"""
Did a protocol emerge, or is the channel decoration? Roadmap steps 7 and 8.

    .venv\\Scripts\\python.exe scripts/analyse_messages.py models/comms_final.zip --num-cartons 4

WHY THIS SCRIPT IS THE RESULT AND THE REWARD CURVE IS NOT

A communicating run and a silent one produce the same shape of learning curve. Worse,
the two obvious summary statistics both mislead on their own:

  - MESSAGE ENTROPY IS NOT EVIDENCE. PPO's entropy bonus pushes the token head toward
    uniform, so a policy that ignores the channel completely sits at the 4.0-bit
    ceiling. High entropy is the null result here, not the positive one.
  - VARIED TOKENS ARE NOT EVIDENCE EITHER. Emitting all 16 symbols proves the head is
    not collapsed; it says nothing about whether the symbols mean anything or whether
    any robot acts on them.

So this script asks three separate questions, and a claim needs all three:

  1. IS THERE INFORMATION IN THE SIGNAL? Mutual information between the token and the
     speaker's own state - am I carrying, where am I, which carton is nearest, who am I.
     Reported against a shuffled-pairing floor, because a mutual information estimate on
     finite samples is biased upward and a small positive number proves nothing without
     knowing what zero looks like on this much data.

  2. DOES THE LISTENER REACT? Causal influence: hold the observation fixed, zero only the
     48 message floats, and measure how far the movement policy's distribution moves
     (mean KL in nats) and how often its argmax changes. A policy that has learned to
     ignore those inputs scores zero here no matter what the speakers emit.

  3. DOES IT MATTER TO THE TASK? The same fixed seeds re-run under four channels -
     learned, silent, shuffled, random - comparing makespan and completion. This is the
     one that decides it. `shuffled` is the sharp comparison: real tokens from the same
     step, attributed to the wrong speakers, so the input distribution is untouched and
     only the meaning is destroyed.

A protocol is: information present (1), listeners react (2), AND performance degrades
when the channel is broken (3). Any two of those without the third is worth reporting as
exactly that and no more.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

import numpy as np

import torch
from stable_baselines3 import PPO

from hivemind_env.env import (
    MESSAGE_MODES,
    MSG_TOKENS,
    NUM_AGENTS,
    OBS_SLICES,
    HiveMindMultiAgentEnv,
    joint_from_slot_actions,
    policy_uses_comms,
)
from hivemind_env.training import (
    INFERENCE_CUSTOM_OBJECTS,
    entropy_bits,
    get_device,
)

MAX_ENTROPY_BITS = math.log2(MSG_TOKENS)


# ---------------------------------------------------------------------------------
# Information theory. Everything in bits, everything zero-safe.
# ---------------------------------------------------------------------------------
def label_entropy_bits(labels) -> float:
    """Entropy of a sequence of discrete labels. Thin wrapper over training.entropy_bits."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0.0
    return entropy_bits(np.unique(labels, return_counts=True)[1], counts=True)


def mutual_information_bits(x, y) -> float:
    """I(X;Y) = H(X) - H(X|Y), both discrete."""
    x, y = np.asarray(x), np.asarray(y)
    if x.size == 0:
        return 0.0
    h_cond = 0.0
    for value in np.unique(y):
        mask = y == value
        h_cond += mask.mean() * label_entropy_bits(x[mask])
    return max(0.0, label_entropy_bits(x) - h_cond)


def mi_with_floor(tokens, feature, rng, repeats=8):
    """
    I(token; feature) alongside the value the same estimator returns on shuffled pairs.

    The floor is not decoration. H(X) - H(X|Y) is biased upward by roughly
    (|X|-1)(|Y|-1) / (2 N ln 2) bits, which at 16 tokens and a 9-valued feature is
    ~0.05 bits at N=10,000 - the same order as a real but weak signal. Reporting the
    measurement without the floor is how noise gets published as a protocol.
    """
    observed = mutual_information_bits(tokens, feature)
    floor = statistics.mean(
        mutual_information_bits(rng.permutation(tokens), feature) for _ in range(repeats)
    )
    return observed, floor


# ---------------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------------
def _region(x, y, half_extent, bins=3):
    """Coarse arena cell of a world position - 3x3 by default, so 9 values."""
    cx = min(bins - 1, max(0, int((x + half_extent) / (2 * half_extent) * bins)))
    cy = min(bins - 1, max(0, int((y + half_extent) / (2 * half_extent) * bins)))
    return cy * bins + cx


def _nearest_carton_slot(env, agent_idx):
    """Which undelivered carton is closest to this robot, by stable slot index."""
    pos = env._canonical_pose(agent_idx)[:2]
    best, best_d = -1, float("inf")
    for slot, rid in enumerate(env.all_resource_ids):
        if slot >= env.active_cartons or env.delivered[slot]:
            continue
        import pybullet as pb
        p, _ = pb.getBasePositionAndOrientation(rid, physicsClientId=env.client_id)
        d = math.hypot(p[0] - pos[0], p[1] - pos[1])
        if d < best_d:
            best, best_d = slot, d
    return best


def rollout(model, episodes, num_cartons, message_mode, deterministic, seed0,
            collect_tokens=False, causal=False, device="cpu"):
    """
    Run `episodes` fixed-seed episodes and return the metrics this script reports.

    `collect_tokens` additionally records, for every robot on every step, the token it
    emitted paired with four facts about the state it emitted that token in. Those pairs
    are what question 1 is computed from.

    `causal` additionally measures, on every step, how far the movement policy's
    distribution moves when the 48 message floats - and only those - are zeroed. That is
    question 2, and it needs no second rollout because the counterfactual is a forward
    pass rather than a different trajectory.
    """
    makespans, completions, deliveries, lengths = [], 0, [], []
    tokens, feats = [], {"carrying": [], "region": [], "nearest": [], "agent": [], "phase": []}
    kls, flips, causal_n = 0.0, 0, 0

    for ep in range(episodes):
        env = HiveMindMultiAgentEnv(
            render_mode=None, num_cartons=num_cartons, comms=True,
            message_mode=message_mode, msg_dropout=0.0,
        )
        obs, _ = env.reset(seed=seed0 + ep)
        terminated = False

        for _ in range(env.max_steps):
            batch = np.asarray(obs, dtype=np.float32)

            if causal:
                kl, flip = _causal_influence(model, batch, device)
                kls += kl
                flips += flip
                causal_n += NUM_AGENTS

            if collect_tokens:
                # Sampled BEFORE the step: a token is chosen from the state the robot is
                # in, not the one its action produces.
                for i in range(NUM_AGENTS):
                    x, y = env._canonical_pose(i)[:2]
                    feats["carrying"].append(int(env.is_carrying[i]))
                    feats["region"].append(_region(x, y, env._arena_half_extent))
                    feats["nearest"].append(_nearest_carton_slot(env, i))
                    feats["agent"].append(i)
                    feats["phase"].append(min(3, int(4 * env.current_step / env.max_steps)))

            raw, _ = model.predict(batch, deterministic=deterministic)
            action = joint_from_slot_actions(raw, NUM_AGENTS)
            obs, _, terminated, truncated, info = env.step(action)

            if collect_tokens:
                tokens.extend(int(t) for t in info["message_tokens"])

            if terminated or truncated:
                break

        if terminated:
            completions += 1
            makespans.append(env.current_step)
        lengths.append(env.current_step)
        deliveries.append(info["delivered"])
        env.close()

    out = {
        "episodes": episodes,
        "message_mode": message_mode,
        "completion_rate": completions / max(episodes, 1),
        "makespan": float(np.mean(makespans)) if makespans else None,
        "avg_delivered": float(np.mean(deliveries)),
        "avg_length": float(np.mean(lengths)),
    }
    if collect_tokens:
        out["tokens"] = np.asarray(tokens)
        out["features"] = {k: np.asarray(v) for k, v in feats.items()}
    if causal and causal_n:
        out["mean_kl_nats"] = kls / causal_n
        out["argmax_flip_rate"] = flips / causal_n
    return out


def _causal_influence(model, batch, device):
    """
    How much does hearing change what this robot intends to do?

    Two numbers over the four robots of one step: the mean KL between the movement
    distribution given the real messages and the same distribution with the message
    slots zeroed, and how often the argmax movement changes. Only the 48 message floats
    differ between the two forward passes, so anything non-zero is attributable to them.

    A policy that has learned to ignore the channel returns (0, 0) here regardless of
    what its speakers emit - which is exactly the failure mode that message entropy
    alone cannot see.
    """
    muted = batch.copy()
    muted[:, OBS_SLICES["messages"]] = 0.0

    with torch.no_grad():
        heard_t, _ = model.policy.obs_to_tensor(batch)
        muted_t, _ = model.policy.obs_to_tensor(muted)
        p = _move_probs(model, heard_t)
        q = _move_probs(model, muted_t)

    kl = (p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum(dim=1)
    flips = (p.argmax(dim=1) != q.argmax(dim=1)).sum().item()
    return float(kl.sum().item()), int(flips)


def _move_probs(model, obs_tensor):
    """Movement-head probabilities, (batch, 7). The token head is head 1 and is skipped."""
    dist = model.policy.get_distribution(obs_tensor)
    inner = getattr(dist, "distribution", None)
    if isinstance(inner, (list, tuple)):
        return inner[0].probs
    return inner.probs


# ---------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------
def report_signal(tokens, features, rng):
    print("\n" + "=" * 78)
    print("  1. IS THERE INFORMATION IN THE SIGNAL?")
    print("=" * 78)

    counts = np.bincount(tokens, minlength=MSG_TOKENS)
    p = counts / max(counts.sum(), 1)
    h = label_entropy_bits(tokens)
    used = int((p >= 0.01).sum())

    print(f"  samples            : {tokens.size:,} tokens")
    print(f"  entropy H(M)       : {h:.3f} bits of a possible {MAX_ENTROPY_BITS:.1f}")
    print(f"  symbols in use     : {used}/{MSG_TOKENS}  (>= 1% of the mass each)")
    print(f"  most common symbol : {int(p.argmax())} at {p.max():.1%}")
    print("  distribution       : "
          + " ".join(f"{k}:{v:.0%}" for k, v in enumerate(p) if v >= 0.01))
    if h > MAX_ENTROPY_BITS - 0.15:
        print("  NOTE: entropy is at its ceiling. That is what an unused token head looks")
        print("        like under PPO's entropy bonus - read the mutual information below")
        print("        before treating it as richness.")

    print(f"\n  {'speaker state':<22}{'I(M;F)':>10}{'shuffled':>11}{'excess':>10}"
          f"{'H(F)':>8}")
    print(f"  {'-' * 61}")
    labels = {
        "carrying": "am I carrying",
        "region": "where I am (3x3)",
        "nearest": "nearest carton slot",
        "agent": "which robot I am",
        "phase": "episode phase (4)",
    }
    results = {}
    for key, label in labels.items():
        observed, floor = mi_with_floor(tokens, features[key], rng)
        results[key] = {"mi_bits": round(observed, 4), "floor_bits": round(floor, 4)}
        print(f"  {label:<22}{observed:>10.3f}{floor:>11.3f}{observed - floor:>10.3f}"
              f"{label_entropy_bits(features[key]):>8.2f}")

    best = max(results, key=lambda k: results[k]["mi_bits"] - results[k]["floor_bits"])
    excess = results[best]["mi_bits"] - results[best]["floor_bits"]
    print(f"\n  Strongest association: {labels[best]}, {excess:.3f} bits above the "
          f"shuffled floor.")
    if excess < 0.02:
        print("  VERDICT: no measurable information. The tokens are not conditioned on")
        print("           anything this script looks at.")
    else:
        print(f"  VERDICT: the tokens carry information about {labels[best]}.")
        print("           Note this is a floor on the protocol's content, not a ceiling -")
        print("           a protocol about something not in this table scores zero here.")
    return results


def report_causal(stats):
    print("\n" + "=" * 78)
    print("  2. DOES THE LISTENER REACT?")
    print("=" * 78)
    kl = stats["mean_kl_nats"]
    flip = stats["argmax_flip_rate"]
    print(f"  mean KL(move | heard  ||  move | silenced) : {kl:.4f} nats")
    print(f"  argmax movement changes when silenced      : {flip:.1%} of robot-steps")
    print("\n  Both are counterfactuals on one forward pass: the observation is identical")
    print("  except for the 48 message floats, so anything above zero is caused by them.")
    # KL is checked first and dominates: at a KL this small the distribution has not
    # moved at all, and the argmax "flips" are near-ties broken by float noise. Reading
    # the flip rate on its own reported weak listening from an untrained net that was
    # provably ignoring the channel.
    if kl < 0.001:
        print("  VERDICT: the policy ignores the channel. Whatever the speakers emit, it")
        print("           does not reach the movement decision"
              + (f" (the {flip:.1%} of argmax changes at this KL are near-ties)."
                 if flip else "."))
    elif flip < 0.02:
        print("  VERDICT: the messages shift the distribution but rarely change the chosen")
        print("           action. Weak listening.")
    else:
        print("  VERDICT: the messages change what the robots do.")


def report_interventions(runs, num_cartons):
    print("\n" + "=" * 78)
    print("  3. DOES IT MATTER TO THE TASK?")
    print("=" * 78)
    print(f"  {'channel':<12}{'makespan':>10}{'vs learned':>12}{'complete':>10}"
          f"{'delivered':>11}")
    print(f"  {'-' * 55}")
    base = runs["learned"]
    for mode in MESSAGE_MODES:
        r = runs[mode]
        ms = f"{r['makespan']:.1f}" if r["makespan"] is not None else "n/a"
        if r["makespan"] is not None and base["makespan"] is not None:
            delta = f"{r['makespan'] - base['makespan']:+.1f}"
        else:
            delta = "-"
        print(f"  {mode:<12}{ms:>10}{delta:>12}{r['completion_rate']:>9.0%}"
              f"{r['avg_delivered']:>10.1f}/{num_cartons}")

    print("\n  'shuffled' is the sharp one: identical tokens from the same step, wrong")
    print("  speakers. The token distribution the network sees is unchanged, so a policy")
    print("  that merely learned to tolerate noise in those inputs survives it.")

    if base["makespan"] is None:
        print("\n  VERDICT: the policy does not complete episodes, so makespan cannot")
        print("           separate the channels. Nothing about communication is shown")
        print("           here - fix the task performance first.")
        return
    worst = max(
        (m for m in MESSAGE_MODES if m != "learned" and runs[m]["makespan"] is not None),
        key=lambda m: runs[m]["makespan"], default=None,
    )
    degraded = [m for m in MESSAGE_MODES if m != "learned"
                and (runs[m]["makespan"] is None
                     or runs[m]["makespan"] > base["makespan"] * 1.05
                     or runs[m]["completion_rate"] < base["completion_rate"] - 0.05)]
    if not degraded:
        print("\n  VERDICT: breaking the channel costs nothing measurable. On this")
        print("           evidence the messages are decoration, whatever their entropy.")
    else:
        print(f"\n  VERDICT: performance degrades when the channel is broken "
              f"({', '.join(degraded)}).")
        if worst:
            print(f"           Worst case is '{worst}'. That degradation is the result;")
            print("           report it with the episode count and the seeds.")


def main():
    ap = argparse.ArgumentParser(
        description="Is the learned channel a protocol or decoration? (roadmap steps 7-8)"
    )
    ap.add_argument("model", help="a checkpoint trained with --comms")
    ap.add_argument("--num-cartons", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=20,
                    help="Per channel condition. The intervention table runs this many "
                         "for each of the four modes on the SAME seeds.")
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--stochastic", action="store_true",
                    help="Sample actions instead of taking the argmax. Worth running "
                         "both ways: this project has three checkpoints whose argmax "
                         "collapses while the sampled policy works.")
    ap.add_argument("--out", default=None, help="Write the numbers to a JSON file.")
    args = ap.parse_args()

    device = get_device()
    model = PPO.load(args.model, device=device, custom_objects=INFERENCE_CUSTOM_OBJECTS)
    if not policy_uses_comms(model):
        sys.exit(
            f"{os.path.basename(args.model)} has action space {model.action_space} - no "
            f"token head, so it was trained without --comms and has no protocol to\n"
            f"analyse. This script is for the communicating arm of the comparison; the\n"
            f"silent arm is scored with scripts/run_evaluation.py."
        )

    deterministic = not args.stochastic
    rng = np.random.default_rng(0)

    print("=" * 78)
    print(f"  Message analysis - {os.path.basename(args.model)}")
    print(f"  {args.num_cartons} cartons, {args.episodes} episodes per condition, "
          f"seeds {args.seed0}-{args.seed0 + args.episodes - 1}")
    print(f"  {'deterministic (argmax)' if deterministic else 'stochastic (sampled)'}, "
          f"{MSG_TOKENS}-token vocabulary, dropout off")
    print("=" * 78, flush=True)

    runs = {}
    for mode in MESSAGE_MODES:
        print(f"\n  running channel '{mode}' ...", flush=True)
        runs[mode] = rollout(
            model, args.episodes, args.num_cartons, mode, deterministic, args.seed0,
            collect_tokens=(mode == "learned"), causal=(mode == "learned"),
            device=device,
        )

    signal = report_signal(runs["learned"]["tokens"], runs["learned"]["features"], rng)
    report_causal(runs["learned"])
    report_interventions(runs, args.num_cartons)

    print("\n" + "=" * 78)
    print("  All three questions have to answer yes before this is a protocol. Report")
    print("  whichever of them says no - that is the honest result and it is still a")
    print("  result: it says the channel was available and went unused.")
    print("=" * 78)

    if args.out:
        payload = {
            "model": args.model,
            "num_cartons": args.num_cartons,
            "episodes_per_condition": args.episodes,
            "deterministic": deterministic,
            "signal": {
                "entropy_bits": round(label_entropy_bits(runs["learned"]["tokens"]), 4),
                "max_entropy_bits": round(MAX_ENTROPY_BITS, 4),
                "mutual_information": signal,
            },
            "causal": {
                "mean_kl_nats": round(runs["learned"]["mean_kl_nats"], 6),
                "argmax_flip_rate": round(runs["learned"]["argmax_flip_rate"], 6),
            },
            "interventions": {
                mode: {k: v for k, v in runs[mode].items()
                       if k not in ("tokens", "features")}
                for mode in MESSAGE_MODES
            },
        }
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
