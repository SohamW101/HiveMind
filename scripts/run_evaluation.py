"""
Reproducible evaluation of a policy across the carton-count curriculum levels.

    scripts/run_evaluation.py --baseline greedy --episodes 30
    scripts/run_evaluation.py --model models/run_final.zip --episodes 30

MAKESPAN is the headline - steps to deliver every carton - not success rate, because
that is the number the greedy baseline is quoted against. Distance, collisions and
message entropy are reported alongside (roadmap step 8). Episodes use fixed per-episode
seeds (level * 1000 + index), so a rerun reproduces the same maps and the same numbers,
and --baseline runs the harness with no model at all so the harness itself is testable.

A comms policy is detected from its action space, not a flag, and scored with its
channel live; --message-mode breaks the channel to measure what it was worth.
"""
import argparse
import json
import math
import os
import statistics
import sys
import time
from datetime import date

import numpy as np

from hivemind_env.env import (
    ACTION_NAMES,
    DEFAULT_OBS_DIM,
    MESSAGE_MODES,
    MOVE_ACTIONS,
    MSG_TOKENS,
    HiveMindMultiAgentEnv,
)
from hivemind_env.greedy import GreedyController
from hivemind_env.training import (
    CURRICULUM_CARTONS,
    entropy_bits,
    NUM_AGENTS,
    SUCCESS_REWARD_THRESHOLD,
    load_policy,
)

# PORT NOTE: the single-agent branch imported its feature extractor at module scope
# because SB3 unpickles it by name. models.py was empty when this file was ported, so the
# import moved into _load_model(). It stays there: evaluation of a --baseline run should
# not need the network module at all.


# PORT NOTE: LEVEL_LABELS described obstacle density on the single-agent branch. Here the
# ladder is carton count, mirroring hivemind_env.training.CURRICULUM_CARTONS so the
# curriculum the policy trained under and the levels it is scored on cannot drift apart.
LEVEL_LABELS = {
    level: f"{cartons} cartons, 4 robots, randomised shelf gaps"
    for level, cartons in CURRICULUM_CARTONS.items()
}


def wilson_interval(successes, n, z=1.96):
    """95% Wilson score interval - honest error bars for small-n success rates."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def classify(terminated, info, final_reward):
    """
    Outcome from the terminal event, not the episode total.

    PORT NOTE: the single-agent version thresholded the final reward because distance
    shaping dominated the total. Here `remaining_resources` is an unambiguous statement
    of whether the job finished, so it is preferred; the reward threshold is the fallback
    for when step 4 changes the info dict.
    """
    if isinstance(info, dict) and "remaining_resources" in info:
        if int(info["remaining_resources"]) == 0:
            return "COMPLETE"
        return "TIMEOUT" if not terminated else "TERMINATED_INCOMPLETE"
    if terminated and float(np.max(final_reward)) >= SUCCESS_REWARD_THRESHOLD:
        return "COMPLETE"
    return "TIMEOUT"


def _joint_action(model, obs, policy_mode, recurrent, lstm_states, episode_start, action_space):
    """
    Turn whatever the policy is into the MultiDiscrete([7,7,7,7]) the env wants.

    PORT NOTE: this function is entirely new, and it is the one real design decision in
    the port. The single-agent harness called `model.predict(obs)` and cast to int.

    the project roadmap, step 6 recommends training a *shared* policy over 4 single-agent slots, so
    the saved model's action space will be Discrete(7), not MultiDiscrete([7]*4). Two
    modes cover both outcomes:

      shared : one Discrete(7) policy queried once per robot, results stacked. Matches
               the step-6 "cheap option" wrapper. Default.
      joint  : one policy whose action space is already MultiDiscrete([7]*4, e.g. a
               dedicated MAPPO implementation. Queried once.
      random : no model; samples the action space. Exercises the harness itself.
      greedy : no model; the scripted controller from hivemind_env.greedy. This is
               the baseline a learned policy has to beat, and it is deliberately
               scored through THIS harness rather than its own script - same seeds,
               same metrics, same makespan definition. A baseline measured a
               different way is not a comparison.

    Returns (action, lstm_states).
    """
    if policy_mode == "random":
        return action_space.sample(), None
    if policy_mode == "greedy":
        return np.asarray(model.act(), dtype=int), None

    def _predict(single_obs, state):
        if recurrent:
            return model.predict(
                single_obs, state=state,
                episode_start=np.array([episode_start]), deterministic=True,
            )
        act, _ = model.predict(single_obs, deterministic=True)
        return act, None

    if policy_mode == "joint":
        action, lstm_states = _predict(obs, lstm_states)
        action = np.asarray(action, dtype=int)
        return action.reshape(NUM_AGENTS, -1).squeeze(-1), lstm_states

    # shared: obs is expected to be indexable per agent (obs[i] is agent i's vector).
    #
    # A communicating policy returns TWO numbers per robot - movement and the token it
    # broadcasts. Taking element [0] and discarding the rest, which is what this did
    # before roadmap step 7, would evaluate a comms policy with a permanently silent
    # channel and report the result as its score. The rows are stacked whole instead,
    # so the joint action is (4,) or (4, 2) exactly as the env expects.
    if lstm_states is None:
        lstm_states = [None] * NUM_AGENTS
    rows = []
    for i in range(NUM_AGENTS):
        act, lstm_states[i] = _predict(obs[i], lstm_states[i])
        rows.append(np.asarray(act, dtype=int).reshape(-1))
    return np.stack(rows).squeeze(-1) if rows[0].size == 1 else np.stack(rows), lstm_states


def _distance_travelled(prev_positions, positions):
    """Sum of per-robot XY displacement this step. Project roadmap, step 8 metric."""
    total = 0.0
    for (px, py, _), (x, y, _) in zip(prev_positions, positions):
        total += math.hypot(x - px, y - py)
    return total


def evaluate_level(model, difficulty, num_episodes, obs_dim, recurrent, policy_mode,
                   max_steps=None, verbose=True, comms=False, message_mode="learned",
                   msg_dropout=0.0):
    # `num_cartons` is what makes the level real, and it was missing until 2026-08-31.
    # This passed difficulty_level only - which the world stores and nothing reads - so
    # every level here ran the full 12 cartons while the summary table dutifully
    # labelled them 4, 8 and 12. A cross-level run reported makespan 98 / 102 / 97 and
    # "Delivered 12/12" at every row, which is the tell: the levels were identical.
    #
    # Anything measured through this harness before that date is a 12-carton number
    # whatever its label said.
    cartons = CURRICULUM_CARTONS.get(difficulty, difficulty)
    # The channel is off unless the policy has a token head. `msg_dropout` defaults to
    # 0 here and not to the training value: an evaluation is a measurement, and rerunning
    # it should not move the number by a few percent because a different set of links
    # happened to drop. Ask for dropout explicitly to measure robustness to it.
    env = HiveMindMultiAgentEnv(
        render_mode=None, difficulty_level=difficulty, obs_dim=obs_dim,
        num_cartons=cartons, comms=comms, message_mode=message_mode,
        msg_dropout=msg_dropout,
    )
    # PORT NOTE - this guard is new and it is not optional.
    #
    # The single-agent harness looped `while True` and trusted the env to raise
    # `truncated` at max_steps. This env returns hard-coded `False, False` from step()
    # (roadmap step 4), so that loop never exits: the first smoke run hung until it was
    # killed at the two-minute mark rather than finishing two episodes.
    #
    # The harness therefore enforces its own budget. Keep it even after step 4 lands - an
    # evaluation script that can hang on a buggy env is a bad evaluation script.
    step_budget = max_steps if max_steps is not None else env.max_steps

    episodes = []
    action_counts = np.zeros(MOVE_ACTIONS, dtype=int)
    token_counts = np.zeros(MSG_TOKENS, dtype=int)
    total_steps = 0

    for ep in range(num_episodes):
        seed = difficulty * 1000 + ep
        obs, info = env.reset(seed=seed)
        lstm_states = None
        episode_start = True

        # The greedy controller is stateful per episode: it captures the shelf map and
        # carton slots at reset, and it is handed to _joint_action in place of a model.
        controller = GreedyController(env) if policy_mode == "greedy" else None
        actor = controller if controller is not None else model

        # PORT NOTE: per-agent reward sums replace the single scalar. The 90/10 split
        # (the project roadmap, step 4) gives each robot a different total for the same shared
        # outcome, and the spread across robots is itself a signal about role division.
        agent_rewards = np.zeros(NUM_AGENTS, dtype=float)
        steps = 0
        distance = 0.0
        collisions = 0
        reward = np.zeros(NUM_AGENTS)
        terminated = False
        prev_positions = list(info["robot_pos"])
        cartons_at_start = info["remaining_resources"]

        while True:
            action, lstm_states = _joint_action(
                actor, obs, policy_mode, recurrent, lstm_states, episode_start,
                env.action_space,
            )
            episode_start = False

            obs, reward, terminated, truncated, info = env.step(action)
            agent_rewards += np.asarray(reward, dtype=float)
            steps += 1
            moves = np.asarray(action)
            moves = moves[:, 0] if moves.ndim == 2 else moves
            for a in moves:
                action_counts[int(a)] += 1
            if comms:
                for tok in info["message_tokens"]:
                    if tok >= 0:
                        token_counts[int(tok)] += 1

            if controller is not None:
                controller.sync_after_step()

            distance += _distance_travelled(prev_positions, info["robot_pos"])
            prev_positions = list(info["robot_pos"])
            collisions += int(info.get("collisions", 0))  # step 4 should provide this

            if terminated or truncated:
                break
            if steps >= step_budget:
                truncated = True
                break

        total_steps += steps
        outcome = classify(terminated, info, reward)
        delivered = cartons_at_start - info["remaining_resources"]

        # Makespan is only defined for a completed job. Recording None (rather than
        # max_steps) keeps timeouts out of the mean instead of silently flattering it -
        # report completion rate alongside makespan, never makespan alone.
        makespan = steps if outcome == "COMPLETE" else None

        episodes.append({
            "episode": ep + 1, "seed": seed, "outcome": outcome,
            "makespan": makespan, "steps": steps,
            "delivered": int(delivered), "cartons": int(cartons_at_start),
            "distance": round(distance, 2), "collisions": collisions,
            "reward_total": round(float(agent_rewards.sum()), 2),
            "reward_per_agent": [round(float(r), 2) for r in agent_rewards],
        })
        if verbose:
            ms = f"{makespan:4d}" if makespan is not None else "   -"
            print(f"  Ep {ep+1:02d} | seed {seed:<5} | {outcome:<20} | "
                  f"Makespan: {ms} | Steps: {steps:4d} | "
                  f"Delivered: {delivered:2d}/{cartons_at_start} | "
                  f"Dist: {distance:6.1f}m | Coll: {collisions:3d}", flush=True)

    env.close()

    n = len(episodes)
    completed = [e for e in episodes if e["outcome"] == "COMPLETE"]
    makespans = [e["makespan"] for e in completed]
    lo, hi = wilson_interval(len(completed), n)
    delivered_frac = [e["delivered"] / e["cartons"] if e["cartons"] else 0.0 for e in episodes]

    return {
        "difficulty": difficulty,
        "label": LEVEL_LABELS.get(difficulty, f"level {difficulty}"),
        "num_episodes": n,
        # Headline.
        "completion_rate": len(completed) / n,
        "completion_ci95": [round(lo, 4), round(hi, 4)],
        "avg_makespan": round(float(np.mean(makespans)), 1) if makespans else None,
        "median_makespan": float(statistics.median(makespans)) if makespans else None,
        "min_makespan": min(makespans) if makespans else None,
        "max_makespan": max(makespans) if makespans else None,
        # Supporting metrics (project roadmap, step 8).
        "avg_delivered_fraction": round(float(np.mean(delivered_frac)), 4),
        "avg_distance": round(float(np.mean([e["distance"] for e in episodes])), 1),
        "avg_collisions": round(float(np.mean([e["collisions"] for e in episodes])), 2),
        "avg_steps": round(float(np.mean([e["steps"] for e in episodes])), 1),
        "avg_reward_total": round(float(np.mean([e["reward_total"] for e in episodes])), 2),
        "std_reward_total": round(float(np.std([e["reward_total"] for e in episodes])), 2),
        "action_distribution": {ACTION_NAMES[i]: int(action_counts[i]) for i in range(MOVE_ACTIONS)},
        # Roadmap step 8's fourth headline metric. Reported here so the number that
        # accompanies a makespan comes from the same episodes that produced it - but
        # entropy alone does not show a protocol emerged, and PPO's entropy bonus pushes
        # it up whether or not one did. scripts/analyse_messages.py is where the claim
        # gets made, because it also runs the interventions.
        "comms": comms,
        "message_mode": message_mode if comms else None,
        "token_counts": token_counts.tolist() if comms else None,
        "token_entropy_bits": round(entropy_bits(token_counts, counts=True), 3) if comms and token_counts.sum() else None,
        "episodes": episodes,
    }


def _load_model(path, policy_mode):
    if policy_mode in ("random", "greedy"):
        return None, False
    # SB3 unpickles the feature extractor by name, so the class must be importable
    # before load. Imported lazily: a --baseline run should not need the network module.
    #
    # This named CustomCombinedExtractor until 2026-09-02 - a class from the single-agent
    # branch that has never existed here - so the guard always fired and every model
    # evaluation printed a warning saying models.py was empty.
    from hivemind_env.models import HiveMindExtractor  # noqa: F401
    return load_policy(path, device="cpu")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a policy on the multi-agent warehouse across curriculum levels"
    )
    parser.add_argument("--model", default=None,
                        help="Path to a saved SB3 policy. Omit and pass --baseline instead.")
    parser.add_argument("--baseline", choices=["random", "greedy"], default=None,
                        help="Run without a learned model. 'greedy' is the scripted "
                             "controller whose makespan every policy is quoted against "
                             "(roadmap step 5). 'random' samples the action space and "
                             "only exercises the harness.")
    parser.add_argument("--policy-mode", choices=["shared", "joint"], default="shared",
                        help="shared: one Discrete(7) policy queried per robot (roadmap "
                             "step 6 default). joint: one MultiDiscrete policy.")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--obs-dim", type=int, default=DEFAULT_OBS_DIM,
                        help="Asserted against the pinned width in env.py; env.py "
                             "rejects a mismatch rather than building a bad env.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Hard per-episode step budget enforced by the harness. "
                             "Defaults to env.max_steps (2000). Lower it for smoke runs - "
                             "the env does not raise truncated yet (roadmap step 4), so "
                             "without this the harness would run forever.")
    parser.add_argument("--message-mode", choices=list(MESSAGE_MODES), default="learned",
                        help="Intervention on the communication channel, ignored unless "
                             "the loaded policy has a token head. 'learned' is what the "
                             "speakers said; 'silent', 'shuffled' and 'random' break it "
                             "in three different ways. Scoring the same checkpoint under "
                             "'learned' and 'shuffled' is what turns a channel into a "
                             "result - if the makespan does not move, nothing is being "
                             "communicated. scripts/analyse_messages.py runs all four "
                             "in one pass.")
    parser.add_argument("--msg-dropout", type=float, default=0.0,
                        help="Link dropout during evaluation (default 0: a measurement "
                             "should be repeatable). Set it to the training value to "
                             "measure how much the policy needs a reliable channel.")
    parser.add_argument("--levels", type=int, nargs="+", default=sorted(CURRICULUM_CARTONS))
    parser.add_argument("--out", default="docs_analysis/evaluation_results.json")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite --out even if it holds a larger, more reliable run.")
    args = parser.parse_args()

    if (args.model is None) == (args.baseline is None):
        parser.error("pass exactly one of --model PATH or --baseline random")

    policy_mode = args.baseline if args.baseline else args.policy_mode

    if args.model and not os.path.exists(args.model):
        print(f"ERROR: model not found: {args.model}")
        sys.exit(1)

    # Guard: --out is the source of truth for the docs and the figures. A quick smoke run
    # (--episodes 5) must not silently replace a 30-episode result set with weaker
    # numbers.
    if os.path.exists(args.out) and not args.force:
        try:
            with open(args.out, encoding="utf-8") as f:
                existing = json.load(f).get("episodes_per_level", 0)
        except (OSError, ValueError):
            existing = 0
        if existing > args.episodes:
            sys.exit(
                f"ERROR: {args.out} already holds a {existing}-episode-per-level run, and\n"
                f"this one is only {args.episodes}. Refusing to overwrite better data.\n\n"
                f"  Write elsewhere:  --out docs_analysis/eval_smoke.json\n"
                f"  Or overwrite:     --force"
            )

    print("=" * 84)
    print("  HiveMind Multi-Agent Policy Evaluation")
    print("=" * 84)
    if args.model:
        print(f"  Model      : {args.model} ({os.path.getsize(args.model)/1024/1024:.1f} MB)")
    else:
        print(f"  Model      : none - {args.baseline} baseline")
    if args.baseline == "greedy":
        print("  Note       : this run produces THE reference makespan. Every later "
              "policy\n               result should be quoted against it.")
    print(f"  Policy mode: {policy_mode}")
    print(f"  Agents     : {NUM_AGENTS}")
    print(f"  Episodes   : {args.episodes} per level (fixed seeds, reproducible)")
    print(f"  Date       : {date.today().isoformat()}", flush=True)

    # Honest banner: refuse to imply these numbers mean anything they cannot yet mean.
    probe = HiveMindMultiAgentEnv(render_mode=None)
    obs, _ = probe.reset(seed=0)
    obs_missing = len(obs) == 0
    probe.close()
    if obs_missing:
        print("\n  !! WARNING: _get_obs() returns an empty list (roadmap step 3 not done),")
        print("     and reward/termination are hard-coded (step 4 not done). Every episode")
        print("     will run to max_steps with zero reward. These numbers measure nothing")
        print("     about a policy - they only prove the harness executes.")
        if args.model:
            sys.exit("\nERROR: refusing to evaluate a model against an empty observation.\n"
                     "       Finish roadmap step 3 first, or use --baseline random.")

    model, recurrent = _load_model(args.model, policy_mode)
    if model is not None:
        print(f"  Algorithm  : {'RecurrentPPO' if recurrent else 'PPO'}", flush=True)

    # Whether to build a communicating env is read off the policy, not off a flag. A
    # comms checkpoint emits two numbers per robot; asking the user to remember which
    # kind they trained is how a communicating policy gets scored with its channel
    # switched off and the result written into a results file as its makespan.
    comms = bool(model is not None and getattr(model.action_space, "shape", ()) == (2,))
    if comms:
        print(f"  Comms      : ON - {MSG_TOKENS}-token channel, "
              f"message_mode={args.message_mode}, dropout {args.msg_dropout:.0%}")
    elif model is not None:
        print("  Comms      : off - this checkpoint has no token head (step 6 baseline)")
        if args.message_mode != "learned":
            print("               --message-mode is ignored: there are no messages.")

    started = time.time()
    results = {}
    for level in args.levels:
        label = LEVEL_LABELS.get(level, f"level {level}")
        print(f"\n{'='*84}\n  Level {level} - {label}\n{'='*84}", flush=True)
        results[level] = evaluate_level(
            model, level, args.episodes, args.obs_dim, recurrent, policy_mode,
            max_steps=args.max_steps, comms=comms, message_mode=args.message_mode,
            msg_dropout=args.msg_dropout,
        )
        r = results[level]
        lo, hi = r["completion_ci95"]
        ms = f"{r['avg_makespan']:.0f}" if r["avg_makespan"] is not None else "n/a (no completions)"
        print(f"\n  Makespan  : {ms}   (headline - lower is better)")
        print(f"  Completed : {r['completion_rate']*100:.0f}%  (95% CI {lo*100:.0f}-{hi*100:.0f}%)"
              f"   Delivered: {r['avg_delivered_fraction']*100:.0f}% of cartons")
        print(f"  Distance  : {r['avg_distance']:.1f} m    Collisions: {r['avg_collisions']:.1f}")
        print(f"  Reward    : {r['avg_reward_total']:.2f} +/- {r['std_reward_total']:.2f} "
              f"(summed over {NUM_AGENTS} agents)")
        print(f"  Steps     : mean {r['avg_steps']:.0f}", flush=True)
        if r["token_entropy_bits"] is not None:
            print(f"  Msg entropy: {r['token_entropy_bits']:.2f} bits of a possible "
                  f"{math.log2(MSG_TOKENS):.1f}  (mode: {r['message_mode']})", flush=True)

    elapsed = time.time() - started
    overall_eps = sum(results[l]["num_episodes"] for l in results)
    overall_done = sum(results[l]["completion_rate"] * results[l]["num_episodes"] for l in results)

    print(f"\n{'='*84}\n  CROSS-LEVEL SUMMARY\n{'='*84}")
    print(f"  {'Level':<7}{'Cartons':>9}{'Makespan':>10}{'Complete':>10}{'95% CI':>14}"
          f"{'Dist(m)':>9}{'Coll':>7}{'AvgRew':>9}")
    print(f"  {'-'*75}")
    for level in args.levels:
        r = results[level]
        lo, hi = r["completion_ci95"]
        ms = f"{r['avg_makespan']:.0f}" if r["avg_makespan"] is not None else "-"
        print(f"  {level:<7}{CURRICULUM_CARTONS.get(level, '?'):>9}{ms:>10}"
              f"{r['completion_rate']*100:>9.0f}%{f'{lo*100:.0f}-{hi*100:.0f}%':>14}"
              f"{r['avg_distance']:>9.1f}{r['avg_collisions']:>7.1f}"
              f"{r['avg_reward_total']:>9.2f}")
    print(f"  {'-'*75}")
    print(f"  Overall: {overall_done/overall_eps*100:.1f}% of {overall_eps} episodes completed")
    print(f"  Wall clock: {elapsed/60:.1f} min")
    print("\n  Compare avg_makespan against the greedy baseline (the project roadmap, step 5).")
    print("  A learned policy that does not beat greedy makespan has not shown anything.")

    payload = {
        "model": args.model,
        "policy_mode": policy_mode,
        "num_agents": NUM_AGENTS,
        "obs_dim": args.obs_dim,
        "episodes_per_level": args.episodes,
        "date": date.today().isoformat(),
        "observations_implemented": not obs_missing,
        "overall_completion_rate": round(overall_done / overall_eps, 4),
        "total_episodes": overall_eps,
        "levels": {str(k): v for k, v in results.items()},
    }
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  JSON written to {args.out}")


if __name__ == "__main__":
    main()
