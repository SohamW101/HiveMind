"""
Communication ablation evaluation: run a trained policy headless for N episodes and write
one JSON line per episode.

    python -m scripts.evaluate_ablation --variant comm        --cartons 12 --episodes 100 --out eval/episodes.jsonl
    python -m scripts.evaluate_ablation --variant nocomm      --cartons 12 --episodes 100 --out eval/episodes.jsonl
    python -m scripts.evaluate_ablation --variant comm_zeroed --cartons 12 --episodes 100 --out eval/episodes.jsonl

Variants:
    comm         comm model, messages delivered (as trained)
    nocomm       no-comm model, message slots held at zero (as trained)
    comm_zeroed  comm model with its incoming messages zeroed - the intervention test

Episode e uses warehouse seed `--seed0 + e`, so every variant sees identical layouts.
Records carry per-step message tokens and carrying flags for the communication analysis;
feed the output directory to scripts/make_presentation_graphs.py.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sb3_contrib import RecurrentPPO

from hivemind_env.env import NUM_AGENTS, OBS_SLICES, HiveMindMultiAgentEnv
from hivemind_env.training import INFERENCE_CUSTOM_OBJECTS

MODELS = os.path.join(os.path.dirname(__file__), "..", "models")
VARIANTS = {
    "comm": ("ppo_recurrent_comm_s0_final.zip", False),
    "nocomm": ("ppo_recurrent_final.zip", True),
    "comm_zeroed": ("ppo_recurrent_comm_s0_final.zip", True),
}
MSG = OBS_SLICES["messages"]
CARRY = OBS_SLICES["own_carrying"]


def run_episode(model, env, seed, zero_messages):
    obs, _ = env.reset(seed=seed)
    lstm, starts = None, np.ones(NUM_AGENTS, dtype=bool)
    steps = collisions = invalid = 0
    tokens, moves, carrying = [], [], []
    done, info = False, {}
    while not done:
        if zero_messages:
            obs[:, MSG] = 0.0
        carrying.append(obs[:, CARRY].ravel().astype(int).tolist())
        action, lstm = model.predict(obs, state=lstm, episode_start=starts, deterministic=True)
        starts[:] = False
        moves.append(action[:, 0].astype(int).tolist())
        tokens.append(action[:, 1].astype(int).tolist())
        obs, _, term, trunc, info = env.step(action.flatten())
        steps += 1
        collisions += info["collisions"]  # new collision events this step, per world
        invalid += sum(info["invalid_actions"])
        done = term or trunc
    return {
        "steps": steps,
        "is_success": bool(info["is_success"]),
        "delivered": int(info["delivered"]),
        "collisions": int(collisions),
        "invalid_actions": int(invalid),
        "tokens": tokens,
        "moves": moves,
        "carrying": carrying,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    p.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    p.add_argument("--model", default=None, help="override the variant's default weights")
    p.add_argument("--cartons", type=int, default=12)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--start", type=int, default=0, help="first episode index (for sharding)")
    p.add_argument("--seed0", type=int, default=1000)
    p.add_argument("--out", required=True, help="JSON-lines file to append to")
    args = p.parse_args()

    fname, zero = VARIANTS[args.variant]
    model_path = args.model or os.path.join(MODELS, fname)
    model = RecurrentPPO.load(model_path, device="cpu", custom_objects=INFERENCE_CUSTOM_OBJECTS)
    env = HiveMindMultiAgentEnv(render_mode=None, communication=True, num_cartons=args.cartons)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    wins = 0
    with open(args.out, "a") as f:
        for e in range(args.start, args.start + args.episodes):
            r = run_episode(model, env, args.seed0 + e, zero)
            r.update(label=args.variant, cartons=args.cartons, episode=e, seed=args.seed0 + e)
            f.write(json.dumps(r) + "\n")
            wins += r["is_success"]
            print(f"[{args.variant} c={args.cartons}] ep {e:3d}  success={r['is_success']}  "
                  f"steps={r['steps']:3d}  delivered={r['delivered']:2d}", flush=True)
    env.close()
    print(f"success {wins}/{args.episodes}")


if __name__ == "__main__":
    main()
