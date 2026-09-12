"""
HiveMind Inference & Evaluation Script

Uses HiveMindSharedPolicyVecEnv — exactly the same wrapper used during training.
This gives the correct (N_AGENTS, obs_dim) observation layout the model expects.

Usage:
    # Headless evaluation — 10 episodes, 4 cartons (curriculum stage 3)
    python scripts/inference.py --episodes 10 --num-cartons 4

    # Full 12-carton task
    python scripts/inference.py --episodes 10 --num-cartons 12

    # Generate a GIF of one episode
    python scripts/inference.py --episodes 1 --num-cartons 4 --gif

Curriculum ladder (from training.py):
    Stage 1:  1 carton   → max_steps  60
    Stage 2:  2 cartons  → max_steps  90
    Stage 3:  4 cartons  → max_steps 150
    Stage 4:  8 cartons  → max_steps 250
    Stage 5: 12 cartons  → max_steps 400  (full task)
"""

import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sb3_contrib import RecurrentPPO

from hivemind_env.training import INFERENCE_CUSTOM_OBJECTS, NUM_AGENTS
from hivemind_env.vec_env import HiveMindSharedPolicyVecEnv

N_AGENTS = NUM_AGENTS  # 4

CURRICULUM = {1: 1, 2: 2, 4: 4, 8: 8, 12: 12}
MAX_STEPS = {1: 60, 2: 90, 4: 150, 8: 250, 12: 400}

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "ppo_recurrent_final.zip"
)


def make_vec_env(
    num_cartons: int | None, render: bool = False
) -> HiveMindSharedPolicyVecEnv:
    """
    Create the same VecEnv used during training.
    1 world × N_AGENTS = 4 SB3 slots, each with obs shape (177,).
    After reset/step, the vec-env returns obs shape (4, 177) — already batched.
    """
    kwargs = {"communication": True}
    if num_cartons is not None:
        kwargs["num_cartons"] = num_cartons
    if render:
        kwargs["render_mode"] = "human"
    return HiveMindSharedPolicyVecEnv(num_worlds=1, **kwargs)


def load_model(model_path: str) -> RecurrentPPO:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    # Load WITHOUT env= to skip the stored obs-space check.
    # The dummy env workaround is no longer needed since we patch CustomDict.
    return RecurrentPPO.load(
        model_path, device="cpu", custom_objects=INFERENCE_CUSTOM_OBJECTS
    )


def run_episode(model: RecurrentPPO, vec_env: HiveMindSharedPolicyVecEnv) -> dict:
    """
    Runs exactly one episode until done.
    Returns tracking stats.
    """
    obs = vec_env.reset()
    lstm_states = None
    # shape: (4,)
    ep_starts = np.ones((vec_env.num_envs,), dtype=bool)

    done = False
    steps = 0
    collisions = 0
    last_info = {}

    while not done:
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=ep_starts,
            deterministic=True,
        )
        obs, _rewards, dones, infos = vec_env.step(action)
        ep_starts = np.zeros((vec_env.num_envs,), dtype=bool)

        steps += 1
        # The env sets done=True for ALL slots when the episode ends.
        done = dones[0]

        for i in range(vec_env.num_envs):
            if "collisions" in infos[i]:
                collisions += infos[i]["collisions"]

        # Only slot 0 info has the aggregated metrics.
        last_info = infos[0]

    return {
        "steps": steps,
        "collisions": collisions,
        "delivered": last_info.get("delivered", 0),
        "all_delivered": last_info.get("all_delivered", False),
        "is_success": last_info.get("is_success", False),
    }


def evaluate(
    episodes: int,
    num_cartons: int | None,
    model_path: str,
    gif: bool = False,
    mp4: bool = False,
    render: bool = False,
) -> dict | None:
    carton_label = str(num_cartons) if num_cartons else "12 (full task)"
    max_step_hint = MAX_STEPS.get(num_cartons or 12, 400)

    print(f"\n{'=' * 56}")
    print("    HiveMind Multi-Agent Inference Evaluation")
    print(f"{'=' * 56}")
    if gif or mp4:
        print("  Episodes     : 1")
    else:
        print(f"  Episodes     : {episodes}")
    print(f"  Cartons      : {carton_label}   (max_steps ≈ {max_step_hint})")
    print(f"  Agents       : {N_AGENTS}")
    print(f"{'=' * 56}\n")

    print("Loading model …", end=" ", flush=True)
    try:
        model = load_model(model_path)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return None
    print("OK")

    vec_env = make_vec_env(num_cartons, render)
    print(f"Vec-env ready  : {vec_env.num_envs} slots (1 world × {N_AGENTS} agents)\n")

    successes = 0
    total_steps = 0
    total_collisions = 0

    for ep in range(episodes):
        result = run_episode(model, vec_env)

        success = result["is_success"] or result["all_delivered"]
        if success:
            successes += 1
        total_steps += result["steps"]
        total_collisions += result["collisions"]
        status = "✓ SUCCESS" if success else "✗ TIMEOUT"

        print(
            f"  [Ep {ep + 1:>3}/{episodes}] {status} | "
            f"Delivered: {result['delivered']:>2} | "
            f"Steps: {result['steps']:>4} | "
            f"Collisions: {result['collisions']}"
        )

    vec_env.close()

    print(f"\n{'=' * 56}")
    print("                FINAL INFERENCE REPORT")
    print(f"{'=' * 56}")
    print(f"  Cartons in play      : {carton_label}")
    print(f"  Episodes evaluated   : {episodes}")
    print(
        f"  Success rate         : {successes}/{episodes}  ({100 * successes / episodes:.1f}%)"
    )
    print(f"  Avg makespan         : {total_steps / episodes:.1f} steps")
    print(f"  Avg collisions/ep    : {total_collisions / episodes:.2f}")
    print(f"{'=' * 56}\n")

    if gif:
        _save_gif(model, num_cartons, "demo_inference.gif")
    elif mp4:
        _save_gif(model, num_cartons, "demo_inference.mp4")

    return {
        "episodes": episodes,
        "successes": successes,
        "total_steps": total_steps,
        "total_collisions": total_collisions,
        "success_rate": successes / episodes,
        "avg_makespan": total_steps / episodes,
        "avg_collisions": total_collisions / episodes,
    }


def _save_gif(
    model: RecurrentPPO, num_cartons: int | None, out: str = "demo_inference.gif"
) -> None:
    print(f"Generating GIF → {out} …")
    try:
        import matplotlib.pyplot as plt
        from matplotlib import animation

        from hivemind_env.env import HiveMindMultiAgentEnv

        kwargs = {"communication": True}
        if num_cartons is not None:
            kwargs["num_cartons"] = num_cartons
        env = HiveMindMultiAgentEnv(render_mode="rgb_array", **kwargs)

        obs_raw, _ = env.reset()  # (4, 177)
        lstm_states = None
        ep_starts = np.ones((N_AGENTS,), dtype=bool)
        frames, done = [], False

        while not done:
            actions, lstm_states = model.predict(
                obs_raw, state=lstm_states, episode_start=ep_starts, deterministic=True
            )
            ep_starts = np.zeros((N_AGENTS,), dtype=bool)
            actions_flat = np.asarray(actions, dtype=np.int64).flatten()
            obs_raw, _, terminated, truncated, _ = env.step(actions_flat)
            done = bool(terminated or truncated)
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        env.close()
        if not frames:
            print("[WARN] No frames captured — GIF not saved.")
            return

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis("off")
        img = ax.imshow(frames[0])

        def update(i):
            img.set_data(frames[i])
            return [img]

        ani = animation.FuncAnimation(
            fig, update, frames=len(frames), interval=200, blit=True
        )
        if out.endswith(".mp4"):
            ani.save(out, writer="ffmpeg", fps=5)
        else:
            ani.save(out, writer="pillow", fps=5)
        plt.close(fig)
        print(f"Saved {out}  ({len(frames)} frames)")
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] GIF failed: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate the HiveMind RecurrentPPO model.")
    p.add_argument(
        "--episodes", type=int, default=10, help="Evaluation episodes (default: 10)."
    )
    p.add_argument(
        "--num-cartons",
        type=int,
        default=None,
        help="Cartons in play: 1, 2, 4, 8, or 12 (default 12 = full task). "
        "Match this to the curriculum stage you want to test.",
    )
    p.add_argument(
        "--model-path", type=str, default=MODEL_PATH, help="Path to the model to load."
    )
    p.add_argument("--gif", action="store_true", help="Generate demo_inference.gif")
    p.add_argument("--mp4", action="store_true", help="Generate demo_inference.mp4")
    p.add_argument("--render", action="store_true", help="Render PyBullet GUI")
    args = p.parse_args()

    evaluate(
        episodes=args.episodes,
        num_cartons=args.num_cartons,
        model_path=args.model_path,
        gif=args.gif,
        mp4=args.mp4,
        render=args.render,
    )
