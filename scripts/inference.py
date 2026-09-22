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
import math
import pybullet as pb

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
    
    stuck_counters = [0] * N_AGENTS

    while not done:
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=ep_starts,
            deterministic=True,
        )

        # --- Safety Shield (Dynamic No-Op) ---
        base_env = vec_env.envs[0]
        
        current_cells = []
        yaws = []
        for i in range(N_AGENTS):
            pos, orn = pb.getBasePositionAndOrientation(base_env.robot_ids[i], physicsClientId=base_env.client_id)
            yaw = pb.getEulerFromQuaternion(orn)[2]
            yaw = round(yaw / (math.pi / 2.0)) * (math.pi / 2.0)
            r, c = base_env._world_to_grid(pos[0], pos[1])
            current_cells.append((r, c))
            yaws.append(yaw)
            
        next_cells = list(current_cells)
        for i in range(N_AGENTS):
            if len(action.shape) > 1 and action.shape[1] > 1:
                a = action[i, 0]
            else:
                a = action[i]
            r, c = current_cells[i]
            yaw = yaws[i]
            
            # Hardcoded rule: ALWAYS drop off if 1 cell away from depot (0,0)
            if base_env.is_carrying[i] and r <= 1 and c <= 1:
                a = 5
                if len(action.shape) > 1 and action.shape[1] > 1:
                    action[i, 0] = a
                else:
                    action[i] = a
                    
            nr, nc = r, c
            if a == 0:
                dr = round(-math.sin(yaw))
                dc = round(math.cos(yaw))
                nr, nc = r + dr, c + dc
            elif a == 1:
                dr = round(-math.sin(yaw))
                dc = round(math.cos(yaw))
                nr, nc = r - dr, c - dc
                
            # Physics Prediction: If PyBullet will reject the move (wall/shelf),
            # the bot will physically remain stationary.
            if not (0 <= nr < base_env.grid_size and 0 <= nc < base_env.grid_size) or (nr, nc) in base_env.blocked_cells:
                nr, nc = r, c
                
            next_cells[i] = (nr, nc)

        priority = sorted(range(N_AGENTS), key=lambda x: (not base_env.is_carrying[x], x))
        
        resolved = False
        while not resolved:
            resolved = True
            for i in priority:
                for j in range(N_AGENTS):
                    if i == j:
                        continue
                    
                    conflict_type = None
                    # 1. Swap Conflict (Cross-path)
                    if next_cells[i] == current_cells[j] and next_cells[j] == current_cells[i]:
                        if priority.index(i) > priority.index(j):
                            conflict_type = "swap"
                    
                    # 2. Contention Conflict
                    if not conflict_type and next_cells[i] == next_cells[j]:
                        if next_cells[j] == current_cells[j]:
                            conflict_type = "contention"
                        elif priority.index(i) > priority.index(j):
                            conflict_type = "contention"
                            
                    if conflict_type:
                        if next_cells[i] != current_cells[i]:
                            next_cells[i] = current_cells[i]
                            
                            # Break deadlocks on swaps by forcing a turn.
                            # For simple contention, use a No-Op to wait patiently.
                            if conflict_type == "swap":
                                new_a = 3  # Turn Right
                            else:
                                stuck_counters[i] += 1
                                if stuck_counters[i] > 3:
                                    new_a = 3  # Turn Right to break traffic jam
                                    stuck_counters[i] = 0
                                else:
                                    new_a = 4 if base_env.is_carrying[i] else 5  # No-Op
                                
                            if len(action.shape) > 1 and action.shape[1] > 1:
                                action[i, 0] = new_a
                            else:
                                action[i] = new_a
                            resolved = False
                        break
            if resolved:
                # Anyone who didn't conflict gets their counter reset
                for i in range(N_AGENTS):
                    if next_cells[i] != current_cells[i]:
                        stuck_counters[i] = 0
        # --- End Safety Shield ---

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
