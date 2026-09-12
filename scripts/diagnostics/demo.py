import os

import numpy as np
import pybullet as p
from sb3_contrib import RecurrentPPO

from hivemind_env.env import HiveMindMultiAgentEnv
from hivemind_env.training import INFERENCE_CUSTOM_OBJECTS


def generate_demo():
    print("Initializing environment for rendering...")
    env = HiveMindMultiAgentEnv(render_mode="rgb_array", difficulty_level=1)

    # Load model
    model_path = "models/ppo_recurrent_final.zip"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Exiting.")
        return

    print(f"Loading model {model_path}...")
    model = RecurrentPPO.load(
        model_path, device="cpu", custom_objects=INFERENCE_CUSTOM_OBJECTS
    )

    obs, _info = env.reset(seed=42)

    # LSTM requires hidden states and episode starts
    num_agents = 4
    lstm_states = None
    episode_starts = np.ones((num_agents,), dtype=bool)

    frames = []

    print("Running episode...")
    for step in range(250):
        actions, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_starts, deterministic=True
        )

        # Flatten actions if they are batched shape (4, 1) to (4,)
        if len(actions.shape) > 1:
            actions = actions.flatten()

        obs, _rewards, terminated, truncated, _infos = env.step(actions.tolist())

        # Render frame manually via PyBullet
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[5, 5, 0],
            distance=15,
            yaw=0,
            pitch=-89,
            roll=0,
            upAxisIndex=2,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.1, farVal=100.0
        )
        _, _, rgbImg, _, _ = p.getCameraImage(
            width=640,
            height=640,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=env.client_id,
        )
        # The rgbImg is returned as a 1D array of bytes, reshape it
        rgbImg = np.reshape(rgbImg, (640, 640, 4))
        frame = rgbImg[:, :, :3]  # Drop alpha
        frames.append(frame)

        episode_starts = np.array([terminated or truncated] * num_agents)

        if terminated or truncated:
            print(f"Episode finished at step {step}")
            break

    # Save GIF using matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import animation

    gif_path = "demo_hivemind_final.gif"
    print(f"Saving GIF to {gif_path} with {len(frames)} frames using matplotlib...")

    fig = plt.figure(figsize=(6.4, 6.4))
    plt.axis("off")
    im = plt.imshow(frames[0])

    def update(frame_idx):
        im.set_data(frames[frame_idx])
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=66, blit=True
    )
    ani.save(gif_path, writer="pillow", fps=15)
    print("Done!")


if __name__ == "__main__":
    generate_demo()
