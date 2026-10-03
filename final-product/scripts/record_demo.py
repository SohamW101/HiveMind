"""Record one full episode top-down as MP4 (needs ffmpeg on PATH).

    python -m scripts.record_demo <comm|nocomm|comm_zeroed> <seed> <out.mp4> [--cartons 12]
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import pybullet as pb
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sb3_contrib import RecurrentPPO  # noqa: E402

from hivemind_env.env import OBS_SLICES, HiveMindMultiAgentEnv  # noqa: E402
from hivemind_env.training import INFERENCE_CUSTOM_OBJECTS  # noqa: E402

MODELS = os.path.join(os.path.dirname(__file__), "..", "models")
VARIANTS = {  # model file, zero incoming messages, title, accent colour
    "comm": ("ppo_recurrent_comm_s0_final.zip", False, "Comm", (235, 104, 52)),
    "nocomm": ("ppo_recurrent_final.zip", True, "No comm", (42, 120, 214)),
    "comm_zeroed": ("ppo_recurrent_comm_s0_final.zip", True, "Comm, messages zeroed", (137, 135, 129)),
}
ARENA, HEADER, FPS = 1000, 80, 30


def font(name, size):
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=sorted(VARIANTS))
    ap.add_argument("seed", type=int)
    ap.add_argument("out")
    ap.add_argument("--cartons", type=int, default=12)
    args = ap.parse_args()

    fname, zero, title, accent = VARIANTS[args.variant]
    model = RecurrentPPO.load(os.path.join(MODELS, fname), device="cpu", custom_objects=INFERENCE_CUSTOM_OBJECTS)
    env = HiveMindMultiAgentEnv(render_mode=None, communication=True, num_cartons=args.cartons)
    W, H = ARENA, ARENA + HEADER
    font_b, font_r = font("LiberationSans-Bold.ttf", 40), font("LiberationSans-Regular.ttf", 30)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
         "-movflags", "+faststart", args.out],
        stdin=subprocess.PIPE,
    )
    state = {"step": 0, "delivered": 0}

    def frame():
        cid = env.client_id
        view = pb.computeViewMatrix([0, 0, 60], [0, 0, 0], [0, 1, 0], physicsClientId=cid)
        proj = pb.computeProjectionMatrixFOV(13.2, 1.0, 1, 100, physicsClientId=cid)
        px = pb.getCameraImage(ARENA, ARENA, view, proj, renderer=pb.ER_TINY_RENDERER,
                               lightDirection=[3, 3, 10], shadow=0, physicsClientId=cid)[2]
        img = Image.new("RGB", (W, H), (252, 252, 251))
        img.paste(Image.fromarray(np.array(px, dtype=np.uint8).reshape(ARENA, ARENA, 4)[:, :, :3]), (0, HEADER))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, 12, HEADER], fill=accent)
        d.text((32, 18), title, font=font_b, fill=(11, 11, 11))
        d.text((W - 32, 24), f"step {state['step']:>3}    delivered {state['delivered']:>2} / {args.cartons}",
               font=font_r, fill=(82, 81, 78), anchor="ra")
        ff.stdin.write(img.tobytes())

    obs, _ = env.reset(seed=args.seed)
    for _ in range(FPS):
        frame()
    step_sim = pb.stepSimulation

    def step_and_capture(*a, **k):  # one frame per physics sub-step -> smooth motion
        r = step_sim(*a, **k)
        frame()
        return r

    pb.stepSimulation = step_and_capture
    lstm, starts, done, info = None, np.ones(4, dtype=bool), False, {}
    while not done:
        if zero:
            obs[:, OBS_SLICES["messages"]] = 0.0
        act, lstm = model.predict(obs, state=lstm, episode_start=starts, deterministic=True)
        starts[:] = False
        state["step"] += 1
        obs, _, term, trunc, info = env.step(act.flatten())
        state["delivered"] = info["delivered"]
        done = term or trunc
    pb.stepSimulation = step_sim
    for _ in range(FPS * 2):
        frame()
    ff.stdin.close()
    ff.wait()
    env.close()
    print(f"{args.variant}: steps={state['step']} delivered={state['delivered']} success={info['is_success']}")


if __name__ == "__main__":
    main()
