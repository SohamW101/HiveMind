"""
Live one-line-per-iteration view of a training run.

Stable-Baselines3 prints a 20-line table every iteration, which is unreadable as a live
feed - the numbers you care about are four of them and they scroll past. This tails the
log and prints one compact line per iteration instead, with the two signals that decide
whether a run is worth continuing highlighted.

    .venv\\Scripts\\python.exe scripts/watch_run.py run.log

Start it before or during a run; it picks up wherever the file is and follows. Ctrl+C
stops watching and does not touch the run.

WHAT TO WATCH

    len       episode length. While this sits at max_steps, NOTHING is completing -
              which is the failure mode that wasted a 5M-step run. It dropping below
              the cap is the first real sign of life.
    success   fraction of episodes that delivered every carton. The headline.
    rew       mean episode reward per slot. Rising is necessary, not sufficient: a
              policy that learns to stand still also makes this rise.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

ROW = re.compile(r"\|\s*(\w+)\s*\|\s*([-\d.e+]+)\s*\|")
TOTAL = re.compile(r"timesteps\s*:\s*([\d,]+)")

WATCHED = ("total_timesteps", "ep_len_mean", "ep_rew_mean", "success_rate", "fps",
           "time_elapsed")


def human(n):
    return f"{n:,}"


def main():
    ap = argparse.ArgumentParser(description="Live compact view of a training log")
    ap.add_argument("logfile")
    ap.add_argument("--poll", type=float, default=1.0, help="Seconds between reads.")
    ap.add_argument("--from-start", action="store_true",
                    help="Replay the whole log first instead of following the tail.")
    args = ap.parse_args()

    while not os.path.exists(args.logfile):
        print(f"waiting for {args.logfile} ...", end="\r", flush=True)
        time.sleep(args.poll)

    total = None
    cur: dict[str, float] = {}
    last_step = None
    printed_header = False

    with open(args.logfile, "r", encoding="utf-8", errors="replace") as fh:
        if not args.from_start:
            fh.seek(0, os.SEEK_END)
        while True:
            line = fh.readline()
            if not line:
                time.sleep(args.poll)
                continue

            if total is None:
                m = TOTAL.search(line)
                if m:
                    total = int(m.group(1).replace(",", ""))

            m = ROW.search(line)
            if m and m.group(1) in WATCHED:
                try:
                    cur[m.group(1)] = float(m.group(2))
                except ValueError:
                    pass

            # An iteration is complete once total_timesteps has moved on.
            step = cur.get("total_timesteps")
            if step is None or step == last_step:
                continue
            # Wait until the rollout block has also landed, so the line is not half empty.
            if "ep_len_mean" not in cur and step > 0 and last_step is not None:
                continue
            last_step = step

            if not printed_header:
                print(f"{'steps':>18}  {'pct':>4}  {'len':>7}  {'success':>8}  "
                      f"{'reward':>9}  {'fps':>5}  {'eta':>6}")
                print("-" * 68)
                printed_header = True

            pct = f"{step / total * 100:3.0f}%" if total else "   ?"
            ln = cur.get("ep_len_mean")
            sr = cur.get("success_rate")
            rw = cur.get("ep_rew_mean")
            fps = cur.get("fps")
            eta = ""
            if total and fps:
                secs = max(0.0, (total - step) / max(fps, 1e-9))
                eta = f"{secs / 60:5.0f}m" if secs >= 60 else f"{secs:5.0f}s"

            flag = ""
            if ln is not None and sr is not None:
                if sr > 0:
                    flag = "  <- COMPLETING"
                elif total and step > total * 0.15:
                    flag = "  <- nothing finishing yet"

            print(f"{human(int(step)):>18}  {pct}  "
                  f"{'' if ln is None else f'{ln:7.1f}'}  "
                  f"{'' if sr is None else f'{sr:7.1%}'}  "
                  f"{'' if rw is None else f'{rw:9.1f}'}  "
                  f"{'' if fps is None else f'{fps:5.0f}'}  {eta:>6}{flag}",
                  flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped watching (the run is untouched)")
        sys.exit(0)
