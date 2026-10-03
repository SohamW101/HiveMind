#!/usr/bin/env python3
"""Parse TensorBoard logs to find actual success metrics per run."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

runs = [
    "ppo_recurrent_30M_1",
    "ppo_shared_20260905_201425_1",
    "ppo_shared_20260905_201717_1",
    "ppo_shared_20260905_202240_1",
    "my_first_run_5",
]

for run in runs:
    path = f"tensorboard_logs/{run}"
    if not os.path.exists(path):
        print(f"--- {run} --- [NOT FOUND]")
        continue

    ea = EventAccumulator(path, size_guidance={"scalars": 1000})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    success_tags = [
        t
        for t in tags
        if "success" in t.lower() or "delivered" in t.lower() or "ep_len" in t.lower()
    ]

    print(f"\n--- {run} ---")
    print(f"  All tags: {tags[:10]}")
    for tag in success_tags:
        events = ea.Scalars(tag)
        if events:
            last = events[-1]
            best = max(events, key=lambda e: e.value)
            print(f"  {tag}:")
            print(f"    last = {last.value:.4f}  @ step {last.step:,}")
            print(f"    best = {best.value:.4f}  @ step {best.step:,}")
