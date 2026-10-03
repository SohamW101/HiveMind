#!/usr/bin/env python3
"""Parse TensorBoard logs to check curriculum progression."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = "ppo_recurrent_30M_1"
path = f"tensorboard_logs/{run}"
if not os.path.exists(path):
    print(f"--- {run} --- [NOT FOUND]")
    sys.exit(0)

ea = EventAccumulator(path, size_guidance={"scalars": 2000})
ea.Reload()

tags = ea.Tags().get("scalars", [])
if "curriculum/target_cartons" in tags:
    events = ea.Scalars("curriculum/target_cartons")
    print(f"--- {run} Curriculum Progression ---")
    last_val = -1
    for e in events:
        if e.value != last_val:
            print(f"Step {e.step:,}: Target Cartons -> {e.value}")
            last_val = e.value
    print(f"Final Step {events[-1].step:,}: Target Cartons -> {events[-1].value}")
else:
    print("No curriculum/target_cartons found")
