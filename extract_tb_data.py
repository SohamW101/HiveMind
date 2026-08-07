"""
Extract all scalar data from the PPO_v1_5 TensorBoard logs and write
a detailed CSV + analysis text file for offline inspection.
"""
import os
import sys

try:
    from tensorflow.python.summary.summary_iterator import summary_iterator
    USE_TF = True
except ImportError:
    USE_TF = False

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    USE_TB = True
except ImportError:
    USE_TB = False

LOG_DIR = "tensorboard_logs_ppo_v1/PPO_v1_5"
OUT_FILE = "docs_analysis/tb_ppo_v1_5_full_data.txt"

if not os.path.isdir(LOG_DIR):
    print(f"ERROR: {LOG_DIR} not found. Available:")
    for d in os.listdir("tensorboard_logs_ppo_v1"):
        print(f"  {d}")
    sys.exit(1)

if not USE_TB:
    print("ERROR: tensorboard not installed. Run: pip install tensorboard")
    sys.exit(1)

print(f"Loading TensorBoard data from {LOG_DIR} ...")
ea = EventAccumulator(LOG_DIR)
ea.Reload()

tags = ea.Tags()
print(f"Available scalar tags: {tags.get('scalars', [])}")

results = {}
for tag in tags.get("scalars", []):
    events = ea.Scalars(tag)
    results[tag] = [(e.step, e.value) for e in events]
    print(f"  Tag: {tag:50s}  Points: {len(events):5d}  "
          f"First step: {events[0].step:>10,}  Last step: {events[-1].step:>10,}  "
          f"Last value: {events[-1].value:.6f}")

# Write full analysis file
with open(OUT_FILE, "w") as f:
    f.write("=" * 80 + "\n")
    f.write("PPO_v1_5 Full TensorBoard Data Extract\n")
    f.write("=" * 80 + "\n\n")

    METRICS_OF_INTEREST = [
        "curriculum/difficulty_level",
        "curriculum/success_rate",
        "rollout/ep_rew_mean",
        "rollout/ep_len_mean",
        "train/entropy_loss",
        "train/approx_kl",
        "train/clip_fraction",
        "train/explained_variance",
        "train/learning_rate",
        "train/value_loss",
        "train/loss",
        "train/policy_gradient_loss",
    ]

    for tag in METRICS_OF_INTEREST:
        if tag not in results:
            f.write(f"\n--- {tag}: NOT FOUND ---\n")
            continue

        data = results[tag]
        values = [v for _, v in data]
        steps = [s for s, _ in data]

        f.write(f"\n{'='*70}\n")
        f.write(f"TAG: {tag}\n")
        f.write(f"{'='*70}\n")
        f.write(f"  Total datapoints : {len(data)}\n")
        f.write(f"  Step range       : {steps[0]:,} → {steps[-1]:,}\n")
        f.write(f"  Value at start   : {values[0]:.6f}\n")
        f.write(f"  Value at end     : {values[-1]:.6f}\n")
        f.write(f"  Min value        : {min(values):.6f} @ step {steps[values.index(min(values))]:,}\n")
        f.write(f"  Max value        : {max(values):.6f} @ step {steps[values.index(max(values))]:,}\n")

        # Print key milestones (every 10% of the run)
        n = len(data)
        milestones = [0, n//10, n//5, n//4, n//3, n//2, 2*n//3, 3*n//4, 4*n//5, 9*n//10, n-1]
        f.write(f"\n  Key milestones:\n")
        f.write(f"  {'Step':>12}  {'Value':>12}\n")
        f.write(f"  {'-'*26}\n")
        seen = set()
        for idx in milestones:
            if idx in seen or idx >= len(data):
                continue
            seen.add(idx)
            f.write(f"  {data[idx][0]:>12,}  {data[idx][1]:>12.6f}\n")

    # Special: Find curriculum transition steps
    if "curriculum/difficulty_level" in results:
        f.write(f"\n{'='*70}\n")
        f.write("CURRICULUM TRANSITION STEPS (exact steps each level changed)\n")
        f.write(f"{'='*70}\n")
        diff_data = results["curriculum/difficulty_level"]
        prev_level = diff_data[0][1]
        for step, level in diff_data:
            if level != prev_level:
                f.write(f"  Level {prev_level:.0f} → Level {level:.0f}  at step {step:>10,}\n")
                prev_level = level
        f.write(f"  Final level: {diff_data[-1][1]:.0f} at step {diff_data[-1][0]:,}\n")

print(f"\nFull data written to: {OUT_FILE}")
