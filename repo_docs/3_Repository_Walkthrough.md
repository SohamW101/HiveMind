# Repository Walkthrough & Onboarding

Welcome to the HiveMind Multi-Agent Reinforcement Learning project! This document outlines how to utilize the active tools in the repository.

## 1. Directory Structure

```text
HiveMind/
├── hivemind_env/         # Core PyBullet simulation environment & physics engine
├── models/               # Saved .zip checkpoints (e.g. ppo_recurrent_final.zip)
├── repo_docs/            # Professional architecture and analysis documentation
├── scripts/              # Active execution scripts
│   ├── diagnostics/      # Probes, verifiers, and smoke tests
│   ├── train.py          # Master training loop
│   ├── inference.py      # GUI Rendering, GIF/MP4 export, single-stage eval
│   ├── evaluate_all.py   # Multi-stage curriculum evaluation suite
│   └── parse_metrics.py  # TensorBoard data extraction
└── legacy_archive/       # Historical documentation and retired media/scripts
```

## 2. Setting Up
Activate the Python virtual environment containing `stable-baselines3`, `pybullet`, and `torch`.
```bash
# Example
source /home/raid/Udayraj/venv/bin/activate
```

## 3. Running Inference (Visualization)
To view the agents executing the final trained model (`ppo_recurrent_final.zip`), you can run the inference script.

**Generate an MP4 Video (Realtime 5 FPS):**
```bash
python scripts/inference.py --episodes 1 --num-cartons 12 --mp4
```

**Run Headless Fast Evaluation (10 Episodes):**
```bash
python scripts/inference.py --episodes 10 --num-cartons 12
```

## 4. Running the Evaluation Suite
To test the model against Curriculum Overfitting and analyze its performance across multiple sparse/dense environments in one go:
```bash
python scripts/evaluate_all.py --episodes 50
```
This automatically loops through 4, 8, and 12 carton configurations and outputs a summary table.

## 5. Training a New Model
If you want to train a completely new model from scratch, modify the curriculum ladder in `scripts/train.py` and run:
```bash
python scripts/train.py
```
*Note: Ensure your `tensorboard_logs/` are cleared or properly versioned to avoid metric contamination.*
