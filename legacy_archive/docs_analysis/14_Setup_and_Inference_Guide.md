# 14_Setup_and_Inference_Guide

This guide explains how to properly set up your local environment to run the interactive 3D PyBullet visualizer and view the training logs.

## 1. The Python 3.11 Windows Issue
If you attempt to install `pybullet` on Windows using Python 3.11, it will fail to compile from source and throw a `missing string.h` C++ Compiler Error. This is because PyBullet only provides pre-compiled Windows binaries for Python up to version **3.10**.

## 2. Fixing the Setup (Windows)

To run the live interactive visualizer on your local PC, you MUST install Python 3.10.

### Step 1: Install Python 3.10
If you do not have it, download the official Python 3.10 installer from Python.org, or install Miniconda and create a 3.10 environment.
```powershell
conda create -n hivemind_310 python=3.10 -y
conda activate hivemind_310
```
*(If using standard Python `venv`, ensure you create the venv using your 3.10 executable).*

### Step 2: Install Dependencies
```powershell
pip install pybullet stable-baselines3 gymnasium numpy torch sb3-contrib tensorboard matplotlib
```

## 3. Running the Live 3D Inference

Once PyBullet is properly installed via Python 3.10, you can use the official inference script to watch the robots in real-time.

Navigate to the project root and run:
```powershell
python scripts/inference.py --episodes 5 --level 2 --render
```
*   `--episodes`: How many times the scenario will run.
*   `--level`: The difficulty (1 = Empty Room, 4 = Maximum Obstacles).
*   `--render`: **CRITICAL FLAG**. This tells the script to launch the actual PyBullet GUI window so you can watch it live.

At the end of the runs, the script will print a formal statistical report to your terminal detailing the Success Rate and Average Makespan.

## 4. Viewing the Logs
To view the raw metrics that proved the 100% success rate:
```powershell
tensorboard --logdir tensorboard_logs
```
Then open `http://localhost:6006` in your browser.
