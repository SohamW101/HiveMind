# HiveMind Multi-Agent RL

<p align="center">
  <em>Achieving 100% Cooperative Success in a Continuous Multi-Agent Physics Simulation.</em>
</p>

---

## 🐝 What is HiveMind?
HiveMind is a Multi-Agent Reinforcement Learning (MARL) framework designed to solve warehouse logistics tasks. Built on PyBullet and Stable-Baselines3, the system controls 4 concurrent differential-drive robots (AGVs) in a continuous 2D plane.

Unlike traditional independent networks, HiveMind utilizes a **Shared Recurrent Policy (LSTM)**. All 4 robots act as extensions of a single centralized brain, allowing them to communicate relative positions and flawlessly coordinate dense 12-carton delivery tasks without devastating gridlocks.

## 📚 Documentation 
We have meticulously documented every facet of the project architecture, training methodology, and physics constraints in the `repo_docs/` folder.

1. **[Architecture Deep Dive](repo_docs/1_Architecture_Deep_Dive.md)**: Explore the 177D Observation Space and the Shared Policy network.
2. **[Training Analysis & Logs](repo_docs/2_Training_Analysis_and_Logs.md)**: Understand the 5-Stage Curriculum Ladder and Reward Shaping.
3. **[Repository Walkthrough](repo_docs/3_Repository_Walkthrough.md)**: Onboarding guide for setting up, rendering MP4s, and training models.
4. **[Curriculum Evaluation Report](repo_docs/4_Curriculum_Evaluation_Report.md)**: Empirical 50-episode breakdown of the final model's performance.
5. **[Environment Constraints & Physics](repo_docs/5_Environment_Constraints_and_Physics.md)**: The strict limitations of the PyBullet LiDAR and kinematic simulation.
6. **[Current Implementation State](repo_docs/6_Current_Implementation_State.md)**: A catalog of exactly what has been built and verified to date.
7. **[Future Roadmap & Plans](repo_docs/7_Future_Roadmap_and_Plans.md)**: Next steps (e.g. Domain Randomization, Sim2Real transfer).

## 🚀 Quick Start
*See the [Repository Walkthrough](repo_docs/3_Repository_Walkthrough.md) for full details.*

**Watch the Agents Play:**
```bash
python scripts/inference.py --episodes 1 --num-cartons 12 --mp4
```

**Run the Evaluation Suite:**
```bash
python scripts/evaluate_all.py --episodes 50
```

---

<p align="center">
  <i>Developed for advanced MARL research in dense spatial simulations.</i>
</p>
