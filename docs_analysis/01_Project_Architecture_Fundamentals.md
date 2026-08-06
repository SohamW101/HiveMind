# Mid-Evaluation Report - Part 1: Project Architecture & Fundamentals

This document provides a fundamental, ground-up explanation of the Single-Agent Reinforcement Learning project. It is designed to help you confidently present the core architecture of the simulation during your mid-evaluation.

---

## 1. The Core Objective
The goal of this project is to train an autonomous robot using **Deep Reinforcement Learning (RL)** to navigate a 2D grid-world, locate a resource (green cylinder), pick it up, and transport it to a depot (red zone) while avoiding static obstacles and boundary walls. 

The entire physics engine is simulated using **PyBullet**, which provides realistic collision detection, robotic movement, and physical sensors (LiDAR).

## 2. The Environment Architecture (`hivemind_env/env.py`)
To allow an AI to interact with a PyBullet physics simulation, we wrap the physics engine in a standard **OpenAI Gymnasium (`gym`) Environment**. This forces the simulation to obey the standard RL loop:
1. **Observe:** The environment provides the current state (`obs`).
2. **Act:** The neural network outputs an action.
3. **Step:** The environment executes the action, steps the physics engine forward, and calculates a `reward`.

### The Action Space (What the robot can do)
The robot is controlled via a **Discrete Action Space** of size 7:
* `0`: Move Forward (0.2m)
* `1`: Move Backward (0.2m)
* `2`: Turn Left (90 degrees)
* `3`: Turn Right (90 degrees)
* `4`: Pick Up (If within 0.25m of resource)
* `5`: Drop Off (If exactly on the depot cell)
* `6`: Stay Idle

*Note: All movements are translated into physical PyBullet base-position transformations using 30 micro-substeps to ensure smooth visual rendering during demos.*

### The Observation Space (What the robot sees)
Instead of feeding raw pixels from a camera (which takes weeks to train), we engineered an **Egocentric Grid Representation**. 
The robot extracts a 15x15 local grid centered perfectly on itself. The grid rotates to match the robot's heading, meaning "Forward" is always the same direction in the neural network.

The grid has **5 separate Channels** (like RGB channels in an image), represented as a tensor of shape `(15, 15, 5)`:
* **Channel 0 (Obstacles):** Populated by physical LiDAR raycasts.
* **Channel 1 (Resource):** Shows the resource location if the robot isn't carrying it.
* **Channel 2 (Depot):** Shows the depot location.
* **Channel 3 (Boundaries):** Marks the absolute edges of the 20x20 arena.
* **Channel 4 (Heading):** A marker showing the robot's center and its forward direction.

## 3. The Curriculum Design
To prevent the AI from failing immediately in complex mazes, we built a **Curriculum Learning** pipeline. The environment scales in difficulty based on the agent's success rate:
* **Level 1:** Static, fixed spawn points. No obstacles.
* **Level 2:** Random spawn points for robot, resource, and depot. No obstacles.
* **Level 3:** Random spawns + Random Obstacles.
* **Level 4:** High complexity mazes and tight corridors.

*The transition between these levels is handled mathematically by the `CurriculumCallback` during training, dynamically upgrading the environment when the success rate exceeds 85%.*
