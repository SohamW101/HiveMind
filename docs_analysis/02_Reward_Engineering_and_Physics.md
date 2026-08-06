# Mid-Evaluation Report - Part 2: Reward Engineering & Physics

Training a neural network relies entirely on the mathematical reward signals it receives. If the reward signal is flawed, the AI will confidently learn the wrong behavior. This document breaks down the advanced robotics optimizations we applied to the reward function to master Level 4.

---

## 1. The Trap of Sparse vs. Dense Rewards
In Reinforcement Learning, rewards are categorized as **Sparse** or **Dense**:
* **Sparse Rewards:** The robot only receives a reward at the very end of the task (e.g., `+10` for dropping off the resource). 
  * *Problem:* In a 500-step horizon, the probability of a random robot stumbling upon the exact sequence to reach the depot is near zero. The network learns nothing.
* **Dense Rewards:** The robot receives a reward every single step based on its distance to the target. 
  * *Solution:* We applied **Potential-Based Reward Shaping (PBRS)** using Euclidean (straight-line) distance. If the robot moves closer to the target, it gets a positive reward.

## 2. The Euclidean Trap (Local Minima)
While the dense Euclidean reward flawlessly solved Levels 1 and 2, it caused a critical failure in Levels 3 and 4 (Obstacles). 

**The Local Minima Problem:**
If a U-shaped obstacle blocks the target, the robot must walk *away* from the target to go around it. However, walking away increases the Euclidean distance, giving the robot a **negative penalty**. 
Mathematically, the robot realizes that standing still or spinning in circles (penalty: `-0.01`) is better than walking away to go around the wall (penalty: `-0.20`). The agent traps itself in the corners of walls.

## 3. The Vision Hardware Upgrade (180-Ray LiDAR)
To solve this without cheating, we first needed to upgrade the robot's physical sensors. 
Initially, the PyBullet simulation fired **36 LiDAR rays** (1 ray every 10 degrees). At maximum range (1.5m), the gap between rays was `0.26m`. Since the obstacles were only `0.2m` wide, obstacles were slipping between the rays. The robot was effectively blind.

**The Fix:** We upgraded the sensor payload to **180 independent rays**. The gap shrank to a microscopic `0.05m`. Obstacles are now painted by 3-4 rays simultaneously, guaranteeing perfect physical perception.

## 4. The Breakthrough: Artificial Potential Fields (APF)
To fix the "Local Minima Trap" using purely realistic robotics principles, we implemented **Artificial Potential Fields (APF)** directly into the reward function. APF is a classic robotics algorithm that models the environment as a magnetic field.

1. **The Attractive Force (The Target):** We kept the dense Euclidean distance reward. The target acts like a magnet, constantly pulling the robot forward.
2. **The Repulsive Force (The LiDAR):** We introduced a dynamic penalty based on the new 180-Ray LiDAR. If a ray detects an obstacle getting dangerously close (under 0.4m), it applies an inverse-distance penalty (`-0.02 * (1.0 / distance)`).

**How it works together:**
When the robot gets trapped in a corner, the Attractive Force tells it to push forward into the wall. But as it gets closer, the LiDAR Repulsive Force grows exponentially stronger and pushes back. The two mathematical forces collide and balance out, causing the robot to gracefully "slide" around the corner and out of the trap, entirely driven by local physics sensors!
