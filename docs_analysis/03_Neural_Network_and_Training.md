# Mid-Evaluation Report - Part 3: AI Core, LSTMs, & Pipeline

This final document covers the neural network architecture, the memory upgrades, and the optimized hyperparameter pipeline used to orchestrate the 10,000,000 step training run.

---

## 1. The AI Brain (Feature Extractor)
The observation space is a massive `(15, 15, 5)` grid tensor. To process this, we built a `CustomCombinedExtractor` in `models.py`. 
* **CNN (Convolutional Neural Network):** Processes the 15x15x5 grid using spatial filters to understand walls, resources, and boundaries.
* **MLP (Multi-Layer Perceptron):** Processes the `is_carrying` flag.
* The CNN and MLP outputs are mathematically concatenated into a single 256-dimensional feature vector representing the state of the world.

## 2. The Amnesia Problem (Why standard PPO fails)
Standard Proximal Policy Optimization (PPO) algorithms are feed-forward networks. They process the current snapshot of the world and make a decision. 
**The flaw:** They have absolutely no memory. 
If the robot is looking at the depot, but turns 180 degrees to navigate around a U-shaped obstacle, the depot is no longer in its 15x15 visual grid. Because it has no memory, the robot instantly forgets where the depot is, causing it to spin aimlessly.

## 3. The LSTM Upgrade (Recurrent PPO)
To solve the amnesia problem in complex Partially Observable Markov Decision Processes (POMDPs), we upgraded the entire algorithmic core to **`RecurrentPPO`** (from the `sb3-contrib` library).
* **What is it?** `RecurrentPPO` injects an **LSTM** (Long Short-Term Memory) layer into the network.
* **How it works:** An LSTM maintains an infinite-horizon "Hidden State" vector. Every step, it updates this hidden state with new information. 
* **The Result:** The robot can now look at the depot, turn 180 degrees, spend the next 100 frames navigating an intense maze using its Artificial Potential Field instincts, and *still remember exactly where the depot is located behind it*. It grants the robot true spatial memory.

## 4. Training Pipeline & Hyperparameters (`train_v2.py`)
To train this massive LSTM architecture, we deployed an optimized 10-million step pipeline using `stable-baselines3`.

### Key Hyperparameters Explained:
| Parameter | Value | Purpose |
| :--- | :--- | :--- |
| **Total Steps** | `10,000,000` | Because Level 4 introduces millions of random combinations (spawns + obstacles), 10M steps guarantees the LSTM has enough data to fully mature its hidden state representations. |
| **`ent_coef`** | `0.05` | (Entropy Coefficient). This parameter dictates how much "randomness" the AI injects into its actions. A high value of 0.05 forces the robot to aggressively explore new paths, ensuring it discovers routes out of complex mazes instead of settling for local minima. |
| **`gamma`** | `0.995` | (Discount Factor). This dictates how far into the future the AI looks. By increasing this to 0.995, we force the AI to care heavily about the long-term `+10` dropoff reward, preventing it from being distracted by the immediate APF repulsive forces. |
| **`n_steps`** | `4096` | This increases the amount of data collected before the network updates, stabilizing the gradients for the complex LSTM network. |

### The Logging & Checkpoint System
* **`VecMonitor`:** We explicitly wrapped the parallel environments in a Monitor. This mathematical hook forces the environment to track the raw `ep_rew_mean` (Reward) and `ep_len_mean` (Speed) and stream them directly into TensorBoard for live graphing.
* **`CheckpointCallback`:** To ensure server stability during the overnight run, a backup `.zip` model is saved every `250,000` steps.

This optimized, recurrent pipeline represents the pinnacle of autonomous robotics navigation training.
