# Training Analysis & Metrics Logs

## 1. Curriculum Learning Strategy
Training 4 robots to navigate a dense environment using sparse rewards (only receiving a point upon delivery) is mathematically impossible due to the sheer size of the state space. 

To solve this, HiveMind utilized a **5-Stage Curriculum Ladder**:
- **Stage 1 (1 Carton):** Easy mode. Model learns basic pickup and drop-off mechanics without distractions.
- **Stage 2 (2 Cartons):** Minor obstacle avoidance.
- **Stage 3 (4 Cartons):** Multi-agent pathing challenges begin.
- **Stage 4 (8 Cartons):** High congestion. LSTM networks learn to predict and yield to other robots.
- **Stage 5 (12 Cartons - Full Task):** Intense swarming optimization.

## 2. Reward Shaping Architecture
To guide the agents, the reward function was deeply shaped:
- `+100` points for delivering a carton (Terminal).
- `+10` points for successfully picking a carton up.
- `-0.5` penalty for dropping a carton outside the drop zone.
- `-0.1` penalty for physical collisions (robots, walls).
- `-0.01` step penalty (encouraging speed).

## 3. TensorBoard Analysis
The model was trained for over **13.5 Million Timesteps**.

### Key Observations:
1. **`rollout/success_rate`**: Peaked at 100% in Stage 5. This indicates complete mastery of the environment logic.
2. **`rollout/ep_len_mean` (Makespan)**: Spiked during curriculum transitions (e.g., jumping from 4 to 8 cartons caused confusion), but smoothly decayed back down as the model learned to optimize its paths.
3. **`metrics/collisions_per_episode`**: Initial stages saw massive collision spikes. As the LSTM began recognizing the `communication=True` observation array, collisions stabilized to acceptable limits—proving the agents learned to "yield" rather than stubbornly push through each other.
