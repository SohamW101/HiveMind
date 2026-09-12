# Recommendations and Next Steps

To maximize the performance of the HiveMind MARL environment and prepare it for Step 7 (Emergent Communication), the following steps should be executed.

## 1. Architectural Transition: CTDE and MAPPO
Currently, the shared PPO uses a decentralized critic. 
- **Action**: Implement a true MAPPO actor-critic split.
- **How**: The Actor network receives the 177-dim local observation. The Critic network receives a concatenated global state: `(177 * 4) = 708-dim` vector representing the absolute truth of the entire warehouse.
- **Why**: This drastically reduces variance during training. The Critic no longer has to guess what the other 3 robots are doing based on relative LiDAR; it knows exactly where they are.

## 2. Introduce Recurrence (RNN/LSTM)
The environment is treated as a Markov Decision Process (MDP), but due to LiDAR occlusion, it is actually a Partially Observable MDP (POMDP).
- **Action**: Wrap the `HiveMindExtractor` output into an LSTM cell.
- **Why**: A robot needs to remember that it saw a carton down an aisle 3 steps ago, even if it turned around. PPO alone cannot hold memory; Recurrent PPO (using `stable-baselines3.RecurrentPPO` or a custom Ray RLlib implementation) is strictly required for optimal routing in occluded warehouses.

## 3. Implement Emergent Communication (Roadmap Step 7)
The 48 zeroed slots are waiting for data.
- **Action**: 
  1. Add a communication action to the `MultiDiscrete` action space.
  2. The policy outputs a discrete token $[0, 15]$.
  3. In `env.step()`, take the token from Robot A, encode it, and write it into the `messages` slice of the observation vector for Robots B, C, and D for the *next* step.
- **Research Backing**: As identified in our literature review, low-bandwidth, discrete channels force agents to agree on high-level compositional structures (e.g., Token 5 = "I am taking the top aisle").

## 4. Codebase Maintenance
- **Urgent Fix**: Implement the `_closed` flag in `env.py`'s `close()` method to prevent PyBullet teardown crashes.
- **Urgent Fix**: Refactor `train.py` so that `args.batch_size` is a function of `args.n_steps * slots`, preventing accidental micro-batches when testing with `worlds=1`. 
- **Documentation**: Remove stale warning text in `smoke_test.py` and `run_evaluation.py` to prevent developer confusion regarding Phase 3 completion.
