# 10_Architecture_Deep_Dive

This document details the multi-agent neural network architecture that enabled the HiveMind agents to achieve a 100% success rate in the collaborative delivery environment.

## 1. The Core Problem
In the initial single-agent implementation, scaling to multiple agents resulted in **Value Loss Explosions**. The agents were unable to attribute rewards correctly because multiple agents were acting simultaneously in a shared environment, creating non-stationarity.

## 2. The Solution: Decoupled Actor-Critic with Transformer Communication

To solve this, we implemented a sophisticated custom feature extractor: the `HiveMindExtractor`.

### A. The `MessageAttention` Module (Transformer)
To allow agents to implicitly communicate and coordinate their actions (e.g., avoiding each other, yielding at choke points), we implemented a Multihead Attention mechanism.
*   **Input:** The local LiDAR and state observations of all agents.
*   **Mechanism:** The `MessageAttention` class projects these observations into Queries, Keys, and Values. It computes attention weights, allowing each agent to "attend" to the most critical features of the other agents' states.
*   **Output:** A context vector that is concatenated with the agent's own local observation.

### B. Decoupled Architectures
We completely separated the feature extraction pipelines for the Actor (Policy) and the Critic (Value) networks.
*   **Critic Extractor:** Has full global visibility. It processes the concatenated states of all agents to accurately estimate the value of a state. This stabilizes the value loss because the Critic is no longer blind to the actions of other agents.
*   **Actor Extractor:** Relies on the `MessageAttention` output, allowing it to formulate a policy based on collaborative context without having full "god-mode" global state (which would overfit).

### C. Recurrent PPO (LSTM)
We upgraded from standard PPO to `RecurrentPPO` (`sb3-contrib`). By injecting LSTM cells into the policy network, the agents gained "memory." This was critical for navigating complex obstacles where the optimal path required remembering past LiDAR scans.

### D. Temporal Horizon Tuning
We adjusted the discount factor (`gamma`) from the default `0.99` to `0.999`. In our environment, episodes can last hundreds of steps. A `gamma` of `0.99` caused the agents to aggressively discount future rewards, leading to suboptimal, short-sighted behavior. `0.999` allowed them to plan for the long-term goal of carton delivery.
