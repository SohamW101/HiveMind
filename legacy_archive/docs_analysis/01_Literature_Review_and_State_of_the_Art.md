# Literature Review and State of the Art: Multi-Agent Warehouse RL

In evaluating the HiveMind Multi-Agent Reinforcement Learning (MARL) approach, we benchmark our architecture and algorithms against the latest academic research. We primarily focus on state-of-the-art developments in multi-agent routing, cooperative exploration, and emergent communication (EC).

## 1. Multi-Agent Reinforcement Learning (MARL) in Constrained Environments

The current HiveMind implementation uses a parameter-shared PPO with a decentralized critic ("Cheap MAPPO"). 

**Recent Advancements in the Field:**
- **Path-Guided MAPPO (PGF-MAPPO)**: As detailed in the recent paper *Generalizable Collaborative Search-and-Capture in Cluttered Environments via Path-Guided MAPPO and Directional Frontier Allocation* (ArXiv 2512.09410), standard MARL struggles significantly with sparse rewards and constrained fields of view (FOV) in cluttered spaces (like warehouses). The researchers resolved this by integrating A*-based potential fields for **dense reward shaping**, alongside a parameter-shared decentralized critic. 
  - **HiveMind Comparison**: Our implementation actively utilizes **potential-based reward shaping** derived from Ng, Harada & Russell (1999) (`Phi = -(work remaining)`). Furthermore, our use of a 1-D CNN over LiDAR returns aligns with their findings that parameter-sharing over localized sensory sweeps scales efficiently at $O(1)$ complexity for robotic swarms.

- **Local Cooperation Rewards in MAPPO (MAPPO-LCR)**: *MAPPO-LCR: Multi-Agent Proximal Policy Optimization with Local Cooperation Reward in Spatial Public Goods Games* (ArXiv 2512.17187) highlights that Proximal Policy Optimization often ignores the underlying coupling of individual returns during value estimation. A true Centralized Training Decentralized Execution (CTDE) centralized critic evaluates joint configurations much more effectively.
  - **HiveMind Comparison**: Our shared critic only sees one robot's observation. While this is currently justifiable because our observation state is "close to global" (containing positions and statuses of all cartons and other robots), it remains fundamentally decentralized. True MAPPO with CTDE is the next logical step if parameter sharing plateaus.

## 2. Emergent Communication (EC)

HiveMind allocates 48 discrete slots specifically for EC (`3 robots x 16 tokens`), currently zeroed out as per Roadmap Step 7.

**State of the Art Context:**
- **Spectrum of Communication**: As defined in *Low-Bandwidth Communication Emerges Naturally in Multi-Agent Learning Systems* (ArXiv 2011.14890), communication in MARL scales from low-bandwidth (e.g., stigmergy, pheromone trails) to high-bandwidth (compositional language).
  - **HiveMind Implementation**: The 16-token discrete vocabulary is a medium-to-low bandwidth channel. It forces the agents to compress intention, aligning with research indicating that highly restricted communication channels encourage more robust, grounded, and generalizable protocols than unbounded continuous channels.
  
- **Taxonomy and Benchmarks**: The comprehensive survey *Emergent Language: A Survey and Taxonomy* (ArXiv 2409.02645, JAAMAS 2025) emphasizes that evaluating EC in reinforcement learning goes beyond statistical NLP representations—it requires metrics measuring *utility* and *compositionality*.
  - **HiveMind Comparison**: HiveMind's EC slots are embedded directly into the observation space alongside physical LiDAR data, processed through a multi-branch feature extractor (MLP for messages, CNN for LiDAR). This ensures that communication acts as a direct feature of the state space, tightly coupling language emergence to the physical task of warehouse delivery.

## Conclusion

HiveMind is highly aligned with modern MARL literature. The decision to use potential-based reward shaping to combat sparsity in cluttered environments is validated by recent PGF-MAPPO research. However, the literature clearly signals that transitioning from a decentralized shared-critic PPO to a CTDE MAPPO is essential for optimizing highly coupled cooperative tasks like multi-robot delivery routing.
