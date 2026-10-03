# Future Roadmap & Extension Plans

The current model has achieved 100% success on the 12-carton objective. To push the HiveMind architecture further toward real-world applicability (such as competing in the CUHK-X Challenge or deploying on physical hardware), the following milestones are proposed for future development.

## Phase 1: Dynamic Generalization
Currently, the model was trained sequentially on 1->2->4->8->12 cartons, which caused "Curriculum Overfitting" (it forgot how to solve 4 cartons efficiently because it optimized exclusively for 12).
- **Plan:** Implement **Domain Randomization** during training. Instead of a fixed ladder, the environment should randomly spawn between 4 and 16 cartons every episode.
- **Goal:** Create a globally generalized policy that performs perfectly regardless of the sparsity or density of the warehouse floor.

## Phase 2: Decentralized Inference
The Shared Policy LSTM relies on the `HiveMindSharedPolicyVecEnv` to sync all 4 observations in a single Python thread. 
- **Plan:** Extract the PyTorch weights and build a ROS2 (Robot Operating System) node script. Run 4 isolated Python processes that communicate their `(X, Y, Theta)` via ROS Topics.
- **Goal:** Prove the policy can execute in a truly distributed, asynchronous system without a centralized environment wrapper dictating the physics steps.

## Phase 3: Obstacle Complexity
The warehouse shelves are currently static. 
- **Plan:** Introduce dynamic obstacles (e.g., humans walking through the warehouse, random forklift crossings).
- **Goal:** Force the LSTM to rely heavily on its LiDAR raycasts to avoid unpredictable dynamic objects, rather than memorizing static shelf locations.

## Phase 4: Sim2Real Transfer
- **Plan:** Port the PyBullet parameters (mass, friction, differential drive constraints) to match physical TurtleBot or custom AGV chassis. 
- **Goal:** Deploy the `ppo_recurrent_final.zip` weights onto physical edge compute devices (e.g., Jetson Nano) to validate the reinforcement learning in physical space.
