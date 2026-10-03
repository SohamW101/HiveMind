# Environment Constraints & Physics

This document outlines the strict physical parameters and limitations baked into the `hivemind_env` that the Reinforcement Learning policy must solve around.

## 1. PyBullet Simulation Constraints
The environment operates inside PyBullet on a discrete stepping clock. 
- **Time Step:** 1/240th of a second per physics step.
- **Control Step:** The RL model makes decisions every 10 physics steps, yielding an effective control frequency of 24Hz. This provides enough temporal resolution for smooth driving while preventing the network from over-processing micro-adjustments.

## 2. Spatial & Boundary Constraints
The warehouse floor is a finite Euclidean plane.
- **Boundaries:** The walls are perfectly rigid. If an agent hits a wall, its kinetic energy stops. There is a penalty for hitting the wall, forcing the agent to learn boundary avoidance.
- **Carton Spawn Limits:** Cartons are spawned within a safe interior radius to prevent them from clipping into walls during initialization.
- **Drop Zones:** There are designated `(X, Y)` regions that trigger the `deliver` mechanic. Cartons dropped outside these zones incur negative reward penalties.

## 3. Kinematic Constraints (Differential Drive)
The robotic agents are modeled after real-world differential drive AGVs (Automated Guided Vehicles).
- **No Strafing (Holonomic constraint):** The agents cannot move side-to-side. 
- **Rotation:** They can rotate in place.
- **Velocity Limits:** Angular and linear velocities are clamped. The neural network outputs `[-1.0, 1.0]`, which is multiplied by maximum physical speed coefficients.

## 4. LiDAR (Raycast) Sensing Limitations
Agents do not have god-view vision. They rely on an array of raycasts (LiDAR) to sense their immediate surroundings.
- **Array Size:** 40 rays distributed radially (360 degrees).
- **Data Channels:** Each ray returns 2 values: `(Distance, ObjectType)`.
- **Constraint:** The rays have a finite max distance. If an object is beyond this distance, the agent is entirely blind to it unless it communicates with other agents.

## 5. Collision Mechanics
- **Robot-Robot:** Soft bodies with high friction. If two robots collide, they experience severe drag and potentially gridlock. This makes swarming optimization critical.
- **Robot-Carton:** Cartons are kinematic objects that the robot can "attach" to via a semantic pickup action, rather than relying on complex gripping physics which would drastically increase computational overhead.
