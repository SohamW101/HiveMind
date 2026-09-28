# HiveMind: Slide Content

---

## Slide 1: Title

# HiveMind
### Teaching a team of warehouse robots to cooperate with multi-agent reinforcement learning

---

## Slide 2: Problem Statement

**The problem**
- Warehouses need several robots collecting items and bringing them to a drop-off point.
- Robots share narrow aisles, compete for the same items and get in each other's way.
- Hand-coded coordination breaks as soon as the layout changes.

**Our goal**
- 4 robots, one warehouse: find the cartons, pick them up, deliver them to the depot, as a team.
- No map is given. The layout is shuffled every episode, so robots must discover the aisles through their own sensors.
- Finish fast, and avoid collisions.

**Our approach**
- Robots *learn* the behaviour through reinforcement learning, with no hand-written rules.

---

## Slide 3: The Environment

- A simulated warehouse built in **PyBullet** (a physics engine).
- **The task:**
  - A grid floor with rows of shelves.
  - 12 cartons placed in random gaps between shelves.
  - A depot in one corner.
- **The robots:** 4 differential-drive robots, each with a gripper arm and a LiDAR sensor.
- **Actions:** move forward/back, turn, pick up, drop off, wait.
- **Partial observability:**
  - no map, and a different layout every episode;
  - obstacles are only seen through a limited-range, noisy LiDAR.
- An episode ends when all cartons are delivered or time runs out.

`[Image: top-down screenshot of the warehouse with the 4 robots]`

---

## Slide 4: RL Architecture: One Brain, Four Robots

- **Shared policy:** one neural network drives all 4 robots.
  - Each robot feeds in its own observation and acts on its own.
  - Everything any robot learns benefits the whole team.
- **What each robot observes:**
  - its own state;
  - teammates' positions;
  - carton status;
  - the depot's direction;
  - its LiDAR scan;
  - reserved slots for messages from teammates.
- **Network:** three branches, merged into memory.
  - **World features** → fully connected layers.
  - **LiDAR scan** → 1-D CNN, which spots obstacle shapes in any direction.
  - **Teammate messages** → attention.
  - All three feed an **LSTM**, so robots remember what they have already explored.
- **Algorithm:** Recurrent PPO, an actor-critic method. The actor and critic are kept separate for stability.

`[Diagram: observation → 3 branches → LSTM → actions]`

---

## Slide 5: How We Trained It

- **Reward design:**
  - Team reward for every delivery, and a big bonus for finishing the job quickly.
  - Penalties for collisions (the robot that caused it takes the blame) and for wasted time.
  - **Progress shaping:** a small reward for real progress toward the next carton or the depot, designed so it can't be gamed.
- **Curriculum learning:** start with 1 carton, then 2 → 4 → 8 → 12, moving up automatically once the team succeeds reliably.
- **Parallel simulation:** 8 warehouses run at once, one per CPU process, for faster training.
- **Communication-ready:** robots can send one of 16 tokens each step. This run is the no-communication baseline; the communication ablation comes next.
- **Tech stack:**
  - Python
  - PyBullet
  - Gymnasium
  - PyTorch
  - Stable-Baselines3 / sb3-contrib
  - TensorBoard

---

## Slide 6: Results

`[Graph: presentation_graphs/1_success_rate.png]`
`[Graph: presentation_graphs/3_episode_length.png]`
`[Graph: presentation_graphs/6_eval_success.png]`

- Trained for **13.5 M steps**; the full 12-carton task was reached at 2.5 M.
- **100% success on the full task:** 50/50 evaluation episodes, all 12 cartons delivered.
- Episodes kept getting shorter throughout training, so the team got faster.
- Lower success with only 4 cartons: the policy specialised on the full task.

*Backup graphs:* `2_cartons_delivered.png`, `4_episode_reward.png`, `5_curriculum.png`

---

## Slide 7: Demo

`[Video: 4 robots completing the 12-carton task]`

---

## Slide 8: Challenges and How We Solved Them

| Challenge | Solution |
|---|---|
| Robots learned to just stand still: moving risked collisions long before any reward | Progress-based reward shaping so every step of real progress pays |
| Picking up a carton was accidentally *punished* | Redesigned the progress measure so pick-ups and deliveries always count as progress |
| Constant shelf crashes dominated learning | Made shelves solid: a blocked move is simply refused |
| The team never finished an episode, so never saw the completion bonus | Curriculum starting from a single carton |
| Episodes too long, so the final reward was invisible to the learner | Episode length scaled to the task size |
| LiDAR slowly started seeing *through* shelves | Found the robot body sinking under gravity; fixed its height |
| Training was too slow | One process per warehouse, with lighter physics |
