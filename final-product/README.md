# HiveMind: Cooperative Warehouse Robots with Multi-Agent RL and Emergent Communication

Four simulated warehouse robots learn, from reward alone, to find 12 cartons in a randomly laid-out warehouse, pick them up and deliver them to a shared depot, while avoiding each other.

- **One shared policy** (Recurrent PPO with an LSTM) controls all four robots.
- **Partial observability:** the robots get **no map** and sense shelves only through a noisy 270° LiDAR.
- **Message channel:** each robot can broadcast one of 16 tokens every step. The project tests, as a controlled ablation (**no comm vs comm**), whether a useful protocol emerges.

| Demo | File |
|---|---|
| Comm model, full 12-carton task | `media/demo_videos/demo_comm.mp4` |
| No-comm model, same warehouse | `media/demo_videos/demo_nocomm.mp4` |
| Comm model with incoming messages zeroed (intervention test) | `media/demo_videos/demo_comm_zeroed.mp4` |

The complete technical reference is [`docs/walkthrough.md`](docs/walkthrough.md).

---

## 1. Setup

**Requirements:** Python ≥ 3.10 (tested on 3.12 and 3.14) and Linux/macOS/Windows. A GPU is optional: training is CPU/physics-bound, so cores matter more. `ffmpeg` is needed only for videos.

```bash
cd final-product
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt tensorboard matplotlib pytest
```

**Dependencies:**
- `requirements.txt`: gymnasium, numpy, pybullet, stable-baselines3, sb3-contrib, torch, tqdm, rich, python-dotenv.
- `tensorboard`: training curves.
- `matplotlib`: graphs.
- `pytest`: tests.

**Two things to know:**
- **Run every command from `final-product/`** using `python -m scripts.<name>`.
- **ROS machines:** if ROS is sourced in your shell, run `unset PYTHONPATH` first, otherwise imports fail with `No module named 'catkin_pkg'`.

## 2. How to run

```bash
# sanity checks
python -m pytest tests/test_environment.py                   # 11 environment/communication tests
python -m scripts.train --smoke --communication              # 1-minute end-to-end training check

# train (comm arm); see docs/training_instructions.md for the exact server procedure
python -m scripts.train --communication --curriculum --num-cartons 1 \
  --worlds 8 --timesteps <T> --seed 0 --run-name <name>
tensorboard --logdir tensorboard_logs

# evaluate (100 headless episodes, identical layouts across variants)
python -m scripts.evaluate_ablation --variant comm        --cartons 12 --episodes 100 --out eval/episodes.jsonl
python -m scripts.evaluate_ablation --variant nocomm      --cartons 12 --episodes 100 --out eval/episodes.jsonl
python -m scripts.evaluate_ablation --variant comm_zeroed --cartons 12 --episodes 100 --out eval/episodes.jsonl
python -m scripts.make_presentation_graphs --eval-dir eval --comm-log tensorboard_logs/ppo_recurrent_comm_s0_2

# watch / record
python scripts/inference.py --episodes 5 --num-cartons 12 --model-path models/ppo_recurrent_comm_s0_final.zip
python scripts/inference.py --episodes 1 --render --model-path models/ppo_recurrent_comm_s0_final.zip   # PyBullet GUI
python -m scripts.record_demo comm 1069 media/demo_videos/demo_comm.mp4                                # top-down MP4
```

**Notes on the commands:**
- **Curriculum:** `--curriculum` must be combined with `--num-cartons 1`. The curriculum then promotes 1 → 2 → 4 → 8 → 12 cartons at 85% rolling success.
- **Models:** load trained weights with `RecurrentPPO.load(path, custom_objects=hivemind_env.training.INFERENCE_CUSTOM_OBJECTS)`.

## 3. How the code is structured

```
final-product/
├── hivemind_env/                 Python package
│   ├── env.py                    HiveMindMultiAgentEnv: PyBullet warehouse, step logic, LiDAR, reward, observation, messages
│   ├── models.py                 HiveMindExtractor (MLP + 1-D CNN + message attention), policy kwargs
│   ├── vec_env.py                in-process VecEnv: each 4-robot world → 4 policy slots (parameter sharing)
│   ├── subproc_vec_env.py        same API, one OS process per world (default for training)
│   ├── training.py               curriculum + metrics callbacks, LR schedules, model loading, device selection
│   ├── greedy.py                 scripted BFS controller (map-aware reference / demonstrator)
│   └── assets/                   URDFs: robot, carton, shelves 1–7 m, shelf generator
├── scripts/
│   ├── train.py                  training entry point
│   ├── evaluate_ablation.py      comm / nocomm / comm_zeroed evaluation → JSON lines
│   ├── make_presentation_graphs.py  training comparisons, evaluation and message-analysis graphs
│   ├── inference.py              evaluation printout, GUI, GIF/MP4
│   ├── evaluate_all.py           inference at 4 / 8 / 12 cartons
│   ├── record_demo.py            top-down demo videos
│   └── diagnostics/              reward/observation verification, policy probes, log watcher, smoke test
├── tests/                        environment tests + post-training threshold checks
├── models/                       ppo_recurrent_final.zip (no comm), ppo_recurrent_comm_s0_final.zip (comm)
├── tensorboard_logs/             training logs of both arms
├── presentation_graphs/          all result graphs
├── media/demo_videos/            demo videos
└── docs/
    ├── walkthrough.md            complete technical reference (start here to continue development)
    ├── training_instructions.md  exact server setup + training procedure
    ├── HiveMind_presentation.pptx  project presentation
    └── diagrams/                 1 system architecture → 2 environment step → 3 network architecture
```

**Architecture diagrams, high-level to low-level** (`docs/diagrams/`, regenerate with `python docs/diagrams/make_diagrams.py`):
1. `1_system_architecture.png`: simulation workers, PPO learner, callbacks and outputs.
2. `2_env_step_flow.png`: what one `env.step()` does, in order.
3. `3_network_architecture.png`: every layer of the policy, with verified dimensions.

## 4. Key design facts
- **Observation:** a fixed 177-d vector per robot.
  - own state (6), teammates (12), carton status and positions (36), depot and time (3), LiDAR (72), messages (48);
  - the width is pinned by import-time, constructor and runtime checks.
- **Action:** `MultiDiscrete([7, 16])` per robot, i.e. a move (forward, backward, left, right, pick, drop, stay) and a message token.
- **Reward:** r = 0.8·shared + 0.2·individual + 30·(Φ(s′) − Φ(s)).
  - Shared: +10 per delivery, +100 completion, makespan bonus, −1 per collision, −0.05 per step.
  - Individual: +1 pick-up, +2 delivery, −5 for a collision the robot caused, −0.5 invalid action, −0.02 idle.
  - Φ = −(cartons left + 0.5·not carrying + distance/26).
- **Network:** separate actor and critic extractors (MLP 57→128, CNN 72→576, attention 48→64, concat 768→256) → LSTM 256 → MLP 128·128 → 23 logits / V. 1.64 M parameters.
- **Results:**
  - Both arms solve the full 12-carton task in 100% of 100 evaluation episodes.
  - The message intervention test and token statistics (entropy 3.80 / 4 bits, I(token; carrying) = 0.013 bits) show the comm model does **not** rely on its channel.
  - Full results are in `docs/walkthrough.md` §20.

## 5. Known limitations and next steps

**Limitations:**
- **Motion:** grid-based (one-cell moves, 90° turns) with kinematic interpolation, not torque control.
- **Observability:** teammate poses and carton positions are observed; only the map and obstacles are hidden. This redundancy weakens the incentive to communicate.
- **Critic:** decentralised. **Statistics:** one training run per arm.
- **Code issues** (details in `docs/walkthrough.md` §22):
  - the learning-rate restart on curriculum promotion is a no-op;
  - two TensorBoard metrics (`pickups_per_episode`, `deliveries_per_episode`) always read 0;
  - `--gamma` and `--init-from` in `train.py` do not behave as named;
  - `inference.py` counts collisions 4×.

**Next steps:**
1. Hide teammate and carton information beyond sensor range, so messages must carry it.
2. Add communication incentives (positive listening/signalling, a mutual-information bonus).
3. Use a centralised (MAPPO) critic.
4. Run multiple seeds per arm.
5. Randomise the carton count per episode.
6. Move to continuous control, then Sim2Real via ROS 2.

**Earlier iterations, abandoned approaches and old models/logs** are archived in [`../learning/iteration-phase/`](../learning/iteration-phase/). Learning resources are in [`../resource-guide/`](../resource-guide/).
