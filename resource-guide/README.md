# Resource Guide

Foundational material to learn before working on HiveMind, roughly in order. Each entry says why it matters for this repo.

## 1. Reinforcement learning fundamentals
- **Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.)**, free at http://incompleteideas.net/book/the-book-2nd.html
  - Ch. 3 (MDPs, returns, discounting), Ch. 6 (TD learning), Ch. 13 (policy gradients).
  - The vocabulary for everything else here.
- **OpenAI Spinning Up**, https://spinningup.openai.com
  - "Key Concepts" and "Intro to Policy Optimization" sections.
  - The clearest short path from RL basics to policy-gradient methods.
- **David Silver's RL course (UCL / DeepMind)**: lecture videos and slides, https://www.davidsilver.uk/teaching/

## 2. PPO, advantage estimation, recurrent policies
- **Schulman et al., 2017, *Proximal Policy Optimization Algorithms***, arXiv:1707.06347. The algorithm this repo trains with.
- **Schulman et al., 2015, *High-Dimensional Continuous Control Using Generalized Advantage Estimation***, arXiv:1506.02438. GAE(λ = 0.95) is used here.
- **Huang et al., *The 37 Implementation Details of PPO***, ICLR Blog Track 2022, https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
  - Explains clip fraction, approximate KL, advantage normalisation and LSTM handling.
- **Hochreiter & Schmidhuber, 1997, *Long Short-Term Memory***, and Hausknecht & Stone, 2015, *Deep Recurrent Q-Learning for POMDPs* (arXiv:1507.06527): why memory helps under partial observability.

## 3. Reward design
- **Ng, Harada & Russell, 1999, *Policy Invariance Under Reward Transformations***, ICML.
  - Potential-based shaping, the theory behind this repo's Φ and F terms.
- **Curriculum learning:** Bengio et al., 2009, *Curriculum Learning* (ICML); Narvekar et al., 2020, *Curriculum Learning for RL Domains: A Framework and Survey* (JMLR).

## 4. Multi-agent RL
- **Albrecht, Christianos & Schäfer, *Multi-Agent Reinforcement Learning: Foundations and Modern Approaches*** (MIT Press, 2024), free at https://www.marl-book.com. Dec-POMDPs, parameter sharing, CTDE.
- **Yu et al., 2022, *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games* (MAPPO)**, arXiv:2103.01955. The recommended next step for this repo's critic.
- **Gupta, Egorov & Kochenderfer, 2017, *Cooperative Multi-Agent Control Using Deep RL***: the parameter-sharing approach used here.
- **Papoudakis et al., 2021, *Benchmarking MARL Algorithms in Cooperative Tasks*** (arXiv:2006.07869). Its Robotic Warehouse (RWARE) environment is a close relative of this task.

## 5. Emergent communication
- **Foerster et al., 2016, *Learning to Communicate with Deep Multi-Agent RL*** (arXiv:1605.06676); **Sukhbaatar et al., 2016, *CommNet*** (arXiv:1605.07736).
- **Lowe et al., 2019, *On the Pitfalls of Measuring Emergent Communication*** (arXiv:1903.05168).
  - Why this repo uses intervention tests and mutual information rather than task reward to judge communication.
- **Eccles et al., 2019, *Biases for Emergent Communication in Multi-agent RL*** (arXiv:1912.05676). Positive signalling/listening losses, a suggested next step.
- **Jiang & Lu, 2018, *Learning Attentional Communication* (ATOC)**, and **Vaswani et al., 2017, *Attention Is All You Need*** (arXiv:1706.03762): background for the message-attention module.

## 6. Tools used in this repo
- **Gymnasium** (environment API): https://gymnasium.farama.org
- **Stable-Baselines3** (PPO, VecEnv, callbacks): https://stable-baselines3.readthedocs.io. The *Custom Policy Network* and *Callbacks* guides map directly onto `models.py` and `training.py`.
- **sb3-contrib RecurrentPPO**: https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html
- **PyBullet Quickstart Guide**: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA. Covers loading URDFs, ray tests, contact points and cameras, all used in `env.py`.
- **URDF tutorials (ROS wiki)**: http://wiki.ros.org/urdf/Tutorials, for reading and editing `hivemind_env/assets/*.urdf`.
- **PyTorch tutorials**: https://pytorch.org/tutorials, covering `nn.Module`, `Conv1d` and `MultiheadAttention`.
- **TensorBoard**: https://www.tensorflow.org/tensorboard/get_started

