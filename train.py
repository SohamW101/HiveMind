"""
Train a shared policy on the HiveMind warehouse - roadmap step 6, no communication.

All four robots run one set of weights. `HiveMindSharedPolicyVecEnv` presents each
four-robot world as four policy-facing slots, which is the project roadmap's "cheap option first";
that module documents what this is and is not (parameter sharing with a decentralised
critic, not MAPPO with a centralised one).

    # smoke run - a few thousand steps, checks the pipeline end to end
    .venv\\Scripts\\python.exe train.py --smoke

    # a real run
    .venv\\Scripts\\python.exe train.py --timesteps 5000000 --worlds 8

    # watch it
    tensorboard --logdir tensorboard_logs

WHAT THIS RUN CANNOT TELL YOU

There is no greedy baseline yet (roadmap step 5). Rising reward here means the policy is
learning *something*; it does not mean the policy is good, because nothing says what a
competent robot would have scored on the same seeds. Build the baseline before reading
anything into these curves - that ordering is the whole point of step 5 preceding step 6.

Communication is not in this run. The 48 message slots are present and zero. That is
deliberate: this run is the no-communication baseline that step 7 has to beat, and the
comparison is only clean because the observation width and the architecture are
identical across the two.
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

from hivemind_env.env import (
    DEFAULT_OBS_DIM,
    NUM_AGENTS,
    SHAPING_SCALE_DEFAULT,
    max_steps_for,
)
from hivemind_env.models import DEFAULT_POLICY_KWARGS
from hivemind_env.training import (
    INFERENCE_CUSTOM_OBJECTS,
    CurriculumCallback,
    get_device,
    linear_schedule,
    num_parallel_envs,
)
from hivemind_env.subproc_vec_env import HiveMindSubprocVecEnv
from hivemind_env.vec_env import HiveMindSharedPolicyVecEnv


def tensorboard_available() -> bool:
    """
    TensorBoard is optional and is NOT in requirements.txt on this branch.

    SB3 does not check: passing `tensorboard_log` without it installed raises
    ImportError("Trying to log data to tensorboard but tensorboard is not installed")
    from inside `learn()`, after the env and policy are already built. Probing up front
    turns that into a one-line warning and a run that still trains, which matters more
    than the logs on a machine where installing it is someone else's call.
    """
    try:
        import tensorboard  # noqa: F401
        return True
    except ImportError:
        return False


def build_env(worlds: int, difficulty: int, seed: int | None,
              backend: str = "subproc", substeps: int | None = None,
              max_steps: int | None = None, num_cartons: int | None = None,
              shaping: bool = True, gamma: float = 0.99,
              shaping_scale: float | None = None):
    """
    Wrapped in VecMonitor so ep_rew_mean / ep_len_mean reach TensorBoard.

    `subproc` puts each warehouse in its own process. Training is 95% PyBullet physics
    on one core, so this is the difference between using one core and using the machine
    - measured 4.3x at 12 workers on a 16-thread box. `inprocess` keeps everything
    sequential here: slower, but a traceback inside a worker is far harder to read.
    """
    cls = HiveMindSubprocVecEnv if backend == "subproc" else HiveMindSharedPolicyVecEnv
    kwargs = {"shaping": shaping, "gamma": gamma}
    if substeps is not None:
        kwargs["substeps"] = substeps
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    if num_cartons is not None:
        kwargs["num_cartons"] = num_cartons
    if shaping_scale is not None:
        kwargs["shaping_scale"] = shaping_scale
    vec = cls(
        num_worlds=worlds, difficulty_level=difficulty,
        obs_dim=DEFAULT_OBS_DIM, seed=seed, **kwargs,
    )
    return VecMonitor(vec)


def main():
    p = argparse.ArgumentParser(description="Train the shared HiveMind policy")
    p.add_argument("--timesteps", type=int, default=2_000_000,
                   help="Total robot-steps, not world-steps - see the note on --worlds.")
    p.add_argument("--worlds", type=int, default=None,
                   help="Parallel warehouses. Each contributes 4 policy slots, so the "
                        "effective batch is 4x this. Defaults to a modest CPU-derived "
                        "number rather than the single-agent branch's habit of 16, "
                        "which would be 64 robot-streams here.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--n-steps", type=int, default=512,
                   help="Rollout length PER SLOT. The buffer is n_steps x 4 x worlds.")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--difficulty", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--curriculum", action="store_true",
                   help="Promote on rolling success along the TRAINING ladder "
                        "1 -> 2 -> 4 -> 8 -> 12 cartons. Start at --num-cartons 1: an "
                        "episode ends only when EVERY carton is delivered, so at 4 the "
                        "terminal bonus is never reached by an untrained policy and the "
                        "critic never sees one. At 1 a random policy completes 33%%.")
    p.add_argument("--init-from", default=None,
                   help="Warm-start the policy from a saved checkpoint, typically the "
                        "behaviour-cloned one from scripts/pretrain_bc.py. The critic is "
                        "NOT cloned, so expect the first iterations to dip before they "
                        "improve.")
    p.add_argument("--run-name", default=None)
    p.add_argument("--save-dir", default="models")
    p.add_argument("--log-dir", default="tensorboard_logs")
    p.add_argument("--checkpoint-every", type=int, default=250_000)
    p.add_argument("--max-steps", type=int, default=None,
                   help="Episode cap. Defaults per carton count - 150 / 250 / 400 at "
                        "4 / 8 / 12 - which is roughly 3x greedy's makespan at each. "
                        "The old flat 2000 made terminal rewards vanish under "
                        "discounting: 0.99^2000 = 1.9e-9.")
    p.add_argument("--num-cartons", type=int, default=None,
                   help="Cartons in play (env default 12). Start at 1 with --curriculum; "
                        "greedy scores makespan 23 at 4, 58 at 8, 97 at 12.")
    p.add_argument("--shaping-scale", type=float, default=None,
                   help="Strength of the potential-based shaping (env default 30.0). "
                        "Sweep it with scripts/diagnose_incentives.py --shaping-scale "
                        "before training on a new value; the gates there catch a scale "
                        "that makes a pickup unprofitable.")
    p.add_argument("--no-shaping", action="store_true",
                   help="Train against the specification's sparse reward exactly. It "
                        "provably does not learn from here - a 5M-step run finished "
                        "below the do-nothing floor - so this is for ablations only.")
    p.add_argument("--substeps", type=int, default=None,
                   help="Physics substeps per env step (env default: 5 headless, 30 GUI). "
                        "An interpolation, not the motion model - makespan, collisions, "
                        "deliveries and completion are identical at any value. Lower is "
                        "faster; 1 risks tunnelling past the 6 cm shelf posts.")
    p.add_argument("--backend", choices=["subproc", "inprocess"], default="subproc",
                   help="subproc: one process per warehouse (default; the only way to "
                        "use more than one core). inprocess: everything sequential in "
                        "this process - slower, but debuggable.")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny run that exercises every code path in a minute or two.")
    args = p.parse_args()

    if args.smoke:
        args.timesteps = 4096
        args.worlds = args.worlds or 2
        args.n_steps = 128
        args.batch_size = 256
        args.checkpoint_every = 10 ** 9  # no mid-run checkpoints in a smoke run

    worlds = args.worlds or max(2, min(8, num_parallel_envs(cap=8)))
    slots = worlds * NUM_AGENTS
    run_name = args.run_name or f"ppo_shared_{datetime.now():%Y%m%d_%H%M%S}"

    device = get_device()
    os.makedirs(args.save_dir, exist_ok=True)

    tb_log = args.log_dir if tensorboard_available() else None
    if tb_log is None:
        print("  [warn] tensorboard is not installed - training will run, but no "
              "curves will be written.")
        print(r"         Install it with: "
              r".venv\Scripts\python.exe -m pip install tensorboard")

    print("=" * 78)
    print("  HiveMind - shared-policy PPO, no communication (roadmap step 6)")
    print("=" * 78)
    print(f"  run          : {run_name}")
    print(f"  worlds       : {worlds}  ->  {slots} policy slots ({NUM_AGENTS} per world)")
    print(f"  observation  : {DEFAULT_OBS_DIM} floats per robot")
    print(f"  timesteps    : {args.timesteps:,} robot-steps")
    print(f"  rollout      : {args.n_steps} per slot  ->  buffer {args.n_steps * slots:,}")
    print(f"  batch size   : {args.batch_size}")
    print(f"  lr           : {args.lr:.0e}, linear decay to 0")
    print(f"  device       : {device}")
    print(f"  tensorboard  : {tb_log or 'disabled (not installed)'}")
    print(f"  substeps     : {args.substeps if args.substeps else '5 (env default)'}")
    print(f"  max steps    : {args.max_steps or f'{max_steps_for(args.num_cartons)} (per carton count)'}")
    print(f"  cartons      : {args.num_cartons or '12 (env default)'}")
    print(f"  shaping      : {'off - sparse spec reward' if args.no_shaping else f'on, scale {args.shaping_scale or SHAPING_SCALE_DEFAULT}'}")
    print(f"  backend      : {args.backend}"
          + ("" if args.backend == "subproc" else "  (sequential - one core only)"),
          flush=True)

    env = build_env(worlds, args.difficulty, args.seed, args.backend,
                    args.substeps, args.max_steps, args.num_cartons,
                    shaping=not args.no_shaping, gamma=args.gamma,
                    shaping_scale=args.shaping_scale)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=linear_schedule(args.lr),
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,          # the action space is small and delivery is sparse
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=DEFAULT_POLICY_KWARGS,
        tensorboard_log=tb_log,
        seed=args.seed,
        device=device,
        verbose=1,
    )

    if args.init_from:
        # Load the weights, not the algorithm: a BC checkpoint carries no useful
        # optimiser or schedule state, and its env spaces must match this run's.
        from stable_baselines3 import PPO as _PPO
        donor = _PPO.load(args.init_from, device=device,
                          custom_objects=INFERENCE_CUSTOM_OBJECTS)
        model.policy.load_state_dict(donor.policy.state_dict())
        print(f"  warm start   : policy weights from {args.init_from}", flush=True)

    n_params = sum(q.numel() for q in model.policy.parameters())
    print(f"  policy params: {n_params:,}\n", flush=True)

    callbacks = []
    if args.curriculum:
        callbacks.append(CurriculumCallback(initial_lr=args.lr, check_freq=2048))
    if args.checkpoint_every < args.timesteps:
        callbacks.append(CheckpointCallback(
            save_freq=max(1, args.checkpoint_every // slots),
            save_path=os.path.join(args.save_dir, "checkpoints"),
            name_prefix=run_name,
        ))

    started = time.time()
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks or None,
            tb_log_name=run_name,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\n[train] interrupted - saving what we have", flush=True)
    finally:
        final = os.path.join(args.save_dir, f"{run_name}_final.zip")
        model.save(final)
        env.close()

    elapsed = time.time() - started
    steps = model.num_timesteps
    print(f"\n  saved      : {final}")
    print(f"  wall clock : {elapsed / 60:.1f} min for {steps:,} robot-steps "
          f"({steps / max(elapsed, 1e-9):.0f}/s)")
    print("\n  Next: build the greedy baseline (roadmap step 5) and compare makespan.")
    print("  Until that number exists, a rising reward curve is not evidence of much.")


if __name__ == "__main__":
    # The __main__ guard is load-bearing, not decoration: the subprocess backend uses
    # the "spawn" start method, which re-imports this module in every worker. Without
    # the guard each worker would start its own training run.
    main()
