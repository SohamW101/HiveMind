"""
Train the shared HiveMind policy. All four robots run one set of weights.

    train.py --smoke                                   # pipeline check, ~1 minute
    train.py --num-cartons 1 --curriculum --timesteps 5000000 --worlds 12
    tensorboard --logdir tensorboard_logs

`HiveMindSharedPolicyVecEnv` presents each four-robot world as four policy slots and
documents what that is and is not - parameter sharing with a decentralised critic, not
MAPPO with a centralised one.

READ THE CURVES AGAINST THE BASELINE, NOT AGAINST ZERO. Greedy's makespan is 23 / 59 / 98
at 4 / 8 / 12 cartons; a rising reward curve on its own says the policy learned
something, not that it is any good.

COMMUNICATION IS OFF BY DEFAULT (roadmap step 7). Without `--comms` this is the step-6
baseline: message slots present and zero, Discrete(7) per slot, every pre-step-7
checkpoint still loadable. With it, each robot also emits one of 16 tokens per step and
the slots carry what the other three said one step earlier, so the per-slot action space
becomes MultiDiscrete([7, 16]). The observation width and the network are identical
either way, so the two runs must differ in exactly one flag:

    train.py --run-name nocomm --num-cartons 1 --curriculum --timesteps 5000000
    train.py --run-name comms  --num-cartons 1 --curriculum --timesteps 5000000 --comms

Watch comms/token_entropy_bits and comms/mi_carrying_bits together, never alone: PPO's
entropy bonus pushes the token head toward uniform, so high entropy on its own is the
null result rather than the positive one. See MessageStatsCallback.
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
    MSG_DROPOUT_DEFAULT,
    MSG_TOKENS,
    NUM_AGENTS,
    SHAPING_SCALE_DEFAULT,
    max_steps_for,
)
from hivemind_env.models import DEFAULT_POLICY_KWARGS
from hivemind_env.training import (
    INFERENCE_CUSTOM_OBJECTS,
    CurriculumCallback,
    MessageStatsCallback,
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
              shaping_scale: float | None = None, comms: bool = False,
              msg_dropout: float = MSG_DROPOUT_DEFAULT):
    """
    Wrapped in VecMonitor so ep_rew_mean / ep_len_mean reach TensorBoard.

    `subproc` puts each warehouse in its own process. Training is 95% PyBullet physics
    on one core, so this is the difference between using one core and using the machine
    - measured 4.3x at 12 workers on a 16-thread box. `inprocess` keeps everything
    sequential here: slower, but a traceback inside a worker is far harder to read.
    """
    cls = HiveMindSubprocVecEnv if backend == "subproc" else HiveMindSharedPolicyVecEnv
    kwargs = {"shaping": shaping, "gamma": gamma, "comms": comms}
    if comms:
        kwargs["msg_dropout"] = msg_dropout
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
                   help="Warm-start the policy weights from a saved checkpoint. The "
                        "critic is NOT cloned, so expect the first iterations to dip. "
                        "The action space must match, so a silent checkpoint cannot "
                        "seed a --comms run or the reverse.")
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
    p.add_argument("--comms", action="store_true",
                   help="Turn on the 16-token broadcast channel (roadmap step 7). Each "
                        "robot emits one token per step alongside its movement, and "
                        "hears the other three one step later. The observation width "
                        "and the network are identical with and without this, so the "
                        "run it must be compared against is the SAME command without "
                        "the flag.")
    p.add_argument("--msg-dropout", type=float, default=MSG_DROPOUT_DEFAULT,
                   help=f"Probability each listener-speaker link drops a message "
                        f"(default {MSG_DROPOUT_DEFAULT}). The roadmap asks for ~10%% so "
                        f"the protocol is not brittle. Ignored without --comms.")
    p.add_argument("--ent-coef", type=float, default=0.01,
                   help="PPO entropy bonus. It matters more with --comms than without: "
                        "SB3 sums the entropy of BOTH heads, and the token head's "
                        "maximum is ln(16)=2.77 against the movement head's ln(7)=1.95, "
                        "so the bonus is a standing pressure toward a uniform - that is, "
                        "meaningless - token distribution. Lower it to ~0.003 if "
                        "comms/mi_carrying_bits stays at zero while "
                        "comms/token_entropy_bits sits at its 4.0 ceiling.")
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
    print("  HiveMind - shared-policy PPO"
          + (" WITH communication (roadmap step 7)" if args.comms
             else ", no communication (roadmap step 6)"))
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
    print("  comms        : "
          + (f"ON - {MSG_TOKENS} tokens, {args.msg_dropout:.0%} link dropout, "
             f"MultiDiscrete([7, {MSG_TOKENS}]) per slot"
             if args.comms else "off - message slots zero (step 6 baseline)"))
    print(f"  ent_coef     : {args.ent_coef}")
    print(f"  backend      : {args.backend}"
          + ("" if args.backend == "subproc" else "  (sequential - one core only)"),
          flush=True)

    env = build_env(worlds, args.difficulty, args.seed, args.backend,
                    args.substeps, args.max_steps, args.num_cartons,
                    shaping=not args.no_shaping, gamma=args.gamma,
                    shaping_scale=args.shaping_scale, comms=args.comms,
                    msg_dropout=args.msg_dropout)

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
        ent_coef=args.ent_coef,  # see --ent-coef; it is not neutral once comms are on
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
        # The action head's shape is part of the weights. A no-comms donor has one
        # 7-way head and a comms policy has a 7-way plus a 16-way one, so mixing them
        # fails - as a torch size mismatch several frames deep, which is a poor way to
        # find out.
        if donor.action_space != env.action_space:
            raise SystemExit(
                f"--init-from {args.init_from} has action space {donor.action_space}, "
                f"but this run's is {env.action_space}.\n"
                f"A checkpoint trained without --comms cannot warm-start a run with it, "
                f"or the reverse: the token head does not exist in one of them.\n"
                f"Train the communicating arm from scratch - and note that the whole "
                f"point of the comparison is that both arms start from the same place."
            )
        model.policy.load_state_dict(donor.policy.state_dict())
        print(f"  warm start   : policy weights from {args.init_from}", flush=True)

    n_params = sum(q.numel() for q in model.policy.parameters())
    print(f"  policy params: {n_params:,}\n", flush=True)

    callbacks = []
    if args.comms:
        callbacks.append(MessageStatsCallback(check_freq=2048))
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
    if args.comms:
        print(f"\n  Next: measure whether the channel is a protocol or decoration -")
        print(f"    scripts/analyse_messages.py {final} --num-cartons <n>")
        print("  Emitting varied tokens is not the result. Losing makespan when those")
        print("  tokens are shuffled is.")
    else:
        print("\n  Next: score it against the greedy baseline (makespan 98 at 12 cartons),")
        print("  then run the identical command with --comms and compare the two.")


if __name__ == "__main__":
    # The __main__ guard is load-bearing, not decoration: the subprocess backend uses
    # the "spawn" start method, which re-imports this module in every worker. Without
    # the guard each worker would start its own training run.
    main()
