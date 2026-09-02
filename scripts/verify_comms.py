"""
Check the communication channel against PyBullet ground truth, one property at a time.

Roadmap step 7's verifier, in the shape of verify_observations.py and verify_rewards.py.
It exists for the same reason those do: a broken channel is silent. A policy trained
against messages that never arrive, or that arrive attributed to the wrong speaker, or
that leak reward, still produces a smooth learning curve - and the whole contribution of
this project is a before/after comparison that such a bug quietly invalidates.

    .venv\\Scripts\\python.exe scripts/verify_comms.py

The four properties worth stating plainly, because they are what the analysis depends on:

  1. COMMS=FALSE IS UNCHANGED. Same action space, same zeroed message slots, same
     177-float observation. Every checkpoint trained before step 7 remains loadable and
     remains a valid baseline. If this breaks, the comparison has no control arm.
  2. THE WORLD DOES NOT DEPEND ON THE CHANNEL. Seed 1000 builds the identical warehouse
     with comms on and off. The channel has its own RNG precisely so that switching it
     on does not silently re-roll every layout and turn a controlled comparison into two
     runs on different worlds.
  3. SPEAKING IS FREE AND CHEAP TALK. Identical movements with different tokens produce
     byte-identical rewards. Nothing pays a robot to talk, so any benefit the channel
     produces has to arrive through another robot's behaviour. That is what makes an
     improvement attributable to communication rather than to a reward artifact.
  4. A DROPPED MESSAGE IS ABSENT, NOT WRONG. Dropout zeroes a block; it never
     substitutes a different token. A listener can always tell silence from speech.

Exits non-zero if any check fails.
"""
import sys

import numpy as np

from hivemind_env.env import (
    MESSAGE_MODES,
    MSG_SILENT,
    MSG_TOKENS,
    NUM_AGENTS,
    OBS_DIM_V3,
    OBS_SLICES,
    OBS_WORLD_DIM,
    HiveMindMultiAgentEnv,
    joint_from_slot_actions,
    split_joint_action,
)

STAY = 6
checks = {"pass": 0, "fail": 0}


def check(label, condition, detail=""):
    if condition:
        checks["pass"] += 1
        print(f"  PASS  {label}")
    else:
        checks["fail"] += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def heard(obs, listener):
    """(NUM_AGENTS - 1, MSG_TOKENS) - what `listener` received, in speaker-block order."""
    return np.asarray(obs[listener][OBS_SLICES["messages"]]).reshape(
        NUM_AGENTS - 1, MSG_TOKENS
    )


def speakers_for(listener):
    """The fixed agent order with self skipped - the order the message blocks are in."""
    return [j for j in range(NUM_AGENTS) if j != listener]


def stay_with(tokens):
    """A joint action where nobody moves and everybody says `tokens[i]`."""
    return np.stack([np.full(NUM_AGENTS, STAY), np.asarray(tokens)], axis=1)


def make(**kwargs):
    kwargs.setdefault("num_cartons", 1)
    kwargs.setdefault("msg_dropout", 0.0)
    return HiveMindMultiAgentEnv(render_mode=None, **kwargs)


# ---------------------------------------------------------------------------------
def property_1_backwards_compatible():
    print("\n1. comms=False is byte-for-byte the environment step 6 trained against")
    env = make(comms=False)
    try:
        obs, _ = env.reset(seed=1000)
        check("observation is still the pinned width",
              obs.shape == (NUM_AGENTS, OBS_DIM_V3), f"got {obs.shape}")
        check("action space is still MultiDiscrete([7,7,7,7])",
              list(env.action_space.nvec) == [7] * NUM_AGENTS,
              f"got {env.action_space.nvec.tolist()}")

        zero = True
        for _ in range(20):
            obs, _, _, _, info = env.step([0, 1, 2, 3])
            zero = zero and not np.any(obs[:, OBS_SLICES["messages"]])
        check("message slots stay zero for a whole episode", zero)
        check("info reports every robot silent",
              info["message_tokens"] == [MSG_SILENT] * NUM_AGENTS,
              f"got {info['message_tokens']}")

        try:
            env.step(stay_with([0, 1, 2, 3]))
            check("a token supplied to a silent env is refused", False,
                  "step() accepted tokens that nothing could ever hear")
        except ValueError:
            check("a token supplied to a silent env is refused", True)
    finally:
        env.close()


def property_2_world_independent_of_channel():
    print("\n2. Turning the channel on does not change the warehouse")
    a, b = make(comms=False), make(comms=True)
    try:
        obs_a, _ = a.reset(seed=1000)
        obs_b, _ = b.reset(seed=1000)
        world = slice(0, OBS_WORLD_DIM)
        check("seed 1000 builds an identical world with and without comms",
              np.array_equal(obs_a[:, world], obs_b[:, world]),
              f"max abs difference {np.abs(obs_a[:, world] - obs_b[:, world]).max():.6f} - "
              f"the channel RNG is leaking into world generation, which would make the "
              f"baseline and the comms run two different experiments")
    finally:
        a.close()
        b.close()


def property_3_routing():
    print("\n3. What was said arrives, at the right listener, in the right block")
    env = make(comms=True)
    try:
        obs, _ = env.reset(seed=1000)
        check("nobody has spoken on the first observation of an episode",
              not np.any(obs[:, OBS_SLICES["messages"]]))

        said = [3, 7, 11, 0]
        obs, _, _, _, info = env.step(stay_with(said))
        check("info records what was said", info["message_tokens"] == said,
              f"got {info['message_tokens']}")

        routed = True
        detail = ""
        for i in range(NUM_AGENTS):
            blocks = heard(obs, i)
            for b, speaker in enumerate(speakers_for(i)):
                expected = said[speaker]
                if not (blocks[b].sum() == 1.0 and blocks[b].argmax() == expected):
                    routed = False
                    detail = (f"listener {i} block {b} should be speaker {speaker} "
                              f"saying {expected}, got {blocks[b].tolist()}")
        check("every listener hears every other robot in fixed speaker order", routed, detail)

        # Self-exclusion is structural - the slice is only 3 blocks wide - so what is
        # actually checked is that no block carries the listener's own token when that
        # token is unique to it.
        obs, _, _, _, _ = env.step(stay_with([1, 2, 4, 8]))
        no_self = all(
            not np.array_equal(heard(obs, i)[b].argmax(), tok)
            for i, tok in enumerate([1, 2, 4, 8])
            for b in range(NUM_AGENTS - 1)
            if heard(obs, i)[b].sum() > 0 and tok not in [t for j, t in
                                                          enumerate([1, 2, 4, 8]) if j != i]
        )
        check("no robot hears its own broadcast", no_self)

        obs, _, _, _, info = env.step([STAY] * NUM_AGENTS)
        check("a movement-only action is silence, not an error",
              not np.any(obs[:, OBS_SLICES["messages"]])
              and info["message_tokens"] == [MSG_SILENT] * NUM_AGENTS)

        try:
            env.step(np.array([[STAY, MSG_TOKENS]] * NUM_AGENTS))
            check("a token outside the vocabulary is refused", False)
        except ValueError:
            check("a token outside the vocabulary is refused", True)
    finally:
        env.close()


def property_4_speaking_is_free():
    print("\n4. Speaking costs nothing and pays nothing - the channel is cheap talk")
    quiet, loud = make(comms=True), make(comms=True)
    try:
        quiet.reset(seed=1000)
        loud.reset(seed=1000)
        moves = [0, 2, 6, 3, 1, 6, 2, 0]
        same = True
        for k, m in enumerate(moves):
            _, r_q, _, _, _ = quiet.step(np.stack(
                [np.full(NUM_AGENTS, m), np.zeros(NUM_AGENTS, dtype=int)], axis=1))
            _, r_l, _, _, _ = loud.step(np.stack(
                [np.full(NUM_AGENTS, m), np.full(NUM_AGENTS, (k * 5 + 3) % MSG_TOKENS)],
                axis=1))
            same = same and np.allclose(r_q, r_l)
        check("identical movements with different tokens earn identical rewards", same,
              "a token is influencing reward directly, so any measured benefit of "
              "communication would be partly a reward artifact")
    finally:
        quiet.close()
        loud.close()


def property_5_dropout():
    print("\n5. Dropout removes messages rather than corrupting them")
    env = make(comms=True, msg_dropout=1.0)
    try:
        env.reset(seed=1000)
        obs, _, _, _, info = env.step(stay_with([1, 2, 3, 4]))
        check("dropout=1.0 silences every link",
              not np.any(obs[:, OBS_SLICES["messages"]]))
        # 4x4 minus the diagonal: a robot never hears itself, so its own link is not
        # a link that can drop.
        check("dropout=1.0 reports all 12 listener-speaker links dropped",
              info["messages_dropped"] == NUM_AGENTS * (NUM_AGENTS - 1),
              f"got {info['messages_dropped']}")
        check("the speaker's own record survives the channel",
              info["message_tokens"] == [1, 2, 3, 4])
    finally:
        env.close()

    env = make(comms=True, msg_dropout=0.5)
    try:
        env.reset(seed=1000)
        dropped = total = 0
        corrupt = False
        for _ in range(300):
            obs, _, _, _, info = env.step(stay_with([5, 5, 5, 5]))
            dropped += info["messages_dropped"]
            total += NUM_AGENTS * (NUM_AGENTS - 1)
            for i in range(NUM_AGENTS):
                for block in heard(obs, i):
                    # Every block is either exactly silent or exactly token 5. A block
                    # holding some other token would mean dropout invented information.
                    if block.sum() not in (0.0, 1.0) or (block.sum() == 1.0
                                                         and block.argmax() != 5):
                        corrupt = True
        rate = dropped / total
        check("dropout=0.5 drops roughly half the links", abs(rate - 0.5) < 0.05,
              f"measured {rate:.3f} over {total} links")
        check("a dropped block is exactly silent, never a different token", not corrupt)
    finally:
        env.close()

    a, b = make(comms=True, msg_dropout=0.3), make(comms=True, msg_dropout=0.3)
    try:
        a.reset(seed=4242)
        b.reset(seed=4242)
        identical = True
        for _ in range(50):
            oa, _, _, _, _ = a.step(stay_with([2, 4, 6, 8]))
            ob, _, _, _, _ = b.step(stay_with([2, 4, 6, 8]))
            identical = identical and np.array_equal(oa, ob)
        check("the same seed replays the same dropout pattern", identical,
              "an evaluation would not be reproducible")
    finally:
        a.close()
        b.close()


def property_6_message_modes():
    print("\n6. The evaluation interventions do what they claim")
    said = [3, 7, 11, 0]

    env = make(comms=True, message_mode="silent")
    try:
        env.reset(seed=1000)
        obs, _, _, _, _ = env.step(stay_with(said))
        check("'silent' delivers nothing", not np.any(obs[:, OBS_SLICES["messages"]]))
    finally:
        env.close()

    env = make(comms=True, message_mode="shuffled")
    try:
        env.reset(seed=1000)
        obs, _, _, _, _ = env.step(stay_with([0, 1, 2, 3]))
        # The point of 'shuffled' is that the token distribution is untouched and only
        # the speaker binding breaks, so every block must still be a real one-hot.
        allowed = {0, 1, 2, 3}
        ok = all(block.sum() == 1.0 and int(block.argmax()) in allowed
                 for i in range(NUM_AGENTS) for block in heard(obs, i))
        check("'shuffled' still delivers real tokens from this step", ok)

        # And it must actually misattribute, or it is not an intervention at all. Over
        # many steps at least one listener must hear a block that is not its speaker's.
        misattributed = False
        for _ in range(40):
            obs, _, _, _, info = env.step(stay_with([0, 1, 2, 3]))
            for i in range(NUM_AGENTS):
                for b, speaker in enumerate(speakers_for(i)):
                    block = heard(obs, i)[b]
                    if block.sum() and int(block.argmax()) != info["message_tokens"][speaker]:
                        misattributed = True
        check("'shuffled' really does attribute tokens to the wrong speakers",
              misattributed)
    finally:
        env.close()

    env = make(comms=True, message_mode="random")
    try:
        env.reset(seed=1000)
        seen = set()
        for _ in range(60):
            obs, _, _, _, _ = env.step(stay_with([5, 5, 5, 5]))
            for i in range(NUM_AGENTS):
                for block in heard(obs, i):
                    if block.sum():
                        seen.add(int(block.argmax()))
        check("'random' replaces the content with tokens the speakers never sent",
              len(seen) > 8, f"only saw {sorted(seen)}")
    finally:
        env.close()

    try:
        make(comms=True, message_mode="nonsense")
        check("an unknown message_mode is refused", False)
    except ValueError:
        check("an unknown message_mode is refused", True)


def property_7_action_helpers():
    print("\n7. The action helpers agree with the env about shapes")
    moves, tokens = split_joint_action(np.array([0, 1, 2, 3]))
    check("a movement-only joint action reads as silence",
          moves.tolist() == [0, 1, 2, 3] and tokens.tolist() == [MSG_SILENT] * NUM_AGENTS)

    moves, tokens = split_joint_action(np.array([[0, 5], [1, 6], [2, 7], [3, 8]]))
    check("a two-column joint action splits into movement and tokens",
          moves.tolist() == [0, 1, 2, 3] and tokens.tolist() == [5, 6, 7, 8])

    try:
        split_joint_action(np.array([0, 5, 1, 6, 2, 7, 3, 8]))
        check("a flattened action is refused rather than guessed at", False)
    except ValueError:
        check("a flattened action is refused rather than guessed at", True)

    check("a Discrete policy's output stacks back to (4,)",
          joint_from_slot_actions(np.array([0, 1, 2, 3])).shape == (NUM_AGENTS,))
    check("a MultiDiscrete policy's output stacks back to (4, 2)",
          joint_from_slot_actions(
              np.array([[0, 5], [1, 6], [2, 7], [3, 8]])).shape == (NUM_AGENTS, 2))

    env = make(comms=True)
    try:
        env.reset(seed=1000)
        sampled = env.action_space.sample()
        check("the action space samples something the env accepts",
              sampled.shape == (NUM_AGENTS, 2))
        env.step(sampled)
        check("a sampled action steps without error", True)
    finally:
        env.close()


def main():
    print("=" * 78)
    print("  Communication channel verification - roadmap step 7")
    print(f"  {MSG_TOKENS}-token vocabulary, {NUM_AGENTS} robots, "
          f"observation pinned at {OBS_DIM_V3}")
    print(f"  message modes: {', '.join(MESSAGE_MODES)}")
    print("=" * 78)

    property_1_backwards_compatible()
    property_2_world_independent_of_channel()
    property_3_routing()
    property_4_speaking_is_free()
    property_5_dropout()
    property_6_message_modes()
    property_7_action_helpers()

    print("\n" + "=" * 78)
    print(f"  {checks['pass']} passed, {checks['fail']} failed")
    print("=" * 78)
    return 1 if checks["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
