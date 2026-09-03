import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as pb
import pybullet_data
import random
import math
from collections import deque
import time

# =============================================================================
# OBSERVATION LAYOUT V3 - PINNED AT 177 FLOATS PER ROBOT. READ BEFORE CHANGING.
# =============================================================================
#
# The flatten width is baked into saved policy weights, so a checkpoint only loads into
# an env reporting the identical dimension. Changing this number does not raise at
# training time - it retroactively invalidates every existing model, and the failure
# surfaces much later as a shape error nobody can date. Phase 1 was bitten by exactly
# that. A new width is a new OBS_DIM_V4 alongside V3, never an edit to V3.
#
# Three defences: the per-component constants below are the only source of truth
# (nothing hard-codes 177 or a slice bound); OBS_SLICES is checked AT IMPORT to tile
# [0, OBS_DIM_V3) with no gap or overlap; __init__ rejects a mismatched obs_dim.
#
#   V1   81  no carton positions - five layouts gave one identical observation
#   V2  105  adds carton positions, no LiDAR - added when shelves became solid
#   V3  177  adds 72 LiDAR rays. Current, and what every checkpoint was trained at.
#
# Neither V1 nor V2 was ever trained against; both are refused at construction.
#
#   slice      size  component           encoding
#   [  0:  3]     3  own pose            x, y over arena half-extent; heading wrapped
#                                        into [0,1) so cardinals read 0/.25/.5/.75
#   [  3:  5]     2  own velocity        last-step XY displacement, cells/step
#   [  5:  6]     1  own carrying        0 or 1
#   [  6: 15]     9  other poses         3 x (x, y, yaw), fixed order, self skipped
#   [ 15: 18]     3  other carrying      0 or 1
#   [ 18: 30]    12  carton status       0 available / .33 mine / .67 other's / 1 done
#   [ 30: 54]    24  carton positions    12 x (x, y), same slot index as the statuses
#   [ 54: 56]     2  depot direction     offset, normalised by arena span
#   [ 56: 57]     1  elapsed time        current_step / max_steps
#   [ 57:129]    72  LiDAR               270 deg arc, 0.1-10 m, normalised, noisy
#   [129:177]    48  messages            3 speakers x 16 one-hot tokens; all zero
#                                        unless comms=True
#
# Everything is normalised into [-1, 1]; _get_obs clips to enforce it.
#
# CHOICES THAT COST SOMETHING
# - Heading is one wrapped float, not sin/cos, which would need 4 values and break the
#   pin. Defensible only because step() snaps headings to the four cardinals, so the
#   network learns 4 discrete values rather than regressing an angle across the wrap.
#   A velocity-controlled motion model with arbitrary headings makes this a V4.
# - Poses are the SNAPPED grid poses, not raw base transforms: the chassis settles a few
#   millimetres during the substeps, which made a one-cell move measure 0.992 cells.
# - Velocity is displacement, not getBaseVelocity, which is meaningless noise after the
#   teleport snapback.
# - Carton status is one ordinal float, not a 4-way one-hot (48 more floats). The four
#   values are evenly spaced, but this does encode an ordering that does not exist. If a
#   policy cannot tell "claimed by other" from "delivered", widen it - as a V4.
# - Other robots' poses are absolute, in fixed agent order. Relative-to-self would
#   probably generalise better under a shared policy; that is a clean ablation.
# - Depot direction is redundant (the depot is always grid (0,0)) but costs 2 floats and
#   saves the network learning the transform.
#
# =============================================================================
# THE COMMUNICATION CHANNEL (roadmap step 7, landed 2026-09-02)
# =============================================================================
#
# The 48 slots were reserved and zeroed from 2026-08-29 so that filling them today did
# not move the width. That is what pinning was for: had they been added now, every
# no-comms checkpoint would have become unloadable, destroying the before/after
# comparison that is this project's contribution.
#
# `comms=False` is the default and leaves the env exactly as it was: action space
# MultiDiscrete([7,7,7,7]), slots zero, channel code never entered. With `comms=True`
# each robot emits one of MSG_TOKENS symbols per step alongside its movement, so the
# action space becomes MultiDiscrete([[7,16]] x 4). A robot hears the other three, never
# itself, in the same fixed speaker order the pose slots use.
#
# FOUR DECISIONS, none of them free:
#
# 1. TOKENS ARE DISCRETE AND PART OF THE ACTION (RIAL). The alternative - a continuous
#    vector with gradients flowing listener-to-speaker (DIAL) - learns faster and cannot
#    be built on stock Stable-Baselines3, which has no rollout that keeps the graph
#    across agents. The reinforced version costs sample efficiency and makes the protocol
#    readable: 16 symbols have a measurable entropy and mutual information.
# 2. MESSAGES ARRIVE ONE STEP LATE, unavoidably: all four robots choose from the same
#    observation simultaneously, so a token picked at step t cannot be in the observation
#    used at step t. A protocol here is about state still true a step later - intent,
#    claims, roles - not reflexes.
# 3. SILENCE IS NOT A TOKEN. All 16 symbols are chosen; the all-zero vector means
#    "nothing arrived" and comes only from dropout, comms=False, or the first
#    observation of an episode. So dropout removes information rather than fabricating
#    it, and saying nothing costs a symbol - a convention a policy has to invent.
# 4. DROPOUT IS PER LISTENER-SPEAKER LINK, not per broadcast, so two listeners can
#    disagree about the same step. Strictly harder than losing a broadcast for everyone,
#    which is the point of the roadmap's "~10% dropout so the protocol is not brittle".
#
# `message_mode` supplies the evaluation-time interventions that separate "emits varied
# tokens" from "uses them". See MESSAGE_MODES and scripts/analyse_messages.py.
# =============================================================================

NUM_AGENTS = 4
NUM_CARTONS = 12

# Movement actions. Named because the vec-env wrappers restate the size of this space
# to SB3 and every diagnostic prints the labels - six copies of this list had drifted
# into six files.
ACTION_NAMES = ["fwd", "back", "turnL", "turnR", "PICKUP", "DROP", "stay"]
MOVE_ACTIONS = len(ACTION_NAMES)

OBS_OWN_POSE = 3
OBS_OWN_VELOCITY = 2
OBS_OWN_CARRYING = 1
OBS_OTHER_POSES = 9
OBS_OTHER_CARRYING = 3
OBS_CARTON_STATUS = 12
OBS_CARTON_POSITIONS = NUM_CARTONS * 2  # 24 - new in V2
OBS_DEPOT_DIRECTION = 2
OBS_ELAPSED_TIME = 1

# ---------------------------------------------------------------------------
# LiDAR (spec S2.2)
# ---------------------------------------------------------------------------
# The spec's FOV, range and noise model are taken as written. Its ray COUNT of 720 is
# not: that would be seven times the rest of the observation and 2,880 raycasts per
# step. 72 rays is one tenth of it, still 3.75 deg apart - a 20 cm gap between beams at
# 3 m - so 1 m obstacles in a 1 m grid stay resolvable. Raising it is a V4.
LIDAR_NUM_RAYS = 72
LIDAR_FOV_RAD = math.radians(270.0)
LIDAR_MIN_RANGE = 0.1
LIDAR_MAX_RANGE = 10.0

# Rays start this far out so the sweep does not range on the robot's own body. The
# widest link at beam height is the front wheel at 0.2363 m. Casting from the spec's
# 0.1 m put the origin inside the chassis and every beam stopped on a wheel: the whole
# scan read 0.12-2.2 m in a 13 m arena. Distances are still measured from the centre.
LIDAR_START_RADIUS = 0.28

# Beam height, a constant rather than the live chassis z, for two reasons.
#
# It must sit inside the chassis band (0.094-0.194) so the sweep reports what the body
# collides with, AND inside the bottom shelf plate (0.140-0.220) so shelves are visible
# at all. 0.17 clears both by ~25 mm. The robot's modelled LiDAR link is at 0.21-0.25
# and rises to 0.5 m while carrying - faithful to the URDF and wrong for the task, since
# a beam at 0.23 passes over the plate the chassis is about to hit.
#
# And it is a constant because reading the live z once let the beam sink with the
# settling chassis: after ~250 steps every ray flew *under* the shelving and a robot one
# cell from a shelf reported 2.5 m of clear space. The sink is fixed (see _spawn_z), but
# perception must not be able to depend on chassis dynamics again.
LIDAR_BEAM_Z = 0.17
LIDAR_NOISE_SIGMA = 0.01       # metres, constant term
LIDAR_NOISE_RANGE_FRAC = 0.01  # plus 1% of the measured range

OBS_LIDAR = LIDAR_NUM_RAYS

# Everything the robot observes about the world, before any communication.
OBS_WORLD_DIM = (
    OBS_OWN_POSE + OBS_OWN_VELOCITY + OBS_OWN_CARRYING
    + OBS_OTHER_POSES + OBS_OTHER_CARRYING
    + OBS_CARTON_STATUS + OBS_CARTON_POSITIONS
    + OBS_DEPOT_DIRECTION + OBS_ELAPSED_TIME + OBS_LIDAR
)  # 129

# Roadmap step 7: each robot broadcasts one of MSG_TOKENS symbols; a robot hears the
# other three, never itself. Spec section 2.4 sets the vocabulary at K = 16 tokens.
MSG_TOKENS = 16
OBS_MESSAGE_DIM = MSG_TOKENS * (NUM_AGENTS - 1)  # 48

# The token index that means "nothing arrived". It is NOT a symbol a robot can choose -
# see choice 3 in the header. Emitted as the all-zero vector, which no one-hot equals.
MSG_SILENT = -1

# Roadmap step 7: "Include ~10% message dropout during training so the protocol is not
# brittle." Applied per listener-speaker link, so the three links into one robot fail
# independently and two listeners can disagree about the same step.
MSG_DROPOUT_DEFAULT = 0.10

# Evaluation-time interventions on the channel: what separates "emits varied tokens"
# from "uses them". A channel that can be replaced by noise without costing makespan was
# never a protocol.
#
#   learned   what the speakers said. The training setting.
#   silent    everyone hears the zero vector - how much competence rests on hearing.
#   shuffled  real tokens from this step, wrong speakers. THE SHARP ONE: the marginal
#             token distribution is untouched, so only the meaning dies. `silent` and
#             `random` can both be survived by a policy that merely learned to tolerate
#             noise in those inputs.
#   random    uniform tokens - same information rate, zero correlation with the world.
MESSAGE_MODES = ("learned", "silent", "shuffled", "random")

OBS_DIM_V1 = 81    # historical - no carton positions, no LiDAR. Never trained against.
OBS_DIM_V2 = 105   # historical - carton positions, no LiDAR. Never trained against.
OBS_DIM_V3 = OBS_WORLD_DIM + OBS_MESSAGE_DIM  # 177
DEFAULT_OBS_DIM = OBS_DIM_V3
_SUPERSEDED_DIMS = {
    OBS_DIM_V1: "V1 (2026-08-28): no carton positions - the observation could not tell "
                "two warehouse layouts apart",
    OBS_DIM_V2: "V2 (2026-08-28): no LiDAR - added when shelves became solid obstacles, "
                "which a robot otherwise had no way to perceive",
}

# Curriculum ladder: difficulty level -> cartons in play. Mirrored by
# hivemind_env.training.CURRICULUM_CARTONS, which drives promotion.
CURRICULUM_CARTONS = {1: 4, 2: 8, 3: 12}

# Episode cap per carton count, roughly 3x greedy's makespan at each (23 / 59 / 98 over
# 30 seeds). A cap far above the job length is not free: the time penalty accrues, the
# makespan bonus is normalised by it, and every surplus step is another chance to
# collide. A flat 400 at 4 cartons was 17x greedy and cost -18 per episode.
MAX_STEPS_BY_CARTONS = {1: 60, 2: 90, 4: 150, 8: 250, 12: 400}


def max_steps_for(num_cartons):
    """
    Episode cap for a carton count. Interpolates upward for counts not on the ladder
    so an arbitrary --num-cartons still gets a sane cap rather than the full-task one.
    """
    n = NUM_CARTONS if num_cartons is None else int(num_cartons)
    for cartons in sorted(MAX_STEPS_BY_CARTONS):
        if n <= cartons:
            return MAX_STEPS_BY_CARTONS[cartons]
    return MAX_STEPS_BY_CARTONS[max(MAX_STEPS_BY_CARTONS)]


CARTON_AVAILABLE = 0.0
CARTON_CLAIMED_BY_ME = 1.0 / 3.0
CARTON_CLAIMED_BY_OTHER = 2.0 / 3.0
CARTON_DELIVERED = 1.0


# =============================================================================
# REWARD - transcribed from MAWC_Technical_Specification.pdf, S3
# =============================================================================
#
# S3.1 Shared (90% weight), identical for all 4 agents:
#     all resources delivered  +100.0                          once per episode
#     per delivery             +10.0
#     makespan bonus           +50 x (T_max - T_actual) / T_max once per episode
#     collision (any pair)     -5.0                            per event
#     time penalty             -0.05                           every step
#
# S3.2 Individual (10% weight), per agent:
#     own pickup               +1.0
#     own delivery             +2.0
#     idle (v < 0.1, off depot) -0.02                          per step
#     replanning (A* re-run)   -0.1                            NOT IMPLEMENTED
#     invalid action           -0.5
#
# S3.3  R_total_i = 0.90 * R_shared + 0.10 * R_individual_i
#
# THREE PLACES THE SPEC NEEDED A DECISION
#
# 1. The replanning penalty has no trigger here. It fires when A* re-runs, and project
#    decision 3 puts A*, DWA and EKF out of scope - robots are placed on a grid, not
#    driven along a planned path. R_REPLAN_PENALTY is defined and deliberately unused so
#    the omission is visible rather than silent.
# 2. "Per collision event" is billed on contact onset, not per step: robots teleport, so
#    an overlapping pair stays overlapped until one moves and charging every step would
#    bill -5.0 repeatedly for one mistake.
# 3. The idle penalty is on linear velocity, so by the spec's letter a robot turning on
#    the spot IS idle, and that is implemented literally. The effect is small (0.002 per
#    step against a 0.045 time penalty). Pass idle_penalises_turning=False to exempt it.
#
# T_max is max_steps - this env counts steps, not the spec's seconds. Otherwise the
# formula is unchanged.
# =============================================================================

# Shared (S3.1)
R_ALL_DELIVERED = 100.0
R_PER_DELIVERY = 10.0
R_MAKESPAN_SCALE = 50.0
R_COLLISION = -5.0
R_TIME_PENALTY = -0.05

# Individual (S3.2)
R_OWN_PICKUP = 1.0
R_OWN_DELIVERY = 2.0
R_IDLE_PENALTY = -0.02
R_REPLAN_PENALTY = -0.1   # defined by the spec; no trigger exists here (see above)
R_INVALID_ACTION = -0.5

# S3.3 split
SHARED_WEIGHT = 0.90
INDIVIDUAL_WEIGHT = 0.10

# "v < 0.1 m/s" in cells per step. A move covers exactly 1.0 cell/step, so any threshold
# below 1.0 separates moving from not moving; 0.1 keeps the spec's number.
IDLE_SPEED_THRESHOLD = 0.1

# The spec's depot is a 2x2 m zone; here it is one cell, and "at depot" reuses the
# drop-off action's 1.5-cell radius so a robot parked where it can legally deliver is
# not also charged for idling there.
DEPOT_RADIUS_CELLS = 1.5

# ---------------------------------------------------------------------------
# Potential-based reward shaping - an ADDITION to the specification's table
# ---------------------------------------------------------------------------
# The spec's reward is sparse: nothing pays until a pickup, and the large terms only pay
# on completion. That is unlearnable from scratch here, measured rather than guessed - a
# 5,013,504-step run completed zero of ~2,500 episodes and converged to -103, BELOW the
# -94 a robot scores by standing still. It had correctly learned that moving costs more
# in collisions than delivery pays.
#
# The fix is potential-based shaping (Ng, Harada & Russell 1999), F = Phi(s') - Phi(s),
# with Phi = -(work remaining) in cartons. It cannot be farmed by hovering: the sum
# telescopes to Phi(end) - Phi(start) whatever path is taken between them.
#
#     R_total_i = 0.90 * R_shared + 0.10 * R_individual_i + F_i
#
# F_i sits OUTSIDE the 90/10 split. Every R_* constant above is the specification's and
# untouched; this added term is the only extension. Set shaping=False for the spec
# exactly, and expect it not to learn.
#
# THE FIRST VERSION WAS WRONG IN TWO WAYS, both measured 2026-08-31: Phi was plain
# distance to the current objective, so it jumped downward when is_carrying flipped and
# completing a pickup scored NEGATIVE; and F was added inside the 0.10 bucket, so a
# scale of 15.0 delivered 1.5. Both are described where they were fixed.
#
# THE SCALE IS SET BY ONE MEASURED QUANTITY - the expected value of moving:
#
#     EV(move) = shaping gain per cell - P(collision | move) x 4.5 - time penalty
#
#     scale   gain/cell   EV(move) random   EV(move) dispersed
#       6.0     +0.150         -0.379            -0.134
#      20.0     +0.499         -0.030            +0.215
#      30.0     +0.750         +0.221            +0.466
#
# Only movement can collide - turning, staying, PICKUP and DROP cannot - so the
# collision penalty is a risk premium on the one action class that makes progress, and
# when it exceeds the gain the optimal policy is to turn, grab and stand still. At scale
# 6.0 a canary run did exactly that: `stay` 100% under argmax, fwd 2% under sampling.
#
# P(collision | move) is 0.108 under random play, peaking at 0.146 mid-episode when
# random walkers jam the aisles, 0.053 once dispersed, 0.031 for greedy. The scale is
# set against the pessimistic figure because that is the regime PPO starts in.
#
# A scale this large is safe because with the shaping gamma at 1 the episode total
# telescopes exactly to scale x (Phi_end - Phi_start), independent of path: it changes
# gradient magnitude and nothing else. Re-run scripts/diagnose_incentives.py after ANY
# change to Phi - none of this arithmetic survives a redefinition, as the next paragraph
# is the proof of.
#
# 30.0 -> 60.0 on 2026-09-03, and it is NOT a retune. Phi's distance term was bounded
# into [0, 0.5] that day (to make a pickup and a delivery provably non-negative - see
# _potential), which halved the gain per cell and dropped EV(move) to +0.048 against a
# +0.05 gate. Doubling the scale restores the per-cell gradient to exactly what it was:
#
#     old   30 x (1 cell / 2*span metres)          = 30 x 0.0385 = 1.155 per cell
#     new   60 x 0.5 x (1 cell / 26 cells)         = 60 x 0.0192 = 1.155 per cell
#
# So the gradient the policy sees is unchanged and only its source is - Euclidean
# distance became geodesic. That is the point: the number moved so the behaviour would
# not.
SHAPING_SCALE_DEFAULT = 60.0

# Upper bound on the geodesic distance Phi measures, in grid cells. The distance term is
# 0.5 * min(1, cells / this), so it stays inside [0, 0.5] and both task transitions come
# out non-negative - see _potential. 2 x grid_size = 26; the longest real path measured
# across seeds 1000-1009 is 24 cells, so it saturates only on routes that do not occur.
GEODESIC_MAX_CELLS = 26


def _build_slices():
    """Named slices in layout order. Nothing else in this file indexes by number."""
    bounds, cursor = {}, 0
    for name, width in (
        ("own_pose", OBS_OWN_POSE),
        ("own_velocity", OBS_OWN_VELOCITY),
        ("own_carrying", OBS_OWN_CARRYING),
        ("other_poses", OBS_OTHER_POSES),
        ("other_carrying", OBS_OTHER_CARRYING),
        ("carton_status", OBS_CARTON_STATUS),
        ("carton_positions", OBS_CARTON_POSITIONS),
        ("depot_direction", OBS_DEPOT_DIRECTION),
        ("elapsed_time", OBS_ELAPSED_TIME),
        ("lidar", OBS_LIDAR),
        ("messages", OBS_MESSAGE_DIM),
    ):
        bounds[name] = slice(cursor, cursor + width)
        cursor += width
    return bounds, cursor


OBS_SLICES, _OBS_TOTAL = _build_slices()

# Defence 2. This runs at import, so a component added without updating the total
# cannot reach a training run.
if _OBS_TOTAL != OBS_DIM_V3:
    raise AssertionError(
        f"Observation layout is inconsistent: components sum to {_OBS_TOTAL} but "
        f"OBS_DIM_V3 is {OBS_DIM_V3}. Adding a component is a NEW observation "
        f"version (bump to V4), not an edit to V3 - every checkpoint trained "
        f"against V3 becomes unloadable the moment this number moves."
    )
if OBS_SLICES["messages"].stop != OBS_DIM_V3:
    raise AssertionError("Message slots must be last so world features keep stable indices.")


def describe_observation_layout():
    """
    The pinned layout as text. Printed by smoke_test.py so the dimension is visible
    in a run log rather than only in this docstring.
    """
    lines = [f"Observation layout V3 - {OBS_DIM_V3} floats per robot, "
             f"{NUM_AGENTS} robots -> shape ({NUM_AGENTS}, {OBS_DIM_V3})"]
    for name, sl in OBS_SLICES.items():
        note = ""
        if name == "messages":
            note = (f" ({NUM_AGENTS - 1} speakers x {MSG_TOKENS} tokens; zero unless "
                    f"comms=True)")
        if name == "lidar":
            note = f" ({LIDAR_NUM_RAYS} rays, 270 deg, {LIDAR_MAX_RANGE} m)"
        lines.append(f"  [{sl.start:3d}:{sl.stop:3d}]  {sl.stop - sl.start:2d}  {name}{note}")
    lines.append(f"  world features: {OBS_WORLD_DIM}   message slots: {OBS_MESSAGE_DIM}")
    return "\n".join(lines)


def joint_from_slot_actions(slot_actions, num_agents=NUM_AGENTS):
    """
    Stack one shared policy's per-robot output into the joint action the env accepts.

    `model.predict` over a batch of `num_agents` observations returns (n,) for a
    Discrete(7) policy and (n, 2) for the MultiDiscrete([7, 16]) a communicating one
    has - and this collapses the first case back to (n,) so the env sees the shape it
    has always seen.

    Every diagnostic script needs this and each of them used to write
    `np.asarray(a).reshape(-1)[:4]`, which reads a comms policy's output as
    [move0, token0, move1, token1] and hands the env four movements that are half
    tokens. Silent, and it produces plausible-looking nonsense.
    """
    arr = np.asarray(slot_actions, dtype=int).reshape(num_agents, -1)
    return arr[:, 0] if arr.shape[1] == 1 else arr


def policy_uses_comms(model) -> bool:
    """
    Does this checkpoint have a token head? Read off the saved action space.

    Asking the caller to remember is how a communicating policy gets evaluated with a
    silent channel and the number filed as its score.
    """
    return tuple(getattr(getattr(model, "action_space", None), "shape", ()) or ()) == (2,)


def split_joint_action(joint_action, num_agents=NUM_AGENTS):
    """
    Split whatever a caller passed into (movement actions, message tokens).

    The joint action has two accepted shapes and this is the ONLY place that knows it:

        (num_agents,)     movement only. Every token is MSG_SILENT. This is what the
                          env has always accepted, and it is what the scripted greedy
                          controller and any pre-step-7 checkpoint produce - so they
                          keep running unchanged against a comms env, saying nothing.
        (num_agents, 2)   column 0 movement, column 1 the token in [0, MSG_TOKENS).
                          What a comms policy's MultiDiscrete([7, 16]) emits.

    Returned tokens are always length num_agents with MSG_SILENT (-1) for silence, so
    callers never branch on the input shape a second time.
    """
    arr = np.asarray(joint_action)
    if arr.ndim == 1 and arr.size == num_agents:
        return arr.astype(int), np.full(num_agents, MSG_SILENT, dtype=int)
    if arr.ndim == 2 and arr.shape == (num_agents, 2):
        tokens = arr[:, 1].astype(int)
        if np.any((tokens < 0) | (tokens >= MSG_TOKENS)):
            raise ValueError(
                f"message tokens must lie in [0, {MSG_TOKENS}); got {tokens.tolist()}"
            )
        return arr[:, 0].astype(int), tokens
    raise ValueError(
        f"joint action must have shape ({num_agents},) for movement only or "
        f"({num_agents}, 2) for movement + message token; got shape {arr.shape}. "
        f"A flattened (num_agents * 2,) array is deliberately NOT accepted - it is "
        f"indistinguishable from an 8-robot movement action and guessing between them "
        f"is how a silent miswiring gets into a training run."
    )


class HiveMindMultiAgentEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    carton_size = 0.5
    gripper_reach = 0.3
    lidar_initial_height = 0.0
    lidar_carry_height = 0.5

    def __init__(self, render_mode=None, difficulty_level=1, obs_dim=DEFAULT_OBS_DIM,
                 show_lidar=None, obs_size=None, idle_penalises_turning=True,
                 lidar_noise=True, substeps=None, max_steps=None,
                 num_cartons=None, shaping=True,
                 shaping_scale=SHAPING_SCALE_DEFAULT, gamma=0.99,
                 comms=False, msg_dropout=MSG_DROPOUT_DEFAULT,
                 message_mode="learned"):
        super().__init__()

        # --- Communication (roadmap step 7) ---------------------------------------
        # Off by default, and that default is doing real work: with comms=False this
        # env is what it was before step 7 landed, so every no-comms checkpoint loads
        # and evaluates unchanged and stays a valid baseline. See the header.
        self.comms = bool(comms)
        self.msg_dropout = float(msg_dropout)
        if not 0.0 <= self.msg_dropout <= 1.0:
            raise ValueError(f"msg_dropout must be in [0, 1], got {msg_dropout}")
        if message_mode not in MESSAGE_MODES:
            raise ValueError(
                f"message_mode={message_mode!r} is not one of {MESSAGE_MODES}. "
                f"These are evaluation-time interventions on the channel - see the "
                f"header block and scripts/analyse_messages.py."
            )
        self.message_mode = message_mode

        # How many of the NUM_CARTONS slots carry a carton this episode; None means all
        # 12. The observation always reserves 12 - the width is pinned - so a smaller
        # task marks the unused slots delivered from the start rather than shrinking the
        # vector, and a checkpoint survives a curriculum promotion. The callback sets
        # this attribute directly; until 2026-08-31 it set `difficulty_level`, which
        # nothing in the world read, so promotion did nothing at all.
        self.num_cartons = num_cartons

        # Potential-based reward shaping (Ng, Harada & Russell). See _shaping_reward.
        # Off by default it would be honest to the specification; on by default it is
        # honest to the fact that the unshaped reward provably cannot be learned from
        # here - a 5M-step run scored below the do-nothing floor.
        self.shaping = shaping
        self.shaping_scale = float(shaping_scale)
        self.gamma = float(gamma)
        # Substeps are an interpolation, NOT the motion model: the final pose is the
        # snapped grid target regardless and collisions are read from it, so the count
        # does not affect makespan, collisions, deliveries or completion. Measured over
        # the 30-seed greedy baseline at 30 / 10 / 5 / 1 substeps: identical to the
        # decimal every time. It changes animation smoothness, how far a carton is flung
        # when a robot ploughs into it, and speed by ~10x end to end.
        #
        # 5 headless (5x the throughput of 30, no behavioural difference), 30 in the GUI
        # so it animates. Not 1: a robot then jumps a full metre at once and could tunnel
        # past the 6 cm shelf posts.
        if substeps is None:
            substeps = 30 if render_mode == "human" else 5
        self.substeps = int(substeps)
        if self.substeps < 1:
            raise ValueError(f"substeps must be >= 1, got {substeps}")
        self.idle_penalises_turning = idle_penalises_turning
        self.lidar_noise = lidar_noise
        self._lidar_cache = None
        self._dist_cache = None
        self.render_mode = render_mode
        self.num_agents = NUM_AGENTS
        self.difficulty_level = difficulty_level
        self.show_lidar = (render_mode == "human") if show_lidar is None else show_lidar
        self.dt = 1.0 / 240.0
        self.grid_size = 13
        self.cell_size = 1.0

        # `obs_size` was the single-agent branch's CNN window width (15 or 21). This env
        # has no grid observation and no CNN, so the name meant nothing here - it was
        # carried over with the class. Accepted only so an older call site fails with a
        # sentence instead of a TypeError.
        if obs_size is not None:
            raise TypeError(
                "obs_size is from the single-agent CNN observation and does not apply "
                f"here. The observation is a flat vector; pass obs_dim={OBS_DIM_V1} or "
                "omit it entirely."
            )

        # Defence 3. A stale call site must not be able to build a mismatched env.
        if obs_dim != OBS_DIM_V3:
            hint = _SUPERSEDED_DIMS.get(obs_dim)
            hint = f" That width is superseded: {hint}." if hint else ""
            raise ValueError(
                f"obs_dim={obs_dim} does not match the pinned observation width "
                f"OBS_DIM_V3={OBS_DIM_V3}. The width is baked into saved policy weights, "
                f"so changing it silently invalidates every existing checkpoint.{hint} If a "
                f"new width is genuinely wanted, add OBS_DIM_V4 alongside V3 and select it "
                f"explicitly - see the layout block at the top of this module."
            )
        self.obs_dim = obs_dim

        # Actions: 0: Forward, 1: Backward, 2: Turn Left, 3: Turn Right, 4: Pick Up, 5: Drop Off, 6: Stay
        #
        # With comms on, every robot also emits one of MSG_TOKENS symbols on the same
        # step, so the per-robot action becomes a pair and the joint action gains a
        # second column. The movement column keeps index 0 and keeps its meaning, so
        # everything that reads `action[..., 0]` is unaffected by the switch.
        if self.comms:
            self.action_space = spaces.MultiDiscrete(
                np.array([[MOVE_ACTIONS, MSG_TOKENS]] * self.num_agents,
                         dtype=np.int64)
            )
        else:
            self.action_space = spaces.MultiDiscrete([MOVE_ACTIONS] * self.num_agents)

        # One row per robot. Everything is normalised into [-1, 1] so the bounds are
        # honest rather than +/-inf placeholders; _get_obs clips to enforce them.
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.num_agents, self.obs_dim),
            dtype=np.float32,
        )

        # Normalisation constants, derived once. half_extent is the arena half-width, so
        # a position at the wall normalises to just under 1.0.
        self._arena_half_extent = self.grid_size * self.cell_size / 2.0
        self._arena_span = self.grid_size * self.cell_size

        if self.render_mode == "human":
            self.client_id = pb.connect(pb.GUI)
        else:
            self.client_id = pb.connect(pb.DIRECT)
            
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        
        self.robot_ids = []
        self.resource_ids = []
        self.depot_id = None
        self.obstacle_ids = []
        self.depot_pos_grid = (0, 0)
        # Episode budget, ~3x greedy's makespan at each carton count. It was a flat
        # 2000 until 2026-08-31, which was actively harmful: 0.99^2000 = 1.9e-9, so the
        # +100 completion and makespan bonuses - the two largest terms in the table -
        # discounted to nothing from the first step (0.99^98 = 0.37, so the horizon
        # suited the task and not the episode it was embedded in). A 5M-step run at 2000
        # completed zero of ~2,500 episodes. Pass max_steps to override.
        self.max_steps = int(max_steps) if max_steps is not None \
            else max_steps_for(self.num_cartons)
        self.current_step = 0

    def _grid_to_world(self, r, c):
        x = (c - self.grid_size/2.0) * self.cell_size + (self.cell_size/2.0)
        y = (self.grid_size/2.0 - r) * self.cell_size - (self.cell_size/2.0)
        return x, y
        
    def _world_to_grid(self, x, y):
        c = int(round((x / self.cell_size) + self.grid_size/2.0 - 0.5))
        r = int(round(self.grid_size/2.0 - (y / self.cell_size) - 0.5))
        return r, c

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.is_carrying = [False] * self.num_agents
        self.carried_resource_ids = [None] * self.num_agents
        self.current_step = 0

        # Observation bookkeeping, all rebuilt per episode.
        #
        # `resource_ids` is the *available* cartons and shrinks as they are picked up -
        # Scripted controllers read it that way, so its meaning must not change. The
        # observation needs something different: a stable slot per carton that survives
        # pickup and delivery, so carton 7 is always index 7 for the whole episode.
        self.all_resource_ids = []          # fixed order, length NUM_CARTONS
        self.resource_slot = {}             # pybullet body id -> observation index
        self.delivered = [False] * NUM_CARTONS
        self._prev_xy = [(0.0, 0.0)] * self.num_agents
        self._velocity = [(0.0, 0.0)] * self.num_agents

        # --- Communication state (roadmap step 7) -----------------------------------
        # `messages[j]` is what robot j broadcast on the last step, as a one-hot over the
        # vocabulary, or all zeros for silence. It starts silent: on the first
        # observation of an episode nobody has spoken yet, and that is a fact the
        # listener should see rather than a zero standing in for a token.
        self.messages = np.zeros((self.num_agents, MSG_TOKENS), dtype=np.float32)
        self.message_tokens = np.full(self.num_agents, MSG_SILENT, dtype=int)
        # `_heard[listener, speaker]` is what arrived AFTER dropout and any message_mode
        # intervention - so the listener's diagonal is never read, and two listeners can
        # legitimately hold different views of the same speaker.
        self._heard = np.zeros(
            (self.num_agents, self.num_agents, MSG_TOKENS), dtype=np.float32
        )
        self._dropped = np.zeros((self.num_agents, self.num_agents), dtype=bool)

        # The channel gets its own generator, derived from the episode seed. Sharing
        # np.random would make dropout consume draws from the same stream the world
        # layout uses, so switching comms on or off would silently change every
        # warehouse - and the no-comms baseline would no longer be on the same seeds
        # as the run it is being compared with.
        self._msg_rng = np.random.default_rng(
            None if seed is None else (int(seed) * 7919 + 13)
        )

        # Reward bookkeeping (spec S3), all per-episode.
        # `_colliding_pairs` holds the pairs already in contact, so a collision is
        # charged once per *event* rather than once per step for as long as two
        # robots stay overlapped - the spec says "per collision event".
        self._dist_cache = None
        self._colliding_pairs = set()
        self._episode_reward = np.zeros(self.num_agents, dtype=np.float64)
        self._makespan_awarded = False
        self.last_reward_breakdown = None

        pb.resetSimulation(physicsClientId=self.client_id)
        pb.setGravity(0, 0, -9.81, physicsClientId=self.client_id)

        # Keep the warehouse floor flush with z=0 for the shelf and carton assets.
        floor_half_extents = [self.grid_size * self.cell_size / 2.0, self.grid_size * self.cell_size / 2.0, 0.05]
        floor_col = pb.createCollisionShape(
            pb.GEOM_BOX,
            halfExtents=floor_half_extents,
            physicsClientId=self.client_id,
        )
        floor_vis = pb.createVisualShape(
            pb.GEOM_BOX,
            halfExtents=floor_half_extents,
            rgbaColor=[0.68, 0.48, 0.28, 1.0],
            physicsClientId=self.client_id,
        )
        pb.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=floor_col,
            baseVisualShapeIndex=floor_vis,
            basePosition=[0, 0, -0.05],
            physicsClientId=self.client_id,
        )

        # Depot position (Corner: r=0, c=0)
        self.depot_pos_grid = (0, 0)
        dx, dy = self._grid_to_world(0, 0)
        depot_vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[self.cell_size*0.5, self.cell_size*0.5, 0.01], rgbaColor=[0, 0, 0, 0.5], physicsClientId=self.client_id)
        self.depot_id = pb.createMultiBody(baseMass=0, baseVisualShapeIndex=depot_vis, basePosition=[dx, dy, 0.01], physicsClientId=self.client_id)

        # Spawn 4 bots near the depot
        spawn_cells = [(0, 1), (1, 0), (0, 2), (2, 0)]
        
        self.robot_ids = []
        self.agent_state = []

        import os
        urdf_path = os.path.join(os.path.dirname(__file__), "assets", "diff_drive_bot.urdf")
        
        for i in range(self.num_agents):
            rx, ry = self._grid_to_world(spawn_cells[i][0], spawn_cells[i][1])
            rid = pb.loadURDF(urdf_path, basePosition=[rx, ry, 0.1], physicsClientId=self.client_id)
            self.robot_ids.append(rid)
            
            state = {
                'left_wheel_indices': [],
                'right_wheel_indices': [],
                'arm_yaw_joint_idx': None,
                'left_finger_joint_idx': None,
                'right_finger_joint_idx': None,
                'lidar_joint_idx': None,
                'current_arm_yaw': 0.0,
                'current_finger_pos': 0.03,  # Scaled by 2x from original 0.015
                'current_lidar_height': self.lidar_initial_height
            }
            
            for j in range(pb.getNumJoints(rid, physicsClientId=self.client_id)):
                info = pb.getJointInfo(rid, j, physicsClientId=self.client_id)
                jname = info[1].decode("utf-8")
                if "left_wheel" in jname:
                    state['left_wheel_indices'].append(j)
                elif "right_wheel" in jname:
                    state['right_wheel_indices'].append(j)
                elif jname == "arm_yaw_joint":
                    state['arm_yaw_joint_idx'] = j
                elif jname == "left_finger_joint":
                    state['left_finger_joint_idx'] = j
                elif jname == "right_finger_joint":
                    state['right_finger_joint_idx'] = j
                elif jname == "lidar_joint":
                    state['lidar_joint_idx'] = j
                    
            self.agent_state.append(state)
            self._set_arm_and_lidar_joints(i, state['current_arm_yaw'], state['current_finger_pos'], state['current_lidar_height'])

        # 13x13 Warehouse Generation
        self.obstacle_ids = []
        self.resource_ids = []
        
        assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        shelf_rows = [1, 3, 5, 7, 9, 11]
        for r in shelf_rows:
            # Randomize x, y, z such that x + y + z = 9 and x, y, z >= 1
            cuts = sorted(random.sample(range(1, 9), 2))
            x = cuts[0]
            y = cuts[1] - cuts[0]
            z = 9 - cuts[1]
            
            partitions = [
                (1, x),
                (x + 2, y),
                (x + y + 3, z)
            ]
            
            for (start_c, length) in partitions:
                cx, cy = self._grid_to_world(r, start_c + length/2.0 - 0.5)
                shelf_urdf_path = os.path.join(assets_dir, f"shelf_{length}m.urdf")
                obs_id = pb.loadURDF(shelf_urdf_path, basePosition=[cx, cy, 0.0], useFixedBase=True, physicsClientId=self.client_id)
                self.obstacle_ids.append(obs_id)
            
            # Gaps are at c = x + 1 and c = x + y + 2
            for c in [x + 1, x + y + 2]:
                resx, resy = self._grid_to_world(r, c)
                carton_urdf_path = os.path.join(assets_dir, "carton.urdf")
                res_id = pb.loadURDF(carton_urdf_path, basePosition=[resx, resy, 0.0], physicsClientId=self.client_id)
                self.resource_ids.append(res_id)

        # Static Boundary Walls around the 13x13 grid
        self.wall_ids = []
        b_size = self.grid_size * self.cell_size / 2.0
        wall_half_extents = [(b_size + 0.1, 0.1, 0.5), (b_size + 0.1, 0.1, 0.5), (0.1, b_size, 0.5), (0.1, b_size, 0.5)]
        wall_positions = [(0, b_size + 0.1, 0.5), (0, -b_size - 0.1, 0.5), (-b_size - 0.1, 0, 0.5), (b_size + 0.1, 0, 0.5)]
        for he, pos in zip(wall_half_extents, wall_positions):
            w_col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=he, physicsClientId=self.client_id)
            w_vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=he, rgbaColor=[0.38, 0.20, 0.08, 1], physicsClientId=self.client_id)
            w_id = pb.createMultiBody(baseMass=0, baseCollisionShapeIndex=w_col, baseVisualShapeIndex=w_vis, basePosition=pos, physicsClientId=self.client_id)
            self.wall_ids.append(w_id)
            
        if self.render_mode == "human":
            pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0, physicsClientId=self.client_id)

        for _ in range(20):
            pb.stepSimulation(physicsClientId=self.client_id)

        # Resting chassis height, sampled once the spawn has settled. step() snaps every
        # robot back to this each step so the chassis cannot sink - see the note there.
        self._spawn_z = pb.getBasePositionAndOrientation(
            self.robot_ids[0], physicsClientId=self.client_id
        )[0][2]

        # Freeze the carton ordering now that all 12 exist. Generation walks shelf rows
        # in a fixed order, so slot i is the same aisle position for a given seed.
        self.all_resource_ids = list(self.resource_ids)
        if len(self.all_resource_ids) != NUM_CARTONS:
            raise RuntimeError(
                f"World generation produced {len(self.all_resource_ids)} cartons but the "
                f"observation reserves exactly {NUM_CARTONS} slots. These must agree."
            )
        self.resource_slot = {rid: i for i, rid in enumerate(self.all_resource_ids)}

        # Where every carton STARTS, captured before the curriculum removes any. These
        # are the aisle gaps, and they stay walkable whether or not a carton sits in
        # them - anything deriving a shelf map from live carton positions would wall off
        # the gaps of the cartons the curriculum took out.
        self.carton_home_cells = []
        for rid in self.all_resource_ids:
            p_, _ = pb.getBasePositionAndOrientation(rid, physicsClientId=self.client_id)
            self.carton_home_cells.append(self._world_to_grid(p_[0], p_[1]))

        # Cells a robot cannot enter: odd grid rows are solid shelf except the two
        # gaps, which is where the cartons start.
        #
        # The env used to let robots drive into shelves and charge the collision.
        # Measured 2026-08-31: a random policy took 105.8 collision events per episode,
        # 93.8 of them shelf, while greedy took 0.0. At -5.0 shared that is -1.19
        # reward/agent/step for moving against -0.045 for standing still, so the only
        # thing PPO could learn was to freeze - which it did, three runs running. The
        # penalty was taxing exploration for a mistake the optimal policy never makes.
        # Blocking the move costs the spec's -0.5 invalid action instead, the same as
        # driving off the grid, and changes no reward constant. Robot-robot collisions
        # are untouched and still cost -5.0.
        _gaps = set(self.carton_home_cells)
        self.blocked_cells = {
            (r, c)
            for r in range(1, self.grid_size - 1, 2)
            for c in range(1, self.grid_size - 1)
            if (r, c) not in _gaps
        }

        # Curriculum: keep only `active` cartons. The rest are removed from the world
        # and marked delivered, so termination fires when the active ones are done while
        # the observation keeps all 12 slots - inactive ones read "delivered" - and the
        # pinned width survives a promotion. None means the full task; the curriculum
        # opts in by setting `num_cartons` explicitly, because deriving it from
        # difficulty_level (which defaults to 1) would silently change what the
        # verifiers and the greedy baseline measure.
        active = NUM_CARTONS if self.num_cartons is None else self.num_cartons
        active = max(1, min(int(active), NUM_CARTONS))
        self.active_cartons = active

        if active < NUM_CARTONS:
            # Drop from the end of the fixed order, so slot i means the same carton at
            # every difficulty and a policy does not have to relearn the indexing.
            for slot in range(active, NUM_CARTONS):
                rid = self.all_resource_ids[slot]
                pb.removeBody(rid, physicsClientId=self.client_id)
                self.resource_ids.remove(rid)
                self.delivered[slot] = True

        # Baseline for the displacement-based velocity; the first observation of an
        # episode therefore reports zero velocity, which is true.
        self._prev_xy = self._current_xy()
        self._velocity = [(0.0, 0.0)] * self.num_agents
        self._lidar_cache = None

        # Baseline for potential-based shaping. Must be sampled after the curriculum has
        # removed inactive cartons, or the first step pays a spurious jump.
        self._prev_potential = [self._potential(i) for i in range(self.num_agents)]

        return self._get_obs(), self._get_info()

    def _set_arm_and_lidar_joints(self, agent_idx, arm_yaw, finger_pos, lidar_height):
        rid = self.robot_ids[agent_idx]
        st = self.agent_state[agent_idx]
        if st['arm_yaw_joint_idx'] is not None:
            pb.resetJointState(rid, st['arm_yaw_joint_idx'], arm_yaw, physicsClientId=self.client_id)
        if st['left_finger_joint_idx'] is not None:
            pb.resetJointState(rid, st['left_finger_joint_idx'], finger_pos, physicsClientId=self.client_id)
        if st['right_finger_joint_idx'] is not None:
            pb.resetJointState(rid, st['right_finger_joint_idx'], -finger_pos, physicsClientId=self.client_id)
        if st['lidar_joint_idx'] is not None:
            pb.resetJointState(rid, st['lidar_joint_idx'], lidar_height, physicsClientId=self.client_id)

    def _get_cardinal_direction_angle(self, target_world_pos, robot_world_pos, robot_yaw):
        dx = target_world_pos[0] - robot_world_pos[0]
        dy = target_world_pos[1] - robot_world_pos[1]
        target_angle = math.atan2(dy, dx)
        rel_angle = target_angle - robot_yaw
        rel_angle = math.atan2(math.sin(rel_angle), math.cos(rel_angle))
        cardinal_step = round(rel_angle / (math.pi / 2.0))
        cardinal_angle = cardinal_step * (math.pi / 2.0)
        if cardinal_angle == math.pi or cardinal_angle == -math.pi:
            cardinal_angle = math.pi
        return cardinal_angle

    def step(self, joint_action):
        # Movement and speech are separated here and nowhere else. Everything below
        # this line sees `actions` exactly as it did before communication existed; the
        # tokens are used once, at the very end, to fill the channel.
        actions, tokens = split_joint_action(joint_action, self.num_agents)
        if not self.comms and np.any(tokens != MSG_SILENT):
            raise ValueError(
                "message tokens were supplied but this env was built with comms=False, "
                "so nothing would ever hear them and the message slots would stay zero. "
                "Build the env with comms=True, or drop the second action column."
            )

        self.current_step += 1
        num_substeps = self.substeps
        self._lidar_cache = None
        self._dist_cache = None

        # Per-step reward events (spec S3). Filled by the action loop below and
        # consumed by _compute_rewards() after the physics has settled.
        did_pickup = [False] * self.num_agents
        did_deliver = [False] * self.num_agents
        invalid_action = [False] * self.num_agents

        # Pre-compute trajectories
        starts = []
        targets = []
        
        for i in range(self.num_agents):
            rid = self.robot_ids[i]
            pos, orn = pb.getBasePositionAndOrientation(rid, physicsClientId=self.client_id)
            yaw = pb.getEulerFromQuaternion(orn)[2]
            
            # Snap yaw to exact cardinal direction to prevent drift
            yaw = round(yaw / (math.pi / 2.0)) * (math.pi / 2.0)
            
            # Snap to the exact cell centre to prevent drift - z included, which it was
            # not until 2026-08-29: the chassis settled ~0.17 mm per step under gravity
            # and nothing put it back, so a robot sank 0.051 m per 300 steps and would
            # have gone through the floor over a full episode. Unnoticed until the LiDAR
            # beam depended on height.
            r, c = self._world_to_grid(pos[0], pos[1])
            gx, gy = self._grid_to_world(r, c)
            pos = (gx, gy, self._spawn_z)
            
            action = actions[i]
            
            st = self.agent_state[i]
            start_state = {
                'pos': pos, 'yaw': yaw,
                'arm_yaw': st['current_arm_yaw'],
                'finger': st['current_finger_pos'],
                'lidar': st['current_lidar_height'],
                'wheel_delta': 0.0
            }
            target_state = start_state.copy()
            
            if action == 0:  # Forward
                nxt = (pos[0] + self.cell_size * math.cos(yaw), pos[1] + self.cell_size * math.sin(yaw), pos[2])
                if self._can_enter(nxt[0], nxt[1]):
                    target_state['pos'] = nxt
                    target_state['wheel_delta'] = 0.119
                else:
                    # Driving off the grid, or into a shelf, is an invalid action (spec
                    # S3.2). The move is refused rather than executed - nothing in the
                    # world model gives meaning to a robot outside the arena, and letting
                    # one teleport past the boundary wall would corrupt every downstream
                    # metric. See `blocked_cells` in reset() for why shelves joined it.
                    invalid_action[i] = True
            elif action == 1:  # Backward
                nxt = (pos[0] - self.cell_size * math.cos(yaw), pos[1] - self.cell_size * math.sin(yaw), pos[2])
                if self._can_enter(nxt[0], nxt[1]):
                    target_state['pos'] = nxt
                    target_state['wheel_delta'] = -0.119
                else:
                    invalid_action[i] = True
            elif action == 2:  # Turn Left
                target_state['yaw'] = yaw + (math.pi / 2.0)
                target_state['wheel_delta'] = 0.05
            elif action == 3:  # Turn Right
                target_state['yaw'] = yaw - (math.pi / 2.0)
                target_state['wheel_delta'] = -0.05
            elif action == 4 and self.is_carrying[i]:  # Pick Up while already loaded
                invalid_action[i] = True
            elif action == 5 and not self.is_carrying[i]:  # Drop Off with empty gripper
                invalid_action[i] = True
            elif action == 4 and not self.is_carrying[i]:  # Pick Up
                nearest_res = None
                min_dist = float('inf')
                for res_id in self.resource_ids:
                    res_pos, _ = pb.getBasePositionAndOrientation(res_id, physicsClientId=self.client_id)
                    dist = math.hypot(res_pos[0] - pos[0], res_pos[1] - pos[1])
                    if dist < min_dist:
                        min_dist = dist
                        nearest_res = res_id
                
                if nearest_res is not None and min_dist <= self.cell_size * 1.5:
                    res_pos, _ = pb.getBasePositionAndOrientation(nearest_res, physicsClientId=self.client_id)
                    target_state['arm_yaw'] = self._get_cardinal_direction_angle(res_pos, pos, yaw)
                    target_state['finger'] = -0.01
                    target_state['lidar'] = self.lidar_carry_height
                    self.is_carrying[i] = True
                    self.carried_resource_ids[i] = nearest_res
                    self.resource_ids.remove(nearest_res)
                    target_state['res_start_pos'] = res_pos
                    target_state['pickup_target'] = nearest_res
                    did_pickup[i] = True
                else:
                    # Grabbing at nothing - no carton within reach (spec S3.2).
                    invalid_action[i] = True
            elif action == 5 and self.is_carrying[i]:  # Drop Off
                dep_pos, _ = pb.getBasePositionAndOrientation(self.depot_id, physicsClientId=self.client_id)
                dist = math.hypot(pos[0] - dep_pos[0], pos[1] - dep_pos[1])
                if dist <= self.cell_size * DEPOT_RADIUS_CELLS:
                    target_state['arm_yaw'] = self._get_cardinal_direction_angle(dep_pos, pos, yaw)
                    target_state['finger'] = 0.03
                    target_state['lidar'] = 0.0
                    target_state['dropoff'] = True
                    target_state['drop_target'] = dep_pos
                    did_deliver[i] = True
                else:
                    # Carrying, but not at the depot - the drop does not happen.
                    invalid_action[i] = True

            starts.append(start_state)
            targets.append(target_state)

        # Simultaneous Execution
        for step_idx in range(1, num_substeps + 1):
            alpha = step_idx / float(num_substeps)
            for i in range(self.num_agents):
                st = self.agent_state[i]
                rid = self.robot_ids[i]
                s = starts[i]
                t = targets[i]
                
                # Interpolate Pos & Yaw
                ix = s['pos'][0] + (t['pos'][0] - s['pos'][0]) * alpha
                iy = s['pos'][1] + (t['pos'][1] - s['pos'][1]) * alpha
                iyaw = s['yaw'] + (t['yaw'] - s['yaw']) * alpha
                iorn = pb.getQuaternionFromEuler([0, 0, iyaw], physicsClientId=self.client_id)
                pb.resetBasePositionAndOrientation(rid, [ix, iy, s['pos'][2]], iorn, physicsClientId=self.client_id)
                
                # Interpolate Joints
                st['current_arm_yaw'] = s['arm_yaw'] + (t['arm_yaw'] - s['arm_yaw']) * alpha
                st['current_finger_pos'] = s['finger'] + (t['finger'] - s['finger']) * alpha
                st['current_lidar_height'] = s['lidar'] + (t['lidar'] - s['lidar']) * alpha
                self._set_arm_and_lidar_joints(i, st['current_arm_yaw'], st['current_finger_pos'], st['current_lidar_height'])
                
                # Wheels
                if actions[i] in [0, 1]:
                    wd = t['wheel_delta']
                    for idx in st['left_wheel_indices']:
                        pos = pb.getJointState(rid, idx, physicsClientId=self.client_id)[0]
                        pb.resetJointState(rid, idx, pos + wd, physicsClientId=self.client_id)
                    for idx in st['right_wheel_indices']:
                        pos = pb.getJointState(rid, idx, physicsClientId=self.client_id)[0]
                        pb.resetJointState(rid, idx, pos + wd, physicsClientId=self.client_id)
                elif actions[i] == 2: # Turn Left
                    wd = t['wheel_delta']
                    for idx in st['left_wheel_indices']:
                        pos = pb.getJointState(rid, idx, physicsClientId=self.client_id)[0]
                        pb.resetJointState(rid, idx, pos - wd, physicsClientId=self.client_id)
                    for idx in st['right_wheel_indices']:
                        pos = pb.getJointState(rid, idx, physicsClientId=self.client_id)[0]
                        pb.resetJointState(rid, idx, pos + wd, physicsClientId=self.client_id)
                elif actions[i] == 3: # Turn Right
                    wd = t['wheel_delta']
                    for idx in st['left_wheel_indices']:
                        pos = pb.getJointState(rid, idx, physicsClientId=self.client_id)[0]
                        pb.resetJointState(rid, idx, pos + wd, physicsClientId=self.client_id)
                    for idx in st['right_wheel_indices']:
                        pos = pb.getJointState(rid, idx, physicsClientId=self.client_id)[0]
                        pb.resetJointState(rid, idx, pos - wd, physicsClientId=self.client_id)

                # Resource interpolation
                arm_world_angle = iyaw + st['current_arm_yaw']
                carried_rx = ix + self.gripper_reach * math.cos(arm_world_angle)
                carried_ry = iy + self.gripper_reach * math.sin(arm_world_angle)
                
                if actions[i] == 4 and 'pickup_target' in t: # Picking up
                    res_id = t['pickup_target']
                    start_res_pos = t['res_start_pos']
                    cur_res_x = start_res_pos[0] + alpha * (carried_rx - start_res_pos[0])
                    cur_res_y = start_res_pos[1] + alpha * (carried_ry - start_res_pos[1])
                    pb.resetBasePositionAndOrientation(res_id, [cur_res_x, cur_res_y, self.carton_size / 2.0], iorn, physicsClientId=self.client_id)
                elif actions[i] == 5 and 'dropoff' in t: # Dropping off
                    res_id = self.carried_resource_ids[i]
                    if res_id:
                        dep_pos = t['drop_target']
                        cur_res_x = carried_rx + alpha * (dep_pos[0] - carried_rx)
                        cur_res_y = carried_ry + alpha * (dep_pos[1] - carried_ry)
                        pb.resetBasePositionAndOrientation(res_id, [cur_res_x, cur_res_y, self.carton_size / 2.0], iorn, physicsClientId=self.client_id)
                elif self.is_carrying[i] and self.carried_resource_ids[i]: # Carrying
                    res_id = self.carried_resource_ids[i]
                    pb.resetBasePositionAndOrientation(res_id, [carried_rx, carried_ry, self.carton_size / 2.0], iorn, physicsClientId=self.client_id)

            pb.stepSimulation(physicsClientId=self.client_id)
            if self.render_mode == "human":
                time.sleep(0.01)

        # Post-substep handling
        for i in range(self.num_agents):
            if actions[i] == 4 and 'pickup_target' in targets[i]:
                self.agent_state[i]['current_arm_yaw'] = targets[i]['arm_yaw']
                self._set_arm_and_lidar_joints(i, targets[i]['arm_yaw'], -0.01, self.lidar_carry_height)
                
            elif actions[i] == 5 and 'dropoff' in targets[i]:
                res_id = self.carried_resource_ids[i]
                if res_id:
                    # Record the delivery against the carton's stable slot before the
                    # body is removed - once removed, the id can tell us nothing.
                    slot = self.resource_slot.get(res_id)
                    if slot is not None:
                        self.delivered[slot] = True
                    pb.removeBody(res_id, physicsClientId=self.client_id)
                self.is_carrying[i] = False
                self.carried_resource_ids[i] = None
                self.agent_state[i]['current_arm_yaw'] = targets[i]['arm_yaw']

        self._update_kinematics()

        # --- Rewards and episode boundaries (spec S3) -------------------------
        collisions = self._detect_collisions()
        all_delivered = sum(self.delivered) >= NUM_CARTONS

        terminated = bool(all_delivered)
        truncated = bool(not terminated and self.current_step >= self.max_steps)

        rewards, breakdown = self._compute_rewards(
            actions=actions,
            did_pickup=did_pickup,
            did_deliver=did_deliver,
            invalid_action=invalid_action,
            collisions=collisions,
            terminated=terminated,
        )
        self._episode_reward += rewards
        self.last_reward_breakdown = breakdown

        # The channel is filled last, so the observation this step returns already
        # carries what was said during it. That is the one-step delay from the header:
        # a token chosen at step t from the observation at t-1 is heard at t+1.
        self._broadcast(tokens)

        info = self._get_info()
        info.update({
            "collisions": len(collisions),
            "collision_pairs": sorted(collisions),
            "pickups": [bool(p) for p in did_pickup],
            "deliveries": [bool(d) for d in did_deliver],
            "invalid_actions": [bool(v) for v in invalid_action],
            "all_delivered": all_delivered,
            "is_success": all_delivered,
            "reward_breakdown": breakdown,
            "episode_reward": self._episode_reward.copy(),
        })

        return self._get_obs(), rewards.tolist(), terminated, truncated, info

    def action_masks(self):
        """
        Which movement actions are legal right now, per robot: (num_agents, MOVE_ACTIONS)
        of bool, plus an all-True block per token when comms are on.

        Only read by MaskablePPO (`train.py --masked`). Plain PPO never calls it, so
        this changes nothing unless the flag is passed.

        WHY IT IS WORTH HAVING

        Measured on nocomm2_final at 4 cartons: 1,480 of 6,000 robot-steps were PICKUP
        pressed with nothing in reach - 25% of every episode, and 88% of all PICKUP
        presses. `stay` took another 29%. Those actions are knowable-invalid from the
        state the policy is already given, so the network is spending capacity learning a
        rule the environment could simply enforce.

        Note the masks describe the SAME conditions step() already checks before charging
        R_INVALID_ACTION - this method and that branch must not drift apart. Anything
        masked out here would have been refused there anyway.

        Turning and staying are always legal. Only forward and backward can be blocked by
        geometry, and only they can collide.
        """
        masks = np.ones((self.num_agents, MOVE_ACTIONS), dtype=bool)

        for i in range(self.num_agents):
            x, y, yaw_norm = self._canonical_pose(i)
            yaw = yaw_norm * 2.0 * math.pi
            ahead = (x + self.cell_size * math.cos(yaw), y + self.cell_size * math.sin(yaw))
            behind = (x - self.cell_size * math.cos(yaw), y - self.cell_size * math.sin(yaw))
            masks[i, 0] = self._can_enter(*ahead)
            masks[i, 1] = self._can_enter(*behind)

            carrying = self.is_carrying[i]
            masks[i, 4] = (not carrying) and self._carton_within_reach(i)
            masks[i, 5] = carrying and self._at_depot(i)

        if not self.comms:
            return masks
        # Every token is always sayable; the mask exists only so the widths line up with
        # MultiDiscrete([MOVE_ACTIONS, MSG_TOKENS]).
        tokens = np.ones((self.num_agents, MSG_TOKENS), dtype=bool)
        return np.concatenate([masks, tokens], axis=1)

    def _carton_within_reach(self, agent_idx):
        """Is an unclaimed carton inside the pickup radius? The same test step() makes."""
        x, y, _ = self._canonical_pose(agent_idx)
        for rid in self.resource_ids:
            p, _ = pb.getBasePositionAndOrientation(rid, physicsClientId=self.client_id)
            if math.hypot(p[0] - x, p[1] - y) <= self.cell_size * 1.5:
                return True
        return False

    def _broadcast(self, tokens):
        """
        Encode this step's tokens and push them through the channel.

        Produces `self.messages` (what was said, one-hot per speaker) and `self._heard`
        (what arrived, per listener-speaker pair, after dropout and any message_mode
        intervention). Keeping those two separate is what makes the analysis honest:
        the speaker's own record is never altered by the channel, so a token that was
        emitted and then dropped is still counted as emitted when entropy is measured.

        With comms off this writes silence and returns - the message slots of every
        observation stay zero, exactly as they were before step 7.
        """
        n = self.num_agents
        self.message_tokens = np.asarray(tokens, dtype=int).copy()

        if not self.comms:
            self.messages[:] = 0.0
            self._heard[:] = 0.0
            self._dropped[:] = False
            return

        # Speak: one-hot per speaker, all-zero for silence.
        self.messages[:] = 0.0
        for j, tok in enumerate(self.message_tokens):
            if tok != MSG_SILENT:
                self.messages[j, int(tok)] = 1.0

        # Listen: start from a perfect channel, one row per listener.
        heard = np.repeat(self.messages[None, :, :], n, axis=0)

        if self.message_mode == "silent":
            heard[:] = 0.0
        elif self.message_mode == "shuffled":
            # Real tokens from this step, attributed to the wrong speakers. Each
            # listener gets its own derangement-ish permutation, so the marginal token
            # distribution it hears is untouched and only the speaker binding breaks.
            for i in range(n):
                heard[i] = self.messages[self._msg_rng.permutation(n)]
        elif self.message_mode == "random":
            heard[:] = 0.0
            picks = self._msg_rng.integers(0, MSG_TOKENS, size=(n, n))
            for i in range(n):
                for j in range(n):
                    heard[i, j, picks[i, j]] = 1.0

        # Drop: independently per listener-speaker link. A dropped message is the
        # all-zero vector, which is distinguishable from every one-hot - the listener
        # learns "I heard nothing", not a wrong token.
        if self.msg_dropout > 0.0:
            self._dropped = self._msg_rng.random((n, n)) < self.msg_dropout
            np.fill_diagonal(self._dropped, False)   # nobody hears themselves anyway
            heard[self._dropped] = 0.0
        else:
            self._dropped[:] = False

        self._heard = heard.astype(np.float32, copy=False)

    def _in_bounds(self, x, y):
        """Is this world position inside the 13x13 grid?"""
        r, c = self._world_to_grid(x, y)
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    def _can_enter(self, x, y):
        """
        Is this world position a cell a robot may move into - in bounds and not shelf?

        Note what this does NOT include: other robots. Two robots may still occupy the
        same cell and are still charged the spec's -5.0 shared collision for it. That
        stays penalise-and-continue, because unlike a shelf it is a genuinely joint
        mistake and avoiding it is part of the coordination problem being studied.
        """
        r, c = self._world_to_grid(x, y)
        if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
            return False
        return (r, c) not in self.blocked_cells

    def _detect_collisions(self):
        """
        Contacts that count as collisions, as a set of hashable keys.

        Spec S3.1 charges -5.0 per "collision (any pair)", "per collision event". Pairs
        are robot-robot, robot-shelf and robot-wall. (Shelves became solid on 2026-08-29
        - before that the bottom plate sat at 0.30 above a 0.194 chassis and robots drove
        straight underneath.)

        "Event" is read as ONSET, not per step: robots teleport, so an overlapping pair
        stays overlapped until one moves and per-step charging would bill -5.0 repeatedly
        for one mistake. `_colliding_pairs` carries the previous step's contacts so only
        new ones are billed.
        """
        current = set()
        for a in range(self.num_agents):
            for b in range(a + 1, self.num_agents):
                if pb.getContactPoints(bodyA=self.robot_ids[a], bodyB=self.robot_ids[b],
                                       physicsClientId=self.client_id):
                    current.add(("robot", a, b))

        obstacles = list(self.obstacle_ids) + list(getattr(self, "wall_ids", []))
        for i in range(self.num_agents):
            for oid in obstacles:
                if pb.getContactPoints(bodyA=self.robot_ids[i], bodyB=oid,
                                       physicsClientId=self.client_id):
                    # One event per robot per step however many shelf segments it is
                    # inside, so a robot straddling two segments is not billed twice.
                    current.add(("obstacle", i))
                    break

        new_events = current - self._colliding_pairs
        self._colliding_pairs = current
        return new_events

    def _shelf_contacts(self):
        """How many robots are currently touching a shelf or wall. Reported in info."""
        count = 0
        obstacles = list(self.obstacle_ids) + list(getattr(self, "wall_ids", []))
        for i in range(self.num_agents):
            for oid in obstacles:
                if pb.getContactPoints(bodyA=self.robot_ids[i], bodyB=oid,
                                       physicsClientId=self.client_id):
                    count += 1
                    break
        return count

    def _at_depot(self, agent_idx):
        dep, _ = pb.getBasePositionAndOrientation(self.depot_id, physicsClientId=self.client_id)
        x, y, _ = self._canonical_pose(agent_idx)
        return math.hypot(x - dep[0], y - dep[1]) <= self.cell_size * DEPOT_RADIUS_CELLS

    def _geodesic_from(self, cell):
        """
        BFS over enterable cells from `cell` -> {cell: steps}. Cached per env step.

        Four of these per step over a 169-cell grid is nothing next to the physics, and
        it is the difference between a distance a robot can actually walk and one that
        pretends shelves are not there.
        """
        if self._dist_cache is None:
            self._dist_cache = {}
        cached = self._dist_cache.get(cell)
        if cached is not None:
            return cached

        dist = {cell: 0}
        queue = deque([cell])
        while queue:
            cur = queue.popleft()
            r, c = cur
            for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nb in dist or nb in self.blocked_cells:
                    continue
                if not (0 <= nb[0] < self.grid_size and 0 <= nb[1] < self.grid_size):
                    continue
                dist[nb] = dist[cur] + 1
                queue.append(nb)

        self._dist_cache[cell] = dist
        return dist

    def _potential(self, agent_idx):
        """
        Phi(s) for one robot: minus the work left to do, in units of cartons.

            Phi_i = -( n_undelivered
                       + (0.5 if not carrying else 0.0)
                       + 0.5 * min(1, geodesic_cells / GEODESIC_MAX_CELLS) )

        Objective is the depot while carrying, else the nearest carton still on the
        floor, else - if every remaining carton is in someone's gripper - the nearest
        undelivered carton wherever it is.

        THREE DEFECTS THIS HAS HAD, ALL MEASURED

        1. PICKING UP WAS PUNISHED (fixed 2026-08-31). Phi was plain distance to the
           current objective, which switched target the moment `is_carrying` flipped, so
           it jumped downward at the one transition the task is built around. On a
           perfect greedy episode all four pickups scored negative total reward (-0.469,
           -0.107, -0.399, -0.698). The 0.5 handoff term is the fix: not carrying is half
           a carton of work away from carrying.

        2. IDLE ROBOTS HAD NO GRADIENT (fixed 2026-09-03). The target search skipped any
           carton not in `resource_ids`, which excludes cartons being carried - so a
           robot with nothing left to claim got d = 0.0, a constant Phi, and no reward
           for moving at all. Since only movement can collide and an invalid PICKUP
           cannot, the cheapest action for such a robot is to grab at thin air. Measured
           on nocomm2_final at 4 cartons: 1,480 PICKUP presses with nothing in reach out
           of 6,000 robot-steps, 25% of the entire episode, with movement at 21%.

        3. THE DISTANCE WAS EUCLIDEAN, IN A WAREHOUSE FULL OF SHELVES (fixed
           2026-09-03). Straight-line distance ignores the shelf rows the robot has to
           drive around, so **10.2% of free cells were local minima** - measured over
           seeds 1000-1009 - where every legal move increased the distance to the nearest
           carton and the shaping therefore punished every one of them. A robot in the
           wrong aisle was paid to stand still. The distance is now geodesic: BFS over
           the cells a robot can actually enter, which has no local minima by
           construction.

        WHY THE DISTANCE TERM IS BOUNDED BY 0.5

        It makes both task transitions provably non-negative, which the old form did not:

            pickup   dPhi = 0.5 + 0.5*(dist_carton - dist_depot) >= 0
            delivery dPhi = 0.5 - 0.5*dist_next                  >= 0

        because each bounded term is in [0, 0.5]. Under the old `d / (2 * span)` a pickup
        in the far corner scored dPhi = 0.5 - 18.4/26 = -0.21, i.e. negative, and only
        avoided being caught because no test happened to pick that corner.

        GEODESIC_MAX_CELLS is 2 x grid_size = 26. The longest real path measured across
        seeds 1000-1009 is 24 cells, so the term saturates only on routes that do not
        occur; beyond it the gradient is flat, which is the correct behaviour for a
        target that is unreachable.
        """
        x, y, _ = self._canonical_pose(agent_idx)
        n_left = sum(1 for slot in range(self.active_cartons) if not self.delivered[slot])
        dist = self._geodesic_from(self._world_to_grid(x, y))

        def cells_to(world_xy):
            """Geodesic cells to a world position; saturated when unreachable."""
            target = self._world_to_grid(world_xy[0], world_xy[1])
            d = dist.get(target)
            if d is not None:
                return d
            # A carton shoved inside a shelf is not enterable, so BFS never reaches its
            # cell. The nearest enterable neighbour is what a robot would actually stand
            # on to pick it up, and the env's 1.5-cell reach makes that a real pickup.
            r, c = target
            neighbours = [dist.get(n) for n in
                          ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))]
            reachable = [n for n in neighbours if n is not None]
            return min(reachable) if reachable else GEODESIC_MAX_CELLS

        if self.is_carrying[agent_idx]:
            dep, _ = pb.getBasePositionAndOrientation(
                self.depot_id, physicsClientId=self.client_id
            )
            d_cells = cells_to(dep)
            handoff = 0.0
        else:
            # Prefer a carton nobody is holding; fall back to any undelivered one so a
            # robot with nothing to claim still has somewhere to be. See defect 2.
            free_best = held_best = None
            for slot, rid in enumerate(self.all_resource_ids):
                if slot >= self.active_cartons or self.delivered[slot]:
                    continue
                p, _ = pb.getBasePositionAndOrientation(rid, physicsClientId=self.client_id)
                d = cells_to(p)
                if rid in self.resource_ids:
                    free_best = d if free_best is None else min(free_best, d)
                else:
                    held_best = d if held_best is None else min(held_best, d)
            best = free_best if free_best is not None else held_best
            d_cells = 0.0 if best is None else best
            handoff = 0.5

        dist_term = 0.5 * min(1.0, d_cells / GEODESIC_MAX_CELLS)
        return -(n_left + handoff + dist_term)

    def _shaping_reward(self):
        """
        F_i = scale * (Phi_i(s') - Phi_i(s)), one per robot.

        WHY THERE IS NO GAMMA HERE

        The textbook form `gamma * Phi(s') - Phi(s)` is exactly policy-invariant, but Phi
        is negative everywhere here, so it carries a per-step drift of `-(1-gamma)*Phi` -
        POSITIVE, largest when furthest from finishing, i.e. a standing bonus for
        loitering worth ~+0.005*scale every step. gamma = 1 drops the drift at the cost
        of strict policy-invariance, which is why shaping is declared as a deviation
        wherever these runs are reported.

        `_prev_potential` is refreshed here, so this must be called exactly once per
        step, after the physics has settled.
        """
        out = np.zeros(self.num_agents, dtype=np.float64)
        if not self.shaping:
            return out
        for i in range(self.num_agents):
            phi = self._potential(i)
            out[i] = self.shaping_scale * (phi - self._prev_potential[i])
            self._prev_potential[i] = phi
        return out

    def _compute_rewards(self, actions, did_pickup, did_deliver, invalid_action,
                         collisions, terminated):
        """
        Spec S3: R_total_i = 0.90 * R_shared + 0.10 * R_individual_i

        Returns (rewards array of length num_agents, breakdown dict). The breakdown is
        what scripts/verify_rewards.py prints, so every term can be read off a real
        episode instead of inferred from the total.
        """
        deliveries_this_step = sum(1 for d in did_deliver if d)

        # ---- Shared (S3.1), identical for all four agents --------------------
        shared = 0.0
        shared_terms = {}

        if deliveries_this_step:
            shared_terms["per_delivery"] = R_PER_DELIVERY * deliveries_this_step
        if collisions:
            shared_terms["collision"] = R_COLLISION * len(collisions)
        shared_terms["time_penalty"] = R_TIME_PENALTY

        if terminated:
            shared_terms["all_delivered"] = R_ALL_DELIVERED
            # Makespan bonus, once per episode, awarded on the terminating step.
            # T_max is max_steps and T_actual is the step the job finished on; the
            # spec's formula is otherwise unchanged. Finishing at the limit pays 0,
            # finishing instantly pays the full +50.
            if not self._makespan_awarded:
                t_actual = min(self.current_step, self.max_steps)
                shared_terms["makespan_bonus"] = (
                    R_MAKESPAN_SCALE * (self.max_steps - t_actual) / self.max_steps
                )
                self._makespan_awarded = True

        shared = sum(shared_terms.values())

        # ---- Individual (S3.2), per agent ------------------------------------
        individual = np.zeros(self.num_agents, dtype=np.float64)
        individual_terms = []
        for i in range(self.num_agents):
            terms = {}
            if did_pickup[i]:
                terms["own_pickup"] = R_OWN_PICKUP
            if did_deliver[i]:
                terms["own_delivery"] = R_OWN_DELIVERY
            if invalid_action[i]:
                terms["invalid_action"] = R_INVALID_ACTION

            # Idle: linear speed below threshold and not parked at the depot.
            # `idle_penalises_turning` follows the spec literally when True (a robot
            # turning on the spot has v ~ 0, so it is idle); set False to exempt turns.
            speed = math.hypot(*self._velocity[i])
            turning = int(actions[i]) in (2, 3)
            idle = speed < IDLE_SPEED_THRESHOLD and not self._at_depot(i)
            if idle and turning and not self.idle_penalises_turning:
                idle = False
            if idle:
                terms["idle"] = R_IDLE_PENALTY

            # R_REPLAN_PENALTY is intentionally never applied - there is no planner
            # in this environment to re-trigger. See the reward block at module top.

            individual[i] = sum(terms.values())
            individual_terms.append(terms)

        # ---- Potential-based shaping (NOT in the specification) ---------------
        # R_i = 0.90 * R_shared + 0.10 * R_individual_i + F_i
        #
        # F_i sits OUTSIDE the 90/10 split, which it did not until 2026-08-31: it was
        # added to the individual bucket and so multiplied by 0.10, making
        # shaping_scale=15.0 silently deliver 1.5. The split and every R_* constant are
        # the spec's and untouched; shaping is our addition and is declared as such in
        # TRAINING.md.
        shaping = self._shaping_reward()

        rewards = SHARED_WEIGHT * shared + INDIVIDUAL_WEIGHT * individual + shaping
        for i in range(self.num_agents):
            if shaping[i] != 0.0:
                # Recorded in the individual breakdown for readability only. It is not
                # part of the individual total and is not weighted by it.
                individual_terms[i]["shaping (unweighted)"] = shaping[i]

        breakdown = {
            "shared_terms": shared_terms,
            "shared_total": shared,
            "individual_terms": individual_terms,
            "individual_totals": individual.tolist(),
            "weights": (SHARED_WEIGHT, INDIVIDUAL_WEIGHT),
            "rewards": rewards.tolist(),
        }
        return rewards, breakdown

    def _canonical_pose(self, agent_idx):
        """
        (x, y, yaw) snapped to the grid - the pose step() actually acts on.

        Raw base positions drift by a few millimetres as the chassis settles during the
        substeps, which is why step() re-snaps to grid centres and cardinal headings
        before computing anything. The observation has to agree with the dynamics: if
        pickup range is measured from the snapped pose, the robot must observe the
        snapped pose too.

        Reading raw positions here instead produced a forward move that measured 0.992
        cells and a stationary robot with non-zero velocity - settling noise showing up
        as motion the environment does not believe happened.
        """
        rid = self.robot_ids[agent_idx]
        pos, orn = pb.getBasePositionAndOrientation(rid, physicsClientId=self.client_id)
        yaw = pb.getEulerFromQuaternion(orn)[2]
        yaw = round(yaw / (math.pi / 2.0)) * (math.pi / 2.0)
        r, c = self._world_to_grid(pos[0], pos[1])
        gx, gy = self._grid_to_world(r, c)
        return gx, gy, yaw

    def _current_xy(self):
        """Planar position of every robot, snapped. See _canonical_pose."""
        return [self._canonical_pose(i)[:2] for i in range(self.num_agents)]

    def _update_kinematics(self):
        """
        Velocity as displacement since the previous step, in cells per step.

        Deliberately not pb.getBaseVelocity: motion is interpolated over substeps and
        then snapped back to the grid, so the base velocity sampled afterwards is
        settling noise. Displacement reports what the robot actually did - a forward
        move reads as exactly 1.0 cell - and stays meaningful when the velocity-
        controlled motion model replaces the teleport.
        """
        current = self._current_xy()
        self._velocity = [
            ((x - px) / self.cell_size, (y - py) / self.cell_size)
            for (x, y), (px, py) in zip(current, self._prev_xy)
        ]
        self._prev_xy = current

    def _pose_features(self, agent_idx):
        """
        (x, y, heading) normalised into [-1, 1]. Exactly OBS_OWN_POSE values.

        Heading is wrapped into [0, 1) rather than divided by pi. Snapping rounds to the
        nearest multiple of pi/2, so a west-facing robot lands on +pi or -pi depending on
        which side of the boundary it settled - and yaw/pi would then encode one physical
        heading as either +1.0 or -1.0, two extremes of the range for the same fact.
        Wrapping first removes that: each cardinal has exactly one encoding, and the four
        come out as 0.0, 0.25, 0.5, 0.75.
        """
        x, y, yaw = self._canonical_pose(agent_idx)
        heading = (yaw % (2.0 * math.pi)) / (2.0 * math.pi)
        return (
            x / self._arena_half_extent,
            y / self._arena_half_extent,
            heading,
        )

    def _carton_status(self, agent_idx):
        """
        One float per carton slot, in the fixed order frozen at reset().

        Availability is read from `resource_ids` (the cartons still on the floor)
        rather than tracked separately, so the observation cannot drift out of sync
        with the pickup logic that mutates it.
        """
        available = set(self.resource_ids)
        mine = self.carried_resource_ids[agent_idx]
        status = []
        for slot, rid in enumerate(self.all_resource_ids):
            if self.delivered[slot]:
                status.append(CARTON_DELIVERED)
            elif rid == mine:
                status.append(CARTON_CLAIMED_BY_ME)
            elif rid in available:
                status.append(CARTON_AVAILABLE)
            else:
                # Not on the floor, not delivered, not mine - another robot has it.
                status.append(CARTON_CLAIMED_BY_OTHER)
        return status

    def _get_lidar_scan(self, agent_idx):
        """
        One 2D planar scan for a robot: LIDAR_NUM_RAYS normalised distances in [0, 1].

        Spec S2.2 geometry - 270 degrees front-facing (-135 to +135 about the heading),
        LIDAR_MIN_RANGE to LIDAR_MAX_RANGE metres, Gaussian noise of
        LIDAR_NOISE_SIGMA + LIDAR_NOISE_RANGE_FRAC of the reading. Ray count is reduced
        from the spec's 720; see the constants block for why.

        Ported in spirit from the single-agent branch's `_get_lidar_scan`: one
        rayTestBatch for the whole sweep rather than LIDAR_NUM_RAYS separate rayTest
        calls. That mattered there and matters four times as much here.

        Cast from the snapped pose at chassis height, so the sweep agrees with the
        collision geometry (see the LiDAR constants block). Noise is drawn from
        `self.np_random`, which `reset(seed=...)` seeds, so a scan is reproducible for
        a given seed and action sequence.
        """
        x, y, yaw = self._canonical_pose(agent_idx)
        z = LIDAR_BEAM_Z

        half = LIDAR_FOV_RAD / 2.0
        angles = yaw + np.linspace(-half, half, LIDAR_NUM_RAYS)
        froms, tos = [], []
        for a in angles:
            ca, sa = math.cos(a), math.sin(a)
            froms.append([x + LIDAR_START_RADIUS * ca, y + LIDAR_START_RADIUS * sa, z])
            tos.append([x + LIDAR_MAX_RANGE * ca, y + LIDAR_MAX_RANGE * sa, z])

        results = pb.rayTestBatch(froms, tos, physicsClientId=self.client_id)

        # hit_fraction is along the cast segment, which starts at LIDAR_START_RADIUS -
        # convert back to a distance from the robot centre before normalising.
        cast_span = LIDAR_MAX_RANGE - LIDAR_START_RADIUS
        span = LIDAR_MAX_RANGE - LIDAR_MIN_RANGE
        fracs = np.array([r[2] for r in results], dtype=np.float64)
        hit = np.array([r[0] >= 0 for r in results], dtype=bool)
        dists = LIDAR_START_RADIUS + fracs * cast_span

        # Noise applies to real returns only. A ray that hit nothing reports max range
        # as a clean "no obstacle" rather than a jittered one.
        if self.lidar_noise and hit.any():
            sigma = LIDAR_NOISE_SIGMA + LIDAR_NOISE_RANGE_FRAC * dists
            dists = np.where(
                hit, dists + self.np_random.normal(0.0, 1.0, dists.shape) * sigma, dists
            )
        dists = np.clip(dists, LIDAR_MIN_RANGE, LIDAR_MAX_RANGE)
        return ((dists - LIDAR_MIN_RANGE) / span).astype(np.float32)

    def _cached_lidar(self):
        """
        The scans used by both _get_obs() and _get_info(), swept once.

        Both run at the same settled physics state, so a single sweep is correct. The
        cache is cleared at the top of every step and at the end of reset, so it never
        spans a physics update. Without it each step fired 2 x 4 x LIDAR_NUM_RAYS rays.
        """
        if self._lidar_cache is None:
            self._lidar_cache = [
                self._get_lidar_scan(i) for i in range(self.num_agents)
            ]
        return self._lidar_cache

    def _carton_positions(self, depot_pos):
        """
        Flat (x, y) per carton slot, normalised like pose. Same index as _carton_status.

        A carton being carried still exists as a body and moves with the gripper, so its
        real position is reported. A delivered carton has been removed from the world;
        the depot is reported for it, which is factually where it went - and the status
        slot already marks it delivered, so a policy can ignore the value.
        """
        out = []
        for slot, rid in enumerate(self.all_resource_ids):
            if self.delivered[slot]:
                x, y = depot_pos[0], depot_pos[1]
            else:
                p, _ = pb.getBasePositionAndOrientation(rid, physicsClientId=self.client_id)
                x, y = p[0], p[1]
            out.append(x / self._arena_half_extent)
            out.append(y / self._arena_half_extent)
        return out

    def _get_obs(self):
        """
        One row per robot, shape (num_agents, OBS_DIM_V3), float32 in [-1, 1].

        Every write goes through OBS_SLICES, so a layout change moves the data and the
        indices together. The final assertion is cheap and catches a component that
        silently produced the wrong number of values.
        """
        obs = np.zeros((self.num_agents, self.obs_dim), dtype=np.float32)

        depot_pos, _ = pb.getBasePositionAndOrientation(
            self.depot_id, physicsClientId=self.client_id
        )
        elapsed = self.current_step / float(self.max_steps)
        poses = [self._pose_features(i) for i in range(self.num_agents)]
        carton_xy = self._carton_positions(depot_pos)
        scans = self._cached_lidar()

        for i in range(self.num_agents):
            row = obs[i]
            x, y, _ = poses[i]

            row[OBS_SLICES["own_pose"]] = poses[i]
            row[OBS_SLICES["own_velocity"]] = self._velocity[i]
            row[OBS_SLICES["own_carrying"]] = 1.0 if self.is_carrying[i] else 0.0

            # Fixed agent order with self skipped: robot 0 sees 1, 2, 3; robot 1 sees
            # 0, 2, 3. Consistent per robot, which is what a shared policy needs.
            others = [j for j in range(self.num_agents) if j != i]
            row[OBS_SLICES["other_poses"]] = [v for j in others for v in poses[j]]
            row[OBS_SLICES["other_carrying"]] = [
                1.0 if self.is_carrying[j] else 0.0 for j in others
            ]

            row[OBS_SLICES["carton_status"]] = self._carton_status(i)
            row[OBS_SLICES["carton_positions"]] = carton_xy

            # Offset to the depot rather than a unit bearing: same 2 floats, but it
            # carries distance as well as direction.
            robot_x = x * self._arena_half_extent
            robot_y = y * self._arena_half_extent
            row[OBS_SLICES["depot_direction"]] = [
                (depot_pos[0] - robot_x) / self._arena_span,
                (depot_pos[1] - robot_y) / self._arena_span,
            ]

            row[OBS_SLICES["elapsed_time"]] = elapsed
            row[OBS_SLICES["lidar"]] = scans[i]

            # Message slots, in the same fixed speaker order the pose slots use, so
            # block b of this slice is the same robot for the whole episode.
            #
            # Read from `_heard[i]`, not from `self.messages`: what robot i receives is
            # what survived the channel into robot i specifically. Reading the raw
            # broadcasts here would make dropout invisible in the observation while
            # still costing time in the log, which is the kind of bug that produces a
            # protocol robust to nothing.
            row[OBS_SLICES["messages"]] = np.concatenate(
                [self._heard[i][j] for j in others]
            )

        np.clip(obs, -1.0, 1.0, out=obs)

        if obs.shape != (self.num_agents, OBS_DIM_V3):
            raise RuntimeError(
                f"Observation shape {obs.shape} does not match the pinned "
                f"({self.num_agents}, {OBS_DIM_V3})."
            )
        return obs

    def _get_info(self):
        poses = []
        for rid in self.robot_ids:
            p, _ = pb.getBasePositionAndOrientation(rid, physicsClientId=self.client_id)
            poses.append(p)
        return {
            "robot_pos": poses,
            "remaining_resources": len(self.resource_ids) + sum(self.is_carrying),
            # Deliveries the robots actually made this episode. Curriculum-disabled
            # slots are marked delivered at reset so termination works, but counting
            # them here reported "8 delivered" on a fresh 4-carton reset - true for the
            # flag array, actively misleading as a metric.
            "delivered": sum(self.delivered[:self.active_cartons]),
            "delivered_flags_total": sum(self.delivered),
            "obs_dim": self.obs_dim,
            "active_cartons": self.active_cartons,
            "is_carrying": list(self.is_carrying),
            # Communication (roadmap step 7). `message_tokens[j]` is what robot j SAID
            # this step - MSG_SILENT (-1) when it said nothing, which is every step
            # under comms=False and the first observation of every episode.
            # `messages_dropped` counts listener-speaker links the channel lost.
            # Neither is visible anywhere in the reward, and both are what
            # scripts/analyse_messages.py reads.
            "comms": self.comms,
            "message_tokens": self.message_tokens.tolist(),
            "messages_dropped": int(self._dropped.sum()),
            "shelf_contacts": self._shelf_contacts(),
            # Normalised scans as the policy sees them, plus the same thing in metres
            # so a human reading a log does not have to undo the normalisation.
            "lidar": [s.copy() for s in self._cached_lidar()],
            "lidar_distances": [
                (LIDAR_MIN_RANGE + s.astype(np.float64) * (LIDAR_MAX_RANGE - LIDAR_MIN_RANGE))
                for s in self._cached_lidar()
            ],
        }

    def render(self):
        pass

    def close(self):
        """
        Disconnect from the physics server. Safe to call more than once.

        The guard is not decoration: a second call used to raise
        pybullet.error("Not connected to physics server."). That was a harmless nuisance
        while everything ran in one process, but a vectorised env closes its workers on
        teardown *and* again from __del__ during interpreter shutdown, where the second
        exception surfaces as a confusing traceback that hides whatever actually went
        wrong.
        """
        if getattr(self, "_closed", False):
            return
        if hasattr(self, "client_id"):
            try:
                pb.disconnect(physicsClientId=self.client_id)
            except pb.error:
                pass  # already gone - nothing to release
        self._closed = True
