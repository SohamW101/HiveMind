import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as pb
import pybullet_data
import random
import math
import time

# =============================================================================
# OBSERVATION LAYOUT V2 - PINNED AT 105 FLOATS PER ROBOT. READ BEFORE CHANGING.
# =============================================================================
#
# VERSION HISTORY
#   V1  81 floats. Shipped 2026-08-28, superseded the same day, NEVER TRAINED
#       AGAINST - no checkpoint exists at this width. It carried carton *status*
#       but no carton *positions*, which made it blind to the warehouse: five
#       different seeds produced five different layouts and one byte-identical
#       observation. A policy on it could only search at random.
#   V2 105 floats. Adds 12 carton positions (2 floats each). Current.
#
# The version number exists so that any future width change is dated rather than
# silent. V1 is kept as a constant, not as a code path: selecting it is refused,
# because the env can no longer produce it.
#
# Phase 1 was bitten by an unpinned observation width: the flatten width is baked
# into the saved weights, so a checkpoint can only ever be loaded back into an env
# that reports the identical dimension. A silent change here does not raise at
# training time - it invalidates every model trained before it, retroactively, and
# the failure surfaces much later as a shape error nobody can date.
#
# Three defences, in order of how much they actually help:
#   1. The per-component constants below are the single source of truth. Nothing
#      in this file hard-codes 81, 33 or any slice bound.
#   2. _OBS_SLICES is checked at import time to tile [0, OBS_DIM_V1) exactly with
#      no gap and no overlap. Add a component without updating the total and the
#      module fails to import - loudly, immediately, before any training starts.
#   3. HiveMindMultiAgentEnv.__init__ rejects an obs_dim that disagrees with
#      OBS_DIM_V1, so a stale call site cannot quietly build a mismatched env.
#
# LAYOUT (per robot; the full observation is (num_agents, OBS_DIM_V1) float32)
#
#   slice     size  component              encoding
#   -------   ----  ---------------------  --------------------------------------
#   [ 0: 3]      3  own pose               x, y normalised by arena half-extent;
#                                          heading wrapped into [0, 1), so the four
#                                          cardinals read 0.0 / 0.25 / 0.5 / 0.75
#   [ 3: 5]      2  own velocity           XY displacement over the last step,
#                                          in cells/step, clipped to [-1, 1]
#   [ 5: 6]      1  own carrying flag      0.0 or 1.0
#   [ 6:15]      9  other robots' poses    3 robots x (x, y, yaw), same encoding
#                                          as own pose, in fixed agent order with
#                                          self skipped
#   [15:18]      3  other carrying flags   0.0 or 1.0
#   [18:30]     12  carton status          one float per carton, stable index:
#                                          0.00 available    (still on the floor)
#                                          0.33 claimed by me (I am carrying it)
#                                          0.67 claimed by other
#                                          1.00 delivered
#   [30:54]     24  carton positions       12 x (x, y), same index as the status
#                                          slots above, same normalisation as
#                                          pose. A carried carton reports where
#                                          it currently is; a delivered one
#                                          reports the depot
#   [54:56]      2  depot direction        offset to depot, normalised by arena
#                                          span - carries bearing AND distance
#   [56:57]      1  elapsed time           current_step / max_steps
#   [57:105]    48  message slots          3 other robots x MSG_TOKENS, ALL ZERO
#                                          in v2 (roadmap step 7 fills these)
#   -------   ----
#   total      105
#
# WHY THE MESSAGE SLOTS EXIST NOW, ZEROED
# Reserving them today is the whole point of pinning. Roadmap step 7 adds the
# 16-token broadcast; if the slots were added *then*, the observation would grow
# from 33 to 81 and every no-comms checkpoint from step 6 would become unloadable
# - destroying exactly the before/after comparison that is the contribution. With
# the slots reserved, step 7 writes into them and the dimension never moves.
#
# CHOICES MADE HERE THAT THE ROADMAP LEFT OPEN (all reversible, none free)
# - Pose is 3 values, so heading is one number, not sin/cos (which would need 4 and
#   break the pinned 33). Defensible only because step() snaps headings to the four
#   cardinals: the network learns 4 discrete values, it does not regress a
#   continuous angle across the wrap. If the physics motion model later allows
#   arbitrary headings, revisit this - sin/cos is then worth its extra float, and
#   that is an OBS_DIM_V2.
# - Every pose in the observation is the SNAPPED pose, matching what step() acts on
#   rather than the raw base transform. The chassis settles a few millimetres during
#   the substeps; reading that raw made a one-cell move measure 0.992 cells and gave
#   a stationary robot a non-zero velocity.
# - Velocity is last-step displacement, NOT pb.getBaseVelocity. Under the current
#   grid-teleport motion the base velocity after snapback is meaningless noise;
#   displacement says what the robot actually did, and it stays correct when the
#   velocity-controlled motion model replaces it.
# - Carton status is one ordinal float per carton, not a 4-way one-hot (which
#   would need 48). The four values are evenly spaced so no pair is closer than
#   any other, but this does encode an ordering that does not really exist. If
#   step 6 struggles to tell "claimed by other" from "delivered", this is the
#   first thing to widen - and it is a dimension change, so it must be a new
#   OBS_DIM_V2 with its own constant, never an edit to V1.
# - Other robots' poses are absolute, in fixed agent order with self skipped.
#   Relative-to-self coordinates would likely generalise better under a shared
#   policy; that is a clean ablation, not a v1 requirement.
# - Depot direction is redundant - the depot is fixed at grid (0,0) every episode,
#   so it is fully determined by own pose. Kept because it is 2 floats and saves
#   the network learning the transform.
#
# - Carton positions are absolute, and share the status slots' index, so slot i is
#   the same carton in both. A carton being carried reports where it actually is
#   (it moves with the gripper); a delivered carton reports the depot, which is
#   where it ended up. The status slot already says "delivered", so a policy that
#   wants to ignore that position can.
# =============================================================================

NUM_AGENTS = 4
NUM_CARTONS = 12

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
# LiDAR (spec S2.2), new in V3
# ---------------------------------------------------------------------------
# Spec asks for 720 rays over a 270-degree front-facing arc, 0.1 - 10.0 m, with
# Gaussian noise of sigma = 0.01 m + 1% of range.
#
# The FOV, range and noise model are taken as written. The ray COUNT is not: 720
# floats per robot would be seven times the entire rest of the observation (105),
# and 4 x 720 = 2880 raycasts every step is a real cost in a loop that already runs
# 30 physics substeps. 72 rays is exactly one tenth of the spec's resolution, which
# is still 3.75 degrees per ray - at 3 m that is a 20 cm gap between beams and at
# 10 m it is 65 cm, so 1 m obstacles in a 1 m grid stay resolvable everywhere in the
# arena. Raising it is a V4, not an edit.
LIDAR_NUM_RAYS = 72
LIDAR_FOV_RAD = math.radians(270.0)
LIDAR_MIN_RANGE = 0.1
LIDAR_MAX_RANGE = 10.0

# Rays start this far from the robot centre so the sweep does not range on the robot's
# own body. The widest link at beam height is the front wheel, at 0.2363 m from centre;
# 0.28 clears it with margin for the arm swinging during a pickup.
#
# Casting from the spec's 0.1 m instead put the origin *inside* the chassis, and the
# scan came back reading 0.12 m minimum and 2.2 m maximum in a 13 m arena - every beam
# stopped on the robot's own wheels. Reported distances are still measured from the
# robot centre, so the geometry stays honest; the practical effect is that this robot
# cannot see anything closer than 0.28 m, which the spec's 0.1 m minimum assumed a
# smaller body for.
LIDAR_START_RADIUS = 0.28

# Beam height above the floor, absolute rather than read off the chassis.
#
# It has to sit inside two bands at once: the robot chassis (0.094 - 0.194) so the
# sweep reports what the body would collide with, and the bottom shelf plate
# (0.140 - 0.220) so shelves are actually visible. 0.17 is inside both with ~25 mm of
# margin either side.
#
# Reading the live base z instead was wrong twice over: the chassis settles downward
# a fraction of a millimetre per step, so after ~250 steps the beam had sunk below the
# plate's 0.14 underside and every ray flew *under* the shelving into the far wall -
# a robot one cell from a shelf reported 2.5 m of clear space. The sink itself is now
# fixed (see _spawn_z), but the beam stays on a constant so perception cannot silently
# depend on chassis dynamics again.
LIDAR_BEAM_Z = 0.17
LIDAR_NOISE_SIGMA = 0.01      # metres, constant term
LIDAR_NOISE_RANGE_FRAC = 0.01  # plus 1% of the measured range

# WHY THE BEAM SITS AT CHASSIS HEIGHT
# The robot carries a modelled LiDAR link at z ~ 0.21 - 0.25 that rises to 0.5 m while
# carrying. Casting from there would be faithful to the URDF and wrong for the task:
# the bottom shelf plate spans 0.14 - 0.22, so a beam at 0.23 passes just over the
# plate the chassis is about to hit, and a carrying robot's raised mast would see over
# the shelving entirely. The sensor must see what the body collides with, so rays are
# cast at the chassis centre instead and the height does not change with carrying.
OBS_LIDAR = LIDAR_NUM_RAYS

# Everything the robot observes about the world, before any communication.
OBS_WORLD_DIM = (
    OBS_OWN_POSE + OBS_OWN_VELOCITY + OBS_OWN_CARRYING
    + OBS_OTHER_POSES + OBS_OTHER_CARRYING
    + OBS_CARTON_STATUS + OBS_CARTON_POSITIONS
    + OBS_DEPOT_DIRECTION + OBS_ELAPSED_TIME + OBS_LIDAR
)  # 129

# Roadmap step 7: each robot broadcasts MSG_TOKENS values; a robot hears the other
# three, never itself. Spec section 2.4 sets the vocabulary at K = 16 tokens.
MSG_TOKENS = 16
OBS_MESSAGE_DIM = MSG_TOKENS * (NUM_AGENTS - 1)  # 48

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

# Carton status values. Evenly spaced across [0, 1] - see the note above about
# this being ordinal rather than categorical.
CARTON_AVAILABLE = 0.0
CARTON_CLAIMED_BY_ME = 1.0 / 3.0
CARTON_CLAIMED_BY_OTHER = 2.0 / 3.0
CARTON_DELIVERED = 1.0


# =============================================================================
# REWARD SPECIFICATION - transcribed from MAWC_Technical_Specification.pdf, S3
# =============================================================================
#
# S3.1 Shared rewards (90% weight), applied identically to all 4 agents:
#     all 12 resources delivered   +100.0                        once per episode
#     per successful delivery      +10.0                          per delivery
#     makespan bonus               +50 x (T_max - T_actual)/T_max once per episode
#     collision (any pair)         -5.0                           per event
#     time penalty                 -0.05                          every step
#
# S3.2 Individual rewards (10% weight), per agent:
#     successful personal pickup   +1.0                           per pickup
#     successful personal delivery +2.0                           per delivery
#     idle penalty                 -0.02   (v < 0.1 m/s, not at depot), per step
#     replanning penalty           -0.1    (A* re-triggered)       per replan
#     invalid action penalty       -0.5                           per invalid action
#
# S3.3  R_total_i = 0.90 * R_shared + 0.10 * R_individual_i
#
# TWO SPEC ITEMS AND HOW THEY LAND HERE
#
# 1. The replanning penalty has no trigger in this environment and is NOT
#    implemented. It fires when A* is re-run, and project decision 3 puts A*,
#    DWA and EKF out of scope - robots here are placed on a grid, not driven
#    along a planned path. There is nothing to replan, so charging for it would
#    be inventing an event. The constant is defined below and left unused so the
#    omission is visible rather than silent.
#
# 2. The idle penalty is defined on linear velocity ("v < 0.1 m/s"), and the
#    spec's own velocity component is (v, omega) - so by its letter a robot
#    turning on the spot IS idle. That is implemented literally. It slightly
#    charges for turning, which navigation needs; the effect is small (0.10
#    weight x 0.02 = 0.002 per step, against a 0.045 time penalty) but it is a
#    real choice, not an oversight. Turn off with `idle_penalises_turning=False`
#    if step 6 shows the policy avoiding turns.
#
# The spec's example uses T_max = 500 s. This env has no seconds - it counts
# steps - so T_max is max_steps and T_actual is the step the episode ended on.
# The bonus formula is unchanged.
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

# "v < 0.1 m/s" in cells per step. A move covers exactly 1.0 cell/step, so any
# threshold below 1.0 separates moving from not moving; 0.1 keeps the spec's number.
IDLE_SPEED_THRESHOLD = 0.1

# The spec's depot is a 2 m x 2 m zone; here it is one grid cell, and "at depot" uses
# the same 1.5-cell radius that the drop-off action already uses, so a robot parked
# where it can legally deliver is not also charged for idling there.
DEPOT_RADIUS_CELLS = 1.5


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
        note = " (zeros in v3)" if name == "messages" else ""
        if name == "lidar":
            note = f" ({LIDAR_NUM_RAYS} rays, 270 deg, {LIDAR_MAX_RANGE} m)"
        lines.append(f"  [{sl.start:3d}:{sl.stop:3d}]  {sl.stop - sl.start:2d}  {name}{note}")
    lines.append(f"  world features: {OBS_WORLD_DIM}   message slots: {OBS_MESSAGE_DIM}")
    return "\n".join(lines)


class HiveMindMultiAgentEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    carton_size = 0.5
    gripper_reach = 0.3
    lidar_initial_height = 0.0
    lidar_carry_height = 0.5

    def __init__(self, render_mode=None, difficulty_level=1, obs_dim=DEFAULT_OBS_DIM,
                 show_lidar=None, obs_size=None, idle_penalises_turning=True,
                 lidar_noise=True, substeps=None):
        super().__init__()
        # Physics substeps per environment step: a one-cell move is executed by
        # teleporting the robot across this many resetBasePositionAndOrientation +
        # stepSimulation pairs.
        #
        # It is an interpolation, not the motion model. The final pose is the snapped
        # grid target regardless, and collisions are read from that final pose, so the
        # count does NOT affect makespan, collisions, deliveries, invalid actions or
        # completion. Measured across the full 30-seed greedy baseline at 30, 10, 5 and
        # 1 substeps: identical to the decimal every time (97.6 mean, 96.5 median,
        # sd 8.3, 30/30 complete, 6.3 collisions).
        #
        # What it does change is animation smoothness and how far a carton is flung when
        # a robot ploughs into it (max 3.98 m at 30, 1.17 m at 1 under random actions) -
        # and speed, by about 10x end to end.
        #
        # Default is 5, chosen on 2026-08-30: 5x the raw throughput of 30 with no
        # behavioural difference. Not 1, because a robot then jumps a full metre in one
        # go and could tunnel past the 6 cm shelf posts - no evidence it does, but a
        # thin margin for a further 2.6x, and thinner still once the velocity-controlled
        # motion model replaces the teleport.
        #
        # None means "pick for the mode": GUI keeps 30 so `play_multi.py` animates
        # smoothly. Behaviour is identical either way, so the demo and training runs
        # disagreeing on this costs nothing.
        if substeps is None:
            substeps = 30 if render_mode == "human" else 5
        self.substeps = int(substeps)
        if self.substeps < 1:
            raise ValueError(f"substeps must be >= 1, got {substeps}")
        self.idle_penalises_turning = idle_penalises_turning
        self.lidar_noise = lidar_noise
        self._lidar_cache = None
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
        self.action_space = spaces.MultiDiscrete([7] * self.num_agents)

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
        self.max_steps = 2000
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
        # play_multi.py reads it that way, so its meaning must not change. The
        # observation needs something different: a stable slot per carton that survives
        # pickup and delivery, so carton 7 is always index 7 for the whole episode.
        self.all_resource_ids = []          # fixed order, length NUM_CARTONS
        self.resource_slot = {}             # pybullet body id -> observation index
        self.delivered = [False] * NUM_CARTONS
        self._prev_xy = [(0.0, 0.0)] * self.num_agents
        self._velocity = [(0.0, 0.0)] * self.num_agents

        # Roadmap step 7 writes here; until then these stay zero and the message slots
        # of every observation are zero with them.
        self.messages = np.zeros((self.num_agents, MSG_TOKENS), dtype=np.float32)

        # Reward bookkeeping (spec S3), all per-episode.
        # `_colliding_pairs` holds the pairs already in contact, so a collision is
        # charged once per *event* rather than once per step for as long as two
        # robots stay overlapped - the spec says "per collision event".
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

        # Baseline for the displacement-based velocity; the first observation of an
        # episode therefore reports zero velocity, which is true.
        self._prev_xy = self._current_xy()
        self._velocity = [(0.0, 0.0)] * self.num_agents
        self._lidar_cache = None

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

    def step(self, actions):
        self.current_step += 1
        num_substeps = self.substeps
        self._lidar_cache = None

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
            
            # Snap position to exact grid cell center to prevent drift.
            #
            # z is snapped too, which it was not until 2026-08-29. The chassis settles
            # ~0.17 mm per step under gravity and nothing ever put it back, so a robot
            # sank 0.051 m over 300 steps and would have dropped ~0.34 m across a full
            # 2000-step episode - through the floor, with the wheels buried long before
            # that. It went unnoticed while nothing depended on height; the LiDAR beam
            # depends on it, which is how it surfaced.
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
                if self._in_bounds(nxt[0], nxt[1]):
                    target_state['pos'] = nxt
                    target_state['wheel_delta'] = 0.119
                else:
                    # Driving off the grid is an invalid action (spec S3.2). The move is
                    # refused rather than executed - nothing in the world model gives
                    # meaning to a robot outside the arena, and letting one teleport past
                    # the boundary wall would corrupt every downstream metric.
                    invalid_action[i] = True
            elif action == 1:  # Backward
                nxt = (pos[0] - self.cell_size * math.cos(yaw), pos[1] - self.cell_size * math.sin(yaw), pos[2])
                if self._in_bounds(nxt[0], nxt[1]):
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

    def _in_bounds(self, x, y):
        """Is this world position inside the 13x13 grid?"""
        r, c = self._world_to_grid(x, y)
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    def _detect_collisions(self):
        """
        Contacts that count as collisions, as a set of hashable keys.

        Spec S3.1 charges -5.0 for a "collision (any pair)", "per collision event".

        WHAT COUNTS AS A PAIR
        Robot-robot contacts, plus robot-vs-shelf and robot-vs-wall. The obstacle half
        was added on 2026-08-29 with the shelf geometry fix: until then the bottom shelf
        plate sat at 0.30 and the chassis topped out at 0.194, so robots drove straight
        under the shelving and there was nothing to charge. Lowering the plate to 0.18
        made shelves solid, and the penalty is what teaches robots to route around them
        - "let the collision penalty do the work" rather than blocking the move outright,
        so a wedged robot has to learn its way out instead of the env silently refusing.

        WHY "EVENT" MEANS ONSET
        Two readings are possible: charge every step a pair overlaps, or charge once when
        the contact begins. The second is used - robots here are teleported rather than
        driven, so an overlapping pair stays overlapped until one of them moves away, and
        per-step charging would bill -5.0 repeatedly for a single mistake.
        `_colliding_pairs` carries the previous step's contacts so only new ones are
        billed. A robot that parks itself inside a shelf pays once, not forever; the time
        penalty is what stops it sitting there.
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

        rewards = SHARED_WEIGHT * shared + INDIVIDUAL_WEIGHT * individual

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

            # Message slots. self.messages is all zeros until roadmap step 7, so this
            # writes zeros over zeros - the wiring is here so step 7 changes behaviour
            # without changing the dimension.
            row[OBS_SLICES["messages"]] = np.concatenate(
                [self.messages[j] for j in others]
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
            # Delivered count is what termination and the evaluation harness both key
            # off, so it is reported explicitly rather than inferred.
            "delivered": sum(self.delivered),
            "obs_dim": self.obs_dim,
            "is_carrying": list(self.is_carrying),
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
