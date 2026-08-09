from gymnasium.envs.registration import register

from hivemind_env.env import (  # noqa: F401  (re-exported for convenience)
    DEFAULT_OBS_SIZE,
    LIDAR_MAX_RANGE,
    LIDAR_NUM_RAYS,
    OBS_SIZE_V1,
    OBS_SIZE_V2,
)

# max_episode_steps matches HiveMindSingleAgentEnv.max_steps. It used to be 200, which
# meant scripts using gym.make() got a 200-step TimeLimit while scripts constructing the
# class directly got 500 - the same policy looked worse through gym.make().
MAX_EPISODE_STEPS = 500

register(
    id="HiveMind-SingleAgent-v0",
    entry_point="hivemind_env.env:HiveMindSingleAgentEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
)

# No unversioned alias is registered: gymnasium raises
#   "Can't register the unversioned environment `HiveMind-SingleAgent` when the
#    versioned environment `HiveMind-SingleAgent-v0` of the same name already exists"
# and it is redundant anyway - gym.make("HiveMind-SingleAgent") resolves to the latest
# registered version on its own, so older call sites keep working.
