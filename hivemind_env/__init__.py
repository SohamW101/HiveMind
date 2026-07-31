from gymnasium.envs.registration import register

register(
    id="HiveMind-SingleAgent",
    entry_point="hivemind_env.env:HiveMindSingleAgentEnv",
    max_episode_steps=200,
)
