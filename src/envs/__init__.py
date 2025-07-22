from gymnasium.envs.registration import register

register(
    id="NormarlHomeostatic-v0",
    entry_point="src.gymnasium_env.envs:NormarlHomeostaticEnv",
    max_episode_steps=3000
)
