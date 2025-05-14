from gymnasium.envs.registration import register

register(
    id="GridWorld-v0",
    entry_point="src.gymnasium_env.envs:GridWorldEnv",
    max_episode_steps=3000
)

register(
    id="LimitedResources-v0",
    entry_point="src.gymnasium_env.envs:LimitedResourcesEnv",
    max_episode_steps=3000
)

# Available parameters for the `register` function in Gymnasium:

# reward_threshold : float (default: None)
#     The reward threshold at which the task is considered solved.

# nondeterministic : bool (default: False)
#     Indicates whether the environment is non-deterministic even after seeding.

# max_episode_steps : int (default: None)
#     The maximum number of steps an episode can last.
#     If not None, a TimeLimit wrapper is automatically added.

# order_enforce : bool (default: True)
#     If True, wraps the environment with an OrderEnforcing wrapper to ensure correct API usage order.

# kwargs : dict (default: {})
#     A dictionary of default keyword arguments to pass to the environment class.

