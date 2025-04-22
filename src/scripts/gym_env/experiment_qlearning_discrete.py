import numpy as np
from src.gymnasium_env.envs.discrete_homeoenv import DiscreteHomeoEnv
from src.gymnasium_env.envs.wrappers.digitize_continuos import DiscretizeWrapper

from ...agents.q_learning import QLearning

def main():
    # === Parameters for discretization ===
    maxh = 5

    # === Create base environment ===
    config_path = "config/config.yaml"
    drive_type = "base_drive"  # "base_drive", "elliptic_drive", "interoceptive_drive"
    env = DiscreteHomeoEnv(config_path=config_path, drive_type=drive_type, render_mode=None, maxh=maxh)

    # === Wrap with discretization ===
    #wrapped_env = DiscretizeWrapper(env, n_bins=n_bins, low=low, high=high)

    # === Calculate number of states and actions ===
    internal_state_dim = env.observation_space.shape[0]
    state_size = (2 * maxh + 1) ** internal_state_dim

    action_size = env.action_space.n

    # === Initialize the Q-learning agent ===
    agent = QLearning(
        env=env,
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )

    # === Train the agent ===
    num_episodes = 1000
    rewards = agent.train(env, num_episodes=num_episodes)
    
    agent.save_q_table("models/gym_env/discrete_homeoenv")

    # Continua com a avaliação
    env_eval = DiscreteHomeoEnv(config_path=config_path, drive_type=drive_type, render_mode='human')
    agent.evaluate(env_eval, num_episodes=10, render=True)

    env.plot_rewards(rewards)

if __name__ == "__main__":
    main()
