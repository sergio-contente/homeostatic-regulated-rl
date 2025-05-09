import numpy as np
from src.custom_env.homeoenv import HomeoEnv
from ...agents.q_learning import QLearning

def main():
    # === Parameters for discretization ===
    maxh = 5

    # === Create base environment ===
    config_path = "config/config.yaml"
    drive_type = "interoceptive_drive"  # "base_drive", "elliptic_drive", "interoceptive_drive"
    env = HomeoEnv(config_path=config_path, drive_type=drive_type, render_mode=None, maxh=maxh)

    # === Calculate number of states and actions ===
    size = env.drive.get_internal_state_size()
    state_size = (2 * maxh + 1) ** size

    action_size = len(env.action_space)

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
    num_episodes = 500
    rewards = agent.train(env, num_episodes=num_episodes)
    
    agent.save_q_table("models/custom/homeoenv")
    env.plot_rewards(rewards)

    env_eval = HomeoEnv(config_path=config_path, drive_type=drive_type, render_mode='human')
    agent.evaluate(env_eval, num_episodes=10, render=True)


if __name__ == "__main__":
    main()
