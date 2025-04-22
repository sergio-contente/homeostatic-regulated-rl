import numpy as np
from ...gymnasium_env.envs.grid_world_2_resources import GridWorldEnv2Resources
from ...gymnasium_env.envs.clementine import ClementineEnvironment
from ...agents.q_learning import QLearning
from ...gymnasium_env.envs.wrappers.digitize_continuos import DiscretizeWrapper
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def evaluate_agent(agent, env, rewards, internal_state_dim=2, success_threshold=0.0):
    """
    Visual and numerical evaluation of the Q-Learning agent.
    
    Parameters:
        - agent: instance of QLearning
        - env: Gymnasium environment (discretized)
        - rewards: list of rewards per episode
        - internal_state_dim: number of internal state dimensions (default=2)
        - success_threshold: minimum reward to consider the episode successful
    """
    
    rewards_series = pd.Series(rewards)
    rolling_avg = rewards_series.rolling(window=10).mean()
    
    # RMSE from the max reward value obtained
    max_reward = np.max(rewards)
    mse = np.square(np.array(rewards) - max_reward).mean(axis=0)
    mse_series = pd.Series([np.square(rewards[:i+1] - max_reward).mean() 
                            for i in range(len(rewards))])
    mse_rolling = mse_series.rolling(window=10).mean()
    
    # Plotar recompensas
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    ax1.plot(rewards_series, alpha=0.3, label="Raw reward")
    ax1.plot(rolling_avg, color="blue", linewidth=2, label="Moving average (10)")
    ax1.set_ylabel("Total reward")
    ax1.set_title("Q-Learning Agent Performance")
    ax1.legend()
    ax1.grid(True)
    
    # Plotar MSE
    ax2.plot(mse_series, alpha=0.3, color="red", label="MSE")
    ax2.plot(mse_rolling, color="darkred", linewidth=2, label="MSE Moving average (10)")
    ax2.set_xlabel("Episodes")
    ax2.set_ylabel("Mean Squared Error")
    ax2.set_title(f"Error relative to max reward ({max_reward:.2f})")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"📊 Final MSE: {mse:.4f}")
    
    successes = np.sum(np.array(rewards) >= success_threshold)
    success_rate = successes / len(rewards)
    print(f"✅ Success rate (reward >= {success_threshold}): {success_rate:.2%}")
    
    action_counts = Counter()
    state, _ = env.reset()
    state = agent._process_state(state)
    done = False
    while not done:
        action = agent.get_action(state)
        action_counts[action] += 1
        next_state, _, done, _, _ = env.step(action)
        state = agent._process_state(next_state)
    
    print("📊 Actions taken in one evaluation episode:")
    for a in range(agent.action_size):
        print(f" - Action {a}: {action_counts[a]} times")

    fig, axes = plt.subplots(1, agent.action_size, figsize=(5 * agent.action_size, 5))
    for a in range(agent.action_size):
        try:
            q_vals_action = agent.q_table[:, a].reshape([agent.n_bins] * internal_state_dim)
            ax = axes[a] if agent.action_size > 1 else axes
            im = ax.imshow(q_vals_action, cmap="viridis", origin="lower")
            ax.set_title(f"Action {a}")
            plt.colorbar(im, ax=ax)
        except:
            print(f"⚠️ Could not plot Q-table for action {a} (check shape).")
    plt.suptitle("Q-table per action")
    plt.tight_layout()
    plt.show()

    try:
        policy = np.argmax(agent.q_table, axis=1)
        policy_grid = policy.reshape([agent.n_bins] * internal_state_dim)

        plt.figure(figsize=(6, 5))
        plt.imshow(policy_grid, cmap="Accent", origin="lower")
        plt.title("Optimal policy (action per state)")
        plt.colorbar(label="Action")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(False)
        plt.show()
    except:
        print("⚠️ Could not visualize the policy (state not 2D?)")
    
def collect_trajectory(agent, env, max_steps=500):
    obs, _ = env.reset()
    trajectory = [tuple(obs)]
    done = False
    steps = 0

    while not done and steps < max_steps:
        state_idx = agent.get_state_index(obs)
        action = np.argmax(agent.q_table[state_idx])
        obs, _, done, _, _ = env.step(action)
        trajectory.append(tuple(obs))
        steps += 1

    return np.array(trajectory)


# === Parameters for discretization ===
n_bins = 50
low, high = 0.0, 300.0

# === Create base environment ===
config_path = "config/config.yaml"
drive_type = "base_drive"  # "base_drive", "elliptic_drive", "interoceptive_drive"
#env = ClementineEnvironment(config_path=config_path, drive_type=drive_type, render_mode=None)
env = ClementineEnvironment(config_path=config_path, drive_type=drive_type, render_mode=None)

# === Calculate number of states and actions ===
internal_state_dim = env.observation_space.shape[0]
state_size = n_bins ** internal_state_dim
action_size = env.action_space.n

# === Initialize the Q-learning agent ===
agent = QLearning(
    env=env,
    state_size=state_size,
    action_size=action_size,
    n_bins=n_bins,
    learning_rate=0.1,
    discount_factor=0.99,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995
)

# === Train the agent ===
num_episodes = 500
rewards = agent.train(env, num_episodes=num_episodes)

# Continua com a avaliação
env_eval = ClementineEnvironment(config_path=config_path, drive_type=drive_type, render_mode='human')
#evaluate_agent(agent, env_eval, rewards, internal_state_dim=2)
#agent.evaluate(env_eval, num_episodes=1, render=True)
