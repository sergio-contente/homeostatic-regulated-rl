import os
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd

import src.gymnasium_env  # Import your custom environment

# Path to the saved model
model_path = "runs/DQN_GridWorld_ellipitc_drive_20250517-230337/dqn_model.pth"

# Environment settings
config_path = "config/test_config.yaml"
drive_type = "elliptic_drive"  # Change to "drive" if using the drive class

# Create environment with human rendering
env = gym.make("GridWorld-v0", config_path=config_path, drive_type=drive_type, render_mode="human", size=5)

# Get the base environment
def get_unwrapped_env(env):
    if hasattr(env, 'env'):
        return get_unwrapped_env(env.env)
    return env

base_env = get_unwrapped_env(env)

# Check available device
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using device: {device}")

# Function to process observation
def process_observation(observation):
    position = float(observation['position'])
    internal_states = observation['internal_states'].astype(np.float32)
    flat_observation = np.concatenate(([position], internal_states))
    return flat_observation, position, internal_states

# Define the same network architecture used in training
class DQN(torch.nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = torch.nn.Linear(n_observations, 128)
        self.layer2 = torch.nn.Linear(128, 128)
        self.layer3 = torch.nn.Linear(128, n_actions)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        return self.layer3(x)

# Get dimensions of observation and action spaces
state, info = env.reset()
processed_state, pos, internal = process_observation(state)
n_observations = len(processed_state)
n_actions = env.action_space.n

# Create the network and load saved weights
policy_net = DQN(n_observations, n_actions).to(device)
checkpoint = torch.load(model_path, map_location=device)
policy_net.load_state_dict(checkpoint['policy_net'])
policy_net.eval()  # Evaluation mode

print("Model loaded successfully!")

# Function to select action (no exploration during testing)
def select_action(state):
    with torch.no_grad():
        return policy_net(state).max(1).indices.view(1, 1)

# Create folder for results
output_dir = "test_results"
os.makedirs(output_dir, exist_ok=True)

# Function to map numerical action to descriptive names
def action_to_name(action):
    # Adjust for your specific case
    if hasattr(base_env, 'action_meanings'):
        return base_env.action_meanings[action]
    
    # Generic example - replace with correct values for your environment
    action_names = ["Left", "Right", "Stay"]  
    if action < len(action_names):
        return action_names[action]
    return f"Action {action}"

# Run test episodes
num_test_episodes = 3
total_rewards = []

for i_episode in range(num_test_episodes):
    state, info = env.reset()
    processed_state, position, internal_states = process_observation(state)
    state_tensor = torch.tensor(processed_state, dtype=torch.float32, device=device).unsqueeze(0)
    
    episode_reward = 0
    done = False
    step = 0
    
    # Try to get internal state names
    try:
        state_names = base_env.drive.get_internal_states_names()
    except AttributeError:
        state_names = [f"state_{i}" for i in range(len(internal_states))]
    
    # List to store history for plotting
    drive_history = []
    internal_states_history = {name: [] for name in state_names}
    positions = []
    actions_history = []
    rewards_history = []
    
    # Add initial values
    try:
        current_drive = base_env.drive.get_current_drive()
        drive_history.append(current_drive)
    except AttributeError:
        current_drive = None
        
    positions.append(position)
    for i, state_name in enumerate(state_names):
        if i < len(internal_states):
            internal_states_history[state_name].append(internal_states[i])
    
    while not done and step < 500:  # Limit of 500 steps
        # Select and execute action
        action = select_action(state_tensor)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        
        # Process new observation
        processed_obs, new_position, new_internal_states = process_observation(observation)
        state_tensor = torch.tensor(processed_obs, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Store action and reward
        actions_history.append(action.item())
        rewards_history.append(reward)
        
        # Accumulate reward
        episode_reward += reward
        
        # Store position for plotting
        positions.append(new_position)
        
        # Store drive and internal states information for plotting
        try:
            new_drive = base_env.drive.get_current_drive()
            drive_history.append(new_drive)
            current_drive = new_drive
        except AttributeError:
            pass
        
        # Store internal states
        for i, state_name in enumerate(state_names):
            if i < len(new_internal_states):
                internal_states_history[state_name].append(new_internal_states[i])
        
        # Check if finished
        done = terminated or truncated
        step += 1
        
        # Show information every 50 steps
        if step % 50 == 0:
            if current_drive is not None:
                print(f"Episode {i_episode+1}, Step {step}, Drive: {current_drive:.4f}")
            else:
                print(f"Episode {i_episode+1}, Step {step}")
    
    total_rewards.append(episode_reward)
    print(f"Episode {i_episode+1}/{num_test_episodes} completed. Reward: {episode_reward:.2f}, Steps: {step}")
    
    # Create a detailed visualization of the trajectory
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Position graph
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(positions, label='Position')
    ax1.set_title('Agent Trajectory')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Position')
    ax1.grid(True)
    
    # Action markers
    action_colors = ['r', 'g', 'b', 'y', 'c', 'm']  # Colors for different actions
    for action_type in set(actions_history):
        action_steps = [i for i, a in enumerate(actions_history) if a == action_type]
        if action_steps:
            action_positions = [positions[i+1] for i in action_steps]  # +1 because action i leads to position i+1
            ax1.scatter(
                [step+1 for step in action_steps],  # +1, since we plot actions at resulting points
                action_positions,
                color=action_colors[action_type % len(action_colors)],
                alpha=0.5,
                s=30,
                label=f'Action: {action_to_name(action_type)}'
            )
    ax1.legend()
    
    # 2. Drive graph
    ax2 = fig.add_subplot(2, 2, 2)
    if drive_history:
        ax2.plot(drive_history, label='Drive', color='green')
        ax2.set_title('Drive Value')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Drive')
        ax2.grid(True)
        
        # Add cumulative rewards line
        cum_rewards = np.cumsum(rewards_history)
        ax2_reward = ax2.twinx()
        ax2_reward.plot(range(1, len(cum_rewards)+1), cum_rewards, 'r--', label='Cumulative Reward')
        ax2_reward.set_ylabel('Cumulative Reward', color='r')
        ax2_reward.tick_params(axis='y', labelcolor='r')
        
        # Add two legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_reward.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 3. Internal states graph
    ax3 = fig.add_subplot(2, 2, 3)
    for state_name, values in internal_states_history.items():
        if values:  # Check if there are recorded values
            ax3.plot(values, label=state_name)
    ax3.set_title('Internal States')
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Value')
    ax3.grid(True)
    ax3.legend()
    
    # 4. Rewards per step graph
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.bar(range(len(rewards_history)), rewards_history, color='orange', width=1.0)
    ax4.set_title('Rewards per Step')
    ax4.set_xlabel('Steps')
    ax4.set_ylabel('Reward')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"analysis_episode_{i_episode+1}.png"), dpi=150)
    print(f"Analysis saved in: {os.path.join(output_dir, f'analysis_episode_{i_episode+1}.png')}")
    plt.close()
    
    # Save episode data to a CSV file
    data = {
        'step': list(range(len(positions)-1)),  # -1 because we have the initial position
        'position': positions[1:],  # Ignore initial position
        'action': actions_history,
        'reward': rewards_history
    }
    
    # Add drive and internal states if available
    if drive_history and len(drive_history) > 1:  # Check if there's more than one value (beyond initial)
        data['drive'] = drive_history[1:]  # Ignore initial drive
    
    for state_name, values in internal_states_history.items():
        if values and len(values) > 1:  # Check if there's more than one value
            data[state_name] = values[1:]  # Ignore initial state
    
    # Create and save DataFrame
    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, f"data_episode_{i_episode+1}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Data saved in: {csv_path}")

print(f"Test completed! Average reward: {sum(total_rewards)/len(total_rewards):.2f}")

# Close the environment
env.close()
