import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import DQN
import src.gymnasium_env  # Ensure custom environment is registered

# === CONFIG ===
config_path = "config/config.yaml"
drive_type = "base_drive"  # or "elliptic_drive", "interoceptive_drive"
n_agents = 1  # Single agent
# Update this path to your latest run directory
model_path = "runs/SB3_DQN_SingleAgentHomeostatic_base_drive_20250613-161906/dqn_model_final"  # Update this with your actual model path
output_dir = "sb3_test_results"
os.makedirs(output_dir, exist_ok=True)

# === ENVIRONMENT ===
raw_env = gym.make("MultiAgentHomeostatic-v0", 
                  config_path=config_path, 
                  drive_type=drive_type, 
                  n_agents=n_agents,
                  render_mode="human",  # Enable rendering
                  unlimited_resources=True,
                  disable_social_norms=True,
                  learning_rate=1e-4)
base_env = raw_env.unwrapped
env = FlattenObservation(raw_env)

# === GET STATE NAMES ===
try:
    state_names = base_env.drives[0].get_internal_states_names()
except Exception:
    state_dim = base_env.drives[0].get_internal_state_dimension()
    state_names = [f"state_{i}" for i in range(state_dim)]

# === LOAD MODEL ===
model = DQN.load(model_path, env=env)

# === RUN ONE EPISODE ===
obs, _ = env.reset()
done = False
step = 0
max_steps = 500
episode_rewards = np.zeros(n_agents)

positions = [[] for _ in range(n_agents)]
rewards = [[] for _ in range(n_agents)]
actions = [[] for _ in range(n_agents)]
drive_history = [[] for _ in range(n_agents)]
internal_states_history = [{name: [] for name in state_names} for _ in range(n_agents)]
resource_stock_history = []
total_intake_history = []
belief_avg_history = [[] for _ in range(n_agents)]

print("\nStarting evaluation episode...")
print("Press Ctrl+C to stop the visualization")

try:
    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store data for analysis
        positions[0].append(info["agents"][0]["position"])
        rewards[0].append(reward)
        actions[0].append(action)
        drive_history[0].append(base_env.drives[0].get_current_drive())
        
        for i, state_name in enumerate(state_names):
            internal_states_history[0][state_name].append(info["agents"][0]["internal_states"][i])
        
        resource_stock_history.append(info["resource_stock"])
        if len(base_env.intake_history) > 0:
            total_intake_history.append(np.sum(base_env.intake_history[-1]))
        
        step += 1
        if step % 10 == 0:
            print(f"Step {step}: Reward = {reward:.2f}, Drive = {drive_history[0][-1]:.2f}")
        
        # Add a small delay to make the visualization more visible
        import time
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nEvaluation stopped by user")

finally:
    env.close()
    print("\nEvaluation finished")

# === PLOT RESULTS ===
plt.figure(figsize=(15, 10))

# Plot 1: Internal States
plt.subplot(2, 2, 1)
for state_name in state_names:
    plt.plot(internal_states_history[0][state_name], label=state_name)
plt.title('Internal States Over Time')
plt.xlabel('Step')
plt.ylabel('State Value')
plt.legend()

# Plot 2: Drive Value
plt.subplot(2, 2, 2)
plt.plot(drive_history[0])
plt.title('Drive Value Over Time')
plt.xlabel('Step')
plt.ylabel('Drive')

# Plot 3: Rewards
plt.subplot(2, 2, 3)
plt.plot(rewards[0])
plt.title('Rewards Over Time')
plt.xlabel('Step')
plt.ylabel('Reward')

# Plot 4: Actions
plt.subplot(2, 2, 4)
plt.plot(actions[0])
plt.title('Actions Over Time')
plt.xlabel('Step')
plt.ylabel('Action')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'evaluation_results.png'))
plt.close()

print(f"\nResults saved to {output_dir}/evaluation_results.png")

# === SAVE CSV ===
data = {"step": list(range(step))}
data.update({f"position_{i}": positions[i], f"drive_{i}": drive_history[i], f"belief_avg_{i}": belief_avg_history[i]})
for i, state_name in enumerate(state_names):
    data[f"{state_name}_{i}"] = internal_states_history[i][state_name]
data["resource_stock"] = resource_stock_history
data["total_intake"] = total_intake_history

df = pd.DataFrame(data)
csv_path = os.path.join(output_dir, f"sb3_data_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv")
df.to_csv(csv_path, index=False)
print(f"📁 CSV saved: {csv_path}") 
