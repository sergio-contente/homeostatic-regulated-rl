import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pygame
import cv2
from datetime import datetime
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import DQN
import src.gymnasium_env  # your custom environment

# === CONFIG ===
config_path = "config/config.yaml"
drive_type = "base_drive"
model_path = "runs/SB3_DQN_GridWorld_base_drive_m2_n2/dqn_model_final"
output_dir = "sb3_test_results"
videos_dir = os.path.join(output_dir, "videos")
os.makedirs(videos_dir, exist_ok=True)

# === ENVIRONMENT ===
raw_env = gym.make("GridWorld-v0", config_path=config_path, drive_type=drive_type, render_mode="human")
env = FlattenObservation(raw_env)
base_env = raw_env.unwrapped

# === UTILS ===
def capture_pygame_screen():
    return pygame.surfarray.array3d(pygame.display.get_surface())

def get_state_names():
    try:
        return base_env.drive.get_internal_states_names()
    except:
        dim = len(base_env.agent_info["internal_states"])
        return [f"state_{i}" for i in range(dim)]

# === LOAD MODEL ===
model = DQN.load(model_path, env=env)

# === RUN ONE EPISODE ===
obs, _ = env.reset()
done = False
step = 0
episode_reward = 0
max_steps = 500

positions, rewards, actions, frames, drive_history = [], [], [], [], []
state_names = get_state_names()
internal_states_history = {name: [] for name in state_names}

positions.append(base_env.agent_info["position"])
frames.append(capture_pygame_screen())
drive_history.append(base_env.drive.get_current_drive())
for i, name in enumerate(state_names):
    internal_states_history[name].append(base_env.agent_info["internal_states"][i])

while not done and step < max_steps:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    actions.append(action)
    rewards.append(reward)
    episode_reward += reward
    step += 1

    positions.append(base_env.agent_info["position"])
    frames.append(capture_pygame_screen())
    drive_history.append(base_env.drive.get_current_drive())
    for i, name in enumerate(state_names):
        internal_states_history[name].append(base_env.agent_info["internal_states"][i])

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

# === SAVE VIDEO ===
video_path = os.path.join(videos_dir, f"sb3_episode_{timestamp}.mp4")
frames_bgr = [cv2.cvtColor(f.transpose(1, 0, 2), cv2.COLOR_RGB2BGR) for f in frames]
h, w, _ = frames_bgr[0].shape
writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
for f in frames_bgr:
    writer.write(f)
writer.release()
print(f"🎥 Video saved: {video_path}")

# === PLOT ===
fig = plt.figure(figsize=(15, 12))

ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(positions, label="Position")
ax1.set_title("Agent Trajectory")
ax1.set_xlabel("Steps")
ax1.set_ylabel("Position")
ax1.grid(True)

ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(drive_history, label="Drive", color="green")
ax2.set_title("Drive Value")
ax2.set_xlabel("Steps")
ax2.set_ylabel("Drive")
ax2.grid(True)
ax2r = ax2.twinx()
cum_rewards = np.cumsum(rewards)
ax2r.plot(range(1, len(cum_rewards) + 1), cum_rewards, 'r--', label="Cumulative Reward")
ax2r.set_ylabel("Cumulative Reward", color='r')
ax2r.tick_params(axis='y', labelcolor='r')
ax2.legend(loc="upper left")

ax3 = fig.add_subplot(2, 2, 3)
for name, vals in internal_states_history.items():
    ax3.plot(vals, label=name)
ax3.set_title("Internal States")
ax3.set_xlabel("Steps")
ax3.set_ylabel("Value")
ax3.grid(True)
ax3.legend()

ax4 = fig.add_subplot(2, 2, 4)
ax4.bar(range(len(rewards)), rewards, color="orange")
ax4.set_title("Reward per Step")
ax4.set_xlabel("Steps")
ax4.set_ylabel("Reward")
ax4.grid(True)

plt.tight_layout()
analysis_path = os.path.join(output_dir, f"sb3_analysis_{timestamp}.png")
plt.savefig(analysis_path)
plt.close()
print(f"📊 Analysis plot saved: {analysis_path}")

# === SAVE CSV ===
data = {
    "step": list(range(len(rewards))),
    "position": positions[1:],
    "action": actions,
    "reward": rewards,
    "drive": drive_history[1:]
}
for name, vals in internal_states_history.items():
    data[name] = vals[1:]
df = pd.DataFrame(data)
csv_path = os.path.join(output_dir, f"sb3_data_{timestamp}.csv")
df.to_csv(csv_path, index=False)
print(f"📁 CSV saved: {csv_path}")

print(f"✅ SB3 test episode completed. Total reward: {episode_reward:.2f}")
env.close()
