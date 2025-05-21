import os
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import src.gymnasium_env

# === CONFIG ===
config_path = "config/config.yaml"
drive_type = "base_drive"
model_path = "runs/SB3_DQN_GridWorld_base_drive_20250521-163256/dqn_model_final"
# Replace with actual folder

# === LOAD ENV ===
raw_env = gym.make("GridWorld-v0", config_path=config_path, drive_type=drive_type, render_mode="human")
env = FlattenObservation(raw_env)

# === LOAD MODEL ===
model = DQN.load(model_path, env=env)

# === EVALUATION ===
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True)
print(f"Mean reward: {mean_reward:.2f} \u00b1 {std_reward:.2f}")

# === RUN ONE EPISODE WITH RENDERING ===
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    done = done or truncated

env.close()
