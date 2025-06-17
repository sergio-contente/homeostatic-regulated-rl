import os
from datetime import datetime
import numpy as np

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

import src.gymnasium_env  # Ensure custom environment is registered

# === CONFIG ===
config_path = "config/config.yaml"
drive_type = "base_drive"  # or "elliptic_drive", "interoceptive_drive"
n_agents = 1  # Single agent
learning_rate = 1e-4  # Higher learning rate for simpler single-agent scenario

current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = os.path.join('runs', f'SB3_DQN_SingleAgentHomeostatic_{drive_type}_{current_time}')
os.makedirs(log_dir, exist_ok=True)

# === TENSORBOARD LOGGER ===
logger = configure(log_dir, ["stdout", "tensorboard"])

# === ENVIRONMENT SETUP ===
raw_env = gym.make("MultiAgentHomeostatic-v0", 
                  config_path=config_path, 
                  drive_type=drive_type, 
                  n_agents=n_agents, 
                  learning_rate=learning_rate,
                  unlimited_resources=True,  # Enable unlimited resources
                  disable_social_norms=True)  # Disable social norms
base_env = raw_env.unwrapped
env = FlattenObservation(Monitor(raw_env))

# === GET STATE NAMES ===
try:
    state_names = base_env.drives[0].get_internal_states_names()
except Exception:
    state_dim = base_env.drives[0].get_internal_state_dimension()
    state_names = [f"state_{i}" for i in range(state_dim)]

# === CUSTOM CALLBACK TO LOG DRIVES, STATES AND SOCIAL METRICS ===
class SingleAgentLoggingCallback(BaseCallback):
    def __init__(self, base_env, state_names, verbose=0):
        super().__init__(verbose)
        self.base_env = base_env
        self.state_names = state_names
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        try:
            agent = self.base_env.agents_info[0]
            drive_value = self.base_env.drives[0].get_current_drive()
            self.logger.record('Agent/drive', drive_value)
            for j, (state_name, state_value) in enumerate(zip(self.state_names, agent["internal_states"])):
                self.logger.record(f'Agent/States/{state_name}', state_value)
            self.logger.record('Environment/resource_stock', self.base_env.resource_stock)
            if len(self.base_env.intake_history) > 0:
                total_intake = np.sum(self.base_env.intake_history[-1])
                self.logger.record('Environment/total_intake', total_intake)
            self.current_episode_reward += self.locals['rewards'][0]
            self.current_episode_length += 1
            if self.locals['dones'][0]:
                self.episode_count += 1
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lengths.append(self.current_episode_length)
                self.logger.record('Episode/reward', self.current_episode_reward)
                self.logger.record('Episode/length', self.current_episode_length)
                self.current_episode_reward = 0
                self.current_episode_length = 0
                if self.episode_count % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    avg_length = np.mean(self.episode_lengths[-10:])
                    self.logger.record('Episode/avg_reward', avg_reward)
                    self.logger.record('Episode/avg_length', avg_length)
        except Exception as e:
            print(f"Logging error: {e}")
        return True

# === MODEL AND CALLBACKS ===
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=learning_rate,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=128,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    exploration_initial_eps=0.9,
    exploration_final_eps=0.05,
    policy_kwargs=dict(net_arch=[256, 256]),
)
model.set_logger(logger)

checkpoint_cb = CheckpointCallback(save_freq=5000, save_path=log_dir, name_prefix="sb3_dqn")
singleagent_cb = SingleAgentLoggingCallback(base_env=base_env, state_names=state_names)

# === TRAINING ===
print(f"\nStarting training. Logs will be saved to {log_dir}")
print("You can monitor the training progress using TensorBoard:")
print(f"tensorboard --logdir {log_dir}\n")

model.learn(total_timesteps=150_000, callback=[checkpoint_cb, singleagent_cb])
model.save(os.path.join(log_dir, "dqn_model_final"))
print(f"\nTraining completed. Model saved in {log_dir}/dqn_model_final") 
