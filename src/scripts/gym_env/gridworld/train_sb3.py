import os
import math
from datetime import datetime

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from torch.utils.tensorboard import SummaryWriter

import src.gymnasium_env  # Ensure env is registered

# === CONFIG ===
config_path = "config/config.yaml"
drive_type = "base_drive"

current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = os.path.join('runs', f'SB3_DQN_GridWorld_{drive_type}_{current_time}')
os.makedirs(log_dir, exist_ok=True)

# === TENSORBOARD LOGGER ===
writer = SummaryWriter(log_dir)
logger = configure(log_dir, ["stdout", "tensorboard"])

# === ENVIRONMENT SETUP ===
raw_env = gym.make("GridWorld-v0", config_path=config_path, drive_type=drive_type)
base_env = raw_env.unwrapped
env = FlattenObservation(Monitor(raw_env))

# === GET STATE NAMES ===
try:
    state_names = base_env.drive.get_internal_states_names()
except:
    state_dim = base_env.drive.get_internal_state_dimension()
    state_names = [f"state_{i}" for i in range(state_dim)]

# === CUSTOM CALLBACK TO LOG DRIVE AND STATES ===
class DriveLoggingCallback(BaseCallback):
    def __init__(self, base_env, state_names, writer, verbose=0):
        super().__init__(verbose)
        self.base_env = base_env
        self.state_names = state_names
        self.writer = writer

    def _on_step(self) -> bool:
        try:
            drive = self.base_env.drive.get_current_drive()
            internal_states = self.base_env.agent_info["internal_states"]
            position = self.base_env.agent_info["position"]

            self.writer.add_scalar('States/drive', drive, self.num_timesteps)
            self.writer.add_scalar('States/position', float(position), self.num_timesteps)

            for i, value in enumerate(internal_states):
                name = self.state_names[i] if i < len(self.state_names) else f"state_{i}"
                self.writer.add_scalar(f'States/{name}', value, self.num_timesteps)
        except Exception as e:
            print(f"Logging error: {e}")
        return True

# === MODEL AND CALLBACKS ===
model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
model.set_logger(logger)

checkpoint_cb = CheckpointCallback(save_freq=5000, save_path=log_dir, name_prefix="sb3_dqn")
drive_cb = DriveLoggingCallback(base_env=base_env, state_names=state_names, writer=writer)

# === TRAINING ===
model.learn(total_timesteps=100_000, callback=[checkpoint_cb, drive_cb])
model.save(os.path.join(log_dir, "dqn_model_final"))
writer.close()
print(f"Model saved in {log_dir}/dqn_model_final")
