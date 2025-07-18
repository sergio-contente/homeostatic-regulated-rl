import os
from datetime import datetime

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from torch.utils.tensorboard import SummaryWriter

import src.gymnasium_env  # registers custom env

# === CONFIGURATION ===
config_path = "config/config.yaml"
drive_type = "base_drive"
current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = os.path.join('runs', f'SB3_DQN_LimitedResources2D_{drive_type}_{current_time}')
os.makedirs(log_dir, exist_ok=True)

# === TENSORBOARD LOGGER ===
writer = SummaryWriter(log_dir)
logger = configure(log_dir, ["stdout", "tensorboard"])

# === ENVIRONMENT SETUP ===
raw_env = gym.make("LimitedResources2D-v0", config_path=config_path, drive_type=drive_type)
base_env = raw_env.unwrapped
env = FlattenObservation(Monitor(raw_env))

# === GET STATE NAMES ===
try:
    state_names = base_env.drive.get_internal_states_names()
except Exception:
    dim = base_env.drive.get_internal_state_dimension()
    state_names = [f"state_{i}" for i in range(dim)]

# === CUSTOM CALLBACK TO LOG TO TENSORBOARD ===
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
            resources_map = self.base_env.agent_info["resources_map"]

            self.writer.add_scalar("States/drive", drive, self.num_timesteps)

            # Handle both scalar and 2D positions
            if hasattr(position, '__len__') and len(position) == 2:
                self.writer.add_scalar("States/position_x", float(position[0]), self.num_timesteps)
                self.writer.add_scalar("States/position_y", float(position[1]), self.num_timesteps)
            else:
                self.writer.add_scalar("States/position", float(position), self.num_timesteps)

            for i, value in enumerate(internal_states):
                name = self.state_names[i] if i < len(self.state_names) else f"state_{i}"
                self.writer.add_scalar(f"States/{name}", value, self.num_timesteps)
            
            # Log resources_map (availability of each resource type)
            for i, availability in enumerate(resources_map):
                # Assuming state_names correspond to resource names for simplicity in logging
                res_name = self.state_names[i] if i < len(self.state_names) else f"resource_type_{i}"
                self.writer.add_scalar(f"Resources/{res_name}_available", float(availability), self.num_timesteps)

        except Exception as e:
            print(f"⚠️ Logging error: {e}")
        return True

# === MODEL SETUP ===
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=1e-4,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=128,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    exploration_initial_eps=0.9,
    exploration_final_eps=0.05,
)

model.set_logger(logger)

# === CALLBACKS ===
checkpoint_cb = CheckpointCallback(
    save_freq=5000,
    save_path=log_dir,
    name_prefix="sb3_dqn"
)
drive_cb = DriveLoggingCallback(base_env=base_env, state_names=state_names, writer=writer)

# === TRAINING ===
model.learn(total_timesteps=500_000, callback=[checkpoint_cb, drive_cb])
model.save(os.path.join(log_dir, "dqn_model_final"))
writer.close()

print(f"✅ Model saved in {log_dir}/dqn_model_final")

# === EVALUATION & RENDERING ===
print("Evaluating trained agent with rendering...")

# Re-create the environment with render_mode="human"
eval_env_raw = gym.make("LimitedResources2D-v0", config_path=config_path, drive_type=drive_type, render_mode="human")
# Important: Wrap with FlattenObservation if the policy expects flattened obs
eval_env = FlattenObservation(Monitor(eval_env_raw)) 

obs, _ = eval_env.reset()
for i in range(1000): # Evaluate for 1000 steps
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    eval_env.render() # Render the environment
    if terminated or truncated:
        print(f"Episode finished after {i+1} steps.")
        obs, _ = eval_env.reset()

eval_env.close()
print("Evaluation finished.")
