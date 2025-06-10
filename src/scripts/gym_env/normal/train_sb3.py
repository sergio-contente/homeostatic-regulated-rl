import os
from datetime import datetime
import numpy as np

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from torch.utils.tensorboard import SummaryWriter

import src.gymnasium_env  # Ensure custom environment is registered

# === CONFIG ===
config_path = "config/config.yaml"
drive_type = "base_drive"  # or "elliptic_drive", "interoceptive_drive"
n_agents = 5  # Number of agents in the environment

current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = os.path.join('runs', f'SB3_DQN_MultiAgentHomeostatic_{drive_type}_{current_time}')
os.makedirs(log_dir, exist_ok=True)

# === TENSORBOARD LOGGER ===
writer = SummaryWriter(log_dir)
logger = configure(log_dir, ["stdout", "tensorboard"])

# === ENVIRONMENT SETUP ===
raw_env = gym.make("MultiAgentHomeostatic-v0", config_path=config_path, drive_type=drive_type, n_agents=n_agents)
base_env = raw_env.unwrapped
env = FlattenObservation(Monitor(raw_env))

# === GET STATE NAMES ===
try:
    state_names = base_env.drives[0].get_internal_states_names()
except Exception:
    state_dim = base_env.drives[0].get_internal_state_dimension()
    state_names = [f"state_{i}" for i in range(state_dim)]

# === CUSTOM CALLBACK TO LOG DRIVES, STATES AND SOCIAL METRICS ===
class MultiAgentLoggingCallback(BaseCallback):
    def __init__(self, base_env, state_names, writer, verbose=0):
        super().__init__(verbose)
        self.base_env = base_env
        self.state_names = state_names
        self.writer = writer

    def _on_step(self) -> bool:
        try:
            # Log resource stock
            self.writer.add_scalar('Environment/resource_stock', 
                                 self.base_env.resource_stock, 
                                 self.num_timesteps)

            # Log average consumption
            if len(self.base_env.intake_history) > 0:
                total_intake = np.sum(self.base_env.intake_history[-1])
                self.writer.add_scalar('Environment/total_intake', 
                                     total_intake, 
                                     self.num_timesteps)

            # Log per-agent metrics
            for i, agent in enumerate(self.base_env.agents_info):
                # Log drive value
                drive_value = self.base_env.drives[i].get_current_drive()
                self.writer.add_scalar(f'Agent{i}/drive', 
                                     drive_value, 
                                     self.num_timesteps)

                # Log internal states
                for j, (state_name, state_value) in enumerate(zip(self.state_names, agent["internal_states"])):
                    self.writer.add_scalar(f'Agent{i}/States/{state_name}', 
                                         state_value, 
                                         self.num_timesteps)
                
                # Log social metrics
                self.writer.add_scalar(f'Agent{i}/Social/beta', 
                                     self.base_env.beta[i], 
                                     self.num_timesteps)
                self.writer.add_scalar(f'Agent{i}/Social/alpha', 
                                     self.base_env.alpha[i], 
                                     self.num_timesteps)
                self.writer.add_scalar(f'Agent{i}/Social/belief_avg', 
                                     np.mean(self.base_env.belief_avg_consumption[i]), 
                                     self.num_timesteps)

        except Exception as e:
            print(f"Logging error: {e}")
        return True

# === MODEL AND CALLBACKS ===
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

checkpoint_cb = CheckpointCallback(save_freq=5000, save_path=log_dir, name_prefix="sb3_dqn")
multiagent_cb = MultiAgentLoggingCallback(base_env=base_env, state_names=state_names, writer=writer)

# === TRAINING ===
model.learn(total_timesteps=150_000, callback=[checkpoint_cb, multiagent_cb])
model.save(os.path.join(log_dir, "dqn_model_final"))
writer.close()
print(f"Model saved in {log_dir}/dqn_model_final") 
