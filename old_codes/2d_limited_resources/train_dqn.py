import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import src.envs

config_path = "config/config.yaml"
drive_type = "base_drive"
current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = os.path.join('runs', f'DQN_LimitedResources2D_{drive_type}_{current_time}')
writer = SummaryWriter(log_dir)

env = gym.make("LimitedResources2D-v0", config_path=config_path, drive_type=drive_type)

def get_unwrapped_env(env):
    if hasattr(env, 'env'):
        return get_unwrapped_env(env.env)
    return env

base_env = get_unwrapped_env(env)

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

n_actions = env.action_space.n

def process_observation(obs):
    position = obs['position'].astype(np.float32)
    internal_states = obs['internal_states'].astype(np.float32)
    return np.concatenate((position, internal_states)), position, internal_states

state, info = env.reset()
processed_state, position, internal_states = process_observation(state)
n_observations = len(processed_state)

try:
    state_names = base_env.drive.get_internal_states_names()
except:
    state_names = [f"state_{i}" for i in range(len(internal_states))]

writer.add_text('Environment/Action Space', f'{n_actions}')
writer.add_text('Environment/Observation Space', f'{n_observations}')
writer.add_text('Environment/Internal States', f'{len(internal_states)}')

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)
writer.add_graph(policy_net, torch.zeros((1, n_observations), device=device))

steps_done = 0
total_steps = 0

def select_action(state_tensor):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    writer.add_scalar('Training/Epsilon', eps_threshold, steps_done)
    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_net(state_tensor).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    writer.add_scalar('Training/Loss', loss.item(), total_steps)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    return loss.item()

num_episodes = 300
episode_rewards = []
episode_durations = []

for i_episode in range(num_episodes):
    state, _ = env.reset()
    processed_state, position, internal_states = process_observation(state)
    state_tensor = torch.tensor(processed_state, dtype=torch.float32, device=device).unsqueeze(0)
    episode_reward = 0
    episode_loss = 0

    for t in count():
        total_steps += 1
        action = select_action(state_tensor)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        processed_obs, new_position, new_internal_states = process_observation(observation)

        # Log internal states every step
        for i, value in enumerate(new_internal_states):
            state_name = state_names[i] if i < len(state_names) else f"state_{i}"
            writer.add_scalar(f'States/{state_name}', value, total_steps)

        # Log drive value every step
        try:
            drive_value = base_env.drive.get_current_drive()
            writer.add_scalar('States/drive', drive_value, total_steps)
        except AttributeError:
            pass

        next_state_tensor = None if (terminated or truncated) else torch.tensor(processed_obs, dtype=torch.float32, device=device).unsqueeze(0)
        reward_tensor = torch.tensor([reward], device=device)

        memory.push(state_tensor, action, next_state_tensor, reward_tensor)
        state_tensor = next_state_tensor

        episode_reward += reward
        writer.add_scalar('Training/StepReward', reward, total_steps)

        loss = optimize_model()
        if loss is not None:
            episode_loss += loss

        for key in policy_net.state_dict():
            target_net.state_dict()[key].copy_(TAU * policy_net.state_dict()[key] + (1 - TAU) * target_net.state_dict()[key])

        if terminated or truncated:
            episode_durations.append(t + 1)
            episode_rewards.append(episode_reward)
            writer.add_scalar('Training/EpisodeReward', episode_reward, i_episode)
            writer.add_scalar('Training/EpisodeDuration', t + 1, i_episode)
            if t > 0:
                writer.add_scalar('Training/AverageEpisodeLoss', episode_loss / t, i_episode)
            break

    if i_episode % 10 == 0:
        writer.flush()
    print(f"Episode {i_episode + 1}/{num_episodes} | Reward: {episode_reward:.2f}")

writer.close()
model_path = os.path.join(log_dir, 'dqn_model.pth')
torch.save({
    'policy_net': policy_net.state_dict(),
    'target_net': target_net.state_dict(),
    'optimizer': optimizer.state_dict(),
}, model_path)
print(f"Treinamento concluído. Modelo salvo em: {model_path}")
