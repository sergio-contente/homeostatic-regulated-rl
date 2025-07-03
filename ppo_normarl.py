"""PPO implementation adapted for the NORMARL homeostatic environment using CleanRL principles.

This code is adapted from the original PPO implementation to work with the NORMARL environment
which uses dictionary observations and discrete actions for homeostatic agents.

Author: Adapted from Jet's CleanRL implementation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import sys
import os

# Add src to path to import NORMARL environment
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.pettingzoo_env.normarl import NormalHomeostaticEnv


class NORMARLAgent(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        
        # Extract dimensions from observation space
        self.position_size = observation_space["position"].n
        self.internal_states_dim = observation_space["internal_states"].shape[0]
        self.social_norm_dim = observation_space["perceived_social_norm"].shape[0]
        
        # Total observation dimension
        total_obs_dim = self.position_size + self.internal_states_dim + self.social_norm_dim
        
        # Network architecture for dictionary observations
        self.network = nn.Sequential(
            self._layer_init(nn.Linear(total_obs_dim, 256)),
            nn.ReLU(),
            self._layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            self._layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
        )
        
        # Actor and critic heads
        self.actor = self._layer_init(nn.Linear(128, action_space.n), std=0.01)
        self.critic = self._layer_init(nn.Linear(128, 1))

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def _process_observation(self, obs):
        """Convert dictionary observation to flat tensor."""
        # One-hot encode position
        position_onehot = torch.zeros(obs["position"].shape[0], self.position_size, device=obs["position"].device, dtype=torch.float32)
        position_onehot.scatter_(1, obs["position"].unsqueeze(1), 1)
        
        # Concatenate all observation components
        processed_obs = torch.cat([
            position_onehot,
            obs["internal_states"].float(),
            obs["perceived_social_norm"].float()
        ], dim=-1)
        
        return processed_obs

    def get_value(self, obs):
        processed_obs = self._process_observation(obs)
        return self.critic(self.network(processed_obs))

    def get_action_and_value(self, obs, action=None):
        processed_obs = self._process_observation(obs)
        hidden = self.network(processed_obs)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


def batchify_obs(obs, device):
    """Converts NORMARL style observations to batch of torch arrays."""
    # Convert dictionary observations to batched format
    batched_obs = {}
    for key in obs[list(obs.keys())[0]].keys():
        if key == "position":
            # Stack positions as integers
            batched_obs[key] = torch.tensor([obs[a][key] for a in obs], device=device, dtype=torch.long)
        else:
            # Stack arrays as float32 - ensure proper stacking
            values = [obs[a][key] for a in obs]
            # Convert to numpy arrays if they aren't already
            values = [np.array(v) for v in values]
            batched_obs[key] = torch.tensor(np.stack(values, axis=0), device=device, dtype=torch.float32)
    return batched_obs


def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    # convert to list of np arrays
    x = np.stack([x[a] for a in x], axis=0)
    # convert to torch as float32
    x = torch.tensor(x, dtype=torch.float32).to(device)
    return x


def unbatchify(x, env):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.agents)}
    return x


if __name__ == "__main__":
    """ALGO PARAMS"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ent_coef = 0.01
    vf_coef = 0.5
    clip_coef = 0.2
    gamma = 0.99
    batch_size = 64
    max_cycles = 1000
    total_episodes = 100
    
    # NORMARL specific parameters
    config_path = "config/config.yaml"
    drive_type = "base_drive"
    learning_rate = 0.1
    beta = 0.5
    number_resources = 1
    n_agents = 5
    size = 10

    """ ENV SETUP """
    env = NormalHomeostaticEnv(
        config_path=config_path,
        drive_type=drive_type,
        learning_rate=learning_rate,
        beta=beta,
        number_resources=number_resources,
        n_agents=n_agents,
        size=size
    )
    
    # Reset environment to initialize action and observation spaces
    observations, info = env.reset()
    
    num_agents = len(env.agents)
    num_actions = env.action_space(env.agents[0]).n
    observation_space = env.observation_space(env.agents[0])

    """ LEARNER SETUP """
    agent = NORMARLAgent(observation_space, env.action_space(env.agents[0])).to(device)
    # Ensure agent uses float32
    agent = agent.float()
    optimizer = optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5)

    """ ALGO LOGIC: EPISODE STORAGE"""
    end_step = 0
    total_episodic_return = 0
    
    # Storage for dictionary observations
    rb_obs_positions = torch.zeros((max_cycles, num_agents), dtype=torch.long).to(device)
    rb_obs_internal_states = torch.zeros((max_cycles, num_agents, observation_space["internal_states"].shape[0]), dtype=torch.float32).to(device)
    rb_obs_social_norms = torch.zeros((max_cycles, num_agents, observation_space["perceived_social_norm"].shape[0]), dtype=torch.float32).to(device)
    
    rb_actions = torch.zeros((max_cycles, num_agents), dtype=torch.long).to(device)
    rb_logprobs = torch.zeros((max_cycles, num_agents), dtype=torch.float32).to(device)
    rb_rewards = torch.zeros((max_cycles, num_agents), dtype=torch.float32).to(device)
    rb_terms = torch.zeros((max_cycles, num_agents), dtype=torch.float32).to(device)
    rb_values = torch.zeros((max_cycles, num_agents), dtype=torch.float32).to(device)

    """ TRAINING LOGIC """
    # train for n number of episodes
    for episode in range(total_episodes):
        # collect an episode
        with torch.no_grad():
            # collect observations and convert to batch of torch tensors
            if episode == 0:
                next_obs = observations  # Use observations from initial reset
            else:
                next_obs, info = env.reset(seed=None)
            # reset the episodic return
            total_episodic_return = 0

            # each episode has num_steps
            for step in range(0, max_cycles):
                # rollover the observation
                obs = batchify_obs(next_obs, device)

                # get action from the agent
                actions, logprobs, _, values = agent.get_action_and_value(obs)

                # execute the environment and log data
                next_obs, rewards, terms, truncs, infos = env.step(
                    unbatchify(actions, env)
                )

                # add to episode storage
                rb_obs_positions[step] = obs["position"]
                rb_obs_internal_states[step] = obs["internal_states"]
                rb_obs_social_norms[step] = obs["perceived_social_norm"]
                rb_rewards[step] = batchify(rewards, device)
                rb_terms[step] = batchify(terms, device)
                rb_actions[step] = actions
                rb_logprobs[step] = logprobs
                rb_values[step] = values.flatten()

                # compute episodic return
                total_episodic_return += rb_rewards[step].cpu().numpy()

                # if we reach termination or truncation, end
                if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                    end_step = step
                    break

        # bootstrap value if not done
        with torch.no_grad():
            rb_advantages = torch.zeros_like(rb_rewards).to(device)
            for t in reversed(range(end_step)):
                delta = (
                    rb_rewards[t]
                    + gamma * rb_values[t + 1] * rb_terms[t + 1]
                    - rb_values[t]
                )
                rb_advantages[t] = delta + gamma * gamma * rb_advantages[t + 1]
            rb_returns = rb_advantages + rb_values

        # convert our episodes to batch of individual transitions
        b_obs_positions = torch.flatten(rb_obs_positions[:end_step], start_dim=0, end_dim=1)
        b_obs_internal_states = torch.flatten(rb_obs_internal_states[:end_step], start_dim=0, end_dim=1)
        b_obs_social_norms = torch.flatten(rb_obs_social_norms[:end_step], start_dim=0, end_dim=1)
        b_logprobs = torch.flatten(rb_logprobs[:end_step], start_dim=0, end_dim=1)
        b_actions = torch.flatten(rb_actions[:end_step], start_dim=0, end_dim=1)
        b_returns = torch.flatten(rb_returns[:end_step], start_dim=0, end_dim=1)
        b_values = torch.flatten(rb_values[:end_step], start_dim=0, end_dim=1)
        b_advantages = torch.flatten(rb_advantages[:end_step], start_dim=0, end_dim=1)

        # Optimizing the policy and value network
        b_index = np.arange(len(b_obs_positions))
        clip_fracs = []
        for repeat in range(4):
            # shuffle the indices we use to access the data
            np.random.shuffle(b_index)
            for start in range(0, len(b_obs_positions), batch_size):
                # select the indices we want to train on
                end = start + batch_size
                batch_index = b_index[start:end]

                # Create batch observations
                batch_obs = {
                    "position": b_obs_positions[batch_index],
                    "internal_states": b_obs_internal_states[batch_index],
                    "perceived_social_norm": b_obs_social_norms[batch_index]
                }

                _, newlogprob, entropy, value = agent.get_action_and_value(
                    batch_obs, b_actions.long()[batch_index]
                )
                logratio = newlogprob - b_logprobs[batch_index]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fracs += [
                        ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                    ]

                # normalize advantages
                advantages = b_advantages[batch_index]
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                # Policy loss
                pg_loss1 = -b_advantages[batch_index] * ratio
                pg_loss2 = -b_advantages[batch_index] * torch.clamp(
                    ratio, 1 - clip_coef, 1 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                value = value.flatten()
                v_loss_unclipped = (value - b_returns[batch_index]) ** 2
                v_clipped = b_values[batch_index] + torch.clamp(
                    value - b_values[batch_index],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        print(f"Training episode {episode}")
        print(f"Episodic Return: {np.mean(total_episodic_return)}")
        print(f"Episode Length: {end_step}")
        print("")
        print(f"Value Loss: {v_loss.item()}")
        print(f"Policy Loss: {pg_loss.item()}")
        print(f"Old Approx KL: {old_approx_kl.item()}")
        print(f"Approx KL: {approx_kl.item()}")
        print(f"Clip Fraction: {np.mean(clip_fracs)}")
        print(f"Explained Variance: {explained_var.item()}")
        print("\n-------------------------------------------\n")

    """ RENDER THE POLICY """
    env = NormalHomeostaticEnv(
        config_path=config_path,
        drive_type=drive_type,
        learning_rate=learning_rate,
        beta=beta,
        number_resources=number_resources,
        n_agents=n_agents,
        size=size,
        render_mode="human"
    )

    agent.eval()

    with torch.no_grad():
        # render 5 episodes out
        for episode in range(5):
            obs, infos = env.reset(seed=None)
            obs = batchify_obs(obs, device)
            terms = [False]
            truncs = [False]
            while not any(terms) and not any(truncs):
                actions, logprobs, _, values = agent.get_action_and_value(obs)
                obs, rewards, terms, truncs, infos = env.step(unbatchify(actions, env))
                obs = batchify_obs(obs, device)
                terms = [terms[a] for a in terms]
                truncs = [truncs[a] for a in truncs] 
