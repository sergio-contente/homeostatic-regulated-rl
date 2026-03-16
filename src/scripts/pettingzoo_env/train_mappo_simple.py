#!/usr/bin/env python3
"""
MAPPO training for multi-agent homeostatic environment.

Trains agents using PPO with parameter sharing in a tragedy-of-the-commons setting
where agents must balance homeostatic needs against shared resource consumption.
"""

import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import argparse
from dataclasses import dataclass

from src.envs.multiagent import create_env
from pettingzoo.utils.conversions import aec_to_parallel


@dataclass
class SimpleArgs:
    # Environment
    config_path: str = "config/config.yaml"
    drive_type: str = "base_drive"
    learning_rate_social: float = 0.1
    beta: float = 0.1
    number_resources: int = 1
    n_agents: int = 10
    env_size: int = 1
    max_steps: int = 1000
    initial_resource_stock: float = 1.0
    scarcity_mode: str = "original"

    # Training
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    num_envs: int = 1
    num_steps: int = 200  # 20 rounds * 10 agents
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Other
    seed: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir: str = "mappo_logs"
    verbose_logging: bool = False
    track_individual_agents: bool = True
    max_agents_to_track: int = 10


class SimpleAgent(nn.Module):
    """MAPPO agent with shared parameters."""

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        shared_features = self.shared(x)
        return self.actor(shared_features), self.critic(shared_features)

    def get_action_and_value(self, x, action=None):
        logits, value = self.forward(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value


class ObservationProcessor:
    """Flatten dict observations into numpy arrays."""

    def __init__(self, sample_obs):
        if isinstance(sample_obs, dict):
            self.obs_dim = sum(
                np.array(v).size if hasattr(v, 'size') else 1
                for v in sample_obs.values()
            )
            self.is_dict = True
        else:
            self.obs_dim = np.array(sample_obs).size
            self.is_dict = False

    def process(self, obs):
        if self.is_dict:
            flat_obs = []
            for v in obs.values():
                if hasattr(v, '__len__') and not isinstance(v, (int, float)):
                    flat_obs.extend(np.array(v).flatten())
                else:
                    flat_obs.append(float(v))
            return np.array(flat_obs, dtype=np.float32)
        return np.array(obs).flatten().astype(np.float32)


def collect_rollout(env, agent, obs_processor, args, device):
    """Collect a rollout from the parallel environment."""
    observations, actions, log_probs, rewards, values = [], [], [], [], []

    obs_dict, _ = env.reset()

    for step in range(args.num_steps):
        processed_obs = {}
        obs_tensor_dict = {}

        for agent_id in env.agents:
            if agent_id in obs_dict:
                processed_obs[agent_id] = obs_processor.process(obs_dict[agent_id])
                obs_tensor_dict[agent_id] = torch.FloatTensor(
                    processed_obs[agent_id]
                ).unsqueeze(0).to(device)

        actions_dict, log_probs_dict, values_dict = {}, {}, {}
        for agent_id in env.agents:
            if agent_id in obs_tensor_dict:
                with torch.no_grad():
                    action, log_prob, _, value = agent.get_action_and_value(
                        obs_tensor_dict[agent_id]
                    )
                    actions_dict[agent_id] = action.item()
                    log_probs_dict[agent_id] = log_prob.item()
                    values_dict[agent_id] = value.item()

        obs_dict, rewards_dict, terminations, truncations, infos = env.step(actions_dict)

        observations.append(processed_obs)
        actions.append(actions_dict)
        log_probs.append(log_probs_dict)
        rewards.append(rewards_dict)
        values.append(values_dict)

        if all(terminations.values()) or all(truncations.values()) or len(env.agents) == 0:
            break

    return observations, actions, log_probs, rewards, values


def _log_detailed_metrics(env, writer, global_step, args=None):
    """Log homeostatic environment metrics to TensorBoard."""
    try:
        base_env = env
        for _ in range(10):
            if not hasattr(base_env, 'unwrapped'):
                break
            next_env = base_env.unwrapped
            if next_env is base_env:
                break
            base_env = next_env

        if not hasattr(base_env, 'homeostatic_agents'):
            return

        agents = base_env.homeostatic_agents
        resource_stock = getattr(base_env, 'resource_stock', None)
        track_individual = getattr(args, 'track_individual_agents', True)
        max_track = getattr(args, 'max_agents_to_track', 10)

        if agents and len(agents) > 0:
            internal_states, drives, social_norms, last_intakes = [], [], [], []
            agent_items = list(agents.items())
            agents_to_track = agent_items[:max_track] if track_individual else []

            for agent_id, agent in agent_items:
                if hasattr(agent, 'internal_states'):
                    internal_state = agent.internal_states[0]
                    internal_states.append(internal_state)
                if hasattr(agent, 'get_current_drive'):
                    drive = agent.get_current_drive()
                    drives.append(drive)
                if hasattr(agent, 'perceived_social_norm'):
                    social_norm = agent.perceived_social_norm[0] if len(agent.perceived_social_norm) > 0 else 0
                    social_norms.append(social_norm)
                if hasattr(agent, 'last_intake'):
                    intake = agent.last_intake[0] if len(agent.last_intake) > 0 else 0
                    last_intakes.append(intake)

                # Individual agent tracking
                if track_individual and (agent_id, agent) in agents_to_track:
                    if hasattr(agent, 'internal_states'):
                        writer.add_scalar(f"individual_agents/{agent_id}/internal_state", internal_state, global_step)
                    if hasattr(agent, 'get_current_drive'):
                        writer.add_scalar(f"individual_agents/{agent_id}/drive", drive, global_step)
                    if hasattr(agent, 'perceived_social_norm'):
                        writer.add_scalar(f"individual_agents/{agent_id}/social_norm", social_norm, global_step)
                    if hasattr(agent, 'last_intake'):
                        writer.add_scalar(f"individual_agents/{agent_id}/intake", intake, global_step)
                    if hasattr(agent, 'position'):
                        writer.add_scalar(f"individual_agents/{agent_id}/position", agent.position, global_step)
                    if hasattr(agent, 'internal_states') and hasattr(agent, 'get_current_drive'):
                        writer.add_scalar(f"individual_agents/{agent_id}/homeostatic_balance", 1.0 - abs(internal_state), global_step)
                        writer.add_scalar(f"individual_agents/{agent_id}/urgency", drive * abs(internal_state), global_step)

            if internal_states:
                writer.add_scalar("agents/avg_internal_state", np.mean(internal_states), global_step)
                writer.add_scalar("agents/std_internal_state", np.std(internal_states), global_step)
                writer.add_scalar("agents/min_internal_state", np.min(internal_states), global_step)
                writer.add_scalar("agents/max_internal_state", np.max(internal_states), global_step)
                writer.add_histogram("distributions/internal_states", np.array(internal_states), global_step)
            if drives:
                writer.add_scalar("agents/avg_drive", np.mean(drives), global_step)
                writer.add_scalar("agents/std_drive", np.std(drives), global_step)
                writer.add_scalar("agents/max_drive", np.max(drives), global_step)
                writer.add_histogram("distributions/drives", np.array(drives), global_step)
            if social_norms:
                writer.add_scalar("agents/avg_social_norm", np.mean(social_norms), global_step)
                writer.add_scalar("agents/std_social_norm", np.std(social_norms), global_step)
                writer.add_histogram("distributions/social_norms", np.array(social_norms), global_step)
            if last_intakes:
                writer.add_scalar("agents/avg_consumption", np.mean(last_intakes), global_step)
                writer.add_scalar("agents/total_consumption", np.sum(last_intakes), global_step)
                writer.add_histogram("distributions/consumptions", np.array(last_intakes), global_step)

            writer.add_scalar("population/alive_agents", len(agents), global_step)

            # Cooperation metrics
            if last_intakes and internal_states:
                non_critical = [i for i, s in enumerate(internal_states) if s > -0.5]
                if non_critical:
                    cooperation_index = 1.0 - (np.mean([last_intakes[i] for i in non_critical]) / 0.1)
                    writer.add_scalar("cooperation/cooperation_index", max(0, cooperation_index), global_step)
                if resource_stock is not None:
                    initial_stock = getattr(env.unwrapped, 'initial_resource_stock', [1000])[0]
                    resource_ratio = resource_stock[0] / initial_stock
                    population_ratio = len(agents) / 10
                    writer.add_scalar("cooperation/sustainability_index", (resource_ratio + population_ratio) / 2, global_step)

        if resource_stock is not None:
            writer.add_scalar("resources/stock_food", resource_stock[0], global_step)
            initial_stock = getattr(env.unwrapped, 'initial_resource_stock', [1000])[0]
            writer.add_scalar("resources/stock_percentage", resource_stock[0] / initial_stock * 100, global_step)
            if hasattr(env.unwrapped, '_last_resource_stock'):
                writer.add_scalar("resources/depletion_rate", env.unwrapped._last_resource_stock - resource_stock[0], global_step)
            env.unwrapped._last_resource_stock = resource_stock[0]

    except Exception as e:
        print(f"Warning: Could not log detailed metrics: {e}")


def train_mappo_simple(args):
    """MAPPO training loop."""
    print(f"MAPPO Training | agents={args.n_agents} | beta={args.beta} | stock={args.initial_resource_stock} | scarcity={args.scarcity_mode} | device={args.device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Create environment
    env = create_env(
        config_path=args.config_path,
        drive_type=args.drive_type,
        learning_rate=args.learning_rate_social,
        beta=args.beta,
        number_resources=args.number_resources,
        n_agents=args.n_agents,
        size=args.env_size,
        max_steps=args.max_steps,
        seed=args.seed,
        initial_resource_stock=args.initial_resource_stock,
        scarcity_mode=args.scarcity_mode
    )
    env = aec_to_parallel(env)

    sample_obs_dict, _ = env.reset()
    if not sample_obs_dict or not env.agents:
        raise ValueError("Environment returned empty observations or no agents.")

    sample_obs = next(iter(sample_obs_dict.values()))
    obs_processor = ObservationProcessor(sample_obs)
    action_dim = env.action_space(env.agents[0]).n

    print(f"  obs_dim={obs_processor.obs_dim}, action_dim={action_dim}, agents={len(env.agents)}")

    # Create agent and optimizer
    agent = SimpleAgent(obs_processor.obs_dim, action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)

    # TensorBoard
    run_name = f"simple_mappo_{int(time.time())}"
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(f"{args.log_dir}/{run_name}")

    global_step = 0
    num_iterations = args.total_timesteps // args.num_steps

    for iteration in range(num_iterations):
        # Collect rollout
        observations, actions_list, log_probs_list, rewards_list, values_list = collect_rollout(
            env, agent, obs_processor, args, device
        )

        if not observations:
            env.reset()
            continue

        # Flatten rollout data
        all_obs, all_actions, all_log_probs, all_rewards, all_values = [], [], [], [], []
        for step_obs, step_act, step_lp, step_rew, step_val in zip(
            observations, actions_list, log_probs_list, rewards_list, values_list
        ):
            for agent_id in step_obs:
                if all(agent_id in d for d in [step_act, step_lp, step_rew, step_val]):
                    all_obs.append(step_obs[agent_id])
                    all_actions.append(step_act[agent_id])
                    all_log_probs.append(step_lp[agent_id])
                    all_rewards.append(step_rew[agent_id])
                    all_values.append(step_val[agent_id])

        if len(all_obs) < 32:
            continue

        # Compute advantages
        advantages = [r - v for r, v in zip(all_rewards, all_values)]
        returns = list(all_rewards)

        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(all_obs)).to(device)
        actions_tensor = torch.LongTensor(all_actions).to(device)
        old_log_probs = torch.FloatTensor(all_log_probs).to(device)
        advantages_tensor = torch.FloatTensor(advantages).to(device)
        returns_tensor = torch.FloatTensor(returns).to(device)

        # Normalize advantages
        if advantages_tensor.std() > 1e-8:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # Skip if NaN
        if torch.isnan(obs_tensor).any() or torch.isnan(advantages_tensor).any():
            continue

        # PPO update
        for epoch in range(args.update_epochs):
            _, new_log_probs, entropy, values_pred = agent.get_action_and_value(obs_tensor, actions_tensor)

            if torch.isnan(new_log_probs).any() or torch.isinf(new_log_probs).any():
                break

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values_pred.squeeze(), returns_tensor)
            entropy_loss = entropy.mean()
            loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                break
            optimizer.step()

        # Log metrics
        avg_reward = np.mean(all_rewards) if all_rewards else 0
        global_step += len(all_obs)

        writer.add_scalar("train/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("train/value_loss", value_loss.item(), global_step)
        writer.add_scalar("train/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("train/total_loss", loss.item(), global_step)
        writer.add_scalar("train/avg_reward", avg_reward, global_step)
        writer.add_scalar("train/data_points", len(all_obs), global_step)

        if all_rewards:
            writer.add_scalar("rewards/avg_homeostatic", np.mean(all_rewards), global_step)
            writer.add_scalar("rewards/std_homeostatic", np.std(all_rewards), global_step)

        _log_detailed_metrics(env, writer, global_step, args)

        # Progress logging
        if (iteration + 1) % 10 == 0:
            print(f"  [{iteration+1}/{num_iterations}] reward={avg_reward:.2f} policy_loss={policy_loss.item():.4f} points={len(all_obs)}")

        # Save model periodically
        if (iteration + 1) % 50 == 0:
            os.makedirs("models", exist_ok=True)
            torch.save({
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iteration': iteration,
                'args': args
            }, f"models/simple_mappo_iter_{iteration + 1}.pt")

    # Save final model
    os.makedirs("models", exist_ok=True)
    final_model_path = f"models/simple_mappo_final_{run_name}.pt"
    torch.save({
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': num_iterations,
        'args': args
    }, final_model_path)

    print(f"Training completed! Model: {final_model_path} | Logs: {args.log_dir}")
    env.close()
    writer.close()


def main():
    parser = argparse.ArgumentParser(description="MAPPO training for homeostatic environment")
    parser.add_argument("--config_path", type=str, default="config/config.yaml")
    parser.add_argument("--n_agents", type=int, default=10)
    parser.add_argument("--initial_resource_stock", type=float, default=5.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--total_timesteps", type=int, default=100_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log_dir", type=str, default="mappo_logs")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--scarcity_mode", type=str, default="original",
                        choices=["original", "adjusted_ab", "soft", "combined"])
    parser.add_argument("--no-individual-tracking", action="store_true")
    parser.add_argument("--max-agents-tracked", type=int, default=10)
    args = parser.parse_args()

    train_args = SimpleArgs(
        config_path=args.config_path,
        n_agents=args.n_agents,
        initial_resource_stock=args.initial_resource_stock,
        beta=args.beta,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        log_dir=args.log_dir,
        verbose_logging=args.verbose,
        track_individual_agents=not args.no_individual_tracking,
        max_agents_to_track=args.max_agents_tracked,
        scarcity_mode=args.scarcity_mode
    )
    train_mappo_simple(train_args)


if __name__ == "__main__":
    main()
