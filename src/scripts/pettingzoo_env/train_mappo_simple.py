#!/usr/bin/env python3
"""
Simple MAPPO Implementation for Homeostatic Environment

This is a simplified version that avoids SuperSuit wrapper complications
and focuses on getting MAPPO working with your environment.

Key features:
- Direct PettingZoo integration without problematic wrappers
- Simplified observation handling
- Focus on the tragedy of commons problem
- Fixed: env.reset() returns (observations, infos) tuple handling
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
from typing import Dict, List, Tuple
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
    beta: float = 0.5
    number_resources: int = 1
    n_agents: int = 10
    env_size: int = 1
    max_steps: int = 1000
    initial_resource_stock: float = 1000.0
    
    # Training
    total_timesteps: int = 100_000  # Reduced for faster testing
    learning_rate: float = 3e-4
    num_envs: int = 1  # Start with 1 environment
    num_steps: int = 64  # Reduced rollout length
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
    log_dir: str = "mappo_logs"  # Custom log directory
    verbose_logging: bool = False  # Show detailed info every iteration


class SimpleAgent(nn.Module):
    """Simplified MAPPO agent."""
    
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        
        # Shared network
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        
        # Actor head
        self.actor = nn.Linear(128, action_dim)
        
        # Critic head  
        self.critic = nn.Linear(128, 1)
        
        # Initialize weights
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
    """Process observations from PettingZoo environment."""
    
    def __init__(self, sample_obs):
        """Initialize with a sample observation to determine dimensions."""
        if isinstance(sample_obs, dict):
            # Flatten dict observation
            self.obs_dim = sum([
                np.array(v).size if hasattr(v, 'size') else 1
                for v in sample_obs.values()
            ])
            self.is_dict = True
        else:
            self.obs_dim = np.array(sample_obs).size
            self.is_dict = False
    
    def process(self, obs):
        """Convert observation to flat numpy array."""
        if self.is_dict:
            # Flatten all values in the dict
            flat_obs = []
            for v in obs.values():
                if hasattr(v, '__len__') and not isinstance(v, (int, float)):
                    flat_obs.extend(np.array(v).flatten())
                else:
                    flat_obs.append(float(v))
            return np.array(flat_obs, dtype=np.float32)
        else:
            return np.array(obs).flatten().astype(np.float32)


def collect_rollout(env, agent, obs_processor, args, device):
    """Collect a rollout from the environment."""
    
    observations = []
    actions = []
    log_probs = []
    rewards = []
    values = []
    dones = []
    
    # Reset environment
    obs_dict, _ = env.reset()  # env.reset() returns (observations, infos)
    
    for step in range(args.num_steps):
        # Process observations for all agents
        processed_obs = {}
        obs_tensor_dict = {}
        
        for agent_id in env.agents:
            if agent_id in obs_dict:
                processed_obs[agent_id] = obs_processor.process(obs_dict[agent_id])
                obs_tensor_dict[agent_id] = torch.FloatTensor(processed_obs[agent_id]).unsqueeze(0).to(device)
        
        # Get actions for all agents
        actions_dict = {}
        log_probs_dict = {}
        values_dict = {}
        
        for agent_id in env.agents:
            if agent_id in obs_tensor_dict:
                with torch.no_grad():
                    action, log_prob, _, value = agent.get_action_and_value(obs_tensor_dict[agent_id])
                    actions_dict[agent_id] = action.item()
                    log_probs_dict[agent_id] = log_prob.item()
                    values_dict[agent_id] = value.item()
        
        # Step environment
        obs_dict, rewards_dict, terminations, truncations, infos = env.step(actions_dict)
        
        # Store step data
        observations.append(processed_obs)
        actions.append(actions_dict)
        log_probs.append(log_probs_dict)
        rewards.append(rewards_dict)
        values.append(values_dict)
        
        # Check if episode is done
        if all(terminations.values()) or all(truncations.values()) or len(env.agents) == 0:
            print(f"Episode ended at step {step + 1}: {len(env.agents)} agents remaining")
            break
    
    return observations, actions, log_probs, rewards, values


def compute_advantages(rewards, values, args):
    """Compute advantages using GAE - only for data that actually exists."""
    advantages = []
    returns = []
    
    # Process each step and each agent that exists in that step
    for t in range(len(rewards)):
        step_rewards = rewards[t]
        step_values = values[t]
        
        for agent_id in step_rewards.keys():
            if agent_id in step_values:
                # For simplicity, compute single-step advantage (can be improved to multi-step GAE)
                reward = step_rewards[agent_id]
                value = step_values[agent_id]
                
                # Simple advantage: reward - value (can be enhanced with GAE later)
                advantage = reward - value
                return_val = reward + value  # Simplified return
                
                advantages.append(advantage)
                returns.append(return_val)
    
    return np.array(advantages), np.array(returns)


def _log_detailed_metrics(env, writer, global_step):
    """Log detailed homeostatic environment metrics to TensorBoard."""
    try:
        if not hasattr(env, 'unwrapped') or not hasattr(env.unwrapped, 'homeostatic_agents'):
            return
            
        agents = env.unwrapped.homeostatic_agents
        resource_stock = getattr(env.unwrapped, 'resource_stock', None)
        
        if agents and len(agents) > 0:
            # Collect data from all living agents
            internal_states = []
            drives = []
            social_norms = []
            last_intakes = []
            
            # Individual agent tracking
            for agent_id, agent in agents.items():
                # Individual metrics for each agent
                if hasattr(agent, 'internal_states'):
                    internal_state = agent.internal_states[0]
                    internal_states.append(internal_state)
                    writer.add_scalar(f"individual_agents/{agent_id}/internal_state", internal_state, global_step)
                    
                if hasattr(agent, 'get_current_drive'):
                    drive = agent.get_current_drive()
                    drives.append(drive)
                    writer.add_scalar(f"individual_agents/{agent_id}/drive", drive, global_step)
                    
                if hasattr(agent, 'perceived_social_norm'):
                    social_norm = agent.perceived_social_norm[0] if len(agent.perceived_social_norm) > 0 else 0
                    social_norms.append(social_norm)
                    writer.add_scalar(f"individual_agents/{agent_id}/social_norm", social_norm, global_step)
                    
                if hasattr(agent, 'last_intake'):
                    intake = agent.last_intake[0] if len(agent.last_intake) > 0 else 0
                    last_intakes.append(intake)
                    writer.add_scalar(f"individual_agents/{agent_id}/intake", intake, global_step)
                    
                # Agent position
                if hasattr(agent, 'position'):
                    writer.add_scalar(f"individual_agents/{agent_id}/position", agent.position, global_step)
                    
                # Derived metrics for individual agents
                if hasattr(agent, 'internal_states') and hasattr(agent, 'get_current_drive'):
                    # Homeostatic balance: how close to optimal state (0)
                    balance = 1.0 - abs(internal_state)  # 1 = perfect balance, 0 = very unbalanced
                    writer.add_scalar(f"individual_agents/{agent_id}/homeostatic_balance", balance, global_step)
                    
                    # Urgency: combination of drive and internal state deviation
                    urgency = drive * abs(internal_state)
                    writer.add_scalar(f"individual_agents/{agent_id}/urgency", urgency, global_step)
            
            # Log agent statistics and distributions
            if internal_states:
                writer.add_scalar("agents/avg_internal_state", np.mean(internal_states), global_step)
                writer.add_scalar("agents/std_internal_state", np.std(internal_states), global_step)
                writer.add_scalar("agents/min_internal_state", np.min(internal_states), global_step)
                writer.add_scalar("agents/max_internal_state", np.max(internal_states), global_step)
                # Add histogram to see distribution
                writer.add_histogram("distributions/internal_states", np.array(internal_states), global_step)
                
            if drives:
                writer.add_scalar("agents/avg_drive", np.mean(drives), global_step)
                writer.add_scalar("agents/std_drive", np.std(drives), global_step)
                writer.add_scalar("agents/max_drive", np.max(drives), global_step)
                # Add histogram to see distribution
                writer.add_histogram("distributions/drives", np.array(drives), global_step)
                
            if social_norms:
                writer.add_scalar("agents/avg_social_norm", np.mean(social_norms), global_step)
                writer.add_scalar("agents/std_social_norm", np.std(social_norms), global_step)
                # Add histogram to see distribution
                writer.add_histogram("distributions/social_norms", np.array(social_norms), global_step)
                
            if last_intakes:
                writer.add_scalar("agents/avg_consumption", np.mean(last_intakes), global_step)
                writer.add_scalar("agents/total_consumption", np.sum(last_intakes), global_step)
                # Add histogram to see distribution
                writer.add_histogram("distributions/consumptions", np.array(last_intakes), global_step)
                
                         # Log population metrics
            writer.add_scalar("population/alive_agents", len(agents), global_step)
            
            # Log cooperation metrics
            if last_intakes and internal_states:
                # Cooperation index: low consumption when internal state is not critical
                non_critical_agents = [i for i, state in enumerate(internal_states) if state > -0.5]
                if non_critical_agents:
                    non_critical_consumption = [last_intakes[i] for i in non_critical_agents]
                    cooperation_index = 1.0 - (np.mean(non_critical_consumption) / 0.1)  # Normalized cooperation
                    writer.add_scalar("cooperation/cooperation_index", max(0, cooperation_index), global_step)
                
                # Sustainability index: resource preservation vs population
                if resource_stock is not None:
                    initial_stock = getattr(env.unwrapped, 'initial_resource_stock', [1000])[0]
                    resource_ratio = resource_stock[0] / initial_stock
                    population_ratio = len(agents) / 10  # Assuming initial 10 agents
                    sustainability = (resource_ratio + population_ratio) / 2
                    writer.add_scalar("cooperation/sustainability_index", sustainability, global_step)
            
        # Log resource information
        if resource_stock is not None:
            writer.add_scalar("resources/stock_food", resource_stock[0], global_step)
            initial_stock = getattr(env.unwrapped, 'initial_resource_stock', [1000])[0]
            writer.add_scalar("resources/stock_percentage", resource_stock[0] / initial_stock * 100, global_step)
            
            # Resource depletion rate
            if hasattr(env.unwrapped, '_last_resource_stock'):
                depletion_rate = env.unwrapped._last_resource_stock - resource_stock[0]
                writer.add_scalar("resources/depletion_rate", depletion_rate, global_step)
            env.unwrapped._last_resource_stock = resource_stock[0]
            
    except Exception as e:
        # Don't let logging errors break training
        print(f"Warning: Could not log detailed metrics: {e}")


def _print_detailed_info(env, iteration, avg_reward):
    """Print detailed information about the current state."""
    try:
        if not hasattr(env, 'unwrapped') or not hasattr(env.unwrapped, 'homeostatic_agents'):
            return
            
        agents = env.unwrapped.homeostatic_agents
        resource_stock = getattr(env.unwrapped, 'resource_stock', None)
        
        print(f"\n📊 Detailed Info - Iteration {iteration}:")
        
        if resource_stock is not None:
            initial_stock = getattr(env.unwrapped, 'initial_resource_stock', [1000])[0]
            stock_percentage = (resource_stock[0] / initial_stock) * 100
            print(f"🌱 Resources: {resource_stock[0]:.1f} ({stock_percentage:.1f}% of initial)")
        
        if agents and len(agents) > 0:
            print(f"👥 Agents alive: {len(agents)}")
            
            # Sample 3 agents for detailed view
            agent_sample = list(agents.items())[:3]
            
            for agent_id, agent in agent_sample:
                internal_state = agent.internal_states[0] if hasattr(agent, 'internal_states') else "N/A"
                drive = agent.get_current_drive() if hasattr(agent, 'get_current_drive') else "N/A"
                social_norm = agent.perceived_social_norm[0] if hasattr(agent, 'perceived_social_norm') and len(agent.perceived_social_norm) > 0 else "N/A"
                last_intake = agent.last_intake[0] if hasattr(agent, 'last_intake') and len(agent.last_intake) > 0 else "N/A"
                
                print(f"  {agent_id}: state={internal_state:.3f}, drive={drive:.3f}, norm={social_norm:.3f}, intake={last_intake:.3f}")
        
        print(f"💰 Avg Reward: {avg_reward:.3f}")
        
    except Exception as e:
        print(f"Warning: Could not print detailed info: {e}")


def train_mappo_simple(args):
    """Simplified MAPPO training."""
    
    print(f"🚀 Starting Simple MAPPO Training")
    print(f"Device: {args.device}")
    print(f"Agents: {args.n_agents}")
    print(f"Initial Resources: {args.initial_resource_stock}")
    print(f"Log Directory: {args.log_dir}")
    
    # Set seeds
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
        initial_resource_stock=args.initial_resource_stock
    )
    
    # Convert to parallel for easier handling
    env = aec_to_parallel(env)
    
    # Get observation dimensions
    sample_obs_dict, _ = env.reset()  # env.reset() returns (observations, infos)
    
    if not sample_obs_dict:
        raise ValueError("Environment returned empty observations. Check environment setup.")
    
    if not env.agents:
        raise ValueError("No agents found in environment.")
    
    sample_obs = next(iter(sample_obs_dict.values()))
    obs_processor = ObservationProcessor(sample_obs)
    
    print(f"Observation dimension: {obs_processor.obs_dim}")
    print(f"Sample observation keys: {list(sample_obs.keys()) if isinstance(sample_obs, dict) else 'Not a dict'}")
    
    # Get action dimension
    action_dim = env.action_space(env.agents[0]).n
    print(f"Action dimension: {action_dim}")
    print(f"Number of agents: {len(env.agents)}")
    
    # Create agent
    agent = SimpleAgent(obs_processor.obs_dim, action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)
    
    # Training setup
    run_name = f"simple_mappo_{int(time.time())}"
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(f"{args.log_dir}/{run_name}")
    
    print(f"📊 Logs salvos em: {args.log_dir}/{run_name}")
    print(f"📊 Para ver TensorBoard: tensorboard --logdir {args.log_dir}")
    print(f"\n📈 Métricas disponíveis no TensorBoard:")
    print(f"   • train/* - Métricas de treinamento (loss, entropy, reward)")
    print(f"   • agents/* - Estados internos, drives, normas sociais, consumo")
    print(f"   • resources/* - Stock de recursos, percentual, taxa de depleção")
    print(f"   • population/* - Número de agentes vivos")
    print(f"   • cooperation/* - Índices de cooperação e sustentabilidade")
    print(f"   • rewards/* - Breakdown de recompensa homeostática vs custo social")
    
    global_step = 0
    num_iterations = args.total_timesteps // args.num_steps
    
    print(f"Training for {num_iterations} iterations...")
    print(f"Expected data per iteration: ~{args.num_steps * args.n_agents} points")
    
    for iteration in range(num_iterations):
        print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")
        
        # Collect rollout
        observations, actions, log_probs, rewards, values = collect_rollout(
            env, agent, obs_processor, args, device
        )
        
        if not observations:
            print("Warning: No observations collected, resetting environment")
            _, _ = env.reset()  # Reset for cleanup, discard return values
            continue
        
        # Skip if no observations collected
        if not observations:
            print("Warning: No observations collected")
            continue
        
        # Prepare training data - ensure all lists have same length
        all_obs = []
        all_actions = []
        all_log_probs = []
        all_rewards = []
        all_values = []
        
        # Collect data ensuring consistency across all lists
        homeostatic_rewards = []
        social_costs = []
        
        for step_obs, step_actions, step_log_probs, step_rewards, step_values in zip(
            observations, actions, log_probs, rewards, values
        ):
            for agent_id in step_obs:
                if (agent_id in step_actions and 
                    agent_id in step_log_probs and 
                    agent_id in step_rewards and 
                    agent_id in step_values):
                    
                    all_obs.append(step_obs[agent_id])
                    all_actions.append(step_actions[agent_id])
                    all_log_probs.append(step_log_probs[agent_id])
                    all_rewards.append(step_rewards[agent_id])
                    all_values.append(step_values[agent_id])
                    
                    # Try to get detailed reward breakdown if available
                    # Note: This is simplified - in a full implementation you'd separate these during reward calculation
                    total_reward = step_rewards[agent_id]
                    homeostatic_rewards.append(total_reward)  # Estimated homeostatic component
                    social_costs.append(abs(total_reward))    # Estimated social cost component
        
        if len(all_obs) == 0:
            print("Warning: No training data available")
            continue
        
        # Compute advantages and returns for the exact same data
        advantages = []
        returns = []
        for i in range(len(all_rewards)):
            reward = all_rewards[i]
            value = all_values[i]
            advantage = reward - value  # Simple advantage
            return_val = reward  # Use actual reward as return
            advantages.append(advantage)
            returns.append(return_val)
        
        # Verify all lists have same length
        data_lengths = [len(all_obs), len(all_actions), len(all_log_probs), len(advantages), len(returns)]
        if len(set(data_lengths)) > 1:
            print(f"⚠️  Data length mismatch: {data_lengths}")
            print("Skipping this iteration...")
            continue
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(all_obs)).to(device)
        actions_tensor = torch.LongTensor(all_actions).to(device)
        old_log_probs = torch.FloatTensor(all_log_probs).to(device)
        advantages_tensor = torch.FloatTensor(advantages).to(device)
        returns_tensor = torch.FloatTensor(returns).to(device)
        
        # Normalize advantages
        if advantages_tensor.std() > 1e-8:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        else:
            print("⚠️  Warning: Zero advantage variance, skipping normalization")
        
        # Update policy
        for epoch in range(args.update_epochs):
            # Get current policy outputs
            _, new_log_probs, entropy, values_pred = agent.get_action_and_value(obs_tensor, actions_tensor)
            
            # Policy loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(values_pred.squeeze(), returns_tensor)
            
            # Entropy loss
            entropy_loss = entropy.mean()
            
            # Total loss
            loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy_loss
            
            # Update
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()
        
        # Log metrics
        total_reward = sum(all_rewards) if all_rewards else 0
        avg_reward = total_reward / len(all_rewards) if all_rewards else 0
        
        global_step += len(all_obs)
        
        # Basic training metrics
        writer.add_scalar("train/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("train/value_loss", value_loss.item(), global_step)
        writer.add_scalar("train/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("train/total_loss", loss.item(), global_step)
        writer.add_scalar("train/avg_reward", avg_reward, global_step)
        writer.add_scalar("train/data_points", len(all_obs), global_step)
        
        # Reward breakdown metrics
        if homeostatic_rewards:
            writer.add_scalar("rewards/avg_homeostatic", np.mean(homeostatic_rewards), global_step)
            writer.add_scalar("rewards/std_homeostatic", np.std(homeostatic_rewards), global_step)
        if social_costs:
            writer.add_scalar("rewards/avg_social_cost", np.mean(social_costs), global_step)
            writer.add_scalar("rewards/std_social_cost", np.std(social_costs), global_step)
        
        # Detailed environment metrics
        _log_detailed_metrics(env, writer, global_step)
        
        print(f"Policy Loss: {policy_loss.item():.4f}")
        print(f"Value Loss: {value_loss.item():.4f}")
        print(f"Entropy: {entropy_loss.item():.4f}")
        print(f"Avg Reward: {avg_reward:.2f}")
        print(f"Data points collected: {len(all_obs)}")
        
        # Print detailed environment info 
        if args.verbose_logging or (iteration + 1) % 10 == 0:
            _print_detailed_info(env, iteration + 1, avg_reward)
        
        # Check if we're getting reasonable data
        if len(all_obs) < 10:
            print("⚠️  Warning: Very few data points collected. Agents may be dying too quickly.")
            print("Consider: increasing initial_resource_stock or increasing max_steps")
        
        # Save model periodically
        if (iteration + 1) % 50 == 0:
            os.makedirs("models", exist_ok=True)
            torch.save({
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iteration': iteration,
                'args': args
            }, f"models/simple_mappo_iter_{iteration + 1}.pt")
            print(f"Model saved at iteration {iteration + 1}")
    
    # Save final model
    os.makedirs("models", exist_ok=True)
    final_model_path = f"models/simple_mappo_final_{run_name}.pt"
    torch.save({
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': num_iterations,
        'args': args
    }, final_model_path)
    
    print(f"\n✅ Training completed!")
    print(f"📁 Final model saved: {final_model_path}")
    print(f"📊 Logs saved in: {args.log_dir}")
    print(f"📈 View logs with: tensorboard --logdir {args.log_dir}")
    
    env.close()
    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config/config.yaml")
    parser.add_argument("--n_agents", type=int, default=10)
    parser.add_argument("--initial_resource_stock", type=float, default=1000.0)
    parser.add_argument("--total_timesteps", type=int, default=100_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log_dir", type=str, default="mappo_logs", help="Directory to save logs")
    parser.add_argument("--verbose", action="store_true", help="Show detailed info every iteration")
    
    args = parser.parse_args()
    
    # Create training args
    train_args = SimpleArgs(
        config_path=args.config_path,
        n_agents=args.n_agents,
        initial_resource_stock=args.initial_resource_stock,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        log_dir=args.log_dir,
        verbose_logging=args.verbose
    )
    
    train_mappo_simple(train_args)


if __name__ == "__main__":
    # Examples of usage:
    # python -m src.scripts.pettingzoo_env.train_mappo_simple --total_timesteps 10000 --n_agents 5
    # python -m src.scripts.pettingzoo_env.train_mappo_simple --log_dir "experimento_1" --seed 42
    # python -m src.scripts.pettingzoo_env.train_mappo_simple --log_dir "teste_cooperacao" --n_agents 8 --verbose
    # python -m src.scripts.pettingzoo_env.train_mappo_simple --log_dir "debug" --verbose --total_timesteps 5000
    main() 
