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

# Add some cost when the resource stock is 0 or agents die
# cost dynamically updated = e-(excess)

# people are starving => social cost becomes irrelevant ... and the otherwise is true
# to do this change the scarcity of the environment and see what happens

# second: try to change the discont factor dynamically for future rewards (selfigh or not)

# do not change the amount of resources, instead change the amount of consomption/intake 

 
@dataclass
class SimpleArgs:
    # Environment
    config_path: str = "config/config.yaml"
    drive_type: str = "base_drive"
    learning_rate_social: float = 0.1
    beta: float = 0.0
    number_resources: int = 1
    n_agents: int = 10
    env_size: int = 1
    max_steps: int = 1000
    initial_resource_stock: float = 1.0
    
    # Training
    total_timesteps: int = 100_000  # Reduced for faster testing
    learning_rate: float = 3e-4
    num_envs: int = 1  # Start with 1 environment
    num_steps: int = 20 * n_agents  # Complete rounds: 20 rounds * 10 agents = 200 steps
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
    log_dir: str = "ippo_logs"  # Custom log directory
    verbose_logging: bool = False  # Show detailed info every iteration
    track_individual_agents: bool = True  # Track individual agent metrics in TensorBoard
    max_agents_to_track: int = 10  # Maximum number of agents to track individually


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
    
    completed_rounds = 0
    steps_in_current_round = 0
    
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
        
        # Track round completion (AEC environments cycle through all agents)
        steps_in_current_round += 1
        if step > 0 and len(env.agents) > 0 and steps_in_current_round >= len(env.agents):
            completed_rounds += 1
            steps_in_current_round = 0
            
            # Log progress every few completed rounds
            if completed_rounds % 5 == 0:
                print(f"  📍 Completed {completed_rounds} rounds in rollout (step {step+1}/{args.num_steps})")
        
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
    
    print(f"🔄 Rollout completed: {len(observations)} steps collected")
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


def _log_detailed_metrics(env, writer, global_step, args=None):
    """Log detailed homeostatic environment metrics to TensorBoard."""
    try:
        print("    📊 Accessing base environment...")
        
        # SAFE environment unwrapping with loop limit
        base_env = env
        max_unwrap_depth = 10  # Prevent infinite loops
        unwrap_count = 0
        
        print(f"    📊 Starting unwrapping, env type: {type(base_env)}")
        
        while hasattr(base_env, 'unwrapped') and unwrap_count < max_unwrap_depth:
            print(f"    📊 Unwrapping step {unwrap_count + 1}, current type: {type(base_env)}")
            next_env = base_env.unwrapped
            if next_env is base_env:  # Prevent circular references
                print("    ⚠️ Circular reference detected, breaking unwrap loop")
                break
            base_env = next_env
            unwrap_count += 1
            
        if unwrap_count >= max_unwrap_depth:
            print(f"    ⚠️ Max unwrap depth ({max_unwrap_depth}) reached, stopping")
            
        print(f"    📊 Unwrapping completed, final type: {type(base_env)}")
        
        if hasattr(base_env, 'env'):
            print("    📊 Found .env attribute, accessing...")
            base_env = base_env.env
            print(f"    📊 After .env access, type: {type(base_env)}")
            
        print("    📊 Checking for homeostatic_agents...")
        if not hasattr(base_env, 'homeostatic_agents'):
            print(f"    ⚠️ No homeostatic_agents found in {type(base_env)}, returning...")
            print(f"    📊 Available attributes: {[attr for attr in dir(base_env) if not attr.startswith('_')][:10]}")
            return
            
        print("    📊 Getting agents and resource stock...")
        agents = base_env.homeostatic_agents
        resource_stock = getattr(base_env, 'resource_stock', None)
        print(f"    📊 Found {len(agents) if agents else 0} agents, resource_stock: {resource_stock}")
        
        # Set tracking preferences from args
        track_individual = getattr(args, 'track_individual_agents', True)
        max_track = getattr(args, 'max_agents_to_track', 10)
        
        if agents and len(agents) > 0:
            print("    📊 Starting agent data collection...")
            # Collect data from all living agents
            internal_states = []
            drives = []
            social_norms = []
            last_intakes = []
            
            # Collect data from all agents and optionally track individuals
            agent_items = list(agents.items())
            agents_to_track = agent_items[:max_track] if track_individual else []
            print(f"    📊 Processing {len(agent_items)} agents, tracking {len(agents_to_track)} individually...")
            
            for agent_id, agent in agent_items:
                # Collect aggregate data from all agents
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
                
                # Individual agent tracking (limited number)
                if track_individual and (agent_id, agent) in agents_to_track:
                    # Individual metrics for selected agents
                    if hasattr(agent, 'internal_states'):
                        writer.add_scalar(f"individual_agents/{agent_id}/internal_state", internal_state, global_step)
                        
                    if hasattr(agent, 'get_current_drive'):
                        writer.add_scalar(f"individual_agents/{agent_id}/drive", drive, global_step)
                        
                    if hasattr(agent, 'perceived_social_norm'):
                        writer.add_scalar(f"individual_agents/{agent_id}/social_norm", social_norm, global_step)
                        
                    if hasattr(agent, 'last_intake'):
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
            
            print("    📊 Starting TensorBoard logging...")
            # Log agent statistics and distributions
            if internal_states:
                print("    📊 Logging internal states...")
                writer.add_scalar("agents/avg_internal_state", np.mean(internal_states), global_step)
                writer.add_scalar("agents/std_internal_state", np.std(internal_states), global_step)
                writer.add_scalar("agents/min_internal_state", np.min(internal_states), global_step)
                writer.add_scalar("agents/max_internal_state", np.max(internal_states), global_step)
                print("    📊 Logging internal states histogram...")
                # Add histogram to see distribution
                writer.add_histogram("distributions/internal_states", np.array(internal_states), global_step)
                print("    ✅ Internal states logged")
                
            if drives:
                print("    📊 Logging drives...")
                writer.add_scalar("agents/avg_drive", np.mean(drives), global_step)
                writer.add_scalar("agents/std_drive", np.std(drives), global_step)
                writer.add_scalar("agents/max_drive", np.max(drives), global_step)
                # Add histogram to see distribution
                writer.add_histogram("distributions/drives", np.array(drives), global_step)
                print("    ✅ Drives logged")
                
            if social_norms:
                print("    📊 Logging social norms...")
                writer.add_scalar("agents/avg_social_norm", np.mean(social_norms), global_step)
                writer.add_scalar("agents/std_social_norm", np.std(social_norms), global_step)
                # Add histogram to see distribution
                writer.add_histogram("distributions/social_norms", np.array(social_norms), global_step)
                print("    ✅ Social norms logged")
                
            if last_intakes:
                print("    📊 Logging consumption...")
                writer.add_scalar("agents/avg_consumption", np.mean(last_intakes), global_step)
                writer.add_scalar("agents/total_consumption", np.sum(last_intakes), global_step)
                # Add histogram to see distribution
                writer.add_histogram("distributions/consumptions", np.array(last_intakes), global_step)
                print("    ✅ Consumption logged")
                
            print("    📊 Logging population metrics...")        
            # Log population metrics
            writer.add_scalar("population/alive_agents", len(agents), global_step)
            print("    ✅ Population metrics logged")
            
            print("    📊 Logging cooperation metrics...")
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
            print("    ✅ Cooperation metrics logged")
            
        print("    📊 Logging resource information...")
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
        print("    ✅ Resource information logged")
            
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
            
            # Show all agents if 5 or fewer, otherwise sample
            if len(agents) <= 5:
                agent_sample = list(agents.items())
                print(f"📊 All agents:")
            else:
                agent_sample = list(agents.items())[:3]
                print(f"📊 Sample of agents (first 3):")
            
            for agent_id, agent in agent_sample:
                internal_state = agent.internal_states[0] if hasattr(agent, 'internal_states') else "N/A"
                drive = agent.get_current_drive() if hasattr(agent, 'get_current_drive') else "N/A"
                social_norm = agent.perceived_social_norm[0] if hasattr(agent, 'perceived_social_norm') and len(agent.perceived_social_norm) > 0 else "N/A"
                last_intake = agent.last_intake[0] if hasattr(agent, 'last_intake') and len(agent.last_intake) > 0 else "N/A"
                position = getattr(agent, 'position', "N/A")
                
                # Calculate derived metrics
                if internal_state != "N/A":
                    balance = 1.0 - abs(internal_state)
                    urgency = drive * abs(internal_state) if drive != "N/A" else "N/A"
                    status = "🟢" if balance > 0.7 else "🟡" if balance > 0.4 else "🔴"
                else:
                    balance = "N/A"
                    urgency = "N/A"
                    status = "❓"
                
                print(f"  {status} {agent_id}: state={internal_state:.3f}, drive={drive:.3f}, norm={social_norm:.3f}")
                print(f"      intake={last_intake:.3f}, pos={position}, balance={balance:.3f}, urgency={urgency:.3f}")
            
            # Show summary statistics
            if len(agents) > 5:
                all_states = [agent.internal_states[0] for agent in agents.values() if hasattr(agent, 'internal_states')]
                all_drives = [agent.get_current_drive() for agent in agents.values() if hasattr(agent, 'get_current_drive')]
                
                if all_states:
                    print(f"📈 Population stats:")
                    print(f"   Internal states: avg={np.mean(all_states):.3f}, std={np.std(all_states):.3f}, range=[{np.min(all_states):.3f}, {np.max(all_states):.3f}]")
                if all_drives:
                    print(f"   Drives: avg={np.mean(all_drives):.3f}, std={np.std(all_drives):.3f}, max={np.max(all_drives):.3f}")
        
        print(f"💰 Avg Reward: {avg_reward:.3f}")
        
    except Exception as e:
        print(f"Warning: Could not print detailed info: {e}")


def train_mappo_simple(args):
    """Simplified MAPPO training (ordem corrigida para logging inicial)."""

    print("🚀 Starting Simple MAPPO Training")
    print(f"Device: {args.device} | Agents: {args.n_agents} | Initial Resources: {args.initial_resource_stock}")
    print(f"Log Directory: {args.log_dir}")

    # Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # ====== ENV ======
    base_env = create_env(
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
    )
    env = aec_to_parallel(base_env)

    # ====== LOGGING (primeiro!) ======
    run_name = f"simple_ippo_{int(time.time())}"
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(f"{args.log_dir}/{run_name}")
    global_step = 0

    print(f"📊 Logs em: {args.log_dir}/{run_name}")
    print(f"➡️  tensorboard --logdir {args.log_dir}")

    # ====== RESET + LOG INICIAL ======
    sample_obs_dict, _ = env.reset()  # (observations, infos)

    # baseline para depletion_rate
    try:
        if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "resource_stock"):
            env.unwrapped._last_resource_stock = float(env.unwrapped.resource_stock[0])
    except Exception:
        pass

    # log do estado inicial (estoque começa no valor real no TB)
    _log_detailed_metrics(env, writer, global_step=global_step, args=args)

    # ====== CHECAGENS ======
    if not sample_obs_dict:
        raise ValueError("Environment returned empty observations. Check environment setup.")
    if not env.agents:
        raise ValueError("No agents found in environment.")

    sample_obs = next(iter(sample_obs_dict.values()))
    obs_processor = ObservationProcessor(sample_obs)

    print(f"Observation dim: {obs_processor.obs_dim}")
    if isinstance(sample_obs, dict):
        print(f"Observation keys: {list(sample_obs.keys())}")

    action_dim = env.action_space(env.agents[0]).n
    print(f"Action dim: {action_dim} | Active agents: {len(env.agents)}")

    # ====== AGENTE ======
    agent = SimpleAgent(obs_processor.obs_dim, action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)

    # ====== TREINAMENTO ======
    num_iterations = args.total_timesteps // args.num_steps
    print(f"Training for {num_iterations} iterations (~{args.num_steps * args.n_agents} samples/iter)")

    for iteration in range(num_iterations):
        it_start = time.time()
        print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")

        # Coleta
        observations, actions, log_probs, rewards, values = collect_rollout(
            env, agent, obs_processor, args, device
        )
        if not observations:
            print("⚠️ No observations collected; resetting env and skipping.")
            env.reset()
            continue

        # Preparação dos dados
        all_obs, all_actions, all_log_probs, all_rewards, all_values = [], [], [], [], []
        homeo_rewards, social_costs = [], []

        for step_obs, step_actions, step_log_probs, step_rewards, step_values in zip(
            observations, actions, log_probs, rewards, values
        ):
            for aid in step_obs:
                if (aid in step_actions and aid in step_log_probs
                        and aid in step_rewards and aid in step_values):
                    all_obs.append(step_obs[aid])
                    all_actions.append(step_actions[aid])
                    all_log_probs.append(step_log_probs[aid])
                    all_rewards.append(step_rewards[aid])
                    all_values.append(step_values[aid])

                    r = step_rewards[aid]
                    homeo_rewards.append(r)
                    social_costs.append(abs(r))

        npts = len(all_obs)
        print(f"📊 Data points: {npts}")
        if npts < 32:
            print("⚠️ Few points (<32). Likely early termination. Skipping updates.")
            continue

        # Vantagens e retornos (simples)
        advantages = []
        returns = []
        for r, v in zip(all_rewards, all_values):
            advantages.append(r - v)
            returns.append(r)
        # Tensores
        obs_tensor = torch.as_tensor(np.array(all_obs), dtype=torch.float32, device=device)
        actions_tensor = torch.as_tensor(all_actions, dtype=torch.long, device=device)
        old_log_probs = torch.as_tensor(all_log_probs, dtype=torch.float32, device=device)
        advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)

        if advantages_tensor.std() > 1e-8:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # Atualizações PPO
        for epoch in range(args.update_epochs):
            try:
                _, new_log_probs, entropy, values_pred = agent.get_action_and_value(obs_tensor, actions_tensor)
            except Exception as e:
                print(f"❌ Error on forward pass: {e}")
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
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

        # Métricas de treino
        total_reward = sum(all_rewards) if all_rewards else 0.0
        avg_reward = total_reward / len(all_rewards) if all_rewards else 0.0
        global_step += npts

        writer.add_scalar("train/policy_loss", float(policy_loss.item()), global_step)
        writer.add_scalar("train/value_loss", float(value_loss.item()), global_step)
        writer.add_scalar("train/entropy", float(entropy_loss.item()), global_step)
        writer.add_scalar("train/total_loss", float(loss.item()), global_step)
        writer.add_scalar("train/avg_reward", float(avg_reward), global_step)
        writer.add_scalar("train/data_points", int(npts), global_step)

        if homeo_rewards:
            writer.add_scalar("rewards/avg_homeostatic", float(np.mean(homeo_rewards)), global_step)
            writer.add_scalar("rewards/std_homeostatic", float(np.std(homeo_rewards)), global_step)
        if social_costs:
            writer.add_scalar("rewards/avg_social_cost", float(np.mean(social_costs)), global_step)
            writer.add_scalar("rewards/std_social_cost", float(np.std(social_costs)), global_step)

        # Log detalhado do ambiente (inclusive recursos)
        try:
            _log_detailed_metrics(env, writer, global_step, args)
        except Exception as e:
            print(f"⚠️ Detailed metrics failed: {e}")

        # Opcional: debug textual periódico
        if args.verbose_logging or (iteration + 1) % 10 == 0:
            _print_detailed_info(env, iteration + 1, avg_reward)

        # Save periódico
        if (iteration + 1) % 50 == 0:
            os.makedirs("models", exist_ok=True)
            torch.save(
                {
                    "model_state_dict": agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "iteration": iteration,
                    "args": args,
                },
                f"models/simple_mappo_iter_{iteration + 1}.pt",
            )
            print(f"💾 Model saved at iteration {iteration + 1}")

        print(f"✅ Iter {iteration + 1} done in {time.time() - it_start:.2f}s")

    # Fim do treino
    os.makedirs("models", exist_ok=True)
    final_model_path = f"models/simple_mappo_final_{run_name}.pt"
    torch.save(
        {
            "model_state_dict": agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": num_iterations,
            "args": args,
        },
        final_model_path,
    )
    print(f"\n✅ Training completed! Final model: {final_model_path}")
    print(f"📈 View logs: tensorboard --logdir {args.log_dir}")

    env.close()
    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config/config.yaml")
    parser.add_argument("--n_agents", type=int, default=10)
    parser.add_argument("--initial_resource_stock", type=float, default=2.0)
    parser.add_argument("--total_timesteps", type=int, default=100_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log_dir", type=str, default="mappo_logs", help="Directory to save logs")
    parser.add_argument("--verbose", action="store_true", help="Show detailed info every iteration")
    parser.add_argument("--no-individual-tracking", action="store_true", help="Disable individual agent tracking in TensorBoard")
    parser.add_argument("--max-agents-tracked", type=int, default=10, help="Maximum number of agents to track individually")
    
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
        verbose_logging=args.verbose,
        track_individual_agents=not args.no_individual_tracking,
        max_agents_to_track=args.max_agents_tracked
    )
    
    train_mappo_simple(train_args)


if __name__ == "__main__":
    # Examples of usage:
    # python -m src.scripts.pettingzoo_env.train_mappo_simple --total_timesteps 10000 --n_agents 5
    # python -m src.scripts.pettingzoo_env.train_mappo_simple --log_dir "experimento_1" --seed 42
    # python -m src.scripts.pettingzoo_env.train_mappo_simple --log_dir "teste_cooperacao" --n_agents 8 --verbose
    # python -m src.scripts.pettingzoo_env.train_mappo_simple --log_dir "debug" --verbose --total_timesteps 5000
    # python -m src.scripts.pettingzoo_env.train_mappo_simple --log_dir "individual_analysis" --max-agents-tracked 5
    # python -m src.scripts.pettingzoo_env.train_mappo_simple --log_dir "aggregate_only" --no-individual-tracking
    main() 
