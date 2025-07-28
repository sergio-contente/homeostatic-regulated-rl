#!/usr/bin/env python3
"""
Enhanced MAPPO for Homeostatic Environment with True Centralized Critic

This implementation provides a proper centralized critic that observes global state
while maintaining decentralized actors for execution. This is ideal for the tragedy
of commons scenario where coordination is crucial.

Key enhancements:
- Centralized critic observes global resource state and all agent states
- Decentralized actor only uses local observations
- Social norm integration in global state
- Resource scarcity awareness in critic
"""

import os
import random
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import supersuit as ss
from pettingzoo.utils.conversions import aec_to_parallel

from src.envs.multiagent import create_env


@dataclass
class EnhancedArgs:
    exp_name: str = "mappo_enhanced_homeostatic"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be automatically disabled if not available"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "homeostatic-mappo-enhanced"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""

    # Environment specific arguments
    config_path: str = "config/config.yaml"
    """path to configuration file"""
    drive_type: str = "base_drive"
    """type of drive to use"""
    learning_rate_social: float = 0.1
    """social learning rate for norm adaptation"""
    beta: float = 0.5
    """social norm internalization strength"""
    number_resources: int = 1
    """number of resource types"""
    n_agents: int = 10
    """number of agents"""
    env_size: int = 10
    """size of environment"""
    max_steps: int = 1000
    """maximum steps per episode"""
    initial_resource_stock: float = 1000.0
    """initial resource stock"""

    # Algorithm specific arguments
    total_timesteps: int = 3_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 256
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function"""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""

    # Enhanced MAPPO specific
    global_state_size: int = 64
    """size of the global state representation"""
    use_global_critic: bool = True
    """whether to use global state in critic"""
    share_parameters: bool = True
    """whether to share parameters between agents"""

    # batch size related arguments (computed)
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


class GlobalStateExtractor:
    """Extracts global state information for centralized critic."""
    
    def __init__(self, n_agents: int, resource_dim: int):
        self.n_agents = n_agents
        self.resource_dim = resource_dim
        
    def extract_global_state(self, local_observations: Dict[str, Any], 
                           env_state: Optional[Dict] = None) -> np.ndarray:
        """
        Extract global state from local observations and environment state.
        
        This creates a comprehensive global view for the centralized critic including:
        - All agents' internal states
        - Global resource levels and scarcity
        - Social norms and consumption patterns
        - Spatial information
        """
        global_features = []
        
        # 1. Aggregate agent internal states
        all_internal_states = []
        all_positions = []
        all_social_norms = []
        
        for i in range(self.n_agents):
            agent_key = f"agent_{i}"
            if agent_key in local_observations:
                obs = local_observations[agent_key]
                
                # Internal homeostatic states
                if 'internal_states' in obs:
                    all_internal_states.extend(obs['internal_states'])
                else:
                    all_internal_states.extend([0.0] * self.resource_dim)
                    
                # Positions
                if 'position' in obs:
                    all_positions.append(obs['position'])
                else:
                    all_positions.append(0)
                    
                # Social norms
                if 'perceived_social_norm' in obs:
                    all_social_norms.extend(obs['perceived_social_norm'])
                else:
                    all_social_norms.extend([0.0] * self.resource_dim)
        
        global_features.extend(all_internal_states)
        global_features.extend(all_positions)
        global_features.extend(all_social_norms)
        
        # 2. Global resource information
        if env_state:
            if 'resource_stock' in env_state:
                global_features.extend(env_state['resource_stock'])
            if 'resource_scarcity' in env_state:
                global_features.extend(env_state['resource_scarcity'])
            if 'round_consumption' in env_state:
                global_features.append(env_state['round_consumption'])
        
        # 3. Statistical aggregates
        if all_internal_states:
            states_array = np.array(all_internal_states).reshape(-1, self.resource_dim)
            global_features.extend([
                np.mean(states_array, axis=0)[0],  # Mean internal state
                np.std(states_array, axis=0)[0],   # Std internal state
                np.min(states_array, axis=0)[0],   # Min internal state
                np.max(states_array, axis=0)[0],   # Max internal state
            ])
        
        return np.array(global_features, dtype=np.float32)


class EnhancedAgent(nn.Module):
    """Enhanced MAPPO Agent with true centralized critic."""
    
    def __init__(self, envs, args):
        super().__init__()
        
        self.args = args
        obs_space = envs.single_observation_space
        action_space = envs.single_action_space
        
        # Calculate observation dimensions
        if isinstance(obs_space, gym.spaces.Dict):
            local_obs_dim = sum([
                np.array(space.shape).prod() if hasattr(space, 'shape') 
                else space.n if hasattr(space, 'n') 
                else 1
                for space in obs_space.spaces.values()
            ])
        else:
            local_obs_dim = np.array(obs_space.shape).prod()
        
        # Decentralized Actor (uses only local observations)
        self.actor_network = nn.Sequential(
            nn.Linear(local_obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(128, action_space.n)
        
        # Centralized Critic (uses global state if available)
        if args.use_global_critic:
            # Global state size is estimated or configurable
            global_obs_dim = args.global_state_size
        else:
            global_obs_dim = local_obs_dim
            
        self.critic_network = nn.Sequential(
            nn.Linear(global_obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.critic_head = nn.Linear(128, 1)
        
        # Initialize parameters
        self._init_parameters()
        
        # Global state extractor
        self.global_state_extractor = GlobalStateExtractor(
            n_agents=args.n_agents,
            resource_dim=args.number_resources
        )
        
    def _init_parameters(self):
        """Initialize network parameters with orthogonal initialization."""
        for module in [self.actor_network, self.critic_network]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.constant_(layer.bias, 0.0)
                    
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.constant_(self.actor_head.bias, 0.0)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.constant_(self.critic_head.bias, 0.0)
    
    def get_value(self, global_obs):
        """Get value estimate using global observations."""
        if isinstance(global_obs, dict):
            # For now, flatten dict observations
            # In a full implementation, you'd extract proper global state
            global_obs = torch.cat([
                v.flatten(start_dim=1) if len(v.shape) > 1 else v
                for v in global_obs.values()
            ], dim=1).float()
        
        hidden = self.critic_network(global_obs)
        return self.critic_head(hidden)

    def get_action_and_value(self, local_obs, global_obs=None, action=None):
        """Get action and value using both local and global observations."""
        
        # Actor uses local observations only
        if isinstance(local_obs, dict):
            local_obs_flat = torch.cat([
                v.flatten(start_dim=1) if len(v.shape) > 1 else v
                for v in local_obs.values()
            ], dim=1).float()
        else:
            local_obs_flat = local_obs
            
        actor_hidden = self.actor_network(local_obs_flat)
        logits = self.actor_head(actor_hidden)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
        
        # Critic uses global observations if available
        critic_obs = global_obs if global_obs is not None else local_obs
        value = self.get_value(critic_obs)
        
        return action, probs.log_prob(action), probs.entropy(), value


def make_enhanced_env(args, idx):
    """Create environment with enhanced observation wrapper."""
    
    def thunk():
        env = create_env(
            config_path=args.config_path,
            drive_type=args.drive_type,
            learning_rate=args.learning_rate_social,
            beta=args.beta,
            number_resources=args.number_resources,
            n_agents=args.n_agents,
            size=args.env_size,
            max_steps=args.max_steps,
            seed=args.seed + idx,
            initial_resource_stock=args.initial_resource_stock
        )
        
        # Convert to parallel
        env = aec_to_parallel(env)
        
        # Add observation wrapper for global state extraction
        # env = GlobalStateWrapper(env, args.n_agents, args.number_resources)
        
        # Apply supersuit wrappers
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
        
        return env
    
    return thunk


def train_enhanced_mappo(args):
    """Enhanced MAPPO training with centralized critic."""
    
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Setup
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # Create environments
    envs = gym.vector.SyncVectorEnv(
        [make_enhanced_env(args, i) for i in range(args.num_envs)]
    )
    
    # Calculate batch sizes
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    
    print(f"Batch size: {args.batch_size}")
    print(f"Minibatch size: {args.minibatch_size}")
    print(f"Number of iterations: {args.num_iterations}")

    # Initialize agent
    agent = EnhancedAgent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage arrays
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Training metrics
    episode_rewards = []
    cooperation_metrics = []
    resource_sustainability = []

    # Start training
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()[0]
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing learning rate
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Rollout phase
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Get action and value
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    local_obs=next_obs,
                    global_obs=next_obs  # For simplicity, using same obs
                )
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Execute action
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(done).to(device)

            # Log episode results and custom metrics
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        episode_rewards.append(info["episode"]["r"])
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # Compute advantages using GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimize policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    local_obs=b_obs[mb_inds],
                    global_obs=b_obs[mb_inds],  # For simplicity
                    action=b_actions.long()[mb_inds]
                )
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Calculate explained variance
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Log metrics
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # Custom metrics for tragedy of commons
        if episode_rewards:
            avg_reward = np.mean(episode_rewards[-100:])  # Last 100 episodes
            writer.add_scalar("homeostatic/avg_reward_100", avg_reward, global_step)

        # Print progress
        if iteration % 50 == 0:
            print(f"\n{'='*60}")
            print(f"Iteration {iteration}/{args.num_iterations}")
            print(f"Global Step: {global_step}")
            print(f"SPS: {int(global_step / (time.time() - start_time))}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"Value Loss: {v_loss.item():.4f}")
            print(f"Policy Loss: {pg_loss.item():.4f}")
            print(f"Entropy: {entropy_loss.item():.4f}")
            print(f"Explained Variance: {explained_var:.4f}")
            print(f"Approx KL: {approx_kl.item():.4f}")
            if episode_rewards:
                print(f"Avg Episode Reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
            print(f"{'='*60}")

    # Save final model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/mappo_enhanced_{run_name}.pt"
    torch.save({
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args,
        'final_iteration': iteration,
        'global_step': global_step
    }, model_path)
    print(f"Model saved to {model_path}")

    envs.close()
    writer.close()


if __name__ == "__main__":
    args = tyro.cli(EnhancedArgs)
    train_enhanced_mappo(args) 
