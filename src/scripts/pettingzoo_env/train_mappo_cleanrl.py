#!/usr/bin/env python3
"""
MAPPO Training Script for Homeostatic Environment using CleanRL

This script implements Multi-Agent PPO (MAPPO) for the tragedy of commons problem
using CleanRL's implementation which has excellent PettingZoo support.

Key features:
- Centralized Training Decentralized Execution (CTDE)
- Global state observation for centralized critic
- Compatible with homeostatic social learning
"""

import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# Supersuit for PettingZoo preprocessing
import supersuit as ss
from pettingzoo.utils.conversions import aec_to_parallel

# Your environment
from src.envs.multiagent import create_env


@dataclass
class Args:
    exp_name: str = "mappo_homeostatic"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be automatically disabled if not available"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "homeostatic-mappo"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

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
    total_timesteps: int = 2_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""

    # batch size related arguments
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(args, idx):
    """Create and wrap environment for training."""
    
    def thunk():
        # Create the base environment
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
        
        # Convert to parallel for easier handling
        env = aec_to_parallel(env)
        
        # Apply supersuit wrappers for better training
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
        
        return env
    
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize layer with orthogonal weights."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """MAPPO Agent with centralized critic and decentralized actor."""
    
    def __init__(self, envs, agent_id=0):
        super().__init__()
        
        # Get observation and action spaces
        obs_space = envs.single_observation_space
        action_space = envs.single_action_space
        
        # Shared feature extractor
        if isinstance(obs_space, gym.spaces.Dict):
            # Handle dict observation space (which your env uses)
            obs_dim = sum([
                np.array(space.shape).prod() if hasattr(space, 'shape') 
                else space.n if hasattr(space, 'n') 
                else 1
                for space in obs_space.spaces.values()
            ])
        else:
            obs_dim = np.array(obs_space.shape).prod()
            
        self.shared_network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
        )
        
        # Actor network (decentralized)
        self.actor = layer_init(nn.Linear(128, action_space.n), std=0.01)
        
        # Critic network (will use global state)
        self.critic = layer_init(nn.Linear(128, 1), std=1.0)
        
    def get_value(self, x):
        """Get value estimate."""
        if isinstance(x, dict):
            # Flatten dict observations
            x = torch.cat([
                v.flatten(start_dim=1) if len(v.shape) > 1 else v
                for v in x.values()
            ], dim=1).float()
        
        hidden = self.shared_network(x)
        return self.critic(hidden)

    def get_action_and_value(self, x, action=None):
        """Get action distribution and value estimate."""
        if isinstance(x, dict):
            # Flatten dict observations  
            x = torch.cat([
                v.flatten(start_dim=1) if len(v.shape) > 1 else v
                for v in x.values()
            ], dim=1).float()
            
        hidden = self.shared_network(x)
        
        # Actor forward pass
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


def train_mappo(args):
    """Main MAPPO training function."""
    
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
        [make_env(args, i) for i in range(args.num_envs)]
    )
    
    # Calculate batch sizes
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    
    print(f"Batch size: {args.batch_size}")
    print(f"Minibatch size: {args.minibatch_size}")
    print(f"Number of iterations: {args.num_iterations}")

    # Initialize agent
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage arrays
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

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
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Execute action in environment
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(done).to(device)

            # Log episode results
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # Bootstrap value if not done
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

        # Flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # Calculate approximate KL divergence
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

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Log training metrics
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # Print progress
        if iteration % 10 == 0:
            print(f"Iteration {iteration}/{args.num_iterations}")
            print(f"SPS: {int(global_step / (time.time() - start_time))}")
            print(f"Value Loss: {v_loss.item():.4f}")
            print(f"Policy Loss: {pg_loss.item():.4f}")
            print(f"Explained Variance: {explained_var:.4f}")
            print(f"Approx KL: {approx_kl.item():.4f}")
            print("="*50)

    # Save final model
    os.makedirs("models", exist_ok=True)
    torch.save(agent.state_dict(), f"models/mappo_{run_name}.pt")
    print(f"Model saved to models/mappo_{run_name}.pt")

    envs.close()
    writer.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    train_mappo(args) 
