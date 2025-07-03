"""Advanced PPO implementation for NORMARL environment adapted from CleanRL.

This is a full training script including CLI, logging and integration with TensorBoard 
and WandB for experiment tracking, specifically adapted for the NORMARL homeostatic environment.

Authors: Adapted from CleanRL by Costa and Elliot
"""

import argparse
import os
import random
import time
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import sys

# Add src to path to import NORMARL environment
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.pettingzoo_env.normarl import NormalHomeostaticEnv


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="normarl-ppo",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances")

    # NORMARL Environment specific arguments
    parser.add_argument("--config-path", type=str, default="config/config.yaml",
        help="path to the NORMARL configuration file")
    parser.add_argument("--drive-type", type=str, default="base_drive",
        help="type of drive system to use")
    parser.add_argument("--learning-rate", type=float, default=0.1,
        help="social learning rate for agents")
    parser.add_argument("--beta", type=float, default=0.5,
        help="social cost coefficient")
    parser.add_argument("--number-resources", type=int, default=1,
        help="number of resources in the environment")
    parser.add_argument("--n-agents", type=int, default=5,
        help="number of agents in the environment")
    parser.add_argument("--size", type=int, default=10,
        help="size of the environment grid")

    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=100000,
        help="total timesteps of the experiments")
    parser.add_argument("--lr", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=1000,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps * args.n_agents)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


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
            layer_init(nn.Linear(total_obs_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
        )
        
        # Actor and critic heads
        self.actor = layer_init(nn.Linear(128, action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1))

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
    args = parse_args()
    print(args)
    run_name = f"normarl__{args.exp_name}__{args.seed}__{int(time.time())}"
    
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
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Environment setup
    env = NormalHomeostaticEnv(
        config_path=args.config_path,
        drive_type=args.drive_type,
        learning_rate=args.learning_rate,
        beta=args.beta,
        number_resources=args.number_resources,
        n_agents=args.n_agents,
        size=args.size
    )
    
    # Reset environment to initialize action and observation spaces
    observations, info = env.reset(seed=args.seed)
    
    num_agents = len(env.agents)
    num_actions = env.action_space(env.agents[0]).n
    observation_space = env.observation_space(env.agents[0])

    agent = NORMARLAgent(observation_space, env.action_space(env.agents[0])).to(device)
    # Ensure agent uses float32
    agent = agent.float()
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    # ALGO Logic: Storage setup
    rb_obs_positions = torch.zeros((args.num_steps, num_agents), dtype=torch.long).to(device)
    rb_obs_internal_states = torch.zeros((args.num_steps, num_agents, observation_space["internal_states"].shape[0]), dtype=torch.float32).to(device)
    rb_obs_social_norms = torch.zeros((args.num_steps, num_agents, observation_space["perceived_social_norm"].shape[0]), dtype=torch.float32).to(device)
    
    rb_actions = torch.zeros((args.num_steps, num_agents), dtype=torch.long).to(device)
    rb_logprobs = torch.zeros((args.num_steps, num_agents), dtype=torch.float32).to(device)
    rb_rewards = torch.zeros((args.num_steps, num_agents), dtype=torch.float32).to(device)
    rb_terms = torch.zeros((args.num_steps, num_agents), dtype=torch.float32).to(device)
    rb_values = torch.zeros((args.num_steps, num_agents), dtype=torch.float32).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = observations
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.lr
            optimizer.param_groups[0]["lr"] = lrnow

        with torch.no_grad():
            obs = next_obs
            for step in range(0, args.num_steps):
                global_step += num_agents
                batch_obs = batchify_obs(obs, device)
                actions, logprobs, _, values = agent.get_action_and_value(batch_obs)
                next_obs, rewards, terms, truncs, infos = env.step(unbatchify(actions, env))
                rb_obs_positions[step] = batch_obs["position"]
                rb_obs_internal_states[step] = batch_obs["internal_states"]
                rb_obs_social_norms[step] = batch_obs["perceived_social_norm"]
                rb_rewards[step] = batchify(rewards, device)
                rb_terms[step] = batchify(terms, device)
                rb_actions[step] = actions
                rb_logprobs[step] = logprobs
                rb_values[step] = values.flatten()
                if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                    end_step = step
                    break
                obs = next_obs

        with torch.no_grad():
            rb_advantages = torch.zeros_like(rb_rewards).to(device)
            for t in reversed(range(end_step)):
                delta = (
                    rb_rewards[t]
                    + args.gamma * rb_values[t + 1] * rb_terms[t + 1]
                    - rb_values[t]
                )
                rb_advantages[t] = delta + args.gamma * args.gae_lambda * rb_advantages[t + 1]
            rb_returns = rb_advantages + rb_values

        b_obs_positions = torch.flatten(rb_obs_positions[:end_step], start_dim=0, end_dim=1)
        b_obs_internal_states = torch.flatten(rb_obs_internal_states[:end_step], start_dim=0, end_dim=1)
        b_obs_social_norms = torch.flatten(rb_obs_social_norms[:end_step], start_dim=0, end_dim=1)
        b_logprobs = torch.flatten(rb_logprobs[:end_step], start_dim=0, end_dim=1)
        b_actions = torch.flatten(rb_actions[:end_step], start_dim=0, end_dim=1)
        b_returns = torch.flatten(rb_returns[:end_step], start_dim=0, end_dim=1)
        b_values = torch.flatten(rb_values[:end_step], start_dim=0, end_dim=1)
        b_advantages = torch.flatten(rb_advantages[:end_step], start_dim=0, end_dim=1)

        b_inds = np.arange(len(b_obs_positions))
        clipfracs = []
        minibatch_size = max(1, len(b_obs_positions) // args.num_minibatches)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_obs_positions), minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                batch_obs = {
                    "position": b_obs_positions[mb_inds],
                    "internal_states": b_obs_internal_states[mb_inds],
                    "perceived_social_norm": b_obs_social_norms[mb_inds]
                }
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    batch_obs, b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        avg_reward = rb_rewards[:end_step].mean().item()
        writer.add_scalar("charts/avg_reward", avg_reward, global_step)
        writer.add_scalar("charts/episode_length", end_step, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # --- CUSTOM METRICS ---
        # Para cada agente, loga reward médio, consumo, social cost, estados internos, normas sociais
        for agent_id, ag in env.homeostatic_agents.items():
            # Consumo total do agente
            if hasattr(ag, 'intake_history') and len(ag.intake_history) > 0:
                writer.add_scalar(f"custom/agent_{agent_id}_total_consumption", np.sum(ag.intake_history), global_step)
            # Social cost médio (se você armazenar isso)
            if hasattr(ag, 'social_cost_history') and len(ag.social_cost_history) > 0:
                writer.add_scalar(f"custom/agent_{agent_id}_mean_social_cost", np.mean(ag.social_cost_history), global_step)
            # Estados internos
            for i, value in enumerate(ag.internal_states):
                writer.add_scalar(f"custom/agent_{agent_id}_internal_state_{i}", value, global_step)
            # Normas sociais
            for i, value in enumerate(ag.perceived_social_norm):
                writer.add_scalar(f"custom/agent_{agent_id}_social_norm_{i}", value, global_step)
        # Resource stock global
        writer.add_scalar("custom/final_resource_stock", np.sum(env.resource_stock), global_step)

    torch.save(agent.state_dict(), f"runs/{run_name}/agent.pt")
    print(f"Agent saved to runs/{run_name}/agent.pt")
    writer.close() 
