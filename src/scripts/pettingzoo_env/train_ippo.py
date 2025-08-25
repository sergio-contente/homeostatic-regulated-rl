#!/usr/bin/env python3
"""
Simple ippo Implementation for Homeostatic Environment (with GAE)

- Direct PettingZoo integration (no SuperSuit wrappers)
- Proper use of gamma and GAE(lambda) to avoid myopic learning
- Tracks detailed homeostatic metrics in TensorBoard
- Keeps the actor decentralized; critic stays shared (you can swap to a centralized critic by concatenating global features before the value head if desired)
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

# second: try to change the discount factor dynamically for future rewards (selfish or not)

# do not change the amount of resources, instead change the amount of consumption/intake

# exp1 : regen 0.03, beta 1, stock 2
# exp2 : regen 0.03, beta 0.8, stock 2
# exp3 : regen 0.03, beta 0.6, stock 2
# exp4 : regen 0.03, beta 0.4, stock 2
# exp5 : regen 0.03, beta 0.0, stock 2


@dataclass
class SimpleArgs:
    # Environment
    config_path: str = "config/config.yaml"
    drive_type: str = "base_drive"
    learning_rate_social: float = 0.1
    beta: float = 0.8
    number_resources: int = 1
    n_agents: int = 10
    env_size: int = 1
    max_steps: int = 1000
    initial_resource_stock: float = 1.0

    # Training
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    num_envs: int = 1
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
    log_dir: str = "ippo_logs"  # unified with CLI default
    verbose_logging: bool = False
    track_individual_agents: bool = True
    max_agents_to_track: int = 10


class SimpleAgent(nn.Module):
    """Simplified ippo agent with shared trunk, actor + critic heads."""

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        z = self.shared(x)
        return self.actor(z), self.critic(z)

    def get_action_and_value(self, x, action=None):
        logits, value = self.forward(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        return action, log_prob, entropy, value


class ObservationProcessor:
    """Process observations from PettingZoo environment (dict or array)."""

    def __init__(self, sample_obs):
        if isinstance(sample_obs, dict):
            self.obs_dim = sum([
                np.array(v).size if hasattr(v, 'size') else 1
                for v in sample_obs.values()
            ])
            self.is_dict = True
        else:
            self.obs_dim = np.array(sample_obs).size
            self.is_dict = False

    def process(self, obs):
        if self.is_dict:
            flat = []
            for v in obs.values():
                if hasattr(v, '__len__') and not isinstance(v, (int, float)):
                    flat.extend(np.array(v).flatten())
                else:
                    flat.append(float(v))
            return np.array(flat, dtype=np.float32)
        else:
            return np.array(obs).flatten().astype(np.float32)


def collect_rollout(env, agent, obs_processor, args, device):
    """Collect a rollout from the environment (parallel API).

    Returns lists over time of: observations, actions, log_probs, rewards, values, dones
    where each item is a dict keyed by agent_id.
    """

    observations: List[Dict] = []
    actions: List[Dict] = []
    log_probs: List[Dict] = []
    rewards: List[Dict] = []
    values: List[Dict] = []
    dones: List[Dict] = []

    # Reset environment
    obs_dict, _ = env.reset()

    completed_rounds = 0
    steps_in_current_round = 0

    for step in range(args.num_steps):
        # Build tensors for existing agents
        processed_obs, obs_tensor_dict = {}, {}
        for agent_id in env.agents:
            if agent_id in obs_dict:
                processed = obs_processor.process(obs_dict[agent_id])
                processed_obs[agent_id] = processed
                obs_tensor_dict[agent_id] = torch.as_tensor(processed, dtype=torch.float32, device=device).unsqueeze(0)

        # Get actions
        actions_dict, log_probs_dict, values_dict = {}, {}, {}
        for agent_id, tens in obs_tensor_dict.items():
            with torch.no_grad():
                action, logp, _, val = agent.get_action_and_value(tens)
            actions_dict[agent_id] = int(action.item())
            log_probs_dict[agent_id] = float(logp.item())
            values_dict[agent_id] = float(val.item())

        # Step
        next_obs, rewards_dict, terminations, truncations, infos = env.step(actions_dict)

        # Dones per agent (if agent missing, treat as done this step)
        done_dict = {aid: bool(terminations.get(aid, False) or truncations.get(aid, False))
                     for aid in set(list(rewards_dict.keys()) + list(values_dict.keys()))}

        steps_in_current_round += 1
        if step > 0 and len(env.agents) > 0 and steps_in_current_round >= len(env.agents):
            completed_rounds += 1
            steps_in_current_round = 0
            if completed_rounds % 5 == 0:
                print(f"  📍 Completed {completed_rounds} rounds in rollout (step {step+1}/{args.num_steps})")

        # Store
        observations.append(processed_obs)
        actions.append(actions_dict)
        log_probs.append(log_probs_dict)
        rewards.append(rewards_dict)
        values.append(values_dict)
        dones.append(done_dict)

        obs_dict = next_obs

        # Episode end
        if (all(terminations.values()) if len(terminations) else False) or \
           (all(truncations.values()) if len(truncations) else False) or \
           len(env.agents) == 0:
            print(f"Episode ended at step {step + 1}: {len(env.agents)} agents remaining")
            break

    print(f"🔄 Rollout completed: {len(observations)} steps collected")
    return observations, actions, log_probs, rewards, values, dones


def compute_gae_per_agent(rewards, values, dones, gamma: float, lam: float):
    """Compute GAE(γ, λ) and returns per agent across time.

    Parameters are lists over time of dicts keyed by agent_id.
    Missing agents at a timestep are treated as terminal at that step.
    """
    T = len(rewards)
    all_agents = set().union(*[r.keys() for r in rewards]) if T > 0 else set()

    advantages_time: List[Dict[str, float]] = []
    returns_time: List[Dict[str, float]] = []

    last_adv = {aid: 0.0 for aid in all_agents}
    last_val_next = {aid: 0.0 for aid in all_agents}  # will carry v_{t+1} when walking backwards

    for t in reversed(range(T)):
        adv_t, ret_t = {}, {}
        for aid in all_agents:
            r = rewards[t].get(aid, 0.0)
            v = values[t].get(aid, 0.0)
            done = dones[t].get(aid, True)  # if not present, mark as terminal

            v_next = 0.0 if done else last_val_next[aid]
            delta = r + gamma * v_next * (1.0 - float(done)) - v
            last_adv[aid] = delta + gamma * lam * (1.0 - float(done)) * last_adv[aid]

            adv_t[aid] = last_adv[aid]
            ret_t[aid] = v + adv_t[aid]

            # prepare for previous time index: current v becomes next step's v
            last_val_next[aid] = v

        advantages_time.append(adv_t)
        returns_time.append(ret_t)

    advantages_time.reverse()
    returns_time.reverse()
    return advantages_time, returns_time


# ====== LOGGING HELPERS (unchanged except small safety tweaks) ======

def _log_detailed_metrics(env, writer, global_step, args=None):
    try:
        print("    📊 Accessing base environment...")
        base_env = env
        max_unwrap_depth = 10
        unwrap_count = 0
        print(f"    📊 Starting unwrapping, env type: {type(base_env)}")
        while hasattr(base_env, 'unwrapped') and unwrap_count < max_unwrap_depth:
            next_env = base_env.unwrapped
            if next_env is base_env:
                break
            base_env = next_env
            unwrap_count += 1
        if hasattr(base_env, 'env'):
            base_env = base_env.env

        if not hasattr(base_env, 'homeostatic_agents'):
            print(f"    ⚠️ No homeostatic_agents found in {type(base_env)}")
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
                else:
                    internal_state = 0.0

                if hasattr(agent, 'get_current_drive'):
                    drive = agent.get_current_drive()
                    drives.append(drive)
                else:
                    drive = 0.0

                if hasattr(agent, 'perceived_social_norm') and len(agent.perceived_social_norm) > 0:
                    social_norm = agent.perceived_social_norm[0]
                    social_norms.append(social_norm)
                else:
                    social_norm = 0.0

                if hasattr(agent, 'last_intake') and len(agent.last_intake) > 0:
                    intake = agent.last_intake[0]
                    last_intakes.append(intake)
                else:
                    intake = 0.0

                if track_individual and (agent_id, agent) in agents_to_track:
                    writer.add_scalar(f"individual_agents/{agent_id}/internal_state", internal_state, global_step)
                    writer.add_scalar(f"individual_agents/{agent_id}/drive", drive, global_step)
                    writer.add_scalar(f"individual_agents/{agent_id}/social_norm", social_norm, global_step)
                    writer.add_scalar(f"individual_agents/{agent_id}/intake", intake, global_step)
                    if hasattr(agent, 'position'):
                        writer.add_scalar(f"individual_agents/{agent_id}/position", getattr(agent, 'position'), global_step)
                    balance = 1.0 - abs(internal_state)
                    urgency = drive * abs(internal_state)
                    writer.add_scalar(f"individual_agents/{agent_id}/homeostatic_balance", balance, global_step)
                    writer.add_scalar(f"individual_agents/{agent_id}/urgency", urgency, global_step)

            if internal_states:
                writer.add_scalar("agents/avg_internal_state", float(np.mean(internal_states)), global_step)
                writer.add_scalar("agents/std_internal_state", float(np.std(internal_states)), global_step)
                writer.add_scalar("agents/min_internal_state", float(np.min(internal_states)), global_step)
                writer.add_scalar("agents/max_internal_state", float(np.max(internal_states)), global_step)
                writer.add_histogram("distributions/internal_states", np.array(internal_states), global_step)

            if drives:
                writer.add_scalar("agents/avg_drive", float(np.mean(drives)), global_step)
                writer.add_scalar("agents/std_drive", float(np.std(drives)), global_step)
                writer.add_scalar("agents/max_drive", float(np.max(drives)), global_step)
                writer.add_histogram("distributions/drives", np.array(drives), global_step)

            if social_norms:
                writer.add_scalar("agents/avg_social_norm", float(np.mean(social_norms)), global_step)
                writer.add_scalar("agents/std_social_norm", float(np.std(social_norms)), global_step)
                writer.add_histogram("distributions/social_norms", np.array(social_norms), global_step)

            if last_intakes:
                writer.add_scalar("agents/avg_consumption", float(np.mean(last_intakes)), global_step)
                writer.add_scalar("agents/total_consumption", float(np.sum(last_intakes)), global_step)
                writer.add_histogram("distributions/consumptions", np.array(last_intakes), global_step)

            writer.add_scalar("population/alive_agents", int(len(agents)), global_step)

            if last_intakes and internal_states:
                non_critical = [i for i, s in enumerate(internal_states) if s > -0.5]
                if non_critical:
                    non_critical_consumption = [last_intakes[i] for i in non_critical]
                    cooperation_index = 1.0 - (float(np.mean(non_critical_consumption)) / 0.1)
                    writer.add_scalar("cooperation/cooperation_index", max(0.0, cooperation_index), global_step)

                if resource_stock is not None:
                    initial_stock = getattr(env.unwrapped, 'initial_resource_stock', [1000])[0]
                    resource_ratio = resource_stock[0] / initial_stock
                    population_ratio = len(agents) / 10
                    sustainability = (resource_ratio + population_ratio) / 2
                    writer.add_scalar("cooperation/sustainability_index", float(sustainability), global_step)

        if resource_stock is not None:
            writer.add_scalar("resources/stock_food", float(resource_stock[0]), global_step)
            initial_stock = getattr(env.unwrapped, 'initial_resource_stock', [1000])[0]
            writer.add_scalar("resources/stock_percentage", float(resource_stock[0] / initial_stock * 100.0), global_step)
            if hasattr(env.unwrapped, '_last_resource_stock'):
                depletion_rate = env.unwrapped._last_resource_stock - float(resource_stock[0])
                writer.add_scalar("resources/depletion_rate", float(depletion_rate), global_step)
            env.unwrapped._last_resource_stock = float(resource_stock[0])

    except Exception as e:
        print(f"Warning: Could not log detailed metrics: {e}")


def _print_detailed_info(env, iteration, avg_reward):
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
            if len(agents) <= 5:
                agent_sample = list(agents.items())
                print(f"📊 All agents:")
            else:
                agent_sample = list(agents.items())[:3]
                print(f"📊 Sample of agents (first 3):")
            for agent_id, agent in agent_sample:
                internal_state = agent.internal_states[0] if hasattr(agent, 'internal_states') else 0.0
                drive = agent.get_current_drive() if hasattr(agent, 'get_current_drive') else 0.0
                social_norm = agent.perceived_social_norm[0] if hasattr(agent, 'perceived_social_norm') and len(agent.perceived_social_norm) > 0 else 0.0
                last_intake = agent.last_intake[0] if hasattr(agent, 'last_intake') and len(agent.last_intake) > 0 else 0.0
                position = getattr(agent, 'position', "N/A")
                balance = 1.0 - abs(internal_state)
                urgency = drive * abs(internal_state)
                status = "🟢" if balance > 0.7 else ("🟡" if balance > 0.4 else "🔴")
                print(f"  {status} {agent_id}: state={internal_state:.3f}, drive={drive:.3f}, norm={social_norm:.3f}")
                print(f"      intake={last_intake:.3f}, pos={position}, balance={balance:.3f}, urgency={urgency:.3f}")
            if len(agents) > 5:
                all_states = [a.internal_states[0] for a in agents.values() if hasattr(a, 'internal_states')]
                all_drives = [a.get_current_drive() for a in agents.values() if hasattr(a, 'get_current_drive')]
                if all_states:
                    print(f"📈 Population stats:")
                    print(f"   Internal states: avg={np.mean(all_states):.3f}, std={np.std(all_states):.3f}, range=[{np.min(all_states):.3f}, {np.max(all_states):.3f}]")
                if all_drives:
                    print(f"   Drives: avg={np.mean(all_drives):.3f}, std={np.std(all_drives):.3f}, max={np.max(all_drives):.3f}")
        print(f"💰 Avg Reward: {avg_reward:.3f}")
    except Exception as e:
        print(f"Warning: Could not print detailed info: {e}")


def train_ippo_simple(args: SimpleArgs):
    print("🚀 Starting Simple ippo Training")
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

    # ====== LOGGING ======
    run_name = f"simple_ippo_{int(time.time())}"
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(f"{args.log_dir}/{run_name}")
    global_step = 0

    print(f"📊 Logs em: {args.log_dir}/{run_name}")
    print(f"➡️  tensorboard --logdir {args.log_dir}")

    # ====== RESET + INITIAL LOG ======
    sample_obs_dict, _ = env.reset()
    try:
        if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "resource_stock"):
            env.unwrapped._last_resource_stock = float(env.unwrapped.resource_stock[0])
    except Exception:
        pass
    _log_detailed_metrics(env, writer, global_step=global_step, args=args)

    # ====== CHECKS ======
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

    # ====== AGENT ======
    agent = SimpleAgent(obs_processor.obs_dim, action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)

    # ====== TRAINING LOOP ======
    num_iterations = max(1, args.total_timesteps // max(1, args.num_steps))
    print(f"Training for {num_iterations} iterations (~{args.num_steps * args.n_agents} samples/iter)")

    for iteration in range(num_iterations):
        it_start = time.time()
        print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")

        observations, actions, log_probs, rewards, values, dones = collect_rollout(
            env, agent, obs_processor, args, device
        )
        if not observations:
            print("⚠️ No observations collected; resetting env and skipping.")
            env.reset()
            continue

        # GAE(γ, λ)
        advantages_time, returns_time = compute_gae_per_agent(
            rewards, values, dones, args.gamma, args.gae_lambda
        )

        # Flatten step-agent dictionaries into arrays
        all_obs, all_actions, all_log_probs, all_advs, all_rets = [], [], [], [], []
        for step_obs, step_actions, step_log_probs, adv_t, ret_t in zip(
            observations, actions, log_probs, advantages_time, returns_time
        ):
            for aid in step_obs:
                if (aid in step_actions and aid in step_log_probs and aid in adv_t and aid in ret_t):
                    all_obs.append(step_obs[aid])
                    all_actions.append(step_actions[aid])
                    all_log_probs.append(step_log_probs[aid])
                    all_advs.append(adv_t[aid])
                    all_rets.append(ret_t[aid])

        npts = len(all_obs)
        print(f"📊 Data points: {npts}")
        if npts < 32:
            print("⚠️ Few points (<32). Likely early termination. Skipping updates.")
            continue

        obs_tensor = torch.as_tensor(np.array(all_obs), dtype=torch.float32, device=device)
        actions_tensor = torch.as_tensor(all_actions, dtype=torch.long, device=device)
        old_log_probs = torch.as_tensor(all_log_probs, dtype=torch.float32, device=device)
        advantages_tensor = torch.as_tensor(all_advs, dtype=torch.float32, device=device)
        returns_tensor = torch.as_tensor(all_rets, dtype=torch.float32, device=device)

        if advantages_tensor.std() > 1e-8:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # PPO updates
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
            value_loss = nn.MSELoss()(values_pred.squeeze(-1), returns_tensor)
            entropy_loss = entropy.mean()
            loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

        # Training metrics
        avg_reward = float(np.mean([r for step in rewards for r in step.values()])) if rewards else 0.0
        global_step += npts

        writer.add_scalar("train/policy_loss", float(policy_loss.item()), global_step)
        writer.add_scalar("train/value_loss", float(value_loss.item()), global_step)
        writer.add_scalar("train/entropy", float(entropy_loss.item()), global_step)
        writer.add_scalar("train/total_loss", float(loss.item()), global_step)
        writer.add_scalar("train/avg_reward", float(avg_reward), global_step)
        writer.add_scalar("train/data_points", int(npts), global_step)

        # Optional reward breakdowns (heuristics)
        if rewards:
            flat_r = [r for step in rewards for r in step.values()]
            writer.add_scalar("rewards/avg_all", float(np.mean(flat_r)), global_step)
            writer.add_scalar("rewards/std_all", float(np.std(flat_r)), global_step)

        # Detailed env metrics
        try:
            _log_detailed_metrics(env, writer, global_step, args)
        except Exception as e:
            print(f"⚠️ Detailed metrics failed: {e}")

        if args.verbose_logging or (iteration + 1) % 10 == 0:
            _print_detailed_info(env, iteration + 1, avg_reward)

        if (iteration + 1) % 50 == 0:
            os.makedirs("models", exist_ok=True)
            torch.save(
                {
                    "model_state_dict": agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "iteration": iteration,
                    "args": args,
                },
                f"models/simple_ippo_iter_{iteration + 1}.pt",
            )
            print(f"💾 Model saved at iteration {iteration + 1}")

        print(f"✅ Iter {iteration + 1} done in {time.time() - it_start:.2f}s")

    # Save final
    os.makedirs("models", exist_ok=True)
    final_model_path = f"models/simple_ippo_final_{run_name}.pt"
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
    parser.add_argument("--log_dir", type=str, default="ippo_logs", help="Directory to save logs")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--verbose", action="store_true", help="Show detailed info every iteration")
    parser.add_argument("--no-individual-tracking", action="store_true", help="Disable individual agent tracking in TensorBoard")
    parser.add_argument("--max-agents-tracked", type=int, default=10, help="Maximum number of agents to track individually")

    args = parser.parse_args()

    train_args = SimpleArgs(
        config_path=args.config_path,
        n_agents=args.n_agents,
        initial_resource_stock=args.initial_resource_stock,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        log_dir=args.log_dir,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        verbose_logging=args.verbose,
        track_individual_agents=not args.no_individual_tracking,
        max_agents_to_track=args.max_agents_tracked,
    )

    train_ippo_simple(train_args)


if __name__ == "__main__":
    # Examples of usage:
    # python -m src.scripts.pettingzoo_env.train_ippo_simple --total_timesteps 10000 --n_agents 5
    # python -m src.scripts.pettingzoo_env.train_ippo_simple --log_dir "experimento_1" --seed 42
    # python -m src.scripts.pettingzoo_env.train_ippo_simple --log_dir "teste_cooperacao" --n_agents 8 --verbose
    # python -m src.scripts.pettingzoo_env.train_ippo_simple --log_dir "debug" --verbose --total_timesteps 5000
    # python -m src.scripts.pettingzoo_env.train_ippo_simple --log_dir "individual_analysis" --max-agents-tracked 5
    # python -m src.scripts.pettingzoo_env.train_ippo_simple --log_dir "aggregate_only" --no-individual-tracking
    main()
