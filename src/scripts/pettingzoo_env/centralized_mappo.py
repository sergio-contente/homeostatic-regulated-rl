#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MAPPO (Crítico Centralizado) para Homeostatic Environment (PettingZoo Parallel API)

- Crítico centralizado: V_i = f([estado_global || one_hot(agent_i)])
- Ator descentralizado compartilhado (IPPO-style)
- PPO (clipped) + GAE(gamma, lambda)
- TensorBoard com as mesmas métricas detalhadas do script anterior
"""

import os
import time
import random
import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from src.envs.multiagent import create_env
from pettingzoo.utils.conversions import aec_to_parallel

# exp1 : regen 0.03, beta 1, stock 2
# exp2 : regen 0.03, beta 0.8, stock 2
# exp3 : regen 0.03, beta 0.6, stock 2
# exp3 : regen 0.03, beta 0.4, stock 2
# exp3 : regen 0.03, beta 0.6, stock 2
# exp3 : regen 0.03, beta 0.0, stock 2


# gamma 0.8


# =======================================================
# Hyperparameters / Args
# =======================================================
@dataclass
class SimpleArgs:
    # Env
    config_path: str = "config/config.yaml"
    drive_type: str = "base_drive"
    learning_rate_social: float = 0.1
    beta: float = 2.0
    number_resources: int = 1
    n_agents: int = 10
    env_size: int = 1
    max_steps: int = 1000
    initial_resource_stock: float = 2.0

    # PPO / MAPPO
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    num_steps: int = 200            # ~20 "rounds" * 10 agentes
    gamma: float = 0.8
    gae_lambda: float = 0.95
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Misc
    seed: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir: str = "mappo_logs"
    verbose_logging: bool = False


# =======================================================
# Redes
# =======================================================
class ActorPolicy(nn.Module):
    """Ator compartilhado (descentralizado)."""
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x)

    def act(self, x):
        logits = self.forward(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        ent = dist.entropy()
        return action, logp, ent

    def evaluate_actions(self, x, actions):
        logits = self.forward(x)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(actions)
        ent = dist.entropy()
        return logp, ent


class CentralCritic(nn.Module):
    """Crítico centralizado: input = [estado_global || one_hot(agent_id)] -> V_i."""
    def __init__(self, state_dim: int, n_agents: int):
        super().__init__()
        self.n_agents = n_agents
        self.net = nn.Sequential(
            nn.Linear(state_dim + n_agents, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def forward(self, state_global, agent_onehot):
        x = torch.cat([state_global, agent_onehot], dim=-1)
        v = self.net(x)
        return v.squeeze(-1)


# =======================================================
# Obs processing e estado global
# =======================================================
class ObservationProcessor:
    """Achata observação (dict -> vetor; array -> vetor)."""
    def __init__(self, sample_obs):
        if isinstance(sample_obs, dict):
            self.is_dict = True
            self.obs_dim = sum(np.array(v).size if hasattr(v, "size") else 1 for v in sample_obs.values())
        else:
            self.is_dict = False
            self.obs_dim = np.array(sample_obs).size

    def process(self, obs) -> np.ndarray:
        if self.is_dict:
            flat = []
            for v in obs.values():
                if hasattr(v, "__len__") and not isinstance(v, (int, float)):
                    flat.extend(np.array(v, dtype=np.float32).ravel())
                else:
                    flat.append(float(v))
            return np.array(flat, dtype=np.float32)
        return np.array(obs, dtype=np.float32).ravel()


def build_global_state(obs_dict: Dict[str, dict],
                       agent_order: List[str],
                       obs_processor: ObservationProcessor,
                       obs_dim: int) -> np.ndarray:
    """Concatena obs de todos os agentes numa ordem fixa."""
    chunks = []
    for aid in agent_order:
        if aid in obs_dict:
            chunks.append(obs_processor.process(obs_dict[aid]))
        else:
            chunks.append(np.zeros(obs_dim, dtype=np.float32))
    return np.concatenate(chunks, axis=0).astype(np.float32)


def one_hot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(n, dtype=torch.float32, device=device)
    v[idx] = 1.0
    return v


# =======================================================
# Coleta de rollout (Parallel API)
# =======================================================
def collect_rollout(env, actor, critic, obs_processor, args, device, agent_order):
    """
    Retorna:
      - observações por passo (dict por agente)
      - actions/log_probs/rewards/dones/values/next_values por agente
      - global_states (lista de vetores concatenados)
    """
    obs_dict, _ = env.reset()
    id2idx = {aid: i for i, aid in enumerate(agent_order)}

    observations, actions, log_probs = [], [], []
    rewards, dones = [], []
    values, next_values, global_states = [], [], []

    for _ in range(args.num_steps):
        # estado global atual
        gstate_np = build_global_state(obs_dict, agent_order, obs_processor, obs_processor.obs_dim)
        global_states.append(gstate_np)

        # prepara obs locais
        obs_tensors: Dict[str, torch.Tensor] = {}
        for aid in env.agents:
            if aid in obs_dict:
                o = obs_processor.process(obs_dict[aid])
                obs_tensors[aid] = torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)

        # escolhe ações e valores atuais
        step_actions, step_logp, step_values = {}, {}, {}
        with torch.no_grad():
            gstate = torch.tensor(gstate_np, dtype=torch.float32, device=device).unsqueeze(0)
            for aid in obs_tensors.keys():
                a, lp, _ = actor.act(obs_tensors[aid])
                step_actions[aid] = int(a.item())
                step_logp[aid] = float(lp.item())
                oh = one_hot(id2idx[aid], len(agent_order), device).unsqueeze(0)
                v = critic(gstate, oh)
                step_values[aid] = float(v.item())

        # step no ambiente
        next_obs, rew, term, trunc, _ = env.step(step_actions)
        step_dones = {aid: bool(term.get(aid, False) or trunc.get(aid, False)) for aid in step_actions.keys()}

        # valores no próximo estado (p/ GAE)
        with torch.no_grad():
            gstate_next_np = build_global_state(next_obs, agent_order, obs_processor, obs_processor.obs_dim)
            gstate_next = torch.tensor(gstate_next_np, dtype=torch.float32, device=device).unsqueeze(0)
            step_next_values = {}
            for aid in step_actions.keys():
                oh = one_hot(id2idx[aid], len(agent_order), device).unsqueeze(0)
                v_n = critic(gstate_next, oh)
                step_next_values[aid] = float(v_n.item())

        # guarda buffers
        observations.append({aid: obs_tensors[aid].squeeze(0).cpu().numpy() for aid in obs_tensors})
        actions.append(step_actions)
        log_probs.append(step_logp)
        rewards.append(rew)
        dones.append(step_dones)
        values.append(step_values)
        next_values.append(step_next_values)

        obs_dict = next_obs
        if (term and all(term.values())) or (trunc and all(trunc.values())) or len(env.agents) == 0:
            break

    return observations, actions, log_probs, rewards, dones, values, next_values, global_states


# =======================================================
# GAE (por agente) com discount factor
# =======================================================
def compute_gae_central(rewards, values, next_values, dones, gamma, lam):
    """
    rewards/values/next_values/dones: listas no tempo de dicts {agent_id: val}
    Retorna:
      adv_dict[aid] -> lista A_t
      ret_dict[aid] -> lista R_t^λ = A_t + V_t
    """
    agent_ids = set()
    for step_v in values:
        agent_ids.update(step_v.keys())

    adv_dict, ret_dict = {}, {}
    for aid in agent_ids:
        r_seq, v_seq, vn_seq, d_seq = [], [], [], []
        for t in range(len(rewards)):
            if aid in values[t]:
                r_seq.append(float(rewards[t].get(aid, 0.0)))
                v_seq.append(float(values[t][aid]))
                vn_seq.append(float(next_values[t].get(aid, 0.0)))
                d_seq.append(bool(dones[t].get(aid, False)))

        if len(r_seq) == 0:
            continue

        r_seq, v_seq, vn_seq, d_seq = map(np.asarray, (r_seq, v_seq, vn_seq, d_seq))
        adv = np.zeros_like(r_seq, dtype=np.float32)
        gae = 0.0
        for i in reversed(range(len(r_seq))):
            delta = r_seq[i] + gamma * vn_seq[i] * (1.0 - d_seq[i]) - v_seq[i]
            gae = delta + gamma * lam * (1.0 - d_seq[i]) * gae
            adv[i] = gae
        ret = adv + v_seq

        adv_dict[aid] = adv.tolist()
        ret_dict[aid] = ret.tolist()

    return adv_dict, ret_dict


# =======================================================
# Logging detalhado (mesmo conjunto do script anterior)
# =======================================================
def _log_detailed_metrics(env, writer: SummaryWriter, global_step: int, args: SimpleArgs = None):
    """Loga recursos, consumo, normas, estados internos, histogramas, coop/sustentabilidade."""
    try:
        base_env = env
        # unwrap seguro
        for _ in range(8):
            if hasattr(base_env, "unwrapped"):
                nxt = base_env.unwrapped
                if nxt is base_env:
                    break
                base_env = nxt
            else:
                break
        if hasattr(base_env, "env"):
            base_env = base_env.env

        if not hasattr(base_env, "homeostatic_agents"):
            return

        agents = getattr(base_env, "homeostatic_agents", {})
        resource_stock = getattr(base_env, "resource_stock", None)

        internal_states, drives, social_norms, last_intakes = [], [], [], []

        # coleta por agente
        for agent_id, agent in agents.items():
            if hasattr(agent, "internal_states"):
                internal_states.append(float(agent.internal_states[0]))
            if hasattr(agent, "get_current_drive"):
                drives.append(float(agent.get_current_drive()))
            if hasattr(agent, "perceived_social_norm") and len(agent.perceived_social_norm) > 0:
                social_norms.append(float(agent.perceived_social_norm[0]))
            if hasattr(agent, "last_intake") and len(agent.last_intake) > 0:
                last_intakes.append(float(agent.last_intake[0]))
        
        for agent_id, agent in agents.items():
         prefix = f"individual/{agent_id}"
         if hasattr(agent, "internal_states"):
           writer.add_scalar(f"{prefix}/internal_state", float(agent.internal_states[0]), global_step)
         if hasattr(agent, "get_current_drive"):
           writer.add_scalar(f"{prefix}/drive", float(agent.get_current_drive()), global_step)
         if hasattr(agent, "perceived_social_norm") and len(agent.perceived_social_norm) > 0:
           writer.add_scalar(f"{prefix}/social_norm", float(agent.perceived_social_norm[0]), global_step)
         if hasattr(agent, "last_intake") and len(agent.last_intake) > 0:
           writer.add_scalar(f"{prefix}/consumption", float(agent.last_intake[0]), global_step)


        # estatísticas agregadas + histogramas
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

        # população viva (aproximação: len dict)
        writer.add_scalar("population/alive_agents", int(len(agents)), global_step)

        # cooperação e sustentabilidade (mesmo cálculo do script anterior)
        if last_intakes and internal_states:
            non_critical_idx = [i for i, st in enumerate(internal_states) if st > -0.5]
            if non_critical_idx:
                nc_cons = [last_intakes[i] for i in non_critical_idx if i < len(last_intakes)]
                if nc_cons:
                    coop_idx = 1.0 - (float(np.mean(nc_cons)) / 0.1)
                    writer.add_scalar("cooperation/cooperation_index", float(max(0.0, coop_idx)), global_step)

        if resource_stock is not None:
            # baseline depleção
            try:
                if not hasattr(base_env, "_last_resource_stock"):
                    base_env._last_resource_stock = float(resource_stock[0])
            except Exception:
                pass

            writer.add_scalar("resources/stock_food", float(resource_stock[0]), global_step)
            initial_stock = getattr(base_env, "initial_resource_stock", [1.0])
            init_val = float(initial_stock[0]) if isinstance(initial_stock, (list, tuple)) else float(initial_stock)
            if init_val > 0:
                writer.add_scalar("resources/stock_percentage", float(resource_stock[0]) / init_val * 100.0, global_step)

            # taxa de depleção
            try:
                last = float(getattr(base_env, "_last_resource_stock", float(resource_stock[0])))
                depletion_rate = last - float(resource_stock[0])
                writer.add_scalar("resources/depletion_rate", float(depletion_rate), global_step)
                base_env._last_resource_stock = float(resource_stock[0])
            except Exception:
                pass

            # sustentabilidade (recursos + população)
            try:
                resource_ratio = float(resource_stock[0]) / init_val if init_val > 0 else 0.0
                population_ratio = len(agents) / float(getattr(base_env, "initial_n_agents", len(agents) or 1))
                sustainability = (resource_ratio + population_ratio) / 2.0
                writer.add_scalar("cooperation/sustainability_index", float(sustainability), global_step)
            except Exception:
                pass

    except Exception as e:
        print(f"[log] métricas detalhadas falharam: {e}")


# =======================================================
# Treinamento
# =======================================================
def train_mappo_central(args: SimpleArgs):
    print("🚀 MAPPO (Crítico Centralizado)")
    print(f"Device: {args.device} | Agents: {args.n_agents}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # ----- Ambiente
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

    # ----- Reset inicial -> dimensões e ordem
    sample_obs_dict, _ = env.reset()
    assert len(env.agents) > 0, "No agents found."
    agent_order = list(env.possible_agents) if hasattr(env, "possible_agents") else list(env.agents)

    sample_obs = next(iter(sample_obs_dict.values()))
    obs_proc = ObservationProcessor(sample_obs)
    obs_dim = obs_proc.obs_dim
    action_dim = env.action_space(env.agents[0]).n
    n_agents = len(agent_order)
    state_dim = obs_dim * n_agents

    # ----- Redes / Optimizador
    actor = ActorPolicy(obs_dim, action_dim).to(device)
    critic = CentralCritic(state_dim, n_agents).to(device)
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.learning_rate)

    # ----- Logging
    run_name = f"mappo_central_{int(time.time())}"
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(f"{args.log_dir}/{run_name}")
    global_step = 0

    print(f"📊 Logs em: {args.log_dir}/{run_name} | tensorboard --logdir {args.log_dir}")

    # baseline de recursos para taxa de depleção e log inicial
    try:
        if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "resource_stock"):
            env.unwrapped._last_resource_stock = float(env.unwrapped.resource_stock[0])
    except Exception:
        pass
    _log_detailed_metrics(env, writer, global_step, args)

    # ----- Loop de treino
    num_iters = args.total_timesteps // max(1, args.num_steps)
    for it in range(num_iters):
        it_start = time.time()
        (obs_steps, act_steps, logp_steps, rew_steps, done_steps,
         val_steps, nval_steps, gstates) = collect_rollout(
            env, actor, critic, obs_proc, args, device, agent_order
        )

        T = len(obs_steps)
        if T == 0:
            print("⚠️ rollout vazio; resetando env…")
            env.reset()
            continue

        # GAE por agente (usa gamma e lambda)
        adv_dict, ret_dict = compute_gae_central(
            rew_steps, val_steps, nval_steps, done_steps, args.gamma, args.gae_lambda
        )

        # Achata para batch
        batch_obs, batch_actions, batch_old_logp = [], [], []
        batch_adv, batch_ret, batch_gstate, batch_onehot = [], [], [], []

        id2idx = {aid: i for i, aid in enumerate(agent_order)}
        counters = {aid: 0 for aid in adv_dict.keys()}

        # também calculamos média de recompensa p/ logging
        all_rewards_flat = []

        for t in range(T):
            gstate_t = torch.tensor(gstates[t], dtype=torch.float32)
            # média de reward deste passo (apenas p/ log simples)
            if rew_steps[t]:
                all_rewards_flat.extend(list(rew_steps[t].values()))
            for aid, a in act_steps[t].items():
                if aid not in adv_dict:
                    continue
                idx = counters[aid]
                if idx >= len(adv_dict[aid]):
                    continue
                # obs local
                o = obs_steps[t][aid]
                batch_obs.append(o)
                batch_actions.append(a)
                batch_old_logp.append(logp_steps[t][aid])
                batch_adv.append(adv_dict[aid][idx])
                batch_ret.append(ret_dict[aid][idx])
                batch_gstate.append(gstate_t.numpy())
                oh = np.zeros(n_agents, dtype=np.float32)
                oh[id2idx[aid]] = 1.0
                batch_onehot.append(oh)
                counters[aid] += 1

        if len(batch_obs) < 32:
            print(f"⚠️ poucos samples ({len(batch_obs)}). pulando update.")
            continue

        # Tensores
        obs_tensor = torch.tensor(np.array(batch_obs), dtype=torch.float32, device=device)
        act_tensor = torch.tensor(np.array(batch_actions), dtype=torch.long, device=device)
        old_logp_tensor = torch.tensor(np.array(batch_old_logp), dtype=torch.float32, device=device)
        adv_tensor = torch.tensor(np.array(batch_adv), dtype=torch.float32, device=device)
        ret_tensor = torch.tensor(np.array(batch_ret), dtype=torch.float32, device=device)
        gstate_tensor = torch.tensor(np.array(batch_gstate), dtype=torch.float32, device=device)
        onehot_tensor = torch.tensor(np.array(batch_onehot), dtype=torch.float32, device=device)

        # Normalização de vantagens
        if adv_tensor.std() > 1e-8:
            adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        # Atualizações PPO
        for epoch in range(args.update_epochs):
            # Ator
            new_logp, entropy = actor.evaluate_actions(obs_tensor, act_tensor)
            ratio = (new_logp - old_logp_tensor).exp()
            surr1 = ratio * adv_tensor
            surr2 = torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef) * adv_tensor
            policy_loss = -torch.min(surr1, surr2).mean()

            # Crítico centralizado
            values_pred = critic(gstate_tensor, onehot_tensor)
            value_loss = nn.MSELoss()(values_pred, ret_tensor)

            # Total
            loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), args.max_grad_norm)
            optimizer.step()

        # Logging básico de treino
        global_step += len(batch_obs)
        writer.add_scalar("train/policy_loss", float(policy_loss.item()), global_step)
        writer.add_scalar("train/value_loss", float(value_loss.item()), global_step)
        writer.add_scalar("train/entropy", float(entropy.mean().item()), global_step)
        writer.add_scalar("train/num_samples", int(len(batch_obs)), global_step)
        if all_rewards_flat:
            writer.add_scalar("train/avg_reward", float(np.mean(all_rewards_flat)), global_step)

        # Logging detalhado igual ao script anterior
        _log_detailed_metrics(env, writer, global_step, args)

        dt = time.time() - it_start
        print(f"✅ iter {it+1}/{num_iters} | steps={T} | batch={len(batch_obs)} | time={dt:.2f}s")

        # checkpoint periódico
        if (it + 1) % 50 == 0:
            os.makedirs("models", exist_ok=True)
            torch.save(
                {
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": args,
                },
                f"models/mappo_central_iter_{it+1}.pt",
            )

    # Final
    os.makedirs("models", exist_ok=True)
    final_path = f"models/mappo_central_final_{int(time.time())}.pt"
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": args,
        },
        final_path,
    )
    print(f"🏁 Treino concluído. Modelo salvo em {final_path}")
    writer.close()
    env.close()


# =======================================================
# CLI
# =======================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", type=str, default="config/config.yaml")
    p.add_argument("--n_agents", type=int, default=10)
    p.add_argument("--initial_resource_stock", type=float, default=2.0)
    p.add_argument("--total_timesteps", type=int, default=100_000)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--log_dir", type=str, default="mappo_logs")
    p.add_argument("--num_steps", type=int, default=200)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_coef", type=float, default=0.2)
    p.add_argument("--update_epochs", type=int, default=4)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--verbose", action="store_true")

    args_cli = p.parse_args()
    args = SimpleArgs(
        config_path=args_cli.config_path,
        n_agents=args_cli.n_agents,
        initial_resource_stock=args_cli.initial_resource_stock,
        total_timesteps=args_cli.total_timesteps,
        learning_rate=args_cli.learning_rate,
        seed=args_cli.seed,
        log_dir=args_cli.log_dir,
        num_steps=args_cli.num_steps,
        gamma=args_cli.gamma,
        gae_lambda=args_cli.gae_lambda,
        clip_coef=args_cli.clip_coef,
        update_epochs=args_cli.update_epochs,
        ent_coef=args_cli.ent_coef,
        vf_coef=args_cli.vf_coef,
        max_grad_norm=args_cli.max_grad_norm,
        verbose_logging=args_cli.verbose,
    )
    train_mappo_central(args)


if __name__ == "__main__":
    main()
