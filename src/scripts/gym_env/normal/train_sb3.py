import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import os

from src.gymnasium_env.envs.normal import NormarlHomeostaticEnv

def unwrap_env(env):
    while hasattr(env, 'env'):
        env = env.env
    return env

class MultiAgentNormarlWrapper(gym.Wrapper):
    def __init__(self, env, n_agents=2):
        super().__init__(env)
        self.n_agents = n_agents
        self.current_agent = 0
        self.episode_step = 0
        self.max_episode_steps = 200

        self.agents_intake_history = [[] for _ in range(n_agents)]
        self.trial_intakes = []
        self.avg_intake = np.zeros(self.env.dimension_internal_states)

        self.agent_rewards = [[] for _ in range(n_agents)]
        self.agent_social_costs = [[] for _ in range(n_agents)]
        self.agent_consumptions = [[] for _ in range(n_agents)]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_agent = 0
        self.episode_step = 0
        self.avg_intake = np.zeros(self.env.dimension_internal_states)
        self.trial_intakes = []
        self.agents_intake_history = [[] for _ in range(self.n_agents)]
        self.agent_rewards = [[] for _ in range(self.n_agents)]
        self.agent_social_costs = [[] for _ in range(self.n_agents)]
        self.agent_consumptions = [[] for _ in range(self.n_agents)]

        obs["agent_id"] = self.current_agent
        obs["episode_step"] = self.episode_step
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action, self.avg_intake)

        agent_intake = info["last_intake"]
        self.agents_intake_history[self.current_agent].append(agent_intake.copy())
        self.trial_intakes.append(agent_intake.copy())

        self.agent_rewards[self.current_agent].append(reward)
        self.agent_consumptions[self.current_agent].append(np.sum(agent_intake))
        social_cost = self.env.compute_social_cost(agent_intake)
        self.agent_social_costs[self.current_agent].append(social_cost)

        self.episode_step += 1
        is_last_agent = self.current_agent == self.n_agents - 1
        if is_last_agent:
            self.avg_intake = np.mean(self.trial_intakes, axis=0)
            self.trial_intakes = []

        self.current_agent = (self.current_agent + 1) % self.n_agents

        if self.episode_step >= self.max_episode_steps:
            done = True
            print(f"Episode ended due to max steps ({self.max_episode_steps})")

        obs["agent_id"] = self.current_agent
        obs["episode_step"] = self.episode_step

        if done:
            info.update(self._get_episode_stats())

        return obs, reward, done, truncated, info

    def _get_episode_stats(self):
        stats = {}
        for agent_id in range(self.n_agents):
            if self.agent_rewards[agent_id]:
                stats[f"agent_{agent_id}_mean_reward"] = np.mean(self.agent_rewards[agent_id])
                stats[f"agent_{agent_id}_total_consumption"] = np.sum(self.agent_consumptions[agent_id])
                stats[f"agent_{agent_id}_mean_social_cost"] = np.mean(self.agent_social_costs[agent_id])
                stats[f"agent_{agent_id}_steps"] = len(self.agent_rewards[agent_id])

        all_rewards = [r for rewards in self.agent_rewards for r in rewards]
        if all_rewards:
            stats["mean_episode_reward"] = np.mean(all_rewards)
            stats["total_episode_consumption"] = np.sum([np.sum(c) for c in self.agent_consumptions])
            stats["final_resource_stock"] = np.sum(self.env.resource_stock)
            stats["episode_length"] = self.episode_step

        return stats

class NormarlCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_consumptions = []
        self.episode_social_costs = []
        self.resource_stocks = []

    def _on_step(self) -> bool:
        if self.locals.get("dones", [False])[0]:
            info = self.locals.get("infos", [{}])[0]
            if "mean_episode_reward" in info:
                self.episode_rewards.append(info["mean_episode_reward"])
                self.episode_consumptions.append(info["total_episode_consumption"])
                self.resource_stocks.append(info["final_resource_stock"])
                self.logger.record("normarl/mean_episode_reward", info["mean_episode_reward"])
                self.logger.record("normarl/total_consumption", info["total_episode_consumption"])
                self.logger.record("normarl/final_resource_stock", info["final_resource_stock"])
                self.logger.record("normarl/episode_length", info["episode_length"])
                for agent_id in range(2):
                    if f"agent_{agent_id}_mean_reward" in info:
                        self.logger.record(f"normarl/agent_{agent_id}_reward", info[f"agent_{agent_id}_mean_reward"])
                        self.logger.record(f"normarl/agent_{agent_id}_consumption", info[f"agent_{agent_id}_total_consumption"])
                        self.logger.record(f"normarl/agent_{agent_id}_social_cost", info[f"agent_{agent_id}_mean_social_cost"])

        try:
            vec_env = self.training_env.envs[0]
            base_env = unwrap_env(vec_env)
            for agent_id in range(2):
                drive = base_env.drive.get_current_drive()
                position = base_env.agent_info["position"]
                internal_states = base_env.agent_info["internal_states"]
                state_names = base_env.drive.get_internal_states_names()
                self.logger.record(f"agent_{agent_id}/States/drive", drive)
                self.logger.record(f"agent_{agent_id}/States/position", float(position))
                for i, value in enumerate(internal_states):
                    name = state_names[i] if i < len(state_names) else f"state_{i}"
                    self.logger.record(f"agent_{agent_id}/States/{name}", float(value))
        except Exception as e:
            print(f"[Logging warning] Could not log agent states: {e}")

        self.logger.dump(self.num_timesteps)
        return True

    def save_metrics(self, filepath):
        data = {
            'episode_rewards': self.episode_rewards,
            'episode_consumptions': self.episode_consumptions,
            'resource_stocks': self.resource_stocks
        }
        np.save(filepath, data)


def create_normarl_env(config_path="config/config.yaml", n_agents=2):
    def _init():
        env = NormarlHomeostaticEnv(
            config_path=config_path,
            drive_type="base_drive",
            learning_rate=0.1,
            size=1,
            render_mode=None
        )
        env.beta = 0.8
        env.alpha = 0.1
        env = MultiAgentNormarlWrapper(env, n_agents=n_agents)
        env = Monitor(env)
        return env
    return _init

def train_ppo_normarl(total_timesteps=200000, n_envs=1, n_agents=2, config_path="config/config.yaml", save_path="./normarl_models/", log_path="./normarl_logs/"):
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    env = make_vec_env(
        create_normarl_env(config_path, n_agents),
        n_envs=n_envs,
        vec_env_cls=DummyVecEnv
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=log_path,
        device="auto"
    )

    normarl_callback = NormarlCallback()
    eval_env = DummyVecEnv([create_normarl_env(config_path, n_agents)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=log_path,
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[normarl_callback, eval_callback],
        progress_bar=True
    )

    final_model_path = os.path.join(save_path, "ppo_normarl_final")
    model.save(final_model_path)
    metrics_path = os.path.join(save_path, "training_metrics.npy")
    normarl_callback.save_metrics(metrics_path)
    return model, normarl_callback

if __name__ == "__main__":
    config = {
        'total_timesteps': 200000,
        'n_envs': 1,  # Debug mode
        'n_agents': 2,
        'config_path': 'config/config.yaml'
    }

    print("Starting NORMARL PPO Training\n" + "="*50)
    model, callback = train_ppo_normarl(**config)
    print("\nTraining completed!\n" + "="*50)
