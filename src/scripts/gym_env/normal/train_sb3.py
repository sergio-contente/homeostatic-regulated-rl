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

        # --- Fix Observation Space ---
        old_space = self.env.observation_space
        assert isinstance(old_space, gym.spaces.Dict), "Original observation space must be a Dict"

        new_spaces = old_space.spaces.copy()
        new_spaces['agent_id'] = spaces.Discrete(self.n_agents)
        new_spaces['episode_step'] = spaces.Box(low=0, high=self.max_episode_steps, shape=(1,), dtype=np.int32)
        self.observation_space = spaces.Dict(new_spaces)
        # ---------------------------

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
        obs["episode_step"] = np.array([self.episode_step], dtype=np.int32)
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
        obs["episode_step"] = np.array([self.episode_step], dtype=np.int32)

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
        #regen = 2e-4
            #ppo_1 = 0.8
            #ppo_2 = 0.5
            #ppo_3 = 0.1
            #ppo_4 = 3e-4
        #regen = 2e-6
            #ppo_5 = 0.8
            #ppo_6 = 0.8 but 10*total_intake
            #ppo_7 = 0.8 but 5*total_intake
            #ppo_8 = 0.1 but 5*total_intake
        #regen = 2e-4
            #ppo_9 = 0.8 but 5*total_intake
            #ppo_10 = 0.8 but 2.5*total_intake
            #ppo_11 = 0.8 but 1.25*total_intake
            #ppo_12 = 0.5 but 1.25*total_intake
            #ppo_13 = 0.1 but 1.25*total_intake
            #ppo_14 = 0 but 1.25*total_intake
        #regen = 2e-6
            #ppo_15 = 0.8 but 1.25*total_intake
            #ppo_16 = 0 but 1.25*total_intake
            #ppo_17 = 0 but 2*total_intake
            #ppo_18 = 0 but 2*total_intake with more timesteps
            #ppo_19 = 0 but 2*total_intake with decay rate 0.01
            #ppo_20 = pp0_17
            #ppo_21 = 0 but 2*total_intake with decay rate 0.006
            #ppo_22 = 0 but 2*total_intake with decay rate 0.003 and 20000 timesteps
            #ppo_23 = 0 but 2*total_intake with decay rate 0.0045 and 20000 timesteps
        # regen = 2e-6 and decay 0.006
           #ppo_24 = 0.8 but 2*total_intake
           #ppo_25 = 0.6 but 2*total_intake
           #ppo_26 = 0.3 but 2*total_intake
           #ppo_27 = ppo_26
           #ppo_28 = 0.1 but 2*total_intake
           #ppo_29 = 0.2 but 2*total_intake
           #ppo_30 = 0.2 but 2*total_intake with social cost 1x
           #ppo_31 = 0.8 but 2*total_intake with social cost 1x

            
        learning_rate = 3e-4 
        # learning rate of the other agent belief is really low
        # put more agents in the training to analyse the distribution of beliefs
        # try for 10 agents to begin -> visualize and aggregate the population methods
        # visualize the society and not single agents
        # It can be a solution for a coordination problem <-> we have a drive and a social cost, we want them to coordinate them with each other
        
        beta = 0.8
        env = NormarlHomeostaticEnv(
            config_path=config_path,
            drive_type="base_drive",
            learning_rate=learning_rate,
            beta=beta,
            size=1,
            render_mode=None
        )
        env = MultiAgentNormarlWrapper(env, n_agents=n_agents)
        env = Monitor(env)
        return env
    return _init

def train_ppo_normarl(total_timesteps=20000, n_envs=1, n_agents=2, config_path="config/config.yaml", save_path="./normarl_models/", log_path="./normarl_logs/"):
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
        'total_timesteps': 20000,
        'n_envs': 1,  # Debug mode
        'n_agents': 2,
        'config_path': 'config/config.yaml'
    }

    print("Starting NORMARL PPO Training\n" + "="*50)
    model, callback = train_ppo_normarl(**config)
    print("\nTraining completed!\n" + "="*50)
