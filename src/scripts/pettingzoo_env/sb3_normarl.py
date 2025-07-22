"""
Trains and evaluates agents in the NormalHomeostaticEnv using Stable-Baselines3 and SuperSuit.
"""

import glob
import os
import time

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from pettingzoo.utils.conversions import aec_to_parallel

from src.envs.multiagent import NormalHomeostaticEnv


def train(env_fn, steps: int = 100_000, seed: int = 0, **env_kwargs):
    """Train agents in a vectorized environment using PPO."""

    env = aec_to_parallel(env_fn())
    #env = ss.flatten_v0(env)

    # Optionally: enable this if your environment removes dead agents (not needed here)
    env = ss.black_death_v3(env)
    _, _ = env.reset()
    # Vectorize for SB3
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")

    print(f"Starting training on {env.unwrapped.metadata.get('name')}")

    model = PPO(
        MlpPolicy,
        env,
        verbose=3,
        batch_size=256
    )

    model.learn(total_timesteps=steps)

    timestamp = time.strftime('%Y%m%d-%H%M%S')
    model.save(f"{env.unwrapped.metadata.get('name')}_{timestamp}")
    print("Model has been saved.")

    env.close()
    print("Finished training.")


def eval(env_fn, num_games: int = 10, render_mode: str | None = None, **env_kwargs):
    """Evaluate a trained agent against random or itself."""

    env = env_fn()
    print(f"\nStarting evaluation on {env.metadata['name']}")

    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("No trained policy found.")
        return

    model = PPO.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

    for i in range(num_games):
        env.reset()

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            for a in env.agents:
                rewards[a] += env.rewards[a]

            if termination or truncation:
                env.step(None)
            else:
                if agent == env.possible_agents[0]:
                    action = env.action_space(agent).sample()  # baseline: random
                else:
                    action = model.predict(obs, deterministic=True)[0]
                env.step(action)

    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: rewards[agent] / num_games for agent in env.possible_agents
    }

    print(f"Avg reward: {avg_reward:.3f}")
    print("Avg reward per agent:", avg_reward_per_agent)
    return avg_reward


if __name__ == "__main__":
    # Define env constructor as a function
    def env_fn():
        return NormalHomeostaticEnv(
            config_path="config/config.yaml",
            drive_type="base_drive",
            learning_rate=0.1,
            beta=0.5,
            number_resources=1,
            n_agents=2,
            size=1
        )

    env_kwargs = dict(max_cycles=100)

    # Train for 81k steps
    train(env_fn, steps=81_920, seed=0, **env_kwargs)

    # Evaluate trained agent
    eval(env_fn, num_games=10, render_mode=None, **env_kwargs)

    # Optionally render
    eval(env_fn, num_games=2, render_mode="human", **env_kwargs)
