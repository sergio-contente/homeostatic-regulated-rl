"""
Training script for NORMARL using IndependentAgentsWrapper.
This version provides true agent independence without PettingZoo.
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.gymnasium_env.envs.normal import NormarlHomeostaticEnv
from src.gymnasium_env.wrappers.independent_agents import IndependentAgentsWrapper


def unwrap_env(env):
    """Unwrap environment to access base properties"""
    if hasattr(env, 'unwrapped'):
        return env.unwrapped
    return env


class IndependentAgentsCallback(BaseCallback):
    """Callback específico para logs do IndependentAgentsWrapper"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_consumptions = []
        self.episode_social_costs = []
        self.resource_stocks = []

    def _on_step(self) -> bool:
        # Log estatísticas quando episódio termina
        if self.locals.get("dones", [False])[0]:
            info = self.locals.get("infos", [{}])[0]
            
            if "mean_episode_reward" in info:
                self.episode_rewards.append(info["mean_episode_reward"])
                self.episode_consumptions.append(info["total_episode_consumption"])
                self.resource_stocks.append(info["final_resource_stock"])
                
                # Log principais métricas
                self.logger.record("independent/mean_episode_reward", info["mean_episode_reward"])
                self.logger.record("independent/total_consumption", info["total_episode_consumption"])
                self.logger.record("independent/final_resource_stock", info["final_resource_stock"])
                self.logger.record("independent/episode_length", info["episode_length"])
                
                # Log métricas por agente
                for agent_id in range(2):  # Ajustar se necessário
                    if f"agent_{agent_id}_mean_reward" in info:
                        self.logger.record(f"independent/agent_{agent_id}_reward", info[f"agent_{agent_id}_mean_reward"])
                        self.logger.record(f"independent/agent_{agent_id}_consumption", info[f"agent_{agent_id}_total_consumption"])
                        self.logger.record(f"independent/agent_{agent_id}_social_cost", info[f"agent_{agent_id}_mean_social_cost"])

        # Log estados dos agentes individuais
        try:
            vec_env = self.training_env.envs[0]
            
            # Acessar o wrapper independente
            if hasattr(vec_env, 'get_agent_states'):
                agent_states = vec_env.get_agent_states()
                
                for agent_id, state in agent_states.items():
                    self.logger.record(f"agent_{agent_id}/States/drive", state["drive"])
                    self.logger.record(f"agent_{agent_id}/States/position", float(state["position"]))
                    
                    # Log estados internos individuais
                    for i, value in enumerate(state["internal_states"]):
                        state_names = vec_env.env.unwrapped.drive.get_internal_states_names()
                        name = state_names[i] if i < len(state_names) else f"state_{i}"
                        self.logger.record(f"agent_{agent_id}/States/{name}", float(value))
                        
        except Exception as e:
            if self.verbose:
                print(f"[Logging warning] Could not log agent states: {e}")

        self.logger.dump(self.num_timesteps)
        return True

    def save_metrics(self, filepath):
        """Save training metrics"""
        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_consumptions': self.episode_consumptions,
            'episode_social_costs': self.episode_social_costs,
            'resource_stocks': self.resource_stocks
        }
        np.save(filepath, metrics)


def create_independent_env(config_path="config/config.yaml", n_agents=2, 
                          ppo_learning_rate=3e-4, social_learning_rate=0.02):
    """Create environment with independent agents wrapper"""
    def _init():
        # Criar ambiente base
        env = NormarlHomeostaticEnv(
            config_path=config_path,
            drive_type="base_drive",
            social_learning_rate=social_learning_rate,
            beta=0.8,
            size=1,
            render_mode=None
        )
        
        # Aplicar wrapper de independência
        env = IndependentAgentsWrapper(env, n_agents=n_agents)
        env = Monitor(env)
        return env
    
    return _init


def train_independent_agents(total_timesteps=20000, n_envs=1, n_agents=2, 
                           config_path="config/config.yaml", 
                           save_path="./independent_models/", 
                           log_path="./independent_logs/", 
                           ppo_learning_rate=3e-4, 
                           social_learning_rate=0.02):
    """Train PPO with independent agents"""
    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # Criar ambiente vetorizado
    env = make_vec_env(
        create_independent_env(config_path, n_agents, ppo_learning_rate, social_learning_rate),
        n_envs=n_envs,
        vec_env_cls=DummyVecEnv
    )

    # Criar modelo PPO
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=ppo_learning_rate,
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

    # Callbacks
    callback = IndependentAgentsCallback()
    eval_env = DummyVecEnv([create_independent_env(config_path, n_agents, ppo_learning_rate, social_learning_rate)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=log_path,
        eval_freq=5000,
        deterministic=True,
        verbose=1
    )

    # Treinar
    print("Starting independent agents training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[callback, eval_callback],
        progress_bar=True
    )

    # Salvar modelo final
    model.save(os.path.join(save_path, "final_model"))
    
    # Salvar métricas
    metrics_path = os.path.join(save_path, "training_metrics.npy")
    callback.save_metrics(metrics_path)
    
    return model, callback


def test_independent_agents(n_agents=2, n_episodes=5):
    """Test the independent agents wrapper with smart consumption"""
    print("🧪 Testing Independent Agents Wrapper (Smart Consumption)")
    print("=" * 50)
    
    # Criar ambiente usando a mesma função de criação do treinamento
    env = create_independent_env(n_agents=n_agents)()
    
    for episode in range(n_episodes):
        print(f"\n🎬 Episode {episode + 1}")
        obs, info = env.reset()
        
        total_steps = 0
        while total_steps < 50:  # Limite por episódio
            
            # 🧠 SMART ACTION: Consume when at resource position
            current_agent = unwrap_env(env).current_agent
            agent_pos = unwrap_env(env).agents_states[current_agent]["position"]
            
            # 🔍 DEBUG: Show agent and resource positions
            base_env = unwrap_env(env).env.unwrapped
            print(f"🤖 Agent {current_agent} at position {agent_pos}")
            print(f"🏭 Resources: {[(rid, info['position']) for rid, info in base_env.resources_info.items()]}")
            print(f"🌍 Environment size: {base_env.size}")
            
            # Check if agent is at any resource position
            should_consume = False
            
            for resource_id, resource_info in base_env.resources_info.items():
                if agent_pos == resource_info["position"]:
                    action = 3 + resource_id  # Consume this resource
                    should_consume = True
                    print(f"🎯 SMART: Agent {current_agent} at resource pos {agent_pos} → consuming resource {resource_id}")
                    break
            
            if not should_consume:
                action = env.action_space.sample()  # Random action
                print(f"🎲 RANDOM: Agent {current_agent} not at any resource → random action {action}")
            
            obs, reward, done, truncated, info = env.step(action)
            total_steps += 1
            
            if done:
                print(f"Episode {episode + 1} finished after {total_steps} steps")
                break
        
        # Mostrar resumo final do episódio
        unwrap_env(env).print_agent_summary()
    
    print("✅ Test completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train", 
                       help="Run training or testing")
    parser.add_argument("--n_agents", type=int, default=2, help="Number of agents")
    parser.add_argument("--timesteps", type=int, default=20000, help="Training timesteps")
    
    args = parser.parse_args()
    
    if args.mode == "test":
        test_independent_agents(n_agents=args.n_agents)
    else:
        config = {
            'total_timesteps': args.timesteps,
            'n_envs': 1,
            'n_agents': args.n_agents,
            'config_path': 'config/config.yaml',
            'ppo_learning_rate': 3e-4,
            'social_learning_rate': 0.02
        }

        print("Starting Independent Agents PPO Training")
        print("=" * 50)
        print(f"Number of agents: {config['n_agents']}")
        print(f"PPO Learning Rate: {config['ppo_learning_rate']}")
        print(f"Social Learning Rate: {config['social_learning_rate']}")
        print(f"Total timesteps: {config['total_timesteps']}")
        print("=" * 50)
        
        model, callback = train_independent_agents(**config)
        print("\n✅ Training completed!")
        print("=" * 50) 
