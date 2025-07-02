#!/usr/bin/env python3
"""
PPO Training Script for NORMARL Homeostatic Environment
Using Stable Baselines 3 with PettingZoo

This script trains PPO agents in the multi-agent NORMARL environment
where agents learn homeostatic regulation and social norms.
"""

import os
import time
import argparse
from typing import Dict, Any, Optional
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, SubprocVecEnv
import torch

# PettingZoo environment wrappers
from pettingzoo.utils import parallel_to_aec
from pettingzoo.utils.conversions import aec_to_parallel
import supersuit as ss

# Import our NORMARL environment
from src.pettingzoo_env.normarl import NormalHomeostaticEnv


def unwrap_env(env):
    """Unwrap vectorized environment to access the base NORMARL environment."""
    current = env
    
    # Unwrap through VecEnv, Monitor, and Wrapper layers
    while hasattr(current, 'env'):
        current = current.env
        
        # If we find a PettingZooToGymnasiumWrapper, get its environment
        if hasattr(current, 'env') and hasattr(current.env, 'homeostatic_agents'):
            return current.env
    
    # If we reach here, check if current env has the attributes we need
    if hasattr(current, 'homeostatic_agents'):
        return current
        
    return None


class NormarlCallback(BaseCallback):
    """
    Custom callback for logging NORMARL-specific metrics to TensorBoard.
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_consumptions = []
        self.episode_social_costs = []
        self.resource_stocks = []
        self.agent_stats = {}
        
    def _on_step(self) -> bool:
        # Log when episode is done
        if self.locals.get("dones", [False])[0]:
            info = self.locals.get("infos", [{}])[0]
            
            # Episode-level metrics
            if "episode_reward" in info:
                episode_reward = info["episode_reward"]
                episode_length = info.get("episode_length", 0)
                
                self.episode_rewards.append(episode_reward)
                
                # Log to TensorBoard
                self.logger.record("normarl/episode_reward", episode_reward)
                self.logger.record("normarl/episode_length", episode_length)
                
                if self.verbose > 0:
                    print(f"Episode completed - Reward: {episode_reward:.3f}, Length: {episode_length}")
        
        # Try to access environment state for detailed logging
        try:
            # Get the wrapper environment
            vec_env = self.training_env.envs[0]
            normarl_env = unwrap_env(vec_env)
            
            if normarl_env is None:
                if self.verbose > 1:
                    print(f"[NormarlCallback] Could not unwrap to NORMARL environment")
                return True
            
            # Debug: Print environment type
            if self.verbose > 1:
                print(f"[NormarlCallback] Successfully accessed NORMARL env: {type(normarl_env).__name__}")
            
            # Log resource information
            if hasattr(normarl_env, 'resource_stock'):
                try:
                    resource_stock = normarl_env.resource_stock
                    total_resources = np.sum(resource_stock)
                    self.logger.record("normarl/total_resource_stock", float(total_resources))
                    
                    # Log individual resource stocks
                    for i, stock in enumerate(resource_stock):
                        self.logger.record(f"normarl/resource_{i}_stock", float(stock))
                    
                    if self.verbose > 1:
                        print(f"[NormarlCallback] Logged resource stock: {total_resources:.2f}")
                except Exception as e:
                    if self.verbose > 0:
                        print(f"[NormarlCallback] Error logging resource stock: {e}")
            
            # Log agent information
            if hasattr(normarl_env, 'homeostatic_agents'):
                try:
                    agents = normarl_env.homeostatic_agents
                    
                    if self.verbose > 1:
                        print(f"[NormarlCallback] Found {len(agents)} homeostatic agents")
                    
                    for agent_id, agent in agents.items():
                        try:
                            # Agent drives and states
                            current_drive = agent.get_current_drive()
                            position = agent.position
                            internal_states = agent.internal_states
                            social_norms = agent.perceived_social_norm
                            last_intake = agent.last_intake
                            
                            # Log agent metrics
                            self.logger.record(f"agents/{agent_id}/drive", float(current_drive))
                            self.logger.record(f"agents/{agent_id}/position", float(position))
                            
                            # Log internal states by name
                            state_names = agent.get_state_names()
                            for i, (state_name, state_value) in enumerate(zip(state_names, internal_states)):
                                self.logger.record(f"agents/{agent_id}/states/{state_name}", float(state_value))
                                
                            # Log social norms
                            for i, (state_name, norm_value) in enumerate(zip(state_names, social_norms)):
                                self.logger.record(f"agents/{agent_id}/social_norms/{state_name}", float(norm_value))
                            
                            # Log consumption
                            total_intake = np.sum(last_intake)
                            self.logger.record(f"agents/{agent_id}/total_consumption", float(total_intake))
                            
                            for i, (state_name, intake_value) in enumerate(zip(state_names, last_intake)):
                                self.logger.record(f"agents/{agent_id}/consumption/{state_name}", float(intake_value))
                            
                            # Calculate and log social cost
                            if hasattr(normarl_env, 'resource_stock') and hasattr(normarl_env, 'initial_resource_stock'):
                                # Calculate resource scarcity for social cost
                                normalized_stock = normarl_env.resource_stock / normarl_env.initial_resource_stock
                                scarcity = np.maximum(0, 1.0 - 0.5 * normalized_stock)
                                social_cost = agent.compute_social_cost(last_intake, scarcity)
                                self.logger.record(f"agents/{agent_id}/social_cost", float(social_cost))
                            
                            if self.verbose > 1:
                                print(f"[NormarlCallback] Logged metrics for {agent_id}: drive={current_drive:.3f}, pos={position}")
                                
                        except Exception as e:
                            if self.verbose > 0:
                                print(f"[NormarlCallback] Error logging agent {agent_id}: {e}")
                            
                except Exception as e:
                    if self.verbose > 0:
                        print(f"[NormarlCallback] Error accessing homeostatic agents: {e}")
            
            # Log global environment metrics
            try:
                if hasattr(normarl_env, 'num_moves'):
                    self.logger.record("normarl/environment_steps", int(normarl_env.num_moves))
                
                # Log current agent selection (for AEC environments)
                if hasattr(normarl_env, 'agent_selection') and hasattr(normarl_env, 'agents'):
                    if normarl_env.agent_selection in normarl_env.agents:
                        current_agent_idx = normarl_env.agents.index(normarl_env.agent_selection)
                        self.logger.record("normarl/current_agent_idx", int(current_agent_idx))
                    
            except Exception as e:
                if self.verbose > 0:
                    print(f"[NormarlCallback] Error logging global metrics: {e}")
                    
        except Exception as e:
            if self.verbose > 0:
                print(f"[NormarlCallback] Warning: Could not log detailed metrics: {e}")
                print(f"[NormarlCallback] Environment type: {type(self.training_env.envs[0])}")
                
                # Try to debug the wrapper structure
                current = self.training_env.envs[0]
                depth = 0
                while hasattr(current, 'env') and depth < 10:
                    print(f"[NormarlCallback] Wrapper depth {depth}: {type(current).__name__}")
                    current = current.env
                    depth += 1
                print(f"[NormarlCallback] Final env: {type(current).__name__}")
                if hasattr(current, 'homeostatic_agents'):
                    print(f"[NormarlCallback] Final env HAS homeostatic_agents!")
                else:
                    print(f"[NormarlCallback] Final env does NOT have homeostatic_agents")
        
        # Always dump the logger
        self.logger.dump(self.num_timesteps)
        return True
    
    def _on_training_end(self) -> None:
        """Log summary statistics at the end of training."""
        if self.episode_rewards:
            self.logger.record("normarl/final_mean_reward", np.mean(self.episode_rewards))
            self.logger.record("normarl/final_std_reward", np.std(self.episode_rewards))
            self.logger.record("normarl/final_max_reward", np.max(self.episode_rewards))
            self.logger.record("normarl/final_min_reward", np.min(self.episode_rewards))
            self.logger.record("normarl/total_episodes", len(self.episode_rewards))
        
        self.logger.dump(self.num_timesteps)
    
    def save_metrics(self, filepath):
        """Save collected metrics to a numpy file."""
        data = {
            'episode_rewards': self.episode_rewards,
            'episode_consumptions': self.episode_consumptions,
            'resource_stocks': self.resource_stocks,
            'agent_stats': self.agent_stats
        }
        np.save(filepath, data)
        if self.verbose > 0:
            print(f"Metrics saved to {filepath}")


class PettingZooToGymnasiumWrapper(gym.Env):
    """
    Wrapper to convert PettingZoo environment to Gymnasium format for SB3.
    Handles multi-agent environment by training a single policy for all agents.
    """
    
    def __init__(self, env_fn, **env_kwargs):
        """
        Initialize wrapper.
        
        Args:
            env_fn: Function that creates the PettingZoo environment
            **env_kwargs: Keyword arguments for environment creation
        """
        self.env = env_fn(**env_kwargs)
        self.env_kwargs = env_kwargs
        
        # Reset environment to initialize agents
        self.env.reset()
        
        # Get spaces from first agent (assuming homogeneous agents)
        if hasattr(self.env, 'agents') and len(self.env.agents) > 0:
            agent_id = self.env.agents[0]
            raw_obs_space = self.env.observation_space(agent_id)
            self.action_space = self.env.action_space(agent_id)
            
            # Create flattened observation space
            obs_size = self._get_obs_size()
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
            )
        else:
            raise ValueError("Environment has no agents after reset")
        
        # Multi-agent state tracking
        self.current_agent_idx = 0
        self.agents = self.env.agents.copy()
        self.n_agents = len(self.agents)
        
        # Episode statistics
        self.episode_rewards = []
        self.episode_length = 0
        self.total_episodes = 0
        
    def reset(self, seed=None, options=None):
        """Reset environment and return first agent's observation."""
        observations, info = self.env.reset(seed=seed, options=options)
        self.current_agent_idx = 0
        self.episode_length = 0
        self.episode_rewards = []
        
        # Return observation for current agent
        current_agent = self.env.agent_selection
        obs = self.env.observe(current_agent)
        
        # Flatten observation if it's a dict
        flat_obs = self._flatten_observation(obs)
        
        return flat_obs, info
        
    def step(self, action):
        """Execute action for current agent and return next observation."""
        # Get current agent before step (the one that will act)
        acting_agent = self.env.agent_selection
        
        # Execute action in environment
        self.env.step(action)
        
        # Get reward for the agent that just acted
        reward = self.env.rewards.get(acting_agent, 0.0)
        terminated = self.env.terminations.get(acting_agent, False)
        truncated = self.env.truncations.get(acting_agent, False)
        
        # Track episode statistics
        self.episode_rewards.append(reward)
        self.episode_length += 1
        
        # Get observation for next agent (current agent_selection after step)
        next_agent = self.env.agent_selection
        obs = self.env.observe(next_agent)
        flat_obs = self._flatten_observation(obs)
        
        # Check if episode is done
        episode_done = all(self.env.terminations.values()) or all(self.env.truncations.values())
        
        # Create info dict with useful statistics
        info = {
            'episode_length': self.episode_length,
            'acting_agent': acting_agent,
            'next_agent': next_agent,
            'agent_reward': reward,
            'total_agents': self.n_agents,
            'episode_done': episode_done
        }
        
        # Check if all agents are done (episode ended)
        if episode_done:
            self.total_episodes += 1
            info['episode_reward'] = sum(self.episode_rewards)
            info['episode_number'] = self.total_episodes
            
        return flat_obs, reward, episode_done, False, info
    
    def _flatten_observation(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Flatten dictionary observation to numpy array.
        
        Args:
            obs: Dictionary observation from environment
            
        Returns:
            Flattened numpy array
        """
        if not obs:
            # Return zero observation if empty
            return np.zeros(self._get_obs_size())
        
        flat_components = []
        
        # Handle position (convert to float)
        if 'position' in obs:
            pos = obs['position']
            if isinstance(pos, (int, np.integer)):
                flat_components.append([float(pos)])
            else:
                flat_components.append(pos.flatten())
        
        # Handle internal states
        if 'internal_states' in obs:
            flat_components.append(obs['internal_states'].flatten())
            
        # Handle perceived social norms
        if 'perceived_social_norm' in obs:
            flat_components.append(obs['perceived_social_norm'].flatten())
        
        # Concatenate all components
        if flat_components:
            return np.concatenate(flat_components).astype(np.float32)
        else:
            return np.zeros(self._get_obs_size(), dtype=np.float32)
    
    def _get_obs_size(self) -> int:
        """Calculate the size of flattened observation."""
        # This should match the structure in _flatten_observation
        size = 1  # position
        size += self.env_kwargs.get('number_resources', 3)  # internal_states
        size += self.env_kwargs.get('number_resources', 3)  # perceived_social_norm
        return size
    
    def render(self, mode='human'):
        """Render environment."""
        return self.env.render()
    
    def close(self):
        """Close environment."""
        self.env.close()


def create_normarl_env(**kwargs) -> PettingZooToGymnasiumWrapper:
    """
    Create NORMARL environment with wrapper.
    
    Args:
        **kwargs: Environment creation arguments
        
    Returns:
        Wrapped environment
    """
    def env_fn(**env_kwargs):
        return NormalHomeostaticEnv(**env_kwargs)
    
    return PettingZooToGymnasiumWrapper(env_fn, **kwargs)


def train_ppo_normarl(
    config_path: str = "config/config.yaml",
    drive_type: str = "base_drive",
    learning_rate: float = 0.1,
    beta: float = 0.5,
    number_resources: int = 1,
    n_agents: int = 3,
    size: int = 5,
    total_timesteps: int = 100000,
    n_envs: int = 4,
    eval_freq: int = 10000,
    save_freq: int = 25000,
    log_dir: str = "logs/ppo_normarl",
    model_dir: str = "models/ppo_normarl",
    tensorboard_log: str = "tensorboard_logs/ppo_normarl",
    **ppo_kwargs
):
    """
    Train PPO agent on NORMARL environment.
    
    Args:
        config_path: Path to configuration file
        drive_type: Type of drive system to use
        learning_rate: Social learning rate for agents
        beta: Social norm internalization strength
        number_resources: Number of resource types
        n_agents: Number of agents in environment
        size: Environment size
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        eval_freq: Evaluation frequency
        save_freq: Model save frequency
        log_dir: Directory for logging
        model_dir: Directory for saving models
        tensorboard_log: Directory for tensorboard logs
        **ppo_kwargs: Additional PPO hyperparameters
    """
    
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tensorboard_log, exist_ok=True)
    
    # Environment parameters
    env_kwargs = {
        'config_path': config_path,
        'drive_type': drive_type,
        'learning_rate': learning_rate,
        'beta': beta,
        'number_resources': number_resources,
        'n_agents': n_agents,
        'size': size
    }
    
    print("🏗️ Creating training environments...")
    print(f"Environment parameters: {env_kwargs}")
    
    # Create vectorized environment
    def make_env():
        env = create_normarl_env(**env_kwargs)
        env = Monitor(env)
        return env
    
    # Use DummyVecEnv for now to avoid multiprocessing issues
    train_env = DummyVecEnv([make_env for _ in range(n_envs)])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env])
    
    print(f"✅ Created {n_envs} training environments")
    
    # PPO hyperparameters optimized for multi-agent scenarios
    default_ppo_kwargs = {
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'policy_kwargs': {
            'net_arch': {'pi': [256, 256], 'vf': [256, 256]},
            'activation_fn': torch.nn.Tanh
        },
        'verbose': 1,
        'tensorboard_log': tensorboard_log
    }
    
    # Update with user-provided kwargs
    default_ppo_kwargs.update(ppo_kwargs)
    
    # Handle ppo_lr parameter (rename to learning_rate for PPO)
    if 'ppo_lr' in default_ppo_kwargs:
        default_ppo_kwargs['learning_rate'] = default_ppo_kwargs.pop('ppo_lr')
    
    print("🧠 Creating PPO model...")
    print(f"PPO hyperparameters: {default_ppo_kwargs}")
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        train_env,
        **default_ppo_kwargs
    )
    
    # Set up callbacks
    callbacks = []
    
    # NORMARL metrics callback (for detailed logging)
    normarl_callback = NormarlCallback(verbose=2)  # Increased verbosity for debugging
    callbacks.append(normarl_callback)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, "best_model"),
        log_path=log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        verbose=1
    )
    callbacks.append(eval_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=os.path.join(model_dir, "checkpoints"),
        name_prefix="ppo_normarl",
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    
    callback_list = CallbackList(callbacks)
    
    print(f"🚀 Starting training for {total_timesteps:,} timesteps...")
    start_time = time.time()
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    
    print(f"✅ Training completed in {training_time:.2f} seconds")
    print(f"💾 Saving final model to {model_dir}")
    
    # Save final model
    model.save(os.path.join(model_dir, "final_model"))
    
    # Save training metrics
    metrics_path = os.path.join(model_dir, "training_metrics.npy")
    normarl_callback.save_metrics(metrics_path)
    print(f"📊 Training metrics saved to {metrics_path}")
    
    # Close environments
    train_env.close()
    eval_env.close()
    
    return model, normarl_callback


def evaluate_model(
    model_path: str,
    config_path: str = "config/config.yaml",
    drive_type: str = "base_drive",
    learning_rate: float = 0.1,
    beta: float = 0.5,
    number_resources: int = 1,
    n_agents: int = 3,
    size: int = 1,
    n_episodes: int = 10,
    render: bool = False
):
    """
    Evaluate trained PPO model.
    
    Args:
        model_path: Path to saved model
        config_path: Path to configuration file
        drive_type: Type of drive system
        learning_rate: Social learning rate
        beta: Social norm internalization strength
        number_resources: Number of resource types
        n_agents: Number of agents
        size: Environment size
        n_episodes: Number of evaluation episodes
        render: Whether to render episodes
    """
    
    print(f"📊 Evaluating model: {model_path}")
    
    # Environment parameters
    env_kwargs = {
        'config_path': config_path,
        'drive_type': drive_type,
        'learning_rate': learning_rate,
        'beta': beta,
        'number_resources': number_resources,
        'n_agents': n_agents,
        'size': size
    }
    
    # Create environment
    env = create_normarl_env(**env_kwargs)
    
    # Load model
    model = PPO.load(model_path)
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        print(f"\n📺 Episode {episode + 1}/{n_episodes}")
        
        while not done:
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)
            
            # Execute action
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.render()
                time.sleep(0.1)
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"  Reward: {episode_reward:.3f}, Length: {episode_length}")
    
    # Print evaluation summary
    print(f"\n📈 Evaluation Summary ({n_episodes} episodes):")
    print(f"  Average Reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    print(f"  Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  Best Reward: {np.max(episode_rewards):.3f}")
    print(f"  Worst Reward: {np.min(episode_rewards):.3f}")
    
    env.close()
    
    return {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths)
    }


def main():
    """Main training script with command line arguments."""
    
    parser = argparse.ArgumentParser(description="Train PPO agent on NORMARL environment")
    
    # Environment parameters
    parser.add_argument("--config-path", type=str, default="config/config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--drive-type", type=str, default="base_drive",
                       help="Type of drive system")
    parser.add_argument("--learning-rate", type=float, default=0.1,
                       help="Social learning rate for agents")
    parser.add_argument("--beta", type=float, default=0.5,
                       help="Social norm internalization strength")
    parser.add_argument("--number-resources", type=int, default=1,
                       help="Number of resource types")
    parser.add_argument("--n-agents", type=int, default=3,
                       help="Number of agents")
    parser.add_argument("--size", type=int, default=5,
                       help="Environment size")
    
    # Training parameters
    parser.add_argument("--total-timesteps", type=int, default=100000,
                       help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=4,
                       help="Number of parallel environments")
    parser.add_argument("--eval-freq", type=int, default=10000,
                       help="Evaluation frequency")
    parser.add_argument("--save-freq", type=int, default=25000,
                       help="Model save frequency")
    
    # PPO hyperparameters
    parser.add_argument("--ppo-lr", type=float, default=3e-4,
                       help="PPO learning rate")
    parser.add_argument("--n-steps", type=int, default=2048,
                       help="Number of steps per update")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--n-epochs", type=int, default=10,
                       help="Number of epochs per update")
    
    # Directories
    parser.add_argument("--log-dir", type=str, default="logs/ppo_normarl",
                       help="Logging directory")
    parser.add_argument("--model-dir", type=str, default="models/ppo_normarl",
                       help="Model save directory")
    parser.add_argument("--tensorboard-log", type=str, default="tensorboard_logs/ppo_normarl",
                       help="Tensorboard log directory")
    
    # Evaluation
    parser.add_argument("--evaluate", type=str, default=None,
                       help="Path to model to evaluate (instead of training)")
    parser.add_argument("--eval-episodes", type=int, default=10,
                       help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true",
                       help="Render evaluation episodes")
    
    args = parser.parse_args()
    
    if args.evaluate:
        # Evaluation mode
        evaluate_model(
            model_path=args.evaluate,
            config_path=args.config_path,
            drive_type=args.drive_type,
            learning_rate=args.learning_rate,
            beta=args.beta,
            number_resources=args.number_resources,
            n_agents=args.n_agents,
            size=args.size,
            n_episodes=args.eval_episodes,
            render=args.render
        )
    else:
        # Training mode
        ppo_kwargs = {
            'ppo_lr': args.ppo_lr,  # Use ppo_lr to avoid conflict with social learning_rate
            'n_steps': args.n_steps,
            'batch_size': args.batch_size,
            'n_epochs': args.n_epochs
        }
        
        model, normarl_callback = train_ppo_normarl(
            config_path=args.config_path,
            drive_type=args.drive_type,
            learning_rate=args.learning_rate,
            beta=args.beta,
            number_resources=args.number_resources,
            n_agents=args.n_agents,
            size=args.size,
            total_timesteps=args.total_timesteps,
            n_envs=args.n_envs,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq,
            log_dir=args.log_dir,
            model_dir=args.model_dir,
            tensorboard_log=args.tensorboard_log,
            **ppo_kwargs
        )
        
        print("🎉 Training completed successfully!")
        print(f"📊 View training logs with: tensorboard --logdir {args.tensorboard_log}")
        
        # Print some final statistics
        if normarl_callback.episode_rewards:
            print(f"📈 Final Training Statistics:")
            print(f"  Total Episodes: {len(normarl_callback.episode_rewards)}")
            print(f"  Average Reward: {np.mean(normarl_callback.episode_rewards):.3f}")
            print(f"  Best Episode: {np.max(normarl_callback.episode_rewards):.3f}")
            print(f"  Worst Episode: {np.min(normarl_callback.episode_rewards):.3f}")


if __name__ == "__main__":
    main() 
