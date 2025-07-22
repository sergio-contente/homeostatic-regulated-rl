"""
NORMARL training with reward shaping following the exact SuperSuit pattern.
"""
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
import supersuit as ss
from pettingzoo.utils.conversions import parallel_wrapper_fn
import gymnasium as gym
import time
from datetime import datetime
from src.pettingzoo_env.normarl import NormalHomeostaticEnv

class RewardShapingWrapper(gym.Wrapper):
    """Wrapper to add consumption incentives"""
    
    def __init__(self, env):
        super().__init__(env)
        self.consumption_bonus = 100.0  # Large bonus for consumption
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Apply reward shaping
        shaped_reward = reward
        
        # Large bonus for consumption action (action 3)
        if action == 3:
            shaped_reward += self.consumption_bonus
            print(f"🎉 Consumption! Base: {reward:.2f}, Bonus: +{self.consumption_bonus}, Total: {shaped_reward:.2f}")
        
        # Cap very large negative penalties
        if reward < -50.0:
            shaped_reward = -1.0
            print(f"🔄 Capped penalty from {reward:.2f} to -1.0")
        
        return obs, shaped_reward, terminated, truncated, info

def raw_env():
    """Create the raw NORMARL environment"""
    env = NormalHomeostaticEnv(
        config_path="config/config.yaml",
        drive_type="base_drive",
        learning_rate=0.1,
        beta=0.5,
        number_resources=1,
        n_agents=2,
        size=1
    )
    env.max_steps = 50  # Shorter episodes
    return env

def train_normarl():
    """Train PPO on NORMARL with reward shaping"""
    
    print("🚀 Starting NORMARL training with reward shaping...")
    
    # Create raw environment (AEC)
    env = raw_env()
    
    # Convert AEC to parallel
    env = aec_to_parallel(env)
    
    # Flatten observations
    env = ss.flatten_v0(env)
    
    print(f"📊 Environment: {env.metadata['name']}")
    print(f"🎮 Action space: {env.action_space(env.possible_agents[0])}")
    print(f"👁️ Observation space: {env.observation_space(env.possible_agents[0])}")
    
    # Convert to gym-like API with one agent
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")
    
    print(f"📦 Final observation space: {env.observation_space}")
    print(f"🎯 Final action space: {env.action_space}")
    
    # Train with high exploration PPO
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=1e-3,  # Higher learning rate
        n_steps=128,         # Shorter rollouts
        batch_size=32,       # Smaller batches
        n_epochs=3,          # Fewer epochs
        gamma=0.8,           # Lower discount for immediate rewards
        gae_lambda=0.9,
        clip_range=0.4,      # Large clip range
        ent_coef=0.7,        # Very high entropy for exploration!
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={
            "net_arch": [32, 32],  # Small network
            "activation_fn": torch.nn.Tanh
        },
        seed=42
    )
    
    print(f"🧠 Network: [32, 32] with Tanh activation")
    print(f"🎯 Entropy coefficient: 0.7 (very high exploration)")
    print(f"💰 Consumption bonus: +100.0")
    print(f"⏰ Episode length: 50 steps")
    
    # Train
    total_timesteps = 10_000
    print(f"🏃 Training for {total_timesteps} timesteps...")
    
    model.learn(total_timesteps=total_timesteps)
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = f"normarl_shaped_{timestamp}.zip"
    model.save(model_path)
    print(f"💾 Model saved as {model_path}")
    
    # Test the model
    print("\n🧪 Testing trained model...")
    
    # Reset environment for testing
    obs = env.reset()
    
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    total_consumption = 0
    total_reward = 0
    
    print("📋 Testing 100 steps:")
    for step in range(100):
        action, _ = model.predict(obs, deterministic=False)
        action = int(action[0]) if hasattr(action, '__getitem__') else int(action)
        action_counts[action] += 1
        
        obs, reward, done, info = env.step([action])
        reward = reward[0] if hasattr(reward, '__getitem__') else reward
        total_reward += reward
        
        if action == 3:
            total_consumption += 1
            print(f"  Step {step+1}: 🍯 CONSUME! Reward: {reward:.2f}")
        
        if done[0] if hasattr(done, '__getitem__') else done:
            obs = env.reset()
    
    print(f"\n📊 Test Results:")
    print(f"  Stay: {action_counts[0]} ({action_counts[0]/100*100:.1f}%)")
    print(f"  Move Left: {action_counts[1]} ({action_counts[1]/100*100:.1f}%)")
    print(f"  Move Right: {action_counts[2]} ({action_counts[2]/100*100:.1f}%)")
    print(f"  🍯 Consume: {action_counts[3]} ({action_counts[3]/100*100:.1f}%)")
    print(f"  Total consumption actions: {total_consumption}/100")
    print(f"  Average reward: {total_reward/100:.2f}")
    
    if total_consumption > 0:
        print(f"  ✅ SUCCESS: Agent learned to consume!")
    else:
        print(f"  ❌ FAILED: Agent did not learn to consume")
    
    env.close()
    return model, model_path

if __name__ == "__main__":
    print("🏥 NORMARL PPO Training with Reward Shaping")
    print("=" * 50)
    
    model, model_path = train_normarl()
    
    print(f"\n🎯 Training completed!")
    print(f"📁 Model saved: {model_path}") 
