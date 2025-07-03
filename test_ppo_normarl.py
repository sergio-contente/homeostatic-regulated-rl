#!/usr/bin/env python3
"""
Test script for the adapted PPO implementation on the NORMARL environment.
This script runs a few training episodes to verify everything works correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import torch
from ppo_normarl import NORMARLAgent, batchify_obs, batchify, unbatchify
from src.pettingzoo_env.normarl import NormalHomeostaticEnv

def test_environment_setup():
    """Test that the environment and agent can be initialized correctly."""
    print("🧪 Testing Environment and Agent Setup")
    print("=" * 50)
    
    try:
        # Environment parameters
        config_path = "config/config.yaml"
        drive_type = "base_drive"
        learning_rate = 0.1
        beta = 0.5
        number_resources = 2
        n_agents = 3
        size = 5
        
        # Create environment
        env = NormalHomeostaticEnv(
            config_path=config_path,
            drive_type=drive_type,
            learning_rate=learning_rate,
            beta=beta,
            number_resources=number_resources,
            n_agents=n_agents,
            size=size
        )
        print("✅ Environment created successfully")
        
        # Reset environment to initialize action and observation spaces
        observations, info = env.reset()
        print("✅ Environment reset successfully")
        
        # Get environment specs
        num_agents = len(env.agents)
        num_actions = env.action_space(env.agents[0]).n
        observation_space = env.observation_space(env.agents[0])
        
        print(f"📊 Number of agents: {num_agents}")
        print(f"📊 Number of actions: {num_actions}")
        print(f"📊 Observation space: {observation_space}")
        
        # Create agent
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent = NORMARLAgent(observation_space, env.action_space(env.possible_agents[0])).to(device)
        print("✅ Agent created successfully")
        
        return env, agent, device
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def test_single_episode(env, agent, device):
    """Test running a single episode with the agent."""
    print("\n🧪 Testing Single Episode")
    print("=" * 50)
    
    if env is None or agent is None:
        print("❌ Skipping episode test - setup failed")
        return False
    
    try:
        # Reset environment
        obs, info = env.reset(seed=42)
        print("✅ Environment reset successfully")
        
        total_reward = 0
        step_count = 0
        max_steps = 100
        
        # Run episode
        for step in range(max_steps):
            # Convert observations to batch format
            batch_obs = batchify_obs(obs, device)
            
            # Get agent action
            with torch.no_grad():
                actions, logprobs, entropy, values = agent.get_action_and_value(batch_obs)
            
            # Execute action
            next_obs, rewards, terms, truncs, infos = env.step(unbatchify(actions, env))
            
            # Update total reward
            episode_reward = sum(rewards.values())
            total_reward += episode_reward
            
            print(f"Step {step}: Reward = {episode_reward:.3f}, Actions = {actions.cpu().numpy()}")
            
            # Check termination
            if any(terms.values()) or any(truncs.values()):
                print(f"Episode ended at step {step}")
                break
            
            obs = next_obs
            step_count = step + 1
        
        print(f"✅ Episode completed successfully")
        print(f"📊 Total steps: {step_count}")
        print(f"📊 Total reward: {total_reward:.3f}")
        print(f"📊 Average reward per step: {total_reward/step_count:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Episode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_step(env, agent, device):
    """Test a single training step."""
    print("\n🧪 Testing Training Step")
    print("=" * 50)
    
    if env is None or agent is None:
        print("❌ Skipping training test - setup failed")
        return False
    
    try:
        import torch.optim as optim
        
        # Setup optimizer
        optimizer = optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5)
        
        # Collect some experience
        obs, info = env.reset(seed=42)
        batch_obs = batchify_obs(obs, device)
        
        # Get initial action and value
        actions, logprobs, entropy, values = agent.get_action_and_value(batch_obs)
        
        # Execute action
        next_obs, rewards, terms, truncs, infos = env.step(unbatchify(actions, env))
        
        # Convert to tensors
        rewards_tensor = batchify(rewards, device)
        terms_tensor = batchify(terms, device)
        
        # Simple value loss
        value_loss = 0.5 * ((values.flatten() - rewards_tensor) ** 2).mean()
        
        # Policy loss (simplified)
        policy_loss = -logprobs.mean()
        
        # Total loss
        loss = value_loss + policy_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print("✅ Training step completed successfully")
        print(f"📊 Value loss: {value_loss.item():.6f}")
        print(f"📊 Policy loss: {policy_loss.item():.6f}")
        print(f"📊 Total loss: {loss.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Starting PPO NORMARL Tests")
    print("=" * 60)
    
    # Test 1: Environment and agent setup
    env, agent, device = test_environment_setup()
    
    # Test 2: Single episode
    episode_success = test_single_episode(env, agent, device)
    
    # Test 3: Training step
    training_success = test_training_step(env, agent, device)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    print(f"Setup: {'✅ PASS' if env is not None else '❌ FAIL'}")
    print(f"Episode: {'✅ PASS' if episode_success else '❌ FAIL'}")
    print(f"Training: {'✅ PASS' if training_success else '❌ FAIL'}")
    
    if env is not None and episode_success and training_success:
        print("\n🎉 All tests passed! The PPO implementation is ready to use.")
        print("You can now run the full training with: python ppo_normarl.py")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 
