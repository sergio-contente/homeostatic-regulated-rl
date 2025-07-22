"""Test the fixed reward calculation."""

from src.envs.multiagent import create_env
import numpy as np

def test_movement_decay_reward():
    print("🚶 Testing Movement Decay Reward (should be non-zero)")
    print("=" * 60)
    
    env = create_env(
        config_path="config/config.yaml",
        drive_type="base_drive",
        learning_rate=0.1,
        beta=0.1,  # Low social cost
        number_resources=1,
        n_agents=1,
        size=1,
        log_level="INFO"
    )
    
    env.reset(seed=42)
    
    agent_id = env.agents[0]
    agent = env.homeostatic_agents[agent_id]
    
    print(f"Initial state: {agent.internal_states}")
    print(f"Initial drive: {agent.get_current_drive():.6f}")
    
    # Test movement action (should have decay reward)
    action = 0  # stay (pure movement, no consumption)
    env.step(action)
    
    reward = env.rewards.get(agent_id, 0)
    
    print(f"Final state: {agent.internal_states}")
    print(f"Final drive: {agent.get_current_drive():.6f}")
    print(f"Reward: {reward:.6f}")
    
    if abs(reward) > 1e-3:
        print("✅ Movement gives non-zero reward (decay effect captured!)")
    else:
        print("❌ Movement reward still zero - check implementation")
    
    return reward

def test_consumption_reward():
    print("\n🍽️ Testing Consumption Reward (should be different from movement)")
    print("=" * 60)
    
    env = create_env(
        config_path="config/config.yaml",
        drive_type="base_drive",
        learning_rate=0.1,
        beta=0.1,
        number_resources=1,
        n_agents=1,
        size=1,
        log_level="INFO"
    )
    
    env.reset(seed=42)
    
    agent_id = env.agents[0]
    agent = env.homeostatic_agents[agent_id]
    
    # Force a state that will benefit from intake
    agent.internal_states = np.array([-0.5])
    
    print(f"Forced initial state: {agent.internal_states}")
    print(f"Initial drive: {agent.get_current_drive():.6f}")
    
    # Test consumption action
    action = 3  # consume
    env.step(action)
    
    reward = env.rewards.get(agent_id, 0)
    intake = agent.last_intake[0]
    
    print(f"Final state: {agent.internal_states}")
    print(f"Final drive: {agent.get_current_drive():.6f}")
    print(f"Intake: {intake:.3f}")
    print(f"Reward: {reward:.6f}")
    
    if intake > 0 and reward > 0:
        print("✅ Consumption gives positive reward!")
    else:
        print("❌ Consumption reward issue - check intake or calculation")
    
    return reward

def test_reward_comparison():
    print("\n📊 Comparing Movement vs Consumption Rewards")
    print("=" * 60)
    
    movement_reward = test_movement_decay_reward()
    consumption_reward = test_consumption_reward()
    
    print(f"\nSummary:")
    print(f"Movement reward: {movement_reward:.6f}")
    print(f"Consumption reward: {consumption_reward:.6f}")
    
    if abs(movement_reward) > 1e-3:
        print("✅ Movement captures decay effects")
    else:
        print("❌ Movement not capturing decay")
    
    if consumption_reward > movement_reward:
        print("✅ Consumption generally more rewarding than pure movement")
    else:
        print("⚠️ Consumption not more rewarding - check parameters")

if __name__ == "__main__":
    test_reward_comparison()
    
    print("\n🎯 Expected Behavior:")
    print("- Movement actions should give small negative rewards (due to decay)")
    print("- Consumption actions should give positive rewards (when beneficial)")
    print("- Reward = (decay effect + intake effect) - social cost") 
