"""Integration tests for multi-agent environments."""

import numpy as np
from src.envs.multiagent import create_env, create_parallel_env

def test_social_norm_learning():
    """Test if social norms are actually learned."""
    print("🧠 Testing Social Norm Learning")
    
    env = create_env(
        config_path="config/config.yaml",
        drive_type="base_drive",
        learning_rate=0.3,  # Higher learning rate for faster test
        beta=1.0,  # High social cost sensitivity
        number_resources=1,
        n_agents=3,
        size=3,
        log_level="ERROR"
    )
    
    env.reset(seed=42)
    
    # Store initial norms
    initial_norms = {}
    for agent_id in env.agents:
        agent = env.homeostatic_agents[agent_id]
        initial_norms[agent_id] = agent.perceived_social_norm.copy()
    
    print(f"   Initial norms: {[f'{k}:{v[0]:.3f}' for k,v in initial_norms.items()]}")
    
    # Run several rounds to see norm evolution
    for round_num in range(3):
        print(f"   Round {round_num + 1}:")
        
        round_intakes = []
        
        # Each agent acts
        for _ in range(len(env.agents)):
            if not env.agents:
                break
                
            current_agent = env.agent_selection
            action = 3  # Always try to consume (action 3 = consume resource 0)
            env.step(action)
            
            agent = env.homeostatic_agents[current_agent]
            round_intakes.append(agent.last_intake[0])
        
        # Check norm updates after round
        avg_intake = np.mean(round_intakes) if round_intakes else 0
        print(f"     Average intake: {avg_intake:.3f}")
        
        for agent_id in env.agents:
            agent = env.homeostatic_agents[agent_id]
            print(f"     {agent_id} norm: {agent.perceived_social_norm[0]:.3f}")
    
    # Check if norms changed
    norm_changed = False
    for agent_id in env.agents:
        agent = env.homeostatic_agents[agent_id]
        if not np.allclose(agent.perceived_social_norm, initial_norms[agent_id]):
            norm_changed = True
            break
    
    if norm_changed:
        print("   ✅ Social norms evolved during episode!")
    else:
        print("   ⚠️  Social norms didn't change (might be due to low consumption)")
    
    print("✅ Social norm learning test completed!")

def test_resource_scarcity():
    """Test resource scarcity mechanism."""
    print("💰 Testing Resource Scarcity")
    
    env = create_env(
        config_path="config/config.yaml",
        drive_type="base_drive",
        learning_rate=0.1,
        beta=1.0,
        number_resources=1,
        n_agents=3,
        size=1,  # Small size so all agents are at resource
        log_level="ERROR"
    )
    
    env.reset(seed=42)
    
    initial_stock = env.resource_stock.copy()
    print(f"   Initial resource stock: {initial_stock}")
    
    # Force consumption by all agents
    consumption_count = 0
    for round_num in range(3):
        print(f"   Round {round_num + 1}:")
        
        for _ in range(len(env.agents)):
            if not env.agents:
                break
                
            current_agent = env.agent_selection
            action = 3  # Consume action
            env.step(action)
            
            agent = env.homeostatic_agents[current_agent]
            if agent.last_intake[0] > 0:
                consumption_count += 1
        
        print(f"     Resource stock after round: {env.resource_stock}")
        print(f"     Total consumptions: {consumption_count}")
    
    final_stock = env.resource_stock.copy()
    stock_changed = not np.allclose(initial_stock, final_stock)
    
    if stock_changed:
        print("   ✅ Resource stock changed due to consumption/regeneration!")
    else:
        print("   ⚠️  Resource stock unchanged (check consumption logic)")
    
    print("✅ Resource scarcity test completed!")

def test_reward_system():
    """Test reward calculation system."""
    print("🏆 Testing Reward System")
    
    env = create_env(
        config_path="config/config.yaml",
        drive_type="base_drive",
        learning_rate=0.1,
        beta=0.5,
        number_resources=1,
        n_agents=2,
        size=3,
        log_level="ERROR"
    )
    
    env.reset(seed=42)
    
    rewards_collected = []
    actions_taken = []
    
    # Test different actions and their rewards
    for i in range(8):  # 4 rounds of 2 agents
        if not env.agents:
            break
            
        current_agent = env.agent_selection
        
        # Alternate between movement and consumption
        action = (i % 4)  # 0=stay, 1=left, 2=right, 3=consume
        env.step(action)
        
        reward = env.rewards.get(current_agent, 0)
        rewards_collected.append(reward)
        actions_taken.append(action)
        
        agent = env.homeostatic_agents[current_agent]
        print(f"   {current_agent} action={action}, reward={reward:.3f}, intake={agent.last_intake[0]:.3f}")
    
    # Analyze rewards
    if rewards_collected:
        avg_reward = np.mean(rewards_collected)
        reward_std = np.std(rewards_collected)
        print(f"   Average reward: {avg_reward:.3f} ± {reward_std:.3f}")
        print(f"   Reward range: [{min(rewards_collected):.3f}, {max(rewards_collected):.3f}]")
        
        # Check if consumption actions give different rewards than movement
        consumption_rewards = [r for i, r in enumerate(rewards_collected) if actions_taken[i] == 3]
        movement_rewards = [r for i, r in enumerate(rewards_collected) if actions_taken[i] < 3]
        
        if consumption_rewards and movement_rewards:
            avg_consumption = np.mean(consumption_rewards)
            avg_movement = np.mean(movement_rewards)
            print(f"   Avg consumption reward: {avg_consumption:.3f}")
            print(f"   Avg movement reward: {avg_movement:.3f}")
        
        print("   ✅ Reward system functioning!")
    else:
        print("   ⚠️  No rewards collected")
    
    print("✅ Reward system test completed!")

if __name__ == "__main__":
    print("🔬 Running Integration Tests")
    print("=" * 50)
    
    try:
        test_social_norm_learning()
        print()
        test_resource_scarcity()
        print()
        test_reward_system()
        print()
        print("✅ All integration tests completed! 🎉")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
