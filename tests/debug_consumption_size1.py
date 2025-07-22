"""Debug consumption and rewards with size=1 (all agents at resource)."""

from src.envs.multiagent import create_env
import numpy as np

def debug_consumption_size1():
    print("🔍 Debug: Consumption with Size=1 (All agents at resource)")
    print("=" * 60)
    
    env = create_env(
        config_path="config/config.yaml",
        drive_type="base_drive",
        learning_rate=0.1,
        beta=0.5,
        number_resources=1,
        n_agents=3,
        size=1,  # 🎯 CRITICAL: Only 1 position = all agents at resource
        log_level="INFO"
    )
    
    env.reset(seed=42)
    
    # Check setup
    resource_pos = env.resources_info[0]["position"]
    print(f"🎯 Resource at position: {resource_pos}")
    print(f"🏪 Initial resource stock: {env.resource_stock}")
    print(f"👥 Agents: {env.agents}")
    
    for agent_id in env.agents:
        agent = env.homeostatic_agents[agent_id]
        print(f"   {agent_id} at position: {agent.position}")
    
    print(f"\n🔬 All agents should be at position {resource_pos} (resource position)")
    
    print("\n🧪 Testing Different Actions:")
    
    action_names = {0: "STAY", 1: "LEFT", 2: "RIGHT", 3: "CONSUME"}
    
    for round_num in range(3):
        print(f"\n--- Round {round_num + 1} ---")
        print(f"Resource stock before round: {env.resource_stock}")
        
        round_rewards = []
        round_intakes = []
        round_actions = []
        
        # Test each agent with different actions
        for agent_idx, action in enumerate([0, 1, 3]):  # stay, left, consume
            if not env.agents:
                break
                
            current_agent_id = env.agent_selection
            current_agent = env.homeostatic_agents[current_agent_id]
            
            print(f"\n  {current_agent_id} (action {action}={action_names[action]}):")
            print(f"    Position before: {current_agent.position}")
            print(f"    Internal states before: {current_agent.internal_states}")
            print(f"    Current drive: {current_agent.get_current_drive():.3f}")
            print(f"    Social norm: {current_agent.perceived_social_norm}")
            
            # Execute action
            env.step(action)
            
            # Get results
            reward = env.rewards.get(current_agent_id, 0)
            intake = current_agent.last_intake[0]
            
            print(f"    Position after: {current_agent.position}")
            print(f"    Internal states after: {current_agent.internal_states}")
            print(f"    Intake: {intake:.3f}")
            print(f"    Reward: {reward:.3f}")
            
            round_rewards.append(reward)
            round_intakes.append(intake)
            round_actions.append(action)
        
        print(f"\n  Round {round_num + 1} Summary:")
        print(f"    Actions: {[action_names[a] for a in round_actions]}")
        print(f"    Intakes: {[f'{i:.3f}' for i in round_intakes]}")
        print(f"    Rewards: {[f'{r:.3f}' for r in round_rewards]}")
        print(f"    Resource stock after round: {env.resource_stock}")
        
        # Check if consumption actions gave higher rewards
        consumption_rewards = [r for i, r in enumerate(round_rewards) if round_actions[i] == 3]
        movement_rewards = [r for i, r in enumerate(round_rewards) if round_actions[i] in [0, 1, 2]]
        
        if consumption_rewards:
            avg_consumption = np.mean(consumption_rewards)
            print(f"    Average consumption reward: {avg_consumption:.3f}")
        
        if movement_rewards:
            avg_movement = np.mean(movement_rewards)
            print(f"    Average movement reward: {avg_movement:.3f}")
        
        # Check social norm updates
        print(f"    Social norms after round:")
        for agent_id in env.agents:
            agent = env.homeostatic_agents[agent_id]
            print(f"      {agent_id}: {agent.perceived_social_norm[0]:.3f}")

def test_consumption_vs_movement():
    print("\n🔬 Specific Test: Consumption vs Movement Rewards")
    print("=" * 60)
    
    env = create_env(
        config_path="config/config.yaml",
        drive_type="base_drive",
        learning_rate=0.1,
        beta=0.1,  # Low social cost for clearer homeostatic rewards
        number_resources=1,
        n_agents=2,
        size=1,  # All at resource position
        log_level="ERROR"  # Silent
    )
    
    env.reset(seed=123)
    
    print("Testing systematic consumption vs movement...")
    
    consumption_rewards = []
    movement_rewards = []
    
    # Test consumption actions
    for i in range(4):
        if not env.agents:
            break
            
        current_agent_id = env.agent_selection
        current_agent = env.homeostatic_agents[current_agent_id]
        
        # Alternate between movement and consumption
        if i % 2 == 0:
            action = 0  # movement (stay)
            action_type = "movement"
        else:
            action = 3  # consumption
            action_type = "consumption"
        
        print(f"\nStep {i}: {current_agent_id} doing {action_type} (action {action})")
        print(f"  States before: {current_agent.internal_states}")
        print(f"  Drive before: {current_agent.get_current_drive():.3f}")
        print(f"  Resource stock: {env.resource_stock}")
        
        env.step(action)
        
        reward = env.rewards.get(current_agent_id, 0)
        intake = current_agent.last_intake[0]
        
        print(f"  States after: {current_agent.internal_states}")
        print(f"  Drive after: {current_agent.get_current_drive():.3f}")
        print(f"  Intake: {intake:.3f}")
        print(f"  Reward: {reward:.3f}")
        
        if action_type == "consumption":
            consumption_rewards.append(reward)
        else:
            movement_rewards.append(reward)
    
    print(f"\n📊 Results:")
    print(f"Movement rewards: {movement_rewards}")
    print(f"Consumption rewards: {consumption_rewards}")
    
    if consumption_rewards and movement_rewards:
        avg_consumption = np.mean(consumption_rewards)
        avg_movement = np.mean(movement_rewards)
        
        print(f"Average movement reward: {avg_movement:.3f}")
        print(f"Average consumption reward: {avg_consumption:.3f}")
        print(f"Consumption advantage: {avg_consumption - avg_movement:.3f}")
        
        if avg_consumption > avg_movement:
            print("✅ Consumption gives higher rewards than movement!")
        else:
            print("⚠️  Consumption not giving higher rewards")
    
    print(f"\nFinal resource stock: {env.resource_stock}")

def test_resource_depletion():
    print("\n💰 Test: Resource Depletion and Regeneration")
    print("=" * 60)
    
    env = create_env(
        config_path="config/config.yaml",
        drive_type="base_drive",
        learning_rate=0.1,
        beta=0.1,
        number_resources=1,
        n_agents=3,
        size=1,  # All at resource position
        log_level="ERROR"
    )
    
    env.reset(seed=42)
    
    print(f"Initial resource stock: {env.resource_stock}")
    print(f"Regeneration rate: {env.resource_regeneration_rate}")
    
    # Force all agents to consume
    for round_num in range(3):
        print(f"\n--- Round {round_num + 1} ---")
        print(f"Stock before round: {env.resource_stock}")
        
        round_consumption = 0
        
        # All agents try to consume
        for i in range(len(env.agents)):
            if not env.agents:
                break
                
            current_agent_id = env.agent_selection
            action = 3  # Always consume
            
            stock_before = env.resource_stock[0]
            env.step(action)
            stock_after = env.resource_stock[0]
            
            agent = env.homeostatic_agents[current_agent_id]
            consumption = agent.last_intake[0]
            round_consumption += consumption
            
            print(f"  {current_agent_id}: consumed {consumption:.3f}, stock: {stock_before:.3f} → {stock_after:.3f}")
        
        print(f"Round {round_num + 1} total consumption: {round_consumption:.3f}")
        print(f"Stock after round: {env.resource_stock}")
        
        if env.resource_stock[0] <= 0:
            print("  ⚠️  Resource depleted!")
        else:
            print("  ✅ Resource still available")

if __name__ == "__main__":
    debug_consumption_size1()
    test_consumption_vs_movement()
    test_resource_depletion()
    
    print("\n🎯 Key Insights:")
    print("- With size=1, all position issues are eliminated")
    print("- Consumption should give positive rewards when successful")
    print("- Movement should give ~0 rewards (no drive change)")
    print("- Resource depletion affects consumption success")
    print("- Social norms should evolve based on actual consumption") 
