"""Test regeneration with only agent_0 consuming - SIMPLIFIED."""

from src.envs.multiagent import create_env
import numpy as np

def test_single_consumer():
    print("🧪 Testing Single Consumer Scenario")
    print("=" * 60)
    print("📋 Scenario: Only agent_0 consumes, others move only")
    
    env = create_env(
        config_path="config/config.yaml",
        drive_type="base_drive",
        learning_rate=0.1,
        beta=0.5,
        number_resources=1,
        n_agents=3,
        size=3,  # size=3 para permitir movimento
        log_level="INFO"
    )
    
    env.reset(seed=42)
    
    # Access the base environment through wrappers
    base_env = env.env.env  # OrderEnforcing -> AssertOutOfBounds -> NormalHomeostaticEnv
    
    print(f"📦 Initial resource: {base_env.resource_stock}")
    print(f"👥 Agents: {env.agents}")
    
    # Track data
    round_data = []
    max_rounds = 8
    
    for round_num in range(max_rounds):
        if not env.agents:
            break
            
        print(f"\n🎯 Round {round_num + 1}")
        print("-" * 40)
        
        round_start_resource = base_env.resource_stock.copy()
        round_actions = {}
        round_rewards = {}
        
        # Complete one round (all 3 agents act)
        for step in range(3):
            if not env.agents:
                break
                
            agent_id = env.agent_selection
            
            # Get agent state before action
            agent = base_env.homeostatic_agents[agent_id]
            before_states = agent.internal_states.copy()
            before_drive = agent.get_current_drive()
            before_resource = base_env.resource_stock.copy()
            
            # Determine action based on agent
            if agent_id == "agent_0":
                action = 3  # Always consume
                action_type = "consume"
            else:
                action = np.random.choice([0, 1, 2])  # Only movement
                action_type = f"move_{action}"
            
            print(f"  {agent_id}: {action_type} (action {action})")
            print(f"    Before - States: {before_states}, Drive: {before_drive:.3f}, Resource: {before_resource}")
            
            # Execute action
            env.step(action)
            
            # Get state after action
            if agent_id in env.agents:  # Agent might be terminated
                after_states = agent.internal_states.copy()
                after_drive = agent.get_current_drive()
                intake = agent.last_intake.copy()
            else:
                after_states = before_states
                after_drive = before_drive
                intake = np.zeros_like(before_states)
                
            after_resource = base_env.resource_stock.copy()
            reward = env.rewards[agent_id]
            
            # Store data
            round_actions[agent_id] = {
                "action": action,
                "action_type": action_type,
                "intake": intake.copy(),
                "reward": reward
            }
            
            print(f"    After  - States: {after_states}, Drive: {after_drive:.3f}, Resource: {after_resource}")
            print(f"    Intake: {intake}, Reward: {reward:.3f}")
            
        round_end_resource = base_env.resource_stock.copy()
        
        # Round summary
        resource_change = round_end_resource - round_start_resource
        total_consumption = sum([np.sum(data["intake"]) for data in round_actions.values()])
        net_change = np.sum(resource_change)
        regeneration = net_change + total_consumption
        
        round_data.append({
            "round": round_num + 1,
            "start_resource": round_start_resource.copy(),
            "end_resource": round_end_resource.copy(),
            "resource_change": resource_change.copy(),
            "total_consumption": total_consumption,
            "regeneration": regeneration,
            "net_change": net_change,
            "actions": round_actions.copy(),
            "active_agents": len(env.agents)
        })
        
        print(f"\n📊 Round {round_num + 1} Summary:")
        print(f"  Resource: {round_start_resource} → {round_end_resource} (change: {resource_change})")
        print(f"  Total consumption: {total_consumption:.3f}")
        print(f"  Net regeneration: {regeneration:.3f}")
        print(f"  Net change: {net_change:.3f}")
        print(f"  Active agents: {len(env.agents)}")
        
        if total_consumption > 0:
            balance = regeneration / total_consumption
            print(f"  Regeneration/Consumption ratio: {balance:.2f}")
            if balance > 1:
                print("  🌱 Sustainable (regeneration > consumption)")
            elif balance == 1:
                print("  ⚖️ Balanced (regeneration = consumption)")
            else:
                print("  📉 Unsustainable (regeneration < consumption)")
        
        # Social norms (if accessible)
        try:
            if hasattr(base_env, 'social_norms'):
                consumer_norm = base_env.social_norms.get("agent_0", np.array([0.0]))
                print(f"  Consumer social norm (agent_0): {consumer_norm}")
        except:
            print("  (Social norms not accessible)")
    
    # Final analysis
    print(f"\n🏁 Final Results after {len(round_data)} rounds:")
    print("=" * 60)
    print(f"📦 Final resource stock: {base_env.resource_stock}")
    print(f"👥 Agents remaining: {len(env.agents)}")
    
    if len(env.agents) == 0:
        print("💀 System collapsed - all agents terminated")
    else:
        print("✅ System survived")
        
    # Resource trend analysis
    resource_history = [data["end_resource"][0] for data in round_data]
    print(f"📈 Resource history: {[f'{r:.2f}' for r in resource_history]}")
    
    if len(resource_history) > 1:
        trend = resource_history[-1] - resource_history[0]
        if trend > 0:
            print(f"📈 Resource trend: +{trend:.3f} (growing)")
        elif trend < 0:
            print(f"📉 Resource trend: {trend:.3f} (declining)")
        else:
            print("⚖️ Resource trend: stable")
    
    return round_data

def analyze_sustainability(round_data):
    """Analyze the sustainability of the system."""
    print(f"\n🔬 Sustainability Analysis:")
    print("=" * 50)
    
    total_consumption = sum([data["total_consumption"] for data in round_data])
    total_regeneration = sum([data["regeneration"] for data in round_data])
    
    for data in round_data:
        print(f"Round {data['round']}: consumption={data['total_consumption']:.3f}, regeneration={data['regeneration']:.3f}")
    
    print(f"\nTotals:")
    print(f"  Total consumption: {total_consumption:.3f}")
    print(f"  Total regeneration: {total_regeneration:.3f}")
    
    if total_consumption > 0:
        efficiency = total_regeneration / total_consumption
        print(f"  System efficiency: {efficiency:.3f}")
        
        if efficiency > 1.1:
            print("  🌱 Highly sustainable system")
        elif efficiency > 0.9:
            print("  ⚖️ Balanced system")  
        else:
            print("  ⚠️ Unsustainable system")

def test_regeneration_mechanism():
    """Test the regeneration mechanism specifically."""
    print("\n🔬 Testing Regeneration Mechanism:")
    print("=" * 50)
    
    env = create_env(
        config_path="config/config.yaml",
        drive_type="base_drive",
        learning_rate=0.1,
        beta=0.5,
        number_resources=1,
        n_agents=1,  # Just 1 agent to isolate regeneration
        size=1,
        log_level="DEBUG"
    )
    
    env.reset(seed=42)
    base_env = env.env.env
    
    print(f"Initial resource: {base_env.resource_stock}")
    
    # Test: No consumption for one round (only movement)
    print("\n1️⃣ Test: No consumption round")
    for _ in range(1):  # 1 agent, 1 action
        before = base_env.resource_stock.copy()
        env.step(0)  # Movement action
        after = base_env.resource_stock.copy()
        print(f"  Resource: {before} → {after} (regeneration only)")
    
    # Test: Consumption round
    print("\n2️⃣ Test: Consumption round")
    for _ in range(1):  # 1 agent, 1 action
        before = base_env.resource_stock.copy()
        env.step(3)  # Consumption action
        after = base_env.resource_stock.copy()
        agent = base_env.homeostatic_agents["agent_0"]
        intake = np.sum(agent.last_intake)
        print(f"  Resource: {before} → {after} (consumption: {intake:.3f})")

if __name__ == "__main__":
    print("🔧 Make sure you've applied the regeneration fix to multiagent.py!")
    print("Replace the _check_resource_regeneration method with Option 1")
    print()
    
    # Test regeneration mechanism first
    test_regeneration_mechanism()
    
    # Then run the full single consumer test
    round_data = test_single_consumer()
    analyze_sustainability(round_data) 
