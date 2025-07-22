"""Test regeneration with only agent_0 consuming while others move - FIXED."""

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
        log_level="INFO"  # Reduzindo verbosidade
    )
    
    env.reset(seed=42)
    print(f"📦 Initial resource: {env.resource_stock}")
    print(f"👥 Agents: {env.agents}")
    
    # Track data
    round_data = []
    max_rounds = 8
    
    for round_num in range(max_rounds):
        if not env.agents:
            break
            
        print(f"\n🎯 Round {round_num + 1}")
        print("-" * 40)
        
        round_start_resource = env.resource_stock.copy()
        agents_data = {}
        
        # Complete one round (all 3 agents act)
        for step in range(3):
            if not env.agents:
                break
                
            agent_id = env.agent_selection
            
            # Get agent state before action
            agent = env.homeostatic_agents[agent_id]
            before_states = agent.internal_states.copy()
            before_drive = agent.get_current_drive()  # ✅ FIXED: Correct method name
            before_resource = env.resource_stock.copy()
            before_norm = env.social_norms[agent_id].copy()
            
            # Determine action based on agent
            if agent_id == "agent_0":
                action = 3  # Always consume
                action_type = "consume"
            else:
                action = np.random.choice([0, 1, 2])  # Only movement
                action_type = f"move_{action}"
            
            print(f"  {agent_id}: {action_type} (action {action})")
            
            # Execute action
            env.step(action)
            
            # Get state after action
            if agent_id in env.agents:  # Agent might be terminated
                after_states = agent.internal_states.copy()
                after_drive = agent.get_current_drive()  # ✅ FIXED: Correct method name
                intake = agent.last_intake.copy()
                after_norm = env.social_norms[agent_id].copy()
            else:
                after_states = before_states
                after_drive = before_drive
                intake = np.zeros_like(before_states)
                after_norm = before_norm
                
            after_resource = env.resource_stock.copy()
            reward = env.rewards[agent_id]
            
            # Store agent data
            agents_data[agent_id] = {
                "action": action,
                "action_type": action_type,
                "states_before": before_states.copy(),
                "states_after": after_states.copy(),
                "drive_before": before_drive,
                "drive_after": after_drive,
                "intake": intake.copy(),
                "reward": reward,
                "social_norm_before": before_norm.copy(),
                "social_norm_after": after_norm.copy(),
                "resource_before": before_resource.copy(),
                "resource_after": after_resource.copy()
            }
            
            # Print step details
            print(f"    States: {before_states} → {after_states}")
            print(f"    Drive: {before_drive:.3f} → {after_drive:.3f}")
            print(f"    Intake: {intake}")
            print(f"    Reward: {reward:.3f}")
            print(f"    Resource: {before_resource} → {after_resource}")
            print(f"    Social norm: {before_norm} → {after_norm}")
            
        round_end_resource = env.resource_stock.copy()
        
        # Round summary
        resource_change = round_end_resource - round_start_resource
        
        round_data.append({
            "round": round_num + 1,
            "start_resource": round_start_resource.copy(),
            "end_resource": round_end_resource.copy(),
            "resource_change": resource_change.copy(),
            "agents_data": agents_data.copy(),
            "active_agents": len(env.agents)
        })
        
        print(f"\n📊 Round {round_num + 1} Summary:")
        print(f"  Resource: {round_start_resource} → {round_end_resource} (change: {resource_change})")
        print(f"  Active agents: {len(env.agents)}")
        
        # Calculate consumption and regeneration
        total_consumption = sum([np.sum(data["intake"]) for data in agents_data.values()])
        net_change = np.sum(resource_change)
        regeneration = net_change + total_consumption
        
        print(f"  Total consumption: {total_consumption:.3f}")
        print(f"  Net regeneration: {regeneration:.3f}")
        print(f"  Net change: {net_change:.3f}")
        
        if total_consumption > 0:
            balance = regeneration / total_consumption
            print(f"  Regeneration/Consumption ratio: {balance:.2f}")
            if balance > 1:
                print("  🌱 Sustainable (regeneration > consumption)")
            elif balance == 1:
                print("  ⚖️ Balanced (regeneration = consumption)")
            else:
                print("  📉 Unsustainable (regeneration < consumption)")
        
        # Social norm analysis
        consumer_norm = env.social_norms["agent_0"] if "agent_0" in env.agents else np.array([0.0])
        non_consumer_norms = [env.social_norms[aid] for aid in env.agents if aid != "agent_0"]
        
        print(f"  Consumer norm (agent_0): {consumer_norm}")
        if non_consumer_norms:
            avg_non_consumer_norm = np.mean(non_consumer_norms, axis=0)
            print(f"  Non-consumer avg norm: {avg_non_consumer_norm}")
    
    # Final analysis
    print(f"\n🏁 Final Results after {len(round_data)} rounds:")
    print("=" * 60)
    print(f"📦 Final resource stock: {env.resource_stock}")
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
    
    # Social norm evolution
    if round_data:
        initial_norms = round_data[0]["agents_data"].get("agent_0", {}).get("social_norm_after", [0])
        final_norms = round_data[-1]["agents_data"].get("agent_0", {}).get("social_norm_after", [0])
        norm_change = final_norms - initial_norms
        print(f"🧠 Social norm evolution (agent_0): {initial_norms} → {final_norms} (change: {norm_change})")
    
    return round_data

def analyze_sustainability(round_data):
    """Analyze the sustainability of the system."""
    print(f"\n🔬 Sustainability Analysis:")
    print("=" * 50)
    
    total_consumption = 0
    total_regeneration = 0
    
    for data in round_data:
        round_consumption = sum([np.sum(agent_data["intake"]) for agent_data in data["agents_data"].values()])
        round_regeneration = np.sum(data["resource_change"]) + round_consumption
        
        total_consumption += round_consumption
        total_regeneration += round_regeneration
        
        print(f"Round {data['round']}: consumption={round_consumption:.3f}, regeneration={round_regeneration:.3f}")
    
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

if __name__ == "__main__":
    # First apply the regeneration fix
    print("🔧 Make sure you've applied the regeneration fix to multiagent.py!")
    print("Replace the _check_resource_regeneration method with Option 1")
    print()
    
    round_data = test_single_consumer()
    analyze_sustainability(round_data) 
