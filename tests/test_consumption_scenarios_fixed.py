"""
Test consumption scenarios: single consumer vs all consumers.
Compare sustainability under different consumption pressures.
"""

from src.envs.multiagent import create_env
import numpy as np

def test_scenario_comparison():
    """Compare single consumer vs all consumers scenarios."""
    print("🧪 Testing Consumption Scenarios Comparison")
    print("=" * 70)
    
    scenarios = [
        {"name": "Single Consumer", "description": "Only agent_0 consumes"},
        {"name": "All Consumers", "description": "All agents consume"}
    ]
    
    results = {}
    
    for scenario in scenarios:
        print(f"\n📊 Scenario: {scenario['name']}")
        print(f"📋 {scenario['description']}")
        print("=" * 50)
        
        # Create fresh environment
        env = create_env(
            config_path="config/config.yaml",
            drive_type="base_drive",
            learning_rate=0.1,
            beta=0.5,
            number_resources=1,
            n_agents=3,
            size=1,  # All agents at resource
            log_level="INFO"
        )
        
        env.reset(seed=42)
        base_env = env.env.env
        
        # Set consistent initial conditions
        initial_stock = 10.0
        base_env.resource_stock = np.array([initial_stock])
        
        print(f"📦 Initial resource stock: {initial_stock}")
        print(f"🌱 Regeneration rate: {base_env.global_resource_manager.get_resource_stock_regeneration_array()}")
        
        # Track data
        resource_history = [initial_stock]
        consumption_history = []
        reward_history = []
        max_rounds = 15
        
        for round_num in range(max_rounds):
            if not env.agents:
                print(f"💀 All agents terminated at round {round_num}")
                break
                
            round_start_stock = base_env.resource_stock[0]
            round_consumption = 0
            round_rewards = {}
            
            print(f"\n🎯 Round {round_num + 1} (Stock: {round_start_stock:.3f})")
            
            # Execute one complete round (all agents act)
            for step in range(3):
                if not env.agents:
                    break
                    
                agent_id = env.agent_selection
                agent = base_env.homeostatic_agents[agent_id]
                
                # Determine action based on scenario
                if scenario["name"] == "Single Consumer":
                    if agent_id == "agent_0":
                        action = 3  # Consume
                        action_desc = "consume"
                    else:
                        action = np.random.choice([0, 1, 2])  # Movement only
                        action_desc = f"move_{action}"
                else:  # All Consumers
                    action = 3  # All consume
                    action_desc = "consume"
                
                # Execute action
                before_stock = base_env.resource_stock[0]
                env.step(action)
                after_stock = base_env.resource_stock[0]
                
                # Get consumption and reward
                if agent_id in env.agents:  # Agent still active
                    intake = np.sum(agent.last_intake)
                    reward = env.rewards[agent_id]
                else:
                    intake = 0
                    reward = 0
                
                round_consumption += intake
                round_rewards[agent_id] = reward
                
                print(f"  {agent_id}: {action_desc} | intake={intake:.3f} | reward={reward:.2f} | stock: {before_stock:.3f}→{after_stock:.3f}")
            
            round_end_stock = base_env.resource_stock[0]
            net_change = round_end_stock - round_start_stock
            regeneration = net_change + round_consumption
            
            # Store data
            resource_history.append(round_end_stock)
            consumption_history.append(round_consumption)
            reward_history.append(round_rewards)
            
            # Round summary
            print(f"📊 Round Summary:")
            print(f"   Resource: {round_start_stock:.3f} → {round_end_stock:.3f} (net: {net_change:+.3f})")
            print(f"   Total consumption: {round_consumption:.3f}")
            print(f"   Regeneration: {regeneration:.3f}")
            print(f"   Active agents: {len(env.agents)}")
            
            # Sustainability indicator
            if round_consumption > 0:
                sustainability_ratio = regeneration / round_consumption
                if sustainability_ratio > 1:
                    print(f"   🌱 Sustainable (regen/consumption = {sustainability_ratio:.2f})")
                elif sustainability_ratio > 0.8:
                    print(f"   ⚖️ Balanced (regen/consumption = {sustainability_ratio:.2f})")
                else:
                    print(f"   📉 Unsustainable (regen/consumption = {sustainability_ratio:.2f})")
        
        # Final analysis
        final_stock = resource_history[-1]
        total_consumption = sum(consumption_history)
        avg_consumption_per_round = total_consumption / len(consumption_history) if consumption_history else 0
        
        results[scenario["name"]] = {
            "final_stock": final_stock,
            "resource_history": resource_history,
            "total_consumption": total_consumption,
            "avg_consumption_per_round": avg_consumption_per_round,
            "rounds_survived": len(resource_history) - 1,
            "system_collapsed": len(env.agents) == 0,
            "consumption_history": consumption_history
        }
        
        print(f"\n🏁 Final Results - {scenario['name']}:")
        print(f"   Final stock: {final_stock:.3f} (started: {initial_stock})")
        print(f"   Total consumption: {total_consumption:.3f}")
        print(f"   Avg consumption/round: {avg_consumption_per_round:.3f}")
        print(f"   Rounds survived: {results[scenario['name']]['rounds_survived']}")
        
        if results[scenario["name"]]["system_collapsed"]:
            print(f"   💀 System collapsed (Tragedy of Commons)")
        else:
            print(f"   ✅ System survived")
    
    return results

def compare_scenarios(results):
    """Compare the results from both scenarios."""
    print(f"\n🔬 Scenario Comparison Analysis")
    print("=" * 70)
    
    single = results["Single Consumer"]
    all_consumers = results["All Consumers"]
    
    print(f"📊 Resource Sustainability:")
    print(f"   Single Consumer:  {single['final_stock']:.3f} final stock")
    print(f"   All Consumers:    {all_consumers['final_stock']:.3f} final stock")
    
    print(f"\n📊 Consumption Patterns:")
    print(f"   Single Consumer:  {single['avg_consumption_per_round']:.3f} avg/round")
    print(f"   All Consumers:    {all_consumers['avg_consumption_per_round']:.3f} avg/round")
    
    print(f"\n📊 System Longevity:")
    print(f"   Single Consumer:  {single['rounds_survived']} rounds")
    print(f"   All Consumers:    {all_consumers['rounds_survived']} rounds")
    
    print(f"\n📊 Outcome:")
    single_outcome = "Survived" if not single["system_collapsed"] else "Collapsed"
    all_outcome = "Survived" if not all_consumers["system_collapsed"] else "Collapsed"
    
    print(f"   Single Consumer:  {single_outcome}")
    print(f"   All Consumers:    {all_outcome}")
    
    # Resource trend analysis
    print(f"\n📈 Resource Trends:")
    
    # Single consumer trend
    if len(single["resource_history"]) > 1:
        single_trend = single["resource_history"][-1] - single["resource_history"][0]
        single_direction = "↗️ Growing" if single_trend > 0 else "↘️ Declining" if single_trend < 0 else "→ Stable"
        print(f"   Single Consumer:  {single_direction} ({single_trend:+.3f})")
    
    # All consumers trend
    if len(all_consumers["resource_history"]) > 1:
        all_trend = all_consumers["resource_history"][-1] - all_consumers["resource_history"][0]
        all_direction = "↗️ Growing" if all_trend > 0 else "↘️ Declining" if all_trend < 0 else "→ Stable"
        print(f"   All Consumers:    {all_direction} ({all_trend:+.3f})")
    
    # Tragedy of Commons analysis
    print(f"\n🏛️ Tragedy of Commons Analysis:")
    if single["system_collapsed"] and all_consumers["system_collapsed"]:
        print(f"   Both systems collapsed - regeneration insufficient even for single consumer")
    elif not single["system_collapsed"] and all_consumers["system_collapsed"]:
        print(f"   ✅ Classic Tragedy of Commons: sustainable with 1 consumer, collapses with all")
        print(f"   📈 Single consumer allows regeneration > consumption")  
        print(f"   📉 Multiple consumers create consumption > regeneration")
    elif single["system_collapsed"] and not all_consumers["system_collapsed"]:
        print(f"   🤔 Unexpected: single consumer collapsed but all survived (check social effects)")
    else:
        print(f"   🌱 Both systems sustainable - regeneration rate is very high")

def show_resource_evolution(results):
    """Show detailed resource evolution for both scenarios."""
    print(f"\n📈 Resource Stock Evolution:")
    print("=" * 70)
    
    single = results["Single Consumer"]["resource_history"]
    all_consumers = results["All Consumers"]["resource_history"]
    
    max_rounds = max(len(single), len(all_consumers))
    
    print(f"{'Round':<6} {'Single':<8} {'All':<8} {'Difference':<10}")
    print("-" * 40)
    
    for i in range(min(15, max_rounds)):  # Show first 15 rounds
        single_val = single[i] if i < len(single) else "---"
        all_val = all_consumers[i] if i < len(all_consumers) else "---"
        
        if isinstance(single_val, (int, float)) and isinstance(all_val, (int, float)):
            diff = single_val - all_val
            print(f"{i:<6} {single_val:<8.3f} {all_val:<8.3f} {diff:<+10.3f}")
        else:
            print(f"{i:<6} {single_val:<8} {all_val:<8} {'---':<10}")

def analyze_sustainability_threshold():
    """Analyze the theoretical sustainability threshold."""
    print(f"\n🔢 Theoretical Sustainability Analysis:")
    print("=" * 50)
    
    # From your config: regeneration = 0.3 (30%)
    # Typical consumption per agent ≈ 0.1 (from config: intake = 0.1)
    
    regeneration_rate = 0.3  # 30%
    consumption_per_agent = 0.1  # Typical consumption
    initial_stock = 10.0
    
    print(f"📊 Parameters:")
    print(f"   Initial stock: {initial_stock}")
    print(f"   Regeneration rate: {regeneration_rate} = {regeneration_rate*100}%")
    print(f"   Consumption per agent: ~{consumption_per_agent}")
    
    # Calculate maximum sustainable agents
    max_regeneration_per_round = regeneration_rate * initial_stock
    max_sustainable_agents = max_regeneration_per_round / consumption_per_agent
    
    print(f"\n🔢 Calculations:")
    print(f"   Max regeneration/round: {regeneration_rate} × {initial_stock} = {max_regeneration_per_round}")
    print(f"   Max sustainable agents: {max_regeneration_per_round} ÷ {consumption_per_agent} = {max_sustainable_agents:.1f}")
    
    print(f"\n🎯 Predictions:")
    for n_agents in [1, 2, 3, 4, 5]:
        total_consumption = n_agents * consumption_per_agent
        sustainability = "✅ Sustainable" if total_consumption <= max_regeneration_per_round else "❌ Unsustainable"
        ratio = max_regeneration_per_round / total_consumption
        print(f"   {n_agents} agents: consumption={total_consumption:.1f}, ratio={ratio:.2f} - {sustainability}")

if __name__ == "__main__":
    print("🚀 Starting Consumption Scenarios Test")
    print("This test compares sustainability with different consumption pressures")
    print()
    
    # Theoretical analysis first
    analyze_sustainability_threshold()
    
    # Run scenarios
    results = test_scenario_comparison()
    
    # Analysis
    compare_scenarios(results)
    show_resource_evolution(results)
    
    print(f"\n✅ Test completed!")
    print(f"💡 This demonstrates how consumption pressure affects system sustainability") 
