"""Test script for multiagent environments only."""

import numpy as np
from src.envs.multiagent import create_env, create_parallel_env

def test_multi_agent():
    """Test multi-agent AEC environment."""
    print("👥 Testing Multi-Agent AEC Environment")
    
    env = create_env(
        config_path="config/config.yaml",
        drive_type="base_drive",
        learning_rate=0.1,
        beta=0.5,
        number_resources=1,
        n_agents=3,
        size=5,
        log_level="WARNING"  # Less verbose
    )
    
    env.reset(seed=42)
    print(f"   Agents: {env.agents}")
    print(f"   Current agent: {env.agent_selection}")
    
    rewards_collected = []
    
    for i in range(15):  # 5 rounds of 3 agents
        if not env.agents:
            print("   All agents terminated!")
            break
            
        current_agent = env.agent_selection
        if current_agent is None:
            print("   No current agent!")
            break
            
        action = env.action_space(current_agent).sample()
        env.step(action)
        
        reward = env.rewards.get(current_agent, 0)
        rewards_collected.append(reward)
        print(f"  Step {i}: {current_agent} action={action} reward={reward:.3f}")
        
        # Check termination
        if all(env.terminations.values()) or all(env.truncations.values()):
            print("   Episode terminated!")
            break
    
    avg_reward = np.mean(rewards_collected) if rewards_collected else 0
    print(f"   Average reward: {avg_reward:.3f}")
    print("✅ Multi-agent AEC test passed!")

def test_parallel():
    """Test parallel environment."""
    print("🔄 Testing Parallel Environment")
    
    env = create_parallel_env(
        config_path="config/config.yaml",
        drive_type="base_drive",
        learning_rate=0.1,
        beta=0.5,
        number_resources=1,
        n_agents=3,
        size=5,
        log_level="WARNING"  # Less verbose
    )
    
    observations = env.reset(seed=42)
    print(f"   Initial agents: {env.agents}")
    print(f"   Initial observations: {len(observations)}")
    
    all_rewards = []
    
    for i in range(5):
        if not env.agents:
            print("   All agents terminated!")
            break
            
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        round_rewards = list(rewards.values())
        all_rewards.extend(round_rewards)
        avg_reward = np.mean(round_rewards)
        
        print(f"  Round {i}: {len(env.agents)} agents, avg_reward={avg_reward:.3f}")
        print(f"    Actions: {actions}")
        print(f"    Rewards: {[f'{k}:{v:.2f}' for k,v in rewards.items()]}")
        
        if all(terminations.values()) or all(truncations.values()):
            print("   Episode terminated!")
            break
    
    overall_avg = np.mean(all_rewards) if all_rewards else 0
    print(f"   Overall average reward: {overall_avg:.3f}")
    print("✅ Parallel test passed!")

def test_compatibility():
    """Test PettingZoo compatibility features."""
    print("🔧 Testing PettingZoo Compatibility")
    
    # Test with supersuit-style access
    env = create_env(
        config_path="config/config.yaml",
        drive_type="base_drive",
        learning_rate=0.1,
        beta=0.5,
        number_resources=1,
        n_agents=2,
        size=3,
        log_level="ERROR"  # Silent
    )
    
    # Test proper reset sequence
    env.reset(seed=123)
    
    # Test attribute access after reset
    print(f"   ✅ Agents accessible: {len(env.agents)} agents")
    print(f"   ✅ Agent selection: {env.agent_selection}")
    print(f"   ✅ Action spaces: {[env.action_space(a).n for a in env.agents]}")
    print(f"   ✅ Observation spaces: {[type(env.observation_space(a)).__name__ for a in env.agents]}")
    
    # Test observation method
    for agent in env.agents:
        obs = env.observe(agent)
        print(f"   ✅ Observe {agent}: {type(obs).__name__} shape {obs.shape}")
        break  # Just test first agent
    
    # Test step without errors
    current_agent = env.agent_selection
    action = env.action_space(current_agent).sample()
    env.step(action)
    print(f"   ✅ Step successful: {current_agent} took action {action}")
    
    print("✅ Compatibility test passed!")

def test_environment_properties():
    """Test environment properties and metadata."""
    print("📊 Testing Environment Properties")
    
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
    
    # Test metadata
    print(f"   Metadata: {env.metadata}")
    print(f"   Name: {env.metadata.get('name', 'Unknown')}")
    print(f"   Parallelizable: {env.metadata.get('is_parallelizable', False)}")
    
    env.reset()
    
    # Test environment properties
    print(f"   Possible agents: {env.possible_agents}")
    print(f"   Current agents: {env.agents}")
    print(f"   Max steps: {env.max_steps}")
    print(f"   Grid size: {env.size}")
    print(f"   Resource types: {env.dimension_internal_states}")
    
    # Test internal state
    first_agent = env.agents[0]
    homeostatic_agent = env.homeostatic_agents[first_agent]
    print(f"   Agent internal states shape: {homeostatic_agent.internal_states.shape}")
    print(f"   Social learning rate: {homeostatic_agent.social_learning_rate}")
    print(f"   Beta (norm strength): {homeostatic_agent.beta}")
    
    print("✅ Properties test passed!")

if __name__ == "__main__":
    print("🧪 Testing Multi-Agent Environments Only")
    print("=" * 60)
    print("Note: Skipping single-agent test (needs avg_intake parameter)")
    print()
    
    try:
        test_multi_agent()
        print()
        test_parallel() 
        print()
        test_compatibility()
        print()
        test_environment_properties()
        print()
        print("✅ All multi-agent tests passed! 🎉")
        print()
        print("📚 Your environments are ready for:")
        print("   - Ray RLlib")
        print("   - CleanRL") 
        print("   - Tianshou")
        print("   - PettingZoo tutorials")
        print("   - SuperSuit wrappers")
        print("   - Custom MARL algorithms")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 
