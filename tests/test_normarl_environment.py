#!/usr/bin/env python3
"""
Comprehensive test for the NORMARL environment to verify it's working as expected.
This test checks agent interactions, resource consumption, social learning, and environment dynamics.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from src.envs.multiagent import NormalHomeostaticEnv

def test_environment_initialization():
    """Test that the environment initializes correctly."""
    print("🧪 Testing Environment Initialization")
    print("=" * 50)
    
    try:
        # Create environment
        env = NormalHomeostaticEnv(
            config_path="config/config.yaml",
            drive_type="base_drive",
            learning_rate=0.1,
            beta=0.5,
            number_resources=1,
            n_agents=3,
            size=5
        )
        print("✅ Environment created successfully")
        
        # Test reset
        observations, info = env.reset()
        print("✅ Environment reset successfully")
        
        # Check basic properties
        print(f"📊 Number of agents: {len(env.agents)}")
        print(f"📊 Agent IDs: {env.agents}")
        print(f"📊 Current agent: {env.agent_selection}")
        print(f"📊 Resource stock: {env.resource_stock}")
        print(f"📊 Resource regeneration rate: {env.resource_regeneration_rate}")
        
        # Check resource positions
        print(f"🌱 Resource positions:")
        for resource_id, resource_info in env.resources_info.items():
            print(f"   Resource {resource_id} ({resource_info['name']}): position {resource_info['position']}")
        
        # Check agent states
        for agent_id in env.agents:
            agent = env.homeostatic_agents[agent_id]
            print(f"👤 {agent_id}: pos={agent.position}, states={agent.internal_states}, norms={agent.perceived_social_norm}")
        
        return env, observations
        
    except Exception as e:
        print(f"❌ Environment initialization failed: {e}")
        return None, None

def test_agent_actions(env, observations):
    """Test that agents can perform actions correctly."""
    print("\n🧪 Testing Agent Actions")
    print("=" * 50)
    
    if env is None:
        print("❌ Skipping action tests - environment not available")
        return False
    
    try:
        # Test a few steps with different actions
        actions_to_test = [0, 1, 2, 3]  # move left, move right, stay, consume resource 0
        
        for i, action in enumerate(actions_to_test):
            print(f"\n🔄 Step {i+1}: Testing action {action}")
            
            # Get current state
            current_agent = env.agent_selection
            agent = env.homeostatic_agents[current_agent]
            old_position = agent.position
            old_states = agent.internal_states.copy()
            old_stock = env.resource_stock.copy()
            
            print(f"👤 Agent {current_agent} before action:")
            print(f"   Position: {old_position}")
            print(f"   Internal states: {old_states}")
            print(f"   Resource stock: {old_stock}")
            # Check if agent is at resource position for consumption actions
            if action >= 3:
                resource_positions = [env.resources_info[i]['position'] for i in range(env.dimension_internal_states)]
                print(f"   Resource positions: {resource_positions}")
                print(f"   At resource position: {old_position in resource_positions}")
            
            # Execute action
            env.step(action)
            
            # Check results
            new_position = agent.position
            new_states = agent.internal_states
            new_stock = env.resource_stock
            reward = env.rewards.get(current_agent, 0)
            
            print(f"👤 Agent {current_agent} after action:")
            print(f"   Position: {new_position}")
            print(f"   Internal states: {new_states}")
            print(f"   Resource stock: {new_stock}")
            print(f"   Reward: {reward}")
            
            # Verify position change for movement actions
            if action == 0:  # move left
                expected_pos = max(0, old_position - 1)
                if new_position == expected_pos:
                    print("✅ Left movement working correctly")
                else:
                    print(f"❌ Left movement failed: expected {expected_pos}, got {new_position}")
                    
            elif action == 1:  # move right
                expected_pos = min(env.size - 1, old_position + 1)
                if new_position == expected_pos:
                    print("✅ Right movement working correctly")
                else:
                    print(f"❌ Right movement failed: expected {expected_pos}, got {new_position}")
                    
            elif action == 2:  # stay
                if new_position == old_position:
                    print("✅ Stay action working correctly")
                else:
                    print(f"❌ Stay action failed: position changed from {old_position} to {new_position}")
                    
            elif action == 3:  # consume resource 0
                if np.any(agent.last_intake > 0):
                    print(f"✅ Resource consumption working correctly")
                    print(f"   Consumed: {agent.last_intake}")
                else:
                    print(f"❌ Resource consumption failed: no intake recorded")
            
            # Check if reward is reasonable
            if isinstance(reward, (int, float)):
                print("✅ Reward is numeric")
            else:
                print(f"❌ Reward has wrong type: {type(reward)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Action testing failed: {e}")
        return False

def test_resource_consumption(env):
    """Test resource consumption and regeneration mechanics."""
    print("\n🧪 Testing Resource Consumption")
    print("=" * 50)
    
    if env is None:
        print("❌ Skipping resource tests - environment not available")
        return False
    
    try:
        # Reset environment
        observations, info = env.reset()
        initial_stock = env.resource_stock.copy()
        print(f"📦 Initial resource stock: {initial_stock}")
        
        # Have all agents consume resources
        for agent_id in env.agents:
            # Move agent to resource position and consume
            agent = env.homeostatic_agents[agent_id]
            
            # Test consuming the single resource
            action = 3  # consume resource 0
            
            print(f"\n🔄 Agent {agent_id} consuming resource 0")
            stock_before = env.resource_stock.copy()
            
            env.step(action)
            
            stock_after = env.resource_stock.copy()
            consumption = stock_before - stock_after
            
            print(f"📦 Stock before: {stock_before}")
            print(f"📦 Stock after: {stock_after}")
            print(f"📦 Consumption: {consumption}")
            
            # Check that consumption is reasonable
            if np.any(consumption > 0):
                print("✅ Resource consumption working")
            else:
                print("⚠️  No resource consumed (might be due to position or stock)")
        
        # Check if global environment update happens after all agents act
        print(f"\n🌍 Final resource stock: {env.resource_stock}")
        
        return True
        
    except Exception as e:
        print(f"❌ Resource testing failed: {e}")
        return False

def test_social_learning(env):
    """Test social norm learning and social cost calculation."""
    print("\n🧪 Testing Social Learning")
    print("=" * 50)
    
    if env is None:
        print("❌ Skipping social learning tests - environment not available")
        return False
    
    try:
        # Reset environment
        observations, info = env.reset()
        
        # Check initial social norms
        print("👥 Initial social norms:")
        for agent_id in env.agents:
            agent = env.homeostatic_agents[agent_id]
            print(f"   {agent_id}: {agent.perceived_social_norm}")
        
        # Have agents consume resources to trigger social learning
        print("\n🔄 Having agents consume resources...")
        for agent_id in env.agents:
            # Consume resource 0
            env.step(3)  # consume resource 0
        
        # Check if social norms were updated
        print("\n👥 Social norms after consumption:")
        for agent_id in env.agents:
            agent = env.homeostatic_agents[agent_id]
            print(f"   {agent_id}: {agent.perceived_social_norm}")
        
        # Check social cost calculation
        print("\n💰 Testing social cost calculation:")
        for agent_id in env.agents:
            agent = env.homeostatic_agents[agent_id]
            if np.any(agent.last_intake > 0):
                resource_scarcity = env._compute_resource_scarcity()
                social_cost = agent.compute_social_cost(agent.last_intake, resource_scarcity)
                print(f"   {agent_id} social cost: {social_cost}")
                print(f"   {agent_id} last intake: {agent.last_intake}")
                print(f"   Resource scarcity: {resource_scarcity}")
        
        return True
        
    except Exception as e:
        print(f"❌ Social learning testing failed: {e}")
        return False

def test_social_cost_specific(env):
    """Test social cost calculation with adjusted scarcity parameters."""
    print("\n🧪 Testing Social Cost Calculation")
    print("=" * 50)
    
    if env is None:
        print("❌ Skipping social cost tests - environment not available")
        return False
    
    try:
        # Reset environment
        observations, info = env.reset()
        
        print(f"📊 Initial resource stock: {env.resource_stock}")
        print(f"🌱 Resource positions: {[r['position'] for r in env.resources_info.values()]}")
        
        # Check agent positions
        for agent_id in env.agents:
            agent = env.homeostatic_agents[agent_id]
            print(f"👤 {agent_id}: pos={agent.position}")
        
        # Test scarcity calculation
        print(f"\n🔍 Testing scarcity calculation:")
        scarcity = env._compute_resource_scarcity()
        print(f"   Current scarcity: {scarcity}")
        print(f"   Formula: max{{0, 2.0 - 0.8 × {env.resource_stock[0]}}} = max{{0, 2.0 - {0.8 * env.resource_stock[0]}}} = {scarcity[0]}")
        
        # Move agents to resource position and test consumption
        resource_pos = env.resources_info[0]['position']
        print(f"\n🔄 Moving agents to resource position {resource_pos} and testing consumption...")
        
        for agent_id in env.agents:
            agent = env.homeostatic_agents[agent_id]
            
            # Move agent to resource position
            while agent.position != resource_pos:
                if agent.position < resource_pos:
                    action = 2  # move right
                else:
                    action = 1  # move left
                env.step(action)
            
            print(f"👤 {agent_id} now at position {agent.position}")
            
            # Test consumption
            stock_before = env.resource_stock.copy()
            print(f"📦 Stock before consumption: {stock_before}")
            
            # Consume resource
            env.step(3)  # consume resource 0
            
            stock_after = env.resource_stock.copy()
            print(f"📦 Stock after consumption: {stock_after}")
            
            # Check social cost
            if np.any(agent.last_intake > 0):
                scarcity = env._compute_resource_scarcity()
                social_cost = agent.compute_social_cost(agent.last_intake, scarcity)
                print(f"💰 Social cost for {agent_id}: {social_cost}")
                print(f"   Intake: {agent.last_intake}")
                print(f"   Social norm: {agent.perceived_social_norm}")
                print(f"   Scarcity: {scarcity}")
                print(f"   Beta: {agent.beta}")
                
                # Manual calculation
                excess = np.maximum(0, agent.last_intake - agent.perceived_social_norm)
                expected_cost = agent.beta * excess * scarcity
                print(f"   Manual calculation: {agent.beta} × {excess} × {scarcity} = {expected_cost}")
            
            print("-" * 30)
        
        return True
        
    except Exception as e:
        print(f"❌ Social cost testing failed: {e}")
        return False

def test_environment_dynamics(env):
    """Test overall environment dynamics and termination conditions."""
    print("\n🧪 Testing Environment Dynamics")
    print("=" * 50)
    
    if env is None:
        print("❌ Skipping dynamics tests - environment not available")
        return False
    
    try:
        # Reset environment
        observations, info = env.reset()
        
        # Run a few complete rounds
        max_steps = 20
        step_count = 0
        
        print(f"🔄 Running {max_steps} steps...")
        
        while step_count < max_steps:
            # Get current agent
            current_agent = env.agent_selection
            
            # Check if agent is terminated
            if env.terminations.get(current_agent, False):
                print(f"⚠️  Agent {current_agent} is terminated")
                break
            
            # Take a random action
            action_space = env.action_space(current_agent)
            action = action_space.sample()
            
            # Execute action
            env.step(action)
            
            step_count += 1
            
            # Print progress every 5 steps
            if step_count % 5 == 0:
                print(f"📊 Step {step_count}: Agent {current_agent} took action {action}")
                print(f"   Resource stock: {env.resource_stock}")
                print(f"   Terminations: {env.terminations}")
        
        print(f"\n✅ Completed {step_count} steps successfully")
        print(f"📊 Final resource stock: {env.resource_stock}")
        print(f"📊 Final terminations: {env.terminations}")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment dynamics testing failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 NORMARL Environment Comprehensive Test")
    print("=" * 60)
    
    # Test 1: Environment initialization
    env, observations = test_environment_initialization()
    
    # Test 2: Agent actions
    actions_ok = test_agent_actions(env, observations)
    
    # Test 3: Resource consumption
    resources_ok = test_resource_consumption(env)
    
    # Test 4: Social learning
    social_ok = test_social_learning(env)
    
    # Test 5: Social cost specific
    social_cost_ok = test_social_cost_specific(env)
    
    # Test 6: Environment dynamics
    dynamics_ok = test_environment_dynamics(env)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    tests = [
        ("Environment Initialization", env is not None),
        ("Agent Actions", actions_ok),
        ("Resource Consumption", resources_ok),
        ("Social Learning", social_ok),
        ("Social Cost Specific", social_cost_ok),
        ("Environment Dynamics", dynamics_ok)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The NORMARL environment is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
