"""Debug reward calculation in detail."""

from src.envs.multiagent import create_env
import numpy as np

def debug_reward_calculation():
    print("🔍 Deep Debug: Reward Calculation")
    print("=" * 60)
    
    env = create_env(
        config_path="config/config.yaml",
        drive_type="base_drive",
        learning_rate=0.1,
        beta=0.1,  # Low social cost to focus on homeostatic reward
        number_resources=1,
        n_agents=1,  # Single agent for easier debugging
        size=1,      # All at resource position
        log_level="ERROR"  # Silent
    )
    
    env.reset(seed=42)
    
    agent_id = env.agents[0]
    agent = env.homeostatic_agents[agent_id]
    
    print(f"🔬 Initial state:")
    print(f"   Agent ID: {agent_id}")
    print(f"   Internal states: {agent.internal_states}")
    print(f"   Current drive: {agent.get_current_drive()}")
    print(f"   Resource stock: {env.resource_stock}")
    
    # Test movement action (should have some reward due to natural decay)
    print(f"\n🚶 Testing Movement Action (should have decay reward):")
    
    # Get states before action
    states_before_step = agent.internal_states.copy()
    drive_before_step = agent.get_current_drive()
    
    print(f"   States before action: {states_before_step}")
    print(f"   Drive before action: {drive_before_step}")
    
    # Execute movement action (action 0 = stay)
    action = 0
    env.step(action)
    
    # Get final results
    states_after_step = agent.internal_states.copy()
    drive_after_step = agent.get_current_drive()
    reward = env.rewards.get(agent_id, 0)
    
    print(f"   States after action: {states_after_step}")
    print(f"   Drive after action: {drive_after_step}")
    print(f"   Final reward: {reward}")
    
    # Manual calculation check
    state_diff = states_after_step - states_before_step
    drive_diff = drive_after_step - drive_before_step
    expected_reward = drive_before_step - drive_after_step  # old - new
    
    print(f"\n🧮 Manual Verification:")
    print(f"   State change: {state_diff}")
    print(f"   Drive change: {drive_diff}")
    print(f"   Expected reward (old-new): {expected_reward}")
    print(f"   Actual reward: {reward}")
    print(f"   Match? {np.isclose(expected_reward, reward)}")
    
    # Test the drive calculation directly
    print(f"\n🔧 Direct Drive Testing:")
    drive_system = agent.drive
    
    # Test drive calculation manually
    drive_from_before = drive_system.compute_drive(states_before_step)
    drive_from_after = drive_system.compute_drive(states_after_step)
    manual_reward = drive_system.compute_reward(drive_from_before, drive_from_after)
    
    print(f"   Drive calculated from before states: {drive_from_before}")
    print(f"   Drive calculated from after states: {drive_from_after}")
    print(f"   Manual reward calculation: {manual_reward}")
    
    # Check loss rates and decay
    print(f"\n⏳ Decay Analysis:")
    loss_rates = drive_system.get_array_loss_rates()
    print(f"   Loss rates: {loss_rates}")
    print(f"   Expected state change: -{loss_rates}")
    print(f"   Actual state change: {state_diff}")
    
    # Check if loss rates are too small
    if np.all(np.abs(loss_rates) < 1e-6):
        print("   ⚠️  Loss rates are extremely small!")
    
    if np.all(np.abs(state_diff) < 1e-6):
        print("   ⚠️  State changes are extremely small!")
    
    if np.all(np.abs(drive_diff) < 1e-6):
        print("   ⚠️  Drive changes are extremely small!")

def test_with_consumption():
    print("\n🍽️ Testing Consumption Action:")
    print("=" * 40)
    
    env = create_env(
        config_path="config/config.yaml",
        drive_type="base_drive",
        learning_rate=0.1,
        beta=0.1,
        number_resources=1,
        n_agents=1,
        size=1,
        log_level="ERROR"
    )
    
    env.reset(seed=42)
    
    agent_id = env.agents[0]
    agent = env.homeostatic_agents[agent_id]
    
    # Force agent to have some internal state deviation
    agent.internal_states = np.array([-0.5])  # Far from optimal (likely 0)
    
    print(f"   Forced internal states: {agent.internal_states}")
    print(f"   Current drive: {agent.get_current_drive()}")
    print(f"   Resource stock: {env.resource_stock}")
    
    # Test consumption
    states_before = agent.internal_states.copy()
    drive_before = agent.get_current_drive()
    
    action = 3  # Consume
    env.step(action)
    
    states_after = agent.internal_states.copy()
    drive_after = agent.get_current_drive()
    reward = env.rewards.get(agent_id, 0)
    intake = agent.last_intake[0]
    
    print(f"   States change: {states_before} → {states_after}")
    print(f"   Drive change: {drive_before} → {drive_after}")
    print(f"   Intake: {intake}")
    print(f"   Reward: {reward}")
    
    expected_reward = drive_before - drive_after
    print(f"   Expected reward: {expected_reward}")

def check_config_file():
    print("\n📄 Checking Configuration File:")
    print("=" * 40)
    
    from src.utils.get_params import ParameterHandler
    
    param_handler = ParameterHandler("config/config.yaml")
    config = param_handler.config
    
    print("Drive parameters:")
    print(f"   {config.get('drive_params', {})}")
    
    print("\nOptimal internal states:")
    optimal_states = config.get('global_params', {}).get('optimal_internal_state', {})
    print(f"   {optimal_states}")
    
    # Check loss rates specifically
    for state_name, state_config in optimal_states.items():
        loss_rate = state_config.get('loss', 'Not found')
        intake_rate = state_config.get('intake', 'Not found')
        print(f"   {state_name}: loss={loss_rate}, intake={intake_rate}")

if __name__ == "__main__":
    debug_reward_calculation()
    test_with_consumption()
    check_config_file()
    
    print("\n🎯 Summary:")
    print("- If loss rates are too small, decay rewards will be ~0")
    print("- If intake rates are large, consumption rewards should be significant")
    print("- Check config.yaml for actual parameter values") 
