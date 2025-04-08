import time
import numpy as np
from src.gymnasium_env.envs import GridWorldEnv2Resources

def test_render_environment():
    """
    Tests if the GridWorldEnv2Resources environment can be properly initialized and rendered.
    """
    try:
        print("Starting rendering test...")
        
        # Initialize the environment with visualization
        env = GridWorldEnv2Resources(
            config_path="config/config.yaml",
            drive_type="base_drive",
            render_mode="human"
        )
        
        print("Environment successfully initialized!")
        print(f"Internal state size: {env._internal_state_size}")
        
        # Reset the environment
        obs, info = env.reset(seed=42)
        print("\nReset successfully completed!")
        print(f"Observation after reset: {obs}")
        print(f"Info after reset: {info}")
        
        # Give time to visualize
        print("\nWill keep the window open for 3 seconds...")
        time.sleep(3)
        
        # Execute some actions to test functionality
        print("\nExecuting some random actions...")
        total_reward = 0
        
        for i in range(10):
            # Choose a random action
            action = env.action_space.sample()
            
            # Execute the action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Map actions to more descriptive names
            action_names = {0: "Consume R0", 1: "Consume R1", 2: "Do Nothing"}
            
            # Print information about the step
            print(f"Step {i+1}:")
            print(f"  Action: {action_names[action]}")
            print(f"  Internal states: {obs['internal_states']}")
            print(f"  Available resources: {obs['resources_available']}")
            print(f"  Reward: {reward:.4f}")
            print(f"  Drive: {info['drive']:.4f}")
            
            # Give time to visualize
            time.sleep(1)
            
            # Check if the episode has ended
            if terminated:
                print("\nEpisode ended! Optimal states achieved.")
                obs, info = env.reset()
                print("Environment reset.")
                total_reward = 0
        
        print(f"\nTest completed! Total reward: {total_reward:.4f}")
        
        # Keep the window open for a few more seconds
        print("Will keep the window open for 3 more seconds before closing...")
        time.sleep(3)
        
        # Close the environment
        env.close()
        print("Environment closed.")
        
    except Exception as e:
        import traceback
        print(f"\nTEST ERROR: {e}")
        print(traceback.format_exc())
        
if __name__ == "__main__":
    test_render_environment()
