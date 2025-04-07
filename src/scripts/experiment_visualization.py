import numpy as np
import time
from src.gymnasium_env.envs import GridWorldEnv
from src.gymnasium_env.wrappers.internal_state import InternalStateWrapper
from src.gymnasium_env.wrappers.drive_reward import DriveRewardWrapper
from src.gymnasium_env.wrappers.visualization import VisualizationWrapper

def run_experiment():
    # Create the base environment
    env = GridWorldEnv(render_mode="human", size=7)
    
    # Add internal states
    env = InternalStateWrapper(env, internal_state_size=2, n_resources=5)
    
    # Add drive-based rewards
    env = DriveRewardWrapper(
        env,
        drive_type="elliptic",
        optimal_internal_states=[0.5, 0.5],
        n_vector=[1, 2],
        m=1
    )
    
    # Add visualization
    env = VisualizationWrapper(env, drive_names=["Red Nutrient", "Blue Nutrient"])
    
    # Run a simple experiment
    obs, info = env.reset(seed=42)
    total_reward = 0
    
    for i in range(100):
        # Simple policy: 50% random, 50% towards resource
        if np.random.random() < 0.5:
            action = env.action_space.sample()
        else:
            # Try to move towards a resource if possible
            agent_pos = tuple(env.env.env.env._agent_location)
            closest_resource = None
            min_dist = float('inf')
            
            for res_pos in env.env.env.resource_locations.keys():
                dist = abs(res_pos[0] - agent_pos[0]) + abs(res_pos[1] - agent_pos[1])
                if dist < min_dist:
                    min_dist = dist
                    closest_resource = res_pos
            
            if closest_resource:
                # Simple navigation towards resource
                if closest_resource[0] > agent_pos[0]:
                    action = 0  # right
                elif closest_resource[0] < agent_pos[0]:
                    action = 2  # left
                elif closest_resource[1] > agent_pos[1]:
                    action = 1  # up
                else:
                    action = 3  # down
            else:
                action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {i+1}, Action: {action}, Reward: {reward:.4f}, Total: {total_reward:.4f}")
        
        if terminated:
            print("Environment terminated!")
            time.sleep(1)
            obs, info = env.reset()
            total_reward = 0
        
        time.sleep(0.2)  # Slow down the visualization
    
    env.close()

if __name__ == "__main__":
    run_experiment()
