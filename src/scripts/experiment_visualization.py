import numpy as np
import time
from src.gymnasium_env.envs import GridWorldEnv
# from src.gymnasium_env.wrappers.internal_state import InternalStateWrapper
# from src.gymnasium_env.wrappers.drive_reward import DriveRewardWrapper
# from src.gymnasium_env.wrappers.visualization import VisualizationWrapper

def run_experiment(
    grid_size=7,
    n_steps=100,
    seed=42
):
    """    
    Args:
        grid_size: Size of the grid world
        n_resources: Number of resources to place in the environment
        n_steps: Number of steps to run the experiment
        seed: Random seed for reproducibility
    """
    
    # Create the base environment
    env = GridWorldEnv(render_mode="human", size=grid_size)
    
    # Run experiment
    obs, info = env.reset(seed=seed)
    total_reward = 0
    
    for i in range(n_steps):
        # Simple policy: 50% random, 50% towards resource
        if np.random.random() < 0.5:
            action = env.action_space.sample()
        else:
            # Try to move towards a resource if possible
            agent_pos = tuple(obs['agent'])  # Acesso correto da posição do agente
            target_pos = tuple(obs['target'])
            closest_resource = None
            min_dist = float('inf')
            
            # Acesso correto às localizações de recursos
            dist = abs(target_pos[0] - agent_pos[0]) + abs(target_pos[1] - agent_pos[1])
            if dist < min_dist:
                min_dist = dist
                closest_resource = target_pos
            
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
        
        # Execute step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Print step information
        print(f"Step {i+1}, Action: {action}, Reward: {reward:.4f}, Total: {total_reward:.4f}")
        
        # Check if episode is done
        if terminated:
            print("Environment terminated!")
            time.sleep(1)
            obs, info = env.reset()
            total_reward = 0
        
        # Slow down visualization
        time.sleep(0.2)
    
    env.close()
    print(f"Experiment completed. Total reward: {total_reward:.4f}")

def main():
        run_experiment(
            grid_size=7,
            n_resources=5,
            n_steps=30,  # Reduzido para demonstração
            seed=42
        )
        # Aguarde um pouco entre experimentos
        time.sleep(1)

if __name__ == "__main__":
    # Execute apenas um tipo de drive para testar
    run_experiment(drive_type="base", n_steps=100)
