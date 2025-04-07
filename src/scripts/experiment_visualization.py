import numpy as np
import time
from src.gymnasium_env.envs import GridWorldEnv
from src.gymnasium_env.wrappers.internal_state import InternalStateWrapper
from src.gymnasium_env.wrappers.drive_reward import DriveRewardWrapper
from src.gymnasium_env.wrappers.visualization import VisualizationWrapper

def run_experiment(
    drive_type="base", 
    grid_size=7,
    n_resources=5,
    n_steps=100,
    seed=42
):
    """
    Run an experiment with the specified drive type.
    
    Args:
        drive_type: Type of drive to use (base, interoceptive, elliptic, weighted)
        grid_size: Size of the grid world
        n_resources: Number of resources to place in the environment
        n_steps: Number of steps to run the experiment
        seed: Random seed for reproducibility
    """
    print(f"Starting experiment with {drive_type} drive...")
    
    # Create the base environment
    env = GridWorldEnv(render_mode="human", size=grid_size)
    
    # Add internal states
    internal_state_env = InternalStateWrapper(env, internal_state_size=2, n_resources=n_resources)
    
    # Add drive-based rewards (parameters depend on drive type)
    if drive_type == "base":
        drive_env = DriveRewardWrapper(
            internal_state_env,
            drive_type="base",
            optimal_internal_states=[0.5, 0.5],
            m=1,
            n=2
        )
    elif drive_type == "interoceptive":
        drive_env = DriveRewardWrapper(
            internal_state_env,
            drive_type="interoceptive",
            optimal_internal_states=[0.5, 0.5],
            m=1,
            n=2,
            eta=1.5
        )
    elif drive_type == "elliptic":
        drive_env = DriveRewardWrapper(
            internal_state_env,
            drive_type="elliptic",
            optimal_internal_states=[0.5, 0.5],
            n_vector=[1, 2],
            m=1
        )
    elif drive_type == "weighted":
        drive_env = DriveRewardWrapper(
            internal_state_env,
            drive_type="weighted",
            w_red=1.5,
            w_blue=0.5
        )
    else:
        raise ValueError(f"Unknown drive type: {drive_type}")
    
    # Add visualization
    env = VisualizationWrapper(drive_env, drive_names=["Red Nutrient", "Blue Nutrient"])
    
    # Run experiment
    obs, info = env.env.env.reset(seed=seed)
    total_reward = 0
    
    for i in range(n_steps):
        # Simple policy: 50% random, 50% towards resource
        if np.random.random() < 0.5:
            action = env.action_space.sample()
        else:
            # Try to move towards a resource if possible
            agent_pos = tuple(obs['agent'])  # Acesso correto da posição do agente
            closest_resource = None
            min_dist = float('inf')
            
            # Acesso correto às localizações de recursos
            for res_pos in info['resource_locations'].keys():
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
        
        # Execute step
        obs, reward, terminated, truncated, info = env.env.step(action)
        total_reward += reward
        
        # Print step information
        print(f"Step {i+1}, Action: {action}, Reward: {reward:.4f}, Total: {total_reward:.4f}")
        print(f"Internal states: {obs['internal_states']}")
        
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
    """Run experiments with different drive types."""
    # Lista de tipos de drive para testar
    drive_types = ["base", "interoceptive", "elliptic", "weighted"]
    
    for drive_type in drive_types:
        print(f"\n{'='*60}")
        print(f"Testing {drive_type} drive")
        print(f"{'='*60}")
        
        # Execute o experimento com este tipo de drive
        try:
            run_experiment(
                drive_type=drive_type,
                grid_size=7,
                n_resources=5,
                n_steps=30,  # Reduzido para demonstração
                seed=42
            )
        except Exception as e:
            print(f"Error with {drive_type} drive: {e}")
        
        # Aguarde um pouco entre experimentos
        time.sleep(1)

if __name__ == "__main__":
    # Execute apenas um tipo de drive para testar
    run_experiment(drive_type="base", n_steps=100)
