import gymnasium as gym
from ...drives.base_drive import BaseDrive
from ...drives.interoceptive_drive import InteroceptiveModulationDrive
from ...drives.elliptic_drive import EllipticDrive
from ...drives.weighted_drive import WeightedDrive

class DriveRewardWrapper(gym.Wrapper):
    """
    A wrapper that calculates rewards based on different homeostatic drive functions.
    
    This wrapper supports multiple drive types:
    - "base": Standard drive calculation (BaseDrive)
    - "interoceptive": Drive with interoceptive modulation
    - "elliptic": Drive with different exponents per dimension
    - "weighted": Drive with weighted nutrient-specific values
    """
    
    def __init__(self, env, drive_type="base", **drive_params):
        """
        Initialize the drive reward wrapper.
        
        Args:
            env: The environment to wrap
            drive_type: The type of drive to use (base, interoceptive, elliptic, weighted)
            **drive_params: Parameters for the specific drive type:
                - base: optimal_internal_states, m, n
                - interoceptive: optimal_internal_states, m, n, eta
                - elliptic: optimal_internal_states, n_vector, m
                - weighted: w_red, w_blue
        """
        super().__init__(env)
        self.drive_type = drive_type
        
        # Initialize the appropriate drive based on the specified type
        if drive_type == "base":
            self.drive = BaseDrive(**drive_params)
        elif drive_type == "interoceptive":
            self.drive = InteroceptiveModulationDrive(**drive_params)
        elif drive_type == "elliptic":
            self.drive = EllipticDrive(**drive_params)
        elif drive_type == "weighted":
            self.drive = WeightedDrive(**drive_params)
        else:
            raise ValueError(f"Unknown drive type: {drive_type}")
    
    def step(self, action):
        """
        Execute the environment step and compute the drive-based reward.
        
        Args:
            action: The action to take in the environment
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        obs, _, terminated, truncated, info = self.env.step(action)
        
        # Extract internal states and outcome from observation
        current_internal_states = obs['internal_states']
        outcome = obs['outcome']
        
        # Compute reward based on the drive reduction
        reward = self.drive.compute_reward(
            current_internal_states=current_internal_states, 
            outcome=outcome
        )
        
        # Add drive-specific information to the info dictionary
        info['drive_value'] = self.drive.compute_drive(current_internal_states).item()
        info['drive_type'] = self.drive_type
        info['reward'] = reward.item()
        
        return obs, reward.item(), terminated, truncated, info
