# src/drives/base_drive.py

import torch

class BaseDrive():
    """
    Base class for implementing homeostatic drives in reinforcement learning.
    Provides default drive computation, but subclasses may override it.
    """

    def __init__(self, optimal_internal_states_config, m=1, n=1):
        """
        Initialize the base drive class.

        :param optimal_internal_states_config: Target internal state vector (H*), as dict.
        :param m: Root parameter used in the drive computation.
        :param n: Exponent parameter used in the drive computation.
        """
        self._optimal_internal_states = optimal_internal_states_config
        self.m = m
        self.n = n
        self._current_drive = None

    def _to_tensor(self, x):
        """
        Utility method to ensure input is a detached torch.Tensor.
        """
        if isinstance(x, torch.Tensor):
            return x.clone().detach()
        return torch.tensor(x, dtype=torch.float32)
    
    def get_state_value(self, state_name):
        return self._optimal_internal_states[state_name]
    
    def get_tensor_optimal_states_values(self):
        if isinstance(self._optimal_internal_states, dict):
            values = []
            for state_name, state_value in self._optimal_internal_states.items():
                values.append(state_value)
            return torch.tensor(values, dtype=torch.float32)
        elif isinstance(self._optimal_internal_states, list) or isinstance(self._optimal_internal_states, torch.Tensor):
            return self._to_tensor(self._optimal_internal_states)
        else:
            return self._to_tensor(self._optimal_internal_states)

    def has_state(self, state_name):
        return state_name in self._optimal_internal_states

    def get_internal_state_size(self):
        return len(self._optimal_internal_states.keys())

    def compute_drive(self, current_internal_states):
        """
        Compute the drive as the root of the sum of powered deviations from the optimal state.

        This is a default implementation based on:
        D(H_t) = (sum_i |H*_i - H_{i,t}|^n)^{1/m}

        Subclasses may override this to apply modulation, weighting, or other mechanisms.

        :param current_internal_states: Current internal states (H_t).
        :return: Scalar torch.Tensor representing drive value.
        """
        current_internal_states = self._to_tensor(current_internal_states)
        optimal_states_tensor = self.get_tensor_optimal_states_values()

        diff = optimal_states_tensor - current_internal_states
        drive_sum = torch.sum(torch.abs(diff) ** self.n)
        drive_value = drive_sum ** (1 / self.m)
        return drive_value
    
    def update_drive(self, drive_value):
        self._current_drive = drive_value

    def compute_reward(self, new_drive):
        """
        Compute the reward as the reduction in drive from applying the outcome.

        :param new_drive:  New drive (H_{t+1}).
        :return: Scalar torch.Tensor representing reward.
        """

        reward = self._current_drive - new_drive
        return reward
    
    def has_reached_optimal(self, current_internal_states):
        """
        Checks if the current internal states have reached the optimal values.
        
        This is determined by checking if the drive value is below a small threshold,
        indicating that the internal states are very close to their optimal values.
        
        :param current_internal_states: Current internal states (H_t).
        :return: Boolean indicating whether optimal states are reached.
        """
        drive_value = self.compute_drive(current_internal_states)
        threshold = 1e-3
        
        # Convert tensor to scalar if needed
        if isinstance(drive_value, torch.Tensor):
            drive_value = drive_value.item()
            
        # Check if drive is close enough to zero
        return drive_value < threshold
