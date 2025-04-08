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

    def compute_reward(self, current_internal_states, outcome):
        """
        Compute the reward as the reduction in drive from applying the outcome.

        :param current_internal_states: Current internal states (H_t).
        :param outcome: Change in internal state from the action (K_t).
        :return: Scalar torch.Tensor representing reward.
        """
        current_internal_states = self._to_tensor(current_internal_states)
        outcome = self._to_tensor(outcome)

        initial_drive = self.compute_drive(current_internal_states)
        new_drive = self.compute_drive(current_internal_states + outcome)

        reward = initial_drive - new_drive
        return reward
