# src/drives/base_drive.py

import torch
from abc import ABC, abstractmethod

class BaseDrive(ABC):
    """
    Abstract base class for implementing homeostatic drives in reinforcement learning.
    """

    def __init__(self, optimal_internal_states, m=1, n=1):
        """
        Initialize the base drive class.

        :param optimal_internal_states: Target internal state vector (H*), as list or torch.Tensor.
        :param m: Root parameter used in the drive computation.
        :param n: Exponent parameter used in the drive computation.
        """
        self.optimal_internal_states = self._to_tensor(optimal_internal_states)
        self.m = m
        self.n = n

    def _to_tensor(self, x):
        """
        Utility method to ensure input is a detached torch.Tensor.
        """
        if isinstance(x, torch.Tensor):
            return x.clone().detach()
        return torch.tensor(x, dtype=torch.float32)

    @abstractmethod
    def compute_drive(self, current_internal_states):
        """
        Compute the drive based on current internal states.

        :param current_internal_states: Current internal states (H_t).
        :return: Scalar torch.Tensor representing drive value.
        """
        pass

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
