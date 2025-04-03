import torch
from .base_drive import BaseDrive

class InteroceptiveModulationDrive(BaseDrive):
    def __init__(self, optimal_internal_states, m, n, eta=1.0):
        """
        Subclass of BaseDrive that applies an interoceptive modulation factor (η)
        to the perceived difference between the current and optimal internal states.

        :param optimal_internal_states: Target internal state vector (H*).
        :param m: Root parameter used in the drive computation.
        :param n: Exponent parameter used in the drive computation.
        :param eta: Interoceptive modulation factor (η).
        """
        super().__init__(optimal_internal_states, m, n)
        self.eta = eta

    def compute_drive(self, current_internal_states):
        """
        Compute the interoceptively modulated drive based on the current internal state.

        :param current_internal_states: The current internal state vector (H_t).
        :return: Scalar torch.Tensor representing the modulated drive value.
        """
        current_internal_states = self._to_tensor(current_internal_states)
        
        # Apply interoceptive modulation
        diff = self.eta * (self.optimal_internal_states - current_internal_states)
        
        drive_sum = torch.sum(torch.abs(diff) ** self.n)
        drive_value = drive_sum ** (1 / self.m)
        
        return drive_value
