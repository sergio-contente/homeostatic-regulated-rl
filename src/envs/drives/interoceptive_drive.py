import numpy as np
from .base_drive import BaseDrive

class InteroceptiveModulationDrive(BaseDrive):
    def __init__(self, optimal_internal_states_config, m, n, eta=1.0):
        """
        Subclass of BaseDrive that applies an interoceptive modulation factor (η)
        to the perceived difference between the current and optimal internal states.

        :param optimal_internal_states_config: Target internal state vector (H*).
        :param m: Root parameter used in the drive computation.
        :param n: Exponent parameter used in the drive computation.
        :param eta: Interoceptive modulation factor (η).
        """
        super().__init__(optimal_internal_states_config, m, n)
        self.eta = eta

    def compute_drive(self, current_internal_states):
        """
        Compute the interoceptively modulated drive based on the current internal state.

        :param current_internal_states: The current internal state vector (H_t).
        :return: Scalar torch.Tensor representing the modulated drive value.
        """
        current_internal_states = self._to_array(current_internal_states)
        optimal_states_array = self.get_array_optimal_states_values()
        
        # Apply interoceptive modulation
        diff = self.eta * (optimal_states_array - current_internal_states)
        
        drive_sum = np.sum(np.abs(diff) ** self.n)
        drive_value = drive_sum ** (1 / self.m)
        
        return drive_value
