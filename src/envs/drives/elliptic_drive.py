import numpy as np
from .base_drive import BaseDrive

class EllipticDrive(BaseDrive):
    """
    A drive function where each internal state component has its own exponent.
    Implements:
        D(H_t) = (sum_i |h*_i - h_i,t|^{n_i})^{1/m}
    """

    def __init__(self, optimal_internal_states_config, n_vector, m):
        """
        :param optimal_internal_states_config: Target internal states vector (H*).
        :param n_vector: Vector of exponents [n₁, n₂, ..., n_N], one per internal state dimension.
        :param m: Root parameter (same as original BaseDrive).
        """
        super().__init__(optimal_internal_states_config, m=m, n=1)  # n unused here
        self.n_vector = self._to_array(n_vector)
        
        # Obtenha o tensor dos estados ótimos para verificação de dimensão
        optimal_states_array = self.get_array_optimal_states_values()
        
        if self.n_vector.shape[0] != optimal_states_array.shape[0]:
            raise ValueError(f"n_vector dimension ({self.n_vector.shape[0]}) must match optimal_internal_states dimension ({optimal_states_array.shape[0]})")

    def compute_drive(self, current_internal_states):
        """
        Computes the elliptic drive using element-specific exponents.

        :param current_internal_states: Vector of current internal states H_t.
        :return: Scalar drive value.
        """
        current_internal_states = self._to_array(current_internal_states)
        optimal_states_array = self.get_array_optimal_states_values()

        # Compute |h*_i - h_i,t| for each i
        diff = np.abs(optimal_states_array - current_internal_states)

        # Apply per-element exponent: |h*_i - h_i,t|^{n_i}
        powered = diff ** self.n_vector

        # Sum and apply m-th root
        drive_sum = np.sum(powered)
        drive_value = drive_sum ** (1 / self.m)

        return drive_value
