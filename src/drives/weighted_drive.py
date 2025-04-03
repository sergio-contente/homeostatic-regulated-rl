# src/drives/weighted_drive.py

import torch
from .base_drive import BaseDrive

class WeightedDrive(BaseDrive):
    """
    Implements a drive function with nutrient-specific weights as described by Yoshida et al. (2024):

    d = 2w_red / (w_blue + w_red) * s_red^2 + 2w_blue / (w_blue + w_red) * s_blue^2
    """

    def __init__(self, w_red=1.0, w_blue=1.0):
        """
        :param w_red: Weight for the red nutrient.
        :param w_blue: Weight for the blue nutrient.
        """
        # We assume optimal internal state is always [0, 0] in this model (the origin)
        super().__init__([0.0, 0.0])
        self.w_red = w_red
        self.w_blue = w_blue

    def compute_drive(self, current_internal_states):
        current_internal_states = self._to_tensor(current_internal_states)
        s_red, s_blue = current_internal_states[0], current_internal_states[1]

        wr = self.w_red
        wb = self.w_blue
        total_weight = wb + wr

        d = (2 * wr / total_weight) * (s_red ** 2) + (2 * wb / total_weight) * (s_blue ** 2)
        return d
