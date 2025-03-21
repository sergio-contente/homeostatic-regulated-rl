import torch

class Drive:
    def __init__(self, optimal_internal_states, m, n):
        """
        Initializes the Drive class.

        :param optimal_internal_states: PyTorch tensor representing the optimal internal states (h*).
        :param m: Root parameter in the drive function.
        :param n: Exponent parameter in the drive function.
        """
        self.optimal_internal_states = torch.tensor(optimal_internal_states, dtype=torch.float32)
        self.m = m
        self.n = n

    def compute_drive(self, current_internal_states):
        """
        Computes the drive D(H_t) based on the current internal states.

        :param current_internal_states: PyTorch tensor representing the current internal states (h_i,t).
        :return: Homeostatic drive value.
        """
        current_internal_states = torch.tensor(current_internal_states, dtype=torch.float32)
        
        # Compute the sum of deviations raised to the power of n
        drive_sum = torch.sum(torch.abs(self.optimal_internal_states - current_internal_states) ** self.n)

        # Apply the m-th root
        drive_value = drive_sum ** (1 / self.m)
        
        return drive_value

# Example usage
optimal_states = [1.0, 2.0, 3.0]
drive_model = Drive(optimal_states, m=2, n=3)

current_states = [1.2, 1.8, 2.5]
drive_value = drive_model.compute_drive(current_states)
print("Homeostatic Drive:", drive_value.item()) 
