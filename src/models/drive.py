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

    def _to_tensor(self, x):
        """Ensures input is a PyTorch tensor and detached from computation graph."""
        if isinstance(x, torch.Tensor):
            return x.clone().detach()
        return torch.tensor(x, dtype=torch.float32)

    def compute_drive(self, current_internal_states):
        """
        Computes the drive D(H_t) based on the current internal states.

        :param current_internal_states: PyTorch tensor or list representing the current internal states (h_i,t).
        :return: Homeostatic drive value.
        """
        current_internal_states = self._to_tensor(current_internal_states)
        
        # Compute the sum of deviations raised to the power of n
        drive_sum = torch.sum(torch.abs(self.optimal_internal_states - current_internal_states) ** self.n)

        # Apply the m-th root
        drive_value = drive_sum ** (1 / self.m)
        
        return drive_value

    def compute_reward(self, current_internal_states, outcome):
        """
        Computes the reward based on the reduction in homeostatic drive.

        :param current_internal_states: PyTorch tensor or list representing the current internal states (H_t).
        :param outcome: PyTorch tensor or list representing the impact of the action (K_t).
        :return: Reward value.
        """
        current_internal_states = self._to_tensor(current_internal_states)
        outcome = self._to_tensor(outcome)

        # Compute initial drive D(H_t)
        initial_drive = self.compute_drive(current_internal_states)

        # Compute new drive after applying outcome D(H_t + K_t)
        new_drive = self.compute_drive(current_internal_states + outcome)

        # Compute reward as the reduction in drive
        reward = initial_drive - new_drive

        return reward

# Example usage
optimal_states = [1.0, 2.0, 3.0]  # Example of optimal internal states
drive_model = Drive(optimal_states, m=2, n=3)


current_states = [1.2, 1.8, 2.5]  # Example of current internal states

print("Drive: ", drive_model.compute_drive(current_states).item())

outcome = [-0.2, 0.3, 0.1]  # Example of action impact (K_t)

reward_value = drive_model.compute_reward(current_states, outcome)
print("Reward:", reward_value.item())  # Convert tensor to scalar value
