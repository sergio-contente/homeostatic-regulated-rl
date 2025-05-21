# src/drives/base_drive.py

import numpy as np

class BaseDrive():
    """
    Base class for implementing homeostatic drives in reinforcement learning.
    Provides default drive computation, but subclasses may override it.
    """

    def __init__(self, optimal_internal_states_config, m=1, n=1):
        """
        Initialize the base drive class.

        :param optimal_internal_states_config: Target internal state config as dict with structure:
            {state_name: {'value': float, 'loss': float, 'intake': float}, ...}
        :param m: Root parameter used in the drive computation.
        :param n: Exponent parameter used in the drive computation.
        """
        self._optimal_internal_states_config = optimal_internal_states_config
        self.m = m
        self.n = n
        self._current_drive = None
        
        # Extrair valores ótimos para cada estado interno
        self._optimal_internal_states_values = {
            state_name: state_config['value']
            for state_name, state_config in optimal_internal_states_config.items()
        }
        
        # Extrair taxas de perda para cada estado interno
        self._internal_states_loss_rates = {
            state_name: state_config['loss']
            for state_name, state_config in optimal_internal_states_config.items()
        }
        
        # Extrair taxas de ingestão para cada estado interno
        self._internal_states_intake_rates = {
            state_name: state_config['intake']
            for state_name, state_config in optimal_internal_states_config.items()
        }

        self._internal_states_resources_regen = {
            state_name: state_config['regeneration']
            for state_name, state_config in optimal_internal_states_config.items()
        }

    def _to_array(self, x):
        """
        Utility method to ensure input is a numpy array.
        """
        if isinstance(x, np.ndarray):
            return x.copy()
        return np.array(x, dtype=np.float32)
    
    def get_state_value(self, state_name):
        """
        Get the optimal value for a specific internal state.
        """
        return self._optimal_internal_states_values[state_name]
    
    def get_state_loss_rate(self, state_name):
        """
        Get the loss rate for a specific internal state.
        """
        return self._internal_states_loss_rates[state_name]
    
    def get_state_intake_rate(self, state_name):
        """
        Get the intake rate for a specific internal state.
        """
        return self._internal_states_intake_rates[state_name]
    
    def get_state_resources_regen_rate(self, state_name):
        """
        Get the resource regeneration rate for a specific internal state.
        """
        return self._internal_states_resources_regen[state_name]
    
    def get_array_optimal_states_values(self):
        """
        Convert optimal state values to a numpy array.
        """
        values = []
        for state_name in self._optimal_internal_states_values:
            values.append(self._optimal_internal_states_values[state_name])
        return np.array(values, dtype=np.float32)

    def get_array_loss_rates(self):
        """
        Convert loss rates to a numpy array.
        """
        rates = []
        for state_name in self._internal_states_loss_rates:
            rates.append(self._internal_states_loss_rates[state_name])
        return np.array(rates, dtype=np.float32)
    
    def get_array_intake_rates(self):
        """
        Convert intake rates to a numpy array.
        """
        rates = []
        for state_name in self._internal_states_intake_rates:
            rates.append(self._internal_states_intake_rates[state_name])
        return np.array(rates, dtype=np.float32)
    
    def get_array_resources_regeneration_rate(self):
        """
        Convert resource regeneration rates to a numpy array.
        """
        rates = []
        for state_name in self._internal_states_resources_regen:
            rates.append(self._internal_states_resources_regen[state_name])
        return np.array(rates, dtype=np.float32)

    def has_state(self, state_name):
        """
        Check if a specific internal state exists.
        """
        return state_name in self._optimal_internal_states_config

    def get_internal_state_dimension(self):
        """
        Get the number of internal states.
        """
        return len(self._optimal_internal_states_config)

    def compute_drive(self, current_internal_states):
        """
        Compute the drive as the root of the sum of powered deviations from the optimal state.

        This is a default implementation based on:
        D(H_t) = (sum_i |H*_i - H_{i,t}|^n)^{1/m}

        Subclasses may override this to apply modulation, weighting, or other mechanisms.

        :param current_internal_states: Current internal states (H_t).
        :return: Scalar numpy value representing drive value.
        """
        current_internal_states = self._to_array(current_internal_states)
        optimal_states_array = self.get_array_optimal_states_values()

        diff = optimal_states_array - current_internal_states
        drive_sum = np.sum(np.abs(diff) ** self.n)
        drive_value = drive_sum ** (1 / self.m)
        return drive_value
    
    def update_drive(self, drive_value):
        """
        Update the current drive value.
        """
        self._current_drive = drive_value

    def compute_reward(self, new_drive):
        """
        Compute the reward as the reduction in drive from applying the outcome.

        :param new_drive:  New drive (H_{t+1}).
        :return: Scalar numpy value representing reward.
        """
        reward = self._current_drive - new_drive
        return reward
    
    def has_reached_optimal(self, current_internal_states, threshold):
        """
        Check if the current internal states are within threshold of the optimal states.
        """
        keys = list(self._optimal_internal_states_values.keys())
        
        for i, key in enumerate(keys):
            optimal_value = self._optimal_internal_states_values[key]
            current_value = current_internal_states[i]  # Assumindo que current_internal_states é um array
            
            if abs(optimal_value - current_value) > threshold:
                return False
        return True
    
    def get_current_drive(self):
        """
        Get the current drive value.
        """
        return self._current_drive

    def get_internal_states_names(self):
        """
        Get the names of all internal states.
        """
        return list(self._optimal_internal_states_config.keys())
    
    def apply_natural_decay(self, current_internal_states):
        """
        Apply natural decay to internal states based on loss rates.
        
        :param current_internal_states: Current internal state values.
        :return: Updated internal state values after decay.
        """
        current_states = self._to_array(current_internal_states)
        loss_rates = self.get_array_loss_rates()
        
        # Apply decay: subtract loss rate from each state
        updated_states = current_states - loss_rates
        
        return updated_states
    
    def apply_intake(self, current_internal_states, action_states):
        """
        Apply intake to internal states based on action and intake rates.
        
        :param current_internal_states: Current internal state values.
        :param action_states: Boolean array indicating which states to apply intake to.
        :return: Updated internal state values after intake.
        """
        current_states = self._to_array(current_internal_states)
        intake_rates = self.get_array_intake_rates()
        action_states = self._to_array(action_states)
        
        # Apply intake only to states specified by action_states
        intake = intake_rates * action_states
        updated_states = current_states + intake
        
        return updated_states
    
    def apply_resource_regeneration(self, resource_available, resource_name):
        if not resource_available:
            random_number = np.random.uniform(0, 1)
            if random_number < self.get_state_resources_regen_rate(resource_name):
                return True
            else:
                return False
        else:
            return True
