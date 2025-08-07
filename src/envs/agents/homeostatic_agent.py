import numpy as np
from typing import Dict, Any, Optional
from src.utils.get_params import ParameterHandler


class HomeostaticAgent:
    """
    Homeostatic agent implementation with drive-based behavior and social norm learning.
    
    Each agent maintains internal homeostatic states and learns social norms from 
    observing other agents' behavior. The agent experiences drives based on deviations
    from optimal internal states and incorporates social costs in decision-making.
    """
    
    def __init__(
        self, 
        agent_id: str,
        config_path: str,
        drive_type: str,
        initial_position: int = 0,
        social_learning_rate: float = 0.1,
        beta: float = 0.5,
        initial_internal_states: Optional[np.ndarray] = None
    ):
        """
        Initialize the homeostatic agent.
        
        Args:
            agent_id: Unique identifier for the agent
            config_path: Path to configuration file for drive parameters
            drive_type: Type of drive to use ('base_drive', 'interoceptive_drive', 'elliptic_drive')
            initial_position: Starting position in the environment
            social_learning_rate: Learning rate for social norm adaptation (alpha)
            beta: Social norm internalization strength (beta)
            initial_internal_states: Initial internal state values (if None, randomized)
        """
        self.agent_id = agent_id
        self.social_learning_rate = social_learning_rate
        self.beta = beta
        
        # Initialize drive system
        self.param_manager = ParameterHandler(config_path)
        self.drive = self.param_manager.create_drive(drive_type)
        self.dimension_internal_states = self.drive.get_internal_state_dimension()
        
        # Agent state
        self.position = initial_position
        
        # Initialize internal states
        if initial_internal_states is not None:
            self.internal_states = initial_internal_states.copy()
        else:
            # Random initialization around neutral values
            self.internal_states = np.random.uniform(
                low=-0.3, high=0.3, 
                size=(self.dimension_internal_states,)
            ).astype(np.float64)
        
        # Social norm perception - starts at zero (no prior knowledge)
        self.perceived_social_norm = np.zeros(self.dimension_internal_states, dtype=np.float64)
        
        # Initialize drive
        initial_drive = self.drive.compute_drive(self.internal_states)
        self.drive.update_drive(initial_drive)
        
        # Memory for learning
        self.intake_history = []
        
        # Track agent's own consumption for social learning
        self.last_intake = np.zeros(self.dimension_internal_states, dtype=np.float64)
        
    def get_state_names(self):
        """Get the names of internal states."""
        return self.drive.get_internal_states_names()
        
    def get_current_drive(self):
        """Get the current drive value."""
        return self.drive.get_current_drive()
        
    def update_position(self, new_position: int):
        """Update agent's position."""
        self.position = new_position
        
    def apply_natural_decay(self):
        """Apply natural decay to internal states."""
        self.internal_states = self.drive.apply_natural_decay(self.internal_states)
        
    def consume_resource(self, resource_to_consume: np.ndarray) -> np.ndarray:
        """
        Consume resources and update internal states.
        
        Args:
            resource_to_consume: Boolean array indicating which resources to consume
            
        Returns:
            intake: Array of actual intake amounts
        """
        # Apply intake to internal states
        states_before = self.internal_states.copy()
        self.internal_states = self.drive.apply_intake(self.internal_states, resource_to_consume)
        
        # Calculate actual intake (for social learning)
        self.last_intake = self.drive.get_intake_array(resource_to_consume)
        self.intake_history.append(self.last_intake.copy())
        
        return self.last_intake, states_before
        
    def compute_homeostatic_reward(self, old_states) -> float:
        """
        Compute reward based on drive reduction.
        
        Returns:
            reward: Homeostatic reward from drive reduction
        """
        old_drive = self.drive.compute_drive(old_states)
        print(f" New states: {self.internal_states}")
        new_drive = self.drive.compute_drive(self.internal_states)
        reward = self.drive.compute_reward(old_drive, new_drive)
        self.drive.update_drive(new_drive)
        return reward
        
    def compute_social_cost(self, intake: np.ndarray, resource_scarcity: np.ndarray) -> float:
        """
        Compute social cost based on NORMARL mechanism.
        
        Social cost formula:
        Si(Qi) = beta_i * (Qi - Q̄i) * max{0, (a - b*E)} if Qi ≥ Q̄i
        Si(Qi) = 0 if Qi < Q̄i
        
        Args:
            intake: Agent's consumption in this step
            resource_scarcity: Scarcity factor for each resource type
            
        Returns:
            social_cost: Total social cost for this action
        """
        social_cost = 0.0
        
        for i in range(self.dimension_internal_states):
            if intake[i] >= self.perceived_social_norm[i]:
                excess_consumption = intake[i] - self.perceived_social_norm[i]
                social_cost += self.beta * excess_consumption * resource_scarcity[i]
                
        return social_cost
        
    def update_social_norm_perception(self, observed_average_intake: np.ndarray):
        """
        Update perception of social norms based on observed behavior.
        
        Uses the NORMARL update rule:
        Q̄i(t+1) = (1 - alpha_i) * Q̄i(t) + alpha_i * observed_average
        
        Args:
            observed_average_intake: Average intake observed from all agents
        """
        self.perceived_social_norm = (
            (1 - self.social_learning_rate) * self.perceived_social_norm + 
            self.social_learning_rate * observed_average_intake
        )
                
    def get_state_info(self) -> Dict[str, Any]:
        """
        Get comprehensive state information for debugging/logging.
        
        Returns:
            state_info: Dictionary containing agent's current state
        """
        return {
            'agent_id': self.agent_id,
            'position': self.position,
            'internal_states': self.internal_states.copy(),
            'perceived_social_norm': self.perceived_social_norm.copy(),
            'current_drive': self.get_current_drive(),
            'last_intake': self.last_intake.copy(),
            'beta': self.beta,
            'social_learning_rate': self.social_learning_rate
        }
        
    def is_in_critical_state(self, threshold: float = 1.0) -> bool:
        """
        Check if agent is in a critical homeostatic state.
        
        Args:
            threshold: Threshold for critical state detection
            
        Returns:
            critical: True if any internal state is near extreme values
        """
        return np.any(np.abs(self.internal_states) > threshold)
        
    def reset(self, initial_position: Optional[int] = None, 
              initial_internal_states: Optional[np.ndarray] = None):
        """
        Reset agent to initial state.
        
        Args:
            initial_position: New starting position (if None, use random)
            initial_internal_states: New starting internal states (if None, use random)
        """
        if initial_position is not None:
            self.position = initial_position
        else:
            self.position = 0
            
        if initial_internal_states is not None:
            self.internal_states = initial_internal_states.copy()
        else:
            self.internal_states = np.random.uniform(
                low=-0.3, high=0.3, 
                size=(self.dimension_internal_states,)
            ).astype(np.float64)
            
        # Reset social perception and history
        self.perceived_social_norm = np.zeros(self.dimension_internal_states, dtype=np.float64)
        self.last_intake = np.zeros(self.dimension_internal_states, dtype=np.float64)
        self.intake_history = []
        
        # Reset drive
        initial_drive = self.drive.compute_drive(self.internal_states)
        self.drive.update_drive(initial_drive)
        
    def __str__(self) -> str:
        """String representation of the agent."""
        state_names = self.get_state_names()
        state_str = ", ".join([
            f"{name}: {value:.3f}" 
            for name, value in zip(state_names, self.internal_states)
        ])
        norm_str = ", ".join([
            f"{name}_norm: {value:.3f}" 
            for name, value in zip(state_names, self.perceived_social_norm)
        ])
        
        return (f"HomeostaticAgent(id={self.agent_id}, pos={self.position}, "
                f"drive={self.get_current_drive():.3f}, beta={self.beta:.2f}, "
                f"states=[{state_str}], norms=[{norm_str}])")
