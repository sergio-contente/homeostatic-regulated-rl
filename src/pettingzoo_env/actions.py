from gymnasium import spaces
import numpy as np
from abc import abstractmethod
from typing import Dict, Any, Tuple, Optional

import functools


class ActionFunction:
    """Abstract base class for action functions."""

    def __init__(self, agent, env):
        """
        Initialize action function.

        Args:
            agent (HomeostaticAgent): The agent that will perform actions.
            env (NormarlHomeostaticEnv): The shared environment.
        """
        self.agent = agent
        self.env = env

    @abstractmethod
    def action_space(self):
        """Define the action space for this agent."""
        pass


class DefaultHomeostaticAction(ActionFunction):
    """Default action implementation for homeostatic agents."""
    
    def __init__(self, agent, env):
        super().__init__(agent, env)
        self.size = env.size
        self.dim_states = env.dimension_internal_states
        
        # Action mapping:
        # 0: stay in place
        # 1: move left  
        # 2: move right
        # 3+i: consume resource type i (if available at current position)
        self.num_movement_actions = 3
        self.num_consumption_actions = self.dim_states
        self.total_actions = self.num_movement_actions + self.num_consumption_actions
        

    @functools.lru_cache(maxsize=None)
    def action_space(self):
        """Define discrete action space."""
        return spaces.Discrete(self.total_actions)
    
    def decode_action(self, action: int) -> Dict[str, Any]:
        """
        Decode a discrete action into its semantic components.
        
        Args:
            action: Discrete action index
            
        Returns:
            Dictionary describing the action
        """
        if action < 0 or action >= self.total_actions:
            raise ValueError(f"Invalid action {action}. Must be in range [0, {self.total_actions-1}]")
        
        if action == 0:
            return {"type": "movement", "direction": "stay", "position_delta": 0}
        elif action == 1:
            return {"type": "movement", "direction": "left", "position_delta": -1}
        elif action == 2:
            return {"type": "movement", "direction": "right", "position_delta": +1}
        else:
            # Consumption action
            resource_idx = action - self.num_movement_actions
            resource_name = self.agent.get_state_names()[resource_idx]
            return {
                "type": "consumption", 
                "resource_idx": resource_idx,
                "resource_name": resource_name
            }
    
    def get_valid_actions(self) -> np.ndarray:
        """
        Get boolean mask of currently valid actions.
        
        Returns:
            Boolean array where True means action is valid
        """
        valid_actions = np.ones(self.total_actions, dtype=bool)
        
        current_pos = self.agent.position
        
        # Check movement action validity
        if current_pos <= 0:
            valid_actions[1] = False  # Can't move left at position 0
        if current_pos >= (self.size - 1):
            valid_actions[2] = False  # Can't move right at rightmost position
            
        # Check consumption action validity
        for i in range(self.dim_states):
            consumption_action = self.num_movement_actions + i
            
            # Check if agent is at resource location
            if hasattr(self.env, 'resources_info') and i in self.env.resources_info:
                resource_info = self.env.resources_info[i]
                resource_pos = resource_info["position"]
                
                # Resource must be at agent's position to be consumable
                if current_pos != resource_pos:
                    valid_actions[consumption_action] = False
                    
                # Additional check for availability if it exists
                if "available" in resource_info and not resource_info["available"]:
                    valid_actions[consumption_action] = False
            else:
                # For NORMARL-style environments without fixed resource positions,
                # allow consumption at any position (resources are distributed in stock)
                valid_actions[consumption_action] = True
        
        return valid_actions
    
    def execute_action(self, action: int) -> Dict[str, Any]:
        """
        Execute the given action and return results.
        
        Args:
            action: Discrete action to execute
            
        Returns:
            Dictionary with action results and effects
        """
        action_info = self.decode_action(action)
        results = {
            "action_taken": action_info,
            "valid": False,
            "position_changed": False,
            "resource_consumed": False,
            "intake": np.zeros(self.dim_states),
            "new_position": self.agent.position
        }
        
        # Check if action is valid
        valid_actions = self.get_valid_actions()
        if not valid_actions[action]:
            results["error"] = f"Invalid action {action} at current state"
            return results
        
        results["valid"] = True
        
        if action_info["type"] == "movement":
            # Execute movement
            old_position = self.agent.position
            new_position = old_position + action_info["position_delta"]
            
            # Clamp to valid range (double-check)
            new_position = max(0, min(self.size - 1, new_position))
            
            if new_position != old_position:
                self.agent.update_position(new_position)
                results["position_changed"] = True
                results["new_position"] = new_position
                
        elif action_info["type"] == "consumption":
            # Execute resource consumption
            resource_idx = action_info["resource_idx"]
            
            # Check if agent is at the correct resource position
            can_consume = True
            if hasattr(self.env, 'resources_info') and resource_idx in self.env.resources_info:
                resource_info = self.env.resources_info[resource_idx]
                resource_pos = resource_info["position"]
                
                # Only allow consumption if agent is at resource position
                if self.agent.position != resource_pos:
                    can_consume = False
                    results["error"] = f"Agent not at resource position (agent at {self.agent.position}, resource at {resource_pos})"
                
                # Check availability if it exists
                if "available" in resource_info and not resource_info["available"]:
                    can_consume = False
                    results["error"] = f"Resource {action_info['resource_name']} not available"
            
            if can_consume:
                # Create consumption array
                consumption_array = np.zeros(self.dim_states)
                consumption_array[resource_idx] = 1.0
                
                # Apply consumption to agent
                intake = self.agent.consume_resource(consumption_array)
                results["intake"] = intake
                results["resource_consumed"] = True
                results["consumed_resource"] = action_info["resource_name"]
                
                # Mark resource as consumed if applicable
                if hasattr(self.env, 'resources_info') and resource_idx in self.env.resources_info:
                    if "available" in self.env.resources_info[resource_idx]:
                        self.env.resources_info[resource_idx]["available"] = False
            else:
                results["valid"] = False
        
        return results
    

    
    def sample_action(self, temperature: float = 1.0, random_state: Optional[np.random.RandomState] = None) -> int:
        """
        Sample an action based on current agent state.
        
        Args:
            temperature: Softmax temperature (lower = more greedy)
            random_state: Random state for reproducibility
            
        Returns:
            Sampled action index
        """
        if random_state is None:
            random_state = np.random
            
        # Get valid actions
        valid_actions = self.get_valid_actions()
        valid_indices = np.where(valid_actions)[0]
        
        if len(valid_indices) == 0:
            # No valid actions - return stay action
            return 0
        
        # Simple uniform random selection from valid actions
        return random_state.choice(valid_indices)
    
    def get_action_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the action space.
        
        Returns:
            Dictionary with action space information
        """
        action_descriptions = []
        
        # Movement actions
        action_descriptions.append("Stay in place")
        action_descriptions.append("Move left")
        action_descriptions.append("Move right")
        
        # Consumption actions
        state_names = self.agent.get_state_names()
        for i, name in enumerate(state_names):
            action_descriptions.append(f"Consume {name}")
        
        return {
            "total_actions": self.total_actions,
            "movement_actions": self.num_movement_actions,
            "consumption_actions": self.num_consumption_actions,
            "action_descriptions": action_descriptions,
            "valid_actions": self.get_valid_actions().tolist(),
            "state_names": state_names
        }
    


class DefaultHomeostatic2DAction(ActionFunction):
    """Default action implementation for 2D homeostatic environments."""
    
    def __init__(self, agent, env):
        super().__init__(agent, env)
        self.size = env.size
        self.dim_states = env.dimension_internal_states
        
        # Action mapping for 2D:
        # 0: stay in place
        # 1: move left (x-1)
        # 2: move right (x+1)  
        # 3: move up (y+1)
        # 4: move down (y-1)
        # 5+i: consume resource type i (if available at current position)
        self.num_movement_actions = 5
        self.num_consumption_actions = self.dim_states
        self.total_actions = self.num_movement_actions + self.num_consumption_actions
        
    @functools.lru_cache(maxsize=None)
    def action_space(self):
        """Define discrete action space."""
        return spaces.Discrete(self.total_actions)
    
    def decode_action(self, action: int) -> Dict[str, Any]:
        """
        Decode a discrete action into its semantic components.
        
        Args:
            action: Discrete action index
            
        Returns:
            Dictionary describing the action
        """
        if action < 0 or action >= self.total_actions:
            raise ValueError(f"Invalid action {action}. Must be in range [0, {self.total_actions-1}]")
        
        if action == 0:
            return {"type": "movement", "direction": "stay", "position_delta": (0, 0)}
        elif action == 1:
            return {"type": "movement", "direction": "left", "position_delta": (-1, 0)}
        elif action == 2:
            return {"type": "movement", "direction": "right", "position_delta": (1, 0)}
        elif action == 3:
            return {"type": "movement", "direction": "up", "position_delta": (0, 1)}
        elif action == 4:
            return {"type": "movement", "direction": "down", "position_delta": (0, -1)}
        else:
            # Consumption action
            resource_idx = action - self.num_movement_actions
            resource_name = self.agent.get_state_names()[resource_idx]
            return {
                "type": "consumption", 
                "resource_idx": resource_idx,
                "resource_name": resource_name
            }
    
    def get_valid_actions(self) -> np.ndarray:
        """
        Get boolean mask of currently valid actions.
        
        Returns:
            Boolean array where True means action is valid
        """
        valid_actions = np.ones(self.total_actions, dtype=bool)
        
        current_pos = self.agent.position  # Should be [x, y] array
        
        # Check movement action validity for 2D boundaries
        if current_pos[0] <= 0:
            valid_actions[1] = False  # Can't move left at x=0
        if current_pos[0] >= (self.size - 1):
            valid_actions[2] = False  # Can't move right at x=size-1
        if current_pos[1] >= (self.size - 1):
            valid_actions[3] = False  # Can't move up at y=size-1
        if current_pos[1] <= 0:
            valid_actions[4] = False  # Can't move down at y=0
            
        # Check consumption action validity
        for i in range(self.dim_states):
            consumption_action = self.num_movement_actions + i
            
            # For 2D environments, check if agent is at resource location
            if hasattr(self.env, 'resources_info'):
                resource_info = self.env.resources_info.get(i)
                if resource_info is not None:
                    resource_pos = resource_info["position"]  # Should be [x, y] array
                    resource_available = resource_info["available"]
                    
                    # Resource must be at agent's position and available
                    if not np.array_equal(current_pos, resource_pos) or not resource_available:
                        valid_actions[consumption_action] = False
                else:
                    # No resource info available - disable consumption
                    valid_actions[consumption_action] = False
            else:
                # No resources in environment - allow consumption (for NORMARL-style envs)
                valid_actions[consumption_action] = True
        
        return valid_actions
    
    def execute_action(self, action: int) -> Dict[str, Any]:
        """
        Execute the given action and return results.
        
        Args:
            action: Discrete action to execute
            
        Returns:
            Dictionary with action results and effects
        """
        action_info = self.decode_action(action)
        results = {
            "action_taken": action_info,
            "valid": False,
            "position_changed": False,
            "resource_consumed": False,
            "intake": np.zeros(self.dim_states),
            "new_position": self.agent.position.copy()
        }
        
        # Check if action is valid
        valid_actions = self.get_valid_actions()
        if not valid_actions[action]:
            results["error"] = f"Invalid action {action} at current state"
            return results
        
        results["valid"] = True
        
        if action_info["type"] == "movement":
            # Execute movement
            old_position = self.agent.position.copy()
            delta = action_info["position_delta"]
            new_position = old_position + np.array(delta)
            
            # Clamp to valid range (double-check)
            new_position[0] = max(0, min(self.size - 1, new_position[0]))
            new_position[1] = max(0, min(self.size - 1, new_position[1]))
            
            if not np.array_equal(new_position, old_position):
                self.agent.position = new_position
                results["position_changed"] = True
                results["new_position"] = new_position
                
        elif action_info["type"] == "consumption":
            # Execute resource consumption
            resource_idx = action_info["resource_idx"]
            
            # Check if agent is at the correct resource position
            can_consume = True
            if hasattr(self.env, 'resources_info') and resource_idx in self.env.resources_info:
                resource_info = self.env.resources_info[resource_idx]
                resource_pos = resource_info["position"]  # Should be [x, y] array
                
                # Only allow consumption if agent is at resource position
                if not np.array_equal(self.agent.position, resource_pos):
                    can_consume = False
                    results["error"] = f"Agent not at resource position (agent at {self.agent.position}, resource at {resource_pos})"
                
                # Check availability if it exists
                if "available" in resource_info and not resource_info["available"]:
                    can_consume = False
                    results["error"] = f"Resource {action_info['resource_name']} not available"
            
            if can_consume:
                # Create consumption array
                consumption_array = np.zeros(self.dim_states)
                consumption_array[resource_idx] = 1.0
                
                # Apply consumption to agent
                intake = self.agent.consume_resource(consumption_array)
                results["intake"] = intake
                results["resource_consumed"] = True
                results["consumed_resource"] = action_info["resource_name"]
                
                # Mark resource as consumed in environment if applicable
                if hasattr(self.env, 'resources_info') and resource_idx in self.env.resources_info:
                    if "available" in self.env.resources_info[resource_idx]:
                        self.env.resources_info[resource_idx]["available"] = False
            else:
                results["valid"] = False
        
        return results
    

    
    def sample_action(self, temperature: float = 1.0, random_state: Optional[np.random.RandomState] = None) -> int:
        """
        Sample an action based on current agent state.
        
        Args:
            temperature: Softmax temperature (lower = more greedy)
            random_state: Random state for reproducibility
            
        Returns:
            Sampled action index
        """
        if random_state is None:
            random_state = np.random
            
        # Get valid actions
        valid_actions = self.get_valid_actions()
        valid_indices = np.where(valid_actions)[0]
        
        if len(valid_indices) == 0:
            # No valid actions - return stay action
            return 0
        
        # Simple uniform random selection from valid actions
        return random_state.choice(valid_indices)
    
    def get_action_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the action space.
        
        Returns:
            Dictionary with action space information
        """
        action_descriptions = []
        
        # Movement actions
        action_descriptions.append("Stay in place")
        action_descriptions.append("Move left")
        action_descriptions.append("Move right")
        action_descriptions.append("Move up")
        action_descriptions.append("Move down")
        
        # Consumption actions
        state_names = self.agent.get_state_names()
        for i, name in enumerate(state_names):
            action_descriptions.append(f"Consume {name}")
        
        return {
            "total_actions": self.total_actions,
            "movement_actions": self.num_movement_actions,
            "consumption_actions": self.num_consumption_actions,
            "action_descriptions": action_descriptions,
            "valid_actions": self.get_valid_actions().tolist(),
            "state_names": state_names,
            "environment_type": "2D"
        }
    
    