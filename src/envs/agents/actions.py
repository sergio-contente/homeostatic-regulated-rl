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
    
    def execute_action(self, action: int) -> Dict[str, Any]:
        """
        Execute the given action and return results.
        """
        agent_old_position = self.agent.position
        agent_new_position = agent_old_position
        if action == 0:
            agent_new_position = agent_old_position
        elif action == 1:
            agent_new_position = max(0, agent_old_position - 1)
        elif action == 2:
            agent_new_position = min(self.size - 1, agent_old_position + 1)
        else:
            agent_new_position = agent_old_position

        resources_to_consume = np.zeros(self.dim_states)
        for i in range(self.dim_states):
            if action == 3 + i:
                # Only consume if agent is at the resource position
                if agent_old_position == self.env.resources_info[i]["position"]:
                    resources_to_consume[i] = 1.0

        return {
            "action_taken": action,
            "agent_new_position": agent_new_position,
            "resources_to_consume": resources_to_consume
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
    
    def execute_action(self, action: int) -> Dict[str, Any]:
        """
        Execute the given action and return results.
        """
        action_info = self.decode_action(action)
        results = {
            "action_taken": action_info,
            "valid": False,
        }
    