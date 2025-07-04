from gymnasium import spaces
import numpy as np
from abc import abstractmethod

class ObservationFunction:
    """Abstract base class for observation functions."""

    def __init__(self, agent, env):
        """
        Initialize observation function.

        Args:
            agent (HomeostaticAgent): The agent to observe.
            env (NormarlHomeostaticEnv): The shared environment.
        """
        self.agent = agent
        self.env = env

    @abstractmethod
    def __call__(self):
        pass

    @abstractmethod
    def observation_space(self):
        pass


class DefaultHomeostaticObservation(ObservationFunction):
    def __init__(self, agent, env):
        super().__init__(agent, env)
        self.size = env.size
        self.dim_states = env.dimension_internal_states

    def __call__(self):
        return np.concatenate([
            np.array([self.agent.position], dtype=np.float32),
            self.agent.internal_states.astype(np.float32),
            self.agent.perceived_social_norm.astype(np.float32)
        ])

    def observation_space(self):
        low = np.concatenate([
            np.array([0.0]),
            np.full(self.dim_states, -1.0),
            np.zeros(self.dim_states)
        ])
        high = np.concatenate([
            np.array([self.size - 1.0]),
            np.full(self.dim_states, 1.0),
            np.ones(self.dim_states)
        ])
        return spaces.Box(low=low, high=high, dtype=np.float32)
