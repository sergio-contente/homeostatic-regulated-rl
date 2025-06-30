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
        return {
            "position": self.agent.position,
            "internal_states": self.agent.internal_states,
            "perceived_social_norm": self.agent.perceived_social_norm
        }

    def observation_space(self):
        return spaces.Dict({
            "position": spaces.Discrete(self.size),
            "internal_states": spaces.Box(low=-1.0, high=1.0, shape=(self.dim_states,), dtype=np.float64),
            "perceived_social_norm": spaces.Box(low=0.0, high=1.0, shape=(self.dim_states,), dtype=np.float64)
        })
