import gymnasium as gym
import numpy as np

class DiscretizeWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_bins=10, low=0.0, high=300.0):
        super().__init__(env)
        self.n_bins = n_bins
        self.low = low
        self.high = high
        
        self.bins = np.linspace(low, high, n_bins + 1)[1:-1]  # pontos de corte entre os bins
        self.observation_space = gym.spaces.MultiDiscrete([n_bins] * env.observation_space["internal_states"].shape[0])

    def observation(self, observation):
        internal_states = observation["internal_states"]
        discrete_states = np.digitize(internal_states, bins=self.bins)  # retorna índices de bins
        return discrete_states  # pode retornar como tuple ou array
