import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from ...utils.convertion import ToTensor


class InternalStateWrapper(gym.Wrapper):
    """
    Wrapper que adiciona estados internos à observação e calcula efeitos de recursos
    posicionados aleatoriamente no ambiente.
    """

    def __init__(self, env, internal_state_size=2, n_resources=2, initial_state_range=(0.2, 0.5)):
        super().__init__(env)
        self.internal_state_size = internal_state_size
        self.n_resources = n_resources
        self.initial_state_range = initial_state_range

        self.observation_space = spaces.Dict({
            **self.env.observation_space.spaces,
            "internal_states": spaces.Box(0.0, 1.0, shape=(internal_state_size,), dtype=np.float32),
            "outcome": spaces.Box(-1.0, 1.0, shape=(internal_state_size,), dtype=np.float32)
        })

        self.resource_locations = {}
        self._internal_states = None
        self._previous_internal_states = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._internal_states = self._init_internal_states()
        self._previous_internal_states = self._internal_states.clone()

        self._setup_resources()

        outcome = torch.zeros_like(self._internal_states)
        obs = self._augment_observation(obs, outcome)
        info = self._augment_info(info, outcome)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        total_outcome = self._apply_outcome(tuple(self.env._agent_location))
        self._previous_internal_states = self._internal_states.clone()

        obs = self._augment_observation(obs, total_outcome)
        info = self._augment_info(info, total_outcome)

        return obs, reward, terminated, truncated, info

    # ----------- MÉTODOS AUXILIARES -----------

    def _init_internal_states(self):
        return ToTensor(np.random.uniform(
            self.initial_state_range[0],
            self.initial_state_range[1],
            size=self.internal_state_size
        ))

    def _setup_resources(self):
        """
        Posiciona recursos aleatórios no grid, cada um com um vetor de efeito
        sobre os estados internos.
        """
        self.resource_locations.clear()
        occupied = {tuple(self.env._agent_location), tuple(self.env._target_location)}

        while len(self.resource_locations) < self.n_resources:
            loc = tuple(self.env.np_random.integers(0, self.env.size, size=2, dtype=int))
            if loc in occupied:
                continue
            occupied.add(loc)

            # Cria vetor de efeito aleatório sobre os internal states
            effect = np.zeros(self.internal_state_size)
            affected_idx = self.env.np_random.choice(
                self.internal_state_size, size=1, replace=False)
            effect[affected_idx] = self.env.np_random.uniform(0.2, 0.4)

            self.resource_locations[loc] = {
                'effect': ToTensor(effect)
            }

    def _apply_outcome(self, agent_loc):
        """
        Aplica efeitos de recursos e decaimento homeostático nos estados internos.
        """
        effect = self.resource_locations.get(agent_loc, {}).get('effect', torch.zeros_like(self._internal_states))
        decay = ToTensor(np.ones(self.internal_state_size) * 0.01)
        total_outcome = effect - decay

        self._internal_states = torch.clamp(self._internal_states + total_outcome, 0.0, 1.0)
        return total_outcome

    def _augment_observation(self, obs, outcome):
        return {
            **obs,
            "internal_states": self._internal_states.numpy(),
            "outcome": outcome.numpy()
        }

    def _augment_info(self, info, outcome):
        return {
            **info,
            "internal_states": self._internal_states.numpy(),
            "previous_internal_states": self._previous_internal_states.numpy(),
            "outcome": outcome.numpy(),
            "resource_locations": self.resource_locations
        }
