from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import torch


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3


class DriveType(Enum):
    BASE = "base"
    INTEROCEPTIVE = "interoceptive" 
    ELLIPTIC = "elliptic"
    WEIGHTED = "weighted"


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self, 
        render_mode=None, 
        internal_state_size=2, 
        size=5, 
        initial_state_range=(0.2, 0.7), 
        n_resources=2,
        **drive_params
    ):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.internal_state_size = internal_state_size
        self.resource_locations = {}
        self.initial_state_range = initial_state_range
        self.n_resources = n_resources
        
        # Drive settings
        self.drive_type = DriveType(drive_type)
        self.drive_params = drive_params or {}
        
        self._internal_states = None
        self._previous_internal_states = None
        self._outcome = None

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "internal_states": spaces.Box(0.0, 1.0, shape=(internal_state_size,), dtype=np.float32),
                "outcome": spaces.Box(-1.0, 1.0, shape=(internal_state_size,), dtype=np.float32)
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, 1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        
        # Initialize the appropriate drive
        self._init_drive()

    def _init_drive(self):
        """Initialize the appropriate drive based on the specified type."""
        if self.drive_type == DriveType.BASE:
            self.drive = self._compute_base_drive
        elif self.drive_type == DriveType.INTEROCEPTIVE:
            self.drive = self._compute_interoceptive_drive
        elif self.drive_type == DriveType.ELLIPTIC:
            self.drive = self._compute_elliptic_drive
        elif self.drive_type == DriveType.WEIGHTED:
            self.drive = self._compute_weighted_drive
        else:
            raise ValueError(f"Unknown drive type: {self.drive_type}")

    def _to_tensor(self, array):
        """Convert numpy array to torch tensor."""
        return torch.tensor(array, dtype=torch.float32)

    def _init_internal_states(self):
        """Initialize internal states with random values."""
        return self._to_tensor(np.random.uniform(
            self.initial_state_range[0],
            self.initial_state_range[1],
            size=self.internal_state_size
        ))
    
    def _setup_resources(self):
        """
        Place random resources in the grid, each with an effect vector
        on internal states.
        """
        self.resource_locations.clear()
        occupied = {tuple(self._agent_location), tuple(self._target_location)}

        while len(self.resource_locations) < self.n_resources:
            # Generate random location
            loc = tuple(self.np_random.integers(0, self.size, size=2, dtype=int))
            if loc in occupied:
                continue
            occupied.add(loc)

            # Create random effect vector (affects one internal state)
            effect = np.zeros(self.internal_state_size)
            affected_idx = self.np_random.choice(
                self.internal_state_size, size=1, replace=False)
            effect[affected_idx] = self.np_random.uniform(0.2, 0.4)

            self.resource_locations[loc] = {
                'effect': self._to_tensor(effect)
            }

    def _get_obs(self):  # Private method
        return {
                "agent": self._agent_location, 
                "target": self._target_location,
                "internal_states": self._internal_states,
                "outcome": self._outcome
                }

    def _get_info(self):  # Private method
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            ),
            "drive_value": self._compute_drive_reward()
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not
        # coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        self._internal_states = self._init_internal_states()
        self._previous_internal_states = self._internal_states.clone()
        self._outcome = torch.zeros_like(self._internal_states)

        self._setup_resources()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Save previous internal states
        self._previous_internal_states = self._internal_states.clone()
        
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        
        # Check if agent is at a resource location
        agent_pos_tuple = tuple(self._agent_location)
        if agent_pos_tuple in self.resource_locations:
            # Apply resource effect to internal states
            resource = self.resource_locations[agent_pos_tuple]
            self._internal_states = torch.clamp(
                self._internal_states + resource['effect'], 
                0.0, 1.0
            )
            # Calculate outcome (change in internal states)
            self._outcome = self._internal_states - self._previous_internal_states
        else:
            # Small decay in internal states when no resource is consumed
            decay = self._to_tensor(np.full(self.internal_state_size, 0.01))
            self._internal_states = torch.clamp(
                self._internal_states - decay,
                0.0, 1.0
            )
            self._outcome = self._internal_states - self._previous_internal_states
        
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        
        # Calculate the drive-based reward
        drive_reward = self._compute_drive_reward()
        
        # Combine with goal-reaching reward if desired
        goal_reward = 1.0 if terminated else 0.0
        reward = drive_reward + goal_reward * self.drive_params.get('goal_weight', 1.0)
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _compute_drive_reward(self):
        """Compute the drive-based reward using the selected drive type."""
        return self.drive(self._internal_states, self._outcome)
    

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        
        # Draw resources - each with a unique color
        colors = [(0, 255, 0), (0, 200, 100), (100, 255, 100), (100, 200, 0)]  # Different green shades
        for i, (loc, resource) in enumerate(self.resource_locations.items()):
            color_idx = i % len(colors)
            pygame.draw.rect(
                canvas,
                colors[color_idx],
                pygame.Rect(
                    pix_square_size * np.array(loc),
                    (pix_square_size, pix_square_size),
                ),
                width=3  # Just the outline
            )
            
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Draw internal states as a bar at the top
        bar_height = self.window_size / 20
        for i in range(self.internal_state_size):
            bar_width = self.window_size / self.internal_state_size
            bar_x = i * bar_width
            value = self._internal_states[i].item()
            
            # Draw background (empty) bar
            pygame.draw.rect(
                canvas,
                (200, 200, 200),
                pygame.Rect(bar_x, 0, bar_width, bar_height)
            )
            
            # Draw filled portion
            filled_width = bar_width * value
            color = (0, 150, 0) if value > 0.5 else (150, 150, 0)
            pygame.draw.rect(
                canvas,
                color,
                pygame.Rect(bar_x, 0, filled_width, bar_height)
            )
            
            # Draw setpoint marker
            setpoint = self.drive_params.get('setpoint', 0.5)
            setpoint_x = bar_x + bar_width * setpoint
            pygame.draw.line(
                canvas,
                (255, 0, 0),
                (setpoint_x, 0),
                (setpoint_x, bar_height),
                width=2
            )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
