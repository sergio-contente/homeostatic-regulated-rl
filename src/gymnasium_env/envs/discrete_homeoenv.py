import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

from ...utils.get_params import ParameterHandler


class HomeoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, config_path, drive_type, render_mode=None, maxh=10):
        # Window setup
        self.window_size = 500  # The size of the PyGame window

        # Drive setup
        self.parameter_manager = ParameterHandler(config_path)
        self.drive = self.parameter_manager.create_drive(drive_type)

        # Homeostatic Regulated Environment variables
        self._internal_state_size = self.drive.get_internal_state_size()
        self._outcome = 1
        self._internal_states = np.zeros(self._internal_state_size, dtype=np.float32)

        # Observations are dictionaries with the agent's internal states only
        self.size = maxh
        self.observation_space = spaces.Box(-self.size, self.size + 1, shape=(self._internal_state_size,), dtype=np.int32)

        # Define action space for homeostatic regulation
        self.action_space = spaces.Discrete(5)
        """
        0: Consume resource 0 
        1: Consume resource 1
        2: Not consume resource 0
        3: Not consume resource 0
        4: Do nothing
        """
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Optimal point:
        tensor = self.drive.get_tensor_optimal_states_values()
        self.optimal_point = tuple(tensor.tolist())

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        
       

    def _get_obs(self):
        return self._internal_states

    def _get_info(self):
        return {
            "drive": self.drive.compute_drive(self._internal_states),
            }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Initialize internal states with random values between 0 and 1
        # self._internal_states = np.random.choice(
				# 		range(-self.size, self.size+1), 
				# 		size=len(self.drive._optimal_internal_states)
				# ).astype(np.int32)        
        # Initial drive
        self._internal_states = np.zeros(
					len(self.drive._optimal_internal_states), 
					dtype=np.int32
						)

        initial_drive = self.drive.compute_drive(self._internal_states)
        self.drive.update_drive(initial_drive)
                
        # All resources available at start
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):        
        # Apply the chosen action to modify internal states
        if action == 0:  # Consume resource 0
            self._internal_states[0] = min(self._internal_states[0] + self._outcome, self.size)
        elif action == 1:  # Consume resource 1
            self._internal_states[1] = min(self._internal_states[1] + self._outcome, self.size)
        elif action == 2: # Do not consume resource 0
            self._internal_states[0] = max(self._internal_states[0] - self._outcome, -self.size)
        elif action == 3: # Do not consume resouce 1
            self._internal_states[1] = max(self._internal_states[1] - self._outcome, -self.size)

        # Updates drive and reward
        new_drive = self.drive.compute_drive(self._internal_states)
        reward = self.drive.compute_reward(new_drive)
        self.drive.update_drive(new_drive)
        
        # An episode is done if internal states are close to optimal
        # You might want to define a threshold for "close enough"
        threshold = self._outcome / 2 
        terminated = self.drive.has_reached_optimal(self._internal_states, threshold)
        
        # # Small reward for consuming resources (optional)
        # if resource_consumed:
        #     reward += 0.5
            
        # Big reward if reached optimal internal state
        if terminated:
            print("Achieved Homeostatic Point")

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        """Renders the current state of the environment using pygame."""
        if not hasattr(self, 'screen'):
            pygame.init()
            self.screen_width = 600
            self.screen_height = 600
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Clementine Environment")
            self.clock = pygame.time.Clock()
            
        # Clear the screen
        self.screen.fill((255, 255, 255))
        
        # Draw the current state
        center_x, center_y = self.screen_width // 2, self.screen_height // 2
        grid_size = min(self.screen_width, self.screen_height) - 100
        cell_size = grid_size // (2 * self.size + 1)
        
        # Draw the grid
        for i in range(-self.size, self.size + 1):
            for j in range(-self.size, self.size + 1):
                rect_x = center_x + i * cell_size - cell_size // 2
                rect_y = center_y + j * cell_size - cell_size // 2
                pygame.draw.rect(self.screen, (200, 200, 200), 
                                (rect_x, rect_y, cell_size, cell_size), 1)
                
                # Mark the optimal point
                if i == self.optimal_point[0] and j == self.optimal_point[1]:
                    pygame.draw.rect(self.screen, (0, 255, 0), 
                                    (rect_x, rect_y, cell_size, cell_size))
        
        # Draw the agent at the current position
        agent_x = center_x + self._internal_states[0] * cell_size - cell_size // 2
        agent_y = center_y + self._internal_states[1] * cell_size - cell_size // 2
        pygame.draw.circle(self.screen, (255, 0, 0), 
                        (agent_x + cell_size // 2, agent_y + cell_size // 2), 
                        cell_size // 2)
        
        # Update the display
        pygame.display.flip()
        self.clock.tick(30)
        pygame.event.pump()

    def plot_rewards(self, rewards):
        """Plot the training rewards curve."""
        import os
        import matplotlib.pyplot as plt

        # Garante que a pasta 'images/custom/homeoenv' exista
        os.makedirs('images/gym/homeoenv', exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.plot(rewards)
        plt.title('Rewards per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        
        # Add moving average
        window_size = min(10, len(rewards))
        if window_size > 0:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(rewards)), moving_avg, 'r', label=f'Moving Average ({window_size} episodes)')
        
        plt.legend()
        plt.savefig('images/gym/homeoenv/training_rewards.png')
        plt.show()

