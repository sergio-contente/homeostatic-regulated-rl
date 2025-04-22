from src.agents.q_learning import QLearning
from ..utils.get_params import ParameterHandler
import pygame
import numpy as np
import matplotlib.pyplot as plt
import time

class Clementine:
    def __init__(self, config_path, drive_type, render_mode=None, maxh=5):
        self.state_space = [(i, j) for i in range(maxh + 1) for j in range(maxh + 1)]
        self.action_space = [0, 1, 2, 3, 4]
        self.maxh = maxh
        self.steps = 0
        self.render_mode = render_mode

        self._outcome = 1

        self.agent = QLearning(
            state_size=len(self.state_space),
            action_size=len(self.action_space)
        )

        self.param_manager = ParameterHandler(config_path)
        self.drive = self.param_manager.create_drive(drive_type)

        # Optimal point:
        tensor = self.drive.get_tensor_optimal_states_values()
        self.optimal_point = tuple(tensor.tolist())

        self.size = self.drive.get_internal_state_size()
        self.current_state = np.random.choice(range(-maxh, maxh + 1), size=self.size)
        initial_drive = self.drive.compute_drive(self.current_state)
        self.drive.update_drive(initial_drive)

    def reset(self):
        self.current_state = np.random.choice(range(-self.maxh, self.maxh + 1), size=self.size)
        initial_drive = self.drive.compute_drive(self.current_state)
        self.drive.update_drive(initial_drive)
        self.steps = 0
        return self.current_state, {}
    
    def step(self, action):        
            # Apply the chosen action to modify internal states
            if action == 0:  # Consume resource 0
                    self.current_state[0] = min(self.current_state[0] + self._outcome, self.maxh)
            elif action == 1:  # Consume resource 1
                    self.current_state[1] = min(self.current_state[1] + self._outcome, self.maxh)
            elif action == 2:
                    self.current_state[0] = max(self.current_state[0] - self._outcome, -self.maxh)
            elif action == 3:
                    self.current_state[1] = max(self.current_state[1] - self._outcome, -self.maxh)

            # Updates drive and reward
            new_drive = self.drive.compute_drive(self.current_state)
            reward = self.drive.compute_reward(new_drive)
            self.drive.update_drive(new_drive)
            
            # An episode is done if internal states are close to optimal
            # You might want to define a threshold for "close enough"
            threshold = self._outcome / 2 
            terminated = self.drive.has_reached_optimal(self.current_state, threshold)
            
            # Big reward if reached optimal internal state
            if terminated:
                    print("Achieved Homeostatic Point")

            observation = self.current_state

            if self.render_mode == "human":
                    self._render_frame()

            self.steps += 1
            truncated = self.steps >= 1000  # Limit the number of steps per episode
            
            return observation, reward, terminated, truncated, {}
    
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
        cell_size = grid_size // (2 * self.maxh + 1)
        
        # Draw the grid
        for i in range(-self.maxh, self.maxh + 1):
            for j in range(-self.maxh, self.maxh + 1):
                rect_x = center_x + i * cell_size - cell_size // 2
                rect_y = center_y + j * cell_size - cell_size // 2
                pygame.draw.rect(self.screen, (200, 200, 200), 
                                 (rect_x, rect_y, cell_size, cell_size), 1)
                
                # Mark the optimal point
                if i == self.optimal_point[0] and j == self.optimal_point[1]:
                    pygame.draw.rect(self.screen, (0, 255, 0), 
                                     (rect_x, rect_y, cell_size, cell_size))
        
        # Draw the agent at the current position
        agent_x = center_x + self.current_state[0] * cell_size - cell_size // 2
        agent_y = center_y + self.current_state[1] * cell_size - cell_size // 2
        pygame.draw.circle(self.screen, (255, 0, 0), 
                          (agent_x + cell_size // 2, agent_y + cell_size // 2), 
                          cell_size // 2)
        
        # Update the display
        pygame.display.flip()
        self.clock.tick(30)
        pygame.event.pump()


    def plot_rewards(self, rewards):
        """Plot the training rewards curve."""
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
        plt.savefig('training_rewards.png')
        plt.show()
