import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

from ...utils.get_params import ParameterHandler


class ClementineEnvironment(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, config_path, drive_type, render_mode=None, size=200):
        # Window setup
        self.window_size = 500  # The size of the PyGame window
        self.size = size

        # Drive setup
        self.parameter_manager = ParameterHandler(config_path)
        self.drive = self.parameter_manager.create_drive(drive_type)

        # Homeostatic Regulated Environment variables
        self._internal_state_size = self.drive.get_internal_state_size()
        self._outcome = 10
        self._internal_states = np.zeros(self._internal_state_size, dtype=np.float32)

        # Observations are dictionaries with the agent's internal states only
        self.observation_space = spaces.Box(-size, size + 1, shape=(self._internal_state_size,), dtype=np.int32)

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

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        
        # Colors for rendering
        self.colors = {
            "background": (240, 240, 240),
            "text": (0, 0, 0),
            "resource_available": (0, 200, 0),
            "resource_unavailable": (200, 0, 0),
            "internal_state_0": (0, 100, 255),
            "internal_state_1": (100, 0, 255),
            "optimal_marker": (255, 215, 0),  # Gold color for optimal level marker
        }

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
# Modificar para garantir que _internal_states tenha o mesmo tamanho que self.drive._optimal_internal_states
        self._internal_states = np.random.choice(
						range(-self.size, self.size+1), 
						size=len(self.drive._optimal_internal_states)
				).astype(np.int32)        
        # Initial drive
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
        elif action == 2:
            self._internal_states[0] = max(self._internal_states[0] - self._outcome, -self.size)
        elif action == 3:
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
        try:
            if self.window is None and self.render_mode == "human":
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size + 100))
                pygame.font.init()
                self.font = pygame.font.SysFont("Arial", 20)
                self.title_font = pygame.font.SysFont("Arial", 24, bold=True)
            if self.clock is None and self.render_mode == "human":
                self.clock = pygame.time.Clock()

            canvas = pygame.Surface((self.window_size, self.window_size + 100))
            canvas.fill(self.colors["background"])

            # Title
            title = self.title_font.render("Homeostatic Regulation Environment", True, (50, 50, 50))
            canvas.blit(title, (self.window_size//2 - title.get_width()//2, 10))

            padding = 40
            state_height = 40
            spacing = 80
            top = 60
            bar_width = self.window_size - 2 * padding
            max_value = 300.0  # Max value for visualization normalization

            # Draw internal states as bars
            for i in range(self._internal_state_size):
                y = top + i * spacing

                # Background bar
                pygame.draw.rect(canvas, (200, 200, 200), (padding, y, bar_width, state_height), border_radius=5)

                # Filled bar
                value = self._internal_states[i]
                filled_width = int(bar_width * min(value / max_value, 1.0))  # Normalize for visualization
                color = self.colors.get(f"internal_state_{i}", (0, 150, 150))
                pygame.draw.rect(canvas, color, (padding, y, filled_width, state_height), border_radius=5)

                # Optimal level marker
                try:
                    optimal = self.drive._optimal_internal_states
                    opt_val = optimal[i] if isinstance(optimal, (list, np.ndarray)) else optimal.get(i, None)
                    if opt_val is not None:
                        x = padding + int(bar_width * min(opt_val / max_value, 1.0))
                        pygame.draw.line(canvas, self.colors["optimal_marker"], (x, y), (x, y + state_height), width=3)
                        # Add text indicating optimal value
                        opt_label = self.font.render(f"Optimal: {opt_val:.0f}", True, (0, 0, 0))
                        canvas.blit(opt_label, (x + 5, y - 25))
                except Exception as e:
                    print(f"Could not draw optimal marker for state {i}: {e}")

                # Label with absolute value and percentage
                percentage = ((value + self.size) / (2 * self.size)) * 100
                label = self.font.render(f"State {i}: {value:.1f} ({percentage:.1f}%)", True, self.colors["text"])
                canvas.blit(label, (padding, y - 25))

            # Draw resources section
            resource_top = top + self._internal_state_size * spacing + 40
            resource_section = self.title_font.render("Available Resources", True, (50, 50, 50))
            canvas.blit(resource_section, (padding, resource_top - 30))

            # Positioning for resource icons and text
            resource_spacing = 200  # More horizontal space between resources
            
            for i in range(self._internal_state_size):
                x = padding + i * resource_spacing
                y = resource_top + 20  # Add some vertical padding

                color = self.colors["resource_available"]
                pygame.draw.circle(canvas, color, (x + 40, y), 30)

                status = "Available"
                label = self.font.render(f"Resource {i}: {status}", True, self.colors["text"])
                canvas.blit(label, (x, y + 40))

            # Information section
            info_top = resource_top + 120  # Adjust vertical position to avoid overlap
            info_section = self.title_font.render("Environment Information", True, (50, 50, 50))
            canvas.blit(info_section, (padding, info_top - 30))

            # Drive information
            drive_val = self.drive.compute_drive(self._internal_states)
            drive_label = self.font.render(f"Current Drive: {drive_val:.3f}", True, self.colors["text"])
            canvas.blit(drive_label, (padding, info_top))

            outcome_info = self.font.render(f"Intake Volume (K): {self._outcome}", True, self.colors["text"])
            canvas.blit(outcome_info, (padding, info_top + 50))

            # Final guidance text at the bottom
            guidelines = self.font.render("Actions: 0=+Res0, 1=+Res1, 2=-Res0, 3=-Res1, 4=Wait", True, (100, 100, 100))
            canvas.blit(guidelines, (padding, self.window_size - 50))

            if self.render_mode == "human":
                self.window.blit(canvas, canvas.get_rect())
                pygame.event.pump()
                pygame.display.update()
                self.clock.tick(self.metadata["render_fps"])
            else:
                return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

        except Exception as e:
            import traceback
            print(f"Render error: {e}")
            print(traceback.format_exc())
            return None
