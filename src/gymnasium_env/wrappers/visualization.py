import gymnasium as gym
import numpy as np
import pygame
import torch

class VisualizationWrapper(gym.Wrapper):
    """
    Wrapper to visualize internal states and drive values alongside the GridWorld.
    This adds a sidebar to the standard GridWorld rendering with internal state and drive information.
    """
    
    def __init__(self, env, drive_names=None):
        """
        Initialize the visualization wrapper.
        
        Args:
            env: The environment to wrap
            drive_names: Optional list of names for each drive dimension
        """
        super().__init__(env)
        self.drive_names = drive_names
        
        # Extend the window size to add the sidebar
        self.sidebar_width = 200
        self.original_window_size = self.env.window_size if hasattr(self.env, 'window_size') else 512
        self.window_size = (self.original_window_size + self.sidebar_width, self.original_window_size)
        
        # Sidebar colors and fonts
        self.bg_color = (240, 240, 240)
        self.text_color = (0, 0, 0)
        self.bar_colors = [
            (0, 0, 255),    # Blue
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (255, 165, 0),  # Orange
            (128, 0, 128)   # Purple
        ]
        
        self.font = None
        self.last_info = None
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_info = info
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_info = info
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """
        Render the environment with the sidebar containing internal state and drive information.
        """
        # Get the base render from the original environment
        base_render = self.env.render()
        
        if self.env.render_mode == "human":
            # If we're in human mode, we need to update the window
            self._render_sidebar()
            return None
        elif self.env.render_mode == "rgb_array":
            # In rgb_array mode, we need to create a new array with the sidebar
            sidebar = self._create_sidebar_array()
            # Combine the base render and sidebar
            full_render = np.zeros((self.original_window_size, self.original_window_size + self.sidebar_width, 3), dtype=np.uint8)
            full_render[:, :self.original_window_size, :] = base_render
            full_render[:, self.original_window_size:, :] = sidebar
            return full_render
    
    def _render_sidebar(self):
        """
        Render the sidebar with internal state and drive information in human mode.
        """
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            self.font = pygame.font.Font(None, 24)
        
        # Create sidebar surface
        sidebar = pygame.Surface((self.sidebar_width, self.original_window_size))
        sidebar.fill(self.bg_color)
        
        # Draw sidebar content
        self._draw_sidebar_content(sidebar)
        
        # Blit sidebar to window
        self.window.blit(sidebar, (self.original_window_size, 0))
        pygame.display.update()
    
    def _create_sidebar_array(self):
        """
        Create a numpy array for the sidebar in rgb_array mode.
        """
        # Create a pygame surface for the sidebar
        sidebar = pygame.Surface((self.sidebar_width, self.original_window_size))
        sidebar.fill(self.bg_color)
        
        # Draw sidebar content
        self._draw_sidebar_content(sidebar)
        
        # Convert to numpy array
        sidebar_array = np.transpose(np.array(pygame.surfarray.pixels3d(sidebar)), axes=(1, 0, 2))
        return sidebar_array
    
    def _draw_sidebar_content(self, surface):
        """
        Draw the content of the sidebar (internal states, drive values, etc.)
        """
        if self.font is None:
            pygame.font.init()
            self.font = pygame.font.Font(None, 24)
        
        y_offset = 20
        
        # Draw title
        title = self.font.render("Internal States", True, self.text_color)
        surface.blit(title, (10, y_offset))
        y_offset += 30
        
        # Draw internal states if available
        if self.last_info and 'internal_states' in self.last_info:
            internal_states = self.last_info['internal_states']
            
            for i, state in enumerate(internal_states):
                # Get name if available
                name = f"State {i+1}" if self.drive_names is None else self.drive_names[i]
                
                # Draw name
                state_text = self.font.render(f"{name}: {state:.2f}", True, self.text_color)
                surface.blit(state_text, (10, y_offset))
                y_offset += 20
                
                # Draw bar
                bar_width = int(state * 180)
                pygame.draw.rect(surface, self.bar_colors[i % len(self.bar_colors)], 
                                 (10, y_offset, bar_width, 15))
                pygame.draw.rect(surface, self.text_color, 
                                 (10, y_offset, 180, 15), 1)  # Border
                y_offset += 25
        
        y_offset += 10
        
        # Draw drive value if available
        if self.last_info and 'drive_value' in self.last_info:
            drive_title = self.font.render("Drive Value", True, self.text_color)
            surface.blit(drive_title, (10, y_offset))
            y_offset += 30
            
            drive_value = self.last_info['drive_value']
            drive_type = self.last_info.get('drive_type', 'Unknown')
            
            # Draw drive type
            type_text = self.font.render(f"Type: {drive_type}", True, self.text_color)
            surface.blit(type_text, (10, y_offset))
            y_offset += 25
            
            # Draw drive value
            value_text = self.font.render(f"Value: {drive_value:.4f}", True, self.text_color)
            surface.blit(value_text, (10, y_offset))
            y_offset += 25
        
        # Draw reward if available
        if self.last_info and 'reward' in self.last_info:
            reward = self.last_info['reward']
            reward_text = self.font.render(f"Reward: {reward:.4f}", True, self.text_color)
            surface.blit(reward_text, (10, y_offset))
            y_offset += 25
