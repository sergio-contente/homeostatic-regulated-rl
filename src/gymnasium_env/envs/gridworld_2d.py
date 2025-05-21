from enum import Enum

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

from src.utils.get_params import ParameterHandler


class LimitedResources2DEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config_path, drive_type, render_mode=None, size=5):
        self.param_manager = ParameterHandler(config_path)
        self.drive = self.param_manager.create_drive(drive_type)
        self.size = size  # The size of the square grid
        self.window_size = 600  # Increased window size for better visibility
        self.dimension_internal_states = self.drive.get_internal_state_dimension()


        # Observations are dictionaries with the agent's and the target's location.
        observation_dict = spaces.Dict(
            {
                "position": spaces.Box(
                    low=  0,
                    high= self.size - 1,
                    shape=(2,),
                    dtype=np.int64
                ),
                                        
                "internal_states": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.dimension_internal_states,),
                    dtype=np.float64,
                )
            }
        )
        self.observation_space = spaces.Dict(observation_dict)

        num_actions = 5 + self.dimension_internal_states  # left, right, up, down, stay + consume actions

        self.action_space = spaces.Discrete(num_actions)

        state_names = self.drive.get_internal_states_names()
        
        resource_rng = np.random.RandomState(123)

        # Total possible positions in the grid
        total_positions = self.size * self.size
        
        # Choose unique positions if there are enough grid cells
        if self.dimension_internal_states <= total_positions:
            # Generate all possible positions
            all_positions = []
            for x in range(self.size):
                for y in range(self.size):
                    all_positions.append((x, y))
                    
            # Randomly select positions without replacement
            sampled_indices = resource_rng.choice(
                len(all_positions), 
                size=self.dimension_internal_states, 
                replace=False
            )
            selected_positions = [all_positions[i] for i in sampled_indices]

        self.resources_info = {}
        for i, state_name in enumerate(state_names):
            self.resources_info[i] = {
                "name": state_name,
                "position": np.array(selected_positions[i], dtype=np.int64)
            }

        # Define the valid area where the agent can move (all grid positions)
        self.area = set()
        for x in range(size):
          for y in range(size):
            self.area.add((x, y))

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        self.agent_info = {
            "position": np.zeros(2, dtype=np.int64),
            "internal_states": np.zeros(self.dimension_internal_states, dtype=np.float64)
        }

        self.previous_internal_states = np.array(self.agent_info["internal_states"])

        initial_drive = self.drive.compute_drive(self.agent_info["internal_states"])
        self.drive.update_drive(initial_drive)

        self.reward_scale = 100.

    def set_agent_info(self, position, internal_states):
        self.agent_info["position"] = position
        self.agent_info["internal_states"] = internal_states

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Escolha uma posição aleatória da área válida
        random_position = list(self.area)[self.np_random.integers(0, len(self.area))]
        position_array = np.array(random_position, dtype=np.int64)
        
        self.set_agent_info(
                position=position_array,
                internal_states=self.np_random.uniform(
                    low=-0.3, 
                    high=0.3, 
                    size=(self.dimension_internal_states,)
                )
            )

        self.previous_internal_states = np.array(self.agent_info["internal_states"])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        initial_drive = self.drive.compute_drive(self.agent_info["internal_states"])
        self.drive.update_drive(initial_drive)

        return observation, info

    def _get_obs(self):
        obs = {
            "position": self.agent_info["position"],
            "internal_states": self.agent_info["internal_states"]
        }
        return obs

    def _get_info(self):
        return self.agent_info.copy()

    def step(self, action):
        # Salva o estado anterior
        self.previous_internal_states = np.array(self.agent_info["internal_states"])
        
        # Aplica o decay natural aos estados internos
        states_after_cost = self.drive.apply_natural_decay(self.previous_internal_states)

        # Inicializa new_internal_state com os valores após decay
        new_internal_state = np.array(states_after_cost)

        # Inicializa nova posição com cópia segura
        previous_position = self.agent_info["position"]
        new_position = previous_position.copy()

        # Processa ações de movimento
        if action == 0:  # stay
            pass
        elif action == 1:  # left
            new_position[0] = max(0, previous_position[0] - 1)
        elif action == 2:  # right
            new_position[0] = min(self.size - 1, previous_position[0] + 1)
        elif action == 3:  # up
            new_position[1] = min(self.size - 1, previous_position[1] + 1)
        elif action == 4:  # down
            new_position[1] = max(0, previous_position[1] - 1)

        # Verifica se nova posição é válida
        new_position_tuple = (int(new_position[0]), int(new_position[1]))
        if new_position_tuple not in self.area:
            new_position = previous_position.copy()

        # Processa ações de consumo
        consumption_actions = self.dimension_internal_states
        for i in range(consumption_actions):
            if action == 5 + i:
                resource = self.resources_info[i]
                resource_pos_tuple = tuple(resource["position"])
                if tuple(previous_position) == resource_pos_tuple:
                    action_states = np.zeros(self.dimension_internal_states)
                    action_states[i] = 1.0
                    new_internal_state = self.drive.apply_intake(
                        new_internal_state,
                        action_states
                    )

        # Usa função centralizada para atualizar o estado do agente
        self.set_agent_info(new_position, new_internal_state)

        # Verifica término do episódio
        done = np.any(new_internal_state < -1.0)

        # Calcula recompensa
        reward = self.get_reward()

        # Prepara retorno
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, done, False, info


    def get_reward(self):
        new_drive = self.drive.compute_drive(self.agent_info["internal_states"]) # get new drive
        reward = self.drive.compute_reward(new_drive) #get reward
        self.drive.update_drive(new_drive) # update drive
        return reward * self.reward_scale
    
    def close(self):
      if self.window is not None:
        pygame.display.quit()
        pygame.quit()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            # Configura tamanho da janela adequado para todos os elementos
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
            pygame.display.set_caption("Limited Resources 2D Environment")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((240, 240, 240))  # Light gray background

        # Define areas of the layout - grid ocupando metade da janela
        grid_size = self.window_size * 1 // 2  # Grid takes up 1/2 of the window
        info_start_y = grid_size + 20  # Space for information below the grid
        pix_square_size = grid_size / self.size

        # Draw grid area with background
        pygame.draw.rect(
            canvas,
            (220, 220, 220),  # Light gray for grid area
            pygame.Rect(0, 0, grid_size, grid_size),
        )

        # Draw grid cells
        for x in range(self.size):
            for y in range(self.size):
                rect = pygame.Rect(x * pix_square_size, y * pix_square_size, pix_square_size, pix_square_size)
                # Alternate cell colors for better visibility
                cell_color = (230, 230, 230) if (x + y) % 2 == 0 else (210, 210, 210)
                pygame.draw.rect(canvas, cell_color, rect)
                pygame.draw.rect(canvas, (100, 100, 100), rect, width=1)

        # Initialize resource colors and map
        resource_colors = {}
        for i, resource in self.resources_info.items():
            name = resource["name"]
            if name == "food":
                resource_colors[i] = (220, 50, 50)  # Red for food
            elif name == "water":
                resource_colors[i] = (50, 50, 220)  # Blue for water
            elif name == "energy":
                resource_colors[i] = (220, 220, 50)  # Yellow for energy
            else:
                # Other colors for additional resources
                other_colors = [(50, 200, 50), (200, 50, 200), (50, 200, 200), (200, 150, 50)]
                resource_colors[i] = other_colors[i % len(other_colors)]

        # Draw resources
        for i, resource in self.resources_info.items():
            x, y = resource["position"]
            resource_color = resource_colors[i]
            
            # Draw the resource as a filled square with border
            pygame.draw.rect(
                canvas,
                resource_color,
                pygame.Rect(x * pix_square_size, y * pix_square_size, pix_square_size, pix_square_size),
            )
            
            # Add resource border
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(x * pix_square_size, y * pix_square_size, pix_square_size, pix_square_size),
                width=2
            )

        # Draw agent
        x, y = self.agent_info["position"]
        center = (int(x * pix_square_size + pix_square_size / 2), int(y * pix_square_size + pix_square_size / 2))
        
        # Draw outer circle (black border)
        pygame.draw.circle(canvas, (0, 0, 0), center, int(pix_square_size / 2.5))
        # Draw inner circle (blue fill)
        pygame.draw.circle(canvas, (50, 150, 255), center, int(pix_square_size / 3))
        
        # Draw grid lines
        for i in range(self.size + 1):
            # Vertical lines
            pygame.draw.line(
                canvas,
                (100, 100, 100),
                (pix_square_size * i, 0),
                (pix_square_size * i, grid_size),
                width=2,
            )
            # Horizontal lines
            pygame.draw.line(
                canvas,
                (100, 100, 100),
                (0, pix_square_size * i),
                (grid_size, pix_square_size * i),
                width=2,
            )
        
        # Initialize fonts
        if pygame.font:
            title_font = pygame.font.SysFont('Arial', 24, bold=True)
            label_font = pygame.font.SysFont('Arial', 18, bold=True)
            value_font = pygame.font.SysFont('Arial', 16)
        else:
            # Fallback if no font available
            return
        
        # Display current drive value
        drive_value = self.drive.get_current_drive()
        drive_title = title_font.render("Drive:", True, (0, 0, 0))
        drive_value_text = value_font.render(f"{drive_value:.4f}", True, (0, 0, 0))
        
        canvas.blit(drive_title, (20, grid_size + 20))
        canvas.blit(drive_value_text, (120, grid_size + 20))
        
        # Display agent position
        agent_pos_x, agent_pos_y = self.agent_info["position"]
        pos_text = label_font.render(f"Position: ({agent_pos_x}, {agent_pos_y})", True, (0, 0, 0))
        canvas.blit(pos_text, (250, grid_size + 20))
        
        # Display internal state bars
        state_names = self.drive.get_internal_states_names()
        bar_height = 25
        bar_spacing = 35
        bar_width = self.window_size - 150  # Fixed width for all bars
        
        # Draw resource legend (add this before drawing bars)
        legend_y = grid_size + 50
        legend_title = label_font.render("Resources Legend:", True, (0, 0, 0))
        canvas.blit(legend_title, (20, legend_y))
        
        # Display each resource in the legend with their colors
        legend_items_per_row = 3
        for i, state_name in enumerate(state_names):
            row = i // legend_items_per_row
            col = i % legend_items_per_row
            legend_x = 20 + col * 150
            legend_item_y = legend_y + 25 + row * 30
            
            # Get resource color
            resource_color = resource_colors[i]
            
            # Draw resource color square
            pygame.draw.rect(
                canvas,
                resource_color,
                pygame.Rect(legend_x, legend_item_y, 20, 20),
            )
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(legend_x, legend_item_y, 20, 20),
                width=1
            )
            
            # Resource name and position
            resource = self.resources_info[i]
            res_x, res_y = resource["position"]
            name_text = value_font.render(f"{state_name} ({res_x}, {res_y})", True, (0, 0, 0))
            canvas.blit(name_text, (legend_x + 25, legend_item_y + 2))

        # Start drawing bars below the legend
        bars_start_y = legend_y + 25 + ((len(state_names) - 1) // legend_items_per_row + 1) * 30 + 20
        
        # Draw each internal state bar
        for i in range(self.dimension_internal_states):
            y_pos = bars_start_y + i * bar_spacing
            state_value = self.agent_info["internal_states"][i]
            state_name = state_names[i]
            
            # State title with color indicator
            state_label = label_font.render(f"{state_name}:", True, (0, 0, 0))
            canvas.blit(state_label, (20, y_pos))
            
            # Colored indicator square
            pygame.draw.rect(
                canvas,
                resource_colors[i],  # Use same color as resource
                pygame.Rect(110, y_pos + 2, 15, 15),
            )
            
            # Numerical value
            value_text = value_font.render(f"{state_value:.2f}", True, (0, 0, 0))
            canvas.blit(value_text, (130, y_pos))
            
            # Draw bar background (gray)
            pygame.draw.rect(
                canvas,
                (200, 200, 200),
                pygame.Rect(210, y_pos, bar_width, bar_height),
            )
            
            # Normalize value to [0, 1]
            normalized_value = (state_value + 1) / 2
            filled_width = max(0, int(normalized_value * bar_width))
            
            # Determine bar color based on value
            if state_value > 0:
                # Stronger green for positive values
                bar_color = (0, min(255, int(150 + state_value * 100)), 0)
            else:
                # Stronger red for negative values
                bar_color = (min(255, int(150 - state_value * 100)), 0, 0)
            
            # Draw filled portion of the bar
            pygame.draw.rect(
                canvas,
                bar_color,
                pygame.Rect(210, y_pos, filled_width, bar_height),
            )
            
            # Bar border
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(210, y_pos, bar_width, bar_height),
                width=2
            )
            
            # Bar markings
            mid_x = 210 + bar_width / 2
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (mid_x, y_pos - 5),
                (mid_x, y_pos + bar_height + 5),
                width=2
            )
            
            # Min, mid, and max labels
            min_label = value_font.render("-1", True, (0, 0, 0))
            mid_label = value_font.render("0", True, (0, 0, 0))
            max_label = value_font.render("1", True, (0, 0, 0))
            
            canvas.blit(min_label, (205, y_pos + bar_height + 5))
            canvas.blit(mid_label, (mid_x - 5, y_pos + bar_height + 5))
            canvas.blit(max_label, (210 + bar_width - 10, y_pos + bar_height + 5))
            
            # Show difference from previous state
            diff = state_value - self.previous_internal_states[i]
            if abs(diff) > 0.001:  # Only show significant changes
                diff_text = f"{diff:+.2f}"  # Format with sign
                if diff > 0:
                    diff_color = (0, 150, 0)  # Green for increase
                else:
                    diff_color = (150, 0, 0)  # Red for decrease
                
                diff_label = value_font.render(diff_text, True, diff_color)
                canvas.blit(diff_label, (220 + bar_width + 10, y_pos + 5))

        # Display actions info in a more compact format
        actions_y = bars_start_y + self.dimension_internal_states * bar_spacing + 30
        actions_title = label_font.render("Actions:", True, (0, 0, 0))
        canvas.blit(actions_title, (20, actions_y))
        
        # List available actions more compactly
        move_actions = "Move: 0=Stay, 1=Left, 2=Right, 3=Up, 4=Down"
        move_text = value_font.render(move_actions, True, (0, 0, 0))
        canvas.blit(move_text, (20, actions_y + 25))
        
        # Consumption actions
        consume_text = value_font.render("Consume:", True, (0, 0, 0))
        canvas.blit(consume_text, (20, actions_y + 45))
        
        # List consumption actions with colors
        for i, state_name in enumerate(state_names):
            action_y = actions_y + 45 + (i + 1) * 20
            
            # Color indicator
            pygame.draw.rect(
                canvas,
                resource_colors[i],
                pygame.Rect(40, action_y, 15, 15),
            )
            
            # Action text
            action_text = value_font.render(f"{5 + i}: {state_name}", True, (0, 0, 0))
            canvas.blit(action_text, (60, action_y))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
if __name__ == '__main__':
    config_path = "config/config.yaml"
    drive_type = "base_drive"  # base_drive, elliptic_drive, etc.
    env = LimitedResources2DEnv(render_mode="human", config_path=config_path, drive_type=drive_type)
    env.reset()

    for i in range(1000):
        actions = env.action_space.sample()
        print(actions)
        obs, reward, done, truncate, info = env.step(actions)
        env.render()
        print("obs:", obs)
        print(info)

        if done:
            print("Episódio terminado, resetando ambiente...")
            obs, info = env.reset()
            print("Novo episódio iniciado. Estados internos:", obs["internal_states"])

    env.close()
    print("finish.")
