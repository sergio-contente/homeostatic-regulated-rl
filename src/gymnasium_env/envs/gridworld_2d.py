from enum import Enum

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

from src.utils.get_params import ParameterHandler
from src.utils.resource_manager import GlobalResourceManager


class LimitedResources2DEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config_path, drive_type, render_mode=None, size=5):
        self.param_manager = ParameterHandler(config_path)
        self.drive = self.param_manager.create_drive(drive_type)
        self.size = size  # The size of the square grid
        self.window_size = 600  # Increased window size for better visibility
        self.dimension_internal_states = self.drive.get_internal_state_dimension()
        
        # ✅ Create global resource manager for shared resource regeneration
        self.resource_manager = GlobalResourceManager(config_path, drive_type)


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
                ),
                "resources_map": spaces.MultiBinary([self.size, self.size, self.dimension_internal_states]) # For each cell, for each resource type
            }
        )
        self.observation_space = spaces.Dict(observation_dict)

        num_actions = 5 + self.dimension_internal_states  # left, right, up, down, stay + consume actions

        self.action_space = spaces.Discrete(num_actions)

        state_names = self.drive.get_internal_states_names()
        
        resource_rng = np.random.RandomState(123)

        # Total possible positions in the grid
        total_positions = self.size * self.size
        
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
        else:
            # If more resources than positions, allow replacement (though less ideal)
            all_positions = []
            for x in range(self.size):
                for y in range(self.size):
                    all_positions.append((x, y))
            sampled_indices = resource_rng.choice(
                len(all_positions),
                size=self.dimension_internal_states,
                replace=True 
            )
            selected_positions = [all_positions[i] for i in sampled_indices]


        self.resources_info = {}
        for i, state_name in enumerate(state_names):
            self.resources_info[i] = {
                "name": state_name,
                "position": np.array(selected_positions[i], dtype=np.int64),
                "available": True # Initially all resources are available
            }
        
        # Initialize resources_map: 1 if resource 'k' is at (x,y) and available, 0 otherwise.
        # This map is more about where resources of a certain type *could* be, and their current availability.
        # For simplicity, let's assume each resource type has one fixed location.
        # The "resources_map" in observation will indicate availability at these fixed locations.
        # The provided definition `spaces.MultiBinary([self.size, self.size, self.dimension_internal_states])`
        # suggests a map where each cell (x,y) can have a binary vector indicating which resources are present AND available.
        # Let's refine this: the observation map should indicate resource availability at their fixed locations.
        # The observation's `resources_map` will be a 1D array indicating availability of each resource type.

        # For observation, let's use a simpler 1D binary array for resource availability
        observation_dict["resources_map"] = spaces.MultiBinary(self.dimension_internal_states)
        self.observation_space = spaces.Dict(observation_dict) # Re-assign with corrected resources_map space
        
        # For internal tracking, `self.resources_info` holds positions and availability.
        # `self.agent_info["resources_map"]` will be the 1D availability array for the observation.
        initial_resources_availability = np.ones(self.dimension_internal_states, dtype=np.int8)


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
            "internal_states": np.zeros(self.dimension_internal_states, dtype=np.float64),
            "resources_map": initial_resources_availability.copy()
        }

        self.previous_internal_states = np.array(self.agent_info["internal_states"])

        initial_drive = self.drive.compute_drive(self.agent_info["internal_states"])
        self.drive.update_drive(initial_drive)

        self.reward_scale = 100.

    def set_agent_info(self, position, internal_states, resources_map):
        self.agent_info["position"] = position
        self.agent_info["internal_states"] = internal_states
        self.agent_info["resources_map"] = resources_map

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset resource availability
        current_resources_availability = np.ones(self.dimension_internal_states, dtype=np.int8)
        for i, resource in self.resources_info.items():
            resource["available"] = True
            # current_resources_availability[i] is already 1, so no change needed here

        # Escolha uma posição aleatória da área válida
        random_position_idx = self.np_random.integers(0, len(self.area))
        random_position_tuple = list(self.area)[random_position_idx]
        position_array = np.array(random_position_tuple, dtype=np.int64)
        
        self.set_agent_info(
                position=position_array,
                internal_states=self.np_random.uniform(
                    low=-0.3, 
                    high=0.3, 
                    size=(self.dimension_internal_states,)
                ),
                resources_map=current_resources_availability
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
            "internal_states": self.agent_info["internal_states"],
            "resources_map": self.agent_info["resources_map"]
        }
        return obs

    def _get_info(self):
        return self.agent_info.copy()

    def step(self, action):
        self.previous_internal_states = np.array(self.agent_info["internal_states"])

        states_after_cost = self.drive.apply_natural_decay(self.previous_internal_states)

        new_internal_state = np.array(states_after_cost)

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

        new_position_tuple = (int(new_position[0]), int(new_position[1]))
        if new_position_tuple not in self.area:
            new_position = previous_position.copy()

        # Process consumption actions
        # The first 5 actions are movement/stay: 0:stay, 1:left, 2:right, 3:up, 4:down
        # Consumption actions start from index 5
        for i in range(self.dimension_internal_states):
            if action == 5 + i: # Action to consume resource i
                resource_data = self.resources_info[i]
                # Check if agent is at the resource location and resource is available
                if np.array_equal(previous_position, resource_data["position"]) and resource_data["available"]:
                    action_states = np.zeros(self.dimension_internal_states)
                    action_states[i] = 1.0  # Indicate consumption of this specific resource
                    new_internal_state = self.drive.apply_intake(
                        states_after_cost, # Use states_after_cost as base for intake
                        action_states
                    )
                    resource_data["available"] = False # Mark resource as unavailable after consumption
        
        # ✅ Apply global resource regeneration (replaces per-agent regeneration)
        self.resources_info = self.resource_manager.apply_resource_regeneration(
            self.resources_info
        )
        
        # Update the resources_map for observation based on current availability
        current_resources_availability = np.array([
            int(self.resources_info[res_idx]["available"]) for res_idx in range(self.dimension_internal_states)
        ], dtype=np.int8)

        self.set_agent_info(new_position, new_internal_state, current_resources_availability)

        done = np.any(new_internal_state < -1.0)

        reward = self.get_reward()

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

        # Define areas of the layout
        grid_display_size = self.window_size * 2 // 3  # Grid takes up 2/3 of the window height and width
        info_start_y = grid_display_size + 20      # Y position for info text
        info_start_x = 20                          # X position for info text
        
        pix_square_size = grid_display_size / self.size

        # Draw grid area with background
        pygame.draw.rect(
            canvas,
            (220, 220, 220),  # Light gray for grid area
            pygame.Rect(0, 0, grid_display_size, grid_display_size),
        )

        # Draw grid cells
        for r in range(self.size):
            for c in range(self.size):
                rect = pygame.Rect(c * pix_square_size, r * pix_square_size, pix_square_size, pix_square_size)
                # Alternate cell colors for better visibility
                cell_color = (230, 230, 230) if (r + c) % 2 == 0 else (210, 210, 210)
                pygame.draw.rect(canvas, cell_color, rect)
                pygame.draw.rect(canvas, (100, 100, 100), rect, width=1) # Grid lines

        # Draw the resources
        for resource_id, resource_data in self.resources_info.items():
            res_pos = resource_data["position"] # This is [x,y] or (x,y)
            res_x_on_grid = res_pos[0] * pix_square_size
            res_y_on_grid = res_pos[1] * pix_square_size

            if resource_data["name"] == "food":
                resource_color = (220, 50, 50)  # Red
            elif resource_data["name"] == "water":
                resource_color = (50, 50, 220)  # Blue
            elif resource_data["name"] == "energy":
                resource_color = (220, 220, 50) # Yellow
            else:
                # Default colors for other resources
                colors = [(50, 200, 50), (200, 50, 200), (50, 200, 200), (200, 150, 50)]
                resource_color = colors[resource_id % len(colors)]
            
            resource_rect = pygame.Rect(res_x_on_grid, res_y_on_grid, pix_square_size, pix_square_size)

            if resource_data["available"]:
                pygame.draw.rect(canvas, resource_color, resource_rect) # Filled square
                pygame.draw.rect(canvas, (0,0,0), resource_rect, width=2) # Black border
                 # Optional: Add a small white inner square to indicate availability clearly
                icon_margin = pix_square_size * 0.25
                available_icon_rect = pygame.Rect(
                    res_x_on_grid + icon_margin,
                    res_y_on_grid + icon_margin,
                    pix_square_size - 2 * icon_margin,
                    pix_square_size - 2 * icon_margin
                )
                pygame.draw.rect(canvas, (255, 255, 255), available_icon_rect, width=0) # Small white square

            else: # Not available
                pygame.draw.rect(canvas, (200,200,200), resource_rect) # Grayed out
                pygame.draw.rect(canvas, (100,100,100), resource_rect, width=2) # Darker gray border
                # Draw an X for unavailable
                pygame.draw.line(canvas, (100,100,100), 
                                 (res_x_on_grid + pix_square_size * 0.2, res_y_on_grid + pix_square_size * 0.2),
                                 (res_x_on_grid + pix_square_size * 0.8, res_y_on_grid + pix_square_size * 0.8),
                                 width=3)
                pygame.draw.line(canvas, (100,100,100), 
                                 (res_x_on_grid + pix_square_size * 0.2, res_y_on_grid + pix_square_size * 0.8),
                                 (res_x_on_grid + pix_square_size * 0.8, res_y_on_grid + pix_square_size * 0.2),
                                 width=3)

        # Draw the agent
        agent_pos = self.agent_info["position"] # This is [x,y] or (x,y)
        agent_center_x = agent_pos[0] * pix_square_size + pix_square_size / 2
        agent_center_y = agent_pos[1] * pix_square_size + pix_square_size / 2
        agent_radius = pix_square_size / 3

        pygame.draw.circle(canvas, (0, 0, 0), (agent_center_x, agent_center_y), agent_radius + 2) # Black border
        pygame.draw.circle(canvas, (50, 150, 255), (agent_center_x, agent_center_y), agent_radius) # Blue agent
        
        # Initialize font
        if pygame.font:
            title_font = pygame.font.SysFont('Arial', 24, bold=True)
            label_font = pygame.font.SysFont('Arial', 18, bold=True)
            value_font = pygame.font.SysFont('Arial', 16)
        else:
            # Fallback if no font available
            return # Or handle differently

        # --- Displaying Information --- 
        current_y_offset = info_start_y

        # Drive Value
        drive_value = self.drive.get_current_drive()
        drive_text = title_font.render(f"Drive: {drive_value:.4f}", True, (0,0,0))
        canvas.blit(drive_text, (info_start_x, current_y_offset))
        current_y_offset += 30

        # Agent Position
        pos_text = label_font.render(f"Position: ({self.agent_info['position'][0]}, {self.agent_info['position'][1]})", True, (0,0,0))
        canvas.blit(pos_text, (info_start_x, current_y_offset))
        current_y_offset += 25

        # Internal States Bars
        state_names = self.drive.get_internal_states_names()
        bar_max_width = self.window_size - info_start_x - 150 # Max width for bars
        bar_height = 20
        bar_spacing = 35

        for i in range(self.dimension_internal_states):
            state_name = state_names[i]
            state_value = self.agent_info["internal_states"][i]
            
            # State Name
            name_surf = label_font.render(f"{state_name}:", True, (0,0,0))
            canvas.blit(name_surf, (info_start_x, current_y_offset))
            
            # State Value Text
            value_surf = value_font.render(f"{state_value:.2f}", True, (0,0,0))
            canvas.blit(value_surf, (info_start_x + 80, current_y_offset))

            # Bar position
            bar_x = info_start_x + 150
            bar_y = current_y_offset

            # Background of the bar
            pygame.draw.rect(canvas, (200,200,200), (bar_x, bar_y, bar_max_width, bar_height))
            
            # Foreground of the bar
            normalized_value = (state_value + 1) / 2 # Normalize from [-1,1] to [0,1]
            filled_width = max(0, min(normalized_value * bar_max_width, bar_max_width))
            
            if state_value > 0:
                bar_color = (0, min(255, int(150 + state_value * 100)), 0) # Greenish
            else:
                bar_color = (min(255, int(150 - state_value * 100)), 0, 0) # Reddish
            pygame.draw.rect(canvas, bar_color, (bar_x, bar_y, filled_width, bar_height))
            pygame.draw.rect(canvas, (0,0,0), (bar_x, bar_y, bar_max_width, bar_height), 1) # Border

            # Mid-line marker
            mid_line_x = bar_x + bar_max_width / 2
            pygame.draw.line(canvas, (0,0,0), (mid_line_x, bar_y), (mid_line_x, bar_y + bar_height), 1)
            
            current_y_offset += bar_spacing
        
        current_y_offset += 10 # spacing before legend
        
        # Resource Legend
        legend_title_surf = label_font.render("Resources:", True, (0,0,0))
        canvas.blit(legend_title_surf, (info_start_x, current_y_offset))
        current_y_offset += 25
        legend_item_x = info_start_x
        legend_item_y_start = current_y_offset
        max_legend_width_per_col = (self.window_size - info_start_x * 2) // 2 # two columns for legend
        col_width = 200 # fixed width for each legend item column

        for i, (res_id, res_data) in enumerate(self.resources_info.items()):
            res_name = res_data["name"]
            res_pos_str = f"({res_data['position'][0]},{res_data['position'][1]})"
            status_str = "Available" if res_data["available"] else "Used"

            if res_data["name"] == "food": color = (220,50,50)
            elif res_data["name"] == "water": color = (50,50,220)
            elif res_data["name"] == "energy": color = (220,220,50)
            else: 
                cs = [(50,200,50), (200,50,200), (50,200,200), (200,150,50)]
                color = cs[res_id % len(cs)]
            
            # Determine position for legend item (simple horizontal list for now)
            current_legend_x = legend_item_x + (i % 2) * col_width
            current_legend_y = legend_item_y_start + (i // 2) * 25

            pygame.draw.rect(canvas, color if res_data["available"] else (180,180,180), (current_legend_x, current_legend_y, 15, 15))
            pygame.draw.rect(canvas, (0,0,0), (current_legend_x, current_legend_y, 15, 15), 1) # border
            
            text_surf = value_font.render(f"{res_name} {res_pos_str} - {status_str}", True, (0,0,0))
            canvas.blit(text_surf, (current_legend_x + 20, current_legend_y))

            if current_legend_x + col_width > self.window_size - info_start_x: # next row if no space
                 legend_item_x = info_start_x
                 current_y_offset +=25
            else:
                 legend_item_x += col_width
        
        # Adjust current_y_offset based on how many rows the legend took
        num_legend_rows = (len(self.resources_info) + 1) // 2 # if 2 items per row
        current_y_offset = legend_item_y_start + num_legend_rows * 25 + 10

        # Actions Legend
        actions_title_surf = label_font.render("Actions:", True, (0,0,0))
        canvas.blit(actions_title_surf, (info_start_x, current_y_offset))
        current_y_offset += 25
        
        action_texts = [
            "0: Stay", "1: Left", "2: Right", "3: Up", "4: Down"
        ]
        for i, state_name in enumerate(state_names):
            action_texts.append(f"{5+i}: Consume {state_name}")

        action_col1_x = info_start_x
        action_col2_x = info_start_x + (self.window_size - info_start_x*2) // 2
        action_y = current_y_offset
        
        for i, text in enumerate(action_texts):
            action_surf = value_font.render(text, True, (0,0,0))
            if i < (len(action_texts) +1) // 2: # First half in col 1
                 canvas.blit(action_surf, (action_col1_x, action_y + (i % ((len(action_texts)+1)//2)) * 20))
            else: # Second half in col 2
                 canvas.blit(action_surf, (action_col2_x, action_y + (i % ((len(action_texts)+1)//2)) * 20))

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
