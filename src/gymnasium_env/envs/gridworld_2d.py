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
        self.window_size = 512  # The size of the PyGame window
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
                "position": np.array(selected_positions[i], dtype=np.int64),
                "available": True
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

        self.set_agent_info(
                position=self.np_random.choice(list(self.area)),
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
        
        prob_regenerate = np.random.uniform(0, 1)
        self.agent_info["internal_states"] = states_after_cost
        # Regenerate resources if they are not available
        for resource in self.resources_info.values():
          resource_name = resource["name"]
          resource_regen_rate = self.drive.get_state_resources_regen_rate(resource_name)
          if not resource["available"] and prob_regenerate < resource_regen_rate:
            resource["available"] = True
        # Processa ações de movimento
        previous_position = self.agent_info["position"]
        new_position = self.agent_info["position"]
        if action == 0:  # stay
            new_position = previous_position
        elif action == 1:  # left
            new_position[0] = previous_position[0] - 1
        elif action == 2:  # right
            new_position[0] = previous_position[0] + 1
        elif action == 3:  # up
            new_position[1] = previous_position[1] - 1
        elif action == 4:  # down
            new_position[1] = previous_position[1] + 1
        
        # Convert NumPy array to tuple before checking if it's in the area set
        new_position_tuple = (int(new_position[0]), int(new_position[1]))
        if new_position_tuple not in self.area:
            new_position = previous_position
        
        self.agent_info["position"] = new_position
        
        # Inicializa new_internal_state para caso não ocorra consumo
        new_internal_state = np.array(self.agent_info["internal_states"])
        
        # Processa ações de consumo
        consumption_actions = self.dimension_internal_states
        for i in range(consumption_actions):
            if action == 5 + i:
                resource = self.resources_info[i]
                resource_position = resource["position"]
                
                # Verifica se o agente está na mesma posição do recurso
                resource_pos_tuple = (int(resource["position"][0]), int(resource["position"][1]))
                previous_position_tuple = (int(previous_position[0]), int(previous_position[1]))
                if previous_position_tuple == resource_pos_tuple:
                    # Prepara um vetor indicando qual recurso está sendo consumido
                    action_states = np.zeros(self.dimension_internal_states)
                    action_states[i] = 1.0
                    resource["available"] = False
                    # Aplica a ingestão
                    new_internal_state = self.drive.apply_intake(
                        self.agent_info["internal_states"],
                        action_states
                    )
        
        # Atualiza os estados internos após possível consumo
        self.agent_info["internal_states"] = new_internal_state
        
        # IMPORTANTE: Verifica se algum estado está abaixo de -1.0
        # Adicionando print para debug
        # print("Verificando estados internos:", self.agent_info["internal_states"])
        # print("Algum estado < -1.0?", np.any(self.agent_info["internal_states"] < -1.0))
        
        # Define done como True se qualquer estado for menor que -1.0
        done = False
        if np.any(self.agent_info["internal_states"] < -1.0):
            # print("EPISÓDIO TERMINADO: Um estado está abaixo de -1")
            done = True
        
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
							self.window = pygame.display.set_mode((self.window_size, self.window_size))
					if self.clock is None and self.render_mode == "human":
							self.clock = pygame.time.Clock()

					canvas = pygame.Surface((self.window_size, self.window_size))
					canvas.fill((240, 240, 240))  # Light gray background
					
					# Define layout areas
					grid_size = self.window_size * 2 // 3  # Grid takes up 2/3 of the window
					info_start_y = grid_size + 20  # Space for information below the grid
					
					# Calculate cell size
					pix_square_size = grid_size / self.size
					
					# Draw grid area with border
					pygame.draw.rect(
							canvas,
							(220, 220, 220),  # Light gray for grid area
							pygame.Rect(0, 0, grid_size, grid_size),
					)
					
					# Draw grid lines
					for x in range(self.size + 1):
							pygame.draw.line(
									canvas,
									(100, 100, 100),  # Dark gray for grid lines
									(pix_square_size * x, 0),
									(pix_square_size * x, grid_size),
									width=2 if x == 0 or x == self.size else 1,
							)
					
					for y in range(self.size + 1):
							pygame.draw.line(
									canvas,
									(100, 100, 100),
									(0, pix_square_size * y),
									(grid_size, pix_square_size * y),
									width=2 if y == 0 or y == self.size else 1,
							)
					
					# Draw resources on the grid
					for resource_id, resource in self.resources_info.items():
							# Colors for each resource type
							if resource["name"] == "food":
									resource_color = (220, 50, 50)  # Red for food
							elif resource["name"] == "water":
									resource_color = (50, 50, 220)  # Blue for water
							elif resource["name"] == "energy":
									resource_color = (220, 220, 50)  # Yellow for energy
							else:
									# Other colors for additional resources
									colors = [(50, 200, 50), (200, 50, 200), (50, 200, 200), (200, 150, 50)]
									resource_color = colors[resource_id % len(colors)]
							
							# Draw resource as a filled square with border
							if resource["available"]:
									resource_pos = resource["position"]
									pygame.draw.rect(
											canvas,
											resource_color,
											pygame.Rect(
													(resource_pos[0] * pix_square_size, resource_pos[1] * pix_square_size),
													(pix_square_size, pix_square_size),
											),
									)
									# Add black border
									pygame.draw.rect(
											canvas,
											(0, 0, 0),
											pygame.Rect(
													(resource_pos[0] * pix_square_size, resource_pos[1] * pix_square_size),
													(pix_square_size, pix_square_size),
											),
											width=2
									)
					
					# Draw the agent as a circle with border
					agent_pos = self.agent_info["position"]
					
					# Draw outer circle (black border)
					pygame.draw.circle(
							canvas,
							(0, 0, 0),  # Black border
							(agent_pos[0] * pix_square_size + pix_square_size / 2,
							agent_pos[1] * pix_square_size + pix_square_size / 2),
							pix_square_size / 2.5,
					)
					
					# Draw inner circle (blue fill)
					pygame.draw.circle(
							canvas,
							(50, 150, 255),  # Blue for agent
							(agent_pos[0] * pix_square_size + pix_square_size / 2,
							agent_pos[1] * pix_square_size + pix_square_size / 2),
							pix_square_size / 3,
					)
					
					# Initialize fonts
					if pygame.font:
							title_font = pygame.font.SysFont('Arial', 24, bold=True)
							label_font = pygame.font.SysFont('Arial', 18, bold=True)
							value_font = pygame.font.SysFont('Arial', 16)
					else:
							return
					
					# Display current drive value
					drive_title = title_font.render("Drive:", True, (0, 0, 0))
					drive_value_text = value_font.render(f"{self.current_drive:.4f}", True, (0, 0, 0))
					
					canvas.blit(drive_title, (grid_size + 20, 20))
					canvas.blit(drive_value_text, (grid_size + 100, 20))
					
					# Display internal state bars
					bar_height = 25
					bar_spacing = 35
					bar_width = self.window_size - grid_size - 60
					
					# Title for internal states
					states_title = title_font.render("Internal States:", True, (0, 0, 0))
					canvas.blit(states_title, (grid_size + 20, 60))
					
					# Draw each internal state bar
					for i in range(self.dimension_internal_states):
							y_pos = 100 + i * bar_spacing
							state_value = self.agent_info["internal_states"][i]
							
							# State name
							state_label = label_font.render(f"{self.state_names[i]}:", True, (0, 0, 0))
							canvas.blit(state_label, (grid_size + 20, y_pos))
							
							# Numerical value
							value_text = value_font.render(f"{state_value:.2f}", True, (0, 0, 0))
							canvas.blit(value_text, (grid_size + 20, y_pos + 20))
							
							# Draw background bar (gray)
							pygame.draw.rect(
									canvas,
									(200, 200, 200),
									pygame.Rect(grid_size + 100, y_pos, bar_width, bar_height),
							)
							
							# Normalize value to [0, 1]
							normalized_value = (state_value + 1) / 2
							filled_width = max(0, int(normalized_value * bar_width))
							
							# Bar color based on value
							if state_value > 0:
									# Green for positive values
									bar_color = (0, min(255, int(150 + state_value * 100)), 0)
							else:
									# Red for negative values
									bar_color = (min(255, int(150 - state_value * 100)), 0, 0)
							
							# Draw filled part of the bar
							pygame.draw.rect(
									canvas,
									bar_color,
									pygame.Rect(grid_size + 100, y_pos, filled_width, bar_height),
							)
							
							# Draw bar border
							pygame.draw.rect(
									canvas,
									(0, 0, 0),
									pygame.Rect(grid_size + 100, y_pos, bar_width, bar_height),
									width=1
							)
							
							# Draw middle marker
							mid_x = grid_size + 100 + bar_width / 2
							pygame.draw.line(
									canvas,
									(0, 0, 0),
									(mid_x, y_pos - 3),
									(mid_x, y_pos + bar_height + 3),
									width=1
							)
							
							# Labels for min, mid, max
							min_label = value_font.render("-1", True, (0, 0, 0))
							mid_label = value_font.render("0", True, (0, 0, 0))
							max_label = value_font.render("1", True, (0, 0, 0))
							
							canvas.blit(min_label, (grid_size + 95, y_pos + bar_height + 5))
							canvas.blit(mid_label, (mid_x - 5, y_pos + bar_height + 5))
							canvas.blit(max_label, (grid_size + 100 + bar_width - 10, y_pos + bar_height + 5))
					
					# Display resource legend
					legend_y = 100 + self.dimension_internal_states * bar_spacing + 40
					legend_title = title_font.render("Resources:", True, (0, 0, 0))
					canvas.blit(legend_title, (grid_size + 20, legend_y))
					
					# Show each resource in the legend
					for i, state_name in enumerate(self.state_names):
							legend_y_pos = legend_y + 30 + i * 25
							
							# Resource color
							if state_name == "food":
									resource_color = (220, 50, 50)
							elif state_name == "water":
									resource_color = (50, 50, 220)
							elif state_name == "energy":
									resource_color = (220, 220, 50)
							else:
									colors = [(50, 200, 50), (200, 50, 200), (50, 200, 200), (200, 150, 50)]
									resource_color = colors[i % len(colors)]
							
							# Draw color square
							pygame.draw.rect(
									canvas,
									resource_color,
									pygame.Rect(grid_size + 20, legend_y_pos, 15, 15),
							)
							pygame.draw.rect(
									canvas,
									(0, 0, 0),
									pygame.Rect(grid_size + 20, legend_y_pos, 15, 15),
									width=1
							)
							
							# Resource name
							name_text = value_font.render(state_name, True, (0, 0, 0))
							canvas.blit(name_text, (grid_size + 45, legend_y_pos))
							
							# Resource position
							pos_text = value_font.render(f"Position: ({self.resources_info[i]['position'][0]}, {self.resources_info[i]['position'][1]})", True, (0, 0, 0))
							canvas.blit(pos_text, (grid_size + 120, legend_y_pos))

					# Display controls
					controls_y = legend_y + 30 + self.dimension_internal_states * 25 + 20
					controls_title = title_font.render("Controls:", True, (0, 0, 0))
					canvas.blit(controls_title, (grid_size + 20, controls_y))
					
					controls = [
							"0: Stay",
							"1: Left",
							"2: Right",
							"3: Up",
							"4: Down"
					]
					
					for i, control in enumerate(controls):
							control_text = value_font.render(control, True, (0, 0, 0))
							canvas.blit(control_text, (grid_size + 20, controls_y + 30 + i * 20))

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
    env = LimitedResources2DEnv(render_mode="human", config_path=config_path, drive_type=drive_type, size=10)
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
