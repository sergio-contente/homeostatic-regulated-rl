import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from src.utils.get_params import ParameterHandler


class NormarlHomeostaticEnv(gym.Env):
    """
    Single-agent homeostatic environment with social norms.
    Integrates NORMARL's social cost mechanism with homeostatic drives.
    Q (consumption in NORMARL) = K (intake in homeostatic system)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config_path, drive_type, learning_rate, beta, render_mode=None, size=10):
        self.param_manager = ParameterHandler(config_path)
        self.drive_type = drive_type
        self.size = size  # The size of the 1D grid
        self.window_size = 800
        
        # Initialize drive
        self.drive = self.param_manager.create_drive(drive_type)
        self.dimension_internal_states = self.drive.get_internal_state_dimension()
        
        # NORMARL parameters
        self.beta = beta  # Norm internalization strength
        self.alpha = learning_rate  # Learning rate
        self.gamma = 0.1  # Softmax temperature
        
        # Social cost parameters
        self.a = 1.0  # Base social cost
        self.b = 0.5  # Resource scarcity multiplier
        
        # Belief about average consumption (Q̄ in NORMARL)
        self.perceived_social_norm = np.zeros(self.dimension_internal_states)
        
        # Resource stock (E in NORMARL)
        self.initial_resource_stock = np.ones(self.dimension_internal_states) * 2
        self.resource_stock = self.initial_resource_stock.copy()
        self.resource_regeneration_rate = self.drive.get_array_resources_regeneration_rate()
        
        # Action space
        num_actions = 3 + self.dimension_internal_states
        self.action_space = spaces.Discrete(num_actions)
        
        # Observation space
        observation_dict = spaces.Dict({
            "position": spaces.Discrete(self.size),
            "internal_states": spaces.Box(
                low=-1.0, high=1.0, 
                shape=(self.dimension_internal_states,), 
                dtype=np.float64
            ),
            # "resources_map": spaces.MultiBinary(self.dimension_internal_states),
            "perceived_social_norm": spaces.Box(
                low=0, high=1.0,
                shape=(self.dimension_internal_states,),
                dtype=np.float64
            )
        })
        self.observation_space = spaces.Dict(observation_dict)
        
        # Initialize resources
        self._initialization_resources()
        self.agent_info = {
            "position": 0,
            "internal_states": np.zeros(self.dimension_internal_states, dtype=np.float64),
            # "resources_map": np.zeros(self.dimension_internal_states, dtype=np.int8),
            "perceived_social_norm": np.zeros(self.dimension_internal_states, dtype=np.float64)
        }

        self.intake = np.zeros(self.dimension_internal_states, dtype=np.float64)
        
        # Tracking for rendering
        self.intake_history = []
        self.resource_history = [self.resource_stock.copy()]
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.area = set(range(self.size))
        
        # Initialize drive
        initial_drive = self.drive.compute_drive(self.agent_info["internal_states"])
        self.drive.update_drive(initial_drive)

    def _initialization_resources(self):
        """Initialize resource positions on the grid"""
        state_names = self.drive.get_internal_states_names()
        resource_rng = np.random.RandomState(123)
        
        if self.dimension_internal_states <= self.size:
            random_positions = resource_rng.choice(
                self.size, size=self.dimension_internal_states, replace=False
            )
        else:
            random_positions = resource_rng.choice(
                self.size, size=self.dimension_internal_states, replace=True
            )
        
        self.resources_info = {}
        for i, state_name in enumerate(state_names):
            self.resources_info[i] = {
                "name": state_name,
                "position": random_positions[i]
                # "available": True
            }

    def compute_social_cost(self, intake):
        """
        Compute social cost based on NORMARL equation (2)
        Si(Qi) = 0 if Qi < Q̄i
        Si(Qi) = βi * (Qi - Q̄i) * max{0, (a - b*E)} if Qi ≥ Q̄i
        
        Where Q = K (intake)
        """
        belief = self.perceived_social_norm
        beta = self.beta
        
        # Resource scarcity factor
        scarcity_factor = np.maximum(0, self.a - self.b * self.resource_stock)

        print("Scarcity factor: ", scarcity_factor)
        print("Resource stock: ", self.resource_stock)
        print("Intake: ", intake)
        print("Belief: ", belief)
        
        
        # Social cost for each resource type
        social_cost = 0
        for i in range(self.dimension_internal_states):
            if intake[i] > belief[i]:
                social_cost += beta * (intake[i] - belief[i]) * scarcity_factor[i]
        
        return social_cost

    def update_belief(self, observed_avg_intake):
        """
        Update belief about average consumption using equation (5)
        Q̄i(t+1) = (1 - αi) * Q̄i(t) + αi * q̄(t+1)
        """
        self.perceived_social_norm = (
            (1 - self.alpha) * self.perceived_social_norm + 
            self.alpha * observed_avg_intake
        )

    def update_resource_stock(self, total_intake):
        """
        Update resource stock using equation (1)
        E(t+1) = (1 + δ) * E(t) - Σ(Qi)
        Where Q = K (total intake)
        """
        self.resource_stock = (
            (1 + self.resource_regeneration_rate) * self.resource_stock - 
            10 * total_intake
        )
        self.resource_stock = np.maximum(0, self.resource_stock)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset resource stock
        self.resource_stock = self.initial_resource_stock.copy()
        
        # # Reset resources
        # for resource in self.resources_info.values():
        #     resource["available"] = True
        
        # Reset agent
        self.agent_info["position"] = self.np_random.choice(list(range(self.size)))
        self.agent_info["internal_states"] = self.np_random.uniform(
            low=-0.3, high=0.3, 
            size=(self.dimension_internal_states,)
        )
        
        # Reset drive
        initial_drive = self.drive.compute_drive(self.agent_info["internal_states"])
        self.drive.update_drive(initial_drive)
        
        # Reset belief
        self.perceived_social_norm = np.zeros(self.dimension_internal_states)
        
        # Reset history
        self.intake_history = []
        self.resource_history = [self.resource_stock.copy()]
        
        if self.render_mode == "human":
            self._render_frame()
        
        return self._get_obs(), self._get_info()

    def _get_obs(self):
        """Get observation for the agent"""
        # Resources map
        # resources_map = np.array([
        #     int(resource["available"]) for resource in self.resources_info.values()
        # ], dtype=np.int8)
        
        return {
            "position": self.agent_info["position"],
            "internal_states": self.agent_info["internal_states"],
            # "resources_map": resources_map,
            "perceived_social_norm": self.perceived_social_norm
        }

    def _get_info(self):
        """Get info dictionary with agent's state"""
        return {
            "agent_info": self.agent_info.copy(),
            "last_intake": self.intake.copy()
        }
    def step(self, action, avg_intake):
        """
        Single-agent RL step
        
        Args:
            action: Action for the agent
            
        Returns:
            observation: Observation for the agent
            reward: Reward for the agent
            done: Done flag
            truncated: Truncated flag
            info: Info dict
        """
        if action is None:
            raise ValueError("Action must be provided!")
        
        # 1. Apply natural decay first
        states_after_decay = self.drive.apply_natural_decay(self.agent_info["internal_states"])

        # for resource in self.resources_info.values():
        #     resource["available"] = self.drive.apply_resource_regeneration(
        #         resource["available"],
        #         resource["name"]
        #     )
        # 2. Process MOVEMENT action
        current_position = self.agent_info["position"]
        new_position = current_position
        if action == 0:  # stay
            new_position = current_position
        elif action == 1:  # left
            new_position = max(0, current_position - 1)
        elif action == 2:  # right
            new_position = min(self.size - 1, current_position + 1)
        else:
            # Consumption actions don't change position
            new_position = current_position
        
         # Verifica limites
        if new_position not in self.area:
            new_position = current_position

        new_internal_state = states_after_decay.copy()

        # Processamento de ações de consumo
        intake_resources = np.zeros(self.dimension_internal_states)
        for i in range(self.dimension_internal_states):
            if action == 3 + i:
                resource = self.resources_info[i]
                # if current_position == resource["position"] and resource["available"]:
                if current_position == resource["position"]:
                    intake_resources[i] = 1.0
                    # resource["available"] = False

        new_internal_state = self.drive.apply_intake(states_after_decay, intake_resources)
        self.intake = self.drive.get_intake_array(intake_resources)
        
        # resources_map = np.array([
        #     int(resource["available"]) for resource in self.resources_info.values()
        # ], dtype=np.int8)
        
        # 4. Update agent state
        self.agent_info["position"] = new_position
        self.agent_info["internal_states"] = new_internal_state
        # self.agent_info["resources_map"] = resources_map
        
        # 5. Calculate reward and update drive
        reward = self._compute_reward(self.intake) * 100
        
        # 7. Update global state
        self._update_global_environment(self.intake,avg_intake)
        
        # 8. Check termination
        done = self._check_termination()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return self._get_obs(), reward, done, False, self._get_info()

    def _compute_reward(self, intake):
        """Compute reward combining homeostatic and social components"""
        # Homeostatic reward (drive reduction)
        new_drive = self.drive.compute_drive(self.agent_info["internal_states"]) # get new drive
        homeostatic_reward = self.drive.compute_reward(new_drive) #get reward
        self.drive.update_drive(new_drive) # update drive
        
        # Social cost
        social_cost = self.compute_social_cost(intake)

        print(f"Homeostatic reward: {homeostatic_reward}, Social cost: {social_cost}")
        
        # Total reward
        return homeostatic_reward - social_cost * 10

    def _update_global_environment(self, intake, avg_intake):
        """Update global environment state"""
        # Update resource stock
        self.update_resource_stock(intake)
        
        # Update history
        self.intake_history.append(intake)
        self.resource_history.append(self.resource_stock.copy())
        
        # Calculate actual average across all agents
        self.update_belief(avg_intake)

    def _check_termination(self):
        """Check if episode should terminate"""
        # Check if resource stock is depleted
        if np.any(self.resource_stock <= 0):
            print("Resource stock depleted")
            return True
        
        # Check if agent's internal states are too extreme
        if np.any(np.abs(self.agent_info["internal_states"]) > 1):
            print("Internal states too extreme")
            return True
        
        return False

    def _render_frame(self):
        """Render the single-agent environment"""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((240, 240, 240))
        
        # Layout
        grid_height = 150
        info_height = 300
        graph_height = self.window_size - grid_height - info_height - 50
        
        pix_square_size = self.window_size / self.size
        
        # Draw grid area
        pygame.draw.rect(canvas, (220, 220, 220), pygame.Rect(0, 0, self.window_size, grid_height))
        
        # Draw resources
        state_names = self.drive.get_internal_states_names()
        for resource_id, resource in self.resources_info.items():
            if state_names[resource_id] == "food":
                resource_color = (220, 50, 50)
            elif state_names[resource_id] == "water":
                resource_color = (50, 50, 220)
            elif state_names[resource_id] == "energy":
                resource_color = (220, 220, 50)
            else:
                colors = [(50, 200, 50), (200, 50, 200), (50, 200, 200)]
                resource_color = colors[resource_id % len(colors)]
            
            x = resource["position"] * pix_square_size
            
            # Always draw resources as available
            pygame.draw.rect(canvas, resource_color, 
                           pygame.Rect(x, 50, pix_square_size, 50))
            pygame.draw.rect(canvas, (0, 0, 0), 
                           pygame.Rect(x, 50, pix_square_size, 50), width=2)
            
            # # Draw based on availability
            # if resource["available"]:
            #     pygame.draw.rect(canvas, resource_color, 
            #                    pygame.Rect(x, 50, pix_square_size, 50))
            #     pygame.draw.rect(canvas, (0, 0, 0), 
            #                    pygame.Rect(x, 50, pix_square_size, 50), width=2)
            # else:
            #     pygame.draw.rect(canvas, (200, 200, 200), 
            #                    pygame.Rect(x, 50, pix_square_size, 50))
            #     pygame.draw.rect(canvas, (100, 100, 100), 
            #                    pygame.Rect(x, 50, pix_square_size, 50), width=1)
        
        # Draw agent
        agent_color = (50, 150, 255)
        x = self.agent_info["position"] * pix_square_size + pix_square_size / 2
        y = 75
        
        # Draw agent with norm internalization indicator
        radius = 12 + int(self.beta * 8)  # Larger circle for higher beta
        pygame.draw.circle(canvas, (0, 0, 0), (int(x), int(y)), radius + 2)
        pygame.draw.circle(canvas, agent_color, (int(x), int(y)), radius)
        
        # Draw grid lines
        for x in range(self.size + 1):
            pygame.draw.line(canvas, (100, 100, 100), 
                           (x * pix_square_size, 0), 
                           (x * pix_square_size, grid_height), width=1)
        
        # Font setup
        if pygame.font:
            title_font = pygame.font.SysFont('Arial', 16, bold=True)
            label_font = pygame.font.SysFont('Arial', 12)
            value_font = pygame.font.SysFont('Arial', 10)
        else:
            return
        
        # Agent information
        info_y = grid_height + 20
        
        # Resource stock
        avg_stock = float(np.mean(self.resource_stock))
        stock_color = (0, 150, 0) if avg_stock > 500 else (150, 150, 0) if avg_stock > 200 else (150, 0, 0)
        stock_text = title_font.render(f"Avg Resource Stock: {avg_stock:.1f}", True, stock_color)
        canvas.blit(stock_text, (20, info_y))
        
        # Total consumption this step
        if len(self.intake_history) > 0:
            total_intake = float(np.sum(self.intake_history[-1]))
            total_text = label_font.render(f"Total intake/step: {total_intake:.1f}", True, (0, 0, 0))
            canvas.blit(total_text, (300, info_y))
        
        # Agent parameters
        agent_text = label_font.render(f"Agent (β={self.beta:.2f}, α={self.alpha:.2f})", True, (0, 0, 0))
        canvas.blit(agent_text, (20, info_y + 30))
        
        # Belief about average consumption
        belief_text = value_font.render(f"Belief: {float(self.perceived_social_norm.mean()):.2f}", True, (100, 100, 100))
        canvas.blit(belief_text, (20, info_y + 50))
        
        # Internal states
        for j, (state_name, state_value) in enumerate(zip(state_names, self.agent_info["internal_states"])):
            y_pos = info_y + 70 + j * 15
            
            # Mini bar
            bar_width = 80
            bar_height = 10
            
            pygame.draw.rect(canvas, (200, 200, 200), 
                           pygame.Rect(20, y_pos, bar_width, bar_height))
            
            normalized_value = (state_value + 1) / 2
            filled_width = int(normalized_value * bar_width)
            
            if state_value > 0:
                bar_color = (0, 200, 0)
            else:
                bar_color = (200, 0, 0)
            
            pygame.draw.rect(canvas, bar_color, 
                           pygame.Rect(20, y_pos, filled_width, bar_height))
            pygame.draw.rect(canvas, (0, 0, 0), 
                           pygame.Rect(20, y_pos, bar_width, bar_height), width=1)
            
            # Label with consumption indicator
            consumed = " *" if len(self.intake_history) > 0 and self.intake_history[-1][j] > 0 else ""
            state_text = value_font.render(f"{state_name}: {float(state_value):.2f}{consumed}", True, (0, 0, 0))
            canvas.blit(state_text, (20 + bar_width + 5, y_pos))
        
        # Resource history graph
        graph_y = self.window_size - graph_height - 20
        graph_width = self.window_size - 40
        
        # Graph background
        pygame.draw.rect(canvas, (250, 250, 250), 
                       pygame.Rect(20, graph_y, graph_width, graph_height - 20))
        pygame.draw.rect(canvas, (0, 0, 0), 
                       pygame.Rect(20, graph_y, graph_width, graph_height - 20), width=1)
        
        # Graph title
        graph_title = label_font.render("Resource Stock Over Time", True, (0, 0, 0))
        canvas.blit(graph_title, (self.window_size // 2 - 80, graph_y - 20))
        
        # Plot resource history
        if len(self.resource_history) > 1:
            max_resource = float(max(np.mean(self.initial_resource_stock), 
                                   max(np.mean(stock) for stock in self.resource_history)))
            
            points = []
            for i, stock in enumerate(self.resource_history[-200:]):  # Last 200 points
                x = 20 + (i / min(200, len(self.resource_history))) * graph_width
                y = graph_y + graph_height - 20 - (float(np.mean(stock)) / max_resource) * (graph_height - 40)
                points.append((int(x), int(y)))
            
            if len(points) > 1:
                pygame.draw.lines(canvas, (0, 0, 255), False, points, 2)
            
            # Draw sustainability threshold
            threshold_y = int(graph_y + graph_height - 20 - (200 / max_resource) * (graph_height - 40))
            pygame.draw.line(canvas, (255, 0, 0), 
                           (20, threshold_y), 
                           (20 + int(graph_width), threshold_y), 1)
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == '__main__':
    config_path = "config/config.yaml"
    drive_type = "base_drive"
    learning_rate = 0.02
    n_agents = 2
    n_trials = 1000

    env = NormarlHomeostaticEnv(
        config_path=config_path,
        drive_type=drive_type,
        learning_rate=learning_rate,
        size=1,
        render_mode="human"
    )

    env.beta = 0.8
    env.alpha = learning_rate

    obs, info = env.reset()

    agents_intake_history = [[] for _ in range(n_agents)]
    avg_intake = np.zeros(env.dimension_internal_states)

    print(f"Starting simulation with {n_agents} agents in {n_trials} trials.\n")

    for trial in range(n_trials):
        trial_intakes = []

        for current_agent in range(n_agents):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action, avg_intake)

            agent_intake = info["last_intake"]
            agents_intake_history[current_agent].append(agent_intake.copy())
            trial_intakes.append(agent_intake.copy())

            print(f"[Trial {trial}] Agent {current_agent} intake: {agent_intake}")

            if done:
                print(f"\nEpisode ended at trial {trial}, agent {current_agent}")
                env.close()
                exit()

        # Atualiza a média após todos os agentes agirem
        avg_intake = np.mean(trial_intakes, axis=0)

        print(f"[Trial {trial}] Average intake: {avg_intake}")
        print(f"[Trial {trial}] Perceived norm: {env.perceived_social_norm}\n")

    env.close()
    print("Simulation finished.")
