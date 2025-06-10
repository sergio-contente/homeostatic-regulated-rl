import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from src.utils.get_params import ParameterHandler


class MultiAgentHomeostaticEnv(gym.Env):
    """
    Multi-agent homeostatic environment with social norms.
    Integrates NORMARL's social cost mechanism with homeostatic drives.
    Q (consumption in NORMARL) = K (intake in homeostatic system)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config_path, drive_type, n_agents=5, render_mode=None, size=10):
        self.param_manager = ParameterHandler(config_path)
        self.drive_type = drive_type
        self.n_agents = n_agents
        self.size = size  # The size of the 1D grid
        self.window_size = 800  # Increased for multi-agent display
        
        # Initialize drives for each agent
        self.drives = []
        for _ in range(n_agents):
            self.drives.append(self.param_manager.create_drive(drive_type))
        
        self.dimension_internal_states = self.drives[0].get_internal_state_dimension()
        
        # NORMARL parameters
        self.beta = np.ones(n_agents) * 0.8  # Norm internalization strength
        self.alpha = np.ones(n_agents) * 0.05  # Learning rate
        self.gamma = 0.1  # Softmax temperature
        
        # Social cost parameters
        self.a = 5.0  # Base social cost
        self.b = 0.005  # Resource scarcity multiplier
        
        # Belief about average consumption (Q̄ in NORMARL)
        # This represents what each agent believes others consume on average
        self.belief_avg_consumption = np.zeros((n_agents, self.dimension_internal_states))
        
        # Resource stock (E in NORMARL)
        self.initial_resource_stock = 1000.0
        self.resource_stock = self.initial_resource_stock
        self.resource_regeneration_rate = 0.02  # δ in NORMARL
        
        # Action space remains the same for each agent
        num_actions = 3 + self.dimension_internal_states
        self.action_space = spaces.Discrete(num_actions)
        
        # Observation space now includes social information
        observation_dict = spaces.Dict({
            "position": spaces.Discrete(self.size),
            "internal_states": spaces.Box(
                low=-1.0, high=1.0, 
                shape=(self.dimension_internal_states,), 
                dtype=np.float64
            ),
            "resources_map": spaces.MultiBinary(self.dimension_internal_states),
            "resource_stock": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float64),
            "belief_avg_consumption": spaces.Box(
                low=0, high=1.0,
                shape=(self.dimension_internal_states,),
                dtype=np.float64
            )
        })
        self.observation_space = spaces.Dict(observation_dict)
        
        # Initialize resources
        self._initialize_resources()
        
        # Agent information for all agents
        self.agents_info = []
        for i in range(n_agents):
            self.agents_info.append({
                "id": i,
                "position": 0,
                "internal_states": np.zeros(self.dimension_internal_states, dtype=np.float64),
                "last_intake": np.zeros(self.dimension_internal_states)  # Q = K
            })
        
        # Tracking for rendering
        self.intake_history = []  # History of total consumption/intake
        self.resource_history = [self.resource_stock]
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # Initialize drives
        for i, agent in enumerate(self.agents_info):
            initial_drive = self.drives[i].compute_drive(agent["internal_states"])
            self.drives[i].update_drive(initial_drive)

    def _initialize_resources(self):
        """Initialize resource positions on the grid"""
        state_names = self.drives[0].get_internal_states_names()
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
                "position": random_positions[i],
                "available": True
            }

    def compute_social_cost(self, agent_id, intake):
        """
        Compute social cost based on NORMARL equation (2)
        Si(Qi) = 0 if Qi < Q̄i
        Si(Qi) = βi * (Qi - Q̄i) * max{0, (a - b*E)} if Qi ≥ Q̄i
        
        Where Q = K (intake)
        """
        belief = self.belief_avg_consumption[agent_id]
        beta = self.beta[agent_id]
        
        # Resource scarcity factor
        scarcity_factor = max(0, self.a - self.b * self.resource_stock)
        
        # Social cost for each resource type
        social_cost = 0
        for i in range(self.dimension_internal_states):
            if intake[i] > belief[i]:
                social_cost += beta * (intake[i] - belief[i]) * scarcity_factor
        
        return social_cost

    def get_action_value(self, agent_id, states_after_decay, action):
        """
        Compute action value Vi(Qi) = Ui(Qi) - Ci(Qi) - Si(Qi)
        Where Q = K (intake vector)
        """
        agent = self.agents_info[agent_id]
        
        # Determine potential intake from this action
        potential_intake = np.zeros(self.dimension_internal_states)
        
        # Check if it's a consumption action
        if action >= 3:
            resource_idx = action - 3
            resource = self.resources_info[resource_idx]
            
            # Can only consume if at resource location and it's available
            if agent["position"] == resource["position"] and resource["available"]:
                potential_intake[resource_idx] = 1.0
        
        # Compute utility: homeostatic improvement from intake
        if np.any(potential_intake > 0):
            potential_state = self.drives[agent_id].apply_intake(states_after_decay, potential_intake)
            potential_drive = self.drives[agent_id].compute_drive(potential_state)
            current_drive = self.drives[agent_id].compute_drive(states_after_decay)
            utility = current_drive - potential_drive  # Reduction in drive is good
        else:
            utility = 0  # No consumption = no utility change
        
        # Individual cost: simply the amount consumed
        individual_cost = np.sum(potential_intake)
        
        # Social cost: based on exceeding social norms
        social_cost = self.compute_social_cost(agent_id, potential_intake)
        
        # Total value
        value = utility - individual_cost - social_cost
        
        return value, potential_intake

    def select_action(self, agent_id, states_after_decay):
        """
        Select action using softmax based on action values
        """
        # Compute values for all possible actions
        action_values = []
        
        for action in range(self.action_space.n):
            value, _ = self.get_action_value(agent_id, states_after_decay, action)
            action_values.append(value)
        
        # Softmax selection
        action_values = np.array(action_values)
        exp_values = np.exp(action_values / self.gamma)
        probabilities = exp_values / np.sum(exp_values)
        
        # Select action
        action = np.random.choice(self.action_space.n, p=probabilities)
        
        return action

    def update_belief(self, agent_id, observed_avg_intake):
        """
        Update belief about average consumption using equation (5)
        Q̄i(t+1) = (1 - αi) * Q̄i(t) + αi * q̄(t+1)
        """
        alpha = self.alpha[agent_id]
        self.belief_avg_consumption[agent_id] = (
            (1 - alpha) * self.belief_avg_consumption[agent_id] + 
            alpha * observed_avg_intake
        )

    def update_resource_stock(self, total_intake):
        """
        Update resource stock using equation (1)
        E(t+1) = (1 + δ) * E(t) - Σ(Qi)
        Where Q = K (total intake)
        """
        self.resource_stock = (
            (1 + self.resource_regeneration_rate) * self.resource_stock - 
            np.sum(total_intake)
        )
        self.resource_stock = max(0, self.resource_stock)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset resource stock
        self.resource_stock = self.initial_resource_stock
        
        # Reset resources
        for resource in self.resources_info.values():
            resource["available"] = True
        
        # Reset agents
        for i, agent in enumerate(self.agents_info):
            agent["position"] = self.np_random.choice(list(range(self.size)))
            agent["internal_states"] = self.np_random.uniform(
                low=-0.3, high=0.3, 
                size=(self.dimension_internal_states,)
            )
            agent["last_intake"] = np.zeros(self.dimension_internal_states)
            
            # Reset drive
            initial_drive = self.drives[i].compute_drive(agent["internal_states"])
            self.drives[i].update_drive(initial_drive)
        
        # Reset beliefs
        self.belief_avg_consumption = np.zeros((self.n_agents, self.dimension_internal_states))
        
        # Reset history
        self.intake_history = []
        self.resource_history = [self.resource_stock]
        
        if self.render_mode == "human":
            self._render_frame()
        
        # Return observation for first agent (for single-agent interface compatibility)
        return self._get_obs(0), self._get_info()

    def _get_obs(self, agent_id):
        """Get observation for a specific agent"""
        agent = self.agents_info[agent_id]
        
        # Resources map
        resources_map = np.array([
            int(resource["available"]) for resource in self.resources_info.values()
        ], dtype=np.int8)
        
        return {
            "position": agent["position"],
            "internal_states": agent["internal_states"],
            "resources_map": resources_map,
            "resource_stock": np.array([self.resource_stock]),
            "belief_avg_consumption": self.belief_avg_consumption[agent_id]
        }

    def _get_info(self):
        """Get info dictionary with all agents' states"""
        return {
            "agents": self.agents_info.copy(),
            "resource_stock": self.resource_stock,
            "beliefs": self.belief_avg_consumption.copy()
        }

    def step(self, action=None):
        """
        Multi-agent step: if action provided, use for agent 0, otherwise all agents act
        """
        total_intake = np.zeros(self.dimension_internal_states)
        all_rewards = []
        
        # Store previous states for rendering
        prev_internal_states = [agent["internal_states"].copy() for agent in self.agents_info]
        
        # Each agent takes an action
        for i, agent in enumerate(self.agents_info):
            # Apply natural decay
            states_after_decay = self.drives[i].apply_natural_decay(agent["internal_states"])
            
            # Select action (use provided action for agent 0 if given)
            if action is not None and i == 0:
                selected_action = action
            else:
                selected_action = self.select_action(i, states_after_decay)
            
            # Process movement
            new_position = agent["position"]
            if selected_action == 1:  # left
                new_position = max(0, new_position - 1)
            elif selected_action == 2:  # right
                new_position = min(self.size - 1, new_position + 1)
            
            # Process consumption
            new_internal_state = states_after_decay.copy()
            actual_intake = np.zeros(self.dimension_internal_states)
            
            if selected_action >= 3:
                resource_idx = selected_action - 3
                resource = self.resources_info[resource_idx]
                
                if new_position == resource["position"] and resource["available"]:
                    # Apply intake (Q = K)
                    actual_intake[resource_idx] = 1.0
                    new_internal_state = self.drives[i].apply_intake(states_after_decay, actual_intake)
                    resource["available"] = False
            
            # Update agent state
            agent["position"] = new_position
            agent["internal_states"] = new_internal_state
            agent["last_intake"] = actual_intake
            total_intake += actual_intake
            
            # Compute reward (homeostatic reward minus social cost)
            new_drive = self.drives[i].compute_drive(new_internal_state)
            homeostatic_reward = self.drives[i].compute_reward(new_drive)
            social_cost = self.compute_social_cost(i, actual_intake)
            reward = homeostatic_reward * 100 - social_cost
            self.drives[i].update_drive(new_drive)
            all_rewards.append(reward)
        
        # Update resource stock based on total intake
        self.update_resource_stock(total_intake)
        self.resource_history.append(self.resource_stock)
        
        # Calculate average intake and update beliefs
        avg_intake = total_intake / self.n_agents
        for i in range(self.n_agents):
            self.update_belief(i, avg_intake)
        
        # Regenerate resources
        for resource in self.resources_info.values():
            resource["available"] = self.drives[0].apply_resource_regeneration(
                resource["available"], resource["name"]
            )
        
        # Check termination
        done = self.resource_stock <= 0
        for agent in self.agents_info:
            if np.any(agent["internal_states"] < -1.0) or np.any(agent["internal_states"] > 1.0):
                done = True
                break
        
        self.intake_history.append(total_intake.copy())
        
        if self.render_mode == "human":
            self._render_frame()
        
        # Return for agent 0 (for compatibility)
        return self._get_obs(0), all_rewards[0], done, False, self._get_info()

    def _render_frame(self):
        """Render the multi-agent environment"""
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
        state_names = self.drives[0].get_internal_states_names()
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
            
            if resource["available"]:
                pygame.draw.rect(canvas, resource_color, 
                               pygame.Rect(x, 50, pix_square_size, 50))
                pygame.draw.rect(canvas, (0, 0, 0), 
                               pygame.Rect(x, 50, pix_square_size, 50), width=2)
            else:
                pygame.draw.rect(canvas, (200, 200, 200), 
                               pygame.Rect(x, 50, pix_square_size, 50))
                pygame.draw.rect(canvas, (100, 100, 100), 
                               pygame.Rect(x, 50, pix_square_size, 50), width=1)
        
        # Draw agents
        agent_colors = [(50, 150, 255), (255, 150, 50), (150, 255, 50), 
                       (255, 50, 150), (150, 50, 255)]
        
        for i, agent in enumerate(self.agents_info):
            color = agent_colors[i % len(agent_colors)]
            x = agent["position"] * pix_square_size + pix_square_size / 2
            y = 75
            
            # Draw agent with norm internalization indicator
            radius = 12 + int(self.beta[i] * 8)  # Larger circle for higher beta
            pygame.draw.circle(canvas, (0, 0, 0), (int(x), int(y)), radius + 2)
            pygame.draw.circle(canvas, color, (int(x), int(y)), radius)
            
            # Agent ID
            if pygame.font:
                font = pygame.font.SysFont('Arial', 10, bold=True)
                id_text = font.render(str(i), True, (255, 255, 255))
                canvas.blit(id_text, (int(x) - 5, int(y) - 5))
        
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
        stock_color = (0, 150, 0) if self.resource_stock > 500 else (150, 150, 0) if self.resource_stock > 200 else (150, 0, 0)
        stock_text = title_font.render(f"Resource Stock: {self.resource_stock:.1f}", True, stock_color)
        canvas.blit(stock_text, (20, info_y))
        
        # Total consumption this step
        if len(self.intake_history) > 0:
            total_text = label_font.render(f"Total intake/step: {np.sum(self.intake_history[-1]):.1f}", True, (0, 0, 0))
            canvas.blit(total_text, (300, info_y))
        
        # Agent states in columns
        agents_per_row = 3
        col_width = self.window_size // agents_per_row
        
        for i, agent in enumerate(self.agents_info):
            col = i % agents_per_row
            row = i // agents_per_row
            x_offset = col * col_width + 20
            y_offset = info_y + 30 + row * 100
            
            # Agent header with parameters
            agent_color = agent_colors[i % len(agent_colors)]
            pygame.draw.circle(canvas, agent_color, (x_offset, y_offset), 8)
            agent_text = label_font.render(f"Agent {i} (β={self.beta[i]:.2f}, α={self.alpha[i]:.2f})", True, (0, 0, 0))
            canvas.blit(agent_text, (x_offset + 15, y_offset - 8))
            
            # Belief about average consumption
            belief_text = value_font.render(f"Belief: {self.belief_avg_consumption[i].mean():.2f}", True, (100, 100, 100))
            canvas.blit(belief_text, (x_offset + 15, y_offset + 8))
            
            # Internal states
            state_names = self.drives[0].get_internal_states_names()
            for j, (state_name, state_value) in enumerate(zip(state_names, agent["internal_states"])):
                y_pos = y_offset + 25 + j * 15
                
                # Mini bar
                bar_width = 80
                bar_height = 10
                
                pygame.draw.rect(canvas, (200, 200, 200), 
                               pygame.Rect(x_offset, y_pos, bar_width, bar_height))
                
                normalized_value = (state_value + 1) / 2
                filled_width = int(normalized_value * bar_width)
                
                if state_value > 0:
                    bar_color = (0, 200, 0)
                else:
                    bar_color = (200, 0, 0)
                
                pygame.draw.rect(canvas, bar_color, 
                               pygame.Rect(x_offset, y_pos, filled_width, bar_height))
                pygame.draw.rect(canvas, (0, 0, 0), 
                               pygame.Rect(x_offset, y_pos, bar_width, bar_height), width=1)
                
                # Label with consumption indicator
                consumed = " *" if agent["last_intake"][j] > 0 else ""
                state_text = value_font.render(f"{state_name}: {state_value:.2f}{consumed}", True, (0, 0, 0))
                canvas.blit(state_text, (x_offset + bar_width + 5, y_pos))
        
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
            max_resource = max(self.initial_resource_stock, max(self.resource_history))
            
            points = []
            for i, stock in enumerate(self.resource_history[-200:]):  # Last 200 points
                x = 20 + (i / min(200, len(self.resource_history))) * graph_width
                y = graph_y + graph_height - 20 - (stock / max_resource) * (graph_height - 40)
                points.append((int(x), int(y)))
            
            if len(points) > 1:
                pygame.draw.lines(canvas, (0, 0, 255), False, points, 2)
            
            # Draw sustainability threshold
            threshold_y = graph_y + graph_height - 20 - (200 / max_resource) * (graph_height - 40)
            pygame.draw.line(canvas, (255, 0, 0), (20, threshold_y), (20 + graph_width, threshold_y), 1)
        
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
    
    # Create multi-agent environment
    env = MultiAgentHomeostaticEnv(
        config_path=config_path, 
        drive_type=drive_type, 
        n_agents=5,
        size=10, 
        render_mode="human"
    )
    
    # Set heterogeneous parameters to see different behaviors
    env.beta = np.array([0.2, 0.5, 0.8, 1.0, 0.0])  # Varied internalization (last agent is selfish)
    env.alpha = np.array([0.01, 0.05, 0.1, 0.05, 0.05])  # Varied learning rates
    
    obs, info = env.reset()
    
    for step in range(1000):
        # Let agents act autonomously
        obs, reward, done, truncate, info = env.step()
        
        if step % 50 == 0:
            print(f"Step {step}: Resource stock = {env.resource_stock:.2f}")
            avg_intake = np.mean([agent["last_intake"] for agent in env.agents_info], axis=0)
            print(f"  Average intake: {avg_intake}")
            
        if done:
            print(f"\nEpisode ended at step {step}")
            if env.resource_stock <= 0:
                print("Resources depleted - Tragedy of the commons!")
            else:
                print("Agent(s) died from homeostatic imbalance")
            break
    
    env.close()
    print("Simulation finished.")
