import numpy as np
import gymnasium as gym
from gymnasium import spaces
from src.utils.get_params import ParameterHandler
from src.utils.resource_manager import GlobalResourceManager


class NormarlHomeostaticBaseEnv(gym.Env):
    """
    Single-agent homeostatic environment with social norms.
    Integrates NORMARL's social cost mechanism with homeostatic drives.
    Q (consumption in NORMARL) = K (intake in homeostatic system)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config_path, drive_type, social_learning_rate, beta, single_agent=True, render_mode=None, size=10):
        self.param_manager = ParameterHandler(config_path)
        self.drive_type = drive_type
        self.size = size
        self.window_size = 800
        self.single_agent = single_agent

        # Initialize drive
        self.drive = self.param_manager.create_drive(drive_type)
        self.dimension_internal_states = self.drive.get_internal_state_dimension()

        # Global resource manager
        self.resource_manager = GlobalResourceManager(config_path, drive_type)

        # NORMARL parameters
        self.beta = beta
        self.social_alpha = social_learning_rate
        self.gamma = 0.1  # Softmax temperature
        self.a = 1.0  # Base social cost
        self.b = 0.5  # Resource scarcity multiplier

        # Belief about average consumption
        self.perceived_social_norm = np.zeros(self.dimension_internal_states)

        # Resource stock
        self.initial_resource_stock = np.ones(self.dimension_internal_states) * 1.5
        self.resource_stock = self.initial_resource_stock.copy()
        self.resource_regeneration_rate = self.resource_manager.get_resource_stock_regeneration_array()

        # Action space
        num_actions = 3 + self.dimension_internal_states
        self.action_space = spaces.Discrete(num_actions)

        # Observation space
        self.observation_space = spaces.Dict({
            "position": spaces.Discrete(self.size),
            "internal_states": spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.dimension_internal_states,),
                dtype=np.float64
            ),
            "perceived_social_norm": spaces.Box(
                low=0, high=1.0,
                shape=(self.dimension_internal_states,),
                dtype=np.float64
            )
        })

        # Initialize resources
        self._initialization_resources()
        self.agent_info = {
            "position": 0,
            "internal_states": np.zeros(self.dimension_internal_states, dtype=np.float64),
            "perceived_social_norm": np.zeros(self.dimension_internal_states, dtype=np.float64)
        }

        self.intake = np.zeros(self.dimension_internal_states, dtype=np.float64)
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
        """Initialize resource positions on the grid."""
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
            }

    def compute_social_cost(self, intake):
        """
        Compute social cost based on NORMARL equation (2).
        Si(Qi) = 0 if Qi < Q̄i
        Si(Qi) = βi * (Qi - Q̄i) * max{0, (a - b*E)} if Qi ≥ Q̄i
        """
        belief = self.perceived_social_norm
        scarcity_factor = np.maximum(0, self.a - self.b * self.resource_stock)

        social_cost = 0
        for i in range(self.dimension_internal_states):
            if intake[i] > belief[i]:
                social_cost += self.beta * (intake[i] - belief[i]) * scarcity_factor[i]

        return social_cost

    def update_belief(self, observed_avg_intake):
        """
        Update belief about average consumption using equation (5).
        Q̄i(t+1) = (1 - αi) * Q̄i(t) + αi * q̄(t+1)
        """
        self.perceived_social_norm = (
            (1 - self.social_alpha) * self.perceived_social_norm +
            self.social_alpha * observed_avg_intake
        )

    def update_resource_stock(self, total_intake):
        """
        Update resource stock: E(t+1) = (1 + δ) * E(t) - Σ(Qi)
        """
        self.resource_stock = self.resource_manager.update_resource_stock(
            self.resource_stock, 2 * total_intake
        )
        self.resource_stock = np.minimum(self.resource_stock, self.initial_resource_stock)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.resource_stock = self.initial_resource_stock.copy()
        self.agent_info["position"] = self.np_random.choice(list(range(self.size)))
        self.agent_info["internal_states"] = self.np_random.uniform(
            low=-0.3, high=0.3,
            size=(self.dimension_internal_states,)
        )

        initial_drive = self.drive.compute_drive(self.agent_info["internal_states"])
        self.drive.update_drive(initial_drive)

        self.perceived_social_norm = np.zeros(self.dimension_internal_states)
        self.intake_history = []
        self.resource_history = [self.resource_stock.copy()]

        return self._get_obs(), self._get_info()

    def _get_obs(self):
        return {
            "position": self.agent_info["position"],
            "internal_states": self.agent_info["internal_states"],
            "perceived_social_norm": self.perceived_social_norm
        }

    def _get_info(self):
        return {
            "agent_info": self.agent_info.copy(),
            "last_intake": self.intake.copy()
        }

    def step(self, action, avg_intake):
        if action is None:
            raise ValueError("Action must be provided!")

        # 1. Apply natural decay
        states_after_decay = self.drive.apply_natural_decay(self.agent_info["internal_states"])

        # 2. Process movement action
        current_position = self.agent_info["position"]
        new_position = current_position
        if action == 0:
            new_position = current_position
        elif action == 1:
            new_position = max(0, current_position - 1)
        elif action == 2:
            new_position = min(self.size - 1, current_position + 1)

        if new_position not in self.area:
            new_position = current_position

        new_internal_state = states_after_decay.copy()

        # 3. Process consumption actions
        intake_resources = np.zeros(self.dimension_internal_states)
        for i in range(self.dimension_internal_states):
            if action == 3 + i:
                resource = self.resources_info[i]
                if current_position == resource["position"]:
                    intake_resources[i] = 1.0

        new_internal_state = self.drive.apply_intake(states_after_decay, intake_resources)
        self.intake = self.drive.get_intake_array(intake_resources)

        # 4. Update agent state
        self.agent_info["position"] = new_position
        self.agent_info["internal_states"] = new_internal_state

        # 5. Calculate reward
        reward = self._compute_reward(self.intake) * 100

        # 6. Update global state
        self._update_global_environment(self.intake, avg_intake)

        # 7. Check termination
        done = self._check_termination()

        return self._get_obs(), reward, done, False, self._get_info()

    def _compute_reward(self, intake):
        old_drive = self.drive.get_current_drive()
        new_drive = self.drive.compute_drive(self.agent_info["internal_states"])
        homeostatic_reward = self.drive.compute_reward(old_drive, new_drive)
        self.drive.update_drive(new_drive)

        social_cost = self.compute_social_cost(intake)

        return homeostatic_reward - social_cost * 10

    def _update_global_environment(self, intake, avg_intake):
        self.update_resource_stock(intake)
        self.intake_history.append(intake)
        self.resource_history.append(self.resource_stock.copy())
        self.update_belief(avg_intake)

    def _check_termination(self):
        if np.any(self.resource_stock <= 0):
            return True
        if np.any(np.abs(self.agent_info["internal_states"]) > 1):
            return True
        return False

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
