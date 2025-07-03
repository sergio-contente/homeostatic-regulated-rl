import functools
from pettingzoo import AECEnv
import numpy as np
import gymnasium
import numpy as np
from gymnasium.spaces import Discrete, Box, Dict
from gymnasium.utils import seeding

from pettingzoo.utils import AgentSelector, wrappers

from src.utils.resource_manager import GlobalResourceManager
from .homeostatic_agent import HomeostaticAgent
from .actions import DefaultHomeostaticAction
from .observations import DefaultHomeostaticObservation

class NormalHomeostaticEnv(AECEnv):
    metadata = {
        "name": "normal_homeostatic_env_v0",
    }

    def __init__(self, config_path, drive_type, learning_rate, beta, number_resources, n_agents=10, render_mode=None, size=10):
        # Environment parameters
        self.size = size
        self.dimension_internal_states = number_resources
        self.n_agents = n_agents
        self.render_mode = render_mode
        
        # Agent configuration parameters
        self.config_path = config_path
        self.drive_type = drive_type
        self.learning_rate = learning_rate
        self.beta = beta
        
        # Global resource management
        self.global_resource_manager = GlobalResourceManager(config_path, drive_type)
        
        # Resource stock (NORMARL-style shared resource pool)
        self.initial_resource_stock = np.ones(number_resources) * 3
        self.resource_stock = self.initial_resource_stock.copy()
        self.resource_regeneration_rate = self.global_resource_manager.get_resource_stock_regeneration_array()
        print(f"🔄 Resource regeneration rate: {self.resource_regeneration_rate}")
        
        # Agent identification
        self.possible_agents = [f"agent_{i}" for i in range(n_agents)]
        
        # Initialize containers for agents and their functions
        self.homeostatic_agents = {}
        self.action_functions = {}
        self.observation_functions = {}
        
        # Spaces will be set in reset after agents are created
        self._action_spaces = {}
        self._observation_spaces = {}
        
        # Initialize resource positions (similar to gymnasium environments)
        self._initialize_resources()

    def _initialize_resources(self):
        """
        Initialize resource positions on the grid.
        Creates fixed positions for each resource type that agents must visit to consume.
        """
        # Create a temporary drive to get state names
        temp_drive = self.global_resource_manager.param_manager.create_drive(self.drive_type)
        state_names = temp_drive.get_internal_states_names()
        
        # Use fixed seed for reproducible resource positions
        resource_rng = np.random.RandomState(123)
        
        # Distribute resources across the grid
        if self.dimension_internal_states <= self.size:
            # Enough positions for unique locations
            random_positions = resource_rng.choice(
                self.size, size=self.dimension_internal_states, replace=False
            )
        else:
            # More resources than positions - allow duplicates
            random_positions = resource_rng.choice(
                self.size, size=self.dimension_internal_states, replace=True
            )
        
        # Create resources_info structure
        self.resources_info = {}
        for i, state_name in enumerate(state_names):
            self.resources_info[i] = {
                "name": state_name,
                "position": random_positions[i],
                "available": True  # In NORMARL, resources regenerate from global stock
            }
        
        #print(f"🌱 Initialized {len(self.resources_info)} resources at positions: {[r['position'] for r in self.resources_info.values()]}")

    def reset(self, seed=None, options=None):
        """
        Reset the environment and initialize all homeostatic agents, observations, and actions.
        
        Initializes:
        - agents: List of active agents
        - rewards, terminations, truncations, infos: PettingZoo required attributes
        - homeostatic_agents: HomeostaticAgent instances for each agent
        - action_functions: Action functions for each agent
        - observation_functions: Observation functions for each agent
        - observations: Current observations for each agent
        - resource_stock: Reset shared resource pool
        """
        # Set random seed
        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)
        else:
            if not hasattr(self, 'np_random'):
                self.np_random, self.np_random_seed = seeding.np_random(None)
        
        # Initialize agent list
        self.agents = self.possible_agents[:]
        
        # Initialize PettingZoo required attributes
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.num_moves = 0
        
        # Reset resource system
        self.resource_stock = self.initial_resource_stock.copy()
        
        # Create homeostatic agents for each agent
        self.homeostatic_agents = {}
        self.action_functions = {}
        self.observation_functions = {}
        
        for agent_id in self.agents:
            # Create homeostatic agent with random initial position
            initial_position = self.np_random.integers(0, self.size)
            
            # Create homeostatic agent
            homeostatic_agent = HomeostaticAgent(
                agent_id=agent_id,
                config_path=self.config_path,
                drive_type=self.drive_type,
                initial_position=initial_position,
                social_learning_rate=self.learning_rate,
                beta=self.beta
            )
            self.homeostatic_agents[agent_id] = homeostatic_agent
            
            # Create action function (1D version for NORMARL)
            action_function = DefaultHomeostaticAction(homeostatic_agent, self)
            self.action_functions[agent_id] = action_function
            
            # Create observation function
            observation_function = DefaultHomeostaticObservation(homeostatic_agent, self)
            self.observation_functions[agent_id] = observation_function
            
            # Cache action and observation spaces
            self._action_spaces[agent_id] = action_function.action_space()
            self._observation_spaces[agent_id] = observation_function.observation_space()
        
        # Initialize observations for all agents
        self.observations = {}
        for agent_id in self.agents:
            self.observations[agent_id] = self.observation_functions[agent_id]()
        
        # Initialize agent selector for turn-based stepping
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()
        
        return self.observations, self.infos

    def step(self, action):
        """
        Execute one step for the current agent in the NORMARL homeostatic environment.
        
        This method:
        1. Applies natural decay to current agent
        2. Executes the agent's action via action function
        3. Calculates homeostatic and social rewards
        4. Updates global resource state
        5. Checks for termination conditions
        6. Moves to next agent
        
        Args:
            action: Action index for the current agent
        """
        # Handle dead agents
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        current_agent_id = self.agent_selection
        current_agent = self.homeostatic_agents[current_agent_id]
        action_func = self.action_functions[current_agent_id]

        # Reset cumulative rewards for current agent
        self._cumulative_rewards[current_agent_id] = 0

        print(f"🔄 Applying natural decay to current agent's internal states")
        print(f"🔄 Current agent's internal states: {current_agent.internal_states}")
        old_internal_states = current_agent.internal_states.copy()
        # 1. Apply natural decay to current agent's internal states
        current_agent.apply_natural_decay()
        print(f"🔄 Current agent's internal states after natural decay: {current_agent.internal_states}")

        # 2. Execute action through action function
        action_result = action_func.execute_action(action)
        print(f"🔄 Executing action: {action_result}")
        
        # 3. Update agent position based on action result
        new_position = action_result["agent_new_position"]
        current_agent.update_position(new_position)
        print(f"🔄 Agent moved to position: {new_position}")
        
        # 4. Validate and consume resources
        print(f"📦 Consuming resources: {action_result['resources_to_consume']}")
        resources_to_consume = action_result["resources_to_consume"]
        print(f"📦 Resources to consume: {resources_to_consume}")
        
        # The actions.py already checked position, now just validate stock and deduct
        actual_consumption = np.zeros_like(resources_to_consume)
        resource_consumed = False
        
        for i, amount in enumerate(resources_to_consume):
            if amount > 0:
                # Check if global resource stock has enough
                available_amount = min(amount, self.resource_stock[i])
                if available_amount > 0:
                    actual_consumption[i] = available_amount
                    # Deduct from global resource stock
                    self.resource_stock[i] -= available_amount
                    resource_consumed = True
                    print(f"📦 Consumed {available_amount} of resource {i}")
                else:
                    print(f"📦 No resource {i} available in global stock")
        
        # Apply actual consumption to agent
        if resource_consumed:
            last_intake, states_before = current_agent.consume_resource(actual_consumption)
        else:
            last_intake = np.zeros(self.dimension_internal_states)
            states_before = old_internal_states.copy()  # States BEFORE natural decay
        print(f"📦 Actual consumption: {actual_consumption}")
        print(f"📦 Last intake: {last_intake}")
        print(f"📦 States before: {states_before}")
        print(f"📦 Current agent's internal states after consumption: {current_agent.internal_states}")
        
        # 5. Calculate rewards for current agent
        reward = 0.0
        
        # Homeostatic reward (drive reduction)
        print(f"🏥 Computing homeostatic reward")
        homeostatic_reward = current_agent.compute_homeostatic_reward(states_before)
        print(f"🏥 Homeostatic reward: {homeostatic_reward}")
        
        # Social cost (if consumption occurred)
        social_cost = 0.0
        if resource_consumed:
            # Use actual intake for social cost calculation
            resource_scarcity = self._compute_resource_scarcity()
            social_cost = current_agent.compute_social_cost(last_intake, resource_scarcity)
            print(f"👥 Social cost: {social_cost}")
        
        # Combined reward
        reward = homeostatic_reward - social_cost
        
        # Small penalty for invalid actions (trying to consume when no stock available)
        if np.sum(resources_to_consume) > np.sum(actual_consumption):
            reward -= 0.1
            print(f"⚠️  Invalid action penalty applied (no stock available)")

        # Store reward for current agent
        self.rewards[current_agent_id] = reward

        # 6. If this is the last agent in the round, update global environment
        if self._agent_selector.is_last():
            print(f"🌍 Resource stock before update: {self.resource_stock}")
            self._update_global_environment()
            print(f"🌍 Resource stock after update: {self.resource_stock}")
            self._update_social_norms()
            self._check_resource_regeneration()
            
            # Check termination conditions for all agents
            self._check_termination_conditions()
            
            # Increment episode step counter
            self.num_moves += 1
            
            # Check if episode should end (max steps)
            max_steps = getattr(self, 'max_steps', 1000)
            if self.num_moves >= max_steps:
                self.truncations = {agent: True for agent in self.agents}
        else:
            # Clear rewards for other agents until all have acted
            for agent_id in self.agents:
                if agent_id != current_agent_id:
                    self.rewards[agent_id] = 0

        # 7. Update observations for all agents
        for agent_id in self.agents:
            self.observations[agent_id] = self.observation_functions[agent_id]()

        # 8. Move to next agent
        self.agent_selection = self._agent_selector.next()

        # 9. Accumulate rewards
        self._accumulate_rewards()

        # 10. Render if needed
        if self.render_mode == "human":
            self.render()

    def _compute_resource_scarcity(self):
        """
        Compute resource scarcity factors for social cost calculation.
        
        Returns:
            np.ndarray: Scarcity factor for each resource type
        """
        # NORMARL-style scarcity: max(0, a - b*E)
        a = 1.0  # Base social cost
        b = 0.5  # Resource scarcity multiplier
        
        # Normalize resource stock to [0, 1] range
        normalized_stock = self.resource_stock / self.initial_resource_stock
        scarcity = np.maximum(0, a - b * normalized_stock)
        
        return scarcity

    def _update_global_environment(self):
        """Update global environment state after all agents have acted."""
        # Update resource stock based on actual resource consumption
        # Note: We need to track actual resource units consumed, not intake benefits
        total_consumption = np.zeros(self.dimension_internal_states)
        
        # For now, we'll use a simple approach: count consumption actions
        # Each consumption action (action 3+i) consumes 1.0 resource unit
        # This is a simplification - ideally we'd track actual consumption per agent
        for agent_id in self.agents:
            agent = self.homeostatic_agents[agent_id]
            # If agent consumed anything (last_intake > 0), assume they consumed 1.0 resource unit
            if np.any(agent.last_intake > 0):
                # Find which resource was consumed and add 1.0 to that resource
                consumed_resource = np.where(agent.last_intake > 0)[0]
                for resource_idx in consumed_resource:
                    total_consumption[resource_idx] += 1.0
        
        print(f"🌍 Total consumption this round: {total_consumption}")
        
        # Store stock before update to calculate regeneration
        stock_before = self.resource_stock.copy()
        
        # Apply NORMARL equation: Et+1 = (1 + δ)Et - ΣQi,t
        # where δ is the natural regeneration rate and ΣQi,t is total consumption
        regeneration_rate = self.resource_regeneration_rate[0]  # Get from config file
        self.resource_stock = (1 + regeneration_rate) * stock_before - total_consumption
        
        # Ensure resource stock doesn't go negative
        self.resource_stock = np.maximum(0, self.resource_stock)
        
        # Clamp resource stock to initial levels (prevent exceeding maximum)
        self.resource_stock = np.minimum(self.resource_stock, self.initial_resource_stock)
        
        print(f"🌍 Stock before regeneration: {stock_before}")
        print(f"🌍 Stock after regeneration: {self.resource_stock}")
        print(f"🌍 Regeneration rate (δ): {regeneration_rate}")

    def _update_social_norms(self):
        """Update each agent's perception of social norms based on observed behavior."""
        # Calculate average consumption across all agents
        if len(self.agents) > 0:
            total_intake = np.zeros(self.dimension_internal_states)
            
            for agent_id in self.agents:
                agent = self.homeostatic_agents[agent_id]
                total_intake += agent.last_intake
            
            avg_intake = total_intake / len(self.agents)
            
            # Update social norm perception for each agent
            for agent_id in self.agents:
                agent = self.homeostatic_agents[agent_id]
                agent.update_social_norm_perception(avg_intake)
                
                # Let agents observe each other's behavior
                for other_agent_id in self.agents:
                    if other_agent_id != agent_id:
                        other_agent = self.homeostatic_agents[other_agent_id]
                        agent.observe_other_agent_behavior(other_agent.last_intake)

    def _check_resource_regeneration(self):
        """Apply resource regeneration if needed."""
        # In NORMARL, resources at fixed positions regenerate each turn
        # This allows multiple agents to potentially access the same resource type
        for resource_info in self.resources_info.values():
            resource_info["available"] = True

    def _check_termination_conditions(self):
        """Check if any agents should terminate due to critical states."""
        for agent_id in self.agents:
            agent = self.homeostatic_agents[agent_id]
            
            # Check if agent is in critical homeostatic state
            if agent.is_in_critical_state(threshold=1.0):
                self.terminations[agent_id] = True
                print(f"Agent {agent_id} terminated due to critical homeostatic state")
            
            # Check if all resources are depleted
            if np.all(self.resource_stock <= 0):
                self.terminations[agent_id] = True
                print(f"Agent {agent_id} terminated due to resource depletion")

    def _clear_rewards(self):
        """Clear rewards for all agents."""
        for agent in self.agents:
            self.rewards[agent] = 0

    def _accumulate_rewards(self):
        """Accumulate rewards for all agents."""
        for agent in self.agents:
            self._cumulative_rewards[agent] += self.rewards[agent]

    def _was_dead_step(self, action):
        """Handle step for an agent that is already terminated."""
        # Move to next agent without processing action
        self.agent_selection = self._agent_selector.next()

    def render(self):
        pass

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # Get current observation from the observation function
        if agent in self.observation_functions:
            return self.observation_functions[agent]()
        else:
            # Fallback to cached observation if available
            return self.observations.get(agent, {})
    
    def action_space(self, agent):
        """
        Return the action space for the specified agent.
        """
        return self._action_spaces[agent]
    
    def observation_space(self, agent):
        """
        Return the observation space for the specified agent.
        """
        return self._observation_spaces[agent]


def main():
    """
    Test function to verify NORMARL environment functionality.
    """
    print("🧪 Testing NORMARL Homeostatic Environment")
    print("=" * 50)
    
    # Environment parameters
    config_path = "config/config.yaml"
    drive_type = "base_drive" 
    learning_rate = 0.1
    beta = 0.5
    number_resources = 1
    n_agents = 3
    size = 1
    
    # Create environment
    print("🏗️  Creating environment...")
    env = NormalHomeostaticEnv(
        config_path=config_path,
        drive_type=drive_type,
        learning_rate=learning_rate,
        beta=beta,
        number_resources=number_resources,
        n_agents=n_agents,
        size=size
    )
    
    # Reset environment
    print("\n🔄 Resetting environment...")
    observations, info = env.reset()
    
    print(f"✅ Environment reset successfully!")
    print(f"📊 Agents: {env.agents}")
    print(f"🧠 Current agent: {env.agent_selection}")
    print(f"🏃 Agent positions: {[agent.position for agent in env.homeostatic_agents.values()]}")
    print(f"📍 Resource positions: {[r['position'] for r in env.resources_info.values()]}")
    print(f"📦 Initial resource stock: {env.resource_stock}")
    
    # Print initial agent states
    print("\n👥 Initial Agent States:")
    for agent_id, agent in env.homeostatic_agents.items():
        state_names = agent.get_state_names()
        state_str = ", ".join([f"{name}: {val:.3f}" for name, val in zip(state_names, agent.internal_states)])
        print(f"  {agent_id}: pos={agent.position}, drive={agent.get_current_drive():.3f}, states=[{state_str}]")
    
    # Test action spaces
    print("\n🎮 Action Spaces:")
    for agent_id in env.agents:
        action_space = env.action_space(agent_id)
        action_func = env.action_functions[agent_id]
        #action_info = action_func.get_action_info()
        #print(f"  {agent_id}: {action_space.n} actions - {action_info['action_descriptions']}")
        print(f"  {agent_id}: {action_space.n} actions")
    
    # Run simulation for several steps
    print("\n🚀 Running simulation...")
    print("-" * 30)
    
    max_steps = 20
    step_count = 0
    
    while env.agents and step_count < max_steps:
        # Get current agent
        current_agent_id = env.agent_selection
        current_agent = env.homeostatic_agents[current_agent_id]
        action_func = env.action_functions[current_agent_id]
        
        # Choose a random action for testing
        action = env.np_random.integers(0, action_func.action_space().n)
        print(f"Step {step_count}: {current_agent_id} taking action {action}")
        
        # Execute step
        env.step(action)
        
        # Print reward and state changes
        reward = env.rewards.get(current_agent_id, 0)
        print(f"  Reward: {reward:.3f}")
        
        # If all agents have acted this round, show summary
        if env._agent_selector.is_last():
            print(f"  📈 Resource stock: {env.resource_stock}")
            print(f"  🌍 Global update completed")
            
            # Show social norms
            print("  📋 Social norms learned:")
            for agent_id, agent in env.homeostatic_agents.items():
                norm_str = ", ".join([f"{name}: {val:.3f}" for name, val in zip(agent.get_state_names(), agent.perceived_social_norm)])
                print(f"    {agent_id}: [{norm_str}]")
        
        step_count += 1
        print()
        
        # Check if episode ended
        if all(env.terminations.values()) or all(env.truncations.values()):
            print("🏁 Episode ended!")
            break
    
    # Final summary
    print("\n📊 Final Summary:")
    print(f"Steps completed: {step_count}")
    print(f"Final resource stock: {env.resource_stock}")
    print(f"Agent terminations: {env.terminations}")
    print(f"Agent truncations: {env.truncations}")
    
    print("\n👥 Final Agent States:")
    for agent_id, agent in env.homeostatic_agents.items():
        state_names = agent.get_state_names()
        state_str = ", ".join([f"{name}: {val:.3f}" for name, val in zip(state_names, agent.internal_states)])
        norm_str = ", ".join([f"{name}: {val:.3f}" for name, val in zip(state_names, agent.perceived_social_norm)])
        print(f"  {agent_id}:")
        print(f"    Position: {agent.position}")
        print(f"    Drive: {agent.get_current_drive():.3f}")
        print(f"    States: [{state_str}]")
        print(f"    Social norms: [{norm_str}]")
        print(f"    Total intake: {sum(agent.intake_history) if agent.intake_history else 0}")
    
    print("\n✅ Test completed successfully!")


if __name__ == "__main__":
    main()
