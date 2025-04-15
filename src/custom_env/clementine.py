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

    def train(self, num_episodes, max_steps_per_episode=1000):
        """
        Trains the agent using the Q-Learning algorithm.
        
        Args:
            num_episodes (int): Number of episodes for training
            max_steps_per_episode (int, optional): Maximum number of steps per episode
                
        Returns:
            list: List of total rewards per episode
        """
        rewards_per_episode = []
        
        for episode in range(num_episodes):
            state, _ = self.reset() 
            state_idx = self._process_state(state)
            
            total_reward = 0
            done = False
            truncated = False
            
            for step in range(max_steps_per_episode):
                # Select an action
                action = self.agent.get_action(state_idx)
                
                # Execute the action in the environment
                next_state, reward, done, truncated, _ = self.step(action)
                next_state_idx = self._process_state(next_state)
                
                # Update the Q-table
                self.agent.update_q_table(state_idx, action, reward, next_state_idx, done)
                
                # Update the state and total reward
                state = next_state
                state_idx = next_state_idx
                total_reward += reward
                
                # End the episode if necessary
                if done or truncated:
                    break
            
            # Record the total reward for the episode
            rewards_per_episode.append(total_reward)
            
            # Decay epsilon after each episode
            self.agent.decay_epsilon()
            
            # Display progress every 10 episodes
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(rewards_per_episode[-10:])
                print(f"Episode: {episode + 1}/{num_episodes}, Average Reward: {avg_reward:.2f}, Epsilon: {self.agent.epsilon:.4f}")
        
        return rewards_per_episode
    
    def _process_state(self, state):
        """Process the state for use in the Q-table."""
        # If it's a dict from the original environment
        if isinstance(state, dict) and "internal_states" in state:
            state = state["internal_states"]
        
        if isinstance(state, np.ndarray):
            # Discretize each value in the array to a value between 0 and maxh
            discrete_state = []
            for i, val in enumerate(state):
                # Normalize to [0, maxh]
                bin_idx = int((val + self.maxh) / (2 * self.maxh) * self.maxh)
                bin_idx = max(0, min(self.maxh, bin_idx))  # Clip to ensure limits
                discrete_state.append(bin_idx)
                
            # Convert the multi-dimensional state to a single index
            return np.ravel_multi_index(tuple(discrete_state), dims=[self.maxh + 1] * len(state))
                
        if isinstance(state, (int, np.integer)):
            # If already an index, return directly
            return state
                
        raise ValueError(f"Unsupported state format: {type(state)}")

    def evaluate(self, num_episodes=10, render=False):
        """
        Evaluates the performance of the trained agent.
        
        Args:
            num_episodes (int, optional): Number of episodes for evaluation
            render (bool, optional): If True, renders the environment during evaluation
                
        Returns:
            float: Average reward per episode
        """
        total_rewards = []
        
        for episode in range(num_episodes):
            state, _ = self.reset()
            state_idx = self._process_state(state)
            
            episode_reward = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                # Always choose the best action during evaluation (epsilon = 0)
                action = np.argmax(self.agent.q_table[state_idx])
                
                # Execute the action in the environment
                next_state, reward, done, truncated, _ = self.step(action)
                next_state_idx = self._process_state(next_state)
                
                if render and self.render_mode == "human":
                    self._render_frame()
                
                # Update the state and reward
                state = next_state
                state_idx = next_state_idx
                episode_reward += reward
            
            total_rewards.append(episode_reward)
            print(f"Evaluation episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")
        
        avg_reward = np.mean(total_rewards)
        print(f"Evaluation completed with {num_episodes} episodes. Average reward: {avg_reward:.2f}")
        
        return avg_reward

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
                if i == 0 and j == 0:
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


def plot_rewards(rewards):
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


def main():
    """Main function for training and evaluating the Clementine agent."""
    # Parameters
    config_path = "config/config.yaml"  # Adjust to the correct path of your configuration file
    drive_type = "brase_drive"  # Type of drive to use
    render_mode = None  # Set to "human" to visualize training
    num_episodes = 1000  # Number of episodes for training
    eval_episodes = 10  # Number of episodes for evaluation
    
    print("Initializing Clementine environment...")
    env = Clementine(config_path, drive_type, render_mode=render_mode)
    
    print(f"\nStarting training for {num_episodes} episodes...")
    start_time = time.time()
    rewards = env.train(num_episodes)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds!")
    
    # Plot the rewards curve
    plot_rewards(rewards)
    
    # Evaluation with rendering
    print(f"\nEvaluating the trained agent for {eval_episodes} episodes...")
    # Create a new instance for evaluation with rendering
    eval_env = Clementine(config_path, drive_type, render_mode="human")
    # Copy the Q-table from the trained agent
    eval_env.agent.q_table = env.agent.q_table.copy()
    eval_env.agent.epsilon = 0.0  # Disable exploration during evaluation
    
    avg_reward = eval_env.evaluate(num_episodes=eval_episodes, render=True)
    print(f"Evaluation completed! Average reward: {avg_reward:.2f}")
    
    # Close pygame if active
    if render_mode == "human" or eval_env.render_mode == "human":
        pygame.quit()


if __name__ == "__main__":
    main()
