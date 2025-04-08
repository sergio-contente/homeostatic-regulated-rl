import torch
import numpy as np
from utils.e_greedy import epsilon_greedy_policy

class QLearning:
    """
    Implementation of a Q-Learning agent with Q-table.
    
    Attributes:
        state_size (int): Number of possible states
        action_size (int): Number of possible actions
        learning_rate (float): Learning rate (alpha)
        discount_factor (float): Discount factor for future rewards (gamma)
        epsilon (float): Initial probability of choosing a random action
        epsilon_min (float): Minimum value for epsilon
        epsilon_decay (float): Decay rate for epsilon after each episode
        q_table (np.ndarray): Q-table to store Q-values for state-action pairs
    """
    
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """
        Initializes the Q-Learning agent.
        
        Args:
            state_size (int): Number of possible states
            action_size (int): Number of possible actions
            learning_rate (float, optional): Learning rate. Default: 0.1
            discount_factor (float, optional): Discount factor. Default: 0.99
            epsilon (float, optional): Initial probability of random action. Default: 1.0
            epsilon_min (float, optional): Minimum value for epsilon. Default: 0.01
            epsilon_decay (float, optional): Decay rate for epsilon. Default: 0.995
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((state_size, action_size))
        
    def get_action(self, state):
        """
        Selects an action using epsilon-greedy policy.
        
        Args:
            state (int): The current state of the environment
            
        Returns:
            int: The selected action
        """
        return epsilon_greedy_policy(self.q_table[state], self.epsilon)
    
    def update_q_table(self, state, action, reward, next_state, done):
        """
        Updates the Q-table using the Bellman equation.
        
        Args:
            state (int): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (int): Next state
            done (bool): Indicates if the episode is finished
            
        Returns:
            float: Value of the error (difference between old and new value)
        """
        # Current Q-value for the state-action pair
        current_q = self.q_table[state, action]
        
        # Maximum Q-value for the next state
        if done:
            max_next_q = 0  # No future estimate if the episode is finished
        else:
            max_next_q = np.max(self.q_table[next_state])
        
        # Calculate the target value using the Bellman equation
        target_q = reward + self.discount_factor * max_next_q
        
        # Update the Q-value
        self.q_table[state, action] = current_q + self.learning_rate * (target_q - current_q)
        
        # Return the TD (Temporal Difference) error
        return target_q - current_q
    
    def decay_epsilon(self):
        """
        Reduces the value of epsilon to balance exploration and exploitation.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save_q_table(self, filepath):
        """
        Saves the Q-table to a file.
        
        Args:
            filepath (str): Path to save the file
        """
        np.save(filepath, self.q_table)
        
    def load_q_table(self, filepath):
        """
        Loads the Q-table from a file.
        
        Args:
            filepath (str): Path of the file to be loaded
        """
        self.q_table = np.load(filepath)
        
    def train(self, env, num_episodes, max_steps_per_episode=1000):
        """
        Trains the agent using the Q-Learning algorithm.
        
        Args:
            env: Training environment that implements reset() and step() functions
            num_episodes (int): Number of episodes for training
            max_steps_per_episode (int, optional): Maximum number of steps per episode
            
        Returns:
            list: List of total rewards per episode
        """
        rewards_per_episode = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()  # Compatible with Gymnasium
            state = self._process_state(state)  # Convert state to index if necessary
            
            total_reward = 0
            done = False
            truncated = False
            
            for step in range(max_steps_per_episode):
                # Select an action
                action = self.get_action(state)
                
                # Execute the action in the environment
                next_state, reward, done, truncated, _ = env.step(action)  # Compatible with Gymnasium
                next_state = self._process_state(next_state)  # Convert state to index if necessary
                
                # Update the Q-table
                self.update_q_table(state, action, reward, next_state, done)
                
                # Update the state and total reward
                state = next_state
                total_reward += reward
                
                # End the episode if necessary
                if done or truncated:
                    break
            
            # Record the total reward for the episode
            rewards_per_episode.append(total_reward)
            
            # Decay epsilon after each episode
            self.decay_epsilon()
            
            # Display progress every 10 episodes
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(rewards_per_episode[-10:])
                print(f"Episode: {episode + 1}/{num_episodes}, Average Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.4f}")
        
        return rewards_per_episode
    
    def _process_state(self, state):
        """
        Converts a state to an index in the Q-table if necessary.
        Can be overridden to process more complex states.
        
        Args:
            state: Environment state (can be a number, array, etc.)
            
        Returns:
            int: State index in the Q-table
        """
        # If state is already an integer, return as is
        if isinstance(state, (int, np.integer)):
            return state
        
        # If it's an array or tensor, convert to an index
        if isinstance(state, (np.ndarray, list, torch.Tensor)):
            # Simple implementation - override this method for more complex states
            if hasattr(state, 'item') and callable(getattr(state, 'item')):
                return state.item()  # For PyTorch tensors
            elif hasattr(state, 'tolist') and callable(getattr(state, 'tolist')):
                return hash(tuple(state.tolist()))  # For NumPy arrays
            else:
                return hash(tuple(state))  # For lists
                
        # For other types, try to convert to hash
        try:
            return hash(state)
        except:
            raise ValueError(f"Could not process state: {state}. Implement a suitable state processing method.")

    def evaluate(self, env, num_episodes=10, render=False):
        """
        Evaluates the performance of the trained agent.
        
        Args:
            env: Evaluation environment
            num_episodes (int, optional): Number of episodes for evaluation
            render (bool, optional): If True, renders the environment during evaluation
            
        Returns:
            float: Average reward per episode
        """
        total_rewards = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            state = self._process_state(state)
            
            episode_reward = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                # Always choose the best action during evaluation (epsilon = 0)
                action = np.argmax(self.q_table[state])
                
                # Execute the action in the environment
                next_state, reward, done, truncated, _ = env.step(action)
                next_state = self._process_state(next_state)
                
                if render:
                    env.render()
                
                # Update the state and reward
                state = next_state
                episode_reward += reward
            
            total_rewards.append(episode_reward)
            print(f"Evaluation episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")
        
        avg_reward = np.mean(total_rewards)
        print(f"Evaluation completed with {num_episodes} episodes. Average reward: {avg_reward:.2f}")
        
        return avg_reward
