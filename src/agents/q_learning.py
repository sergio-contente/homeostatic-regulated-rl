import torch
import numpy as np
from ..utils.e_greedy import epsilon_greedy_policy

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
    
    def __init__(self, state_size, action_size, env=None, learning_rate=0.1, discount_factor=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, n_bins = 20):
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
        self.n_bins = n_bins
        self.size = env.size if env != None else state_size -1 
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
        q_values = self.q_table[state]
        return epsilon_greedy_policy(q_values, self.epsilon)
    
    def update_q_table(self, state, action, reward, next_state, done):
        # Converte os estados para índices
        state_idx = self._process_state(state)
        next_state_idx = self._process_state(next_state)

        self.q_table[state_idx, action] += self.learning_rate * (reward + self.discount_factor * np.max(self.q_table[next_state_idx, :]) - self.q_table[state_idx, action])

        
    
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
        """Processa o estado para uso na tabela Q."""
        # Se for dict vindo do ambiente original
        if isinstance(state, dict) and "internal_states" in state:
            state = state["internal_states"]
        
        if isinstance(state, np.ndarray):
            # Discretiza cada valor do array para um valor entre 0 e n_bins-1
            discrete_state = []
            for i, val in enumerate(state):
                # Normaliza o valor para o intervalo [0, n_bins-1]
                bin_idx = int(val * (self.n_bins / self.size))
                bin_idx = max(0, min(self.n_bins - 1, bin_idx))  # Clip para garantir limites
                discrete_state.append(bin_idx)
                
            # Converte o estado multi-dimensional para um índice único
            return np.ravel_multi_index(tuple(discrete_state), dims=[self.n_bins] * len(state))
            
        if isinstance(state, (int, np.integer)):
            # Se já for um índice, retorna direto
            return state
            
        raise ValueError(f"Formato de estado não suportado: {type(state)}")



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
