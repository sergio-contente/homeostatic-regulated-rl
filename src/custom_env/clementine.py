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
            truncated = self.steps >= 1000  # Limita o número de passos por episódio
            
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
        """Processa o estado para uso na tabela Q."""
        # Se for dict vindo do ambiente original
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
        """Renderiza o estado atual do ambiente usando pygame."""
        if not hasattr(self, 'screen'):
            pygame.init()
            self.screen_width = 600
            self.screen_height = 600
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Clementine Environment")
            self.clock = pygame.time.Clock()
            
        # Limpa a tela
        self.screen.fill((255, 255, 255))
        
        # Desenha o estado atual
        center_x, center_y = self.screen_width // 2, self.screen_height // 2
        grid_size = min(self.screen_width, self.screen_height) - 100
        cell_size = grid_size // (2 * self.maxh + 1)
        
        # Desenha a grade
        for i in range(-self.maxh, self.maxh + 1):
            for j in range(-self.maxh, self.maxh + 1):
                rect_x = center_x + i * cell_size - cell_size // 2
                rect_y = center_y + j * cell_size - cell_size // 2
                pygame.draw.rect(self.screen, (200, 200, 200), 
                                 (rect_x, rect_y, cell_size, cell_size), 1)
                
                # Marca o ponto ótimo
                if i == 0 and j == 0:
                    pygame.draw.rect(self.screen, (0, 255, 0), 
                                     (rect_x, rect_y, cell_size, cell_size))
        
        # Desenha o agente na posição atual
        agent_x = center_x + self.current_state[0] * cell_size - cell_size // 2
        agent_y = center_y + self.current_state[1] * cell_size - cell_size // 2
        pygame.draw.circle(self.screen, (255, 0, 0), 
                          (agent_x + cell_size // 2, agent_y + cell_size // 2), 
                          cell_size // 2)
        
        # Atualiza a tela
        pygame.display.flip()
        self.clock.tick(30)
        pygame.event.pump()


def plot_rewards(rewards):
    """Plota a curva de recompensas do treinamento."""
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title('Recompensas por Episódio')
    plt.xlabel('Episódio')
    plt.ylabel('Recompensa Total')
    plt.grid(True)
    
    # Adiciona a média móvel
    window_size = min(10, len(rewards))
    if window_size > 0:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), moving_avg, 'r', label=f'Média Móvel ({window_size} episódios)')
    
    plt.legend()
    plt.savefig('training_rewards.png')
    plt.show()


def main():
    """Função principal para treinamento e avaliação do agente Clementine."""
    # Parâmetros
    config_path = "config/config.yaml"  # Ajuste para o caminho correto do seu arquivo de configuração
    drive_type = "base_drive"  # Tipo de drive a ser usado
    render_mode = None  # Defina como "human" para visualizar o treinamento
    num_episodes = 1000  # Número de episódios para treinamento
    eval_episodes = 10  # Número de episódios para avaliação
    
    print("Iniciando ambiente Clementine...")
    env = Clementine(config_path, drive_type, render_mode=render_mode)
    
    print(f"\nIniciando treinamento por {num_episodes} episódios...")
    start_time = time.time()
    rewards = env.train(num_episodes)
    training_time = time.time() - start_time
    print(f"Treinamento concluído em {training_time:.2f} segundos!")
    
    # Plotar a curva de recompensas
    plot_rewards(rewards)
    
    # Avaliação com renderização
    print(f"\nAvaliando o agente treinado por {eval_episodes} episódios...")
    # Cria uma nova instância para avaliação com renderização
    eval_env = Clementine(config_path, drive_type, render_mode="human")
    # Copia a tabela Q do agente treinado
    eval_env.agent.q_table = env.agent.q_table.copy()
    eval_env.agent.epsilon = 0.0  # Desativa a exploração durante a avaliação
    
    avg_reward = eval_env.evaluate(num_episodes=eval_episodes, render='human')
    print(f"Avaliação concluída! Recompensa média: {avg_reward:.2f}")
    
    # Fecha pygame se estiver ativo
    if render_mode == "human" or eval_env.render_mode == "human":
        pygame.quit()


if __name__ == "__main__":
    main()
