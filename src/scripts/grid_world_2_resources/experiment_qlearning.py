from ...agents.q_learning import QLearning
from ...gymnasium_env.envs.grid_world_2_resources import GridWorldEnv2Resources
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from datetime import datetime
import pygame
import os

def record_pygame_evaluation(agent, env, filepath="evaluation_video.mp4", num_episodes=5, fps=10):
    """
    Records the agent's evaluation in the environment and saves it as an MP4 file.
    
    Args:
        agent: The trained Q-learning agent
        env: The environment to evaluate in
        filepath: Path to save the MP4 file
        num_episodes: Number of episodes to record
        fps: Frames per second for the output video
    
    Returns:
        str: Path to the saved video file
    """
    try:
        import imageio
    except ImportError:
        print("Please install imageio: pip install imageio imageio-ffmpeg")
        return None
    
    # Create a temporary directory for frames
    temp_dir = f"temp_frames_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(temp_dir, exist_ok=True)
    
    
    # Initialize pygame if not already done
    if not pygame.get_init():
        pygame.init()
    
    frames = []
    frame_count = 0
    total_reward = 0
    
    print(f"Recording {num_episodes} episodes to {filepath}...")
    
    for episode in range(num_episodes):
        # Reset the environment
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        step_count = 0
        
        while not (done or truncated) and step_count < 500:  # Limit steps to prevent very long episodes
            # Process the state and select the best action
            if hasattr(agent, '_process_state'):
                state_idx = agent._process_state(state)
                action = np.argmax(agent.q_table[state_idx])
            else:
                # Fallback for agents without _process_state
                action = agent.get_action(state)
            
            # Take the action
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Render the environment
            env.render()
            
            # Capture the frame
            surface = pygame.display.get_surface()
            if surface is not None:
                # Convert pygame surface to numpy array
                frame = pygame.surfarray.array3d(surface)
                frame = np.transpose(frame, (1, 0, 2))  # Reorder dimensions for imageio
                
                # Save frame
                frames.append(frame)
                frame_path = os.path.join(temp_dir, f"frame_{frame_count:06d}.png")
                imageio.imwrite(frame_path, frame)
                frame_count += 1
            
            # Update state and rewards
            state = next_state
            episode_reward += reward
            step_count += 1
            
            # Small delay to make sure pygame renders correctly
            pygame.time.delay(30)
            
        total_reward += episode_reward
        print(f"Episode {episode+1}/{num_episodes} completed with reward: {episode_reward:.2f}")
    
    # Create the video from frames
    print(f"Creating video from {frame_count} frames...")
    
    try:
        # Use imageio to create the video
        writer = imageio.get_writer(filepath, fps=fps)
        for frame_idx in range(frame_count):
            frame_path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.png")
            if os.path.exists(frame_path):
                writer.append_data(imageio.imread(frame_path))
        writer.close()
        print(f"Video saved to {filepath}")
    except Exception as e:
        print(f"Error creating video: {e}")
    
    # Clean up temp directory
    try:
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)
        print(f"Cleaned up temporary files in {temp_dir}")
    except Exception as e:
        print(f"Error cleaning up temporary files: {e}")
    
    # Calculate average reward
    avg_reward = total_reward / num_episodes
    print(f"Evaluation complete: Average reward over {num_episodes} episodes: {avg_reward:.2f}")
    
    return filepath


# Modified test_trained_agent function to include recording
def test_trained_agent_with_recording(agent=None, wrapped_env=None, record=True, video_path="agent_evaluation.mp4"):
    """
    Function to test and record a trained agent.
    """
    # Initialize the environment with visualization for testing if not provided
    if wrapped_env is None:
        env = GridWorldEnv2Resources(
            config_path="config/config.yaml",
            drive_type="base",
            render_mode="human"
        )
        wrapped_env = GridWorldWrapper(env)
    
    # If no agent was provided, load one from a file
    if agent is None:
        initial_state_size = 100  # Same initial size used in training
        action_size = wrapped_env.action_space.n
        
        agent = CustomQLearning(
            state_size=initial_state_size,
            action_size=action_size
        )
        
        try:
            # Try to load the complete model first
            agent.load_model("qlearning_model.pkl")
            print("Model successfully loaded!")
        except FileNotFoundError:
            try:
                # If not found, try to load just the Q-table
                agent.load_q_table("q_table_resource_env.npy")
                print("Q-table successfully loaded!")
                print("States map not available.")
                agent._state_index_map = {}  # Initialize an empty map
            except FileNotFoundError:
                print("Train the agent first, no model was found.")
                return
    
    # Record the evaluation if requested
    if record:
        print(f"Recording evaluation to {video_path}...")
        video_file = record_pygame_evaluation(agent, wrapped_env, filepath=video_path, num_episodes=5)
        if video_file:
            print(f"Evaluation recorded and saved to {video_file}")
    else:
        # Evaluate the trained agent using the specialized evaluation method
        print("Evaluating the agent...")
        avg_reward = agent.evaluate(env=wrapped_env, num_episodes=5, render=True)
        print(f"Finished! Average reward: {avg_reward:.2f}")
        return avg_reward

# Example usage:
# test_trained_agent_with_recording(video_path="homeostatic_agent_evaluation.mp4")


class GridWorldWrapper(gym.Wrapper):
    """
    Wrapper para extrair os estados internos do dicionário de observação
    retornado pelo ambiente GridWorldEnv2Resources.
    """
    def __init__(self, env):
        super().__init__(env)
        # O espaço de observação agora será apenas o vetor de estados internos
        self.observation_space = self.env.observation_space["internal_states"]
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Retorna apenas o vetor de estados internos
        return obs["internal_states"], info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Retorna apenas o vetor de estados internos
        return obs["internal_states"], reward, terminated, truncated, info


class CustomQLearning(QLearning):
    """
    Versão customizada do QLearning para funcionar com o ambiente GridWorld.
    """
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        super().__init__(state_size, action_size, learning_rate, discount_factor,
                        epsilon, epsilon_min, epsilon_decay)
        
        # Dicionário para mapear estados para índices
        self._state_index_map = {}
        
    def get_action(self, state):
        """
        Implementação própria da seleção de ação com epsilon-greedy.
        
        Args:
            state: O estado atual do ambiente
            
        Returns:
            int: A ação selecionada
        """
        state_idx = self._process_state(state)
        
        # Implementação própria de epsilon-greedy
        if np.random.random() < self.epsilon:
            # Exploração: ação aleatória
            return np.random.randint(self.action_size)
        else:
            # Exploração: melhor ação conhecida
            return np.argmax(self.q_table[state_idx])
        
    def _process_state(self, state):
        """
        Processa o estado para ser usado como índice na Q-table.
        Agora lida corretamente com estados que são valores únicos ou arrays.
        
        Args:
            state: Estado do ambiente (pode ser valor único ou array)
            
        Returns:
            int: Um índice para a Q-table
        """
        # Se for dicionário, extrair os estados internos
        if isinstance(state, dict) and "internal_states" in state:
            state = state["internal_states"]
        
        # Verifica se o estado é um valor único
        if isinstance(state, (int, float, np.integer, np.floating)):
            # Se for um valor numérico único, usa diretamente como chave
            state_key = int(state)  # Converte para int para consistência
        else:
            # Se for um array ou lista, converte para tupla arredondada
            try:
                state_key = tuple(np.round(state, 2))
            except Exception as e:
                print(f"Error in processing the state: {e}")
                print(f"State type: {type(state)}")
                print(f"State: {state}")
                # Usa uma representação de string como fallback
                state_key = str(state)
        
        # Mapeia para um índice na Q-table
        if state_key not in self._state_index_map:
            self._state_index_map[state_key] = len(self._state_index_map)
            
            # Expandir Q-table se necessário
            if len(self._state_index_map) > self.q_table.shape[0]:
                new_size = max(self.q_table.shape[0] * 2, len(self._state_index_map))
                new_q_table = np.zeros((new_size, self.action_size))
                new_q_table[:self.q_table.shape[0], :] = self.q_table
                self.q_table = new_q_table
                print(f"Q-table expanded to {new_size} states")
                
        return self._state_index_map[state_key]
    
    def save_model(self, filepath):
        """
        Salva o modelo completo incluindo o mapa de estados.
        
        Args:
            filepath (str): Caminho para salvar o modelo
        """
        import pickle
        model_data = {
            'q_table': self.q_table,
            'state_index_map': self._state_index_map,
            'params': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon': self.epsilon,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model savec in {filepath}")
    
    def load_model(self, filepath):
        """
        Carrega o modelo completo de um arquivo.
        
        Args:
            filepath (str): Caminho do arquivo do modelo
        """
        import pickle
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_table = model_data['q_table']
        self._state_index_map = model_data['state_index_map']
        
        params = model_data['params']
        self.state_size = params['state_size']
        self.action_size = params['action_size']
        self.learning_rate = params['learning_rate']
        self.discount_factor = params['discount_factor']
        self.epsilon = params['epsilon']
        self.epsilon_min = params['epsilon_min']
        self.epsilon_decay = params['epsilon_decay']
        
        print(f"Model loaded from {filepath}")
    
    def evaluate(self, env, num_episodes=10, render=False):
        """
        Avalia o desempenho do agente treinado.
        Versão adaptada para usar o processamento de estado personalizado.
        
        Args:
            env: Ambiente de avaliação
            num_episodes (int, opcional): Número de episódios para avaliação
            render (bool, opcional): Se True, renderiza o ambiente durante a avaliação
            
        Returns:
            float: Recompensa média por episódio
        """
        total_rewards = []
        
        # Salva o epsilon atual e define como 0 para avaliação determinística
        original_epsilon = self.epsilon
        self.epsilon = 0
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            
            episode_reward = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                # Processa o estado para obter o índice
                state_idx = self._process_state(state)
                
                # Escolhe a melhor ação para este estado
                action = np.argmax(self.q_table[state_idx])
                
                # Executa a ação no ambiente
                next_state, reward, done, truncated, _ = env.step(action)
                
                if render:
                    env.render()
                
                # Atualiza o estado e a recompensa
                state = next_state
                episode_reward += reward
            
            total_rewards.append(episode_reward)
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")
        
        # Restaura o epsilon original
        self.epsilon = original_epsilon
        
        avg_reward = np.mean(total_rewards)
        print(f"Evaluating concluded with {num_episodes} episodes. Average reward: {avg_reward:.2f}")
        
        return avg_reward


def train_qlearning_agent():
    """
    Função para treinar o agente Q-Learning adaptado.
    """
    # Inicializa o ambiente sem renderização para treinar mais rápido
    env = GridWorldEnv2Resources(
        config_path="config/config.yaml",
        drive_type="base",
        render_mode=None
    )
    
    # Aplica o wrapper para simplificar as observações
    wrapped_env = GridWorldWrapper(env)
    
    # Inicializa o agente adaptado
    initial_state_size = 100  # Tamanho inicial que será expandido conforme necessário
    action_size = wrapped_env.action_space.n
    
    agent = CustomQLearning(
        state_size=initial_state_size,
        action_size=action_size,
        learning_rate=0.3,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )
    
    # Treina o agente
    num_episodes = 500  # Reduzido para testes iniciais
    rewards = agent.train(env=wrapped_env, num_episodes=num_episodes, max_steps_per_episode=200)
    
    # Salva o modelo completo
    agent.save_model("qlearning_model.pkl")
    # Também salva a Q-table para compatibilidade com código existente
    agent.save_q_table("q_table_resource_env.npy")
    
    # Imprime a recompensa média dos últimos episódios
    print(f"Treinamento concluído com recompensa média final: {np.mean(rewards[-10:]):.2f}")
    
    # Plota a evolução das recompensas
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Rewards by episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Adiciona uma linha de média móvel para visualizar tendência
    window_size = min(20, len(rewards)//5)
    if window_size > 1:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), moving_avg, 'r-', linewidth=2)
        plt.legend(['Reward by episode', f'Moving Average ({window_size} episodes)'])
    
    plt.savefig('rewards_plot.png')
    plt.close()
    
    # Informações sobre o treinamento
    print(f"Number of unique states found: {len(agent._state_index_map)}")
    print(f"Final size of Q-Table: {agent.q_table.shape}")
    
    return agent, rewards, wrapped_env


def test_trained_agent(agent=None, wrapped_env=None):
    """
    Função para testar um agente treinado.
    """
    # Inicializa o ambiente com visualização para teste, se não for fornecido
    if wrapped_env is None:
        env = GridWorldEnv2Resources(
            config_path="config/config.yaml",
            drive_type="base",
            render_mode="human"
        )
        wrapped_env = GridWorldWrapper(env)
    else:
        # Se o ambiente foi passado, ativa a renderização
        wrapped_env.env.render_mode = "human"
    
    # Se nenhum agente foi fornecido, carrega um do arquivo
    if agent is None:
        initial_state_size = 100  # Mesmo tamanho inicial usado no treinamento
        action_size = wrapped_env.action_space.n
        
        agent = CustomQLearning(
            state_size=initial_state_size,
            action_size=action_size
        )
        
        try:
            # Tenta carregar o modelo completo primeiro
            agent.load_model("qlearning_model.pkl")
            print("Model succesfully loaded!")
        except FileNotFoundError:
            try:
                # Se não encontrar, tenta carregar apenas a Q-table
                agent.load_q_table("q_table_resource_env.npy")
                print("Q-table successfully loaded!")
                print("States map not available.")
                agent._state_index_map = {}  # Inicializa um mapa vazio
            except FileNotFoundError:
                print("Train de agent first, no model was found.")
                return
    
    # Avalia o agente treinado usando o método de avaliação especializado
    print("Evaluating the agent...")
    avg_reward = agent.evaluate(env=wrapped_env, num_episodes=5, render=True)
    
    print(f"Finished! Average reward: {avg_reward:.2f}")
    return avg_reward


def test_experiment_qlearning():
    """
    Função principal que treina e então testa o agente Q-Learning.
    """
    # Treina o agente
    print("=== TRAINING ===")
    agent, rewards, wrapped_env = train_qlearning_agent()
    
    # # Testa o agente treinado
    # print("\n=== TESTING ===")
    # test_trained_agent(agent, wrapped_env)


if __name__ == "__main__":
    # Option 1: Train and then test with recording
    test_experiment_qlearning()  # Train the agent
    test_trained_agent_with_recording(video_path="homeostatic_agent.mp4")  # Record the evaluation
    
    # Option 2: Just record a previously trained agent
    # test_trained_agent_with_recording(video_path="homeostatic_agent.mp4")
