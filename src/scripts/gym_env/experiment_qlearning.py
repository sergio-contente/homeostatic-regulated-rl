import numpy as np
import pickle
import os
from src.gymnasium_env.envs.gridworld import GridWorldEnv


class QLearningAgent:
    def __init__(self, state_size, action_size, num_internal_states=3, bins_per_state=5, 
                 learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.995):
        """
        Inicializa o agente Q-learning
        
        Args:
            state_size: Tamanho do espaço de estados (posições)
            action_size: Tamanho do espaço de ações
            num_internal_states: Número de estados internos (food, water, energy, etc.)
            bins_per_state: Número de bins para discretizar cada estado interno
            learning_rate: Taxa de aprendizado (alpha)
            discount_factor: Fator de desconto para recompensas futuras (gamma)
            exploration_rate: Taxa de exploração inicial (epsilon)
            exploration_decay: Taxa de decaimento da exploração
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_internal_states = num_internal_states
        self.bins_per_state = bins_per_state
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = 0.01
        
        # Inicializa a tabela Q com valores zero
        q_table_shape = [state_size] + [bins_per_state] * num_internal_states + [action_size]
        self.q_table = np.zeros(q_table_shape)
        
    def discretize_state(self, internal_states):
        """
        Converte os valores contínuos dos estados internos em índices discretos para a tabela Q
        """
        bins = []
        for state_value in internal_states:
            bin_index = int((state_value + 1) / 2 * (self.bins_per_state - 1))
            bin_index = max(0, min(self.bins_per_state - 1, bin_index))
            bins.append(bin_index)
        return bins
    
    def get_q_value(self, position, internal_states_bins, action):
        """Obtém o valor Q para um estado e ação específicos"""
        index = [position] + internal_states_bins + [action]
        return self.q_table[tuple(index)]
    
    def set_q_value(self, position, internal_states_bins, action, value):
        """Define o valor Q para um estado e ação específicos"""
        index = [position] + internal_states_bins + [action]
        self.q_table[tuple(index)] = value
    
    def get_max_q(self, position, internal_states_bins):
        """Obtém o valor Q máximo para um estado"""
        index = [position] + internal_states_bins
        return np.max(self.q_table[tuple(index)])
    
    def choose_action(self, state):
        """
        Escolhe uma ação usando a política epsilon-greedy
        """
        position = state['position']
        internal_states_bins = self.discretize_state(state['internal_states'])
        
        # Exploração: escolher ação aleatória
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        
        # Exploração: escolher a melhor ação conhecida
        index = [position] + internal_states_bins
        return np.argmax(self.q_table[tuple(index)])
    
    def learn(self, state, action, reward, next_state, done):
        """
        Atualiza a tabela Q com base na experiência
        """
        position = state['position']
        internal_states_bins = self.discretize_state(state['internal_states'])
        
        next_position = next_state['position']
        next_internal_states_bins = self.discretize_state(next_state['internal_states'])
        
        # Obtém o valor Q atual
        current_q = self.get_q_value(position, internal_states_bins, action)
        
        # Fórmula do Q-learning
        if not done:
            next_max = self.get_max_q(next_position, next_internal_states_bins)
            target = reward + self.gamma * next_max
        else:
            target = reward
            
        # Atualiza o valor Q
        new_q = current_q + self.lr * (target - current_q)
        self.set_q_value(position, internal_states_bins, action, new_q)
        
        # Reduz a taxa de exploração
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath):
        """
        Salva o modelo do agente em um arquivo
        
        Args:
            filepath: Caminho para salvar o arquivo
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'state_size': self.state_size,
                'action_size': self.action_size,
                'num_internal_states': self.num_internal_states,
                'bins_per_state': self.bins_per_state,
                'lr': self.lr,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min
            }, f)
        print(f"Modelo salvo em {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Carrega um modelo de agente a partir de um arquivo
        
        Args:
            filepath: Caminho para o arquivo do modelo
            
        Returns:
            QLearningAgent: Instância do agente carregado
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        agent = cls(
            state_size=data['state_size'],
            action_size=data['action_size'],
            num_internal_states=data['num_internal_states'],
            bins_per_state=data['bins_per_state'],
            learning_rate=data['lr'],
            discount_factor=data['gamma'],
            exploration_rate=data['epsilon'],
            exploration_decay=data['epsilon_decay']
        )
        agent.q_table = data['q_table']
        agent.epsilon = data['epsilon']
        agent.epsilon_min = data['epsilon_min']
        
        return agent


def create_environment(config_path, drive_type, size=10, render_mode=None):
    """
    Cria o ambiente GridWorld
    
    Args:
        config_path: Caminho para o arquivo de configuração
        drive_type: Tipo de drive a ser usado
        size: Tamanho do grid
        render_mode: Modo de renderização ('human', 'rgb_array' ou None)
        
    Returns:
        GridWorldEnv: Instância do ambiente
    """
    return GridWorldEnv(
        render_mode=render_mode,
        config_path=config_path,
        drive_type=drive_type,
        size=size
    )


def create_agent(env, bins_per_state=5, learning_rate=0.1, discount_factor=0.95, 
                exploration_rate=1.0, exploration_decay=0.995):
    """
    Cria um agente Q-Learning
    
    Args:
        env: Ambiente GridWorld
        bins_per_state: Número de bins para discretizar cada estado interno
        learning_rate: Taxa de aprendizado
        discount_factor: Fator de desconto
        exploration_rate: Taxa de exploração inicial
        exploration_decay: Taxa de decaimento da exploração
        
    Returns:
        QLearningAgent: Instância do agente
    """
    return QLearningAgent(
        state_size=env.size,
        action_size=env.action_space.n,
        num_internal_states=env.dimension_internal_states,
        bins_per_state=bins_per_state,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        exploration_rate=exploration_rate,
        exploration_decay=exploration_decay
    )


def train_agent(agent, env, num_episodes=1000, max_steps=200, render_interval=100, save_path=None):
    """
    Treina o agente no ambiente
    
    Args:
        agent: Instância do agente Q-Learning
        env: Ambiente GridWorld
        num_episodes: Número de episódios de treinamento
        max_steps: Número máximo de passos por episódio
        render_interval: Intervalo de episódios para renderização
        save_path: Caminho para salvar o modelo (opcional)
        
    Returns:
        list: Lista de recompensas por episódio
    """
    episode_rewards = []
    best_avg_reward = float('-inf')
    
    for episode in range(num_episodes):
        # Reset do ambiente
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        # Define render_mode para "human" em alguns episódios para visualização
        original_render_mode = env.render_mode
        if episode % render_interval == 0:
            env.render_mode = "human"
        else:
            env.render_mode = None
            
        # Loop de um episódio
        for step in range(max_steps):
            # Escolhe a ação
            action = agent.choose_action(state)
            
            # Executa a ação
            next_state, reward, done, _, _ = env.step(action)
            
            # Agente aprende com a experiência
            agent.learn(state, action, reward, next_state, done)
            
            # Atualiza o estado e a recompensa total
            state = next_state
            total_reward += reward
            
            # Encerra o episódio se terminou
            if done:
                break
        
        # Restaura o modo de renderização original
        env.render_mode = original_render_mode
        
        # Registra a recompensa total do episódio
        episode_rewards.append(total_reward)
        
        # Imprime estatísticas a cada N episódios
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            print(f"Episódio {episode}/{num_episodes}, Recompensa média: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
            
            # Salva o melhor modelo se houver um caminho especificado
            if save_path and avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save(save_path)
                print(f"Novo melhor modelo salvo com recompensa média de {avg_reward:.2f}")
    
    # Salva o modelo final se não foi salvo durante o treinamento
    if save_path and (not os.path.exists(save_path) or episode == num_episodes - 1):
        agent.save(save_path)
    
    return episode_rewards


def test_agent(agent, env, num_episodes=5, max_steps=500):
    """
    Testa o agente no ambiente
    
    Args:
        agent: Instância do agente Q-Learning
        env: Ambiente GridWorld
        num_episodes: Número de episódios de teste
        max_steps: Número máximo de passos por episódio
        
    Returns:
        list: Lista de recompensas por episódio
    """
    # Desativa a exploração para avaliação
    original_epsilon = agent.epsilon
    agent.epsilon = 0
    
    test_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            total_reward += reward
            steps += 1
        
        test_rewards.append(total_reward)
        print(f"Episódio de teste {episode+1}/{num_episodes}: Recompensa: {total_reward:.2f}, Passos: {steps}")
    
    # Restaura o epsilon original
    agent.epsilon = original_epsilon
    
    avg_reward = np.mean(test_rewards)
    print(f"\nTeste concluído. Recompensa média: {avg_reward:.2f}")
    
    return test_rewards


def main_train():
    """Função para treinar um agente do zero"""
    config_path = "config/config.yaml"
    drive_type = "base_drive"
    model_path = "models/qlearning_agent.pkl"
    
    # Cria o ambiente e o agente com parâmetros ajustados
    env = create_environment(config_path, drive_type)
    agent = create_agent(
        env, 
        bins_per_state=10,  # Aumento de 5 para 10 bins por estado
        learning_rate=0.2,  # Aumento da taxa de aprendizado
        discount_factor=0.99,  # Aumento do fator de desconto
        exploration_rate=1.0,
        exploration_decay=0.99  # Decaimento mais lento para exploração
    )
    
    # Treina o agente por mais episódios e com mais passos por episódio
    rewards = train_agent(
        agent=agent,
        env=env,
        num_episodes=5000,  # Aumento de 1000 para 5000 episódios
        max_steps=500,      # Aumento de 200 para 500 passos por episódio
        render_interval=500,  # Renderiza menos frequentemente durante o treinamento
        save_path=model_path
    )
    
    print("\nTreinamento concluído!")
    print(f"Recompensa média dos últimos 100 episódios: {np.mean(rewards[-100:]):.2f}")
    env.close()

def main_test():
    """Função para testar um agente já treinado"""
    config_path = "config/config.yaml"
    drive_type = "base_drive"
    model_path = "models/qlearning_agent.pkl"
    
    # Verifica se o modelo existe
    if not os.path.exists(model_path):
        print(f"Erro: Modelo {model_path} não encontrado. Execute o treinamento primeiro.")
        return
    
    # Cria o ambiente e carrega o agente
    env = create_environment(config_path, drive_type, render_mode="human")
    agent = QLearningAgent.load(model_path)
    
    # Testa o agente
    test_agent(agent, env, num_episodes=5, max_steps=500)
    
    env.close()


if __name__ == "__main__":
    # Para treinar um novo agente, descomente a linha abaixo:
    # main_train()
    
    # Para testar um agente já treinado, descomente a linha abaixo:
    # main_test()
    
    # Alternativamente, você pode importar este módulo em outro arquivo
    # e chamar as funções main_train() e main_test() separadamente
    
    # Exemplo de uso como script independente:
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            print("Iniciando treinamento...")
            main_train()
        elif sys.argv[1] == "test":
            print("Iniciando teste...")
            main_test()
        else:
            print("Uso: python qlearning.py [train|test]")
    else:
        print("Uso: python qlearning.py [train|test]")
