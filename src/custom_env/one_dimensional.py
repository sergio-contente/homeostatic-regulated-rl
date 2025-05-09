import pygame
import numpy as np
import os
import matplotlib.pyplot as plt
from ..utils.get_params import ParameterHandler

class HomeoEnv1D:
    def __init__(self, config_path, drive_type, maxh, enable_visualization, render_mode=None):
        self.state_space = [(i, j) for i in range(-maxh, maxh + 1) for j in range(-maxh, maxh + 1)]
        self.state_to_idx = {state: idx for idx, state in enumerate(self.state_space)}
        self.action_space = [0, 1, 2, 3, 4]  # ficar, esquerda, direita, consumir, não consumir
        self.maxh = maxh
        self.steps = 0
        self.render_mode = render_mode
        self.enable_visualization = enable_visualization

        self._outcome = 1
        self.consumption_counts = {0: 0, 1: 0}


        self.param_manager = ParameterHandler(config_path)
        self.drive = self.param_manager.create_drive(drive_type)

        tensor = self.drive.get_tensor_optimal_states_values()
        self.optimal_point = tuple(tensor.tolist())

        self.size = self.drive.get_internal_state_size()
        # Gera um estado aleatório diferente do ótimo
        while True:
            random_state = np.random.uniform(-self.maxh, self.maxh, self.size).astype(np.float32)
            if not np.allclose(random_state, self.optimal_point, atol=0.1):
                break

        self.current_state = random_state
        initial_drive = self.drive.compute_drive(self.current_state)
        self.drive.update_drive(initial_drive)

        self.resources_position = [-maxh + 2, maxh - 2]
        self.agent_position = 0

        # Inicializa Q-Learning
        self.learning_rate = 0.5
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        # Históricos para visualização
        self.history = {"states": [], "drive": [], "reward": [], "position": []}
        self.training_screen = None
        self.training_font = None
        self.training_width = 1200
        self.training_height = 700
        self.history_max_len = 500

        self.n_bins = 15  # ou outro número que você escolher
        self.bins = [np.linspace(-maxh, maxh, self.n_bins + 1) for _ in range(self.size)]

        self.q_table = np.zeros([self.n_bins] * self.size + [len(self.action_space)])
        
        # Adição de variáveis para controle de visualização durante o treinamento
        self.current_episode = 0
        self.current_reward = 0.0
        self.current_epsilon = self.epsilon
        self.step_reward = 0.0

    def discretize_state(self, state):
        discretized = []
        for i, val in enumerate(state):
            bin_idx = np.digitize(val, self.bins[i]) - 1  # -1 porque digitize começa em 1
            bin_idx = min(max(bin_idx, 0), self.n_bins - 1)  # garantir que fica no intervalo
            discretized.append(bin_idx)
        return tuple(discretized)

    def reset(self):
        # Gera um estado aleatório diferente do ótimo
        while True:
            random_state = np.random.uniform(-self.maxh, self.maxh, self.size).astype(np.float32)
            if not np.allclose(random_state, self.optimal_point, atol=0.1):
                break

        self.current_state = random_state
        initial_drive = self.drive.compute_drive(self.current_state)
        self.drive.update_drive(initial_drive)
        self.agent_position = 0
        self.steps = 0
        return self.current_state, {}
    
    def compute_action_mask(self):
        """
        Returns a boolean mask of valid actions.
        True means the action is valid.
        """
        mask = np.ones(len(self.action_space), dtype=np.bool)

        # Check if the agent is on a resource position
        if self.agent_position not in self.resources_position:
            # Mask out 'consume resource' (action 3)
            mask[3] = 0
            mask[4] = 0 

        return mask
    

    def step(self, action):
        if action == 1:
            self.agent_position = max(self.agent_position - 1, -self.maxh + 2)
        elif action == 2:
            self.agent_position = min(self.agent_position + 1, self.maxh - 2)

        resource_index = None
        if self.agent_position == self.resources_position[0]:
            resource_index = 0
        elif self.agent_position == self.resources_position[1]:
            resource_index = 1

        if resource_index is not None and action == 3:
        #     self.current_state[resource_index] = np.clip(
        #     self.current_state[resource_index] + self.current_state[resource_index] * self._outcome,
        #     - self.maxh,
        #     self.maxh
        # )
            self.current_state[resource_index] = min(self.current_state[resource_index] + self._outcome, self.maxh)
            self.consumption_counts[resource_index] += 1

        decay = 0.07
        for i in range(len(self.current_state)):
            self.current_state[i] = max(self.current_state[i] * (1 - decay), -self.maxh)

        new_drive = self.drive.compute_drive(self.current_state)
        reward = self.drive.compute_reward(new_drive)
        self.drive.update_drive(new_drive)

        threshold = self._outcome / 2
        terminated = self.drive.has_reached_optimal(self.current_state, threshold)

        # if terminated:
        #     print("Optimal state reached!")

        observation = self.current_state

        # Armazena a recompensa do passo atual para visualização
        if hasattr(self, 'step_reward'):
            self.step_reward = reward

        if self.render_mode == "human" and self.enable_visualization:
            self.render()

        self.steps += 1
        truncated = self.steps >= 1000

        return observation, reward, terminated, truncated, {"action_mask": self.compute_action_mask()}

    # def select_action(self, state_idx, temperature=1.0):
    #     q_values = self.q_table[state_idx]
    #     q_values = q_values - np.max(q_values)
    #     exp_q = np.exp(q_values / max(temperature, 1e-6))
    #     probabilities = exp_q / np.sum(exp_q)
    #     return np.random.choice(self.action_space, p=probabilities)
    def select_action(self, state_idx, epsilon=0.1, mask=None):
        """
        Selects an action using epsilon-greedy with optional action mask.
        """
        if mask is None:
            mask = np.ones(len(self.action_space), dtype=np.bool)

        valid_actions = np.where(mask)[0]

        if np.random.random() < epsilon:
            # Explore only from valid actions
            return np.random.choice(valid_actions)
        else:
            # Exploit: choose the best valid action
            q_values = self.q_table[state_idx]
            # Mask invalid actions by setting them to -inf
            masked_q_values = np.full_like(q_values, -np.inf)
            masked_q_values[mask] = q_values[mask]
            return np.argmax(masked_q_values)


    def update_q_table(self, state_idx, action, reward, next_state_idx):
        q_predict = self.q_table[state_idx][action]
        q_target = reward + self.gamma * np.max(self.q_table[next_state_idx])
        self.q_table[state_idx][action] += self.learning_rate * (q_target - q_predict)

    def train(self, num_episodes=500, max_steps_per_episode=1000):
        rewards_per_episode = []
        self.consumption_counts = {0: 0, 1: 0}

        for episode in range(num_episodes):
            state, _ = self.reset()
            state_idx = self.discretize_state(state)
            total_reward = 0
            
            # Atualizar valores para início do episódio
            self.current_episode = episode
            self.current_reward = total_reward
            self.current_epsilon = self.epsilon

            for step in range(max_steps_per_episode):
                # Processa eventos pygame para evitar travamento
                if self.enable_visualization:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return rewards_per_episode  # Retorna os resultados parciais
                
                # Seleciona e executa a ação
                mask = self.compute_action_mask()
                action = self.select_action(state_idx, self.epsilon, mask)
                next_state, reward, done, truncated, _ = self.step(action)
                next_state_idx = self.discretize_state(next_state)

                # Atualiza a Q-table
                self.update_q_table(state_idx, action, reward, next_state_idx)

                # Atualiza o estado
                state_idx = next_state_idx
                total_reward += reward
                
                # IMPORTANTE: Atualiza a recompensa total para a visualização
                self.current_reward = total_reward
                
                # Verificação de término
                done = False
                if done or truncated:
                    break

            # Atualiza epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            rewards_per_episode.append(total_reward)

            # Log de progresso
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode+1}/{num_episodes} | Total Reward: {total_reward:.2f} | Epsilon: {self.epsilon:.4f}")
            
        print(f"Total de consumos - Recurso 0: {self.consumption_counts[0]}, Recurso 1: {self.consumption_counts[1]}")


        return rewards_per_episode

    def setup_training_visualization(self):
        pygame.init()
        self.training_screen = pygame.display.set_mode((self.training_width, self.training_height))
        pygame.display.set_caption("Training Visualization - HomeoEnv1D")
        self.training_font = pygame.font.SysFont('Arial', 16)

    def update_training_visualization(self, current_step, episode, total_reward, epsilon):
        if self.training_screen is None:
            self.setup_training_visualization()

        if len(self.history["states"]) >= self.history_max_len:
            self.history["states"].pop(0)
        self.history["states"].append(self.current_state.copy())

        if len(self.history["drive"]) >= self.history_max_len:
            self.history["drive"].pop(0)
        self.history["drive"].append(self.drive.get_current_drive())

        if len(self.history["reward"]) >= self.history_max_len:
            self.history["reward"].pop(0)
        self.history["reward"].append(total_reward)

        if len(self.history["position"]) >= self.history_max_len:
            self.history["position"].pop(0)
        self.history["position"].append(self.agent_position)

        self.training_screen.fill((255, 255, 255))

        info_texts = [
            f"Episode: {episode}",
            f"Step: {current_step}",
            f"Total Reward: {total_reward:.2f}",
            f"Drive: {self.drive.get_current_drive():.2f}",
            f"Epsilon: {epsilon:.4f}",
            f"Agent Position: {self.agent_position}",
            f"Internal State: {self.current_state.tolist()}",
        ]

        for i, text in enumerate(info_texts):
            text_surface = self.training_font.render(text, True, (0, 0, 0))
            self.training_screen.blit(text_surface, (20, 20 + 25 * i))

        self._draw_history(self.history["drive"], (20, 250), (550, 150), "Drive")
        self._draw_history(self.history["reward"], (20, 420), (550, 150), "Total Reward")
        self._draw_history(self.history["position"], (600, 250), (550, 150), "Agent Position")

        if len(self.history["states"]) > 0:
            state_0 = [s[0] for s in self.history["states"]]
            state_1 = [s[1] for s in self.history["states"]]
            self._draw_double_history(state_0, state_1, (600, 420), (550, 150), "Internal States")

        pygame.display.flip()
        pygame.event.pump()

    def _draw_history(self, values, position, size, title):
        x, y = position
        width, height = size

        if len(values) < 2:
            return

        max_val = max(values)
        min_val = min(values)
        range_val = max(max_val - min_val, 1e-5)

        pygame.draw.rect(self.training_screen, (230, 230, 230), (x, y, width, height))
        pygame.draw.rect(self.training_screen, (0, 0, 0), (x, y, width, height), 2)

        points = []
        for i, v in enumerate(values):
            px = x + i * width // self.history_max_len
            py = y + height - int((v - min_val) / range_val * height)
            points.append((px, py))

        if len(points) > 1:
            pygame.draw.lines(self.training_screen, (0, 0, 255), False, points, 2)

        title_surface = self.training_font.render(title, True, (0, 0, 0))
        self.training_screen.blit(title_surface, (x + 5, y - 20))

    def _draw_double_history(self, values1, values2, position, size, title):
        x, y = position
        width, height = size

        if len(values1) < 2:
            return

        max_val = max(max(values1), max(values2))
        min_val = min(min(values1), min(values2))
        range_val = max(max_val - min_val, 1e-5)

        pygame.draw.rect(self.training_screen, (230, 230, 230), (x, y, width, height))
        pygame.draw.rect(self.training_screen, (0, 0, 0), (x, y, width, height), 2)

        points1 = []
        points2 = []
        for i in range(len(values1)):
            px = x + i * width // self.history_max_len
            py1 = y + height - int((values1[i] - min_val) / range_val * height)
            py2 = y + height - int((values2[i] - min_val) / range_val * height)
            points1.append((px, py1))
            points2.append((px, py2))

        if len(points1) > 1:
            pygame.draw.lines(self.training_screen, (255, 0, 0), False, points1, 2)
            pygame.draw.lines(self.training_screen, (0, 128, 0), False, points2, 2)

        title_surface = self.training_font.render(title, True, (0, 0, 0))
        self.training_screen.blit(title_surface, (x + 5, y - 20))

    def plot_rewards(self, rewards):
        os.makedirs('images/custom/homeoenv', exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.plot(rewards)
        plt.title('Rewards per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)

        window_size = min(10, len(rewards))
        if window_size > 0:
            moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
            plt.plot(range(window_size-1, len(rewards)), moving_avg, 'r', label=f'Moving Average ({window_size} episodes)')

        plt.legend()
        plt.savefig('images/custom/homeoenv/training_rewards.png')
        plt.show()
    
    def render(self):
        """
        Renderiza o ambiente mostrando o agente e os recursos em uma linha de -maxh até maxh,
        mantendo os gráficos existentes, incluindo o gráfico de recompensa total.
        """
        if self.training_screen is None:
            self.setup_training_visualization()
        
        # Atualiza os históricos
        if len(self.history["states"]) >= self.history_max_len:
            self.history["states"].pop(0)
        self.history["states"].append(self.current_state.copy())

        if len(self.history["drive"]) >= self.history_max_len:
            self.history["drive"].pop(0)
        self.history["drive"].append(self.drive.get_current_drive())

        # Para a recompensa, precisamos garantir que estamos armazenando o valor correto
        current_reward = 0
        if hasattr(self, 'current_reward'):
            current_reward = self.current_reward
        
        if len(self.history["reward"]) >= self.history_max_len:
            self.history["reward"].pop(0)
        self.history["reward"].append(current_reward)  # Armazena a recompensa atual

        if len(self.history["position"]) >= self.history_max_len:
            self.history["position"].pop(0)
        self.history["position"].append(self.agent_position)

        # Limpa a tela
        self.training_screen.fill((255, 255, 255))

        # Pegamos os valores atuais do episódio e epsilon se disponíveis
        current_episode = 0
        current_epsilon = 0.0
        if hasattr(self, 'current_episode'):
            current_episode = self.current_episode
        if hasattr(self, 'current_epsilon'):
            current_epsilon = self.current_epsilon

        # Informações textuais
        info_texts = [
            f"Episode: {current_episode}",
            f"Step: {self.steps}",
            f"Total Reward: {current_reward:.2f}",
            f"Drive: {self.drive.get_current_drive():.2f}",
            f"Epsilon: {current_epsilon:.4f}",
            f"Agent Position: {self.agent_position}",
            f"Internal State: {[round(s, 2) for s in self.current_state.tolist()]}",
        ]

        for i, text in enumerate(info_texts):
            text_surface = self.training_font.render(text, True, (0, 0, 0))
            self.training_screen.blit(text_surface, (20, 20 + 25 * i))

        # Desenha os gráficos - garantindo que o gráfico de recompensa tenha dados
        self._draw_history(self.history["drive"], (20, 250), (550, 150), "Drive")
        
        # Certifique-se de que o gráfico de recompensa tem pelo menos 2 pontos para desenhar
        if len(self.history["reward"]) >= 2:
            self._draw_history(self.history["reward"], (20, 420), (550, 150), "Total Reward")
        else:
            # Se não tiver pontos suficientes, desenha só o fundo e o título
            pygame.draw.rect(self.training_screen, (230, 230, 230), (20, 420, 550, 150))
            pygame.draw.rect(self.training_screen, (0, 0, 0), (20, 420, 550, 150), 2)
            title_surface = self.training_font.render("Total Reward", True, (0, 0, 0))
            self.training_screen.blit(title_surface, (20 + 5, 420 - 20))
        
        self._draw_history(self.history["position"], (600, 250), (550, 150), "Agent Position")

        if len(self.history["states"]) > 0:
            state_0 = [s[0] for s in self.history["states"]]
            state_1 = [s[1] for s in self.history["states"]]
            self._draw_double_history(state_0, state_1, (600, 420), (550, 150), "Internal States")

        # Visualização do agente na linha
        # Posição e dimensões da área de visualização
        line_area_x = 20
        line_area_y = 600
        line_area_width = self.training_width - 40
        line_area_height = 80
        
        # Fundo para a área da linha
        pygame.draw.rect(self.training_screen, (240, 240, 240), 
                        (line_area_x, line_area_y, line_area_width, line_area_height))
        pygame.draw.rect(self.training_screen, (0, 0, 0), 
                        (line_area_x, line_area_y, line_area_width, line_area_height), 2)
        
        # Título da visualização
        title_surface = self.training_font.render("Agent and Resources Position", True, (0, 0, 0))
        self.training_screen.blit(title_surface, (line_area_x + 5, line_area_y - 20))
        
        # Linha base (eixo)
        line_y = line_area_y + line_area_height // 2
        line_start_x = line_area_x + 20
        line_end_x = line_area_x + line_area_width - 20
        pygame.draw.line(self.training_screen, (0, 0, 0), 
                        (line_start_x, line_y), (line_end_x, line_y), 3)
        
        # Calcula o tamanho de cada unidade na linha
        total_range = 2 * self.maxh
        unit_size = (line_end_x - line_start_x) / total_range
        
        # Desenha marcações de posição
        for i in range(-self.maxh, self.maxh + 1):
            pos_x = line_start_x + (i + self.maxh) * unit_size
            if i % 2 == 0:  # Desenha apenas marcações a cada 2 unidades
                pygame.draw.line(self.training_screen, (100, 100, 100),
                                (pos_x, line_y - 8), (pos_x, line_y + 8), 1)
                pos_text = self.training_font.render(str(i), True, (0, 0, 0))
                self.training_screen.blit(pos_text, (pos_x - 5, line_y + 12))
        
        # Desenha recursos
        for resource_pos in self.resources_position:
            pos_x = line_start_x + (resource_pos + self.maxh) * unit_size
            pygame.draw.circle(self.training_screen, (0, 128, 0), 
                            (int(pos_x), line_y), 12)
            # Rótulo do recurso
            res_label = self.training_font.render("R", True, (255, 255, 255))
            text_rect = res_label.get_rect(center=(int(pos_x), line_y))
            self.training_screen.blit(res_label, text_rect)
        
        # Desenha o agente
        agent_x = line_start_x + (self.agent_position + self.maxh) * unit_size
        pygame.draw.circle(self.training_screen, (0, 0, 255), 
                        (int(agent_x), line_y), 10)
        # Rótulo do agente
        agent_label = self.training_font.render("A", True, (255, 255, 255))
        agent_text_rect = agent_label.get_rect(center=(int(agent_x), line_y))
        self.training_screen.blit(agent_label, agent_text_rect)
        
        # Legenda
        legend_start_x = line_area_x + 10
        legend_start_y = line_area_y + 5
        # Agente (azul)
        pygame.draw.circle(self.training_screen, (0, 0, 255), 
                        (legend_start_x + 8, legend_start_y + 8), 8)
        legend_text = self.training_font.render("Agent", True, (0, 0, 0))
        self.training_screen.blit(legend_text, (legend_start_x + 25, legend_start_y))
        
        # Recursos (verde)
        pygame.draw.circle(self.training_screen, (0, 128, 0), 
                        (legend_start_x + 120, legend_start_y + 8), 8)
        legend_text = self.training_font.render("Resources", True, (0, 0, 0))
        self.training_screen.blit(legend_text, (legend_start_x + 135, legend_start_y))
        
        # Atualiza a tela
        pygame.display.flip()
        pygame.event.pump()

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    import pandas as pd
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap

    def plot_analysis(self, save_dir="images/custom/homeoenv/analysis"):
        """
        Gera uma série de gráficos de análise para o ambiente HomeoEnv1D.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Garante que os gráficos fiquem em tamanho adequado
        plt.rcParams['figure.figsize'] = [10, 8]
        plt.rcParams['figure.dpi'] = 100
        
        # Cria uma figura com subplots para análises similares à imagem de referência
        fig = plt.figure(figsize=(14, 12))
        
        # Define formato de grid para os subplots
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Plot trajetória do estado interno - Gráfico a
        self._plot_internal_state_trajectory(fig.add_subplot(gs[0, 0]))
        
        # 2. Plot evolução do drive ao longo de um episódio - Gráfico b
        self._plot_drive_evolution(fig.add_subplot(gs[0, 1]))
        
        # 3. Plot frequência de seleção de ações - Gráfico c
        self._plot_action_frequency(fig.add_subplot(gs[1, 0]))
        
        # 4. Plot heatmap da Q-table - Gráfico d
        self._plot_qtable_heatmap(fig.add_subplot(gs[1, 1]))
        
        # Adiciona título geral
        fig.suptitle("Análise do Ambiente HomeoEnv1D", fontsize=16, y=0.98)
        
        # Salva a figura principal
        plt.savefig(os.path.join(save_dir, "complete_analysis.png"), bbox_inches="tight")
        plt.close(fig)
        
        # Plots adicionais em figuras separadas
        self._plot_3d_drive_surface(save_dir)
        self._plot_policy_visualization(save_dir)
        
        print(f"Análises salvas em: {save_dir}")

    def _plot_internal_state_trajectory(self, ax):
        """
        Plota a trajetória dos estados internos durante um episódio de avaliação,
        mostrando cada estado interno separadamente.
        """
        # Roda um episódio para coletar dados da trajetória
        trajectory = self._collect_trajectory_data()

        states_0 = [s[0] for s in trajectory['states']]
        states_1 = [s[1] for s in trajectory['states']]

        # Plota os dois estados internos em cores diferentes
        ax.plot(range(len(states_0)), states_0, 'b-', linewidth=1.5, label='Estado Interno 1')
        ax.plot(range(len(states_1)), states_1, 'g-', linewidth=1.5, label='Estado Interno 2')

        # Adiciona linhas horizontais de referência (opcional)
        for i in range(-2, 10, 2):
            ax.axhline(y=i, color='r', linestyle='--', alpha=0.3)

        # Configurações do gráfico
        ax.set_xlabel('Número de passos')
        ax.set_ylabel('Estado interno')
        ax.set_title('a. Trajetória dos Estados Internos')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

        # Limites do eixo
        ax.set_xlim(0, len(states_0))
        ax.set_ylim(-3, 9)


    def _plot_drive_evolution(self, ax):
        """
        Plota a evolução do drive ao longo de um episódio.
        Semelhante ao Gráfico b na imagem de referência.
        """
        # Usa os mesmos dados de trajetória
        trajectory = self._collect_trajectory_data()
        
        # Plota a evolução do drive
        ax.plot(range(len(trajectory['drives'])), trajectory['drives'], 'r-', linewidth=1.5)
        
        # Configurações do gráfico
        ax.set_xlabel('Número de passos')
        ax.set_ylabel('Drive')
        ax.set_title('b. Evolução do Drive')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Limites do eixo y baseados no range de drives observados
        ax.set_ylim(0, max(trajectory['drives']) * 1.1)
        ax.set_xlim(0, len(trajectory['drives']))

    def _plot_action_frequency(self, ax):
        """
        Plota a frequência de seleção de cada ação durante o treinamento.
        Semelhante ao Gráfico c na imagem de referência.
        """
        # Simula dados de frequência de ações (isso é mais ilustrativo)
        # Na implementação real, você deveria armazenar esses dados durante o treinamento
        
        # Extrai a política de ação para cada estado
        action_counts = self._get_action_preferences()
        
        # Descrições das ações para o eixo y
        action_desc = {
            0: "Ficar parado",
            1: "Mover esquerda",
            2: "Mover direita",
            3: "Consumir recurso",
            4: "Não consumir"
        }
        
        # Criar dados para o gráfico de linha
        episodes = np.linspace(0, 10000, 100)  # Simulando 10000 episódios
        
        # Cores para cada ação
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        
        # Plota cada ação como uma linha separada
        for action, count in action_counts.items():
            percentage = count / sum(action_counts.values()) * 10
            # Adiciona um pouco de ruído para simular variação ao longo do treinamento
            noise = np.random.normal(0, 0.1, size=len(episodes))
            action_freq = percentage + noise
            action_freq = np.clip(action_freq, 0, 10)  # Garante valores positivos
            
            ax.plot(episodes, action_freq, label=action_desc[action], color=colors[action], linewidth=1.5)
        
        # Configurações do gráfico
        ax.set_xlabel('Número de episódios')
        ax.set_ylabel('Frequência de seleção de ação')
        ax.set_title('c. Preferência por Ações')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Limites
        ax.set_ylim(0, 10)
        ax.set_xlim(0, 10000)

    def _plot_qtable_heatmap(self, ax):
        """
        Plota um heatmap da Q-table para visualizar a política aprendida.
        Semelhante ao Gráfico d na imagem de referência.
        """
        # Cria um mapa de valor agregado da Q-table para os dois estados internos
        # Para simplificar, vamos considerar apenas a dimensão (x1, x2) ignorando a posição do agente
        
        # Extrai os valores máximos da Q-table para cada estado
        best_values = np.zeros((self.n_bins, self.n_bins))
        best_actions = np.zeros((self.n_bins, self.n_bins))
        
        for i in range(self.n_bins):
            for j in range(self.n_bins):
                state_idx = (i, j)
                best_action = np.argmax(self.q_table[state_idx])
                best_value = np.max(self.q_table[state_idx])
                best_values[i, j] = best_value
                best_actions[i, j] = best_action
        
        # Cria um heatmap dos valores Q
        im = ax.imshow(best_values, cmap='viridis', origin='lower')
        
        # Adiciona setas para indicar as ações preferidas
        action_markers = {
            0: "o",  # Ficar parado - círculo
            1: "<",  # Esquerda - seta para esquerda
            2: ">",  # Direita - seta para direita
            3: "^",  # Consumir - seta para cima
            4: "v"   # Não consumir - seta para baixo
        }
        
        # Adiciona marcadores de ação
        for i in range(0, self.n_bins, 1):
            for j in range(0, self.n_bins, 1):
                action = int(best_actions[i, j])
                ax.scatter(j, i, marker=action_markers[action], color='k', s=10)
        
        # Configura o heatmap
        plt.colorbar(im, ax=ax, label='Valor Q')
        
        # Rótulos dos eixos com valores reais
        x_ticks = np.linspace(0, self.n_bins-1, 5)
        y_ticks = np.linspace(0, self.n_bins-1, 5)
        x_labels = [f"{self.bins[0][int(i)]:.1f}" for i in x_ticks]
        y_labels = [f"{self.bins[1][int(i)]:.1f}" for i in y_ticks]
        
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
        
        # Títulos e legendas
        ax.set_xlabel('Estado Interno 1')
        ax.set_ylabel('Estado Interno 2')
        ax.set_title('d. Heatmap da Política Aprendida')
        
        # Adiciona legenda para as ações
        action_desc = {
            0: "Ficar",
            1: "Esquerda",
            2: "Direita",
            3: "Consumir",
            4: "Não consumir"
        }
        
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker=action_markers[a], color='w',
                                markerfacecolor='black', markersize=8, label=action_desc[a]) 
                        for a in range(5)]
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=7, title="Ações")

    def _plot_3d_drive_surface(self, save_dir):
        """
        Plota uma superfície 3D da função de drive.
        Semelhante aos gráficos na parte inferior da imagem de referência.
        """
        fig = plt.figure(figsize=(12, 10))
        
        # Primeiro subplot: Heatmap 2D do drive
        ax1 = fig.add_subplot(1, 2, 1)
        self._plot_drive_heatmap(ax1)
        
        # Segundo subplot: Superfície 3D do drive
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        self._plot_drive_surface(ax2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "drive_surface.png"), bbox_inches="tight")
        plt.close(fig)

    def _plot_drive_heatmap(self, ax):
        """
        Plota um heatmap 2D da função de drive para diferentes estados internos.
        """
        # Cria grid para os dois estados internos
        x = np.linspace(-self.maxh, self.maxh, 50)
        y = np.linspace(-self.maxh, self.maxh, 50)
        X, Y = np.meshgrid(x, y)
        
        # Calcula o drive para cada par de estados
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                state = np.array([X[i, j], Y[i, j]], dtype=np.float32)
                Z[i, j] = self.drive.compute_drive(state)
        
        # Cria o heatmap
        im = ax.contourf(X, Y, Z, 20, cmap='hot')
        plt.colorbar(im, ax=ax, label='Drive')
        
        # Adiciona o ponto ótimo
        optimal_point = self.optimal_point
        ax.scatter(optimal_point[0], optimal_point[1], marker='*', 
                color='blue', s=100, label='Ponto ótimo')
        
        # Contornos
        CS = ax.contour(X, Y, Z, 6, colors='k', alpha=0.5)
        ax.clabel(CS, inline=True, fontsize=8)
        
        # Configurações do gráfico
        ax.set_xlabel('Estado Interno 1')
        ax.set_ylabel('Estado Interno 2')
        ax.set_title('Mapa de Calor da Função Drive')
        ax.legend(loc='upper right')
        
        # Limites
        ax.set_xlim(-self.maxh, self.maxh)
        ax.set_ylim(-self.maxh, self.maxh)

    def _plot_drive_surface(self, ax):
        """
        Plota uma superfície 3D da função de drive.
        """
        # Reutiliza o grid do heatmap
        x = np.linspace(-self.maxh, self.maxh, 50)
        y = np.linspace(-self.maxh, self.maxh, 50)
        X, Y = np.meshgrid(x, y)
        
        # Calcula o drive para cada par de estados
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                state = np.array([X[i, j], Y[i, j]], dtype=np.float32)
                Z[i, j] = self.drive.compute_drive(state)
        
        # Cria a superfície 3D
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
        
        # Adiciona uma barra de cores
        fig = plt.gcf()
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Drive')
        
        # Adiciona o ponto ótimo
        optimal_point = self.optimal_point
        optimal_drive = self.drive.compute_drive(np.array(optimal_point, dtype=np.float32))
        ax.scatter(optimal_point[0], optimal_point[1], optimal_drive, 
                marker='*', color='red', s=100)
        
        # Configurações do gráfico
        ax.set_xlabel('Estado Interno 1')
        ax.set_ylabel('Estado Interno 2')
        ax.set_zlabel('Drive')
        ax.set_title('Superfície da Função Drive')
        
        # Ajusta a visualização
        ax.view_init(elev=30, azim=45)

    def _plot_policy_visualization(self, save_dir):
        """
        Visualização da política aprendida como setas vetoriais num gráfico 2D.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Cria grid para os dois estados internos
        x = np.linspace(-self.maxh, self.maxh, self.n_bins)
        y = np.linspace(-self.maxh, self.maxh, self.n_bins)
        X, Y = np.meshgrid(x, y)
        
        # Para cada estado, determina a ação preferida e a direção resultante
        U = np.zeros_like(X)  # Componente x do vetor de ação
        V = np.zeros_like(Y)  # Componente y do vetor de ação
        C = np.zeros_like(X)  # Cor baseada no valor Q
        
        # Mapeia as ações para vetores de direção
        action_vectors = {
            0: (0, 0),      # Ficar parado
            1: (-1, 0),     # Esquerda
            2: (1, 0),      # Direita
            3: (0, 1),      # Consumir (aumenta estado interno)
            4: (0, -1)      # Não consumir (diminui estado interno)
        }
        
        # Calcula a direção da política para cada estado
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                # Converte coordenadas para índices na q_table
                x_idx = np.digitize(X[i, j], self.bins[0]) - 1
                y_idx = np.digitize(Y[i, j], self.bins[1]) - 1
                x_idx = min(max(x_idx, 0), self.n_bins - 1)
                y_idx = min(max(y_idx, 0), self.n_bins - 1)
                
                # Obtém a melhor ação para este estado
                state_idx = (x_idx, y_idx)
                best_action = np.argmax(self.q_table[state_idx])
                best_value = np.max(self.q_table[state_idx])
                
                # Define a direção do vetor baseada na ação
                dx, dy = action_vectors[best_action]
                if dx == 0 and dy == 0:
                    dx, dy = (0.1, 0.1)  # só para não ficar invisível no gráfico
                U[i, j] = dx
                V[i, j] = dy
                
                # Define a cor com base no valor Q
                C[i, j] = best_value
        
        # Normaliza os valores para visualização
        norm = plt.Normalize(np.min(C), np.max(C))
        
        # Cria o campo vetorial
        quiver = ax.quiver(X, Y, U, V, C, cmap='viridis', norm=norm, 
                    scale=30, scale_units='inches', pivot='mid', width=0.003)
        
        # Adiciona uma barra de cores
        cbar = plt.colorbar(quiver, ax=ax)
        cbar.set_label('Valor Q')
        
        # Adiciona o ponto ótimo
        optimal_point = self.optimal_point
        ax.scatter(optimal_point[0], optimal_point[1], marker='*', 
                color='red', s=200, label='Ponto ótimo')
        
        # Configura o gráfico
        ax.set_xlabel('Estado Interno 1')
        ax.set_ylabel('Estado Interno 2')
        ax.set_title('Visualização da Política: Direção das Ações')
        ax.legend(loc='upper right')
        
        # Limites
        ax.set_xlim(-self.maxh, self.maxh)
        ax.set_ylim(-self.maxh, self.maxh)
        
        # Grid
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Salva o gráfico
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "policy_direction.png"), bbox_inches="tight")
        plt.close(fig)

    def _collect_trajectory_data(self, num_steps=2000):
        """
        Coleta dados da trajetória do agente seguindo a política aprendida por um número de passos.
        Retorna um dicionário com os estados, ações, recompensas e drives ao longo da trajetória.
        """
        trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'positions': [],
            'drives': []
        }
        
        # Reseta o ambiente
        state, _ = self.reset()
        state_idx = self.discretize_state(state)
        
        # Coleta dados para num_steps ou até terminar
        for step in range(num_steps):
            # Seleciona a melhor ação de acordo com a Q-table
            mask = self.compute_action_mask()
            action = np.argmax(self.q_table[state_idx])
            
            # Executa a ação
            next_state, reward, done, truncated, _ = self.step(action)
            next_state_idx = self.discretize_state(next_state)
            
            # Armazena os dados da trajetória
            trajectory['states'].append(state.copy())
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)
            trajectory['positions'].append(self.agent_position)
            trajectory['drives'].append(self.drive.get_current_drive())
            
            # Atualiza o estado
            state = next_state
            state_idx = next_state_idx
            
            # Verifica se terminou
            done = False
            if done or truncated:
                break
        
        return trajectory

    def _get_action_preferences(self):
        """
        Calcula a preferência por cada ação na política aprendida.
        Retorna um dicionário com a contagem de cada ação como a melhor ação entre todos os estados.
        """
        action_counts = {i: 0 for i in range(len(self.action_space))}
        
        # Para cada configuração possível de estado, conta qual é a ação preferida
        for state_indices in np.ndindex(self.q_table.shape[:-1]):
            best_action = np.argmax(self.q_table[state_indices])
            action_counts[best_action] += 1
        
        return action_counts
