# Transforma o estado atual em um formato aceitável para o Q-learning
"""
Módulo com funções para visualização em tempo real do treinamento
"""
import pygame
import numpy as np

def setup_training_visualization(self):
    """Configura a visualização para monitoramento do treinamento."""
    if not pygame.get_init():
        pygame.init()
    
    # Configuração da janela de visualização
    self.training_width = 1000
    self.training_height = 600
    self.training_screen = pygame.display.set_mode((self.training_width, self.training_height))
    pygame.display.set_caption("Homeostatic Environment - Training Monitor")
    
    if not hasattr(self, 'font'):
        self.font = pygame.font.SysFont('Arial', 16)
        self.title_font = pygame.font.SysFont('Arial', 18, bold=True)
    
    # Histórico para gráficos
    self.state_history = {0: [], 1: []}  # Um para cada estado interno
    self.drive_history = []
    self.reward_history = []
    self.action_history = []
    self.position_history = []
    
    # Configurações de cores
    self.colors = {
        'background': (240, 240, 240),
        'text': (0, 0, 0),
        'grid': (200, 200, 200),
        'axis': (100, 100, 100),
        'state0': (255, 0, 0),    # Vermelho para estado 0
        'state1': (0, 0, 255),    # Azul para estado 1
        'drive': (0, 128, 0),     # Verde para drive
        'reward': (128, 0, 128),  # Roxo para recompensa
        'position': (255, 165, 0), # Laranja para posição
        'optimal': (34, 139, 34)  # Verde floresta para ponto ótimo
    }
    
    # Tamanho máximo dos históricos
    self.history_max_size = 200
    
    # Informações sobre episódios
    self.episode_count = 0
    self.total_steps = 0
    self.episode_steps = 0
    self.last_action = -1
    self.last_reward = 0
    self.episode_total_reward = 0
    
    return self.training_screen

def update_training_visualization(self, episode, step, action, reward, epsilon=None):
    """Atualiza a visualização do treinamento com os dados atuais."""
    if not hasattr(self, 'training_screen'):
        self.setup_training_visualization()
    
    # Atualizar contadores
    self.episode_count = episode
    self.episode_steps = step
    self.total_steps += 1
    self.last_action = action
    self.last_reward = reward
    self.episode_total_reward += reward
    
    # Atualizar históricos
    for i in range(len(self.current_state)):
        if len(self.state_history[i]) >= self.history_max_size:
            self.state_history[i].pop(0)
        self.state_history[i].append(float(self.current_state[i]))
    
    drive_value = self.drive.get_current_drive()
    if len(self.drive_history) >= self.history_max_size:
        self.drive_history.pop(0)
    self.drive_history.append(drive_value)
    
    if len(self.reward_history) >= self.history_max_size:
        self.reward_history.pop(0)
    self.reward_history.append(reward)
    
    if len(self.action_history) >= self.history_max_size:
        self.action_history.pop(0)
    self.action_history.append(action)
    
    if len(self.position_history) >= self.history_max_size:
        self.position_history.pop(0)
    self.position_history.append(self.agent_position)
    
		# # No método update_training_visualization
    # self.state_history[0].append(float(self.current_state[0]) + np.random.uniform(-0.5, 0.5))  # Adiciona ruído para teste
    # self.reward_history.append(reward + np.random.uniform(-0.5, 0.5))  # Adiciona ruído para teste
    
    # Limpar a tela
    self.training_screen.fill(self.colors['background'])
    
    # Desenhar o painel de informações
    self._draw_info_panel(epsilon)
    
    # Desenhar os gráficos
    self._draw_training_graphs()
    
    # Desenhar a representação do ambiente 1D
    self._draw_environment_representation()
    
    # Atualizar a tela
    pygame.display.flip()
    pygame.event.pump()
    
    return self.training_screen

def _draw_info_panel(self, epsilon=None):
    """Desenha o painel de informações sobre o treinamento."""
    # Desenhar título
    title = "MONITORAMENTO DE TREINAMENTO"
    title_surface = self.title_font.render(title, True, self.colors['text'])
    self.training_screen.blit(title_surface, (20, 15))
    
    # Informações básicas - coluna esquerda
    left_info = [
        f"Episódio: {self.episode_count}",
        f"Passos no episódio: {self.episode_steps}",
        f"Total de passos: {self.total_steps}",
        f"Estado interno 0: {self.current_state[0]:.2f}",
        f"Estado interno 1: {self.current_state[1]:.2f}",
        f"Ponto ótimo: {self.optimal_point}",
        f"Epsilon: {epsilon if epsilon is not None else 'N/A'}"
    ]
    
    for i, info in enumerate(left_info):
        text_surface = self.font.render(info, True, self.colors['text'])
        self.training_screen.blit(text_surface, (20, 50 + i * 25))
    
    # Informações básicas - coluna direita
    right_info = [
        f"Drive atual: {self.drive.get_current_drive():.2f}",
        f"Posição do agente: {self.agent_position}",
        f"Última ação: {self._action_to_text(self.last_action)}",
        f"Última recompensa: {self.last_reward:.2f}",
        f"Recompensa acumulada: {self.episode_total_reward:.2f}"
    ]
    
    for i, info in enumerate(right_info):
        text_surface = self.font.render(info, True, self.colors['text'])
        self.training_screen.blit(text_surface, (300, 50 + i * 25))

def _action_to_text(self, action):
    """Converte o código da ação em texto descritivo."""
    actions = ["Ficar parado", "Mover esquerda", "Mover direita", "Consumir", "Não consumir"]
    if 0 <= action < len(actions):
        return actions[action]
    return "Desconhecida"

def _draw_training_graphs(self):
    """Desenha os gráficos de monitoramento do treinamento."""
    # Área para os gráficos
    graphs_area = (10, 200, self.training_width - 20, self.training_height - 210)
    
    # Desenhar os gráficos em grade 2x2
    graph_width = (graphs_area[2] - 20) // 2
    graph_height = (graphs_area[3] - 20) // 2
    
    # Posições dos gráficos
    positions = [
        (graphs_area[0], graphs_area[1]),  # Estados internos
        (graphs_area[0] + graph_width + 20, graphs_area[1]),  # Drive
        (graphs_area[0], graphs_area[1] + graph_height + 20),  # Recompensas
        (graphs_area[0] + graph_width + 20, graphs_area[1] + graph_height + 20)  # Ações e posição
    ]
    
    # Desenhar os 4 gráficos
    self._draw_states_graph(positions[0], (graph_width, graph_height))
    self._draw_drive_graph(positions[1], (graph_width, graph_height))
    self._draw_reward_graph(positions[2], (graph_width, graph_height))
    self._draw_actions_position_graph(positions[3], (graph_width, graph_height))

def _draw_states_graph(self, position, size):
    """Desenha o gráfico de estados internos."""
    x, y = position
    width, height = size
    
    # Desenhar fundo e borda
    pygame.draw.rect(self.training_screen, self.colors['background'], (x, y, width, height))
    pygame.draw.rect(self.training_screen, self.colors['grid'], (x, y, width, height), 1)
    
    # Título
    title = "Estados Internos"
    title_surface = self.font.render(title, True, self.colors['text'])
    self.training_screen.blit(title_surface, (x + 10, y + 5))
    
    # Desenhar eixos
    margin = 30
    graph_x = x + margin
    graph_y = y + margin
    graph_width = width - 2 * margin
    graph_height = height - 2 * margin
    
    # Desenhar grid horizontal
    for i in range(5):
        y_pos = graph_y + (i * graph_height) // 4
        pygame.draw.line(self.training_screen, self.colors['grid'], 
                         (graph_x, y_pos), (graph_x + graph_width, y_pos), 1)
        
        # Calcular o valor correspondente (de -maxh a maxh)
        value = self.maxh - (i * (2 * self.maxh)) // 4
        label = self.font.render(f"{value}", True, self.colors['text'])
        self.training_screen.blit(label, (graph_x - 25, y_pos - 8))
    
    # Desenhar linha para o eixo x (zero)
    zero_y = graph_y + graph_height // 2
    pygame.draw.line(self.training_screen, self.colors['axis'], 
                     (graph_x, zero_y), (graph_x + graph_width, zero_y), 2)
    
    # Desenhar linha para o ponto ótimo
    for i, optimal in enumerate(self.optimal_point):
        optimal_y = graph_y + graph_height - ((optimal + self.maxh) * graph_height) // (2 * self.maxh)
        pygame.draw.line(self.training_screen, self.colors[f'state{i}'], 
                (graph_x, optimal_y), (graph_x + graph_width, optimal_y), 1)  # Linha normal
    
    # Desenhar históricos de estados
    for state_idx, history in self.state_history.items():
        if not history:
            continue
        
        points = []
        for i, value in enumerate(history):
            # Mapear índice para coordenada x
            if len(history) == 1:
                point_x = graph_x + graph_width // 2
            else:
                point_x = graph_x + (i * graph_width) // (len(history) - 1)
            
            # Mapear valor para coordenada y (de -maxh a maxh)
            normalized_value = (value + self.maxh) / (2 * self.maxh)
            point_y = graph_y + graph_height - int(normalized_value * graph_height)
            points.append((point_x, point_y))
        
        # Desenhar linhas
        if len(points) > 1:
            pygame.draw.lines(self.training_screen, self.colors[f'state{state_idx}'], False, points, 2)

def _draw_drive_graph(self, position, size):
    """Desenha o gráfico do drive."""
    x, y = position
    width, height = size
    
    # Desenhar fundo e borda
    pygame.draw.rect(self.training_screen, self.colors['background'], (x, y, width, height))
    pygame.draw.rect(self.training_screen, self.colors['grid'], (x, y, width, height), 1)
    
    # Título
    title = "Drive (distância ao ponto ótimo)"
    title_surface = self.font.render(title, True, self.colors['text'])
    self.training_screen.blit(title_surface, (x + 10, y + 5))
    
    # Desenhar eixos
    margin = 30
    graph_x = x + margin
    graph_y = y + margin
    graph_width = width - 2 * margin
    graph_height = height - 2 * margin
    
    # Determinar o valor máximo para escala
    max_drive = max(self.drive_history) if self.drive_history else 10
    max_drive = max(max_drive, 1) * 1.2  # Adicionar margem
    
    # Desenhar grid horizontal
    for i in range(5):
        y_pos = graph_y + (i * graph_height) // 4
        pygame.draw.line(self.training_screen, self.colors['grid'], 
                         (graph_x, y_pos), (graph_x + graph_width, y_pos), 1)
        
        # Valor no eixo y
        value = max_drive * (4 - i) / 4
        label = self.font.render(f"{value:.1f}", True, self.colors['text'])
        self.training_screen.blit(label, (graph_x - 35, y_pos - 8))
    
    # Desenhar histórico do drive
    if self.drive_history:
        points = []
        for i, value in enumerate(self.drive_history):
            # Mapear índice para coordenada x
            if len(self.drive_history) == 1:
                point_x = graph_x + graph_width // 2
            else:
                point_x = graph_x + (i * graph_width) // (len(self.drive_history) - 1)
            
            # Mapear valor para coordenada y
            normalized_value = value / max_drive
            point_y = graph_y + graph_height - int(normalized_value * graph_height)
            points.append((point_x, point_y))
        
        # Desenhar linhas
        if len(points) > 1:
            pygame.draw.lines(self.training_screen, self.colors['drive'], False, points, 2)

def _draw_reward_graph(self, position, size):
    """Desenha o gráfico de recompensas."""
    x, y = position
    width, height = size
    
    # Desenhar fundo e borda
    pygame.draw.rect(self.training_screen, self.colors['background'], (x, y, width, height))
    pygame.draw.rect(self.training_screen, self.colors['grid'], (x, y, width, height), 1)
    
    # Título
    title = "Recompensas"
    title_surface = self.font.render(title, True, self.colors['text'])
    self.training_screen.blit(title_surface, (x + 10, y + 5))
    
    # Desenhar eixos
    margin = 30
    graph_x = x + margin
    graph_y = y + margin
    graph_width = width - 2 * margin
    graph_height = height - 2 * margin
    
    # Determinar valores mínimo e máximo para escala
    if self.reward_history:
        min_reward = min(min(self.reward_history), -1)
        max_reward = max(max(self.reward_history), 1)
    else:
        min_reward = -1
        max_reward = 1
    
    range_reward = max_reward - min_reward
    
    # Desenhar grid horizontal
    for i in range(5):
        y_pos = graph_y + (i * graph_height) // 4
        pygame.draw.line(self.training_screen, self.colors['grid'], 
                         (graph_x, y_pos), (graph_x + graph_width, y_pos), 1)
        
        # Valor no eixo y
        value = max_reward - (i * range_reward) / 4
        label = self.font.render(f"{value:.1f}", True, self.colors['text'])
        self.training_screen.blit(label, (graph_x - 35, y_pos - 8))
    
    # Desenhar linha para o eixo x (zero)
    if min_reward < 0 < max_reward:
        zero_y = int(graph_y + graph_height - (-min_reward * graph_height) / range_reward)
        pygame.draw.line(self.training_screen, self.colors['axis'], 
                (graph_x, zero_y), (graph_x + graph_width, zero_y), 2)
    
    # Desenhar histórico de recompensas
    if self.reward_history:
        points = []
        for i, value in enumerate(self.reward_history):
            # Mapear índice para coordenada x
            if len(self.reward_history) == 1:
                point_x = graph_x + graph_width // 2
            else:
                point_x = graph_x + (i * graph_width) // (len(self.reward_history) - 1)
            
            # Mapear valor para coordenada y
            normalized_value = (value - min_reward) / range_reward
            point_y = graph_y + graph_height - int(normalized_value * graph_height)
            points.append((point_x, point_y))
        
        # Desenhar linhas
        if len(points) > 1:
            pygame.draw.lines(self.training_screen, self.colors['reward'], False, points, 2)

def _draw_actions_position_graph(self, position, size):
    """Desenha o gráfico de ações e posição do agente."""
    x, y = position
    width, height = size
    
    # Desenhar fundo e borda
    pygame.draw.rect(self.training_screen, self.colors['background'], (x, y, width, height))
    pygame.draw.rect(self.training_screen, self.colors['grid'], (x, y, width, height), 1)
    
    # Título
    title = "Ações e Posição"
    title_surface = self.font.render(title, True, self.colors['text'])
    self.training_screen.blit(title_surface, (x + 10, y + 5))
    
    # Desenhar eixos
    margin = 30
    graph_x = x + margin
    graph_y = y + margin
    graph_width = width - 2 * margin
    graph_height = height - 2 * margin
    
    # Área de ações (metade superior)
    action_height = graph_height // 2 - 5
    
    # Área de posição (metade inferior)
    position_y = graph_y + graph_height // 2 + 5
    position_height = graph_height // 2 - 5
    
    # Desenhar separador
    pygame.draw.line(self.training_screen, self.colors['grid'], 
                    (graph_x, graph_y + graph_height // 2), 
                    (graph_x + graph_width, graph_y + graph_height // 2), 1)
    
    # Rótulos das ações
    actions = ["Ficar", "Esq", "Dir", "Consumir", "Não cons."]
    for i, action in enumerate(actions):
        y_pos = graph_y + (i * action_height) // 4
        pygame.draw.line(self.training_screen, self.colors['grid'], 
                        (graph_x, y_pos), (graph_x + graph_width, y_pos), 1)
        label = self.font.render(action, True, self.colors['text'])
        self.training_screen.blit(label, (graph_x - 60, y_pos - 8))
    
    # Grades para posição
    max_pos = self.maxh
    min_pos = -self.maxh
    range_pos = max_pos - min_pos
    
    for i in range(5):
        y_pos = position_y + (i * position_height) // 4
        pygame.draw.line(self.training_screen, self.colors['grid'], 
                        (graph_x, y_pos), (graph_x + graph_width, y_pos), 1)
        
        value = max_pos - (i * range_pos) // 4
        label = self.font.render(f"{value}", True, self.colors['text'])
        self.training_screen.blit(label, (graph_x - 25, y_pos - 8))
    
    # Desenhar histórico de ações
    if self.action_history:
        action_points = []
        for i, action in enumerate(self.action_history):
            # Mapear índice para coordenada x
            if len(self.action_history) == 1:
                point_x = graph_x + graph_width // 2
            else:
                point_x = graph_x + (i * graph_width) // (len(self.action_history) - 1)
            
            # Mapear ação para coordenada y
            point_y = graph_y + (action * action_height) // 4
            action_points.append((point_x, point_y))
            
            # Desenhar ponto para a ação
            pygame.draw.circle(self.training_screen, (0, 0, 0), (point_x, point_y), 3)
    
    # Desenhar histórico de posições
    if self.position_history:
        position_points = []
        for i, pos in enumerate(self.position_history):
            # Mapear índice para coordenada x
            if len(self.position_history) == 1:
                point_x = graph_x + graph_width // 2
            else:
                point_x = graph_x + (i * graph_width) // (len(self.position_history) - 1)
            
            # Mapear posição para coordenada y
            normalized_pos = (pos - min_pos) / range_pos
            point_y = position_y + position_height - int(normalized_pos * position_height)
            position_points.append((point_x, point_y))
        
        # Desenhar linhas para posições
        if len(position_points) > 1:
            pygame.draw.lines(self.training_screen, self.colors['position'], False, position_points, 2)

def _draw_environment_representation(self):
    """Desenha uma representação simplificada do ambiente 1D."""
    # Área para a representação no topo
    env_area = (600, 50, 380, 80)
    x, y, width, height = env_area
    
    # Desenhar fundo
    pygame.draw.rect(self.training_screen, (250, 250, 250), env_area)
    pygame.draw.rect(self.training_screen, self.colors['grid'], env_area, 1)
    
    # Desenhar linha representando o ambiente 1D
    line_y = y + height // 2
    pygame.draw.line(self.training_screen, (0, 0, 0), 
                    (x + 20, line_y), (x + width - 20, line_y), 3)
    
    # Calcular escala para mapear posições
    scale = (width - 40) / (2 * self.maxh)
    center_x = x + width // 2
    
    # Desenhar recursos
    for i, pos in enumerate(self.resources_position):
        resource_x = center_x + pos * scale
        color = (0, 0, 255) if i == 0 else (0, 255, 0)  # Azul para 0, Verde para 1
        pygame.draw.circle(self.training_screen, color, (int(resource_x), line_y), 10)
    
    # Desenhar agente
    agent_x = center_x + self.agent_position * scale
    pygame.draw.circle(self.training_screen, (255, 0, 0), (int(agent_x), line_y), 7)
    
    # Rótulo
    label = self.font.render("Ambiente 1D", True, self.colors['text'])
    self.training_screen.blit(label, (x + 10, y + 5))

def reset_training_stats(self):
    """Reinicia as estatísticas de treinamento para um novo episódio."""
    self.episode_steps = 0
    self.episode_total_reward = 0
    if hasattr(self, 'state_history'):
        for state in self.state_history:
            self.state_history[state] = []
        self.drive_history = []
        self.reward_history = []
        self.action_history = []
        self.position_history = []

def train_with_visualization(self, num_episodes=500, render_interval=1):
    """
    Treina o agente com visualização em tempo real.

    Args:
        num_episodes: Número de episódios para treinamento
        render_interval: Intervalo de passos para atualizar a visualização (menor = mais lento, maior = mais rápido)

    Returns:
        Lista de recompensas médias por episódio
    """
    rewards_history = []

    # Configura a visualização do treinamento
    self.setup_training_visualization()

    try:
        for episode in range(num_episodes):
            state, _ = self.reset()
            episode_reward = 0
            done = False
            truncated = False
            steps = 0

            self.reset_training_stats()

            # Usa diretamente o dicionário state_to_idx
            state_idx = self.state_to_idx[tuple(state)]

            while not (done or truncated):
                # Escolhe uma ação
                action = self.select_action(state_idx)

                # Executa a ação
                next_state, reward, done, truncated, _ = self.step(action)

                # Obtém o índice do próximo estado
                next_state_idx = self.state_to_idx[tuple(next_state)]

                # Atualiza a Q-Table
                self.update_q_table(state_idx, action, reward, next_state_idx, done)

                # Atualiza o estado
                state_idx = next_state_idx
                episode_reward += reward
                steps += 1

                # Atualiza a visualização a cada N passos
                if steps % render_interval == 0:
                    self.update_training_visualization(episode + 1, steps, action, reward, epsilon=self.epsilon)

                if steps >= 1000:
                    truncated = True

            rewards_history.append(episode_reward)

            print(f"Episódio: {episode + 1}/{num_episodes}, Recompensa: {episode_reward:.2f}, Epsilon: {self.epsilon:.4f}")

            # Atualiza visualização ao final do episódio
            self.update_training_visualization(episode + 1, steps, action, reward, epsilon=self.epsilon)

    except KeyboardInterrupt:
        print("\nTreinamento interrompido pelo usuário.")

    pygame.quit()

    return rewards_history
