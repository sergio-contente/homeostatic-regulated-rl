from src.agents.q_learning import QLearning
from ..utils.get_params import ParameterHandler
import pygame
import numpy as np
import matplotlib.pyplot as plt
import time

# Classe que implementa um ambiente 1D com 3 ações de movimento (direita, esquerda, parado)
# Além disso tem a ideia de ação de consumir ou não os recursos

class HomeoEnv1D:
    def __init__(self, config_path, drive_type, render_mode=None, maxh=5):
        self.state_space = [(i, j) for i in range(-maxh, maxh + 1) for j in range(-maxh, maxh + 1)]
        self.action_space = [0, 1, 2, 3, 4]  # ficar, esquerda, direita, consumir, não consumir
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

        # Optimal point:
        tensor = self.drive.get_tensor_optimal_states_values()
        self.optimal_point = tuple(tensor.tolist())

        self.size = self.drive.get_internal_state_size()
        # Inicializa os estados internos com valores aleatórios entre -maxh e maxh
        self.current_state = np.random.choice(range(-maxh, maxh + 1), size=self.size)
        initial_drive = self.drive.compute_drive(self.current_state)
        self.drive.update_drive(initial_drive)

        # Localization of resources and agent
        # resource 0: limite esquerdo
        # resource 1: limite direito
        self.resources_position = [-maxh, maxh]
        self.agent_position = 0  # Começa no centro do ambiente

    def reset(self):
        # Resetar o estado interno do agente para valores aleatórios
        self.current_state = np.random.choice(range(-self.maxh, self.maxh + 1), size=self.size)
        initial_drive = self.drive.compute_drive(self.current_state)
        self.drive.update_drive(initial_drive)
        
        # Resetar a posição do agente para o centro
        self.agent_position = 0
        
        self.steps = 0
        return self.current_state, {}
    
    def step(self, action):
        # Um problema aqui: ele tá em um recurso, porém o valor interno dele para
        # o outro recurso está caindo e ele deve sair do recurso atual para consumir o outro
        
        # Verifica se atingiu um recurso
        resource_index = None
        if self.agent_position == self.resources_position[0]:
            resource_index = 0  # Está no recurso da esquerda
        elif self.agent_position == self.resources_position[1]:
            resource_index = 1  # Está no recurso da direita
        on_resource = resource_index is not None

        # Tratamento do movimento -> desconto do estado interno para cada timestep
        # Ações: 0 = ficar, 1 = esquerda, 2 = direita, 3 = consumir, 4 = não consumir
        
        # Tratamento do movimento
        if action == 0:  # Ficar parado
            pass  # Mantém a posição atual
        elif action == 1:  # Mover para esquerda
            self.agent_position = max(self.agent_position - 1, -self.maxh)
        elif action == 2:  # Mover para direita
            self.agent_position = min(self.agent_position + 1, self.maxh)
        
        # Verificar novamente se após o movimento o agente está em um recurso
        if self.agent_position == self.resources_position[0]:
            resource_index = 0
        elif self.agent_position == self.resources_position[1]:
            resource_index = 1
        on_resource = resource_index is not None
        
        # o agente tem que decidir se vai continuar no recurso atual (overeating ou não) ou ir pro outro
        # Tratamento do consumo
        if on_resource:
            if action == 3:  # Consumir
                # Aumenta o valor interno correspondente ao recurso
                # Se estiver no recurso da esquerda (índice 0), aumenta o primeiro estado interno
                # Se estiver no recurso da direita (índice 1), aumenta o segundo estado interno
                consumption_value = 1  # Quantidade a aumentar no estado interno
                
                # Atualiza o estado interno correspondente ao recurso
                self.current_state[resource_index] = min(
                    self.current_state[resource_index] + consumption_value, 
                    self.maxh
                )
            elif action == 4:  # Não consumir (decidiu não consumir mesmo estando no recurso)
                pass  # Não faz nada, apenas permanece no local sem consumir
        
        # Decaimento natural dos estados internos a cada passo
        decay_rate = 0.1  # Taxa de decaimento dos estados internos
        for i in range(len(self.current_state)):
            # Reduz o valor do estado interno, mas não abaixo de -maxh
            self.current_state[i] = max(self.current_state[i] - decay_rate, -self.maxh)
        
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
            reward += 10  # Bônus de recompensa por atingir o ponto homeostático

        observation = self.current_state

        if self.render_mode == "human":
            self._render_frame()

        self.steps += 1
        truncated = self.steps >= 1000  # Limit the number of steps per episode
        
        return observation, reward, terminated, truncated, {}
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        """Renders the current state of the environment using pygame."""
        if not hasattr(self, 'screen'):
            pygame.init()
            self.screen_width = 800
            self.screen_height = 400
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Homeostatic Environment 1D")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 16)
            
        # Clear the screen
        self.screen.fill((255, 255, 255))
        
        # Desenhar a "linha" onde o agente se move (ambiente 1D)
        line_y = self.screen_height // 2
        pygame.draw.line(
            self.screen, 
            (0, 0, 0), 
            (50, line_y), 
            (self.screen_width - 50, line_y), 
            3
        )
        
        # Calcular a escala para mapear de [-maxh, maxh] para o espaço da tela
        scale = (self.screen_width - 100) / (2 * self.maxh)
        center_x = self.screen_width // 2
        
        # Desenhar os recursos
        for i, pos in enumerate(self.resources_position):
            resource_x = center_x + pos * scale
            resource_color = (0, 0, 255) if i == 0 else (0, 255, 0)  # Azul para recurso 0, Verde para recurso 1
            pygame.draw.circle(self.screen, resource_color, (int(resource_x), line_y), 15)
            
            # Desenhar labels dos recursos
            text = f"Recurso {i}"
            text_surface = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (int(resource_x - 30), line_y - 40))
        
        # Desenhar o agente
        agent_x = center_x + self.agent_position * scale
        pygame.draw.circle(self.screen, (255, 0, 0), (int(agent_x), line_y), 10)
        
        # Mostrar os estados internos
        for i, state in enumerate(self.current_state):
            text = f"Estado interno {i}: {state:.1f}"
            text_surface = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (50, 30 + i * 25))
        
        # Mostrar o valor do drive (distância do ponto ótimo)
        drive_value = self.drive.compute_drive(self.current_state)
        drive_text = f"Drive: {drive_value:.2f}"
        drive_surface = self.font.render(drive_text, True, (0, 0, 0))
        self.screen.blit(drive_surface, (50, 100))
        
        # Mostrar a posição do agente
        pos_text = f"Posição: {self.agent_position}"
        pos_surface = self.font.render(pos_text, True, (0, 0, 0))
        self.screen.blit(pos_surface, (50, 130))
        
        # Mostrar se está em um recurso
        resource_status = "Não está em recurso"
        if self.agent_position == self.resources_position[0]:
            resource_status = "No recurso 0 (esquerda)"
        elif self.agent_position == self.resources_position[1]:
            resource_status = "No recurso 1 (direita)"
        
        resource_surface = self.font.render(resource_status, True, (0, 0, 0))
        self.screen.blit(resource_surface, (50, 160))
        
        # Update the display
        pygame.display.flip()
        self.clock.tick(30)
        pygame.event.pump()

    def plot_rewards(self, rewards):
        """Plot the training rewards curve."""
        import os

        # Garante que a pasta 'images/custom/homeoenv' exista
        os.makedirs('images/custom/homeoenv', exist_ok=True)

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
        plt.savefig('images/custom/homeoenv/training_rewards.png')
        plt.show()
