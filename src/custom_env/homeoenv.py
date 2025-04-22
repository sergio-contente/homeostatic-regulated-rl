from src.agents.q_learning import QLearning
from ..utils.get_params import ParameterHandler
import pygame
import numpy as np
import matplotlib.pyplot as plt
import time

class HomeoEnv:
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

        # Optimal point:
        tensor = self.drive.get_tensor_optimal_states_values()
        self.optimal_point = tuple(tensor.tolist())

        self.size = self.drive.get_internal_state_size()
        self.current_state = np.random.choice(range(0, maxh + 1), size=self.size)
        initial_drive = self.drive.compute_drive(self.current_state)
        self.drive.update_drive(initial_drive)

    def reset(self):
        self.current_state = np.random.choice(range(0, self.maxh + 1), size=self.size)
        initial_drive = self.drive.compute_drive(self.current_state)
        self.drive.update_drive(initial_drive)
        self.steps = 0
        return self.current_state, {}
    
    def step(self, action):
            if self.render is not None:
                print(f"[STEP] Estado atual: {self.current_state}, Ação: {action}")
            prev_state = self.current_state.copy()
            # Apply the chosen action to modify internal states
            if action == 0:  # Consume resource 0
                    self.current_state[0] = min(self.current_state[0] + self._outcome, self.maxh)
            elif action == 1:  # Consume resource 1
                    self.current_state[1] = min(self.current_state[1] + self._outcome, self.maxh)
            elif action == 2:
                    self.current_state[0] = max(self.current_state[0] - self._outcome, 0)
            elif action == 3:
                    self.current_state[1] = max(self.current_state[1] - self._outcome, 0)
            elif action == 4:
                    self.current_state = self.current_state

            # Updates drive and reward
            new_drive = self.drive.compute_drive(self.current_state)
            reward = self.drive.compute_reward(new_drive)
            self.drive.update_drive(new_drive)
            
            # An episode is done if internal states are close to optimal
            # You might want to define a threshold for "close enough"
            threshold = self._outcome / 2 
            #terminated = self.drive.has_reached_optimal(self.current_state, threshold)
            terminated = np.array_equal(
                self.current_state, 
                self.drive.get_tensor_optimal_states_values()
            )
            
            # Big reward if reached optimal internal state
            if terminated:
                print("Achieved Homeostatic Point")
                reward += 10

            if np.array_equal(self.current_state, prev_state):
                reward -= 0.1


            observation = self.current_state

            if self.render_mode == "human":
                    self._render_frame()

            self.steps += 1
            truncated = self.steps >= 1000  # Limit the number of steps per episode
            
            return observation, reward, terminated, truncated, {}
    
    def _get_state_index(self, state):
        """
        Converts a state to its corresponding index in the state space.
        Raises an error if the state is not valid.
        """
        if isinstance(state, np.ndarray):
            state_tuple = tuple(state.astype(int))
        else:
            state_tuple = tuple(state)

        if state_tuple not in self.state_space:
            raise ValueError(f"[ERROR] Invalid state: {state_tuple}. This state is not in the defined state space.")
        
        return self.state_space.index(state_tuple)

    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        """Renders the current state of the environment using pygame."""
        if not hasattr(self, 'screen'):
            pygame.init()
            self.screen_width = 600
            self.screen_height = 600
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Custom Homeostatic Environment")
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))

        total_cells = self.maxh + 1  # só de 0 até maxh

        # Definições de margens e tamanho de célula
        margin = 50
        available_width = self.screen_width - 2 * margin
        available_height = self.screen_height - 2 * margin
        cell_size = min(available_width // total_cells, available_height // total_cells)

        # Centraliza a grid
        grid_width = total_cells * cell_size
        grid_height = total_cells * cell_size
        grid_start_x = (self.screen_width - grid_width) // 2
        grid_start_y = (self.screen_height - grid_height) // 2

        # Desenha a grid
        for x in range(total_cells):
            for y in range(total_cells):
                pixel_x = grid_start_x + x * cell_size
                pixel_y = grid_start_y + (self.maxh - y) * cell_size  # inverte o eixo y

                # Cor do fundo da célula
                cell_color = (240, 240, 240)
                pygame.draw.rect(self.screen, cell_color, (pixel_x, pixel_y, cell_size, cell_size))

                # Borda
                pygame.draw.rect(self.screen, (200, 200, 200), (pixel_x, pixel_y, cell_size, cell_size), 1)

                # Coordenadas
                if cell_size >= 20:
                    small_font = pygame.font.SysFont('Arial', max(8, cell_size // 5))
                    coord_text = f"{x},{y}"
                    coord_surf = small_font.render(coord_text, True, (180, 180, 180))
                    coord_rect = coord_surf.get_rect(center=(pixel_x + cell_size // 2, pixel_y + cell_size // 2))
                    self.screen.blit(coord_surf, coord_rect)

                # Ponto ótimo (quadrado verde)
                if (x, y) == self.optimal_point:
                    pygame.draw.rect(self.screen, (0, 255, 0), (pixel_x, pixel_y, cell_size, cell_size))
                    if cell_size >= 30:
                        label_font = pygame.font.SysFont('Arial', 12)
                        label = label_font.render("OPTIMAL", True, (0, 100, 0))
                        label_rect = label.get_rect(center=(pixel_x + cell_size // 2, pixel_y + cell_size // 2))
                        self.screen.blit(label, label_rect)

        # Desenha o agente (círculo vermelho)
        agent_x, agent_y = self.current_state
        agent_pixel_x = grid_start_x + agent_x * cell_size + cell_size // 2
        agent_pixel_y = grid_start_y + (self.maxh - agent_y) * cell_size + cell_size // 2
        pygame.draw.circle(self.screen, (255, 0, 0), (agent_pixel_x, agent_pixel_y), cell_size // 3)

        # Informações de status
        font = pygame.font.SysFont('Arial', 14)
        info_lines = [
            f"Agent: ({agent_x}, {agent_y})",
            f"Optimal: {self.optimal_point}",
            f"Distance to optimal: {sum(abs(np.array(self.current_state) - np.array(self.optimal_point)))}",
            f"Steps: {self.steps}"
        ]
        for i, text in enumerate(info_lines):
            surf = font.render(text, True, (0, 0, 0))
            self.screen.blit(surf, (10, 10 + 20 * i))

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
