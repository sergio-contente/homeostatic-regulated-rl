from enum import Enum

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

from src.utils.get_params import ParameterHandler


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config_path, drive_type, render_mode=None, size=5):
        self.param_manager = ParameterHandler(config_path)
        self.drive = self.param_manager.create_drive(drive_type)
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.dimension_internal_states = self.drive.get_internal_state_dimension()


        # Observations are dictionaries with the agent's and the target's location.
        observation_dict = spaces.Dict(
            {
                "position": spaces.Discrete(self.size),
                "internal_states": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.dimension_internal_states,),
F                    dtype=np.float64,
                )
            }
        )
        self.observation_space = spaces.Dict(observation_dict)

        num_actions = 3 + self.dimension_internal_states  # left, right, stay + consume actions

        self.action_space = spaces.Discrete(num_actions)

        state_names = self.drive.get_internal_states_names()
        
        resource_rng = np.random.RandomState(123)

        if self.dimension_internal_states <= self.size:
            random_positions = resource_rng.choice(
                self.size, 
                size=self.dimension_internal_states, 
                replace=False
            )
        else:
            random_positions = resource_rng.choice(
                self.size, 
                size=self.dimension_internal_states, 
                replace=True
            )

        self.resources_info = {}
        for i, state_name in enumerate(state_names):
            self.resources_info[i] = {
                "name": state_name,
                "position": random_positions[i]
            }

        self.area = set(range(size))

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        self.agent_info = {
            "position": 0,
            "internal_states": np.zeros(self.dimension_internal_states, dtype=np.float64)
        }

        self.previous_internal_states = np.array(self.agent_info["internal_states"])

        initial_drive = self.drive.compute_drive(self.agent_info["internal_states"])
        self.drive.update_drive(initial_drive)

        self.reward_scale = 100.

    def set_agent_info(self, position, internal_states):
        self.agent_info["position"] = position
        self.agent_info["internal_states"] = internal_states

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.set_agent_info(
                position=self.np_random.choice(list(self.area)),
                internal_states=self.np_random.uniform(
                    low=-0.3, 
                    high=0.3, 
                    size=(self.dimension_internal_states,)
                )
            )

        self.previous_internal_states = np.array(self.agent_info["internal_states"])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        initial_drive = self.drive.compute_drive(self.agent_info["internal_states"])
        self.drive.update_drive(initial_drive)

        return observation, info

    def _get_obs(self):
        obs = {
            "position": self.agent_info["position"],
            "internal_states": self.agent_info["internal_states"]
        }
        return obs

    def _get_info(self):
        return self.agent_info.copy()

    def step(self, action):
        # Salva o estado anterior
        self.previous_internal_states = np.array(self.agent_info["internal_states"])
        
        # Aplica o decay natural aos estados internos
        states_after_cost = self.drive.apply_natural_decay(self.previous_internal_states)
        self.agent_info["internal_states"] = states_after_cost
        # Processa ações de movimento
        previous_position = self.agent_info["position"]
        new_position = self.agent_info["position"]
        if action == 0:  # stay
            new_position = previous_position
        elif action == 1:  # left
            new_position = previous_position - 1
        elif action == 2:  # right
            new_position = previous_position + 1
        
        # Verifica limites da área
        if new_position not in self.area:
            new_position = previous_position
        
        self.agent_info["position"] = new_position
        
        # Inicializa new_internal_state para caso não ocorra consumo
        new_internal_state = np.array(self.agent_info["internal_states"])
        
        # Processa ações de consumo
        consumption_actions = self.dimension_internal_states
        for i in range(consumption_actions):
            if action == 3 + i:
                resource = self.resources_info[i]
                resource_position = resource["position"]
                
                # Verifica se o agente está na mesma posição do recurso
                if previous_position == resource_position:
                    # Prepara um vetor indicando qual recurso está sendo consumido
                    action_states = np.zeros(self.dimension_internal_states)
                    action_states[i] = 1.0
                    
                    # Aplica a ingestão
                    new_internal_state = self.drive.apply_intake(
                        self.agent_info["internal_states"],
                        action_states
                    )
        
        # Atualiza os estados internos após possível consumo
        self.agent_info["internal_states"] = new_internal_state
        
        # IMPORTANTE: Verifica se algum estado está abaixo de -1.0
        # Adicionando print para debug
        # print("Verificando estados internos:", self.agent_info["internal_states"])
        # print("Algum estado < -1.0?", np.any(self.agent_info["internal_states"] < -1.0))
        
        # Define done como True se qualquer estado for menor que -1.0
        done = False
        if np.any(self.agent_info["internal_states"] < -1.0):
            # print("EPISÓDIO TERMINADO: Um estado está abaixo de -1")
            done = True
        
        # Calcula recompensa
        reward = self.get_reward()
        
        # Prepara retorno
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, done, False, info

    def get_reward(self):
        new_drive = self.drive.compute_drive(self.agent_info["internal_states"]) # get new drive
        reward = self.drive.compute_reward(new_drive) #get reward
        self.drive.update_drive(new_drive) # update drive
        return reward * self.reward_scale

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((240, 240, 240))  # Fundo ligeiramente cinza claro para melhor contraste
        
        # Definir áreas do layout
        grid_height = self.window_size // 3  # A grade ocupa o terço superior
        info_start_y = grid_height + 20  # Espaço para informações abaixo da grade
        
        # Tamanho dos quadrados na grade
        pix_square_size = grid_height / self.size
        
        # Desenhar área da grade com borda
        pygame.draw.rect(
            canvas,
            (220, 220, 220),  # Cinza claro para a área da grade
            pygame.Rect(0, 0, self.window_size, grid_height),
        )
        
        # Desenhar os recursos na grade
        for resource_id, resource in self.resources_info.items():
            # Cores vibrantes distintas para cada recurso
            if resource["name"] == "food":
                resource_color = (220, 50, 50)  # Vermelho mais visível para comida
            elif resource["name"] == "water":
                resource_color = (50, 50, 220)  # Azul mais visível para água
            elif resource["name"] == "energy":
                resource_color = (220, 220, 50)  # Amarelo mais visível para energia
            else:
                # Outras cores para recursos adicionais
                colors = [(50, 200, 50), (200, 50, 200), (50, 200, 200), (200, 150, 50)]
                resource_color = colors[resource_id % len(colors)]
            
            # Desenhar o recurso como um quadrado preenchido com borda
            pygame.draw.rect(
                canvas,
                resource_color,
                pygame.Rect(
                    (resource["position"] * pix_square_size, 0),
                    (pix_square_size, pix_square_size),
                ),
            )
            # Adicionar uma borda preta ao redor do recurso
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    (resource["position"] * pix_square_size, 0),
                    (pix_square_size, pix_square_size),
                ),
                width=2
            )
        
        # Desenhar o agente como um círculo com borda
        agent_x = self.agent_info["position"]
        agent_y = 0.5  # Centralizado verticalmente no grid
        
        # Círculo exterior (borda preta)
        pygame.draw.circle(
            canvas,
            (0, 0, 0),  # Preto para a borda
            (agent_x * pix_square_size + pix_square_size / 2,
            agent_y * pix_square_size + pix_square_size / 2),
            pix_square_size / 2.5,
        )
        
        # Círculo interior (preenchimento azul)
        pygame.draw.circle(
            canvas,
            (50, 150, 255),  # Azul para o agente
            (agent_x * pix_square_size + pix_square_size / 2,
            agent_y * pix_square_size + pix_square_size / 2),
            pix_square_size / 3,
        )
        
        # Desenhar as linhas da grade
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                (100, 100, 100),  # Cinza escuro para as linhas
                (pix_square_size * x, 0),
                (pix_square_size * x, grid_height),
                width=2,
            )
        # Linhas horizontais da grade
        pygame.draw.line(
            canvas,
            (100, 100, 100),
            (0, 0),
            (self.window_size, 0),
            width=2,
        )
        pygame.draw.line(
            canvas,
            (100, 100, 100),
            (0, grid_height),
            (self.window_size, grid_height),
            width=2,
        )
        
        # Inicializar a fonte
        if pygame.font:
            title_font = pygame.font.SysFont('Arial', 28, bold=True)
            label_font = pygame.font.SysFont('Arial', 20, bold=True)
            value_font = pygame.font.SysFont('Arial', 18)
        else:
            # Fallback se não houver fonte disponível
            return
        
        # Exibir valor do drive atual
        drive_value = self.drive.get_current_drive()
        drive_title = title_font.render("Drive:", True, (0, 0, 0))
        drive_value_text = value_font.render(f"{drive_value:.4f}", True, (0, 0, 0))
        
        canvas.blit(drive_title, (20, grid_height + 20))
        canvas.blit(drive_value_text, (120, grid_height + 20))
        
        # Exibir barras de estados internos com melhor formatação
        state_names = self.drive.get_internal_states_names()
        bar_height = 30
        bar_spacing = 40
        bar_width = self.window_size - 150  # Largura fixa para todas as barras
        
        # Desenhar cada barra de estado interno
        for i in range(self.dimension_internal_states):
            y_pos = info_start_y + 50 + i * bar_spacing
            state_value = self.agent_info["internal_states"][i]
            
            # Título do estado
            state_label = label_font.render(f"{state_names[i]}:", True, (0, 0, 0))
            canvas.blit(state_label, (20, y_pos))
            
            # Valor numérico
            value_text = value_font.render(f"{state_value:.2f}", True, (0, 0, 0))
            canvas.blit(value_text, (20, y_pos + 25))
            
            # Desenhar fundo da barra (cinza)
            pygame.draw.rect(
                canvas,
                (200, 200, 200),
                pygame.Rect(150, y_pos, bar_width, bar_height),
            )
            
            # Normalizar o valor para [0, 1]
            normalized_value = (state_value + 1) / 2
            filled_width = max(0, int(normalized_value * bar_width))
            
            # Determinar cor da barra baseada no valor
            if state_value > 0:
                # Verde mais forte para valores positivos
                bar_color = (0, min(255, int(150 + state_value * 100)), 0)
            else:
                # Vermelho mais forte para valores negativos
                bar_color = (min(255, int(150 - state_value * 100)), 0, 0)
            
            # Desenhar a parte preenchida da barra
            pygame.draw.rect(
                canvas,
                bar_color,
                pygame.Rect(150, y_pos, filled_width, bar_height),
            )
            
            # Borda da barra
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(150, y_pos, bar_width, bar_height),
                width=2
            )
            
            # Marcações na barra
            mid_x = 150 + bar_width / 2
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (mid_x, y_pos - 5),
                (mid_x, y_pos + bar_height + 5),
                width=2
            )
            
            # Rótulos de mínimo, meio e máximo
            min_label = value_font.render("-1", True, (0, 0, 0))
            mid_label = value_font.render("0", True, (0, 0, 0))
            max_label = value_font.render("1", True, (0, 0, 0))
            
            canvas.blit(min_label, (145, y_pos + bar_height + 5))
            canvas.blit(mid_label, (mid_x - 5, y_pos + bar_height + 5))
            canvas.blit(max_label, (150 + bar_width - 10, y_pos + bar_height + 5))
        
        # Mostrar legenda dos recursos
        legend_y = info_start_y + 50 + self.dimension_internal_states * bar_spacing + 30
        legend_title = label_font.render("Resources:", True, (0, 0, 0))
        canvas.blit(legend_title, (20, legend_y))
        
        # Mostrar cada recurso na legenda
        for i, state_name in enumerate(state_names):
            legend_x = 150 + i * 120
            
            # Determinar cor do recurso
            if state_name == "food":
                resource_color = (220, 50, 50)
            elif state_name == "water":
                resource_color = (50, 50, 220)
            elif state_name == "energy":
                resource_color = (220, 220, 50)
            else:
                colors = [(50, 200, 50), (200, 50, 200), (50, 200, 200), (200, 150, 50)]
                resource_color = colors[i % len(colors)]
            
            # Desenhar quadrado de cor do recurso
            pygame.draw.rect(
                canvas,
                resource_color,
                pygame.Rect(legend_x, legend_y, 20, 20),
            )
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(legend_x, legend_y, 20, 20),
                width=1
            )
            
            # Nome do recurso
            name_text = value_font.render(state_name, True, (0, 0, 0))
            canvas.blit(name_text, (legend_x + 25, legend_y + 2))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == '__main__':
    config_path = "config/config.yaml"
    drive_type = "base_drive"  # base_drive, elliptic_drive, etc.
    env = GridWorldEnv(render_mode="human", config_path=config_path, drive_type=drive_type, size=10)
    env.reset()

    for i in range(1000):
        actions = env.action_space.sample()
        print(actions)
        obs, reward, done, truncate, info = env.step(actions)
        env.render()
        print("obs:", obs)
        print(info)

        if done:
            print("Episódio terminado, resetando ambiente...")
            obs, info = env.reset()
            print("Novo episódio iniciado. Estados internos:", obs["internal_states"])

    env.close()
    print("finish.")
