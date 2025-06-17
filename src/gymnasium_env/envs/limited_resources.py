from enum import Enum

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

from src.utils.get_params import ParameterHandler


class LimitedResources1D(gym.Env):
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
                    dtype=np.float64
                ),
                "resources_map":  spaces.MultiBinary(self.dimension_internal_states)
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
                "position": random_positions[i],
                "available": True
            }

        resources_map = np.array([
            int(resource["available"]) for resource in self.resources_info.values()
        ], dtype=np.int8)

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

    def set_agent_info(self, position, internal_states, resources_map):
        self.agent_info["position"] = position
        self.agent_info["internal_states"] = internal_states
        self.agent_info["resources_map"] = resources_map

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        resources_map = np.empty(self.dimension_internal_states, dtype=np.int8)

        for i, resource in self.resources_info.items():
            resource["available"] = True
            resources_map[i] = 1

        self.set_agent_info(
                position=self.np_random.choice(list(self.area)),
                internal_states=self.np_random.uniform(
                    low=-0.3, 
                    high=0.3, 
                    size=(self.dimension_internal_states,)
                ),
                resources_map=resources_map
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
            "internal_states": self.agent_info["internal_states"],

            # IDEA: Add resources availability to the observation
            "resources_map": self.agent_info["resources_map"]
        }
        return obs

    def _get_info(self):
        return self.agent_info.copy()

    def step(self, action):
        self.previous_internal_states = np.array(self.agent_info["internal_states"])

        for resource in self.resources_info.values():
            resource["available"] = self.drive.apply_resource_regeneration(
                resource["available"],
                resource["name"]
            )

        # Decaimento natural dos estados internos
        states_after_cost = self.drive.apply_natural_decay(self.previous_internal_states)

        # Processamento de movimento
        previous_position = self.agent_info["position"]
        new_position = previous_position
        if action == 1:  # left
            new_position -= 1
        elif action == 2:  # right
            new_position += 1

        # Verifica limites
        if new_position not in self.area:
            new_position = previous_position

        # Inicializa novo estado interno (pode ser sobrescrito pelo consumo)
        new_internal_state = np.array(states_after_cost)

        # Processamento de ações de consumo
        for i in range(self.dimension_internal_states):
            if action == 3 + i:
                resource = self.resources_info[i]
                if previous_position == resource["position"] and resource["available"]:
                    action_states = np.zeros(self.dimension_internal_states)
                    action_states[i] = 1.0
                    new_internal_state = self.drive.apply_intake(states_after_cost, action_states)
                    resource["available"] = False

        # Atualiza o vetor binário de disponibilidade dos recursos
        resources_map = np.array([
            int(resource["available"]) for resource in self.resources_info.values()
        ], dtype=np.int8)

        # Atualiza tudo com a função set_agent_info
        self.set_agent_info(
            position=new_position,
            internal_states=new_internal_state,
            resources_map=resources_map
        )

        # Terminação se algum estado interno estiver crítico
        done = np.any(self.agent_info["internal_states"] < -1.0) or np.any(self.agent_info["internal_states"] > 1.0)    

        reward = self.get_reward()
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
            
            # Verificar se o recurso está disponível
            if resource["available"]:
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
                
                # Adicionar ícone para mostrar que o recurso está disponível
                icon_margin = pix_square_size * 0.15
                pygame.draw.rect(
                    canvas,
                    (255, 255, 255),  # Branco para o ícone
                    pygame.Rect(
                        (resource["position"] * pix_square_size + icon_margin, icon_margin),
                        (pix_square_size - 2 * icon_margin, pix_square_size * 0.3),
                    ),
                )
            else:
                # Desenhar o recurso como um quadrado vazio com borda tracejada
                # Primeiro, desenhe um quadrado cinza claro como fundo
                pygame.draw.rect(
                    canvas,
                    (200, 200, 200),  # Cinza claro
                    pygame.Rect(
                        (resource["position"] * pix_square_size, 0),
                        (pix_square_size, pix_square_size),
                    ),
                )
                
                # Adicionar uma borda tracejada para indicar que o recurso está indisponível
                dash_length = 4
                dash_width = 2
                dash_color = (100, 100, 100)  # Cinza escuro
                
                x = resource["position"] * pix_square_size
                y = 0
                width = pix_square_size
                height = pix_square_size
                
                # Desenhar linhas tracejadas horizontais (superior e inferior)
                for i in range(0, int(width), dash_length * 2):
                    # Linha superior
                    pygame.draw.line(
                        canvas,
                        dash_color,
                        (x + i, y),
                        (x + min(i + dash_length, width), y),
                        dash_width
                    )
                    # Linha inferior
                    pygame.draw.line(
                        canvas,
                        dash_color,
                        (x + i, y + height),
                        (x + min(i + dash_length, width), y + height),
                        dash_width
                    )
                
                # Desenhar linhas tracejadas verticais (esquerda e direita)
                for i in range(0, int(height), dash_length * 2):
                    # Linha esquerda
                    pygame.draw.line(
                        canvas,
                        dash_color,
                        (x, y + i),
                        (x, y + min(i + dash_length, height)),
                        dash_width
                    )
                    # Linha direita
                    pygame.draw.line(
                        canvas,
                        dash_color,
                        (x + width, y + i),
                        (x + width, y + min(i + dash_length, height)),
                        dash_width
                    )
                
                # Adicionar um X no centro para indicar indisponibilidade
                x_margin = pix_square_size * 0.2
                pygame.draw.line(
                    canvas,
                    dash_color,
                    (x + x_margin, y + x_margin),
                    (x + width - x_margin, y + height - x_margin),
                    dash_width
                )
                pygame.draw.line(
                    canvas,
                    dash_color,
                    (x + x_margin, y + height - x_margin),
                    (x + width - x_margin, y + x_margin),
                    dash_width
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
        
        # Exibir posição atual do agente
        position_title = label_font.render(f"Posição: {self.agent_info['position']}", True, (0, 0, 0))
        canvas.blit(position_title, (250, grid_height + 20))
        
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
            
            # Mostrar diferença em relação ao estado anterior
            diff = state_value - self.previous_internal_states[i]
            if abs(diff) > 0.001:  # Se houver uma diferença significativa
                diff_text = f"{diff:+.2f}"  # Formato com sinal
                if diff > 0:
                    diff_color = (0, 150, 0)  # Verde para aumento
                else:
                    diff_color = (150, 0, 0)  # Vermelho para diminuição
                
                diff_label = value_font.render(diff_text, True, diff_color)
                canvas.blit(diff_label, (150 + bar_width + 10, y_pos + bar_height // 2 - 8))
        
        # Mostrar legenda dos recursos
        legend_y = info_start_y + 50 + self.dimension_internal_states * bar_spacing + 30
        legend_title = label_font.render("Recursos:", True, (0, 0, 0))
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
            resource = self.resources_info[i]
            
            if resource["available"]:
                # Recurso disponível: quadrado colorido
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
                status_text = "disponível"
            else:
                # Recurso indisponível: quadrado cinza com X
                pygame.draw.rect(
                    canvas,
                    (200, 200, 200),
                    pygame.Rect(legend_x, legend_y, 20, 20),
                )
                pygame.draw.rect(
                    canvas,
                    (100, 100, 100),
                    pygame.Rect(legend_x, legend_y, 20, 20),
                    width=1
                )
                # Desenhar X
                pygame.draw.line(
                    canvas,
                    (100, 100, 100),
                    (legend_x + 4, legend_y + 4),
                    (legend_x + 16, legend_y + 16),
                    2
                )
                pygame.draw.line(
                    canvas,
                    (100, 100, 100),
                    (legend_x + 4, legend_y + 16),
                    (legend_x + 16, legend_y + 4),
                    2
                )
                status_text = "indisponível"
            
            # Nome do recurso
            name_text = value_font.render(f"{state_name} ({status_text})", True, (0, 0, 0))
            canvas.blit(name_text, (legend_x + 25, legend_y + 2))
            
            # Posição do recurso
            pos_text = value_font.render(f"Pos: {resource['position']}", True, (0, 0, 0))
            canvas.blit(pos_text, (legend_x + 25, legend_y + 22))

        # Adicionar informações sobre as ações disponíveis
        actions_y = legend_y + 60
        actions_title = label_font.render("Ações:", True, (0, 0, 0))
        canvas.blit(actions_title, (20, actions_y))
        
        # Listar as ações disponíveis
        action_texts = [
            "0: Ficar parado",
            "1: Mover para esquerda",
            "2: Mover para direita"
        ]
        
        # Adicionar ações de consumo
        for i, state_name in enumerate(state_names):
            action_texts.append(f"{3 + i}: Consumir {state_name}")
        
        # Mostrar textos das ações
        for i, text in enumerate(action_texts):
            action_text = value_font.render(text, True, (0, 0, 0))
            if i < 3:  # Primeiras ações na primeira coluna
                canvas.blit(action_text, (20, actions_y + 25 + i * 20))
            else:  # Ações de consumo na segunda coluna
                canvas.blit(action_text, (250, actions_y + 25 + (i - 3) * 20))

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
    env = LimitedResources1D(render_mode="human", config_path=config_path, drive_type=drive_type, size=10)
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
