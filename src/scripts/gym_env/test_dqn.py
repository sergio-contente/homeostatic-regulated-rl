import os
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd

import src.gymnasium_env  # Importa seu ambiente personalizado

# Caminho para o modelo salvo
model_path = "runs/DQN_GridWorld_interoceptive_drive_20250511-223125/dqn_model.pth"


# Configurações do ambiente
config_path = "config/config.yaml"
drive_type = "interoceptive_drive"

# Criar o ambiente sem renderização
env = gym.make("GridWorld-v0", config_path=config_path, drive_type=drive_type, render_mode="human")

# Obter o ambiente base
def get_unwrapped_env(env):
    if hasattr(env, 'env'):
        return get_unwrapped_env(env.env)
    return env

base_env = get_unwrapped_env(env)

# Verificar dispositivo disponível
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Usando dispositivo: {device}")

# Função para processar observação
def process_observation(observation):
    position = float(observation['position'])
    internal_states = observation['internal_states'].astype(np.float32)
    flat_observation = np.concatenate(([position], internal_states))
    return flat_observation, position, internal_states

# Definir a mesma arquitetura de rede usada no treinamento
class DQN(torch.nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = torch.nn.Linear(n_observations, 128)
        self.layer2 = torch.nn.Linear(128, 128)
        self.layer3 = torch.nn.Linear(128, n_actions)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        return self.layer3(x)

# Obter dimensões do espaço de observação e ação
state, info = env.reset()
processed_state, pos, internal = process_observation(state)
n_observations = len(processed_state)
n_actions = env.action_space.n

# Criar a rede e carregar os pesos salvos
policy_net = DQN(n_observations, n_actions).to(device)
checkpoint = torch.load(model_path, map_location=device)
policy_net.load_state_dict(checkpoint['policy_net'])
policy_net.eval()  # Modo de avaliação

print("Modelo carregado com sucesso!")

# Função para selecionar ação (sem exploração durante teste)
def select_action(state):
    with torch.no_grad():
        return policy_net(state).max(1).indices.view(1, 1)

# Criar pasta para os resultados
output_dir = "test_results"
os.makedirs(output_dir, exist_ok=True)

# Função para mapear ação numérica para nomes descritivos
def action_to_name(action):
    # Ajuste para seu caso específico
    if hasattr(base_env, 'action_meanings'):
        return base_env.action_meanings[action]
    
    # Exemplo genérico - substitua pelos valores corretos do seu ambiente
    action_names = ["Esquerda", "Direita", "Ficar"]  
    if action < len(action_names):
        return action_names[action]
    return f"Ação {action}"

# Executar episódios de teste
num_test_episodes = 3
total_rewards = []

for i_episode in range(num_test_episodes):
    state, info = env.reset()
    processed_state, position, internal_states = process_observation(state)
    state_tensor = torch.tensor(processed_state, dtype=torch.float32, device=device).unsqueeze(0)
    
    episode_reward = 0
    done = False
    step = 0
    
    # Tentar obter nomes de estados internos
    try:
        state_names = base_env.drive.get_internal_states_names()
    except AttributeError:
        state_names = [f"state_{i}" for i in range(len(internal_states))]
    
    # Lista para armazenar histórico para plotagem
    drive_history = []
    internal_states_history = {name: [] for name in state_names}
    positions = []
    actions_history = []
    rewards_history = []
    
    # Adicionar valores iniciais
    try:
        current_drive = base_env.drive.get_current_drive()
        drive_history.append(current_drive)
    except AttributeError:
        current_drive = None
        
    positions.append(position)
    for i, state_name in enumerate(state_names):
        if i < len(internal_states):
            internal_states_history[state_name].append(internal_states[i])
    
    while not done and step < 500:  # Limite de 500 passos
        # Selecionar e executar ação
        action = select_action(state_tensor)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        
        # Processar nova observação
        processed_obs, new_position, new_internal_states = process_observation(observation)
        state_tensor = torch.tensor(processed_obs, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Armazenar ação e recompensa
        actions_history.append(action.item())
        rewards_history.append(reward)
        
        # Acumular recompensa
        episode_reward += reward
        
        # Armazenar posição para plotagem
        positions.append(new_position)
        
        # Armazenar informações sobre drive e estados internos para plotagem
        try:
            new_drive = base_env.drive.get_current_drive()
            drive_history.append(new_drive)
            current_drive = new_drive
        except AttributeError:
            pass
        
        # Armazenar estados internos
        for i, state_name in enumerate(state_names):
            if i < len(new_internal_states):
                internal_states_history[state_name].append(new_internal_states[i])
        
        # Verificar se terminou
        done = terminated or truncated
        step += 1
        
        # Mostrar informações a cada 50 passos
        if step % 50 == 0:
            if current_drive is not None:
                print(f"Episódio {i_episode+1}, Passo {step}, Drive: {current_drive:.4f}")
            else:
                print(f"Episódio {i_episode+1}, Passo {step}")
    
    total_rewards.append(episode_reward)
    print(f"Episódio {i_episode+1}/{num_test_episodes} concluído. Recompensa: {episode_reward:.2f}, Passos: {step}")
    
    # Criar uma visualização detalhada da trajetória
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Gráfico da posição
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(positions, label='Posição')
    ax1.set_title('Trajetória do Agente')
    ax1.set_xlabel('Passos')
    ax1.set_ylabel('Posição')
    ax1.grid(True)
    
    # Marcadores para ações
    action_colors = ['r', 'g', 'b', 'y', 'c', 'm']  # Cores para diferentes ações
    for action_type in set(actions_history):
        action_steps = [i for i, a in enumerate(actions_history) if a == action_type]
        if action_steps:
            action_positions = [positions[i+1] for i in action_steps]  # +1 porque a ação i leva à posição i+1
            ax1.scatter(
                [step+1 for step in action_steps],  # +1, pois plotamos as ações nos pontos resultantes
                action_positions,
                color=action_colors[action_type % len(action_colors)],
                alpha=0.5,
                s=30,
                label=f'Ação: {action_to_name(action_type)}'
            )
    ax1.legend()
    
    # 2. Gráfico do drive
    ax2 = fig.add_subplot(2, 2, 2)
    if drive_history:
        ax2.plot(drive_history, label='Drive', color='green')
        ax2.set_title('Valor do Drive')
        ax2.set_xlabel('Passos')
        ax2.set_ylabel('Drive')
        ax2.grid(True)
        
        # Adicionar linha de recompensas acumuladas
        cum_rewards = np.cumsum(rewards_history)
        ax2_reward = ax2.twinx()
        ax2_reward.plot(range(1, len(cum_rewards)+1), cum_rewards, 'r--', label='Recompensa Acumulada')
        ax2_reward.set_ylabel('Recompensa Acumulada', color='r')
        ax2_reward.tick_params(axis='y', labelcolor='r')
        
        # Adicionar duas legendas
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_reward.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 3. Gráfico dos estados internos
    ax3 = fig.add_subplot(2, 2, 3)
    for state_name, values in internal_states_history.items():
        if values:  # Verificar se há valores registrados
            ax3.plot(values, label=state_name)
    ax3.set_title('Estados Internos')
    ax3.set_xlabel('Passos')
    ax3.set_ylabel('Valor')
    ax3.grid(True)
    ax3.legend()
    
    # 4. Gráfico de recompensas por passo
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.bar(range(len(rewards_history)), rewards_history, color='orange', width=1.0)
    ax4.set_title('Recompensas por Passo')
    ax4.set_xlabel('Passos')
    ax4.set_ylabel('Recompensa')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"analysis_episode_{i_episode+1}.png"), dpi=150)
    print(f"Análise salva em: {os.path.join(output_dir, f'analysis_episode_{i_episode+1}.png')}")
    plt.close()
    
    # Salvar dados do episódio em um arquivo CSV
    data = {
        'step': list(range(len(positions)-1)),  # -1 porque temos a posição inicial
        'position': positions[1:],  # Ignorar posição inicial
        'action': actions_history,
        'reward': rewards_history
    }
    
    # Adicionar drive e estados internos se disponíveis
    if drive_history and len(drive_history) > 1:  # Verificar se há mais de um valor (além do inicial)
        data['drive'] = drive_history[1:]  # Ignorar drive inicial
    
    for state_name, values in internal_states_history.items():
        if values and len(values) > 1:  # Verificar se há mais de um valor
            data[state_name] = values[1:]  # Ignorar estado inicial
    
    # Criar e salvar DataFrame
    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, f"data_episode_{i_episode+1}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Dados salvos em: {csv_path}")

print(f"Teste concluído! Recompensa média: {sum(total_rewards)/len(total_rewards):.2f}")

# Fechar o ambiente
env.close()
