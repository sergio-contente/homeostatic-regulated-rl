import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import src.gymnasium_env

config_path = "config/config.yaml"
drive_type = "elliptic_drive"  # ou "elliptic_drive"

# Criar um diretório único para logs desta execução
current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = os.path.join('runs', f'DQN_GridWorld_{drive_type}_{current_time}')
writer = SummaryWriter(log_dir)

# Criar o ambiente e acessar o ambiente base através do wrapper TimeLimit
env = gym.make("GridWorld-v0", config_path=config_path, drive_type=drive_type)

# Obter o ambiente base (GridWorldEnv) se estiver envolvido por wrappers
def get_unwrapped_env(env):
    """Obtém o ambiente base, ignorando wrappers como TimeLimit"""
    if hasattr(env, 'env'):
        return get_unwrapped_env(env.env)
    return env

base_env = get_unwrapped_env(env)

# Configurar matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# Verificar se GPU está disponível
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Usando dispositivo: {device}")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Salvar uma transição"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Chamado com um elemento para determinar próxima ação, ou um batch
    # durante otimização. Retorna tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# BATCH_SIZE é o número de transições amostradas do buffer de replay
# GAMMA é o fator de desconto
# EPS_START é o valor inicial de epsilon
# EPS_END é o valor final de epsilon
# EPS_DECAY controla a taxa de decaimento exponencial de epsilon
# TAU é a taxa de atualização da rede alvo
# LR é a taxa de aprendizado do otimizador AdamW
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Obter número de ações do espaço de ações do gym
n_actions = env.action_space.n

# Função auxiliar para converter observação em dicionário para tensor
def process_observation(observation):
    """Converter observação em dicionário para array plano para a rede neural"""
    # Extrair posição e estados internos do dicionário de observação
    position = float(observation['position'])  # Converter para float
    internal_states = observation['internal_states'].astype(np.float32)  # Garantir float32
    
    # Concatenar posição e estados internos em um array plano
    flat_observation = np.concatenate(([position], internal_states))
    
    return flat_observation, position, internal_states

# Obter o número de observações de estado
state, info = env.reset()
processed_state, position, internal_states = process_observation(state)
n_observations = len(processed_state)

# Obter nomes dos estados internos
try:
    # Acessar o atributo drive diretamente no ambiente base
    state_names = base_env.drive.get_internal_states_names()
    print(f"Name of internal states: {state_names}")
except AttributeError:
    state_names = [f"state_{i}" for i in range(len(internal_states))]

num_internal_states = len(internal_states)
print(f"Number of internal states: {num_internal_states}")
print(f"Internal states of initial values: {internal_states}")

# Obter valor do drive
try:
    current_drive = base_env.drive.get_current_drive()
    print(f"Value of initial drive: {current_drive:.4f}")
except AttributeError:
    print("Warning: It was not possible to get the current drive value")
    current_drive = None

# Registrar informações no TensorBoard
writer.add_text('Environment/Action Space', f'Number of actions: {n_actions}')
writer.add_text('Environment/Observation Space', f'Observation dimensions: {n_observations}')
writer.add_text('Environment/Internal States', f'Number of internal states: {num_internal_states}')
writer.add_text('Environment/State Names', f'States names: {", ".join(state_names)}')

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

# Registrar arquitetura da rede como grafo
dummy_input = torch.zeros(1, n_observations, device=device)
writer.add_graph(policy_net, dummy_input)

steps_done = 0
total_steps = 0  # Contador global de passos em todos os episódios


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    # Registrar valor de epsilon
    writer.add_scalar('Training/Epsilon', eps_threshold, steps_done)
    
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) retornará o maior valor de coluna de cada linha.
            # A segunda coluna do resultado max é o índice onde o elemento max foi
            # encontrado, então escolhemos a ação com a maior recompensa esperada.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Calcular médias de 100 episódios e plotá-las também
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # Pausa para atualizar os gráficos
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpor o batch (veja https://stackoverflow.com/a/19343/3343043 para
    # explicação detalhada). Isso converte batch-array de Transitions
    # para Transition de batch-arrays.
    batch = Transition(*zip(*transitions))

    # Calcular uma máscara de estados não-finais e concatenar os elementos do batch
    # (um estado final seria aquele após o qual a simulação terminou)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Calcular Q(s_t, a) - o modelo calcula Q(s_t), então selecionamos as
    # colunas das ações tomadas. Essas são as ações que teriam sido tomadas
    # para cada estado do batch de acordo com policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Calcular V(s_{t+1}) para todos os próximos estados.
    # Os valores esperados das ações para non_final_next_states são calculados com base
    # na "antiga" target_net; selecionando sua melhor recompensa com max(1).values
    # Isso é mesclado com base na máscara, de modo que teremos o valor de estado esperado
    # ou 0 caso o estado seja final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Calcular os valores Q esperados
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Calcular a perda de Huber
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Registrar a perda
    global total_steps
    writer.add_scalar('Training/Loss', loss.item(), total_steps)
    
    # Otimizar o modelo
    optimizer.zero_grad()
    loss.backward()
    # Clipping de gradiente in-place
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return loss.item()

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 200
else:
    num_episodes = 50

# Rastrear recompensas cumulativas para cada episódio
episode_rewards = []

for i_episode in range(num_episodes):
    # Inicializar o ambiente e obter seu estado
    state, info = env.reset()
    
    # Processar o estado de dicionário em um array plano e converter para tensor
    processed_state, position, internal_states = process_observation(state)
    state_tensor = torch.tensor(processed_state, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Obter e registrar o valor atual do drive
    try:
        current_drive = base_env.drive.get_current_drive()
        writer.add_scalar('States/drive', current_drive, total_steps)
    except AttributeError:
        current_drive = None
    
    # Registrar os estados internos iniciais
    for i, value in enumerate(internal_states):
        state_name = state_names[i] if i < len(state_names) else f"state_{i}"
        writer.add_scalar(f'States/{state_name}', value, total_steps)
    
    # Registrar posição inicial
    writer.add_scalar('States/position', position, total_steps)
    
    # Rastrear recompensas para este episódio
    episode_reward = 0
    episode_loss = 0
    
    for t in count():
        total_steps += 1  # Incrementar contador global de passos
        
        action = select_action(state_tensor)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        
        # Processar nova observação
        processed_obs, new_position, new_internal_states = process_observation(observation)
        
        # Obter e registrar o novo valor do drive após a ação
        try:
            new_drive = base_env.drive.get_current_drive()
            writer.add_scalar('States/drive', new_drive, total_steps)
        except AttributeError:
            new_drive = None
        
        # Registrar estados internos após a ação
        for i, value in enumerate(new_internal_states):
            state_name = state_names[i] if i < len(state_names) else f"state_{i}"
            writer.add_scalar(f'States/{state_name}', value, total_steps)
        
        # Registrar nova posição
        writer.add_scalar('States/position', new_position, total_steps)
        
        # Rastrear recompensa
        episode_reward += reward
        
        # Registrar a recompensa deste passo
        writer.add_scalar('Training/StepReward', reward, total_steps)
        
        reward_tensor = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(processed_obs, dtype=torch.float32, device=device).unsqueeze(0)

        # Armazenar a transição na memória
        memory.push(state_tensor, action, next_state, reward_tensor)

        # Mover para o próximo estado
        state_tensor = next_state

        # Realizar um passo de otimização (na rede de política)
        loss = optimize_model()
        if loss is not None:
            episode_loss += loss

        # Atualização suave dos pesos da rede alvo
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            episode_rewards.append(episode_reward)
            
            # Registrar métricas do episódio
            writer.add_scalar('Training/EpisodeDuration', t + 1, i_episode)
            writer.add_scalar('Training/EpisodeReward', episode_reward, i_episode)
            
            # Registrar valor final do drive
            if new_drive is not None:
                writer.add_scalar('Training/FinalDrive', new_drive, i_episode)
            
            # Registrar a perda média do episódio
            if t > 0:  # Evitar divisão por zero
                avg_loss = episode_loss / t
                writer.add_scalar('Training/AverageEpisodeLoss', avg_loss, i_episode)
            
            # Adicionar histogramas dos parâmetros da rede
            if i_episode % 10 == 0:  # Registrar histogramas apenas a cada 10 episódios para economizar espaço
                for name, param in policy_net.named_parameters():
                    writer.add_histogram(f'PolicyNet/{name}', param.data, i_episode)
            
            plot_durations()
            
            # Imprimir progresso
            print(f"Episode {i_episode}/{num_episodes}, Duration: {t+1}, Reward: {episode_reward:.2f}")
            
            # Mostrar valores dos estados internos e drive no final do episódio
            if new_drive is not None:
                print(f"  Final Drive: {new_drive:.4f}")
            
            for i, state_name in enumerate(state_names):
                if i < len(new_internal_states):
                    print(f"  {state_name}: {new_internal_states[i]:.4f}")
            
            break

print('Completo')
plot_durations(show_result=True)
plt.ioff()
plt.show()

# Fechar o writer
writer.close()

# Salvar o modelo treinado
model_save_path = os.path.join(log_dir, 'dqn_model.pth')
torch.save({
    'policy_net': policy_net.state_dict(),
    'target_net': target_net.state_dict(),
    'optimizer': optimizer.state_dict(),
}, model_save_path)
print(f"Model saved in {model_save_path}")
