import numpy as np
from ...gymnasium_env.envs.grid_world_2_resources import GridWorldEnv2Resources
from ...agents.q_learning import QLearning
from ...gymnasium_env.envs.wrappers.digitize_continuos import DiscretizeWrapper
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def evaluate_agent(agent, env, rewards, internal_state_dim=2, success_threshold=0.0):
    """
    Avaliação visual e numérica do agente Q-Learning.
    
    Parâmetros:
        - agent: instância do QLearning
        - env: ambiente Gymnasium (discretizado)
        - rewards: lista de recompensas por episódio
        - internal_state_dim: número de dimensões do estado interno (default=2)
        - success_threshold: recompensa mínima para considerar sucesso
    """
    
    # === 1. Gráfico de recompensa ===
    rewards_series = pd.Series(rewards)
    rolling_avg = rewards_series.rolling(window=10).mean()
    
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_series, alpha=0.3, label="Recompensa bruta")
    plt.plot(rolling_avg, color="blue", linewidth=2, label="Média móvel (10)")
    plt.xlabel("Episódios")
    plt.ylabel("Recompensa total")
    plt.title("Desempenho do Agente Q-Learning")
    plt.legend()
    plt.grid(True)
    plt.show()

    # === 2. Taxa de sucesso ===
    successes = np.sum(np.array(rewards) >= success_threshold)
    success_rate = successes / len(rewards)
    print(f"✅ Taxa de sucesso (reward >= {success_threshold}): {success_rate:.2%}")
    
    # === 3. Distribuição das ações (política atual) ===
    action_counts = Counter()
    state, _ = env.reset()
    state = agent._process_state(state)
    done = False
    while not done:
        action = agent.get_action(state)
        action_counts[action] += 1
        next_state, _, done, _, _ = env.step(action)
        state = agent._process_state(next_state)
    
    print("📊 Ações tomadas em um episódio de avaliação:")
    for a in range(agent.action_size):
        print(f" - Ação {a}: {action_counts[a]} vezes")

    # === 4. Q-table por ação (heatmaps) ===
    fig, axes = plt.subplots(1, agent.action_size, figsize=(5 * agent.action_size, 5))
    for a in range(agent.action_size):
        try:
            q_vals_action = agent.q_table[:, a].reshape([agent.n_bins] * internal_state_dim)
            ax = axes[a] if agent.action_size > 1 else axes
            im = ax.imshow(q_vals_action, cmap="viridis", origin="lower")
            ax.set_title(f"Ação {a}")
            plt.colorbar(im, ax=ax)
        except:
            print(f"⚠️ Não foi possível plotar Q-table para ação {a} (verifique shape).")
    plt.suptitle("Q-table por ação")
    plt.tight_layout()
    plt.show()

    # === 5. Política ótima (argmax da Q-table) ===
    try:
        policy = np.argmax(agent.q_table, axis=1)
        policy_grid = policy.reshape([agent.n_bins] * internal_state_dim)

        plt.figure(figsize=(6, 5))
        plt.imshow(policy_grid, cmap="Accent", origin="lower")
        plt.title("Política ótima (ação por estado)")
        plt.colorbar(label="Ação")
        plt.xlabel("Dimensão 1")
        plt.ylabel("Dimensão 2")
        plt.grid(False)
        plt.show()
    except:
        print("⚠️ Não foi possível visualizar a política (estado não 2D?)")
# 1. Parâmetros de discretização
n_bins = 20
low, high = 0.0, 300.0

# 2. Criação do ambiente base
config_path = "config/config.yaml"
drive_type = "interoceptive_drive"  #  "elliptic_drive", "interoceptive_drive"
env = GridWorldEnv2Resources(config_path=config_path, drive_type=drive_type)

# 3. Envelopar com DiscretizeWrapper
wrapped_env = DiscretizeWrapper(env, n_bins=n_bins, low=low, high=high)

# 4. Calcular número de estados
internal_state_dim = wrapped_env.observation_space.shape[0]  # Ex: 2 estados internos
state_size = n_bins ** internal_state_dim
action_size = wrapped_env.action_space.n  # Deve ser 3 no seu caso

# 5. Inicializar o agente
agent = QLearning(
    state_size=state_size,
    action_size=action_size,
    n_bins=n_bins,  # necessário para processar o estado corretamente
    learning_rate=0.1,
    discount_factor=0.99,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995
)

# 6. Treinamento
num_episodes = 500
rewards = agent.train(wrapped_env, num_episodes=num_episodes)


# Avaliação visual e estatística
evaluate_agent(agent, wrapped_env, rewards, internal_state_dim=2)


# # 7. (Opcional) Avaliação
# avg_reward = agent.evaluate(wrapped_env, num_episodes=10, render=False)
# print(f"Average evaluation reward: {avg_reward:.2f}")

# # 8. (Opcional) Plotar gráfico de recompensas
# import matplotlib.pyplot as plt
# plt.plot(rewards)
# plt.xlabel("Episódios")
# plt.ylabel("Recompensa total")
# plt.title("Desempenho do Agente Q-Learning")
# plt.grid(True)
# plt.show()

# import matplotlib.pyplot as plt
# import pandas as pd

# rewards_series = pd.Series(rewards)
# rolling_avg = rewards_series.rolling(window=10).mean()

# plt.plot(rewards_series, alpha=0.3, label="Recompensa bruta")
# plt.plot(rolling_avg, color="blue", linewidth=2, label="Média móvel (10)")
# plt.xlabel("Episódios")
# plt.ylabel("Recompensa total")
# plt.title("Desempenho do Agente Q-Learning")
# plt.legend()
# plt.grid(True)
# plt.show()

# import matplotlib.pyplot as plt

# def plot_q_table(agent, internal_state_dim):
#     q_table = agent.q_table
#     n_actions = q_table.shape[1]

#     fig, axes = plt.subplots(1, n_actions, figsize=(5 * n_actions, 5))
#     for a in range(n_actions):
#         q_vals_action = q_table[:, a].reshape([agent.n_bins] * internal_state_dim)

#         ax = axes[a]
#         im = ax.imshow(q_vals_action, cmap="viridis", origin="lower")
#         ax.set_title(f"Ação {a}")
#         plt.colorbar(im, ax=ax)

#     plt.suptitle("Q-table por ação")
#     plt.tight_layout()
#     plt.show()

# def plot_policy(agent, internal_state_dim):
#     policy = np.argmax(agent.q_table, axis=1)
#     policy_grid = policy.reshape([agent.n_bins] * internal_state_dim)

#     plt.imshow(policy_grid, cmap="Accent", origin="lower")
#     plt.title("Política ótima (ação por estado)")
#     plt.colorbar()
#     plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

