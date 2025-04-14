import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from src.scripts.visualizer import load_and_visualize_qlearning_model

# === Nova função para visualização vetorial colorida ===
def plot_policy_quiver(policy_matrix, state_range=(-200, 200)):
    H, W = policy_matrix.shape
    real_s0 = np.linspace(state_range[0], state_range[1], W)
    real_s1 = np.linspace(state_range[0], state_range[1], H)
    X, Y = np.meshgrid(real_s0, real_s1)

    dx = np.zeros_like(policy_matrix, dtype=float)
    dy = np.zeros_like(policy_matrix, dtype=float)
    colors = np.empty(policy_matrix.shape, dtype=object)

    dx[policy_matrix == 0] = +1
    colors[policy_matrix == 0] = 'blue'

    dy[policy_matrix == 1] = +1
    colors[policy_matrix == 1] = 'green'

    dx[policy_matrix == 2] = -1
    colors[policy_matrix == 2] = 'red'

    dy[policy_matrix == 3] = -1
    colors[policy_matrix == 3] = 'orange'

    colors[policy_matrix == 4] = 'lightgray'  # wait

    plt.figure(figsize=(8, 8))
    for action in range(5):
        mask = (policy_matrix == action)
        plt.quiver(X[mask], Y[mask], dx[mask], dy[mask], color=colors[mask], scale=20, angles='xy', label=f'Ação {action}')

    plt.gca().invert_yaxis()
    plt.title("Policy vector field (quiver plot)")
    plt.xlabel("State 0 (Dimension 1)")
    plt.ylabel("State 1 (Dimension 2)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# === Parâmetros ===
max_val = 300.0
model_path = "models/qlearning_model_base_drive_n1_m1.pkl"  # Ajuste o caminho conforme necessário
output_dir = "./visualizacoes"

# === Carregar e visualizar modelo ===
print(f"Carregando modelo Q-learning de: {model_path}")
print(f"Visualizações serão salvas em: {output_dir}")

visualizer = load_and_visualize_qlearning_model(
    model_path=model_path,
    output_dir=output_dir,
    max_val=max_val
)

# === Exibir estatísticas adicionais se o visualizador foi criado com sucesso ===
if visualizer:
    print("\n=== Estatísticas da Política ===")

    # Política ótima (melhor ação) em cada estado
    policy = np.argmax(visualizer.q_table, axis=1)

    # Reestruturar a policy para um grid 2D, assumindo ambiente 2D
    grid_size = int(np.sqrt(len(policy)))
    policy_matrix = policy.reshape((grid_size, grid_size))

    # Plotar o campo vetorial (quiver)
    plot_policy_quiver(policy_matrix, state_range=(-200, 200))

    # Contagem de ações na política ótima
    action_counts = np.bincount(policy, minlength=5)
    total_states = len(policy)

    for action_id, count in enumerate(action_counts):
        percentage = (count / total_states) * 100
        action_name = visualizer.action_names[action_id] if action_id < len(visualizer.action_names) else f"Ação {action_id}"
        print(f"  - {action_name}: {count} estados ({percentage:.1f}%)")

    # Estatísticas de valores Q
    max_q_values = np.max(visualizer.q_table, axis=1)
    print(f"\n=== Estatísticas da Função Valor ===")
    print(f"  - Valor médio: {np.mean(max_q_values):.4f}")
    print(f"  - Valor máximo: {np.max(max_q_values):.4f}")
    print(f"  - Valor mínimo: {np.min(max_q_values):.4f}")

    # Qualidade da política
    q_best = np.max(visualizer.q_table, axis=1)
    q_second_best = np.array([np.sort(row)[-2] if len(row) > 1 else row[0] for row in visualizer.q_table])
    q_gap = q_best - q_second_best

    print(f"\n=== Qualidade da Política ===")
    print(f"  - Diferença média entre melhor e segunda melhor ação: {np.mean(q_gap):.4f}")
    print(f"  - Porcentagem de estados com diferença significativa (>0.1): {np.mean(q_gap > 0.1) * 100:.1f}%")

    try:
        with open(model_path, 'rb') as f:
            loaded_data = pickle.load(f)

        if isinstance(loaded_data, dict) and 'rewards_history' in loaded_data:
            rewards_history = loaded_data['rewards_history']

            print(f"\n=== Análise de Recompensa por Episódio ===")
            print(f"  - Total de episódios: {len(rewards_history)}")
            print(f"  - Recompensa média: {np.mean(rewards_history):.4f}")
            print(f"  - Recompensa máxima: {np.max(rewards_history):.4f}")
            print(f"  - Recompensa mínima: {np.min(rewards_history):.4f}")

            window_size = min(50, len(rewards_history) // 10) if len(rewards_history) > 10 else 1
            moving_avg = np.convolve(rewards_history, np.ones(window_size)/window_size, mode='valid')

            plt.figure(figsize=(10, 6))
            plt.plot(rewards_history, alpha=0.5, label='Recompensa por episódio')
            if len(moving_avg) > 1:
                plt.plot(np.arange(len(moving_avg)) + window_size-1, moving_avg, 'r-', linewidth=2, label=f'Média móvel ({window_size} eps)')

            plt.title('Recompensa durante o Treinamento')
            plt.xlabel('Episódio')
            plt.ylabel('Recompensa Total')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)

            if len(rewards_history) > 1:
                z = np.polyfit(range(len(rewards_history)), rewards_history, 1)
                p = np.poly1d(z)
                plt.plot(range(len(rewards_history)), p(range(len(rewards_history))), "b--", linewidth=1, label=f'Tendência (m={z[0]:.4f})')
                plt.legend()

                last_window = rewards_history[-min(100, len(rewards_history) // 4):]
                variance = np.var(last_window)
                print(f"  - Variância nos últimos episódios: {variance:.4f}")
                print(f"  - Inclinação da tendência: {z[0]:.6f}")

                if abs(z[0]) < 0.01 and variance < 1.0:
                    print("  - Status: O treinamento parece ter convergido")
                elif abs(z[0]) > 0.05:
                    print("  - Status: O treinamento ainda estava em progresso ativo")
                else:
                    print("  - Status: Convergência parcial")

            reward_plot_path = os.path.join(output_dir, "rewards_history.png")
            plt.savefig(reward_plot_path)
            print(f"  - Gráfico de recompensas salvo em: {reward_plot_path}")

            if len(rewards_history) >= 100:
                print(f"\n=== Progresso do Treinamento ===")
                segments = min(5, len(rewards_history) // 100)
                segment_size = len(rewards_history) // segments

                for i in range(segments):
                    start_idx = i * segment_size
                    end_idx = (i + 1) * segment_size if i < segments - 1 else len(rewards_history)
                    segment_rewards = rewards_history[start_idx:end_idx]

                    print(f"  - Episódios {start_idx+1}-{end_idx}: " + 
                          f"Média={np.mean(segment_rewards):.2f}, " +
                          f"Mediana={np.median(segment_rewards):.2f}, " +
                          f"Desvio={np.std(segment_rewards):.2f}")
        else:
            print("\n=== Análise de Recompensa por Episódio ===")
            print("  - Histórico de recompensas não encontrado no modelo.")
            print("  - Verifique se o seu algoritmo de treinamento está salvando o histórico de recompensas.")

    except Exception as e:
        print("\n=== Análise de Recompensa por Episódio ===")
        print(f"  - Erro ao analisar histórico de recompensas: {str(e)}")
        print("  - Verifique se o formato do modelo é compatível com esta análise.")

    print("\nVisualização concluída com sucesso!")
else:
    print("Não foi possível criar o visualizador. Verifique os erros acima.")
