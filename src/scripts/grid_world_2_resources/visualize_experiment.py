# src/scripts/grid_world_2_resources/visualize_experiment.py

import os
import sys
import pickle
import numpy as np
from src.scripts.grid_world_2_resources.visualizer import load_and_visualize_qlearning_model

# === Parâmetros ===
max_val = 300.0
model_path = "models/qlearning_model_base_drive_n2_m2.pkl"  # Ajuste o caminho conforme necessário
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
    
    # Contagem de ações na política ótima
    action_counts = np.bincount(policy, minlength=3)
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
    
    print("\nVisualização concluída com sucesso!")
else:
    print("Não foi possível criar o visualizador. Verifique os erros acima.")
