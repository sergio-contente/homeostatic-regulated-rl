import numpy as np
import os
import sys
from src.custom_env.one_dimensional import HomeoEnv1D

# Métodos de visualização
from src.custom_env.training_visualization import (
    setup_training_visualization,
    update_training_visualization,
    _draw_info_panel,
    _action_to_text,
    _draw_training_graphs,
    _draw_states_graph,
    _draw_drive_graph,
    _draw_reward_graph,
    _draw_actions_position_graph,
    _draw_environment_representation,
    reset_training_stats,
    train_with_visualization
)

def main():
    # === Parameters ===
    maxh = 5
    config_path = "config/config.yaml"
    drive_type = "base_drive"  # Ex: "base_drive", "elliptic_drive", "interoceptive_drive"
    num_episodes = 2000

    # === Create environment ===
    env = HomeoEnv1D(config_path=config_path, drive_type=drive_type, render_mode=None, maxh=maxh)

    # === Attach visualization methods to environment ===
    HomeoEnv1D.setup_training_visualization = setup_training_visualization
    HomeoEnv1D.update_training_visualization = update_training_visualization
    HomeoEnv1D._draw_info_panel = _draw_info_panel
    HomeoEnv1D._action_to_text = _action_to_text
    HomeoEnv1D._draw_training_graphs = _draw_training_graphs
    HomeoEnv1D._draw_states_graph = _draw_states_graph
    HomeoEnv1D._draw_drive_graph = _draw_drive_graph
    HomeoEnv1D._draw_reward_graph = _draw_reward_graph
    HomeoEnv1D._draw_actions_position_graph = _draw_actions_position_graph
    HomeoEnv1D._draw_environment_representation = _draw_environment_representation
    HomeoEnv1D.reset_training_stats = reset_training_stats
    HomeoEnv1D.train_with_visualization = train_with_visualization

    # === Ask user for visualization ===
    use_visualization = input("Deseja usar visualização durante o treinamento? (s/n): ").lower().startswith('s')

    # === Train agent ===
    if use_visualization:
        print("Treinando com visualização em tempo real...")
        rewards = env.train_with_visualization(num_episodes=num_episodes, render_interval=1)
    else:
        print("Treinando sem visualização...")
        rewards = env.train(num_episodes=num_episodes)

    # === Save Q-table ===
    os.makedirs("models/custom/HomeoEnv1D", exist_ok=True)
    np.save("models/custom/HomeoEnv1D/q_table.npy", env.q_table)

    # === Plot rewards ===
    env.plot_rewards(rewards)

    # === Evaluate agent ===
    print("Iniciando avaliação...")
    env_eval = HomeoEnv1D(config_path=config_path, drive_type=drive_type, render_mode="human", maxh=maxh)
    
    # Copia a tabela treinada
    env_eval.q_table = np.copy(env.q_table)
    env_eval.epsilon = 0.0  # Greedy para avaliação

    for episode in range(10):
        state, _ = env_eval.reset()
        state_idx = env_eval.state_to_idx[tuple(state)]
        done = False
        truncated = False
        total_reward = 0

        while not done and not truncated:
            action = env_eval.select_action(state_idx)
            next_state, reward, done, truncated, _ = env_eval.step(action)
            state_idx = env_eval.state_to_idx[tuple(next_state)]
            total_reward += reward

        print(f"[Avaliação] Episódio {episode+1}: Recompensa total = {total_reward:.2f}")

if __name__ == "__main__":
    main()
