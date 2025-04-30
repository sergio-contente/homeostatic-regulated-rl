import os
import numpy as np
import pygame
import imageio
import time
from src.custom_env.one_dimensional import HomeoEnv1D

def save_training_gif(frames, save_path, duration=0.05):
    """
    Salva um GIF a partir de uma lista de surfaces do pygame.
    """
    images = []
    for frame in frames:
        frame_str = pygame.image.tostring(frame, "RGBA", False)
        frame_surface = pygame.image.fromstring(frame_str, frame.get_size(), "RGBA")
        frame_array = pygame.surfarray.array3d(frame_surface)
        frame_array = frame_array.transpose([1, 0, 2])  # Corrigir eixo
        images.append(frame_array)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.mimsave(save_path, images, duration=duration)

    print(f"GIF salvo em: {save_path}")

def main():
    # === Configurações ===
    maxh = 5
    config_path = "config/config.yaml"
    drive_type = "base_drive"
    q_table_path = "models/custom/HomeoEnv1D/q_table.npy"
    num_eval_episodes = 5
    max_steps_per_episode = 1000
    save_gif_path = "images/custom/homeoenv/evaluation.gif"
    
    # Configuração para visualização
    enable_visualization = True
    render_mode = "human"
    delay_between_steps = 0.1  # Atraso entre passos para melhor visualização

    # === Inicializa Pygame ===
    if enable_visualization:
        pygame.init()

    # === Criar o Ambiente ===
    env = HomeoEnv1D(
        config_path=config_path,
        drive_type=drive_type,
        maxh=maxh,
        enable_visualization=enable_visualization,
        render_mode=render_mode
    )

    # === Inicializa visualização ===
    if enable_visualization:
        env.setup_training_visualization()

    # === Carregar a Q-Table ===
    if os.path.exists(q_table_path):
        env.q_table = np.load(q_table_path)
        print(f"Q-table carregada com sucesso de: {q_table_path}")
    else:
        print(f"ERRO: Arquivo Q-table não encontrado em: {q_table_path}")
        return

    rewards = []
    gif_frames = []

    print(f"\n=== Iniciando avaliação da política para {num_eval_episodes} episódios ===")
    
    for episode in range(num_eval_episodes):
        state, _ = env.reset()
        state_idx = env.discretize_state(state)
        total_reward = 0
        episode_frames = []

        print(f"\nEpisódio {episode+1}/{num_eval_episodes}")
        
        # Configurar as variáveis para visualização
        env.current_episode = episode
        env.current_reward = total_reward
        env.current_epsilon = 0.0  # Avaliação usa política greedy (sem exploração)

        for step in range(max_steps_per_episode):
            # Processa eventos pygame para evitar travamento
            if enable_visualization:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        print("Avaliação interrompida pelo usuário.")
                        return
            
            # Selecionar ação usando a política aprendida (greedy)
            action = np.argmax(env.q_table[state_idx])
            
            # Mapear ação para descrição textual para melhor compreensão
            action_desc = {
                0: "ficar parado",
                1: "mover para esquerda",
                2: "mover para direita",
                3: "consumir recurso",
                4: "não consumir"
            }
            
            print(f"  Passo {step}: Estado interno={[round(s, 2) for s in state]}, " 
                  f"Posição={env.agent_position}, Ação={action} ({action_desc[action]})")
            
            # Executar a ação
            next_state, reward, done, truncated, _ = env.step(action)
            next_state_idx = env.discretize_state(next_state)
            
            # Atualizar estado e recompensa
            state = next_state
            state_idx = next_state_idx
            total_reward += reward
            
            # Atualizar variáveis para visualização
            env.current_reward = total_reward
            
            # Renderizar o ambiente explicitamente
            if enable_visualization:
                env.render()
                
                # Capturar o frame para o GIF
                frame = pygame.display.get_surface().copy()
                gif_frames.append(frame)
                episode_frames.append(frame)
                
                # Atraso para melhor visualização
                time.sleep(delay_between_steps)
            
            # Verificar condição de término
            if done:
                print(f"  Episódio terminado com sucesso após {step+1} passos!")
                break
            
            if truncated:
                print(f"  Episódio truncado após {step+1} passos (limite de passos atingido).")
                break
        
        # Estatísticas do episódio
        print(f"  Total de passos: {step+1}")
        print(f"  Recompensa total: {total_reward:.2f}")
        print(f"  Estado interno final: {[round(s, 2) for s in env.current_state]}")
        print(f"  Drive final: {env.drive.get_current_drive():.2f}")
        
        rewards.append(total_reward)
        
        # Salvar GIF para cada episódio individual (opcional)
        episode_gif_path = f"images/custom/homeoenv/episode_{episode+1}.gif"
        if len(episode_frames) > 0:
            save_training_gif(episode_frames, episode_gif_path)

    # Estatísticas finais
    avg_reward = np.mean(rewards)
    print("\n=== Resultados da Avaliação ===")
    print(f"Recompensa média nos {num_eval_episodes} episódios: {avg_reward:.2f}")
    print(f"Recompensas por episódio: {[round(r, 2) for r in rewards]}")
    
    # Análise da política aprendida
    print("\n=== Análise da Política ===")
    optimal_states = []
    total_states = np.prod(env.q_table.shape[:-1])  # Excluindo a dimensão de ação
    
    # Contagem de ações preferidas
    action_counts = {i: 0 for i in range(len(env.action_space))}
    for state_indices in np.ndindex(env.q_table.shape[:-1]):
        best_action = np.argmax(env.q_table[state_indices])
        action_counts[best_action] += 1
    
    print(f"Total de estados discretizados: {total_states}")
    for action, count in action_counts.items():
        percentage = (count / total_states) * 100
        print(f"Ação {action} ({action_desc[action]}) é a melhor em {count} estados ({percentage:.1f}%)")

    # === Salvar o GIF completo ===
    if len(gif_frames) > 0:
        save_training_gif(gif_frames, save_gif_path, duration=0.05)
    
    # Encerrar pygame
    pygame.quit()
    print("\nAvaliação concluída.")

if __name__ == "__main__":
    main()
