import os
import numpy as np
import pygame
import time
from src.custom_env.one_dimensional import HomeoEnv1D

def main():
    # === Configurações do Experimento ===
    maxh = 5
    config_path = "config/config.yaml"
    drive_type = "brase_drive"  # base_drive, elliptic_drive, etc.
    num_episodes = 2000  # Reduzindo para teste
    enable_visualization = False
    
    # === Inicializa Pygame ===
    pygame.init()
    
    # === Criar o Ambiente ===
    env = HomeoEnv1D(
        config_path=config_path, 
        drive_type=drive_type, 
        maxh=maxh, 
        enable_visualization=enable_visualization,
        render_mode="human"  # Ativa a renderização
    )
    
    # === Inicializar Visualização ===
    if enable_visualization:
        env.setup_training_visualization()
    
    # === Testar a visualização antes do treinamento ===
    print("Visualização inicial. Pressione qualquer tecla para iniciar o treinamento...")
    state, _ = env.reset()
    env.render()  # Renderizando explicitamente
    
    # Pausa para ver a visualização inicial
    waiting = True
    while waiting and enable_visualization:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN or event.type == pygame.QUIT:
                waiting = False
        time.sleep(0.1)  # Evitar uso excessivo de CPU
    
    # === Treinar o Agente ===
    print("Iniciando treinamento...")
    try:
        rewards = env.train(num_episodes=num_episodes)
        print("Treinamento concluído com sucesso.")
    except Exception as e:
        print(f"Erro durante o treinamento: {e}")
        import traceback
        traceback.print_exc()
        rewards = []
    
    
    # === Salvar a Q-Table ===
    if len(rewards) > 0:
        save_dir = "models/custom/HomeoEnv1D"
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "q_table.npy"), env.q_table)
        
        # === Plotar as Recompensas ===
        env.plot_rewards(rewards)
        print("Recompensas plotadas.")
    
    # Encerrar pygame adequadamente
    pygame.quit()

if __name__ == "__main__":
    main()
