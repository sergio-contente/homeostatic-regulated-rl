import os
import numpy as np
from src.custom_env.one_dimensional import HomeoEnv1D

def analyze_agent():
    """
    Cria e analisa o agente HomeoEnv1D usando os métodos de visualização.
    """
    # === Configurações ===
    maxh = 8
    config_path = "config/config.yaml"
    drive_type = "base_drive"
    q_table_path = "models/custom/HomeoEnv1D/q_table.npy"
    
    # === Criar o Ambiente ===
    env = HomeoEnv1D(
        config_path=config_path,
        drive_type=drive_type,
        maxh=maxh,
        enable_visualization=False,  # Não precisa de visualização para análise
        render_mode=None
    )
    
    # === Carregar a Q-Table se existir ===
    if os.path.exists(q_table_path):
        env.q_table = np.load(q_table_path)
        print(f"Q-table carregada com sucesso: {q_table_path}")
    else:
        print(f"AVISO: Arquivo Q-table não encontrado: {q_table_path}")
        print("Gerando gráficos com Q-table não treinada...")
    
    # === Diretório para salvar as análises ===
    analysis_dir = "images/custom/homeoenv/analysis"
    os.makedirs(analysis_dir, exist_ok=True)
    
    # === Gerar todos os gráficos de análise ===
    print("Gerando gráficos de análise...")
    env.plot_analysis(save_dir=analysis_dir)
    
    print(f"Análise concluída! Gráficos salvos em: {analysis_dir}")
    print("Arquivos gerados:")
    for filename in os.listdir(analysis_dir):
        print(f" - {filename}")

if __name__ == "__main__":
    analyze_agent()
