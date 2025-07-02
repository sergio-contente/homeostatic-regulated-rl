#!/usr/bin/env python3
"""
Demonstração do sistema de logging detalhado do NORMARL.
Este script mostra como usar os novos logs avançados do TensorBoard.
"""

from train_ppo_normarl import train_ppo_normarl, evaluate_model
import numpy as np
import os
import time

def demonstrate_detailed_logging():
    """
    Demonstra o sistema de logging detalhado com métricas específicas do NORMARL.
    """
    print("🔍 Demonstração de Logging Detalhado do NORMARL")
    print("=" * 60)
    
    # Configuração para demonstração
    config = {
        'config_path': 'config/config.yaml',
        'drive_type': 'base_drive',
        'learning_rate': 0.15,      # Taxa de aprendizado social mais alta para observar mudanças
        'beta': 0.8,                # Forte internalização de normas sociais
        'number_resources': 1,      # Um tipo de recurso (comida)
        'n_agents': 3,              # Três agentes para observar interação social
        'size': 1,                  # Ambiente simples
        'total_timesteps': 25000,   # Treinamento rápido para demonstração
        'n_envs': 1,               # Um ambiente para logs claros
        'eval_freq': 2500,         # Avaliação frequente
        'save_freq': 5000,         # Salvar checkpoints
        'log_dir': 'demo_logs/detailed_normarl',
        'model_dir': 'demo_models/detailed_normarl', 
        'tensorboard_log': 'demo_tensorboard/detailed_normarl'
    }
    
    # Configuração PPO otimizada para observar aprendizado
    ppo_config = {
        'ppo_lr': 5e-4,      # Learning rate ligeiramente maior
        'n_steps': 512,      # Passos menores para updates mais frequentes
        'batch_size': 32,    # Batch menor para demonstração
        'n_epochs': 8,       # Epochs reduzidas
        'ent_coef': 0.02,   # Entropia maior para exploração
    }
    
    print("📊 Configuração de Demonstração:")
    print(f"  Agentes: {config['n_agents']}")
    print(f"  Beta (força das normas sociais): {config['beta']}")
    print(f"  Taxa de aprendizado social: {config['learning_rate']}")
    print(f"  Timesteps: {config['total_timesteps']:,}")
    print(f"  Logs TensorBoard: {config['tensorboard_log']}")
    
    print("\n🚀 Iniciando treinamento com logging detalhado...")
    start_time = time.time()
    
    # Treinar com logging detalhado
    model, callback = train_ppo_normarl(**config, **ppo_config)
    
    training_time = time.time() - start_time
    print(f"\n✅ Treinamento concluído em {training_time:.1f} segundos")
    
    # Mostrar estatísticas de treinamento
    if callback.episode_rewards:
        rewards = callback.episode_rewards
        print(f"\n📈 Estatísticas de Treinamento:")
        print(f"  Episódios completados: {len(rewards)}")
        print(f"  Recompensa média: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
        print(f"  Melhor episódio: {np.max(rewards):.3f}")
        print(f"  Pior episódio: {np.min(rewards):.3f}")
        print(f"  Tendência (últimos vs primeiros 10): {np.mean(rewards[-10:]):.3f} vs {np.mean(rewards[:10]):.3f}")
    
    # Avaliação final
    print(f"\n🎯 Avaliação final do modelo...")
    model_path = os.path.join(config['model_dir'], 'final_model')
    
    if os.path.exists(model_path + '.zip'):
        results = evaluate_model(
            model_path=model_path,
            config_path=config['config_path'],
            drive_type=config['drive_type'],
            learning_rate=config['learning_rate'],
            beta=config['beta'],
            number_resources=config['number_resources'],
            n_agents=config['n_agents'],
            size=config['size'],
            n_episodes=10,
            render=False
        )
        
        print(f"\n📊 Resultados de Avaliação:")
        print(f"  Recompensa média: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
        print(f"  Duração média dos episódios: {results['mean_length']:.1f}")
        print(f"  Melhor episódio: {results['rewards'].index(max(results['rewards'])) + 1} com {max(results['rewards']):.3f}")
    
    # Instruções para visualizar logs
    print(f"\n🔍 Como visualizar os logs detalhados:")
    print(f"  1. Execute: tensorboard --logdir {config['tensorboard_log']}")
    print(f"  2. Abra http://localhost:6006 no navegador")
    print(f"  3. Explore as seguintes métricas:")
    print(f"     📈 SCALARS > normarl/: Métricas de episódio e recursos")
    print(f"     👥 SCALARS > agents/: Estados individuais dos agentes")
    print(f"     🧠 SCALARS > agents/*/states/: Estados internos (fome, etc.)")
    print(f"     🤝 SCALARS > agents/*/social_norms/: Percepção de normas sociais")
    print(f"     💰 SCALARS > agents/*/consumption/: Consumo por tipo de recurso")
    print(f"     ⚖️ SCALARS > agents/*/social_cost: Custos sociais")
    print(f"     🎯 SCALARS > agents/*/drive: Nível de urgência homeostática")
    
    print(f"\n📁 Arquivos salvos:")
    print(f"  Modelo final: {model_path}")
    print(f"  Métricas: {os.path.join(config['model_dir'], 'training_metrics.npy')}")
    print(f"  Logs TensorBoard: {config['tensorboard_log']}")
    
    return model, callback


def compare_social_learning_rates():
    """
    Compara diferentes taxas de aprendizado social para demonstrar o logging.
    """
    print("\n🧪 Experimento: Comparação de Taxas de Aprendizado Social")
    print("=" * 60)
    
    learning_rates = [0.05, 0.1, 0.2]  # Baixa, média, alta
    results = {}
    
    for lr in learning_rates:
        print(f"\n📊 Testando taxa de aprendizado social: {lr}")
        
        config = {
            'config_path': 'config/config.yaml',
            'drive_type': 'base_drive',
            'learning_rate': lr,
            'beta': 0.6,
            'number_resources': 1,
            'n_agents': 3,
            'size': 1,
            'total_timesteps': 15000,
            'n_envs': 1,
            'eval_freq': 5000,
            'model_dir': f'demo_models/lr_experiment_{lr}',
            'tensorboard_log': f'demo_tensorboard/lr_experiment_{lr}'
        }
        
        model, callback = train_ppo_normarl(**config)
        
        if callback.episode_rewards:
            final_performance = np.mean(callback.episode_rewards[-5:])
            results[lr] = {
                'final_reward': final_performance,
                'total_episodes': len(callback.episode_rewards),
                'callback': callback
            }
            print(f"  Recompensa final média: {final_performance:.3f}")
    
    # Análise comparativa
    print(f"\n📈 Análise Comparativa:")
    print("-" * 40)
    best_lr = max(results.keys(), key=lambda x: results[x]['final_reward'])
    
    for lr, result in results.items():
        performance = "🏆" if lr == best_lr else "  "
        print(f"  {performance} LR {lr:4.2f}: {result['final_reward']:6.3f} recompensa, {result['total_episodes']:3d} episódios")
    
    print(f"\n💡 Interpretação:")
    print(f"  LR baixa (0.05): Aprendizado social lento, adaptação gradual")
    print(f"  LR média (0.10): Equilíbrio entre estabilidade e adaptação")  
    print(f"  LR alta (0.20): Adaptação rápida, possível instabilidade")
    print(f"  Melhor desempenho: LR = {best_lr}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Demonstração de Logging Detalhado NORMARL")
    parser.add_argument("--demo", type=str, default="basic", 
                       choices=["basic", "comparison", "both"],
                       help="Tipo de demonstração")
    
    args = parser.parse_args()
    
    if args.demo in ["basic", "both"]:
        model, callback = demonstrate_detailed_logging()
    
    if args.demo in ["comparison", "both"]:
        comparison_results = compare_social_learning_rates()
    
    print(f"\n🎉 Demonstração concluída!")
    print(f"Explore os logs detalhados com TensorBoard para ver as métricas em tempo real.") 
