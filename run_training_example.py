#!/usr/bin/env python3
"""
Example script demonstrating PPO training on NORMARL environment.
Simple usage with predefined parameters.
"""

from train_ppo_normarl import train_ppo_normarl, evaluate_model
import os

def quick_training_example():
    """
    Quick training example with basic parameters.
    """
    print("🚀 Starting quick PPO training example on NORMARL")
    print("=" * 60)
    
    # Basic configuration
    config = {
        'config_path': 'config/config.yaml',
        'drive_type': 'base_drive',
        'learning_rate': 0.1,  # Social learning rate
        'beta': 0.5,           # Social norm strength
        'number_resources': 1,  # Single resource type (food)
        'n_agents': 3,         # Three agents
        'size': 1,             # 1x1 environment
        'total_timesteps': 50000,  # Quick training
        'n_envs': 1,          # 1 environment for debugging
        'eval_freq': 5000,    # Evaluate every 5k steps
        'save_freq': 10000,   # Save every 10k steps
    }
    
    # PPO hyperparameters (note: learning_rate is renamed to avoid conflict)
    ppo_config = {
        'ppo_lr': 3e-4,        # PPO learning rate (renamed from learning_rate)
        'n_steps': 1024,       # Steps per update (reduced for quick training)
        'batch_size': 64,
        'n_epochs': 10,
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("\nPPO Hyperparameters:")
    for key, value in ppo_config.items():
        print(f"  {key}: {value}")
    
    # Train model
    model, callback = train_ppo_normarl(**config, **ppo_config)
    
    print("\n🎉 Training completed!")
    
    # Show training statistics
    if callback.episode_rewards:
        print(f"📈 Training Statistics:")
        print(f"  Episodes completed: {len(callback.episode_rewards)}")
        print(f"  Average reward: {callback.episode_rewards[-10:] if len(callback.episode_rewards) >= 10 else callback.episode_rewards}")
        print(f"  Final avg reward: {sum(callback.episode_rewards[-10:]) / len(callback.episode_rewards[-10:]) if len(callback.episode_rewards) >= 10 else sum(callback.episode_rewards) / len(callback.episode_rewards):.3f}")
    
    # Quick evaluation
    print("\n📊 Running quick evaluation...")
    model_path = os.path.join(config.get('model_dir', 'models/ppo_normarl'), 'final_model')
    
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
            n_episodes=5,
            render=False
        )
        
        print(f"\n✅ Final Results:")
        print(f"Average Reward: {results['mean_reward']:.3f}")
        print(f"Average Episode Length: {results['mean_length']:.1f}")
    else:
        print(f"❌ Model file not found: {model_path}")


def social_norm_experiment():
    """
    Experiment comparing different social norm parameters.
    """
    print("🧪 Social Norm Learning Experiment")
    print("=" * 50)
    
    # Base configuration
    base_config = {
        'config_path': 'config/config.yaml',
        'drive_type': 'base_drive',
        'learning_rate': 0.1,
        'number_resources': 1,
        'n_agents': 3,
        'size': 1,
        'total_timesteps': 30000,
        'n_envs': 2,
        'eval_freq': 10000,
        'save_freq': 15000,
    }
    
    # Different beta values (social norm strength)
    beta_values = [0.0, 0.5, 1.0]
    
    results = {}
    
    for beta in beta_values:
        print(f"\n🔬 Training with beta = {beta} (social norm strength)")
        
        config = base_config.copy()
        config['beta'] = beta
        config['model_dir'] = f'models/ppo_normarl_beta_{beta}'
        config['tensorboard_log'] = f'tensorboard_logs/ppo_normarl_beta_{beta}'
        
        # Train model
        model, callback = train_ppo_normarl(**config)
        
        # Evaluate
        model_path = os.path.join(config['model_dir'], 'final_model')
        if os.path.exists(model_path + '.zip'):
            eval_results = evaluate_model(
                model_path=model_path,
                beta=beta,
                **{k: v for k, v in config.items() if k in ['config_path', 'drive_type', 'learning_rate', 'number_resources', 'n_agents', 'size']},
                n_episodes=5,
                render=False
            )
            results[beta] = eval_results
    
    # Compare results
    print(f"\n📊 Social Norm Experiment Results:")
    print("-" * 50)
    for beta, result in results.items():
        print(f"Beta {beta:3.1f}: Avg Reward = {result['mean_reward']:6.3f} ± {result['std_reward']:.3f}")
    
    print("\n💡 Interpretation:")
    print("  Beta = 0.0: No social norms (purely selfish)")
    print("  Beta = 0.5: Moderate social awareness")
    print("  Beta = 1.0: Strong social norm internalization")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NORMARL PPO Training Examples")
    parser.add_argument("--experiment", type=str, default="quick", 
                       choices=["quick", "social_norms"],
                       help="Which experiment to run")
    
    args = parser.parse_args()
    
    if args.experiment == "quick":
        quick_training_example()
    elif args.experiment == "social_norms":
        social_norm_experiment() 
