#!/usr/bin/env python3
"""
MAPPO Evaluation Script for Homeostatic Environment

This script evaluates trained MAPPO agents on the tragedy of commons scenario
and compares their performance with baseline behaviors (greedy, abstain, smart).

Key metrics:
- Resource sustainability
- Agent survival rates
- Cooperation levels
- Social norm compliance
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import json
from dataclasses import asdict

# Import your environment and agents
from src.envs.multiagent import create_env
from src.scripts.pettingzoo_env.train_mappo_enhanced import EnhancedAgent, EnhancedArgs
from pettingzoo.utils.conversions import aec_to_parallel
import supersuit as ss


class MAPPOEvaluator:
    """Evaluates trained MAPPO agents."""
    
    def __init__(self, model_path: str, args: EnhancedArgs):
        """Initialize evaluator with trained model."""
        self.model_path = model_path
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load trained agent
        self.agent = self._load_agent()
        
        # Evaluation metrics
        self.eval_results = {}
        
    def _load_agent(self) -> EnhancedAgent:
        """Load trained MAPPO agent from checkpoint."""
        print(f"Loading model from {self.model_path}")
        
        # Create temporary environment to get spaces
        temp_env = create_env(
            config_path=self.args.config_path,
            drive_type=self.args.drive_type,
            learning_rate=self.args.learning_rate_social,
            beta=self.args.beta,
            number_resources=self.args.number_resources,
            n_agents=self.args.n_agents,
            size=self.args.env_size,
            max_steps=self.args.max_steps,
            initial_resource_stock=self.args.initial_resource_stock
        )
        
        # Convert to vector env format for agent initialization
        temp_env = aec_to_parallel(temp_env)
        temp_env = ss.pettingzoo_env_to_vec_env_v1(temp_env)
        temp_env = ss.concat_vec_envs_v1(temp_env, 1, num_cpus=1, base_class="stable_baselines3")
        
        # Create agent
        agent = EnhancedAgent(temp_env, self.args).to(self.device)
        
        # Load trained weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.eval()
        
        temp_env.close()
        print(f"✅ Model loaded successfully from iteration {checkpoint.get('final_iteration', 'unknown')}")
        
        return agent
    
    def evaluate_episode(self, env, max_steps: int = 1000) -> Dict[str, Any]:
        """Evaluate one episode and return metrics."""
        
        episode_data = {
            'total_reward': 0,
            'episode_length': 0,
            'final_resource_stock': 0,
            'agent_survival_count': 0,
            'resource_consumption': [],
            'social_costs': [],
            'agent_states_history': [],
            'cooperation_score': 0,
        }
        
        obs = env.reset()
        done = False
        step_count = 0
        
        # Track per-step metrics
        step_rewards = []
        step_consumptions = []
        step_social_costs = []
        
        while not done and step_count < max_steps:
            if not env.agents:
                break
                
            # Get actions from all agents
            actions = {}
            for agent_id in env.agents:
                if agent_id in obs:
                    agent_obs = obs[agent_id]
                    
                    # Convert observation to tensor
                    if isinstance(agent_obs, dict):
                        obs_tensor = torch.cat([
                            torch.tensor(v).flatten() if hasattr(v, '__len__') else torch.tensor([v])
                            for v in agent_obs.values()
                        ]).float().unsqueeze(0).to(self.device)
                    else:
                        obs_tensor = torch.tensor(agent_obs).float().unsqueeze(0).to(self.device)
                    
                    # Get action from MAPPO agent
                    with torch.no_grad():
                        action, _, _, _ = self.agent.get_action_and_value(obs_tensor)
                        actions[agent_id] = action.item()
                else:
                    actions[agent_id] = 0  # Default action
            
            # Step environment
            obs, rewards, terminations, truncations, infos = env.step(actions)
            
            # Collect metrics
            step_reward = sum(rewards.values()) if rewards else 0
            step_rewards.append(step_reward)
            
            # Check if episode is done
            done = all(terminations.values()) or all(truncations.values()) or len(env.agents) == 0
            step_count += 1
        
        # Calculate final metrics
        episode_data['total_reward'] = sum(step_rewards)
        episode_data['episode_length'] = step_count
        episode_data['agent_survival_count'] = len(env.agents)
        
        # Resource sustainability (if accessible)
        if hasattr(env, 'resource_stock'):
            episode_data['final_resource_stock'] = env.resource_stock[0] if len(env.resource_stock) > 0 else 0
        
        # Calculate cooperation score (high survival + resource preservation)
        initial_agents = self.args.n_agents
        survival_rate = episode_data['agent_survival_count'] / initial_agents
        resource_preservation = episode_data['final_resource_stock'] / self.args.initial_resource_stock
        episode_data['cooperation_score'] = (survival_rate + resource_preservation) / 2
        
        return episode_data
    
    def evaluate_multiple_episodes(self, n_episodes: int = 50) -> Dict[str, Any]:
        """Evaluate multiple episodes and aggregate results."""
        print(f"🔬 Evaluating MAPPO agent over {n_episodes} episodes...")
        
        all_results = []
        
        for episode in range(n_episodes):
            if episode % 10 == 0:
                print(f"Episode {episode}/{n_episodes}")
            
            # Create fresh environment for each episode
            env = create_env(
                config_path=self.args.config_path,
                drive_type=self.args.drive_type,
                learning_rate=self.args.learning_rate_social,
                beta=self.args.beta,
                number_resources=self.args.number_resources,
                n_agents=self.args.n_agents,
                size=self.args.env_size,
                max_steps=self.args.max_steps,
                seed=42 + episode,  # Different seed per episode
                initial_resource_stock=self.args.initial_resource_stock
            )
            
            # Convert to parallel for evaluation
            env = aec_to_parallel(env)
            
            episode_result = self.evaluate_episode(env)
            all_results.append(episode_result)
            
            env.close()
        
        # Aggregate results
        aggregated = {
            'total_reward': {
                'mean': np.mean([r['total_reward'] for r in all_results]),
                'std': np.std([r['total_reward'] for r in all_results]),
                'min': np.min([r['total_reward'] for r in all_results]),
                'max': np.max([r['total_reward'] for r in all_results])
            },
            'episode_length': {
                'mean': np.mean([r['episode_length'] for r in all_results]),
                'std': np.std([r['episode_length'] for r in all_results])
            },
            'agent_survival_count': {
                'mean': np.mean([r['agent_survival_count'] for r in all_results]),
                'std': np.std([r['agent_survival_count'] for r in all_results])
            },
            'final_resource_stock': {
                'mean': np.mean([r['final_resource_stock'] for r in all_results]),
                'std': np.std([r['final_resource_stock'] for r in all_results])
            },
            'cooperation_score': {
                'mean': np.mean([r['cooperation_score'] for r in all_results]),
                'std': np.std([r['cooperation_score'] for r in all_results])
            },
            'success_rate': {
                'resource_preservation': np.mean([1 if r['final_resource_stock'] > self.args.initial_resource_stock * 0.1 else 0 for r in all_results]),
                'agent_survival': np.mean([1 if r['agent_survival_count'] > self.args.n_agents * 0.5 else 0 for r in all_results]),
                'combined_success': np.mean([1 if r['cooperation_score'] > 0.5 else 0 for r in all_results])
            }
        }
        
        self.eval_results = aggregated
        return aggregated


def compare_with_baselines(args: EnhancedArgs, mappo_results: Dict) -> Dict[str, Any]:
    """Compare MAPPO performance with baseline strategies."""
    print("\n🏆 Comparing with baseline strategies...")
    
    baselines = ['greedy', 'abstain', 'smart']
    baseline_results = {}
    
    for strategy in baselines:
        print(f"  Testing {strategy} strategy...")
        
        strategy_results = []
        
        for episode in range(20):  # Fewer episodes for baselines
            env = create_env(
                config_path=args.config_path,
                drive_type=args.drive_type,
                learning_rate=args.learning_rate_social,
                beta=args.beta,
                number_resources=args.number_resources,
                n_agents=args.n_agents,
                size=args.env_size,
                max_steps=args.max_steps,
                seed=100 + episode,
                initial_resource_stock=args.initial_resource_stock
            )
            
            # Run baseline strategy
            episode_result = run_baseline_episode(env, strategy, args.max_steps)
            strategy_results.append(episode_result)
            
            env.close()
        
        # Aggregate baseline results
        baseline_results[strategy] = {
            'cooperation_score': np.mean([r['cooperation_score'] for r in strategy_results]),
            'final_resource_stock': np.mean([r['final_resource_stock'] for r in strategy_results]),
            'agent_survival_count': np.mean([r['agent_survival_count'] for r in strategy_results]),
            'total_reward': np.mean([r['total_reward'] for r in strategy_results])
        }
    
    # Add MAPPO results for comparison
    baseline_results['mappo'] = {
        'cooperation_score': mappo_results['cooperation_score']['mean'],
        'final_resource_stock': mappo_results['final_resource_stock']['mean'],
        'agent_survival_count': mappo_results['agent_survival_count']['mean'],
        'total_reward': mappo_results['total_reward']['mean']
    }
    
    return baseline_results


def run_baseline_episode(env, strategy: str, max_steps: int) -> Dict[str, Any]:
    """Run one episode with a baseline strategy."""
    
    env.reset()
    step_count = 0
    total_reward = 0
    
    while env.agents and step_count < max_steps:
        current_agent = env.agent_selection
        
        if current_agent in env.agents:
            # Choose action based on strategy
            if strategy == 'greedy':
                action = 3  # Always consume
            elif strategy == 'abstain':
                action = 0  # Never consume  
            elif strategy == 'smart':
                # Smart strategy based on internal state
                agent = env.homeostatic_agents[current_agent]
                food_state = agent.internal_states[0]
                action = 3 if food_state < 0.02 else 0  # Consume only when needed
            else:
                action = env.action_space(current_agent).sample()
            
            env.step(action)
            total_reward += env.rewards.get(current_agent, 0)
        
        step_count += 1
    
    # Calculate metrics
    survival_count = len(env.agents)
    final_resources = env.resource_stock[0] if hasattr(env, 'resource_stock') and len(env.resource_stock) > 0 else 0
    
    # Calculate cooperation score
    initial_agents = env.n_agents
    survival_rate = survival_count / initial_agents
    resource_preservation = final_resources / env.initial_resource_stock[0]
    cooperation_score = (survival_rate + resource_preservation) / 2
    
    return {
        'total_reward': total_reward,
        'agent_survival_count': survival_count,
        'final_resource_stock': final_resources,
        'cooperation_score': cooperation_score
    }


def plot_comparison_results(baseline_results: Dict[str, Dict], save_path: str = "mappo_evaluation_results.png"):
    """Create comparison plots."""
    strategies = list(baseline_results.keys())
    metrics = ['cooperation_score', 'final_resource_stock', 'agent_survival_count', 'total_reward']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = [baseline_results[strategy][metric] for strategy in strategies]
        
        bars = axes[i].bar(strategies, values, 
                          color=['red' if s == 'greedy' else 'gray' if s == 'abstain' 
                                else 'blue' if s == 'smart' else 'green' for s in strategies])
        
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_ylabel('Value')
        
        # Highlight MAPPO
        for j, strategy in enumerate(strategies):
            if strategy == 'mappo':
                bars[j].set_color('green')
                bars[j].set_alpha(0.8)
        
        # Add value labels on bars
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.01 * max(values), f'{v:.2f}', 
                        ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Comparison plots saved to {save_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate MAPPO on homeostatic environment")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--n_episodes", type=int, default=50, help="Number of episodes to evaluate")
    parser.add_argument("--config_path", type=str, default="config/config.yaml", help="Environment config path")
    parser.add_argument("--save_results", action="store_true", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Create default EnhancedArgs (you might want to load these from the checkpoint)
    eval_args = EnhancedArgs(
        config_path=args.config_path,
        n_agents=10,
        initial_resource_stock=1000.0,
        max_steps=1000
    )
    
    print("🚀 Starting MAPPO Evaluation")
    print(f"Model: {args.model_path}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Agents: {eval_args.n_agents}")
    print(f"Initial Resources: {eval_args.initial_resource_stock}")
    
    # Initialize evaluator
    evaluator = MAPPOEvaluator(args.model_path, eval_args)
    
    # Evaluate MAPPO
    mappo_results = evaluator.evaluate_multiple_episodes(args.n_episodes)
    
    # Print MAPPO results
    print("\n📊 MAPPO Evaluation Results:")
    print(f"Average Cooperation Score: {mappo_results['cooperation_score']['mean']:.3f} ± {mappo_results['cooperation_score']['std']:.3f}")
    print(f"Average Final Resources: {mappo_results['final_resource_stock']['mean']:.1f} ± {mappo_results['final_resource_stock']['std']:.1f}")
    print(f"Average Agent Survival: {mappo_results['agent_survival_count']['mean']:.1f} ± {mappo_results['agent_survival_count']['std']:.1f}")
    print(f"Resource Preservation Rate: {mappo_results['success_rate']['resource_preservation']:.1%}")
    print(f"Agent Survival Rate: {mappo_results['success_rate']['agent_survival']:.1%}")
    print(f"Combined Success Rate: {mappo_results['success_rate']['combined_success']:.1%}")
    
    # Compare with baselines
    comparison_results = compare_with_baselines(eval_args, mappo_results)
    
    print("\n🏆 Comparison with Baselines:")
    for strategy, results in comparison_results.items():
        print(f"\n{strategy.upper()}:")
        print(f"  Cooperation Score: {results['cooperation_score']:.3f}")
        print(f"  Final Resources: {results['final_resource_stock']:.1f}")
        print(f"  Agent Survival: {results['agent_survival_count']:.1f}")
    
    # Create plots
    plot_comparison_results(comparison_results)
    
    # Save results if requested
    if args.save_results:
        results_file = f"mappo_evaluation_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'mappo_detailed': mappo_results,
                'comparison': comparison_results,
                'evaluation_args': asdict(eval_args)
            }, f, indent=2, default=str)
        print(f"💾 Results saved to {results_file}")
    
    # Final assessment
    print("\n🎯 FINAL ASSESSMENT:")
    mappo_score = comparison_results['mappo']['cooperation_score']
    best_baseline = max([comparison_results[s]['cooperation_score'] for s in ['greedy', 'abstain', 'smart']])
    
    if mappo_score > best_baseline:
        improvement = ((mappo_score - best_baseline) / best_baseline) * 100
        print(f"✅ MAPPO outperforms best baseline by {improvement:.1f}%")
        print("🎉 MAPPO successfully learned to avoid the tragedy of commons!")
    else:
        decline = ((best_baseline - mappo_score) / best_baseline) * 100
        print(f"❌ MAPPO underperforms best baseline by {decline:.1f}%")
        print("🔧 Consider adjusting hyperparameters or training longer")


if __name__ == "__main__":
    import time
    main() 
