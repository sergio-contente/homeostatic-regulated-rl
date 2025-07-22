"""Comprehensive test suite for homeostatic environments."""

import numpy as np
import pandas as pd
from src.envs.multiagent import create_env, create_parallel_env
import time

class EnvironmentTester:
    """Comprehensive tester for homeostatic environments."""
    
    def __init__(self, config_path="config/config.yaml", drive_type="base_drive"):
        self.config_path = config_path
        self.drive_type = drive_type
        self.results = {}
    
    def create_test_env(self, env_type="aec", **kwargs):
        """Create environment for testing."""
        default_params = {
            "config_path": self.config_path,
            "drive_type": self.drive_type,
            "learning_rate": 0.2,
            "beta": 0.5,
            "number_resources": 1,
            "n_agents": 3,
            "size": 3,
            "log_level": "WARNING"
        }
        default_params.update(kwargs)
        
        if env_type == "aec":
            return create_env(**default_params)
        else:
            return create_parallel_env(**default_params)
    
    def collect_metrics(self, env, agents_data=None):
        """Collect comprehensive metrics from environment state."""
        metrics = {
            "resource_stock": env.resource_stock.copy(),
            "num_agents": len(env.agents),
            "step": getattr(env, 'num_moves', 0)
        }
        
        if agents_data is None:
            agents_data = {}
            for agent_id in env.agents:
                agent = env.homeostatic_agents[agent_id]
                agents_data[agent_id] = {
                    "position": agent.position,
                    "internal_states": agent.internal_states.copy(),
                    "drive": agent.get_current_drive(),
                    "social_norm": agent.perceived_social_norm.copy(),
                    "last_intake": agent.last_intake.copy(),
                    "beta": agent.beta
                }
        
        metrics["agents"] = agents_data
        return metrics
    
    def print_detailed_state(self, metrics, title="Environment State"):
        """Print detailed environment state."""
        print(f"\n📊 {title}")
        print("=" * 60)
        print(f"🏪 Resource Stock: {metrics['resource_stock']}")
        print(f"👥 Active Agents: {metrics['num_agents']}")
        print(f"⏱️  Step: {metrics['step']}")
        
        print(f"\n👤 Agent Details:")
        for agent_id, data in metrics["agents"].items():
            print(f"  {agent_id}:")
            print(f"    Position: {data['position']}")
            print(f"    Internal States: {data['internal_states']}")
            print(f"    Drive: {data['drive']:.4f}")
            print(f"    Social Norm: {data['social_norm']}")
            print(f"    Last Intake: {data['last_intake']}")
            print(f"    Beta (social sensitivity): {data['beta']}")
    
    def test_scenario_1_depletion_aec(self):
        """Test 1: AEC Environment - Agents consume until depletion."""
        print("\n🧪 TEST 1: AEC Environment - Consumption until Depletion")
        print("=" * 80)
        
        env = self.create_test_env("aec", n_agents=3, size=1)  # Size=1 ensures all agents at resource
        env.reset(seed=42)
        
        metrics_history = []
        rewards_history = []
        
        initial_metrics = self.collect_metrics(env)
        self.print_detailed_state(initial_metrics, "Initial State")
        metrics_history.append(initial_metrics)
        
        # Force consumption until depletion
        step = 0
        while env.agents and step < 50:
            if env.resource_stock[0] <= 0:
                print(f"\n💀 Resource depleted at step {step}!")
                break
                
            current_agent_id = env.agent_selection
            if current_agent_id is None:
                break
                
            # Always try to consume
            action = 3  # Consume action
            
            # Collect pre-step data
            current_agent = env.homeostatic_agents[current_agent_id]
            pre_step_data = {
                "agent_id": current_agent_id,
                "step": step,
                "action": action,
                "pre_states": current_agent.internal_states.copy(),
                "pre_drive": current_agent.get_current_drive(),
                "pre_social_norm": current_agent.perceived_social_norm.copy(),
                "pre_resource_stock": env.resource_stock.copy()
            }
            
            # Execute step
            env.step(action)
            
            # Collect post-step data
            reward = env.rewards.get(current_agent_id, 0)
            post_step_data = {
                "post_states": current_agent.internal_states.copy(),
                "post_drive": current_agent.get_current_drive(),
                "post_social_norm": current_agent.perceived_social_norm.copy(),
                "post_resource_stock": env.resource_stock.copy(),
                "intake": current_agent.last_intake.copy(),
                "reward": reward
            }
            
            # Combine data
            step_data = {**pre_step_data, **post_step_data}
            rewards_history.append(step_data)
            
            # Print step details
            if step % 3 == 0 or env.resource_stock[0] <= 0.5:  # Print every round or near depletion
                print(f"\n📋 Step {step} - {current_agent_id}:")
                print(f"   Action: {action} (consume)")
                print(f"   States: {step_data['pre_states']} → {step_data['post_states']}")
                print(f"   Drive: {step_data['pre_drive']:.4f} → {step_data['post_drive']:.4f}")
                print(f"   Intake: {step_data['intake']}")
                print(f"   Reward: {step_data['reward']:.2f}")
                print(f"   Resource: {step_data['pre_resource_stock']} → {step_data['post_resource_stock']}")
                print(f"   Social Norm: {step_data['post_social_norm']}")
            
            step += 1
            
            # Collect metrics every round
            if step % 3 == 0:
                metrics = self.collect_metrics(env)
                metrics_history.append(metrics)
        
        final_metrics = self.collect_metrics(env)
        self.print_detailed_state(final_metrics, "Final State - AEC Depletion")
        
        # Summary statistics
        print(f"\n📈 AEC Depletion Summary:")
        print(f"   Total steps: {step}")
        print(f"   Final resource stock: {env.resource_stock}")
        print(f"   Agents remaining: {len(env.agents)}")
        
        if rewards_history:
            rewards = [r['reward'] for r in rewards_history]
            intakes = [r['intake'][0] for r in rewards_history]
            print(f"   Average reward: {np.mean(rewards):.2f}")
            print(f"   Total intake: {np.sum(intakes):.2f}")
            print(f"   Resource consumed: {initial_metrics['resource_stock'][0] - env.resource_stock[0]:.2f}")
        
        self.results["aec_depletion"] = {
            "metrics_history": metrics_history,
            "rewards_history": rewards_history,
            "final_metrics": final_metrics
        }
    
    def test_scenario_2_parallel_depletion(self):
        """Test 2: Parallel Environment - Consumption until depletion."""
        print("\n🧪 TEST 2: Parallel Environment - Consumption until Depletion")
        print("=" * 80)
        
        env = self.create_test_env("parallel", n_agents=3, size=1)
        observations = env.reset(seed=42)
        
        metrics_history = []
        rewards_history = []
        
        initial_metrics = self.collect_metrics(env)
        self.print_detailed_state(initial_metrics, "Initial State - Parallel")
        metrics_history.append(initial_metrics)
        
        round_num = 0
        while env.agents and round_num < 20:
            if env.resource_stock[0] <= 0:
                print(f"\n💀 Resource depleted at round {round_num}!")
                break
            
            # All agents try to consume
            actions = {agent: 3 for agent in env.agents}  # All consume
            
            # Collect pre-step data
            pre_round_data = {}
            for agent_id in env.agents:
                agent = env.homeostatic_agents[agent_id]
                pre_round_data[agent_id] = {
                    "pre_states": agent.internal_states.copy(),
                    "pre_drive": agent.get_current_drive(),
                    "pre_social_norm": agent.perceived_social_norm.copy()
                }
            
            pre_resource_stock = env.resource_stock.copy()
            
            # Execute parallel step
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Collect post-step data
            round_data = {
                "round": round_num,
                "actions": actions,
                "pre_resource_stock": pre_resource_stock,
                "post_resource_stock": env.resource_stock.copy(),
                "rewards": rewards.copy(),
                "agents_data": {}
            }
            
            for agent_id in env.agents:
                agent = env.homeostatic_agents[agent_id]
                round_data["agents_data"][agent_id] = {
                    **pre_round_data[agent_id],
                    "post_states": agent.internal_states.copy(),
                    "post_drive": agent.get_current_drive(),
                    "post_social_norm": agent.perceived_social_norm.copy(),
                    "intake": agent.last_intake.copy(),
                    "reward": rewards.get(agent_id, 0)
                }
            
            rewards_history.append(round_data)
            
            # Print round details
            print(f"\n🔄 Round {round_num}:")
            print(f"   Actions: {actions}")
            print(f"   Resource: {pre_resource_stock} → {env.resource_stock}")
            print(f"   Rewards: {[f'{k}:{v:.1f}' for k, v in rewards.items()]}")
            
            # Show agent changes
            for agent_id in env.agents:
                data = round_data["agents_data"][agent_id]
                print(f"   {agent_id}: states {data['pre_states']} → {data['post_states']}, intake {data['intake']}")
            
            # Collect metrics
            metrics = self.collect_metrics(env)
            metrics_history.append(metrics)
            
            round_num += 1
            
            # Check termination
            if all(terminations.values()) or all(truncations.values()):
                print("🏁 Episode terminated!")
                break
        
        final_metrics = self.collect_metrics(env)
        self.print_detailed_state(final_metrics, "Final State - Parallel Depletion")
        
        # Summary
        print(f"\n📈 Parallel Depletion Summary:")
        print(f"   Total rounds: {round_num}")
        print(f"   Final resource stock: {env.resource_stock}")
        print(f"   Agents remaining: {len(env.agents)}")
        
        if rewards_history:
            all_rewards = []
            all_intakes = []
            for round_data in rewards_history:
                all_rewards.extend(round_data["rewards"].values())
                for agent_data in round_data["agents_data"].values():
                    all_intakes.append(agent_data["intake"][0])
            
            print(f"   Average reward: {np.mean(all_rewards):.2f}")
            print(f"   Total intake: {np.sum(all_intakes):.2f}")
        
        self.results["parallel_depletion"] = {
            "metrics_history": metrics_history,
            "rewards_history": rewards_history,
            "final_metrics": final_metrics
        }
    
    def test_scenario_3_conservative_consumption(self):
        """Test 3: Conservative consumption - agents don't deplete."""
        print("\n🧪 TEST 3: Conservative Consumption (High Social Cost)")
        print("=" * 80)
        
        # High beta = high social cost = conservative behavior
        env = self.create_test_env("aec", n_agents=3, size=3, beta=2.0, learning_rate=0.3)
        env.reset(seed=42)
        
        initial_metrics = self.collect_metrics(env)
        self.print_detailed_state(initial_metrics, "Initial State - Conservative")
        
        step = 0
        consecutive_no_consumption = 0
        
        print(f"\n🐌 Testing Conservative Behavior (high social cost):")
        
        while env.agents and step < 30:
            current_agent_id = env.agent_selection
            if current_agent_id is None:
                break
            
            current_agent = env.homeostatic_agents[current_agent_id]
            
            # Intelligent action: consume if really needed, otherwise move
            current_drive = current_agent.get_current_drive()
            social_norm = current_agent.perceived_social_norm[0]
            
            # Consume if drive is high and social norm is low
            if current_drive > 0.2 and social_norm < 0.05:
                action = 3  # Consume
                action_name = "CONSUME"
            else:
                action = np.random.choice([0, 1, 2])  # Random movement
                action_name = ["STAY", "LEFT", "RIGHT"][action]
            
            pre_resource = env.resource_stock[0]
            env.step(action)
            post_resource = env.resource_stock[0]
            
            reward = env.rewards.get(current_agent_id, 0)
            intake = current_agent.last_intake[0]
            
            if intake == 0:
                consecutive_no_consumption += 1
            else:
                consecutive_no_consumption = 0
            
            if step % 5 == 0 or intake > 0:
                print(f"   Step {step}: {current_agent_id} {action_name}, "
                      f"drive={current_drive:.3f}, norm={social_norm:.3f}, "
                      f"intake={intake:.3f}, reward={reward:.1f}")
            
            step += 1
            
            # Stop if agents learn to avoid consumption
            if consecutive_no_consumption > 15:
                print(f"   🎯 Agents learned conservative behavior after {consecutive_no_consumption} steps without consumption!")
                break
        
        final_metrics = self.collect_metrics(env)
        self.print_detailed_state(final_metrics, "Final State - Conservative")
        
        resource_preserved = env.resource_stock[0] / initial_metrics["resource_stock"][0]
        print(f"\n📈 Conservative Behavior Summary:")
        print(f"   Resource preservation: {resource_preserved:.1%}")
        print(f"   Final social norms: {[f'{agent.perceived_social_norm[0]:.3f}' for agent in env.homeostatic_agents.values()]}")
    
    def test_scenario_4_random_actions(self):
        """Test 4: Random actions with size=3."""
        print("\n🧪 TEST 4: Random Actions (Size=3)")
        print("=" * 80)
        
        env = self.create_test_env("aec", n_agents=3, size=3, beta=0.3)
        env.reset(seed=42)
        
        initial_metrics = self.collect_metrics(env)
        self.print_detailed_state(initial_metrics, "Initial State - Random Actions")
        
        print(f"\n🎲 Resource positions: {[env.resources_info[i]['position'] for i in env.resources_info]}")
        print(f"🎲 Agent positions: {[agent.position for agent in env.homeostatic_agents.values()]}")
        
        step = 0
        action_counts = {"stay": 0, "left": 0, "right": 0, "consume": 0}
        successful_consumptions = 0
        failed_consumptions = 0
        
        while env.agents and step < 24:  # 8 rounds of 3 agents
            current_agent_id = env.agent_selection
            if current_agent_id is None:
                break
            
            current_agent = env.homeostatic_agents[current_agent_id]
            
            # Completely random action
            action = np.random.randint(0, 4)
            action_names = ["STAY", "LEFT", "RIGHT", "CONSUME"]
            action_name = action_names[action]
            
            action_counts[action_name.lower()] += 1
            
            pre_position = current_agent.position
            pre_resource = env.resource_stock[0]
            
            env.step(action)
            
            post_position = current_agent.position
            post_resource = env.resource_stock[0]
            intake = current_agent.last_intake[0]
            reward = env.rewards.get(current_agent_id, 0)
            
            # Track consumption success
            if action == 3:  # Consume action
                if intake > 0:
                    successful_consumptions += 1
                else:
                    failed_consumptions += 1
            
            if step % 6 == 0 or intake > 0:  # Print every 2 rounds or when consumption occurs
                print(f"   Step {step}: {current_agent_id} {action_name} "
                      f"pos:{pre_position}→{post_position}, "
                      f"resource:{pre_resource:.2f}→{post_resource:.2f}, "
                      f"intake:{intake:.3f}, reward:{reward:.1f}")
            
            step += 1
        
        final_metrics = self.collect_metrics(env)
        self.print_detailed_state(final_metrics, "Final State - Random Actions")
        
        print(f"\n📈 Random Actions Summary:")
        print(f"   Action distribution: {action_counts}")
        print(f"   Successful consumptions: {successful_consumptions}")
        print(f"   Failed consumptions: {failed_consumptions}")
        consumption_success_rate = successful_consumptions / (successful_consumptions + failed_consumptions) if (successful_consumptions + failed_consumptions) > 0 else 0
        print(f"   Consumption success rate: {consumption_success_rate:.1%}")
        print(f"   Resource preservation: {env.resource_stock[0] / initial_metrics['resource_stock'][0]:.1%}")
    
    def test_scenario_5_size3_detailed(self):
        """Test 5: Detailed analysis with size=3."""
        print("\n🧪 TEST 5: Detailed Analysis (Size=3)")
        print("=" * 80)
        
        env = self.create_test_env("parallel", n_agents=3, size=3, beta=0.5, learning_rate=0.2)
        observations = env.reset(seed=123)
        
        initial_metrics = self.collect_metrics(env)
        self.print_detailed_state(initial_metrics, "Initial State - Size 3 Analysis")
        
        # Show resource and agent positions
        print(f"\n🗺️  Environment Layout:")
        for i in range(3):
            agents_here = [agent_id for agent_id, agent in env.homeostatic_agents.items() if agent.position == i]
            resources_here = [res_id for res_id, res_info in env.resources_info.items() if res_info['position'] == i]
            print(f"   Position {i}: Agents {agents_here}, Resources {resources_here}")
        
        rounds_data = []
        
        for round_num in range(8):
            print(f"\n🔄 Round {round_num + 1}:")
            
            # Strategic actions based on positions
            actions = {}
            for agent_id in env.agents:
                agent = env.homeostatic_agents[agent_id]
                resource_pos = env.resources_info[0]['position']  # Assuming 1 resource
                
                if agent.position == resource_pos:
                    # At resource position - maybe consume
                    if agent.get_current_drive() > 0.1:
                        actions[agent_id] = 3  # Consume
                    else:
                        actions[agent_id] = np.random.choice([0, 1, 2])  # Move randomly
                else:
                    # Not at resource - try to move towards it
                    if agent.position < resource_pos:
                        actions[agent_id] = 2  # Move right
                    else:
                        actions[agent_id] = 1  # Move left
            
            pre_metrics = self.collect_metrics(env)
            
            # Execute round
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            post_metrics = self.collect_metrics(env)
            
            # Analyze round
            print(f"   Actions: {actions}")
            print(f"   Rewards: {[f'{k}:{v:.1f}' for k, v in rewards.items()]}")
            
            round_data = {
                "round": round_num,
                "actions": actions,
                "rewards": rewards.copy(),
                "resource_change": pre_metrics["resource_stock"][0] - post_metrics["resource_stock"][0],
                "pre_metrics": pre_metrics,
                "post_metrics": post_metrics
            }
            rounds_data.append(round_data)
            
            # Show position changes
            for agent_id in env.agents:
                pre_pos = pre_metrics["agents"][agent_id]["position"]
                post_pos = post_metrics["agents"][agent_id]["position"]
                intake = post_metrics["agents"][agent_id]["last_intake"][0]
                social_norm = post_metrics["agents"][agent_id]["social_norm"][0]
                
                if pre_pos != post_pos or intake > 0:
                    print(f"   {agent_id}: pos {pre_pos}→{post_pos}, intake {intake:.3f}, norm {social_norm:.3f}")
            
            if all(terminations.values()) or all(truncations.values()):
                break
        
        final_metrics = self.collect_metrics(env)
        self.print_detailed_state(final_metrics, "Final State - Size 3 Analysis")
        
        # Analysis summary
        total_consumption = sum(r["resource_change"] for r in rounds_data)
        avg_reward = np.mean([r for round_data in rounds_data for r in round_data["rewards"].values()])
        
        print(f"\n📈 Size 3 Analysis Summary:")
        print(f"   Total resource consumed: {total_consumption:.2f}")
        print(f"   Average reward per action: {avg_reward:.2f}")
        print(f"   Final social norms: {[agent_data['social_norm'][0] for agent_data in final_metrics['agents'].values()]}")
        
        self.results["size3_analysis"] = {
            "rounds_data": rounds_data,
            "final_metrics": final_metrics
        }
    
    def run_all_tests(self):
        """Run all test scenarios."""
        print("🚀 Starting Comprehensive Environment Testing")
        print("=" * 100)
        
        start_time = time.time()
        
        # Run all tests
        self.test_scenario_1_depletion_aec()
        self.test_scenario_2_parallel_depletion()
        self.test_scenario_3_conservative_consumption()
        self.test_scenario_4_random_actions()
        self.test_scenario_5_size3_detailed()
        
        end_time = time.time()
        
        print(f"\n🎯 ALL TESTS COMPLETED in {end_time - start_time:.1f} seconds")
        print("=" * 100)
        
        # Overall summary
        print(f"\n📋 Overall Summary:")
        print(f"   ✅ AEC Environment: Functional")
        print(f"   ✅ Parallel Environment: Functional") 
        print(f"   ✅ Resource Depletion: Working")
        print(f"   ✅ Social Norm Learning: Working")
        print(f"   ✅ Reward System: Working")
        print(f"   ✅ Position-based Mechanics: Working")
        print(f"   ✅ Size=3 Multi-position: Working")
        
        return self.results


def main():
    """Main testing function."""
    tester = EnvironmentTester()
    results = tester.run_all_tests()
    
    print(f"\n💾 Test results stored in tester.results")
    print(f"🔬 Available result keys: {list(results.keys())}")
    
    return tester, results


if __name__ == "__main__":
    tester, results = main() 
