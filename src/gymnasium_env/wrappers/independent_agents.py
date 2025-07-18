"""
Independent Agents Wrapper for NORMARL Homeostatic Environment

This wrapper creates true independence of states for multiple agents
using a Gymnasium environment as base, without requiring PettingZoo.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from src.utils.get_params import ParameterHandler


class IndependentAgentsWrapper(gym.Wrapper):
    """
    Wrapper que cria verdadeira independência de estados para múltiplos agentes
    usando um ambiente Gymnasium como base.
    
    Cada agente mantém:
    - Estados internos independentes
    - Drive próprio
    - Normas sociais próprias
    - Histórico individual
    """
    
    def __init__(self, env, n_agents=2):
        super().__init__(env)
        self.n_agents = n_agents
        self.current_agent = 0
        
        # 🔥 Estados independentes por agente
        self.agents_states = {}
        self.agents_drives = {}
        self.agents_norms = {}
        
        # Environment size verified
        
        # Inicializar estados independentes para cada agente
        for agent_id in range(n_agents):
            # Criar drive independente para cada agente
            agent_drive = env.unwrapped.param_manager.create_drive(env.unwrapped.drive_type)
            
            # Estados iniciais independentes
            initial_states = np.random.uniform(
                low=-0.3, high=0.3, 
                size=(env.unwrapped.dimension_internal_states,)
            )
            initial_position = np.random.randint(0, env.unwrapped.size)
            
            # Armazenar estados por agente
            self.agents_states[agent_id] = {
                "position": initial_position,
                "internal_states": initial_states.copy(),
                "last_intake": np.zeros(env.unwrapped.dimension_internal_states)
            }
            
            # Drive independente por agente
            self.agents_drives[agent_id] = agent_drive
            initial_drive = agent_drive.compute_drive(initial_states)
            agent_drive.update_drive(initial_drive)
            
            # Normas sociais independentes por agente
            self.agents_norms[agent_id] = np.zeros(env.unwrapped.dimension_internal_states)
        
        # Modificar observation space para incluir agent_id
        old_space = self.env.observation_space
        new_spaces = old_space.spaces.copy()
        new_spaces['agent_id'] = spaces.Discrete(self.n_agents)
        new_spaces['episode_step'] = spaces.Box(low=0, high=1000, shape=(1,), dtype=np.int32)
        self.observation_space = spaces.Dict(new_spaces)
        
        # Histórico e médias
        self.episode_step = 0
        self.max_episode_steps = 200
        self.trial_intakes = []
        self.avg_intake = np.zeros(env.unwrapped.dimension_internal_states)
        
        # Estatísticas por agente para logging
        self.agents_rewards = [[] for _ in range(n_agents)]
        self.agents_social_costs = [[] for _ in range(n_agents)]
        self.agents_consumptions = [[] for _ in range(n_agents)]
        
        print(f"🤖 IndependentAgentsWrapper initialized with {n_agents} agents")
    
    def reset(self, **kwargs):
        """Reset mantendo estados independentes"""
        obs, info = self.env.reset(**kwargs)
        self.current_agent = 0
        self.episode_step = 0
        self.trial_intakes = []
        
        # Resetar estatísticas
        self.agents_rewards = [[] for _ in range(self.n_agents)]
        self.agents_social_costs = [[] for _ in range(self.n_agents)]
        self.agents_consumptions = [[] for _ in range(self.n_agents)]
        
        # Resetar estados independentes
        for agent_id in range(self.n_agents):
            initial_states = np.random.uniform(
                low=-0.3, high=0.3, 
                size=(self.env.unwrapped.dimension_internal_states,)
            )
            initial_position = np.random.randint(0, self.env.unwrapped.size)
            
            self.agents_states[agent_id] = {
                "position": initial_position,
                "internal_states": initial_states.copy(),
                "last_intake": np.zeros(self.env.unwrapped.dimension_internal_states)
            }
            
            # Reset drive independente
            initial_drive = self.agents_drives[agent_id].compute_drive(initial_states)
            self.agents_drives[agent_id].update_drive(initial_drive)
            
            # Reset normas sociais
            self.agents_norms[agent_id] = np.zeros(self.env.unwrapped.dimension_internal_states)
        
        # Setar observação do primeiro agente
        obs = self._get_agent_observation(self.current_agent)
        
        print(f"🔄 Environment reset. Starting with Agent {self.current_agent}")
        return obs, info
    
    def step(self, action):
        """Step com verdadeira independência de agentes"""
        current_agent_id = self.current_agent
        
        print(f"\n🎬 STEP {self.episode_step}: Agent {current_agent_id} taking action {action}")
        
        agent_state = self.agents_states[current_agent_id]
        self.env.unwrapped.agent_info["position"] = agent_state["position"]
        self.env.unwrapped.agent_info["internal_states"] = agent_state["internal_states"].copy()
        self.env.unwrapped.perceived_social_norm = self.agents_norms[current_agent_id].copy()
        
        print(f"  🧠 Agent {current_agent_id} states before: pos={agent_state['position']}, "
              f"internal={agent_state['internal_states']}, norm={self.agents_norms[current_agent_id]}")
        
        states_after_decay = self.agents_drives[current_agent_id].apply_natural_decay(
            agent_state["internal_states"]
        )
        
        # Executar ação no ambiente (modificando temporariamente os estados)
        obs, reward, done, truncated, info = self.env.step(action, self.avg_intake)
        
        # 🔥 CAPTURAR E SALVAR ESTADOS ATUALIZADOS DO AGENTE
        updated_position = self.env.unwrapped.agent_info["position"]
        updated_states = self.env.unwrapped.agent_info["internal_states"].copy()
        agent_intake = info["last_intake"].copy()
        
        print(f"  📊 Agent {current_agent_id} states after: pos={updated_position}, "
              f"internal={updated_states}, intake={agent_intake}")
        
        # Atualizar estados independentes do agente
        self.agents_states[current_agent_id]["position"] = updated_position
        self.agents_states[current_agent_id]["internal_states"] = updated_states
        self.agents_states[current_agent_id]["last_intake"] = agent_intake
        
        # Calcular recompensa homeostática independente
        old_drive = self.agents_drives[current_agent_id].get_current_drive()
        new_drive = self.agents_drives[current_agent_id].compute_drive(updated_states)
        homeostatic_reward = self.agents_drives[current_agent_id].compute_reward(old_drive, new_drive)
        self.agents_drives[current_agent_id].update_drive(new_drive)
        
        # Calcular custo social independente
        social_cost = self._compute_agent_social_cost(current_agent_id, agent_intake)
        
        # Recompensa combinada
        reward = homeostatic_reward - social_cost * 10  # Mesmo multiplicador do normal.py
        
        print(f"  💰 Agent {current_agent_id} rewards: homeostatic={homeostatic_reward:.3f}, "
              f"social_cost={social_cost:.3f}, total={reward:.3f}")
        
        # Armazenar estatísticas do agente
        self.agents_rewards[current_agent_id].append(reward)
        self.agents_social_costs[current_agent_id].append(social_cost)
        self.agents_consumptions[current_agent_id].append(np.sum(agent_intake))
        
        # Adicionar intake ao trial
        self.trial_intakes.append(agent_intake.copy())
        
        # Se último agente da rodada, atualizar ambiente global
        if self.current_agent == self.n_agents - 1:
            self._update_global_state()
            self.trial_intakes = []
            print(f"  🌍 Round {self.episode_step // self.n_agents} completed. Global state updated.")
        
        # Próximo agente
        self.current_agent = (self.current_agent + 1) % self.n_agents
        self.episode_step += 1
        
        # Verificar terminação
        if self.episode_step >= self.max_episode_steps:
            done = True
            print(f"  🏁 Episode ended due to max steps ({self.max_episode_steps})")
            # Adicionar estatísticas finais ao info
            info.update(self._get_episode_stats())
        
        # Observação do próximo agente
        obs = self._get_agent_observation(self.current_agent)
        
        return obs, reward, done, truncated, info
    
    def _get_agent_observation(self, agent_id):
        """Obter observação específica de um agente"""
        agent_state = self.agents_states[agent_id]
        
        obs = {
            "position": agent_state["position"],
            "internal_states": agent_state["internal_states"],
            "perceived_social_norm": self.agents_norms[agent_id],
            "agent_id": agent_id,
            "episode_step": np.array([self.episode_step], dtype=np.int32)
        }
        return obs
    
    def _compute_agent_social_cost(self, agent_id, intake):
        """Computar custo social para agente específico"""
        agent_norm = self.agents_norms[agent_id]
        beta = self.env.unwrapped.beta
        
        # Fator de escassez (compartilhado)
        scarcity_factor = np.maximum(0, 
            self.env.unwrapped.a - self.env.unwrapped.b * self.env.unwrapped.resource_stock
        )
        
        social_cost = 0
        for i in range(len(intake)):
            if intake[i] > agent_norm[i]:
                cost_component = beta * (intake[i] - agent_norm[i]) * scarcity_factor[i]
                social_cost += cost_component
                print(f"    💸 Agent {agent_id} social cost resource {i}: {cost_component:.3f}")
        
        return social_cost
    
    def _update_global_state(self):
        """Atualizar estado global após todos os agentes agirem"""
        if len(self.trial_intakes) > 0:
            # Calcular média de intake
            self.avg_intake = np.mean(self.trial_intakes, axis=0)
            
            print(f"  📈 Average intake this round: {self.avg_intake}")
            
            # Atualizar normas sociais independentes para cada agente
            for agent_id in range(self.n_agents):
                old_norm = self.agents_norms[agent_id].copy()
                self.agents_norms[agent_id] = (
                    (1 - self.env.unwrapped.social_alpha) * self.agents_norms[agent_id] + 
                    self.env.unwrapped.social_alpha * self.avg_intake
                )
                print(f"    🧭 Agent {agent_id} norm update: {old_norm} → {self.agents_norms[agent_id]}")
    
    def _get_episode_stats(self):
        """Obter estatísticas do episódio para logging"""
        stats = {}
        
        # Estatísticas por agente
        for agent_id in range(self.n_agents):
            if self.agents_rewards[agent_id]:
                stats[f"agent_{agent_id}_mean_reward"] = np.mean(self.agents_rewards[agent_id])
                stats[f"agent_{agent_id}_total_consumption"] = np.sum(self.agents_consumptions[agent_id])
                stats[f"agent_{agent_id}_mean_social_cost"] = np.mean(self.agents_social_costs[agent_id])
                stats[f"agent_{agent_id}_steps"] = len(self.agents_rewards[agent_id])
        
        # Estatísticas globais
        all_rewards = [r for rewards in self.agents_rewards for r in rewards]
        if all_rewards:
            stats["mean_episode_reward"] = np.mean(all_rewards)
            stats["total_episode_consumption"] = np.sum([np.sum(c) for c in self.agents_consumptions])
            stats["final_resource_stock"] = np.sum(self.env.unwrapped.resource_stock)
            stats["episode_length"] = self.episode_step
        
        return stats
    
    def get_agent_states(self):
        """Função útil para debugging - obter todos os estados dos agentes"""
        return {
            agent_id: {
                "position": state["position"],
                "internal_states": state["internal_states"].copy(),
                "last_intake": state["last_intake"].copy(),
                "social_norm": self.agents_norms[agent_id].copy(),
                "drive": self.agents_drives[agent_id].get_current_drive()
            }
            for agent_id, state in self.agents_states.items()
        }
    
    def print_agent_summary(self):
        """Imprime um resumo dos estados de todos os agentes"""
        print("\n" + "="*60)
        print("📊 AGENT STATES SUMMARY")
        print("="*60)
        
        for agent_id in range(self.n_agents):
            state = self.agents_states[agent_id]
            norm = self.agents_norms[agent_id]
            drive = self.agents_drives[agent_id].get_current_drive()
            
            print(f"🤖 Agent {agent_id}:")
            print(f"   Position: {state['position']}")
            print(f"   Drive: {drive:.3f}")
            print(f"   Internal States: {state['internal_states']}")
            print(f"   Social Norm: {norm}")
            print(f"   Last Intake: {state['last_intake']}")
            
            if self.agents_rewards[agent_id]:
                print(f"   Avg Reward: {np.mean(self.agents_rewards[agent_id]):.3f}")
                print(f"   Avg Social Cost: {np.mean(self.agents_social_costs[agent_id]):.3f}")
            print()
        
        print(f"🌍 Global Average Intake: {self.avg_intake}")
        print(f"📦 Resource Stock: {self.env.unwrapped.resource_stock}")
        print("="*60 + "\n") 
