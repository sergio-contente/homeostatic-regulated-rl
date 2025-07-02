# Sistema de Logging Detalhado NORMARL

## Visão Geral

O sistema de logging detalhado do NORMARL foi implementado para fornecer monitoramento completo do treinamento de agentes homeostáticos com aprendizado de normas sociais. Este sistema permite observar em tempo real:

- Estados internos individuais de cada agente
- Evolução das percepções de normas sociais
- Padrões de consumo de recursos
- Custos sociais e recompensas homeostáticas
- Dinâmicas de recursos globais

## Arquitetura do Sistema

### NormarlCallback Class

O `NormarlCallback` é uma classe customizada baseada no `BaseCallback` do Stable Baselines 3 que captura métricas específicas do NORMARL a cada passo do treinamento.

```python
from train_ppo_normarl import train_ppo_normarl

# O callback é automaticamente incluído no treinamento
model, callback = train_ppo_normarl(**config)
```

## Métricas Registradas

### 1. Métricas de Episódio (`normarl/`)

- `normarl/episode_reward`: Recompensa total do episódio
- `normarl/episode_length`: Duração do episódio em passos
- `normarl/total_resource_stock`: Estoque total de recursos no ambiente
- `normarl/resource_{i}_stock`: Estoque de cada tipo de recurso
- `normarl/environment_steps`: Passos totais do ambiente
- `normarl/current_agent_idx`: Índice do agente atual (AEC)

### 2. Métricas por Agente (`agents/{agent_id}/`)

#### Estados Básicos
- `agents/{agent_id}/drive`: Nível de urgência homeostática atual
- `agents/{agent_id}/position`: Posição do agente no ambiente
- `agents/{agent_id}/total_consumption`: Consumo total no último passo

#### Estados Internos (`agents/{agent_id}/states/`)
- `agents/{agent_id}/states/{state_name}`: Valor de cada estado interno (ex: food, water)

#### Normas Sociais (`agents/{agent_id}/social_norms/`)
- `agents/{agent_id}/social_norms/{state_name}`: Percepção das normas sociais para cada recurso

#### Consumo Detalhado (`agents/{agent_id}/consumption/`)
- `agents/{agent_id}/consumption/{state_name}`: Consumo de cada tipo de recurso

#### Custos Sociais
- `agents/{agent_id}/social_cost`: Custo social calculado pela fórmula NORMARL

### 3. Métricas Finais (`normarl/final_*`)

Estatísticas calculadas ao final do treinamento:
- `normarl/final_mean_reward`: Recompensa média de todos os episódios
- `normarl/final_std_reward`: Desvio padrão das recompensas
- `normarl/final_max_reward`: Melhor episódio
- `normarl/final_min_reward`: Pior episódio
- `normarl/total_episodes`: Total de episódios completados

## Como Usar

### 1. Treinamento Básico com Logging

```python
from train_ppo_normarl import train_ppo_normarl

config = {
    'total_timesteps': 50000,
    'n_agents': 3,
    'beta': 0.5,
    'learning_rate': 0.1,
    'tensorboard_log': 'logs/normarl_experiment'
}

# Treinar com logging automático
model, callback = train_ppo_normarl(**config)

# Acessar métricas coletadas
print(f"Episódios: {len(callback.episode_rewards)}")
print(f"Recompensa média: {np.mean(callback.episode_rewards):.3f}")
```

### 2. Visualização no TensorBoard

```bash
# Inicie o TensorBoard
tensorboard --logdir logs/normarl_experiment

# Abra http://localhost:6006 no navegador
```

### 3. Scripts de Demonstração

#### Demonstração Básica
```bash
python demo_detailed_logging.py --demo basic
```

#### Comparação de Parâmetros
```bash
python demo_detailed_logging.py --demo comparison
```

#### Ambos
```bash
python demo_detailed_logging.py --demo both
```

## Exemplos de Análise

### 1. Observar Aprendizado de Normas Sociais

No TensorBoard, navegue até `agents/agent_0/social_norms/food` para ver como a percepção de normas sociais evolui ao longo do tempo.

### 2. Monitorar Equilíbrio Homeostático

Compare `agents/agent_0/states/food` com `agents/agent_0/drive` para observar como os estados internos influenciam a urgência homeostática.

### 3. Analisar Custos Sociais

Observe `agents/agent_0/social_cost` em relação ao `normarl/total_resource_stock` para entender como a escassez de recursos afeta os custos sociais.

### 4. Comparar Agentes

Compare as métricas entre diferentes agentes (agent_0, agent_1, agent_2) para observar heterogeneidade e emergência de papéis sociais.

## Estrutura de Arquivos Gerados

```
project/
├── models/ppo_normarl/
│   ├── final_model.zip          # Modelo treinado
│   ├── training_metrics.npy     # Métricas em formato NumPy
│   └── checkpoints/             # Checkpoints periódicos
├── logs/ppo_normarl/
│   └── evaluations.npz          # Resultados de avaliação
└── tensorboard_logs/ppo_normarl/
    └── PPO_*/                   # Logs do TensorBoard
        ├── events.out.tfevents.*
        └── ...
```

## Exemplos de Visualização

### 1. Gráfico de Múltiplos Agentes
Selecione `agents/*/drive` no TensorBoard para comparar as urgências homeostáticas de todos os agentes simultaneamente.

### 2. Correlação Estados-Consumo
Compare `agents/agent_0/states/food` com `agents/agent_0/consumption/food` para observar comportamento reativo vs. preventivo.

### 3. Evolução de Normas
Plote `agents/*/social_norms/food` para todos os agentes e observe convergência ou divergência das percepções sociais.

## Configurações Recomendadas

### Para Observar Dinâmicas Sociais
```python
config = {
    'n_agents': 5,              # Mais agentes = dinâmicas sociais mais ricas
    'beta': 0.8,                # Alta sensibilidade a normas sociais
    'learning_rate': 0.15,      # Aprendizado social mais rápido
    'number_resources': 2,      # Múltiplos recursos para estratégias diversas
}
```

### Para Estudar Aprendizado Individual
```python
config = {
    'n_agents': 1,              # Foco em um agente
    'beta': 0.0,                # Sem influência social
    'total_timesteps': 100000,  # Treinamento longo
    'eval_freq': 5000,          # Avaliações frequentes
}
```

## Solução de Problemas

### Logs Vazios
- Verifique se o callback está sendo adicionado corretamente
- Confirme que `tensorboard_log` está especificado na configuração
- Certifique-se de que há episódios sendo completados

### Performance Lenta
- Use `n_envs=1` para debugging
- Reduza a frequência de logging se necessário
- Considere usar `verbose=0` no callback para menos prints

### Métricas Inconsistentes
- Verifique se o ambiente está sendo resetado corretamente
- Confirme que os agentes estão sendo inicializados adequadamente
- Observe se há problemas na função `unwrap_env`

## Personalização

### Adicionar Novas Métricas
```python
class CustomNormarlCallback(NormarlCallback):
    def _on_step(self):
        super()._on_step()
        
        # Sua métrica customizada
        try:
            # Acesse o ambiente
            normarl_env = unwrap_env(self.training_env.envs[0]).env
            
            # Calcule e registre sua métrica
            custom_metric = calculate_my_metric(normarl_env)
            self.logger.record("custom/my_metric", custom_metric)
            
        except Exception as e:
            if self.verbose > 0:
                print(f"Erro ao calcular métrica customizada: {e}")
        
        return True
```

## Próximos Passos

1. **Análise Automatizada**: Implementar scripts que detectem padrões automaticamente
2. **Visualizações Customizadas**: Criar dashboards específicos para análise NORMARL
3. **Métricas Agregadas**: Adicionar métricas de população e emergência social
4. **Comparação de Algoritmos**: Expandir para outros algoritmos de RL 
