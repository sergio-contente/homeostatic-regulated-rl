# PPO Training for NORMARL Homeostatic Environment

Este repositório inclui um sistema completo de treinamento PPO usando **Stable Baselines 3** para o ambiente **NORMARL** (Norm-Adaptive Resource Management with Homeostatic Agents).

## 🎯 Visão Geral

O sistema permite treinar agentes PPO que aprendem:
- **Regulação homeostática**: Manter estados internos equilibrados
- **Normas sociais**: Adaptar comportamento baseado em observações de outros agentes
- **Gestão de recursos**: Navegar e consumir recursos de forma sustentável

## 📦 Instalação

### 1. Dependências Base
```bash
# Instalar dependências do projeto principal
pip install -r requirements.txt

# Instalar dependências específicas para treinamento
pip install -r requirements_training.txt
```

### 2. Dependências Principais
- `stable-baselines3>=2.0.0` - Algoritmos RL
- `pettingzoo>=1.22.0` - Ambiente multi-agente  
- `supersuit>=3.7.0` - Wrappers para PettingZoo
- `torch>=1.11.0` - Deep learning backend
- `tensorboard>=2.9.0` - Logging e visualização

## 🚀 Uso Rápido

### Treinamento Básico
```bash
# Treinamento rápido com parâmetros padrão
python run_training_example.py --experiment quick

# Experimento comparando diferentes valores de norma social
python run_training_example.py --experiment social_norms
```

### Treinamento Customizado
```bash
# Treinamento com parâmetros específicos
python train_ppo_normarl.py \
    --total-timesteps 100000 \
    --n-agents 5 \
    --beta 0.8 \
    --n-envs 4 \
    --ppo-lr 3e-4
```

### Avaliação de Modelo
```bash
# Avaliar modelo treinado
python train_ppo_normarl.py \
    --evaluate models/ppo_normarl/final_model \
    --eval-episodes 20 \
    --render
```

## 🎮 Parâmetros do Ambiente

### Parâmetros Homeostáticos
- `--config-path`: Arquivo de configuração dos drives (`config/config.yaml`)
- `--drive-type`: Tipo de drive (`base_drive`, `interoceptive_drive`, `elliptic_drive`)
- `--number-resources`: Número de tipos de recursos (padrão: 1)
- `--n-agents`: Número de agentes (padrão: 3)
- `--size`: Tamanho do ambiente (padrão: 5)

### Parâmetros NORMARL  
- `--learning-rate`: Taxa de aprendizado social α (padrão: 0.1)
- `--beta`: Força de internalização das normas sociais β (padrão: 0.5)

## 🧠 Parâmetros PPO

### Hiperparâmetros Principais
- `--ppo-lr`: Taxa de aprendizado PPO (padrão: 3e-4)
- `--n-steps`: Passos por atualização (padrão: 2048)
- `--batch-size`: Tamanho do batch (padrão: 64)
- `--n-epochs`: Épocas por atualização (padrão: 10)

### Configuração de Treinamento
- `--total-timesteps`: Total de timesteps (padrão: 100,000)
- `--n-envs`: Ambientes paralelos (padrão: 4)
- `--eval-freq`: Frequência de avaliação (padrão: 10,000)
- `--save-freq`: Frequência de salvamento (padrão: 25,000)

## 📊 Monitoramento

### TensorBoard
```bash
# Visualizar logs de treinamento
tensorboard --logdir tensorboard_logs/ppo_normarl
```

### Métricas Importantes
- **Episode Reward**: Recompensa total do episódio
- **Episode Length**: Duração do episódio
- **Value Loss**: Perda da função valor
- **Policy Loss**: Perda da política
- **Explained Variance**: Qualidade da função valor

## 🏗️ Arquitetura do Sistema

### Wrapper PettingZoo → Gymnasium
```python
class PettingZooToGymnasiumWrapper(gym.Env):
    """
    Converte ambiente PettingZoo multi-agente para formato Gymnasium.
    - Treina política única para todos os agentes
    - Observações: [posição, estados_internos, normas_sociais] 
    - Ações: [movimento, consumo_recursos]
    """
```

### Estrutura de Observação
```python
obs = {
    'position': int,                    # Posição do agente  
    'internal_states': np.array,        # Estados homeostáticos
    'perceived_social_norm': np.array   # Normas sociais percebidas
}
# Flattened: [pos, state1, state2, ..., norm1, norm2, ...]
```

### Espaço de Ação
```python
actions = [
    0: ficar_parado,
    1: mover_esquerda, 
    2: mover_direita,
    3+i: consumir_recurso_tipo_i
]
```

## 🧪 Experimentos Sugeridos

### 1. Impacto das Normas Sociais
```bash
# Sem normas sociais (β=0)
python train_ppo_normarl.py --beta 0.0 --tensorboard-log tb_beta_0

# Normas moderadas (β=0.5)  
python train_ppo_normarl.py --beta 0.5 --tensorboard-log tb_beta_05

# Normas fortes (β=1.0)
python train_ppo_normarl.py --beta 1.0 --tensorboard-log tb_beta_1
```

### 2. Escalabilidade Multi-Agente
```bash
# Variando número de agentes
for agents in 2 5 10; do
    python train_ppo_normarl.py \
        --n-agents $agents \
        --model-dir models/agents_$agents \
        --tensorboard-log tb_agents_$agents
done
```

### 3. Complexidade de Recursos
```bash
# Ambiente com múltiplos recursos
python train_ppo_normarl.py \
    --number-resources 3 \
    --size 10 \
    --total-timesteps 200000
```

## 📁 Estrutura de Arquivos

```
├── train_ppo_normarl.py          # Script principal de treinamento
├── run_training_example.py       # Exemplos de uso  
├── requirements_training.txt     # Dependências PPO
├── PPO_TRAINING_README.md        # Este arquivo
├── models/                       # Modelos salvos
│   ├── ppo_normarl/
│   │   ├── final_model.zip
│   │   ├── best_model.zip  
│   │   └── checkpoints/
├── logs/                         # Logs de avaliação
│   └── ppo_normarl/
├── tensorboard_logs/             # Logs TensorBoard
│   └── ppo_normarl/
└── src/pettingzoo_env/           # Ambiente NORMARL
    ├── normarl.py
    ├── homeostatic_agent.py
    └── actions.py
```

## 🔧 Configuração Avançada

### Otimização de Performance
```python
# Para treinamento mais rápido
ppo_kwargs = {
    'learning_rate': 5e-4,
    'n_steps': 1024,           # Menor para updates mais frequentes
    'batch_size': 128,         # Maior para estabilidade
    'n_epochs': 5,             # Menor para velocidade
}
```

### Multi-Processing
```python
# Usar múltiplos processos
train_ppo_normarl(
    n_envs=8,                  # 8 ambientes paralelos
    total_timesteps=500000,    # Mais timesteps
    eval_freq=20000           # Avaliação menos frequente
)
```

## 📈 Resultados Esperados

### Comportamentos Emergentes
1. **Aprendizado de Normas**: Agentes devem convergir para normas de consumo sustentáveis
2. **Regulação Homeostática**: Manutenção de estados internos equilibrados  
3. **Cooperação**: Redução do consumo excessivo quando recursos são escassos
4. **Navegação Inteligente**: Movimento eficiente para recursos necessários

### Métricas de Sucesso
- **Estabilidade**: Recompensas convergindo e episódios longos
- **Eficiência**: Consumo balanceado entre agentes
- **Sustentabilidade**: Recursos não se esgotando rapidamente
- **Adaptação**: Mudança de comportamento conforme normas evoluem

## 🛠️ Troubleshooting

### Problemas Comuns

1. **"ImportError: No module named 'pettingzoo'"**
   ```bash
   pip install pettingzoo supersuit
   ```

2. **"CUDA out of memory"**
   ```bash
   # Reduzir batch size ou usar CPU
   python train_ppo_normarl.py --batch-size 32
   ```

3. **Treinamento muito lento**
   ```bash
   # Reduzir n_envs ou complexity
   python train_ppo_normarl.py --n-envs 2 --n-steps 512
   ```

4. **Convergência ruim**
   ```bash
   # Ajustar learning rate e exploration
   python train_ppo_normarl.py --ppo-lr 1e-4 --ent-coef 0.05
   ```

## 📚 Referências

- **NORMARL**: [Norm-Adaptive Reinforcement Learning](https://arxiv.org/abs/xxxx.xxxxx)
- **PPO**: [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- **Stable Baselines 3**: [Documentation](https://stable-baselines3.readthedocs.io/)
- **PettingZoo**: [Multi-Agent RL Environments](https://pettingzoo.farama.org/)

## 🤝 Contribuindo

Para contribuir com melhorias no sistema de treinamento:

1. Implemente novos algoritmos (A2C, TD3, SAC)
2. Adicione métricas específicas para homeostase
3. Crie visualizações para normas sociais
4. Otimize hiperparâmetros automaticamente
5. Adicione suporte para ambientes 2D

---

🎯 **Happy Training!** O sistema está pronto para explorar como agentes artificiais podem aprender regulação homeostática e normas sociais através de Deep Reinforcement Learning! 
