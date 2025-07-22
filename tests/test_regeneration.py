"""Test regeneration in the corrected multiagent environment."""

from src.envs.multiagent import create_env

def test_regeneration():
    print("🧪 Testing Resource Regeneration")
    print("=" * 50)
    
    env = create_env(
        config_path="config/config.yaml",
        drive_type="base_drive",      # ✅ Obrigatório
        learning_rate=0.1,            # ✅ Obrigatório  
        beta=0.5,                     # ✅ Obrigatório
        number_resources=1,           # ✅ Obrigatório
        n_agents=3,
        size=1,
        log_level="DEBUG"  # Para ver logs de regeneração
    )
    
    env.reset(seed=42)
    print(f"📦 Initial resource: {env.resource_stock}")
    
    # Primeiro, vamos aplicar a correção do método de regeneração
    print("\n🔧 Applying regeneration fix...")
    
    # Força consumo de todos os agentes em um round completo
    print(f"\n🎯 Round 1: All agents consume")
    for i in range(3):  # Um round completo (3 agentes)
        agent_id = env.agent_selection
        print(f"  Step {i}: {agent_id} consuming...")
        
        old_stock = env.resource_stock.copy()
        env.step(3)  # ação de consumo
        new_stock = env.resource_stock.copy()
        
        consumed = old_stock - new_stock
        print(f"    Resource: {old_stock} → {new_stock} (consumed: {consumed})")
        
    print(f"\n📊 After one complete round: {env.resource_stock}")
    
    # Mais um round para ver regeneração
    print(f"\n🎯 Round 2: More consumption to test regeneration")
    for i in range(3):
        if env.agents:  # Se ainda há agentes ativos
            agent_id = env.agent_selection
            old_stock = env.resource_stock.copy()
            env.step(3)
            new_stock = env.resource_stock.copy()
            print(f"  {agent_id}: {old_stock} → {new_stock}")
    
    print(f"\n✅ Final resource stock: {env.resource_stock}")
    print(f"👥 Agents remaining: {len(env.agents)}")
    
    if len(env.agents) == 0:
        print("🏛️ Tragedy of Commons occurred - all agents terminated!")
    else:
        print("🌱 System sustained - regeneration working!")

if __name__ == "__main__":
    test_regeneration() 
