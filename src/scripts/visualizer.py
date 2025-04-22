import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from collections import Counter
import time
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

class HomeostaticVisualizer:
    """
    Ferramenta de visualização para agentes de regulação homeostática
    com dois recursos (recurso 0 e recurso 1).
    """
    
    def __init__(self, q_table, env=None, max_val=300.0, n_bins=50):
        """
        Inicializa o visualizador com uma tabela Q e ambiente.
        
        Args:
            q_table: Tabela Q do agente a ser visualizada
            env: O ambiente em que o agente foi treinado (opcional)
            max_val: Valor máximo para estados internos (padrão=300.0)
            n_bins: Número de bins para discretização (padrão=50)
        """
        self.q_table = q_table
        self.env = env
        self.max_val = max_val
        self.n_bins = n_bins
        self.action_size = q_table.shape[1]
        
        # Determinar a dimensão do espaço de estados
        state_size = q_table.shape[0]
        self.internal_state_dim = self._determine_state_dim(state_size, n_bins)
        
        print(f"Forma da tabela Q: {q_table.shape}")
        print(f"Número total de estados: {state_size}")
        print(f"Número de ações: {self.action_size}")
        print(f"Dimensão interna de estados detectada: {self.internal_state_dim}")
        print(f"Bins por dimensão: {self.n_bins}")
        
        # Define nomes das ações para este ambiente
        self.action_names = [
            "Consumir Recurso 0",
            "Consumir Recurso 1",
            "Não fazer nada"
        ]
        
        # Define nomes de recursos
        self.resource_names = [
            "Recurso 0",
            "Recurso 1"
        ]
        
        # Cores para visualização
        self.colors = {
            "resource0": "blue",
            "resource1": "red",
            "do_nothing": "gray",
            "background": "#f8f8f8",
            "grid": "#dddddd",
            "text": "#333333",
            "trajectory": "black",
            "optimal": "gold",
        }
    
    def _determine_state_dim(self, state_size, n_bins):
        """
        Determina a dimensão do espaço de estados.
        
        Args:
            state_size: Número total de estados
            n_bins: Número de bins por dimensão
            
        Returns:
            internal_state_dim: Dimensão do espaço de estados
        """
        # Tenta encontrar a dimensão que satisfaz state_size = n_bins^dim
        for dim in range(1, 5):  # Testa dimensões de 1 a 4
            if abs(state_size - n_bins**dim) < 1e-6:
                return dim
        
        # Se não encontrou uma correspondência exata, calcula a melhor aproximação
        best_dim = max(1, round(np.log(state_size) / np.log(n_bins)))
        print(f"⚠️ Aviso: Não foi possível determinar a dimensão exata do espaço de estados.")
        print(f"   Usando aproximação: {best_dim} dimensões com {n_bins} bins cada.")
        print(f"   Isso daria {n_bins**best_dim} estados, mas temos {state_size} estados.")
        return best_dim
    
    def visualize_q_values_per_action(self):
        """
        Cria uma visualização de valores Q para cada ação, mostrando mapas de calor.
        
        Returns:
            fig: O objeto figura do matplotlib
        """
        # Extrai valores Q para cada ação
        q_values_per_action = []
        
        # Para espaço bidimensional
        if self.internal_state_dim == 2:
            for i in range(min(self.action_size, len(self.action_names))):
                # Remodelar valores Q para visualização 2D
                q_values = self.q_table[:, i].reshape(self.n_bins, self.n_bins)
                q_values_per_action.append(q_values)
            
            # Cria figura com subplots para cada ação
            fig, axes = plt.subplots(1, len(q_values_per_action), figsize=(15, 5))
            
            # Define min/max comum para barra de cores consistente
            vmin = min(np.min(q) for q in q_values_per_action)
            vmax = max(np.max(q) for q in q_values_per_action)
            
            # Cria mapas de calor para cada ação
            for i, (values, ax) in enumerate(zip(q_values_per_action, axes)):
                im = ax.imshow(values, cmap='viridis', origin='lower', 
                               extent=[0, self.max_val, 0, self.max_val],
                               vmin=vmin, vmax=vmax)
                ax.set_title(self.action_names[i], fontsize=14)
                ax.set_xlabel('Estado do Recurso 1', fontsize=12)
                ax.set_ylabel('Estado do Recurso 0', fontsize=12)
                
                # Adiciona barra de cores a cada subplot
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(im, cax=cax)
                cbar.set_label('Valor Q', fontsize=10)
        else:
            # Para espaços com mais de 2 dimensões, mostrar apenas fatias representativas
            print(f"⚠️ Espaço de estados com {self.internal_state_dim} dimensões.")
            print("   Mostrando fatias 2D representativas.")
            
            # Criar múltiplas figuras para fatias representativas em 2D
            fig = plt.figure(figsize=(15, 5 * min(self.action_size, len(self.action_names))))
            
            for i in range(min(self.action_size, len(self.action_names))):
                # Para cada ação, mostrar uma fatia do espaço de estados
                ax = fig.add_subplot(min(self.action_size, len(self.action_names)), 1, i+1)
                
                # Selecionar fatia no meio das outras dimensões
                indices = [slice(0, self.n_bins)] * 2
                for d in range(2, self.internal_state_dim):
                    indices.append(self.n_bins // 2)  # Fatia central
                
                try:
                    # Tentar redimensionar para visualizar
                    q_slice = self.q_table[:, i].reshape([self.n_bins] * self.internal_state_dim)
                    q_slice = q_slice[tuple(indices)]
                    
                    im = ax.imshow(q_slice, cmap='viridis', origin='lower', 
                                  extent=[0, self.max_val, 0, self.max_val])
                    ax.set_title(f"{self.action_names[i]} (fatia central)", fontsize=14)
                    ax.set_xlabel('Dimensão 1', fontsize=12)
                    ax.set_ylabel('Dimensão 0', fontsize=12)
                    
                    # Adiciona barra de cores
                    plt.colorbar(im, ax=ax)
                except Exception as e:
                    print(f"Erro ao visualizar fatia para ação {i}: {e}")
                    ax.text(0.5, 0.5, f"Não foi possível visualizar\n(erro: {e})", 
                           ha='center', va='center', transform=ax.transAxes)
        
        # Adiciona título global com parâmetros de drive, se disponíveis
        if self.env and hasattr(self.env, 'drive'):
            drive = self.env.drive
            drive_params = ""
            
            if hasattr(drive, 'n') and hasattr(drive, 'm'):
                drive_params = f"n = {drive.n}, m = {drive.m}"
            
            plt.suptitle(f'Valores Q por Ação {drive_params}', fontsize=16)
        else:
            plt.suptitle('Valores Q por Ação', fontsize=16)
            
        plt.tight_layout()
        
        return fig
    
    def visualize_state_value_and_policy(self):
        """
        Cria uma visualização de valores de estado com política ótima sobreposta como setas.
        
        Returns:
            fig: O objeto figura do matplotlib
        """
        # Verifica se é possível visualizar em 2D
        if self.internal_state_dim != 2:
            print(f"⚠️ Não é possível visualizar diretamente um espaço de estados {self.internal_state_dim}D.")
            print("   Fazendo projeção 2D (fatia central para dimensões extra).")
        
        # Obtém política ótima (melhor ação) em cada estado
        policy = np.argmax(self.q_table, axis=1)
        
        try:
            # Tenta remodelar para visualização
            policy_grid = policy.reshape([self.n_bins] * self.internal_state_dim)
            
            # Se dimensões > 2, pega uma fatia central
            if self.internal_state_dim > 2:
                indices = [slice(0, self.n_bins)] * 2
                for d in range(2, self.internal_state_dim):
                    indices.append(self.n_bins // 2)
                policy_grid = policy_grid[tuple(indices)]
            
            # Obtém valor máximo Q para cada estado
            state_values = np.max(self.q_table, axis=1)
            state_values_grid = state_values.reshape([self.n_bins] * self.internal_state_dim)
            
            # Se dimensões > 2, pega a mesma fatia central
            if self.internal_state_dim > 2:
                state_values_grid = state_values_grid[tuple(indices)]
            
            # Cria figura
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Exibe valores de estado como mapa de calor
            im = ax.imshow(state_values_grid, cmap='plasma', origin='lower',
                          extent=[0, self.max_val, 0, self.max_val])
            
            # Cria uma grade para as setas de política
            arrow_density = 10  # Número de setas em cada dimensão
            step_size = self.n_bins // arrow_density
            
            # Desenha setas para a política ótima
            for i in range(0, self.n_bins, step_size):
                for j in range(0, self.n_bins, step_size):
                    action = policy_grid[i, j]
                    
                    # Calcula posição da seta em coordenadas de mapa de calor
                    x = j * self.max_val / self.n_bins + self.max_val / (2 * self.n_bins)
                    y = i * self.max_val / self.n_bins + self.max_val / (2 * self.n_bins)
                    
                    # Define direção da seta com base na ação
                    dx, dy = 0, 0
                    
                    if action == 0:  # Consumir Recurso 0
                        dx, dy = 0, 5  # Para cima
                    elif action == 1:  # Consumir Recurso 1
                        dx, dy = 5, 0  # Para direita
                    elif action == 2:  # Não fazer nada
                        # Para "não fazer nada", usar um marcador diferente
                        ax.plot(x, y, 'kx', markersize=6, markeredgewidth=2)
                        continue
                    
                    # Desenha a seta
                    ax.arrow(x, y, dx, dy, head_width=5, head_length=5, 
                             fc='black', ec='black', alpha=0.7)
            
            # Adiciona marcador de ponto ótimo, se disponível
            if self.env and hasattr(self.env, 'drive') and hasattr(self.env.drive, '_optimal_internal_states'):
                try:
                    optimal_states = self.env.drive._optimal_internal_states
                    if isinstance(optimal_states, (list, np.ndarray)) and len(optimal_states) >= 2:
                        optimal_x = optimal_states[1]  # Recurso 1
                        optimal_y = optimal_states[0]  # Recurso 0
                        ax.plot(optimal_x, optimal_y, 'o', color=self.colors["optimal"], 
                               markersize=15, markeredgecolor='black')
                except Exception as e:
                    print(f"Não foi possível marcar o ponto ótimo: {e}")
            
            # Adiciona barra de cores
            cbar = plt.colorbar(im)
            cbar.set_label('Valor do Estado', fontsize=12)
            
            # Define rótulos e título
            ax.set_xlabel('Estado do Recurso 1', fontsize=14)
            ax.set_ylabel('Estado do Recurso 0', fontsize=14)
            ax.set_title('Mapa de Valor-Ação-Estado e Política Ótima', fontsize=16)
            
            # Adiciona parâmetros de drive como subtítulo, se disponíveis
            if self.env and hasattr(self.env, 'drive'):
                drive = self.env.drive
                drive_type = type(drive).__name__
                
                params = []
                if hasattr(drive, 'n'):
                    params.append(f"n = {drive.n}")
                if hasattr(drive, 'm'):
                    params.append(f"m = {drive.m}")
                if hasattr(drive, 'eta'):
                    params.append(f"η = {drive.eta}")
                
                drive_params = f"{drive_type}: {', '.join(params)}"
                plt.figtext(0.5, 0.01, drive_params, fontsize=12, ha='center')
            
            plt.tight_layout()
            
            return fig
        except Exception as e:
            print(f"⚠️ Erro ao visualizar política: {e}")
            # Criar figura de erro
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"Não foi possível visualizar a política\nErro: {e}", 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
    
    def visualize_policy(self):
        """
        Cria uma visualização da política do agente.
        
        Returns:
            fig: O objeto figura do matplotlib
        """
        # Obtém política ótima (melhor ação) em cada estado
        policy = np.argmax(self.q_table, axis=1)
        
        try:
            # Tenta remodelar para visualização
            policy_grid = policy.reshape([self.n_bins] * self.internal_state_dim)
            
            # Se dimensões > 2, pega uma fatia central
            if self.internal_state_dim > 2:
                indices = [slice(0, self.n_bins)] * 2
                for d in range(2, self.internal_state_dim):
                    indices.append(self.n_bins // 2)
                policy_grid = policy_grid[tuple(indices)]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Criar um mapa de cores personalizado para as ações
            cmap = plt.cm.get_cmap('tab10', self.action_size)
            
            im = ax.imshow(policy_grid, cmap=cmap, origin="lower", 
                         extent=[0, self.max_val, 0, self.max_val])
            
            # Barra de cores personalizada com labels de ações
            action_names_short = ["Cons. R0", "Cons. R1", "Esperar"]
            action_names_to_use = action_names_short[:self.action_size]
            
            cbar = plt.colorbar(im, ticks=range(self.action_size))
            cbar.set_label('Ação', fontsize=12)
            cbar.set_ticklabels(action_names_to_use)
            
            # Configurar rótulos e título
            ax.set_xlabel('Estado do Recurso 1', fontsize=14)
            ax.set_ylabel('Estado do Recurso 0', fontsize=14)
            
            if self.internal_state_dim > 2:
                title = f'Política Ótima (fatia central do espaço {self.internal_state_dim}D)'
            else:
                title = 'Política Ótima (ação por estado)'
                
            ax.set_title(title, fontsize=16)
            
            plt.tight_layout()
            
            return fig
        except Exception as e:
            print(f"⚠️ Erro ao visualizar política: {e}")
            # Criar figura de erro
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"Não foi possível visualizar a política\nErro: {e}", 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig

    def visualize_3d_value_function(self):
        """
        Cria uma visualização 3D da função de valor.
        
        Returns:
            fig: O objeto figura do matplotlib
        """
        # Obtém valor máximo Q para cada estado
        state_values = np.max(self.q_table, axis=1)
        
        try:
            # Tenta remodelar para visualização
            state_values_grid = state_values.reshape([self.n_bins] * self.internal_state_dim)
            
            # Se dimensões > 2, pega uma fatia central
            if self.internal_state_dim > 2:
                indices = [slice(0, self.n_bins)] * 2
                for d in range(2, self.internal_state_dim):
                    indices.append(self.n_bins // 2)
                state_values_grid = state_values_grid[tuple(indices)]
            
            # Cria figura 3D
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Cria malha de coordenadas
            x = np.linspace(0, self.max_val, self.n_bins)
            y = np.linspace(0, self.max_val, self.n_bins)
            X, Y = np.meshgrid(x, y)
            
            # Traça superfície 3D
            surf = ax.plot_surface(X, Y, state_values_grid, cmap='viridis', 
                                  linewidth=0, antialiased=True)
            
            # Adiciona barra de cores
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Valor do Estado')
            
            # Configura rótulos e título
            ax.set_xlabel('Estado do Recurso 1', fontsize=12)
            ax.set_ylabel('Estado do Recurso 0', fontsize=12)
            ax.set_zlabel('Valor (max Q)', fontsize=12)
            
            if self.internal_state_dim > 2:
                title = f'Função de Valor em 3D (fatia central do espaço {self.internal_state_dim}D)'
            else:
                title = 'Função de Valor em 3D'
                
            ax.set_title(title, fontsize=16)
            
            # Define ângulo de visão
            ax.view_init(elev=30, azim=45)
            
            plt.tight_layout()
            
            return fig
        except Exception as e:
            print(f"⚠️ Erro ao visualizar função de valor 3D: {e}")
            # Criar figura de erro
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"Não foi possível visualizar a função de valor em 3D\nErro: {e}", 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig

def load_and_visualize_qlearning_model(model_path, output_dir='./', env=None, max_val=300.0, n_bins=50):
    """
    Carrega um modelo Q-learning salvo e gera visualizações.
    
    Args:
        model_path: Caminho para o arquivo pickle contendo a tabela Q
        output_dir: Diretório para salvar as visualizações
        env: Ambiente (opcional)
        max_val: Valor máximo para estados internos
        n_bins: Número de bins usados na discretização
    """
    # Carregar a tabela Q
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        # Se o arquivo contém um dicionário com a tabela Q
        if isinstance(data, dict) and 'q_table' in data:
            q_table = data['q_table']
        # Se o arquivo contém diretamente a tabela Q
        elif isinstance(data, np.ndarray):
            q_table = data
        else:
            raise ValueError("Formato de arquivo não reconhecido. Esperava um dicionário com 'q_table' ou um array numpy.")
        
        print(f"Tabela Q carregada do arquivo {model_path}")
        print(f"Forma da tabela Q: {q_table.shape}")
    except Exception as e:
        print(f"⚠️ Erro ao carregar o modelo: {e}")
        return None
    
    # Criar visualizador
    visualizer = HomeostaticVisualizer(q_table, env, max_val, n_bins)
    
    # Gerar e salvar visualizações
    print("Gerando visualizações...")
    
    # Visualização dos valores Q por ação
    fig1 = visualizer.visualize_q_values_per_action()
    if fig1:
        fig1.savefig(f"{output_dir}/q_values_por_acao.png", dpi=300, bbox_inches='tight')
    
    # Visualização de valor-estado e política
    fig2 = visualizer.visualize_state_value_and_policy()
    if fig2:
        fig2.savefig(f"{output_dir}/valor_estado_e_politica.png", dpi=300, bbox_inches='tight')
    
    # Visualização da política
    fig3 = visualizer.visualize_policy()
    if fig3:
        fig3.savefig(f"{output_dir}/politica_otima.png", dpi=300, bbox_inches='tight')
    
    # Visualização 3D da função de valor
    fig4 = visualizer.visualize_3d_value_function()
    if fig4:
        fig4.savefig(f"{output_dir}/funcao_valor_3d.png", dpi=300, bbox_inches='tight')
    
    print(f"Visualizações salvas em {output_dir}")
    
    plt.show()  # Mostrar figuras interativamente
    
    return visualizer

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from collections import Counter

class LinearVisualizer:
    """
    Ferramenta de visualização para agentes Q-learning com estados discretizados.
    Esta versão não tenta reconstituir a estrutura multidimensional do espaço de estados.
    """
    
    def __init__(self, q_table, env=None, max_val=300.0):
        """
        Inicializa o visualizador com uma tabela Q.
        
        Args:
            q_table: Tabela Q do agente a ser visualizada
            env: O ambiente em que o agente foi treinado (opcional)
            max_val: Valor máximo para estados internos (padrão=300.0)
        """
        self.q_table = q_table
        self.env = env
        self.max_val = max_val
        
        # Análise básica da tabela Q
        self.state_size, self.action_size = q_table.shape
        
        print(f"Forma da tabela Q: {q_table.shape}")
        print(f"Número total de estados: {self.state_size}")
        print(f"Número de ações: {self.action_size}")
        
        # Define nomes das ações para este ambiente
        self.action_names = [
            "Consumir Recurso 0",
            "Consumir Recurso 1",
            "Não fazer nada"
        ]
    
    def visualize_q_distribution(self):
        """
        Cria uma visualização da distribuição de valores Q para todas as ações.
        
        Returns:
            fig: O objeto figura do matplotlib
        """
        fig, axes = plt.subplots(self.action_size, 1, figsize=(10, 4*self.action_size))
        
        if self.action_size == 1:
            axes = [axes]  # Garantir que axes seja uma lista
        
        for i, ax in enumerate(axes):
            # Obter valores Q para esta ação
            q_values = self.q_table[:, i]
            
            # Criar histograma
            ax.hist(q_values, bins=50, alpha=0.7, color=f'C{i}')
            ax.set_title(f'Distribuição de Valores Q para {self.action_names[i] if i < len(self.action_names) else f"Ação {i}"}')
            ax.set_xlabel('Valor Q')
            ax.set_ylabel('Frequência')
            
            # Adicionar estatísticas
            mean_q = np.mean(q_values)
            median_q = np.median(q_values)
            max_q = np.max(q_values)
            min_q = np.min(q_values)
            
            stats_text = (f'Mín: {min_q:.3f}, Máx: {max_q:.3f}\n'
                          f'Média: {mean_q:.3f}, Mediana: {median_q:.3f}')
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Linha vertical para a média
            ax.axvline(mean_q, color='red', linestyle='dashed', linewidth=1, label=f'Média: {mean_q:.3f}')
            ax.legend()
        
        plt.tight_layout()
        return fig
    
    def visualize_policy_distribution(self):
        """
        Cria uma visualização da distribuição de ações na política ótima.
        
        Returns:
            fig: O objeto figura do matplotlib
        """
        # Obtém política ótima (melhor ação) em cada estado
        policy = np.argmax(self.q_table, axis=1)
        
        # Conta frequência de cada ação
        action_counts = np.bincount(policy, minlength=self.action_size)
        action_percentages = action_counts / len(policy) * 100
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Cores para cada ação
        colors = plt.cm.tab10(np.arange(self.action_size))
        
        # Criar gráfico de barras
        bars = ax.bar(range(self.action_size), action_percentages, color=colors)
        
        # Adicionar rótulos
        ax.set_xticks(range(self.action_size))
        action_labels = [self.action_names[i] if i < len(self.action_names) else f"Ação {i}" 
                        for i in range(self.action_size)]
        ax.set_xticklabels(action_labels, rotation=30, ha='right')
        ax.set_ylabel('Porcentagem de Estados (%)')
        ax.set_title('Distribuição de Ações na Política Ótima')
        
        # Adicionar valores nas barras
        for i, (bar, count, percentage) in enumerate(zip(bars, action_counts, action_percentages)):
            ax.text(i, percentage + 1, f'{count}\n({percentage:.1f}%)', 
                   ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def visualize_q_value_correlation(self):
        """
        Cria uma visualização da correlação entre valores Q para diferentes ações.
        Útil para entender como as ações se relacionam.
        
        Returns:
            fig: O objeto figura do matplotlib
        """
        if self.action_size <= 1:
            return None  # Precisa de pelo menos duas ações
            
        fig, axes = plt.subplots(1, self.action_size * (self.action_size - 1) // 2, 
                                figsize=(4 * self.action_size, 4))
        
        # Para uma única correlação, garantir que axes seja uma lista
        if self.action_size == 2:
            axes = [axes]
        
        # Para cada par de ações, criar um scatter plot
        plot_idx = 0
        for i in range(self.action_size):
            for j in range(i+1, self.action_size):
                # Obter os valores Q
                q_values_i = self.q_table[:, i]
                q_values_j = self.q_table[:, j]
                
                # Calcular coeficiente de correlação
                corr = np.corrcoef(q_values_i, q_values_j)[0, 1]
                
                # Criar o scatter plot
                ax = axes[plot_idx]
                ax.scatter(q_values_i, q_values_j, alpha=0.1, s=1)
                
                # Adicionar rótulos
                action_i_name = self.action_names[i] if i < len(self.action_names) else f"Ação {i}"
                action_j_name = self.action_names[j] if j < len(self.action_names) else f"Ação {j}"
                
                ax.set_xlabel(f'Q({action_i_name})')
                ax.set_ylabel(f'Q({action_j_name})')
                ax.set_title(f'Correlação: {corr:.3f}')
                
                # Adicionar linha diagonal para referência
                if q_values_i.min() < q_values_j.min():
                    min_val = q_values_i.min()
                else:
                    min_val = q_values_j.min()
                    
                if q_values_i.max() > q_values_j.max():
                    max_val = q_values_i.max()
                else:
                    max_val = q_values_j.max()
                
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.3)
                
                plot_idx += 1
        
        plt.tight_layout()
        return fig
    
    def visualize_value_function(self):
        """
        Cria uma visualização da distribuição da função valor (máximo Q).
        
        Returns:
            fig: O objeto figura do matplotlib
        """
        # Obter valores máximos de Q para cada estado
        value_function = np.max(self.q_table, axis=1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Criar histograma dos valores
        ax.hist(value_function, bins=50, alpha=0.7, color='purple')
        ax.set_title('Distribuição da Função Valor (Máximo Q)')
        ax.set_xlabel('Valor')
        ax.set_ylabel('Frequência')
        
        # Adicionar estatísticas
        mean_val = np.mean(value_function)
        median_val = np.median(value_function)
        max_val = np.max(value_function)
        min_val = np.min(value_function)
        
        stats_text = (f'Mín: {min_val:.3f}, Máx: {max_val:.3f}\n'
                      f'Média: {mean_val:.3f}, Mediana: {median_val:.3f}')
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Adicionar linhas de referência
        ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label=f'Média: {mean_val:.3f}')
        ax.axvline(median_val, color='green', linestyle='dashed', linewidth=1, label=f'Mediana: {median_val:.3f}')
        
        ax.legend()
        plt.tight_layout()
        
        return fig

def load_and_visualize_qlearning_model(model_path, output_dir='./', env=None, max_val=300.0):
    """
    Carrega um modelo Q-learning salvo e gera visualizações.
    
    Args:
        model_path: Caminho para o arquivo pickle contendo a tabela Q
        output_dir: Diretório para salvar as visualizações
        env: Ambiente (opcional)
        max_val: Valor máximo para estados internos
    """
    # Criar diretório de saída se não existir
    if not os.path.exists(output_dir):
        print(f"Criando diretório de saída: {output_dir}")
        os.makedirs(output_dir)
    
    # Carregar a tabela Q
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        # Se o arquivo contém um dicionário com a tabela Q
        if isinstance(data, dict) and 'q_table' in data:
            q_table = data['q_table']
        # Se o arquivo contém diretamente a tabela Q
        elif isinstance(data, np.ndarray):
            q_table = data
        else:
            raise ValueError("Formato de arquivo não reconhecido. Esperava um dicionário com 'q_table' ou um array numpy.")
        
        print(f"Tabela Q carregada do arquivo {model_path}")
    except Exception as e:
        print(f"⚠️ Erro ao carregar o modelo: {e}")
        return None
    
    # Criar visualizador
    visualizer = LinearVisualizer(q_table, env, max_val)
    
    # Gerar e salvar visualizações
    print("Gerando visualizações...")
    
    # Visualização da distribuição de valores Q
    fig1 = visualizer.visualize_q_distribution()
    if fig1:
        fig1.savefig(f"{output_dir}/q_distribuicao.png", dpi=300, bbox_inches='tight')
        print(f"✓ Salvo: {output_dir}/q_distribuicao.png")
    
    # Visualização da distribuição de política
    fig2 = visualizer.visualize_policy_distribution()
    if fig2:
        fig2.savefig(f"{output_dir}/distribuicao_politica.png", dpi=300, bbox_inches='tight')
        print(f"✓ Salvo: {output_dir}/distribuicao_politica.png")
    
    # Visualização da correlação de valores Q
    fig3 = visualizer.visualize_q_value_correlation()
    if fig3:
        fig3.savefig(f"{output_dir}/q_correlacao.png", dpi=300, bbox_inches='tight')
        print(f"✓ Salvo: {output_dir}/q_correlacao.png")
    
    # Visualização da função valor
    fig4 = visualizer.visualize_value_function()
    if fig4:
        fig4.savefig(f"{output_dir}/funcao_valor.png", dpi=300, bbox_inches='tight')
        print(f"✓ Salvo: {output_dir}/funcao_valor.png")
    
    print(f"Visualizações salvas em {output_dir}")
    
    plt.show()  # Mostrar figuras interativamente
    
    return visualizer

# Exemplo de uso
if __name__ == "__main__":
    model_path = "model/clementine/qlearning_model_base_drive_n1_m1.pkl"
    output_dir = "./"
    
    # Carregar e visualizar o modelo
    visualizer = load_and_visualize_qlearning_model(
        model_path=model_path,
        output_dir=output_dir,
        max_val=300.0,
        n_bins=50
    )
