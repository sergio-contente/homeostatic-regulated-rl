"""Homeostatic Environments Package"""

from .base_env import NormarlHomeostaticBaseEnv
from .multiagent import create_env, create_parallel_env, NormalHomeostaticEnv

# Unified interface like SUMO
def single_agent_env(**kwargs):
    """Create single-agent environment."""
    return NormarlHomeostaticBaseEnv(**kwargs)

def multi_agent_env(**kwargs):
    """Create multi-agent AEC environment.""" 
    return create_env(**kwargs)

def parallel_env(**kwargs):
    """Create parallel multi-agent environment."""
    return create_parallel_env(**kwargs)

__all__ = [
    'single_agent_env',
    'multi_agent_env', 
    'parallel_env',
    'NormarlHomeostaticBaseEnv',
    'NormalHomeostaticEnv'
]
