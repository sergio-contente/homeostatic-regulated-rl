import os
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.utils import EzPickle, seeding
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers


def env(**kwargs):
    env = NormarlHomeostaticEnvPZ(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

parallel_env = parallel_wrapper_fn(env)

class NormarlHomeostaticEnv(gym.Env):
    """
    Single-agent homeostatic environment with social norms.
    Integrates NORMARL's social cost mechanism with homeostatic drives.
    Q (consumption in NORMARL) = K (intake in homeostatic system)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
            self,
            config_path,
            drive_type,
            social_learning_rate,
						beta, 
						render_mode=None,
						size=10
		) -> None:

			
