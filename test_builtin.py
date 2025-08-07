import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from src.pettingzoo_env.normarl import NormalHomeostaticEnv
from pettingzoo.utils.conversions import aec_to_parallel
from pettingzoo.test import parallel_api_test

if __name__ == "__main__":
    env = NormalHomeostaticEnv(
        		config_path="config/config.yaml",
            drive_type="base_drive",
            learning_rate=0.1,
            beta=0.5,
            number_resources=1,
            n_agents=3,
            size=5)
    env = aec_to_parallel(env)
    parallel_api_test(env, num_cycles=1_000_000)
