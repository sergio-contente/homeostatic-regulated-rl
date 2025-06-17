import ray
import supersuit as ss
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from torch import nn
from pettingzoo.butterfly import pistonball_v6

class PistonballCNN(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        
        # CNN architecture specifically designed for pistonball
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, [8, 8], stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, [4, 4], stride=(2, 2)),
            nn.ReLU(), 
            nn.Conv2d(64, 64, [3, 3], stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),  # Critical: 3136 calculated from preprocessed shape
            nn.ReLU(),
        )
        self.policy_fn = nn.Linear(512, num_outputs)
        self.value_fn = nn.Linear(512, 1)

    def forward(self, input_dict, state, seq_lens):
        # Convert from NHWC to NCHW format for PyTorch
        model_out = self.model(input_dict["obs"].permute(0, 3, 1, 2))
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state

    def value_function(self):
        return self._value_out.flatten()

def env_creator(args):
    env = pistonball_v6.parallel_env(
        n_pistons=20,
        time_penalty=-0.1,
        continuous=True,
        random_drop=True,
        random_rotate=True,
        ball_mass=0.75,
        ball_friction=0.3,
        ball_elasticity=1.5,
        max_cycles=125,
    )
    
    # Essential SuperSuit preprocessing - transforms [457, 120, 3] to [84, 84, 3]
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.dtype_v0(env, "float32")
    env = ss.resize_v1(env, x_size=84, y_size=84)  # Key transformation
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    env = ss.frame_stack_v1(env, 3)
    return env

# Setup and configuration
ray.init()
env_name = "pistonball_v6"

register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
ModelCatalog.register_custom_model("PistonballCNN", PistonballCNN)

# Use legacy API to avoid new stack complications
config = (
    PPOConfig()
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )
    .environment(env=env_name)
    .framework("torch")
    .env_runners(num_env_runners=4, num_envs_per_env_runner=1)
    .training(
        model={"custom_model": "PistonballCNN"},
        lambda_=0.95,
        kl_coeff=0.5,
        clip_param=0.1,
        vf_loss_coeff=1,
        entropy_coeff=0.01,
        train_batch_size=5000
    )
    .multi_agent(
        policies={"shared_policy": (None, None, None, {})},
        policy_mapping_fn=(lambda agent_id, episode, **kwargs: "shared_policy"),
    )
)

# Train the agent
from ray import tune
tune.run("PPO", name="PPO", config=config, stop={"timesteps_total": 5000000})
