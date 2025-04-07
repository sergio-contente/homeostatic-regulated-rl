import gymnasium as gym
from ...drives.base_drive import BaseDrive

class BaseDriveRewardWrapper(gym.Wrapper):
    def __init__(self, env, optimal_internal_states, m=1, n=1):
        super().__init__(env)
        self.drive = BaseDrive(optimal_internal_states, m=1, n=1)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        current_internal_states = info['internal_states']
        outcome = obs['outcome']
        reward = self.drive.compute_reward(current_internal_states=current_internal_states, outcome=outcome)
        return obs, reward, terminated, truncated, info
