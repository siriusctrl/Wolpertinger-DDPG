import gym
from gym.spaces import Box
import numpy as np
from normalized_env import NormalizedEnv

class test_env(gym.Env):

    def __init__(self, config={}) -> None:
        super().__init__()
        self.config = config
        self.action_space = Box(low=0, high=5, shape=(5,), dtype=np.int32)
        self.observation_space = Box(low=0, high=100, shape=(5,))
    
    def reset(self):
        return self.observation_space.sample()
    
    def step(self, action):
        action = np.round(action)
        reward = (len(set(action)) - 5) * np.random.normal(loc=3)

        new_obs = np.random.normal(loc=reward, size=5)
        # print(f'Action is {action}, reward is {reward}')

        info = {'log': f'Action is {action}, reward is {reward}' }

        return(
            new_obs,
            reward,
            False,
            info
        )

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env

if __name__ == '__main__':
    env = NormalizedEnv(test_env())
    model = PPO(MlpPolicy, env, device='cuda', verbose=1)
    model.learn(total_timesteps=100000)

    n = 0

    obs = env.reset()
    while n < 100:
        action, _state = model.predict(obs)
        # print(action)
        obs, rewards, dones, info = env.step(action)
        print(info['log'])
        n += 1

    