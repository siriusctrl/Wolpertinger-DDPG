#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gym

# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        # use this wrapper in Store environment
        # return act_k * action + act_b, action
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)


if __name__ == '__main__':
    from gym.spaces import Box
    import numpy as np

    def action(high, low, action):
        act_k = (high - low)/ 2.
        act_b = (high + low)/ 2.
        return act_k * action + act_b

    def reverse_action(high, low, action):
        act_k_inv = 2./(high - low)
        act_b = (high + low)/ 2.
        return act_k_inv * (action - act_b)

    action_space = Box(low=0, high=10, shape=(5,), dtype=np.int32)
    sample = action_space.sample()
    print(f'The action space is {sample}')
    ac = reverse_action(action_space.high, action_space.low, sample)
    ra = action(action_space.high, action_space.low, ac)
    print(f'After function action {ac}, after reverse {ra}')
