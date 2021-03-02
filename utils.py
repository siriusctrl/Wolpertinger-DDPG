from gym.spaces import Box
import numpy as np
from itertools import product

def all_actions(low, high, n) -> np.ndarray:
    # print(list(range(low, high+1)))
    return product(range(low, high+1), repeat=n)

def reverse_normalise(high, low, action):
    act_k = (high - low) / 2.
    act_b = (high + low) / 2.
    return act_k * action + act_b

def normalise(high, low, action):
    act_k_inv = 2. / (high - low)
    act_b = (high + low) / 2.
    return act_k_inv * (action - act_b)

if __name__=="__main__":
    action_space = Box(low=0, high=2, shape=(3,), dtype=np.int32)
    res = all_actions(action_space.low[0], action_space.high[0], action_space.shape[0])
    # print(len(res) == len(set(res)))
    res = np.array(list(res))
    print(res)
    res = normalise(action_space.high[0], action_space.low[0], res)
    print(res)