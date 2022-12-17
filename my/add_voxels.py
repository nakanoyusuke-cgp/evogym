import timeit

import gym
import evogym.envs
from evogym import sample_robot
import numpy as np
import random

# seed = 3
# random.seed(seed)
# np.random.seed(seed)

pd = [3./6, 1/6, 1/6, 1/6, 1/6, 0., 1/6, 0.]
body, connections = sample_robot((5, 5), pd=pd)

env = gym.make('Hunting-v0', body=body)
env.reset()


def step(n=1):
    for i in range(n):
        action = env.action_space.sample() - 1
        ob, reward, done, info = env.step(action)
        env.render()
        if done:
            env.reset()

    return ob


def print_state():
    tmp1 = env.sim.object_boxels_pos('robot')
    tmp2 = env.sim.object_boxels_type('robot')
    print(tmp1)
    print(tmp2)
    print(tmp1[:, tmp2 == 6])


print_state()
step(50)



# while True:
#     action = env.action_space.sample()-1
#     ob, reward, done, info = env.step(action)
#     env.render()
#
#     if done:
#         env.reset()
#
# env.close()
