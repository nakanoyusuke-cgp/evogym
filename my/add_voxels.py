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

env = gym.make('Walker-v0', body=body)
env.reset()
# tmp = env.sim.object_boxels_type('robot')
tmp = env.sim.object_boxels_pos('robot')
# tmp = env.sim.get_indices_of_actuators('robot')
print(tmp)
env.render()

def step(n=1):
    for i in range(n):
        action = env.action_space.sample() - 1
        ob, reward, done, info = env.step(action)
        env.render()
        if done:
            env.reset()

    return ob

# while True:
#     action = env.action_space.sample()-1
#     ob, reward, done, info = env.step(action)
#     env.render()
#
#     if done:
#         env.reset()
#
# env.close()
