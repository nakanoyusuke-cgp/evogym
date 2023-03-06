import timeit

import gym
import evogym.envs
from evogym import sample_robot
import numpy as np
import random

# seed = 3
# random.seed(seed)
# np.random.seed(seed)

pd = np.array([3./6, 1/6, 1/6, 1/6, 1/6, 0., 1/6, 0.])
body, connections = sample_robot((5, 5), pd=pd)

# body = np.array([
#     [6, 3, 3, 3, 6],
#     [3, 3, 3, 3, 3],
#     [3, 3, 0, 3, 3],
#     [3, 0, 0, 0, 3],
#     [3, 0, 0, 0, 3],
# ])
# print(body)

env_idx = 2

if env_idx == 0:
    env = gym.make('HuntCreeper-v0', body=body)
elif env_idx == 1:
    env = gym.make('HuntHopper-v0', body=body)
elif env_idx == 2:
    env = gym.make('HuntFlyer-v0', body=body)
else:
    exit(1)

obs = env.reset()
env.render()

print(obs)
print(obs.size)


state = None


def tmp_debug(env):
    print(env.get_robot_prey_diff())


def step(env, n=1, verbose=False, verbose_interval=1):
    global state
    for i in range(n):
        action = env.action_space.sample() - 1
        ob, reward, done, info = env.step(action)
        env.render()

        # d = env.sim.ground_on_robot("prey", "robot")
        # print(d)
        # if 0.0 <= d < 0.1:
        #     return ob

        # if state != info['state']:
        #     state = info['state']
        #     print(state)
        #     # if state == 'after_landing':
        #     #     return obs
        #     if state == 'jumping':
        #         print(env.object_pos_at_time(env.get_time(), 'prey')[1], env.sim.ground_on_robot('prey', 'robot'))

        if verbose and ((i + 1) % verbose_interval == 0):
            print("reward:", reward)
            print("info:", info)
        if done:
            env.reset()

    return ob


def print_state(env):
    tmp1 = env.sim.object_boxels_pos('robot')
    tmp2 = env.sim.object_boxels_type('robot')
    print(tmp1)
    print(tmp2)
    print(tmp1[:, tmp2 == 6])
