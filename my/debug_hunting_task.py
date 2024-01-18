import timeit

import gym
import evogym.envs
from evogym import sample_robot
import numpy as np
import random
import os
import sys


pd = np.array([3., 1., 1., 1., 1., 0., 1., 0., 1.])
body, connections = sample_robot((5, 5), pd=pd)

body = np.array([
    [2, 2, 2, 2, 2],
    [2, 0, 0, 0, 2],
    [6, 6, 3, 1, 2],
    [4, 1, 0, 3, 8],
    [6, 8, 0, 3, 6],
])
# print(body)

env_idx = 31

if env_idx == 1:
    env = gym.make('HuntCreeper-v0', body=body)
elif env_idx == 2:
    env = gym.make('HuntHopper-v0', body=body)
elif env_idx == 3:
    env = gym.make('HuntFlyer-v0', body=body)
elif env_idx == 0:
    env = gym.make('Observer_vis1-v0', body=body)
elif env_idx == 11:
    env = gym.make('HuntCreeper_vis1-v0', body=body)
elif env_idx == 12:
    env = gym.make("HuntCreeper_vis1-v1", body=body)
elif env_idx == 13:
    env = gym.make("HuntHugeCreeper_vis1-v0", body=body)
elif env_idx == 21:
    env = gym.make("HuntHopperVis-v0", body=body)
elif env_idx == 31:
    env = gym.make("HuntFlyerVis-v0", body=body)

# baselines
elif env_idx == 100:
    env = gym.make("HuntCreeperBaseline-v0", body=body)
elif env_idx == 110:
    env = gym.make("HuntHopperBaseline-v0", body=body)
elif env_idx == 120:
    env = gym.make("HuntFlyerBaseline-v0", body=body)

elif env_idx == 101:
    env = gym.make("HuntCreeperBaselineVis-v0", body=body)
elif env_idx == 111:
    env = gym.make("HuntHopperBaselineVis-v0", body=body)
elif env_idx == 121:
    env = gym.make("HuntFlyerBaselineVis-v0", body=body)

elif env_idx == 102:
    env = gym.make("HuntCreeperBaselineVis-v1", body=body)

else:
    exit(1)

obs = env.reset()
# env.render()
#
# print(obs)
# print(obs.size)


state = None


def tmp_debug(env):
    print(env.get_robot_prey_diff())


def step(env, n=1, verbose=False, verbose_interval=1):
    global state
    for i in range(n):
        action = env.action_space.sample() * 0.0
        # action = env.action_space.sample() - 1
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


def check_hunt_rewards(env, n=1):
    global state
    for i in range(n):
        action = env.action_space.sample() * 0.0
        # action = env.action_space.sample() -1.0
        ob, reward, done, info = env.step(action)
        env.render()

        print("reward:", reward)
        # print("info:", info)
        if done:
            env.reset()

def print_state(env):
    tmp1 = env.sim.object_boxels_pos('robot')
    tmp2 = env.sim.object_boxels_type('robot')
    print(tmp1)
    print(tmp2)
    print(tmp1[:, tmp2 == 6])

# ---
#
# env = gym.make("HuntCreeperBaselineVis-v1", body=body)
# for i in range(1):
#     env.seed(1)
#     l = []
#     for i in range(1):
#         env.reset()
#         l += [env.init_prey_x_pos]
#     print(l)

# step(env, 100)
