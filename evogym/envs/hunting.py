import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

from evogym import *
from evogym.envs import BenchmarkBase, PackageBase

import random
import math
import numpy as np
import os


SENSING_RANGE = 0.5
ESCAPE_VELOCITY = 0.002
HOPPING = 0.001


class Hunting(BenchmarkBase):
    def __init__(self, body, connections=None):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Walker-v0.json'))
        self.world.add_from_array('robot', body, 1, 1, connections=connections)
        self.world.add_from_array('prey', np.array([[7]]), 8, 1)
        # robotであるか否かをどこで判断しているか
        # actuator >= 1

        # init sim
        BenchmarkBase.__init__(self, self.world)
        self.default_viewer.track_objects('robot', 'prey')

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size
        num_pred_voxels = np.sum((self.object_voxels_type('robot') == VOXEL_TYPES['PRED']), dtype=np.int)

        self.action_space = spaces.Box(low=0.6, high=1.6, shape=(num_actuators, ), dtype=np.float)
        # robot vel: 2, diffs: 2 * num_pred_voxels, prey vel: 2, relative pos: num_robot_points
        self.observation_space = spaces.Box(
            low=-100, high=100.0, shape=(2 + 2 * num_pred_voxels + 2 + num_robot_points, ), dtype=np.float)

    def step(self, action: np.ndarray):
        # step
        done = super().step({'robot': action})

        # prey behave
        self.prey_behave()

        # get prey pred diffs
        prey_pred_diffs = self.get_prey_pred_diffs()

        # compute reward
        reward = self.get_reward(prey_pred_diffs=prey_pred_diffs)

        # generate observation
        obs = np.concatenate((
            np.mean(self.object_vel_at_time(self.get_time(), 'robot'), axis=1),
            self.get_prey_pred_diffs().reshape(-1),
            np.mean(self.object_vel_at_time(self.get_time(), 'prey'), axis=1),
            self.get_relative_pos_obs('robot'),
        ))

        done_info = ''

        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0
            done_info = 'The simulation was terminated because it became unstable.'

        # the prey completely escaped from predatory robot
        prey_com_pos = np.mean(self.object_pos_at_time(self.get_time(), 'prey'), axis=1)
        if prey_com_pos[0] > 99*self.VOXEL_SIZE:
            done = True
            reward = -1
            done_info = 'The simulation was terminated because the prey completely escaped from the predatory robot.'

        # observation, reward, has simulation met termination conditions, debugging info
        return obs, reward, done, {
            'prey_pred_diffs': prey_pred_diffs,
            'done_info': done_info,
        }

    def reset(self):
        super().reset()

        # observation
        # robot_velocity 2, prey_pred_diffs num_pred_voxels * 2, prey_velocity, relative_pos num_points
        obs = np.concatenate((
            np.mean(self.object_vel_at_time(self.get_time(), 'robot'), axis=1),
            self.get_prey_pred_diffs().reshape(-1),
            np.mean(self.object_vel_at_time(self.get_time(), 'prey'), axis=1),
            self.get_relative_pos_obs('robot'),
        ))

        return obs

    def get_reward(self, prey_pred_diffs):
        # 捕食器官のうち、最も捕食対象に近いブロックを取り上げる
        #  - ボクセルインデックスとボクセルの種類の紐付けが必要
        # 距離が1.8以下の時捕食しているとみなす
        # print()

        sqr_dist = np.min(np.sum((prey_pred_diffs * prey_pred_diffs), axis=1))
        return 0.01 / sqr_dist

    def get_prey_pred_diffs(self):
        robot_boxels_pos = self.object_voxels_pos('robot')
        robot_boxels_type = self.object_voxels_type('robot')
        prey_boxels_pos = self.object_voxels_pos('prey')
        pred_boxels_pos = robot_boxels_pos[:, robot_boxels_type == VOXEL_TYPES['PRED']]
        return prey_boxels_pos.T - pred_boxels_pos

    def prey_behave(self):
        robot_com_pos = np.mean(self.object_pos_at_time(self.get_time(), 'robot'), axis=1)
        prey_com_pos = np.mean(self.object_pos_at_time(self.get_time(), 'prey'), axis=1)
        diff = prey_com_pos - robot_com_pos
        if np.sum(diff ** 2) < SENSING_RANGE ** 2:
            self.sim.translate_object(
                ESCAPE_VELOCITY * ((diff[0] >= 0.0) * 2 - 1),
                HOPPING, 'prey')
