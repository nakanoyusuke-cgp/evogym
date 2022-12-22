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


class Hunting(BenchmarkBase):
    def __init__(self, body, connections=None):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Walker-v0.json'))
        self.world.add_from_array('robot', body, 1, 1, connections=connections)
        self.world.add_from_array('prey', np.array([[7]]), 3, 1)
        # robotであるか否かをどこで判断しているか
        # actuator >= 1

        # init sim
        BenchmarkBase.__init__(self, self.world)
        self.default_viewer.track_objects('robot', 'prey')

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size

        self.action_space = spaces.Box(low=0.6, high=1.6, shape=(num_actuators, ), dtype=np.float)
        self.observation_space = spaces.Box(low=-100, high=100.0, shape=(6 + num_robot_points, ), dtype=np.float)

    # def _get_obs(self):
    #     robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")
    #     robot_vel_final = self.object_vel_at_time(self.get_time(), "robot")
    #     prey_pos_final = self.object_pos_at_time(self.get_time(), "prey")
    #     prey_vel_final = self.object_vel_at_time(self.get_time(), "prey")
    #
    #     robot_com_pos = np.mean(robot_pos_final, axis=1)
    #     robot_com_vel = np.mean(robot_vel_final, axis=1)
    #     prey_com_pos = np.mean(prey_pos_final, axis=1)
    #     prey_com_vel = np.mean(prey_vel_final, axis=1)
    #
    #     # observation
    #     obs = np.array([
    #         robot_com_vel[0], robot_com_vel[1],
    #         prey_com_pos[0] - robot_com_pos[0],
    #         prey_com_pos[1] - robot_com_pos[1],
    #         prey_com_pos[0], prey_com_vel[1],
    #     ])
    #
    #     return obs

    def step(self, action: np.ndarray):

        # collect pre step information
        pos_1 = self.object_pos_at_time(self.get_time(), 'robot')

        # step
        done = super().step({'robot': action})

        self.prey_behave()

        # collect post step information
        pos_2 = self.object_pos_at_time(self.get_time(), 'robot')


        robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")
        robot_vel_final = self.object_vel_at_time(self.get_time(), "robot")
        prey_pos_final = self.object_pos_at_time(self.get_time(), "prey")
        prey_vel_final = self.object_vel_at_time(self.get_time(), "prey")

        # observation
        obs = np.concatenate((
            self.get_obs(robot_pos_final, robot_vel_final, prey_pos_final, prey_vel_final),
            self.get_relative_pos_obs('robot')
        ))

        prey_pred_diffs = self.get_prey_pred_diffs()
        reward = self.get_reward(prey_pred_diffs=prey_pred_diffs)
        # # compute reward
        # com_1 = np.mean(pos_1, 1)
        # com_2 = np.mean(pos_2, 1)
        # reward = (com_2[0] - com_1[0])

        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0
        #
        # # check goal met
        # if com_2[0] > 99*self.VOXEL_SIZE:
        #     done = True
        #     reward += 1.0

        # observation, reward, has simulation met termination conditions, debugging info
        return obs, reward, done, {
            'prey_pred_diffs': prey_pred_diffs,
        }

    def reset(self):
        super().reset()

        robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")
        robot_vel_final = self.object_vel_at_time(self.get_time(), "robot")
        prey_pos_final = self.object_pos_at_time(self.get_time(), "prey")
        prey_vel_final = self.object_vel_at_time(self.get_time(), "prey")

        # observation
        obs = np.concatenate((
            self.get_obs(robot_pos_final, robot_vel_final, prey_pos_final, prey_vel_final),
            self.get_relative_pos_obs('robot')
        ))

        return obs

    def get_obs(self, robot_pos_final, robot_vel_final, prey_pos_final, prey_vel_final):

        robot_com_pos = np.mean(robot_pos_final, axis=1)
        robot_com_vel = np.mean(robot_vel_final, axis=1)
        prey_com_pos = np.mean(prey_pos_final, axis=1)
        prey_com_vel = np.mean(prey_vel_final, axis=1)

        obs = np.array([
            robot_com_vel[0], robot_com_vel[1],
            prey_com_pos[0]-robot_com_pos[0],
            prey_com_pos[1]-robot_com_pos[1],
            prey_com_vel[0], prey_com_vel[1],
        ])

        return obs

    def get_reward(self, prey_pred_diffs):
        # 捕食器官のうち、最も捕食対象に近いブロックを取り上げる
        #  - ボクセルインデックスとボクセルの種類の紐付けが必要
        # 距離が1.8以下の時捕食しているとみなす
        # print()

        sqr_dist = np.min(np.sum((prey_pred_diffs * prey_pred_diffs), axis=1))
        return 0.01 / sqr_dist

    def get_prey_pred_diffs(self):
        robot_boxels_pos = self.sim.object_boxels_pos('robot')
        robot_boxels_type = self.sim.object_boxels_type('robot')
        prey_boxels_pos = self.sim.object_boxels_pos('prey')
        pred_boxels_pos = robot_boxels_pos[:, robot_boxels_type == VOXEL_TYPES['PRED']]
        return prey_boxels_pos.T - pred_boxels_pos

    def prey_behave(self):
        self.sim.translate_object(0.002, 0.001, 'prey')
