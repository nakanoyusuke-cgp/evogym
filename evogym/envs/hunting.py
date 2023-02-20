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


class HuntingBase(BenchmarkBase):
    REWARD_RANGE = 0.7
    PROGRESSIVE_REWARD = 0.05

    def __init__(self, world):
        # init sim
        BenchmarkBase.__init__(self, world)
        self.default_viewer.track_objects('robot', 'prey')
        # self.default_viewer.track_objects(('robot',), ('prey',))

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size
        num_pred_voxels = np.sum((self.object_voxels_type('robot') == VOXEL_TYPES['PRED']), dtype=np.int64)

        self.action_space = spaces.Box(low=0.6, high=1.6, shape=(num_actuators, ), dtype=np.float)
        # robot vel: 2, diffs: 2 * num_pred_voxels, prey vel: 2, relative pos: num_robot_points
        self.observation_space = spaces.Box(
            low=-100, high=100.0, shape=(2 + 2 * num_pred_voxels + 2 + num_robot_points, ), dtype=np.float)

        self._sqr_dist_prev = None

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

        prey_pred_diffs = self.get_prey_pred_diffs()
        self._sqr_dist_prev = np.min(np.sum((prey_pred_diffs * prey_pred_diffs), axis=0))

        return obs

    def get_reward(self, prey_pred_diffs, sqr_dist_prev):
        reward = 0
        sqr_dist = np.min(np.sum((prey_pred_diffs * prey_pred_diffs), axis=0))
        if sqr_dist < (self.REWARD_RANGE ** 2):
            reward += 0.01 / sqr_dist
        if sqr_dist < sqr_dist_prev:
            reward += self.PROGRESSIVE_REWARD

        return np.clip(reward, 0., 1.), sqr_dist

    def get_prey_pred_diffs(self):
        robot_boxels_pos = self.object_voxels_pos('robot')
        robot_boxels_type = self.object_voxels_type('robot')
        prey_boxels_pos = self.object_voxels_pos('prey')
        pred_boxels_pos = robot_boxels_pos[:, robot_boxels_type == VOXEL_TYPES['PRED']]
        return prey_boxels_pos - pred_boxels_pos


class HuntCreeper(HuntingBase):
    SENSING_RANGE = 1.0
    ESCAPE_VELOCITY = 0.0015
    HOPPING = 0.001

    def __init__(self, body: np.ndarray, connections=None):
        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Walker-v0.json'))
        self.world.add_from_array('robot', body, 1, 1, connections=connections)
        self.world.add_from_array('prey', np.array([[7]]), 8, 1)

        HuntingBase.__init__(self, world=self.world)

    def step(self, action: np.ndarray):
        # step
        done = super().step({'robot': action})

        # prey behave
        self.prey_behave()

        # get prey pred diffs
        prey_pred_diffs = self.get_prey_pred_diffs()

        # compute reward
        reward, sqr_dist = self.get_reward(prey_pred_diffs=prey_pred_diffs, sqr_dist_prev=self._sqr_dist_prev)
        self._sqr_dist_prev = sqr_dist

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

    def prey_behave(self):
        robot_com_pos = np.mean(self.object_pos_at_time(self.get_time(), 'robot'), axis=1)
        prey_com_pos = np.mean(self.object_pos_at_time(self.get_time(), 'prey'), axis=1)
        diff = prey_com_pos - robot_com_pos
        if np.sum(diff ** 2) < (self.SENSING_RANGE ** 2):
            if diff[0] >= 0.0:
                self.sim.translate_object(self.ESCAPE_VELOCITY, self.HOPPING, 'prey')
            else:
                self.sim.translate_object(-1 * self.ESCAPE_VELOCITY, self.HOPPING, 'prey')


class HuntHopper(HuntingBase):
    SENSING_RANGE = 20.0
    X_INIT_VELOCITY = 4.0
    Y_INIT_VELOCITY = 10.0
    JUMP_INTERVAL_STEPS = 60.0
    GROUND_THRESHOLD = 0.0075
    AIR_CONTROL_HEIGHT = 0.2
    STATES = {
        "out_of_sensing_range": 0,
        'jumping': 1,
        'after_landing': 2,
    }

    STATES_INV = {v: k for k, v in STATES.items()}

    def __init__(self, body: np.ndarray, connections=None):
        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Walker-v0.json'))
        self.world.add_from_array('robot', body, 1, 1, connections=connections)
        self.world.add_from_array('prey', np.array([[7]]), 8, 1)
        # self.world.add_from_array('prey', np.array([[7]]), 3, 10)
        self.state = self.STATES['out_of_sensing_range']
        self.state_time = 0

        HuntingBase.__init__(self, world=self.world)

    def step(self, action: np.ndarray):
        # step
        done = super().step({'robot': action})

        # prey behave
        self.prey_behave()

        # get prey pred diffs
        prey_pred_diffs = self.get_prey_pred_diffs()

        # compute reward
        reward, sqr_dist = self.get_reward(prey_pred_diffs=prey_pred_diffs, sqr_dist_prev=self._sqr_dist_prev)
        self._sqr_dist_prev = sqr_dist

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
            'state': self.STATES_INV[self.state],
            'state_time': self.state_time,
        }

    def get_robot_prey_diff(self):
        robot_com_pos = np.mean(self.object_pos_at_time(self.get_time(), 'robot'), axis=1)
        prey_com_pos = np.mean(self.object_pos_at_time(self.get_time(), 'prey'), axis=1)
        return prey_com_pos - robot_com_pos

    def is_on_grounding(self):
        is_on_floor = np.any(self.object_pos_at_time(self.get_time(), 'prey')[1] < 0.1 + self.GROUND_THRESHOLD)
        is_on_robot = self.sim.ground_on_robot('prey', 'robot') < self.GROUND_THRESHOLD
        # print(self.object_pos_at_time(self.get_time(), 'prey')[1], self.sim.ground_on_robot('prey', 'robot'))
        return is_on_robot or is_on_floor

    def in_range_air_control(self):
        prey_pos_y = self.object_pos_at_time(self.get_time(), 'prey')[1]
        ground_on_robot = self.sim.ground_on_robot('prey', 'robot')
        is_not_on_floor = np.any((0.1 + self.GROUND_THRESHOLD < prey_pos_y))
        is_not_on_robot = self.GROUND_THRESHOLD < ground_on_robot
        under_air_control_height_floor = np.mean(prey_pos_y) < 0.1 + self.AIR_CONTROL_HEIGHT
        under_air_control_height_robot = 
        return (is_not_on_robot and under_air_control_height_robot) or (is_not_on_floor and under_air_control_height_floor)

    def under_air_control_height(self):
        prey_com_pos = np.mean(self.object_pos_at_time(self.get_time(), 'prey'), axis=1)
        return prey_com_pos[1] < self.AIR_CONTROL_HEIGHT

    def prey_behave(self):
        # ### state of 'out_of_sensing_range' ###
        if self.state == self.STATES['out_of_sensing_range']:
            # センサ範囲外
            if np.sum(self.get_robot_prey_diff() ** 2) < (self.SENSING_RANGE ** 2) and self.is_on_grounding():
                self.state = self.STATES['jumping']
                self.state_time = 0
            else:
                self.state_time += 1

        # ### state of 'jumping' ###
        elif self.state == self.STATES['jumping']:
            # ジャンプ中の状態
            if self.state_time == 0:
                self.sim.add_object_velocity(self.X_INIT_VELOCITY, self.Y_INIT_VELOCITY, 'prey')
                self.state_time += 1
            else:
                if self.is_on_grounding():  # 着地
                    # 着地硬直に移行
                    self.state = self.STATES['after_landing']
                    # self.sim.add_object_velocity(0.0, -self.get_vel_com_obs('prey').mean(), 'prey')
                    self.sim.set_object_velocity(0.0, 0.0, 'prey')
                    self.state_time = 0
                else:
                    self.state_time += 1

        # ### state of after_landing ###
        elif self.state == self.STATES['after_landing']:
            # ジャンプ後の硬直状態
            if self.state_time >= self.JUMP_INTERVAL_STEPS:
                if np.sum(self.get_robot_prey_diff() ** 2) < (self.SENSING_RANGE ** 2):
                    # ジャンプ中状態への移行
                    self.state = self.STATES['jumping']
                    self.state_time = 0
                else:
                    # センサ範囲外状態へ移行
                    self.state = self.STATES['out_of_sensing_range']
                    self.state_time = 0
            else:
                if (not self.is_on_grounding()) and self.under_air_control_height():
                    self.sim.set_object_velocity(0.0, 0.0, 'prey')
                self.state_time += 1

        # ### error of illegal state ###
        else:
            print('The prey_hopper has illegal state number:', self.STATES)
            exit(1)
