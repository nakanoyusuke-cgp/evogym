from gym import spaces

from evogym import *
from evogym.envs import BenchmarkBase
from evogym.envs.hunting import HuntCreeper
from evogym.envs.hunting.hunting_base import HuntingBase
from evogym.envs.hunting.state_handler import StateHandler

import numpy as np
import os


class HuntMultiCreepers(HuntCreeper):
    RELOCATE_PREY_DISTANCE = 7
    MAX_REWARD_PER_PREY = 100

    def __init__(self, body: np.ndarray, connections=None):
        super().__init__(body, connections)
        self.relocate_count = 0
        self.accumulated_rewards = 0.

    def change_params(self):
        super().change_params()
        self.ROBOT_POS = [35, 1]
        self.PREY_POS = [44, 1]
        self.REWARD_RANGE = 0.12

    def reset(self):
        self.relocate_count = 0
        self.accumulated_rewards = 0.
        return super().reset()

    def get_reward(self, prey_pred_diffs, sqr_dist_prev):
        reward = 0.
        sqr_dist = np.min(np.sum((prey_pred_diffs * prey_pred_diffs), axis=0))
        if sqr_dist < (self.REWARD_RANGE ** 2):
            reward += 1.
        if sqr_dist < sqr_dist_prev:
            reward += self.PROGRESSIVE_REWARD

        return np.clip(reward, 0., 1.), sqr_dist

    def step(self, action: np.ndarray):
        # step
        obs, reward, done, info = super().step(action)

        # the prey completely escaped from predatory robot (x < 0)
        prey_com_pos = np.mean(self.object_pos_at_time(self.get_time(), 'prey'), axis=1)
        if prey_com_pos[0] < 0:
            done = True
            reward = -1
            info["done_info"] = ('The simulation was terminated because the prey completely escaped from the predatory '
                                 'robot.')

        # relocate
        info['prey_was_relocated'] = False
        if not done:
            self.accumulated_rewards += reward
            if self.accumulated_rewards >= self.MAX_REWARD_PER_PREY:
                prey_pred_diffs = self.relocate_prey()

                obs = np.concatenate((
                    np.mean(self.object_vel_at_time(self.get_time(), 'robot'), axis=1),
                    self.get_prey_pred_diffs().reshape(-1),
                    np.mean(self.object_vel_at_time(self.get_time(), 'prey'), axis=1),
                    self.get_relative_pos_obs('robot'),
                ))

                info['prey_was_relocated'] = True
                info['prey_pred_diffs'] = prey_pred_diffs

        info["accumulated_reward"] = self.accumulated_rewards

        return obs, reward, done, info

    def relocate_prey(self):
        if self.relocate_count % 2 == 0:
            prey_x = self.get_pos_com_obs('robot')[0] - self.RELOCATE_PREY_DISTANCE * self.VOXEL_SIZE
        else:
            prey_x = self.get_pos_com_obs('robot')[0] + self.RELOCATE_PREY_DISTANCE * self.VOXEL_SIZE

        self.sim.move_object(prey_x, 0.19, 'prey')
        self.sim.set_object_velocity(0., 0., 'prey')
        prey_pred_diffs = self.get_prey_pred_diffs()
        self._sqr_dist_prev = np.min(np.sum((prey_pred_diffs * prey_pred_diffs), axis=0))

        self.relocate_count += 1
        self.accumulated_rewards = 0

        return prey_pred_diffs


class HuntMultiCreepersWithoutOuterInfo(HuntMultiCreepers):
    def __init__(self, body: np.ndarray, connections=None):
        super().__init__(body, connections)

        # set action space and observation space
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size
        # robot vel: 2, relative pos: num_robot_points
        self.observation_space = spaces.Box(
            low=-100, high=100.0, shape=(2 + num_robot_points, ), dtype=np.float)

    def step(self, action: np.ndarray):
        obs, reward, done, info = super().step(action)
        obs = np.concatenate((
            np.mean(self.object_vel_at_time(self.get_time(), 'robot'), axis=1),
            self.get_relative_pos_obs('robot'),
        ))
        return obs, reward, done, info

    def reset(self):
        super().reset()

        obs = np.concatenate((
            np.mean(self.object_vel_at_time(self.get_time(), 'robot'), axis=1),
            self.get_relative_pos_obs('robot'),
        ))

        return obs
