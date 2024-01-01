from gym import spaces

from evogym import *
from evogym.envs import BenchmarkBase
from evogym.envs.hunting.hunt_creeper import HuntCreeper
from evogym.envs.hunting.hunting_base import HuntingBase
from evogym.envs.hunting.state_handler import StateHandler

import numpy as np
import os


class HuntCreeperBaseline(HuntCreeper):
    def __init__(self, body: np.ndarray, connections=None):
        super().__init__(body=body, connections=connections)

    def change_params(self):
        # common
        self.REWARD_RANGE = 0.5
        self.PROGRESSIVE_REWARD = 0.05
        self.ROBOT_POS = [1, 1]
        self.PREY_POS = [15, 1]
        self.PREY_STRUCTURE = [[7, 7], [7, 7]]

        # task specific
        self.SENSING_RANGE = 1.0
        self.ESCAPE_VELOCITY = 0.0021
        self.HOPPING = 0.001

    # 1.5**2 + 0.5**2 = 2.5
    def get_reward(self, prey_pred_diffs, sqr_dist_prev):
        reward = 0
        sqr_dist = np.min(np.sum((prey_pred_diffs * prey_pred_diffs), axis=0))
        if sqr_dist < (self.REWARD_RANGE ** 2):
            # reward += 0.01 / sqr_dist
            reward += 0.025 / sqr_dist
        if sqr_dist < sqr_dist_prev:
            reward += self.PROGRESSIVE_REWARD

        return np.clip(reward, 0., 1.), sqr_dist