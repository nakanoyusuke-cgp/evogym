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
        self.REWARD_RANGE = 0.7
        self.PROGRESSIVE_REWARD = 0.05
        self.ROBOT_POS = [1, 1]
        self.PREY_POS = [15, 1]
        self.PREY_STRUCTURE = [[7, 7], [7, 7]]

        # task specific
        self.SENSING_RANGE = 1.0
        self.ESCAPE_VELOCITY = 0.0021
        self.HOPPING = 0.001
