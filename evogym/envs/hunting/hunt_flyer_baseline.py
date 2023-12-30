from gym import spaces

from evogym import *
from evogym.envs import BenchmarkBase
from evogym.envs.hunting.hunting_base import HuntingBase
from evogym.envs.hunting.hunt_flyer import HuntFlyer
from evogym.envs.hunting.state_handler import StateHandler

import numpy as np
import os
import random


class HuntFlyerBaseline(HuntFlyer):
    def __init__(self, body: np.ndarray, connections=None):
        super().__init__(body, connections)

    def change_params(self):
        # common
        self.REWARD_RANGE = 0.7
        self.PROGRESSIVE_REWARD = 0.05
        self.ROBOT_POS = [1, 1]
        self.PREY_POS = [15, 7]
        self.PREY_STRUCTURE = [[7, 7], [7, 7]]

        # # task specific
        # SENSING_X_RANGE = 1.0
        # X_ACCELERATION = 0.35
        # Y_ACCELERATION = 12.0
        # X_ACCELERATION_DISPERSION = 0.1
        # Y_ACCELERATION_DISPERSION = 0.3
        # Y_FLAP_HEIGHT = 0.6
        # Y_FLAP_HEIGHT_DISPERSION = 0.05
        # FLAP_MIN_INTERVAL = 30.0
        # VELOCITY_SUPPRESSION_MULTIPLIER = 0.5
        # X_RANDOM_FLAP_RANGE = 0.3
        # ACCELERATION_STEP = 10
