from gym import spaces

from evogym import *
from evogym.envs import BenchmarkBase
from evogym.envs.hunting.hunting_base import HuntingBase
from evogym.envs.hunting.hunt_hopper import HuntHopper
from evogym.envs.hunting.state_handler import StateHandler

import numpy as np
import os


class HuntHopperBaseline(HuntHopper):
    def __init__(self, body: np.ndarray, connections=None):
        super().__init__(body, connections)

    def change_params(self):
        # common
        self.REWARD_RANGE = 0.7
        self.PROGRESSIVE_REWARD = 0.05
        self.ROBOT_POS = [1, 1]
        self.PREY_POS = [15, 1]
        self.PREY_STRUCTURE = [[7, 7], [7, 7]]

        # task specific
        self.SENSING_RANGE = 1.0
        self.X_INIT_VELOCITY = 1.25
        self.Y_INIT_VELOCITY = 9.0
        self.INIT_WAIT_STEPS = 0.0
        self.JUMP_INTERVAL_STEPS = 60.0
        self.JUMP_ACCELERATION_STEPS = 10.0
        self.LANDING_CONTROL_STEPS = 5.0
        self.GROUND_THRESHOLD = 0.0075
        