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

