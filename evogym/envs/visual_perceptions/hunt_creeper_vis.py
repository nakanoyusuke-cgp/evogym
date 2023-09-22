from gym import spaces

from evogym import *
from evogym.envs import BenchmarkBase
from evogym.envs.hunting.hunting_base import HuntingBase
from evogym.envs.hunting.hunt_creeper import HuntCreeper
from evogym.envs.hunting.state_handler import StateHandler

import numpy as np
import os


DEFAULT_CONFIG = {
    # task-specific config
    "SENSING_RANGE": 1.0,
    "ESCAPE_VELOCITY": 0.0015,
    "HOPPING": 0.001,

    # task-common config
    "REWARD_RANGE": 0.7,
    "PROGRESSIVE_REWARD": 0.05,

    # vis1 config
    "VIS_LIMIT_LEN": 1.0
}

class HuntCreeperVis1(HuntCreeper):
    VIS_LIMIT_LEN = 1.0

    def __init__(self, body: np.ndarray, connections=None, config=None):
        if config is not None:
            self.change_config(config)

        HuntCreeper.__init__(self, body=body, connections=connections, config=config)

    def change_config(self, config: dict):
        super().change_config(config=config)
        self.VIS_LIMIT_LEN = config["VIS_LIMIT_LEN"]
        print("hunt creeper vis1, change config")

    def step(self, action: np.ndarray):
        obs, reward, done, info = super().step(action=action)

        obs = np.concatenate((
            np.mean(self.object_vel_at_time(self.get_time(), 'robot'), axis=1),
            self.get_prey_pred_diffs().reshape(-1),
            np.mean(self.object_vel_at_time(self.get_time(), 'prey'), axis=1),
            self.get_relative_pos_obs('robot'),
        ))

        return obs, reward, done, info

    def reset(self):
        obs = super().reset()

        obs = np.concatenate((
            np.mean(self.object_vel_at_time(self.get_time(), 'robot'), axis=1),
            self.get_prey_pred_diffs().reshape(-1),
            np.mean(self.object_vel_at_time(self.get_time(), 'prey'), axis=1),
            self.get_relative_pos_obs('robot'),
        ))

        return obs
