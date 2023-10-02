from gym import spaces

from evogym import *
from evogym.envs import BenchmarkBase
from evogym.simulator_cpp import VisualProcessor

import numpy as np
import os


DEFAULT_CONFIG = {
    # task-specific config
    "INIT_BOX_X": 2.0,
    "REWARD_DIST_BOT": 1.0,
    "REWARD_DIST_TOP": 1.2,

    # vis1 config
    "VIS_LIMIT_LEN": 2.2
}


class ObserverVis1(BenchmarkBase):
    INIT_BOX_X = 2.0
    REWARD_DIST_BOT = 1.0
    REWARD_DIST_TOP = 1.2
    VIS_LIMIT_LEN = 2.2

    def __init__(self, body: np.ndarray, connections=None, config=None):
        if config is not None:
            self.change_config(config)

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Walker-v0.json'))
        self.world.add_from_array('robot', body, 1, 1, connections=connections)
        self.world.add_from_array('box', np.array([[7, 7, 7, 7]]), self.INIT_BOX_X / self.sim.VOXEL_SIZE, 1)

        # init sim
        BenchmarkBase.__init__(self, world=self.world)

        # init vis_proc
        self.vis_proc = VisualProcessor(1, sim, self.VIS_LIMIT_LEN, -1)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size
        num_vis_voxels = len(body[body == VOXEL_TYPES["VIS"]])

        self.action_space = spaces.Box(low=0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        # vis, v^r, c^r
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(num_vis_voxels * 2 + 2 + num_robot_points,), dtype=np.float)

    def change_config(self, config: dict):
        self.INIT_BOX_X = config["INIT_BOX_X"]
        self.REWARD_DIST_BOT = config["REWARD_DIST_BOT"]
        self.REWARD_DIST_TOP = config["REWARD_DIST_TOP"]
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
        super().reset()

        vis_types = self.vis_proc.get_vis1_types()
        vis_sqr_dists = self.vis_proc.get_vis1_sqr_depths()

        obs = np.concatenate((
            
            self.get_vel_com_obs("robot"),
            self.get_relative_pos_obs('robot'),
        ))

        return obs
