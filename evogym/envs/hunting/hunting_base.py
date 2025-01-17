from gym import spaces

from evogym import *
from evogym.envs import BenchmarkBase
from evogym.envs.hunting.state_handler import StateHandler

import numpy as np
import os


class HuntingBase(BenchmarkBase):
    REWARD_RANGE = 0.7
    PROGRESSIVE_REWARD = 0.05
    ROBOT_POS = [1, 1]
    PREY_POS = [1, 1]
    PREY_STRUCTURE = [[7]]

    # def change_config(self, config: dict):
    #     self.REWARD_RANGE = config["REWARD_RANGE"]
    #     self.PROGRESSIVE_REWARD = config["PROGRESSIVE_REWARD"]

    def __init__(self, body: np.ndarray, connections=None):
    # def __init__(self, world, config):
        # if config is not None:
        #     self.change_config(config=config)

        self.change_params()

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Walker-v0.json'))
        self.world.add_from_array('robot', body, self.ROBOT_POS[0], self.ROBOT_POS[1], connections=connections)
        self.world.add_from_array('prey', np.array(self.PREY_STRUCTURE), self.PREY_POS[0], self.PREY_POS[1])
        # self.world.add_from_array('robot', body, 1, 1, connections=connections)
        # self.world.add_from_array('prey', np.array([[7]]), self.INIT_POS_X, self.INIT_POS_Y)

        # init sim
        self._sim = EvoSim(self.world)
        self._default_viewer = self.generate_viewer()
        self.default_viewer.track_objects('robot', 'prey')
        # BenchmarkBase.__init__(self, self.world)
        # self.default_viewer.track_objects('robot', 'prey')
        # # self.default_viewer.track_objects(('robot',), ('prey',))

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size
        num_pred_voxels = np.sum((self.object_voxels_type('robot') == VOXEL_TYPES['PRED']), dtype=np.int64)

        self.action_space = spaces.Box(low=0.6, high=1.6, shape=(num_actuators, ), dtype=np.float)
        # robot vel: 2, diffs: 2 * num_pred_voxels, prey vel: 2, relative pos: num_robot_points
        self.observation_space = spaces.Box(
            low=-100, high=100.0, shape=(2 + 2 * num_pred_voxels + 2 + num_robot_points, ), dtype=np.float)

        self._sqr_dist_prev = None

    def generate_viewer(self):
        return EvoViewer(self._sim)

    def change_params(self):
        pass

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
        # prey_boxels_pos = self.object_voxels_pos('prey')

        prey_boxels_pos = self.get_pos_com_obs('prey')[:, np.newaxis]
        pred_boxels_pos = robot_boxels_pos[:, robot_boxels_type == VOXEL_TYPES['PRED']]
        return prey_boxels_pos - pred_boxels_pos


