from gym import spaces

from evogym import *
from evogym.envs import BenchmarkBase
from evogym.envs.hunting.hunting_base import HuntingBase
from evogym.envs.hunting.state_handler import StateHandler

import numpy as np
import os


# DEFAULT_CONFIG = {
#     # task-specific config
#     "SENSING_RANGE": 1.0,
#     "ESCAPE_VELOCITY": 0.0015,
#     "HOPPING": 0.001,
#     "INIT_POS_X": 8,
#     "INIT_POS_Y": 1,

#     # task-common config
#     "REWARD_RANGE": 0.7,
#     "PROGRESSIVE_REWARD": 0.05,
# }


class HuntCreeper(HuntingBase):
    SENSING_RANGE = 1.0
    ESCAPE_VELOCITY = 0.0015
    HOPPING = 0.001
    # INIT_POS_X = 8
    # INIT_POS_Y = 1

    def __init__(self, body: np.ndarray, connections=None):
    # def __init__(self, body: np.ndarray, connections=None, config=None):
        # if config is not None:
        #     self.change_config(config)

        self.PREY_POS = [8, 1]

        # # make world
        # self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Walker-v0.json'))
        # self.world.add_from_array('robot', body, 1, 1, connections=connections)
        # self.world.add_from_array('prey', np.array([[7]]), self.INIT_POS_X, self.INIT_POS_Y)

        HuntingBase.__init__(self, body=body, connections=connections)
        # HuntingBase.__init__(self, world=self.world, config=config)

    # def change_config(self, config: dict):
    #     self.SENSING_RANGE = config["SENSING_RANGE"]
    #     self.ESCAPE_VELOCITY = config["ESCAPE_VELOCITY"]
    #     self.HOPPING = config["HOPPING"]
    #     self.INIT_POS_X = config["INIT_POS_X"]
    #     self.INIT_POS_Y = config["INIT_POS_Y"]

    #     print("hunt creeper change config")

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
