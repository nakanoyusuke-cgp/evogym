from gym import spaces

from evogym import *
from evogym.envs import BenchmarkBase
from evogym.envs.hunting.hunting_base import HuntingBase
from evogym.envs.hunting.state_handler import StateHandler

import numpy as np
import os
import random

# X軸上の距離を頼りに飛び回る
# 一定の高さを保つ
# y>h以下ならはねる
# states:
# - no op
# - 初速を得る（ロボットとの距離に応じてx軸上の初速を確定する, ランダム性を持つ, 確定性が必要）


# DEFAULT_CONFIG = {
#     # task-specific config
#     "SENSING_X_RANGE": 1.0,
#     "X_ACCELERATION": 0.35,
#     "Y_ACCELERATION": 12.0,
#     "X_ACCELERATION_DISPERSION": 0.1,
#     "Y_ACCELERATION_DISPERSION": 0.3,
#     "Y_FLAP_HEIGHT": 0.6,
#     "Y_FLAP_HEIGHT_DISPERSION": 0.05,
#     "FLAP_MIN_INTERVAL": 30.0,
#     "VELOCITY_SUPPRESSION_MULTIPLIER": 0.5,
#     "X_RANDOM_FLAP_RANGE": 0.3,
#     "ACCELERATION_STEP": 10,

#     # task-common config
#     "REWARD_RANGE": 0.7,
#     "PROGRESSIVE_REWARD": 0.05,
# }


class HuntFlyer(HuntingBase):
    SENSING_X_RANGE = 1.0
    X_ACCELERATION = 0.35
    Y_ACCELERATION = 12.0
    X_ACCELERATION_DISPERSION = 0.1
    Y_ACCELERATION_DISPERSION = 0.3
    Y_FLAP_HEIGHT = 0.6
    Y_FLAP_HEIGHT_DISPERSION = 0.05
    FLAP_MIN_INTERVAL = 30.0
    VELOCITY_SUPPRESSION_MULTIPLIER = 0.5
    X_RANDOM_FLAP_RANGE = 0.3
    ACCELERATION_STEP = 10

    # def change_config(self, config: dict):
    #     self.SENSING_X_RANGE = config["SENSING_RANGE"]
    #     self.X_ACCELERATION = config["X_ACCELERATION"]
    #     self.Y_ACCELERATION = config["Y_ACCELERATION"]
    #     self.X_ACCELERATION_DISPERSION = config["X_ACCELERATION_DISPERSION"]
    #     self.Y_ACCELERATION_DISPERSION = config["Y_ACCELERATION_DISPERSION"]
    #     self.Y_FLAP_HEIGHT = config["Y_FLAP_HEIGHT"]
    #     self.Y_FLAP_HEIGHT_DISPERSION = config["Y_FLAP_HEIGHT_DISPERSION"]
    #     self.FLAP_MIN_INTERVAL = config["FLAP_MIN_INTERVAL"]
    #     self.VELOCITY_SUPPRESSION_MULTIPLIER = config["VELOCITY_SUPPRESSION_MULTIPLIER"]
    #     self.X_RANDOM_FLAP_RANGE = config["X_RANDOM_FLAP_RANGE"]
    #     self.ACCELERATION_STEP = config["ACCELERATION_STEP"]

    def __init__(self, body: np.ndarray, connections=None):
    # def __init__(self, body: np.ndarray, connections=None, config=None):
        # if config is not None:
        #     self.change_config(config)

        # # make world
        # self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Walker-v0.json'))
        # self.world.add_from_array('robot', body, 1, 1, connections=connections)
        # self.world.add_from_array('prey', np.array([[7]]), 8, 7)
        # # self.world.add_from_array('prey', np.array([[7]]), 3, 10)

        HuntingBase.__init__(self, body=body, connections=connections)
        # HuntingBase.__init__(self, world=self.world)
        # HuntingBase.__init__(self, world=self.world, config=config)

        self.flap_x_line = 0.0
        self.next_flap_y_height = 0.0
        self.acc_x = 0.0
        self.acc_y = 0.0

        self.rng = random.Random()
        self._seed = 0
        self.rng.seed(self._seed)

        self.state_handler = StateHandler()

        # ### ステートマシンの設定 ###
        def free_fall_callback(s):
            # 羽ばたきのインターバル
            if s == 0:
                self.next_flap_y_height = self.Y_FLAP_HEIGHT + \
                                          self.Y_FLAP_HEIGHT_DISPERSION * (self.rng.random() * 2 - 1.0)

            # if s >= self.FLAP_MIN_INTERVAL - self.ACCELERATION_STEP:
            x, y = self.get_prey_com_pos()

            # flap
            if y < self.next_flap_y_height:
                self.sim.mul_object_velocity(self.VELOCITY_SUPPRESSION_MULTIPLIER, 'prey')
                robot_x, _ = self.get_robot_com_pos()
                diff_x = x - robot_x
                if diff_x ** 2 < self.SENSING_X_RANGE ** 2:
                    # update flap_x_line
                    if diff_x >= 0:
                        self.flap_x_line = max(robot_x + self.SENSING_X_RANGE, self.flap_x_line)
                    else:
                        self.flap_x_line = min(robot_x - self.SENSING_X_RANGE, self.flap_x_line)

                    return 'escape_flap'

                else:
                    return 'random_flap'

        def escape_flap_callback(s):
            if s == 0:
                acc_x_abs = self.X_ACCELERATION + self.X_ACCELERATION_DISPERSION * (self.rng.random() * 2 - 1.0)
                acc_y_abs = self.Y_ACCELERATION + self.Y_ACCELERATION_DISPERSION * (self.rng.random() * 2 - 1.0)
                x, _ = self.get_prey_com_pos()
                if x <= self.flap_x_line:
                    self.acc_x = acc_x_abs / self.ACCELERATION_STEP

                else:
                    self.acc_x = -acc_x_abs / self.ACCELERATION_STEP
                self.acc_y = acc_y_abs / self.ACCELERATION_STEP

            if s < self.ACCELERATION_STEP:
                self.sim.add_object_velocity(self.acc_x, self.acc_y, 'prey')

            if s > self.FLAP_MIN_INTERVAL:
                return 'free_fall'

        def random_flap_callback(s):
            if s == 0:
                acc_x_abs = self.X_ACCELERATION + self.X_ACCELERATION_DISPERSION * (self.rng.random() * 2 - 1.0)
                acc_y_abs = self.Y_ACCELERATION + self.Y_ACCELERATION_DISPERSION * (self.rng.random() * 2 - 1.0)
                x, _ = self.get_prey_com_pos()
                if x < self.flap_x_line - self.X_RANDOM_FLAP_RANGE:
                    # 範囲より左
                    self.acc_x = acc_x_abs / self.ACCELERATION_STEP

                elif x > self.flap_x_line + self.X_RANDOM_FLAP_RANGE:
                    # 範囲より右
                    self.acc_x = -acc_x_abs / self.ACCELERATION_STEP

                else:
                    self.acc_x = (self.rng.randint(0, 1) * 2 - 1) * acc_x_abs / self.ACCELERATION_STEP

                self.acc_y = acc_y_abs / self.ACCELERATION_STEP

            if s < self.ACCELERATION_STEP:
                self.sim.add_object_velocity(self.acc_x, self.acc_y, 'prey')

            if s > self.FLAP_MIN_INTERVAL:
                return 'free_fall'

        self.state_handler.add_state(name="free_fall", callback=free_fall_callback)
        self.state_handler.add_state(name="escape_flap", callback=escape_flap_callback)
        self.state_handler.add_state(name="random_flap", callback=random_flap_callback)
        self.state_handler.set_state(name="free_fall")

    def step(self, action: np.ndarray):
        # step
        done = super().step({'robot': action})

        # prey behave
        self.state_handler.step()

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
        if prey_com_pos[0] > 99 * self.VOXEL_SIZE:
            done = True
            reward = -1
            done_info = 'The simulation was terminated because the prey completely escaped from the predatory robot.'

        # observation, reward, has simulation met termination conditions, debugging info
        return obs, reward, done, {
            'prey_pred_diffs': prey_pred_diffs,
            'done_info': done_info,
            'state': self.state_handler.state,
            'state_time': self.state_handler.state_step,
        }

    def reset(self):
        s = super().reset()

        self.flap_x_line = self.get_prey_com_pos()[0]
        self.rng.seed(self._seed)
        return s

    def seed(self, seed=None):
        if seed is None:
            self._seed = 0
        else:
            self._seed = seed

    def get_prey_com_pos(self):
        return np.mean(self.object_pos_at_time(self.get_time(), 'prey'), axis=1)

    def get_robot_com_pos(self):
        return np.mean(self.object_pos_at_time(self.get_time(), 'robot'), axis=1)

    def get_robot_prey_diff(self):
        robot_com_pos = np.mean(self.object_pos_at_time(self.get_time(), 'robot'), axis=1)
        prey_com_pos = np.mean(self.object_pos_at_time(self.get_time(), 'prey'), axis=1)
        return prey_com_pos - robot_com_pos
