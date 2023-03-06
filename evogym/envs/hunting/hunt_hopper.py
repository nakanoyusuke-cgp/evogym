from gym import spaces

from evogym import *
from evogym.envs import BenchmarkBase
from evogym.envs.hunting.hunting_base import HuntingBase
from evogym.envs.hunting.state_handler import StateHandler

import numpy as np
import os

DEFAULT_CONFIG = {
    # task-specific config
    "SENSING_RANGE": 1.0,
    "X_INIT_VELOCITY": 1.25,
    "Y_INIT_VELOCITY": 9.0,
    "INIT_WAIT_STEPS": 0.0,
    "JUMP_INTERVAL_STEPS": 60.0,
    "JUMP_ACCELERATION_STEPS": 10.0,
    "LANDING_CONTROL_STEPS": 5.0,
    "GROUND_THRESHOLD": 0.0075,

    # task-common config
    "REWARD_RANGE": 0.7,
    "PROGRESSIVE_REWARD": 0.05,
}


class HuntHopper(HuntingBase):
    SENSING_RANGE = 1.0
    X_INIT_VELOCITY = 1.25
    Y_INIT_VELOCITY = 9.0
    INIT_WAIT_STEPS = 0.0
    JUMP_INTERVAL_STEPS = 60.0
    JUMP_ACCELERATION_STEPS = 10.0
    LANDING_CONTROL_STEPS = 5.0
    GROUND_THRESHOLD = 0.0075

    def change_config(self, config: dict):
        self.SENSING_RANGE = config["SENSING_RANGE"]
        self.X_INIT_VELOCITY = config["X_INIT_VELOCITY"]
        self.Y_INIT_VELOCITY = config["Y_INIT_VELOCITY"]
        self.INIT_WAIT_STEPS = config["INIT_WAIT_STEPS"]
        self.JUMP_INTERVAL_STEPS = config["JUMP_INTERVAL_STEPS"]
        self.JUMP_ACCELERATION_STEPS = config["JUMP_ACCELERATION_STEPS"]
        self.LANDING_CONTROL_STEPS = config["LANDING_CONTROL_STEPS"]
        self.GROUND_THRESHOLD = config["GROUND_THRESHOLD"]

    def __init__(self, body: np.ndarray, connections=None, config=None):
        if config is not None:
            self.change_config(config)

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Walker-v0.json'))
        self.world.add_from_array('robot', body, 1, 1, connections=connections)
        self.world.add_from_array('prey', np.array([[7]]), 8, 1)
        # self.world.add_from_array('prey', np.array([[7]]), 3, 10)

        HuntingBase.__init__(self, world=self.world, config=config)

        self.state_handler = StateHandler()

        # ### ステートマシンの設定 ###
        def initial_callback(s):
            # エピソード開始後の待機時間
            if s > self.INIT_WAIT_STEPS:
                return "search"

        def search_callback(s):
            # ロボットを探知
            if np.sum(self.get_robot_prey_diff() ** 2) < (self.SENSING_RANGE ** 2) and self.is_on_grounding():
                return "jumping"

        def jumping_callback(s):
            # ジャンプ
            if s == 0:
                self.sim.add_object_velocity(self.X_INIT_VELOCITY, self.Y_INIT_VELOCITY, 'prey')
            else:
                if s > self.JUMP_ACCELERATION_STEPS and self.is_on_grounding():  # 着地
                    # 着地硬直に移行
                    return "landing"

        def landing_callback(s):
            # 着地と着地硬直
            if s < self.JUMP_INTERVAL_STEPS:
                if s < self.LANDING_CONTROL_STEPS:
                    self.sim.mul_object_velocity(0.1, 'prey')
            else:
                return "search"

        self.state_handler.add_state(name="initial", callback=initial_callback)
        self.state_handler.add_state(name="search", callback=search_callback)
        self.state_handler.add_state(name="jumping", callback=jumping_callback)
        self.state_handler.add_state(name="landing", callback=landing_callback)
        self.state_handler.set_state(name="initial")

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

    def get_robot_prey_diff(self):
        robot_com_pos = np.mean(self.object_pos_at_time(self.get_time(), 'robot'), axis=1)
        prey_com_pos = np.mean(self.object_pos_at_time(self.get_time(), 'prey'), axis=1)
        return prey_com_pos - robot_com_pos

    def is_on_grounding(self):
        is_on_floor = np.any(self.object_pos_at_time(self.get_time(), 'prey')[1] < 0.1 + self.GROUND_THRESHOLD)
        is_on_robot = self.sim.ground_on_robot('prey', 'robot') < self.GROUND_THRESHOLD
        return is_on_robot or is_on_floor
