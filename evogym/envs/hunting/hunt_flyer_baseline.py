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
        # super().__init__(body, connections)
        HuntingBase.__init__(self, body=body, connections=connections)

        self.INIT_WAIT = 20
        self.SENSING_RANGE = 1.0
        self.HOP_ACC_STEP = 10
        self.X_HOP_ACC = 1.
        self.Y_HOP_ACC = 13.
        self.FLAP_ACC_STEP = 10
        self.X_FLAP_ACC = 1.
        self.Y_FLAP_ACC = 13.
        self.Y_FLAP_HEIGHT = 0.6
        self.FLAP_MIN_INTERVAL = 30
        self.VELOCITY_SUPPRESSION_MULTIPLIER = 0.2
        self.GROUND_THRESHOLD = 0.0075

        self.state_handler = StateHandler()
        
        def init_free_fall_callback(s):
            prey_pos = self.get_prey_com_pos()
            if s >= self.INIT_WAIT:
                if self.is_on_grounding():
                    if np.sum(self.get_robot_prey_diff() ** 2) < (self.SENSING_RANGE):
                        return 'escape_hop'
                    else:
                        return 'neutral_hop'
                elif prey_pos[1] < self.Y_FLAP_HEIGHT:
                    if np.sum(self.get_robot_prey_diff() ** 2) < (self.SENSING_RANGE):
                        return 'escape_flap'
                    else:
                        return 'neutral_flap'
                else:
                    pass

        def free_fall_callback(s):
            prey_pos = self.get_prey_com_pos()
            if s >= self.FLAP_MIN_INTERVAL:
                if self.is_on_grounding():
                    if np.sum(self.get_robot_prey_diff() ** 2) < (self.SENSING_RANGE):
                        return 'escape_hop'
                    else:
                        return 'neutral_hop'
                elif prey_pos[1] < self.Y_FLAP_HEIGHT:
                    if np.sum(self.get_robot_prey_diff() ** 2) < (self.SENSING_RANGE):
                        return 'escape_flap'
                    else:
                        return 'neutral_flap'
                else:
                    pass

        def neutral_flap_callback(s):
            if s == 0:
                self.sim.mul_object_velocity(self.VELOCITY_SUPPRESSION_MULTIPLIER, 'prey')
            if s < self.FLAP_ACC_STEP:
                self.sim.add_object_velocity(0, self.Y_FLAP_ACC / self.FLAP_ACC_STEP, 'prey')
            else:
                return 'free_fall'

        def escape_flap_callback(s):
            if s == 0:
                self.sim.mul_object_velocity(self.VELOCITY_SUPPRESSION_MULTIPLIER, 'prey')
            if s < self.FLAP_ACC_STEP:
                if self.get_robot_prey_diff()[0] >= 0:
                    self.sim.add_object_velocity(self.X_FLAP_ACC / self.FLAP_ACC_STEP, self.Y_FLAP_ACC / self.FLAP_ACC_STEP, 'prey')
                else:
                    self.sim.add_object_velocity(-self.X_FLAP_ACC / self.FLAP_ACC_STEP, self.Y_FLAP_ACC / self.FLAP_ACC_STEP, 'prey')
            else:
                return 'free_fall'

        def neutral_hop_callback(s):
            if s == 0:
                self.sim.mul_object_velocity(self.VELOCITY_SUPPRESSION_MULTIPLIER, 'prey')
            if s < self.HOP_ACC_STEP:
                self.sim.add_object_velocity(0, self.Y_HOP_ACC / self.HOP_ACC_STEP, 'prey')
            else:
                return 'free_fall'

        def escape_hop_callback(s):
            if s == 0:
                self.sim.mul_object_velocity(self.VELOCITY_SUPPRESSION_MULTIPLIER, 'prey')
            if s < self.HOP_ACC_STEP:
                if self.get_robot_prey_diff()[0] > 0:
                    self.sim.add_object_velocity(
                        self.X_HOP_ACC / self.HOP_ACC_STEP,
                        self.Y_HOP_ACC / self.HOP_ACC_STEP, 'prey')
                else:
                    self.sim.add_object_velocity(
                        -self.X_HOP_ACC / self.HOP_ACC_STEP, 
                        self.Y_HOP_ACC / self.HOP_ACC_STEP, 'prey')
            else:
                return 'free_fall'

        self.state_handler.add_state(name="init_free_fall", callback=init_free_fall_callback)
        self.state_handler.add_state(name="free_fall", callback=free_fall_callback)
        self.state_handler.add_state(name="neutral_flap", callback=neutral_flap_callback)
        self.state_handler.add_state(name="escape_flap", callback=escape_flap_callback)
        self.state_handler.add_state(name="neutral_hop", callback=neutral_hop_callback)
        self.state_handler.add_state(name="escape_hop", callback=escape_hop_callback)

        self.state_handler.set_state(name="init_free_fall")


    def change_params(self):
        # common
        self.REWARD_RANGE = 0.5
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

    # 1.5**2 + 0.5**2 = 2.5
    def get_reward(self, prey_pred_diffs, sqr_dist_prev):
        reward = 0
        sqr_dist = np.min(np.sum((prey_pred_diffs * prey_pred_diffs), axis=0))
        if sqr_dist < (self.REWARD_RANGE ** 2):
            # reward += 0.01 / sqr_dist
            reward += 0.025 / sqr_dist
        if sqr_dist < sqr_dist_prev:
            reward += self.PROGRESSIVE_REWARD

        return np.clip(reward, 0., 1.), sqr_dist
    
    def is_on_grounding(self):
        is_on_floor = np.any(self.object_pos_at_time(self.get_time(), 'prey')[1] < 0.1 + self.GROUND_THRESHOLD)
        is_on_robot = self.sim.ground_on_robot('prey', 'robot') < self.GROUND_THRESHOLD
        return is_on_robot or is_on_floor
    
    def reset(self):
        return HuntingBase.reset(self)
