from gym import spaces

from evogym import *
from evogym import np
from evogym.envs import BenchmarkBase
from evogym.envs.hunting.hunting_base import HuntingBase
# from evogym.envs.hunting.hunt_creeper import HuntCreeper
from evogym.envs.hunting.hunt_flyer_baseline import HuntFlyerBaseline
from evogym.envs.hunting.state_handler import StateHandler
from evogym.simulator_cpp import VisualProcessor
from evogym.envs.visual_perceptions.vis_line_viewer import VisLineViewer

import numpy as np
import os


class HuntFlyerBaselineVis(HuntFlyerBaseline):
    VIS_LIMIT_LEN = 17.

    def __init__(self, body: np.ndarray, connections=None):
        super().__init__(body=body, connections=connections)

        # set observation space
        # - the action space inherits from "hunting_base"
        # - the observation space has "vis_proc" observations instead prey's positions and velocities
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size
        # num_pred_voxels = np.sum((self.object_voxels_type('robot') == VOXEL_TYPES['PRED']), dtype=np.int64)
        self.vis_proc.update_configuration()
        num_vis_voxels = self.vis_proc.get_num_vis_surfaces()

        self.observation_space = spaces.Box(
            low=-100,
            high=100.0,
            shape=(num_vis_voxels * 2 + 2 + num_robot_points,),
            dtype=np.float
        )

    def change_params(self):
        # inherit super params
        super().change_params()

    def generate_viewer(self):
        self.vis_proc = VisualProcessor(1, self.sim, self.VIS_LIMIT_LEN * self.VOXEL_SIZE, -1)
        return VisLineViewer(vis_proc=self.vis_proc, sim_to_view=self._sim)
        
    def step(self, action: np.ndarray):
        obs, reward, done, info = super().step(action=action)

        self.vis_proc.update_for_timestep()

        # observation
        vis_types = np.array(self.vis_proc.get_vis1_types(), dtype=np.float)
        vis_sqr_depths = np.array(self.vis_proc.get_vis1_sqr_depths(), dtype=float)
        vis_dists = np.clip(1 - (vis_sqr_depths ** 0.5 / self.VOXEL_SIZE / self.VIS_LIMIT_LEN), 0., 1.)

        obs = np.concatenate((
            vis_types,
            vis_dists,
            np.mean(self.object_vel_at_time(self.get_time(), 'robot'), axis=1),
            self.get_relative_pos_obs('robot'),
        ))

        info["vis_types"] = vis_types
        info["vis_dists"] = vis_dists

        return obs, reward, done, info

    def reset(self):
        obs = super().reset()

        self.vis_proc.update_configuration()
        self.vis_proc.update_for_timestep()

        vis_types = np.array(self.vis_proc.get_vis1_types(), dtype=np.float)
        vis_dists = np.clip(1 - (np.array(self.vis_proc.get_vis1_sqr_depths(), dtype=float) ** 0.5 / self.VOXEL_SIZE / self.VIS_LIMIT_LEN), 0., 1.)

        obs = np.concatenate((
            vis_types,
            vis_dists,
            np.mean(self.object_vel_at_time(self.get_time(), 'robot'), axis=1),
            self.get_relative_pos_obs('robot'),
        ))

        return obs
        