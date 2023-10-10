from gym import spaces

from evogym import *
from evogym.envs import BenchmarkBase
from evogym.simulator_cpp import VisualProcessor
from evogym.envs.visual_perceptions.vis_line_viewer import VisLineViewer

import numpy as np
import os


DEFAULT_CONFIG = {
    # task-specific config
    "INIT_BOX_X": 20,
    "REWARD_DIST_BOT": 10.,
    "REWARD_DIST_TOP": 12.,

    # vis1 config
    "VIS_LIMIT_LEN": 22.
}


class ObserverVis1(BenchmarkBase):
    INIT_BOX_X = 20
    REWARD_DIST_BOT = 10.
    REWARD_DIST_TOP = 12.
    VIS_LIMIT_LEN = 22.

    def __init__(self, body: np.ndarray, connections=None, config=None):
        if config is not None:
            self.change_config(config)

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Walker-v0.json'))
        self.world.add_from_array('robot', body, 1, 1, connections=connections)
        self.world.add_from_array('box', np.array([[7, 7], [7, 7], ]), self.INIT_BOX_X, 1)

        # init sim
        BenchmarkBase.__init__(self, world=self.world)

        # self._sim = EvoSim(self.world)
        self.vis_proc = VisualProcessor(1, self.sim, self.VIS_LIMIT_LEN * self.VOXEL_SIZE, -1)

        self._default_viewer = VisLineViewer(vis_proc=self.vis_proc, sim_to_view=self._sim)
        self.default_viewer.track_objects('robot', 'box')

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size
        self.vis_proc.update_configuration()
        num_vis_voxels = self.vis_proc.get_num_vis_surfaces()

        self.action_space = spaces.Box(low=0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        # vis, v^r, c^r
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(num_vis_voxels * 2 + 2 + num_robot_points,), dtype=np.float)

    def change_config(self, config: dict):
        self.INIT_BOX_X = config["INIT_BOX_X"]
        self.REWARD_DIST_BOT = config["REWARD_DIST_BOT"]
        self.REWARD_DIST_TOP = config["REWARD_DIST_TOP"]
        self.VIS_LIMIT_LEN = config["VIS_LIMIT_LEN"]
        # print("hunt creeper vis1, change config")

    def step(self, action: np.ndarray):

        # step
        done = super().step({"robot": action})
        self.vis_proc.update_for_timestep()

        # observation
        vis_types = np.array(self.vis_proc.get_vis1_types(), dtype=np.float)
        vis_sqr_depths = np.array(self.vis_proc.get_vis1_sqr_depths(), dtype=float)
        vis_dists = np.clip(1 - (vis_sqr_depths ** 0.5 / self.VOXEL_SIZE / self.VIS_LIMIT_LEN), 0., 1.)

        obs = np.concatenate((
            vis_types,
            vis_dists,
            self.get_vel_com_obs("robot"),
            self.get_relative_pos_obs('robot'),
        ))

        # compute reward
        pos_r = np.mean(self.object_pos_at_time(self.get_time(), "robot"), 1)
        pos_b = np.mean(self.object_pos_at_time(self.get_time(), "box"), 1)
        robot_box_dist = np.linalg.norm(pos_b - pos_r)

        if self.REWARD_DIST_BOT <= robot_box_dist / self.VOXEL_SIZE <= self.REWARD_DIST_TOP:
            reward = 1.
        else:
            reward = 0.

        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0

        # info
        info = {
            "vis_sqr_depths": vis_sqr_depths,
        }

        return obs, reward, done, info

    def reset(self):
        super().reset()

        # update vis configuration
        self.vis_proc.update_configuration()
        self.vis_proc.update_for_timestep()

        vis_types = np.array(self.vis_proc.get_vis1_types(), dtype=np.float)
        vis_dists = np.clip(1 - (np.array(self.vis_proc.get_vis1_sqr_depths(), dtype=float) ** 0.5 / self.VOXEL_SIZE / self.VIS_LIMIT_LEN), 0., 1.)

        obs = np.concatenate((
            vis_types,
            vis_dists,
            self.get_vel_com_obs("robot"),
            self.get_relative_pos_obs('robot'),
        ))

        return obs
