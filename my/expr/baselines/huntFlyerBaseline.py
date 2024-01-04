import os, sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '../../../examples')
sys.path.insert(0, root_dir)

import random
import numpy as np

from evogym.utils import VOXEL_TYPES
from ga.run import run_ga



if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    
    run_ga(
        pop_size = 25,
        structure_shape = (5,5),
        experiment_name = "huntFlyerBaseline",
        max_evaluations = 500,
        train_iters = 1000,
        num_cores = 3,
        voxels_pd=np.array([
            3/8,  # EMPTY
            1/8,  # RIGID
            1/8,  # SOFT
            1/8,  # H_ACT
            1/8,  # V_ACT
            0,  # FIXED
            1/8,  # PRED
            0,  # PREY
            0,  # VIS
            ], np.float64),
        voxels_limits=np.array([
            -1,  # EMPTY
            -1,  # RIGID
            -1,  # SOFT
            -1,  # H_ACT
            -1,  # V_ACT
            0,  # FIXED
            3,  # PRED
            0,  # PREY
            0,  # VIS
            ], np.int64),
        structure_requirement=lambda robot: np.any(robot==VOXEL_TYPES['PRED']),
    )

