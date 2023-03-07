import random
import numpy as np

from evogym.utils import VOXEL_TYPES
from ga.run import run_ga

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    
    run_ga(
        pop_size = 3,
        structure_shape = (5,5),
        experiment_name = "test_flyer",
        max_evaluations = 6,
        train_iters = 50,
        num_cores = 3,
        voxels_pd=np.array([3/8, 1/8, 1/8, 1/8, 1/8, 0, 1/8, 0], np.float64),
        voxels_limits=np.array([-1, -1, -1, -1, -1, 0, 3, 0], np.int64),
        structure_requirement=lambda robot: np.any(robot==VOXEL_TYPES['PRED']),
    )