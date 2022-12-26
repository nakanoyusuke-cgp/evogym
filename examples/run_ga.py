import random
import numpy as np

from ga.run import run_ga

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    
    run_ga(
        pop_size = 3,
        structure_shape = (5,5),
        experiment_name = "test_hunting_ga",
        max_evaluations = 6,
        train_iters = 50,
        num_cores = 3,
        voxels_limits=np.array([-1, -1, -1, -1, -1, 0, 3, 0], np.int)
    )