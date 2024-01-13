import math

import matplotlib.pyplot as plt
import numpy as np
import random
import os
import sys

cur_path = os.path.dirname(os.path.abspath(__file__))
examples_path = os.path.join(cur_path, "../examples")
sys.path.insert(0, examples_path)

from utils.algo_utils import get_percent_survival_evals


x = np.arange(500, dtype=np.int64)

y = np.array([
    max(2, math.ceil(get_percent_survival_evals(i, 500) * 25)) for i in x
])


def gpse(c, m):
    low = 0.0
    high = 0.6
    return ((m-c-1)/(m-1)) ** (1/3) * (high-low) + low


y2 = np.array([
    max(2, math.ceil(gpse(i, 500) * 25)) for i in x
])

plt.plot(x, y)
plt.plot(x, y2)
plt.ylim((0, 25))

plt.show()
