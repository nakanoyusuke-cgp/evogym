import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(0.1, 0.8, 100)
y = 0.01 / (x ** 2)
# y = 1 / (1 + x**2)

plt.plot(x, y)
plt.show()
