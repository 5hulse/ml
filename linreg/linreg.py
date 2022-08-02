# linreg.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Wed 16 Mar 2022 16:07:27 GMT

from functools import reduce
import numpy as np
import numpy.linalg as nlinalg
from numpy.random import normal


def linear_regression(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return nlinalg.inv(X.T @ X) @ X.T @ y


# y = 2 * x_1 + 3 * x_2 + 4
model = [2, 2, -4]
shape = (24, 24)
dim = len(shape)
prod = reduce(lambda x, y: x * y, shape)

xs = (np.indices(shape).reshape(dim, prod)).T
X = np.ones((prod, dim + 1))
X[:, 1:] = xs

noise = normal(scale=4, size=prod)
y = X @ model + noise

beta = linear_regression(X, y)
plane = X @ beta

x1, x2 = np.meshgrid(*[np.arange(x) for x in shape], indexing="ij")

import matplotlib as mpl
mpl.use("tkAgg")
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(x1, x2, y.reshape(shape), alpha=1, color="deepseagreen")
ax.plot_surface(x1, x2, plane.reshape(shape), alpha=0.4, color="cornflowerblue")
plt.show()
