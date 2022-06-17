import numpy as np

from spiral_model import Spiral

rmin = 4e-6
pitch = 1e-6
rmax = 54e-6
#nturns = 10
#rmax = rmin + nturns * pitch
eps = 6.5

spiral = Spiral(rmin, rmax, pitch, eps=eps, order=2)
#spiral.fit(verbose=True, f0_guess=120.)
#print("f0 = {0:.4f} +/- {1:.4f} GHz".format(spiral.omega / (2. * np.pi), spiral.sigma_omega / (2. * np.pi)))

f0 = np.linspace(1., 10., 10)
L = np.empty(f0.size)
for i, f in enumerate(f0):
    params = np.array([f * 2. * np.pi, 1., 0.2])
    L[i] = spiral.loss_function(params)

import matplotlib.pyplot as plt
plt.plot(f0, L)
plt.show()
