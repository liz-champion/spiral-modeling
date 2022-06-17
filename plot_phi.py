import numpy as np
import matplotlib.pyplot as plt

rmin = 5e-6
rmax = 25e-6
pitch = 2e-6
alpha = pitch / (2. * np.pi * rmax)
nturns = (rmax - rmin) / pitch

rho = lambda phi: rmax * (1. - alpha * phi)
phi_approx = lambda s: (1. / alpha) * (1. - np.sqrt(1. - 2. * alpha * s / rmax))

dphi = 1e-3
phi = np.arange(0., 2. * np.pi * nturns, dphi)
s = np.zeros(phi.size)

for i, p in enumerate(phi[1:]):
    s[i + 1] = s[i] + np.sqrt(alpha**2 * rmax**2 + rho(p)**2) * dphi

plt.figure(figsize=(8, 5))
#plt.plot(s*1e6, phi_approx(s), color="black", label="Approximation")
#plt.plot(s*1e6, phi, linestyle=(0, (5, 5)), color="red", label="Numerical integration")
plt.plot(s, phi_approx(s) - phi)
plt.show()
exit()
plt.xlabel("s (μm)", fontsize=14)
plt.ylabel("$\phi$ (rad)", fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig("latex/phi.pdf")
