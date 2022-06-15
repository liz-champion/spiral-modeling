import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import jv
from scipy.constants import c

c_eff = c / np.sqrt(6.89)

rmax = 40e-6
pitch = 0.5e-6

### Eq. (3) in https://link.aps.org/doi/10.1103/PhysRevApplied.14.044055
def calculate_f0_archimedean(rmin, rmax, pitch):
    xi = 0.81
    f0 = xi * c_eff * 2. * pitch / (np.pi * (2. * rmax)**2)
    return f0

### see http://aip.scitation.org/doi/10.1063/1.4863835
def calculate_p0(rmin, rmax, pitch):
    w = 0.5 * (rmax - rmin)
    target_function = lambda p: (np.log(2. * rmax / w) * pitch * w * jv(1, pitch * w) - jv(0, pitch * w))**2
    res = minimize(target_function, 1./w, tol=1e-16)
    p0 = res.x[0]
    return p0
    
def calculate_f0_ring(rmin, rmax, pitch):
    p0 = calculate_p0(rmin, rmax, pitch)
    alpha = pitch / (2. * np.pi * rmax)
    return c_eff * alpha * p0 / (2. * np.pi)

rmin = np.linspace(0.05 * rmax, 0.95 * rmax, 100)
rho = np.empty(rmin.size)
f0_arch = np.empty(rmin.size)
f0_ring = np.empty(rmin.size)

for i, r in enumerate(rmin):
    rho[i] = (rmax - r) / (rmax + r)
    f0_arch[i] = calculate_f0_archimedean(r, rmax, pitch)
    f0_ring[i] = calculate_f0_ring(r, rmax, pitch)

plt.figure(figsize=(8, 5))
plt.plot(rho, f0_arch*1e-9, color="m", label="Archimedean")
plt.plot(rho, f0_ring*1e-9, color="blue", label="Ring")
plt.xlabel("fill ratio", fontsize=14)
plt.ylabel("$f_0$ (GHz)", fontsize=14)
plt.legend()
plt.show()
