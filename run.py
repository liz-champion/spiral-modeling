import numpy as np
import matplotlib.pyplot as plt
import sys

from spiral_model import Spiral

rmax = 41e-6
rmin_array = np.linspace(0.05 * rmax, 0.25 * rmax, 20)
pitch = 0.5e-6
order = 4

rho = np.empty(rmin_array.size)
f0_arch = np.empty(rmin_array.size)
f0_ring = np.empty(rmin_array.size)
f0_here = np.empty(rmin_array.size)
out = np.empty((rmin_array.size, order + 7))

for i, rmin in enumerate(rmin_array):
    print("iteration", i, ":\t", end="")
    sys.stdout.flush()
    spiral = Spiral(rmin, rmax, pitch, order=order)

    f0_arch[i] = spiral.calculate_f0_archimedean()
    f0_ring[i] = spiral.calculate_f0_ring()

    spiral.fit()

    rho[i] = (rmax - rmin) / (rmax + rmin)
    f0_here[i] = 1e9 * spiral.omega / (2. * np.pi)
    print("f0 = {0:.3f} +/- {1:.3f} GHz".format(f0_here[i]*1e-9, spiral.sigma_omega / (2. * np.pi)))

    out[i][0] = rmin
    out[i][1] = rmax
    out[i][2] = pitch
    out[i][3] = f0_arch[i]
    out[i][4] = f0_ring[i]
    out[i][5] = f0_here[i]
    out[i][6] = spiral.sigma_omega * 1e9 / (2. * np.pi)
    out[i][7:] = spiral.coeffs

header = "rmin \trmax \tpitch \tf0_arch \tf0_ring \tf0_num \tf0_err \tc" + " c".join(str(i + 1) for i in range(order))
fname = "rmax_{0}-p{1}.txt".format(rmax*1e9, pitch*1e9)
np.savetxt(fname, out, header=header)

plt.figure(figsize=(8, 5))
plt.plot(rho, f0_arch*1e-9, color="m", label="Archimedean")
plt.plot(rho, f0_ring*1e-9, color="blue", label="Ring")
plt.plot(rho, f0_here*1e-9, color="black", label="Numerical")
#plt.errorbar(rho, f0_here*1e-9, capsize=5, c="black", fmt="none")
plt.xlabel("fill ratio", fontsize=14)
plt.ylabel("$f_0$ (GHz)", fontsize=14)
plt.axvline((rmax - 4e-6) / (rmax + 4e-6), color="black")
plt.legend()
plt.savefig("41_um.png")
plt.show()
