import numpy as np
from scipy.constants import pi, c, mu_0, epsilon_0
from scipy.integrate import quad
from scipy.misc import derivative
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.special import jv

### constants
I = 1. # fiducial current amplitude in the spiral, doesn't actually matter

def integrate(f, xmin, xmax, n=1000):
    dx = (xmax - xmin) / n
    x = np.arange(xmin, xmax + dx, dx)
    return dx * np.sum(f(x))

class Spiral:

    def __init__(self, rmin, rmax, pitch, order=6, eps=6.5, n_eval=None):
        self.rmin = rmin
        self.rmax = rmax
        self.pitch = pitch
        self.nturns = (rmax - rmin) / pitch
        self.order = order
        self.c_eff = c / np.sqrt(eps)
        self.eps = eps

        self.fill_ratio = (rmax - rmin) / (rmax + rmin)

        self.length = pi * (rmax**2 - rmin**2) / pitch # approximate the length of the spiral as area/pitch
        self.alpha = pitch / (2. * pi * rmax)
        
        if n_eval is None:
            self.n_eval = int(0.75 * self.nturns)
        else:
            self.n_eval = n_eval
        #self.r_eval = np.sqrt(np.random.uniform(1.001 * self.rmin**2, 0.999 * self.rmax**2, self.n_eval))
        #self.theta_eval = np.random.uniform(0., 2. * pi, self.n_eval)
        self.theta_eval = np.sqrt(np.random.uniform(0., (2. * pi * self.nturns)**2, self.n_eval))
        self.r_eval = self.rmax * (1. - self.alpha * self.theta_eval) + 0.5 * self.pitch

        self.omega = None
        self.sigma_omega = None
        self.coeffs = None

    def phi(self, s):
        return (1. / self.alpha) * (1. - np.sqrt(1. - 2. * self.alpha * s / self.rmax))

    def eval_psi(self, s, coeffs):
        psi = 0.
        #r = self.rmax * (1. - self.alpha * self.phi(s))
        #nu = ((r - self.rmin) / (self.rmax - self.rmin))**2
        for n, c in enumerate(coeffs):
            #psi += c * np.sin(pi * (n + 1.) * nu)
            psi += c * np.sin(pi * (n + 1.) * s / self.length)
        return psi

    def vector_potential_radial(self, r, theta, omega, coeffs):
        R = lambda s: np.sqrt(r**2 + self.rmax**2 * (1. - self.alpha * self.phi(s))**2 - 2. * r * self.rmax * (1. - self.alpha * self.phi(s)) * np.cos(self.phi(s) - theta))
        rho = lambda s: self.rmax * (1. - self.alpha * self.phi(s))
        dphi_ds = lambda s: 1. / np.sqrt(self.alpha**2 * self.rmax**2 + rho(s))
        k = omega / self.c_eff
        
        integrand_radial = lambda s: (1. / R(s)) * np.exp(-1.j * k * R(s)) * self.eval_psi(s, coeffs) * dphi_ds(s) * (rho(s) * np.sin(theta - self.phi(s)) - self.rmax * self.alpha * np.cos(theta - self.phi(s)))
        
        n = int(4. * self.length / self.pitch)
        #import matplotlib.pyplot as plt; s = np.linspace(0., self.length, n); plt.scatter(R(s), integrand_radial(s), s=2.); plt.show(); exit()
        A_r = integrate(integrand_radial, 0., self.length, n=n)
        return A_r

    def vector_potential_angular(self, r, theta, omega, coeffs):
        R = lambda s: np.sqrt(r**2 + self.rmax**2 * (1. - self.alpha * self.phi(s))**2 - 2. * r * self.rmax * (1. - self.alpha * self.phi(s)) * np.cos(self.phi(s) - theta))
        rho = lambda s: self.rmax * (1. - self.alpha * self.phi(s))
        dphi_ds = lambda s: 1. / np.sqrt(self.alpha**2 * self.rmax**2 + rho(s))
        k = omega / self.c_eff
    
        integrand_angular = lambda s: (1. / R(s)) * np.exp(-1.j * k * R(s)) * self.eval_psi(s, coeffs) * dphi_ds(s) * (rho(s) * np.cos(theta - self.phi(s)) - self.rmax * self.alpha * np.sin(theta - self.phi(s)))
    
        n = int(4. * self.length / self.pitch)
        A_theta = integrate(integrand_angular, 0., self.length, n=n)
        return A_theta

    def electric_field(self, r, theta, omega, coeffs):
        dr = 1e-2 * self.pitch
        E_r = derivative(lambda r1: (1. / r1) * derivative(lambda r2: r2 * self.vector_potential_radial(r2, theta, omega, coeffs), r1, dr), r, dr)
        #E_r = derivative(lambda r1: self.vector_potential_radial(r1, theta, omega, coeffs), r, dr, n=2)
        E_r /= (1.j * omega * epsilon_0 * mu_0)

        E_theta = -1.j * omega * self.vector_potential_angular(r, theta, omega, coeffs)
        #print(np.abs(self.rmax * self.alpha * E_r), np.abs(r * E_theta))

        return E_r, E_theta

    def loss_function(self, params):
        omega = params[0] * 1e9
        coeffs = params[1:]
        L = 0.
        y = []
        for _theta, _r in zip(self.theta_eval, self.r_eval):
            E_r, E_theta = self.electric_field(_r, _theta, omega, coeffs)
            temp = self.rmax * self.alpha * E_r + _r * E_theta
            y.append(np.abs(temp))
            L += np.abs(temp)
        print(params[0])
        import matplotlib.pyplot as plt; plt.scatter(self.r_eval, y); plt.show()
        return L

    ### Eq. (3) in https://link.aps.org/doi/10.1103/PhysRevApplied.14.044055
    def calculate_f0_archimedean(self):
        xi = 0.81
        f0 = xi * self.c_eff * 2. * self.pitch / (np.pi * (2. * self.rmax)**2)
        return f0

    ### see http://aip.scitation.org/doi/10.1063/1.4863835
    def calculate_p0(self):
        w = 0.5 * (self.rmax - self.rmin)
        target_function = lambda p: (np.log(2. * self.rmax / w) * self.pitch * w * jv(1, self.pitch * w) - jv(0, self.pitch * w))**2
        res = minimize(target_function, 1./w, tol=1e-16)
        p0 = res.x[0]
        return p0
        
    def calculate_f0_ring(self):
        p0 = self.calculate_p0()
        return self.c_eff * self.alpha * p0 / (2. * np.pi)

    def fit(self, verbose=False, f0_guess=None):
        if f0_guess is None:
            f0_arch = self.calculate_f0_archimedean() * 1e-9
            f0_ring = self.calculate_f0_ring() * 1e-9
            f0_avg = self.fill_ratio * f0_arch + (1. - self.fill_ratio) * f0_ring
            omega_avg = f0_avg * 2. * pi
            omega_min = 0.1 * omega_avg
            omega_max = 2. * omega_avg
            omega_guess = omega_avg
        else:
            omega_guess = 2. * pi * f0_guess

        x0 = np.array([omega_guess, 1.] + [0.] * (self.order - 1))

        if verbose:
            def loss(params):
                L = self.loss_function(params)
                print("f0 = {0:.3f} GHz \t L = {1}".format(params[0] / (2. * pi), L))
                return L
        else:
            loss = self.loss_function

        bounds = [[0.5 * omega_guess, 1.4 * omega_guess], [1., 1.]] + [[-0.4, 0.4]] * (self.order - 1)

        #res = minimize(loss, x0, method="L-BFGS-B", bounds=bounds)
        res = minimize(loss, x0, method="Nelder-Mead", bounds=bounds)
        self.omega = res.x[0]
        self.sigma_omega = 0. #np.sqrt(np.diag(res.hess_inv)[0])
        self.coeffs = res.x[1:]
