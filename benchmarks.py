"""
benchmarks.py
Standard single-objective optimisation benchmarks (minimisation) plus two
constrained engineering design problems, for pre-validating the hyper-heuristic
before the ETC application. Pure numpy.

Every Problem exposes:
  name, dim, lo[dim], hi[dim], optimum (known global minimum objective),
  target (success threshold on the objective), eval(x) -> scalar to MINIMISE.
Constrained problems fold the constraints into eval() through a static penalty
(see PenalisedProblem) so the single-objective optimisers can be applied unchanged.
"""
import numpy as np


class Problem:
    def __init__(self, name, dim, lo, hi, optimum, target, fn):
        self.name = name
        self.dim = dim
        self.lo = np.asarray(lo, float) * np.ones(dim)
        self.hi = np.asarray(hi, float) * np.ones(dim)
        self.optimum = optimum
        self.target = target
        self._fn = fn

    def eval(self, x):
        x = np.clip(np.asarray(x, float), self.lo, self.hi)
        return float(self._fn(x))

    def random(self, rng):
        return self.lo + rng.random(self.dim) * (self.hi - self.lo)


# ---------------------------------------------------------- classical functions
def _sphere(x):      return np.sum(x**2)
def _rastrigin(x):   return 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))
def _ackley(x):
    n = len(x)
    return (-20*np.exp(-0.2*np.sqrt(np.sum(x**2)/n))
            - np.exp(np.sum(np.cos(2*np.pi*x))/n) + 20 + np.e)
def _rosenbrock(x):  return np.sum(100*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2)
def _griewank(x):
    i = np.arange(1, len(x)+1)
    return np.sum(x**2)/4000 - np.prod(np.cos(x/np.sqrt(i))) + 1
def _schwefel(x):    return 418.9828872724338*len(x) - np.sum(x*np.sin(np.sqrt(np.abs(x))))


def classical_suite(dim=10):
    return [
        Problem("Sphere",     dim, -5.12, 5.12,  0.0, 1e-6, _sphere),
        Problem("Rastrigin",  dim, -5.12, 5.12,  0.0, 1e-2, _rastrigin),
        Problem("Ackley",     dim, -32.768, 32.768, 0.0, 1e-3, _ackley),
        Problem("Rosenbrock", dim, -5.0, 10.0,  0.0, 1e-1, _rosenbrock),
        Problem("Griewank",   dim, -600.0, 600.0, 0.0, 1e-2, _griewank),
        Problem("Schwefel",   dim, -500.0, 500.0, 0.0, 1e-1, _schwefel),
    ]


# ----------------------------------------------- constrained engineering design
class PenalisedProblem:
    """Objective + sum of squared constraint violations * big penalty."""
    def __init__(self, name, dim, lo, hi, optimum, target, obj, cons, penalty=1e6):
        self.name = name; self.dim = dim
        self.lo = np.asarray(lo, float); self.hi = np.asarray(hi, float)
        self.optimum = optimum; self.target = target
        self._obj = obj; self._cons = cons; self.penalty = penalty

    def eval(self, x):
        # Deb feasibility-first: any feasible solution ranks below any infeasible one
        x = np.clip(np.asarray(x, float), self.lo, self.hi)
        g = self._cons(x)                         # list of g(x) <= 0
        viol = float(np.sum(np.maximum(0.0, g)))
        if viol <= 1e-6:
            return float(self._obj(x))
        return float(1.0e6 + 1.0e3*viol)

    def feasible(self, x):
        x = np.clip(np.asarray(x, float), self.lo, self.hi)
        return bool(np.all(self._cons(x) <= 1e-6))

    def random(self, rng):
        return self.lo + rng.random(self.dim)*(self.hi - self.lo)


def welded_beam():
    # vars: h, l, t, b ; classic Coello formulation. Best known ~1.724852.
    P, L, E, G = 6000.0, 14.0, 30e6, 12e6
    tau_max, sig_max, delta_max = 13600.0, 30000.0, 0.25
    def obj(x):
        h, l, t, b = x
        return 1.10471*h**2*l + 0.04811*t*b*(14.0 + l)
    def cons(x):
        h, l, t, b = x
        Pc = (4.013*E*np.sqrt(t**2*b**6/36.0)/(L**2))*(1 - t/(2*L)*np.sqrt(E/(4*G)))
        sigma = 6*P*L/(b*t**2)
        delta = 6*P*L**3/(E*t**3*b)
        M = P*(L + l/2.0)
        R = np.sqrt(l**2/4.0 + ((h + t)/2.0)**2)
        J = 2*(np.sqrt(2)*h*l*(l**2/12.0 + ((h + t)/2.0)**2))
        tau1 = P/(np.sqrt(2)*h*l)
        tau2 = M*R/J
        tau = np.sqrt(tau1**2 + 2*tau1*tau2*(l/(2*R)) + tau2**2)
        return np.array([tau - tau_max, sigma - sig_max, h - b,
                         0.10471*h**2 + 0.04811*t*b*(14+l) - 5.0,
                         0.125 - h, delta - delta_max, P - Pc])
    return PenalisedProblem("WeldedBeam", 4, [0.1,0.1,0.1,0.1], [2.0,10.0,10.0,2.0],
                            1.724852, 1.74, obj, cons)


def pressure_vessel():
    # vars: Ts, Th, R, L ; classic formulation. Best known ~6059.714.
    def obj(x):
        Ts, Th, R, L = x
        return 0.6224*Ts*R*L + 1.7781*Th*R**2 + 3.1661*Ts**2*L + 19.84*Ts**2*R
    def cons(x):
        Ts, Th, R, L = x
        return np.array([-Ts + 0.0193*R, -Th + 0.00954*R,
                         -np.pi*R**2*L - (4.0/3.0)*np.pi*R**3 + 1296000.0,
                         L - 240.0])
    return PenalisedProblem("PressureVessel", 4, [0.0625,0.0625,10.0,10.0],
                            [99*0.0625, 99*0.0625, 200.0, 200.0],
                            6059.714, 6100.0, obj, cons)


def engineering_suite():
    return [welded_beam(), pressure_vessel()]


if __name__ == "__main__":
    for p in classical_suite(10) + engineering_suite():
        rng = np.random.default_rng(0)
        print(f"{p.name:14s} dim={p.dim} opt={p.optimum} sample f={p.eval(p.random(rng)):.3f}")
