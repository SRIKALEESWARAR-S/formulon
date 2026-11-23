"""Benchmark PDE/ODE problems with closed-form reference solutions.

Each :class:`PDEProblem` bundles the equation residual (backend-agnostic: it
consumes a dict of named derivatives), the exact solution, the domain and the
physical parameters.  Because the exact solution is analytic, results are
reproducible and error metrics are exact -- no reference solver tolerances.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .derivatives import fd_derivatives

__all__ = ["PDEProblem", "get_problem", "list_problems", "heat_slab_from_material",
           "verify_problem"]

PI = math.pi


@dataclass
class PDEProblem:
    name: str
    description: str
    outputs: tuple
    domain: dict
    params: dict
    exact: Callable                      # exact(**coords) -> array | tuple of arrays
    derivative_keys: tuple
    _residual_terms: Callable            # (derivs, coords, params) -> {eq: array}
    use_case: str = ""
    reference: str = ""
    _factory: Callable = field(default=None, repr=False)

    # -- residual evaluation ------------------------------------------------
    def residual_from_derivs(self, derivs: dict, coords: dict) -> dict:
        """Residuals from precomputed derivatives (NumPy *or* torch tensors)."""
        return self._residual_terms(derivs, coords, self.params)

    def residual(self, model, coords: dict, h: float = 1e-3) -> dict:
        """PDE residuals of a NumPy callable ``model(*coords)`` via finite differences."""
        derivs = fd_derivatives(model, coords, self.outputs, self.derivative_keys, h)
        return self.residual_from_derivs(derivs, coords)

    def with_params(self, **kw) -> "PDEProblem":
        """Same problem with different physical parameters (e.g. other ``alpha``)."""
        p = {**self.params, **kw}
        return self._factory(**p)

    def exact_dict(self, **coords) -> dict:
        val = self.exact(**coords)
        val = val if isinstance(val, tuple) else (val,)
        return dict(zip(self.outputs, val))

    # -- conditions / structure ---------------------------------------------
    @property
    def second_order_time(self) -> bool:
        """True if the equation contains a second time derivative (needs an initial velocity)."""
        return any(k.split("_", 1)[-1] == "tt" for k in self.derivative_keys if "_" in k)

    def conditions(self, kind: str, pts: dict) -> dict:
        """Dirichlet targets from the exact solution: ``{derivative key: array}``.

        kind 'boundary' -> values of every output; kind 'initial' -> values, plus the
        initial velocity ``<out>_t`` when the equation is second order in time.
        """
        if kind not in ("boundary", "initial"):
            raise ValueError("kind must be 'boundary' or 'initial'")
        targets = dict(self.exact_dict(**pts))
        if kind == "initial" and self.second_order_time:
            h = 1e-5

            def shifted(sign):
                q = dict(pts)
                q["t"] = np.asarray(pts["t"], dtype=float) + sign * h
                return self.exact_dict(**q)
            hi, lo = shifted(+1), shifted(-1)
            for o in self.outputs:
                targets[o + "_t"] = (hi[o] - lo[o]) / (2 * h)
        return targets

    def linear_operator(self):
        """``({derivative key: coefficient}, source or None)`` for linear single-output
        problems, meaning  sum(coef*deriv) = source(coords); None for nonlinear ones.
        Used by the NumPy (PIELM) solver."""
        p = self.params
        if self.name == "heat_1d":
            return {"u_t": 1.0, "u_xx": -p["alpha"]}, None
        if self.name == "wave_1d":
            return {"u_tt": 1.0, "u_xx": -p["c"] ** 2}, None
        if self.name == "damped_oscillator":
            return {"x_tt": p["m"], "x_t": p["c"], "x": p["k"]}, None
        if self.name == "poisson_2d":
            w = PI**2 * (p["kx"] ** 2 + p["ky"] ** 2)
            return ({"u_xx": -1.0, "u_yy": -1.0},
                    lambda c: w * np.sin(p["kx"] * PI * c["x"]) * np.sin(p["ky"] * PI * c["y"]))
        return None


def _arr(*a):
    return [np.asarray(x, dtype=float) for x in a]


def _np(x):
    """Any array-like (NumPy, torch incl. requires_grad, JAX) -> float NumPy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)

def _like(ref, arr):
    """`arr` as the same array type, dtype and device as `ref` (NumPy / torch / JAX).

    `tensor + ndarray` is unsafe: for a tensor that requires grad, torch defers to NumPy,
    which calls tensor.numpy() and raises "Can't call numpy() on Tensor that requires grad".
    """
    if hasattr(ref, "new_tensor"):                      # torch tensor
        return ref.new_tensor(np.asarray(arr, dtype=float))
    if type(ref).__module__.split(".")[0] in ("jax", "jaxlib"):
        import jax.numpy as jnp
        return jnp.asarray(arr, dtype=ref.dtype)
    return np.asarray(arr, dtype=float)


# ---------------------------------------------------------------- heat 1-D
def _heat_1d(alpha=0.1, length=1.0, t_end=None):
    if alpha <= 0 or length <= 0:
        raise ValueError("alpha and length must be positive")
    k = PI / length
    if t_end is None:
        t_end = 1.0 / (alpha * k * k)          # one thermal decay time

    def exact(x, t):
        x, t = _arr(x, t)
        return np.exp(-alpha * k * k * t) * np.sin(k * x)

    def terms(d, c, p):
        return {"heat": d["u_t"] - p["alpha"] * d["u_xx"]}

    return PDEProblem(
        "heat_1d", "1-D heat conduction u_t = alpha u_xx, u(0,t)=u(L,t)=0, u(x,0)=sin(pi x/L)",
        ("u",), {"x": (0.0, length), "t": (0.0, t_end)},
        {"alpha": alpha, "length": length, "t_end": t_end}, exact, ("u_t", "u_xx"), terms,
        "Transient conduction in slabs: battery cells, PCB/heat-sink layers, furnace walls, "
        "thermal-management surrogates and inverse identification of diffusivity.",
        "Carslaw & Jaeger, Conduction of Heat in Solids (separation of variables)",
        _heat_1d)


# ---------------------------------------------------------------- wave 1-D
def _wave_1d(c=1.0, length=1.0, t_end=1.0):
    if c <= 0 or length <= 0:
        raise ValueError("c and length must be positive")
    k = PI / length

    def exact(x, t):
        x, t = _arr(x, t)
        return np.sin(k * x) * np.cos(c * k * t)

    def terms(d, co, p):
        return {"wave": d["u_tt"] - p["c"] ** 2 * d["u_xx"]}

    return PDEProblem(
        "wave_1d", "1-D wave equation u_tt = c^2 u_xx (standing wave, fixed ends)",
        ("u",), {"x": (0.0, length), "t": (0.0, t_end)},
        {"c": c, "length": length, "t_end": t_end}, exact, ("u_tt", "u_xx"), terms,
        "Vibration / acoustics of strings, rods and ducts; structural-health-monitoring surrogates; "
        "a hard test for PINNs because of their spectral bias.",
        "d'Alembert / separation of variables", _wave_1d)


# ------------------------------------------------------------- Burgers 1-D
def _burgers_1d(nu=0.01, A=2.0, B=1.0, k=PI):
    if nu <= 0 or A <= abs(B):
        raise ValueError("need nu>0 and A>|B| so the Cole-Hopf potential stays positive")

    def phi(x, t):
        return A + B * np.exp(-nu * k * k * t) * np.cos(k * x)

    def exact(x, t):
        x, t = _arr(x, t)
        e = np.exp(-nu * k * k * t)
        return 2 * nu * k * B * e * np.sin(k * x) / phi(x, t)

    def terms(d, c, p):
        return {"burgers": d["u_t"] + d["u"] * d["u_x"] - p["nu"] * d["u_xx"]}

    return PDEProblem(
        "burgers_1d", "Viscous Burgers u_t + u u_x = nu u_xx, exact via Cole-Hopf transform",
        ("u",), {"x": (0.0, 2.0), "t": (0.0, 1.0)}, {"nu": nu, "A": A, "B": B, "k": k}, exact,
        ("u", "u_t", "u_x", "u_xx"), terms,
        "Canonical nonlinear convection-diffusion benchmark: traffic flow, shock-forming flows, "
        "turbulence-model toy problems.", "Cole (1951); Hopf (1950)", _burgers_1d)


# ------------------------------------------------------------- Poisson 2-D
def _poisson_2d(kx=1, ky=1):
    w = PI**2 * (kx**2 + ky**2)

    def exact(x, y):
        x, y = _arr(x, y)
        return np.sin(kx * PI * x) * np.sin(ky * PI * y)

    def terms(d, c, p):
        x, y = _np(c["x"]), _np(c["y"])
        f = w * np.sin(kx * PI * x) * np.sin(ky * PI * y)
        # "* 0.0 + f" keeps the result in the derivative container's own type (NumPy/torch/JAX)
        return {"poisson": -(d["u_xx"] + d["u_yy"]) - _like(d["u_xx"], f)}

    return PDEProblem(
        "poisson_2d", "-Laplace(u) = f on the unit square, u=0 on the boundary (manufactured solution)",
        ("u",), {"x": (0.0, 1.0), "y": (0.0, 1.0)}, {"kx": kx, "ky": ky}, exact,
        ("u_xx", "u_yy"), terms,
        "Electrostatics, steady heat conduction, pressure-Poisson step of CFD, potential flow.",
        "Method of manufactured solutions (Roache)", _poisson_2d)


# ---------------------------------------------------- Kovasznay (Navier-Stokes)
def _kovasznay(Re=40.0):
    if Re <= 0:
        raise ValueError("Re must be positive")
    lam = Re / 2 - math.sqrt(Re**2 / 4 + 4 * PI**2)

    def exact(x, y):
        x, y = _arr(x, y)
        e = np.exp(lam * x)
        return (1 - e * np.cos(2 * PI * y), lam / (2 * PI) * e * np.sin(2 * PI * y),
                0.5 * (1 - np.exp(2 * lam * x)))

    def terms(d, c, p):
        Re_ = p["Re"]
        return {
            "continuity": d["u_x"] + d["v_y"],
            "momentum_x": d["u"] * d["u_x"] + d["v"] * d["u_y"] + d["p_x"] - (d["u_xx"] + d["u_yy"]) / Re_,
            "momentum_y": d["u"] * d["v_x"] + d["v"] * d["v_y"] + d["p_y"] - (d["v_xx"] + d["v_yy"]) / Re_,
        }

    keys = ("u", "v", "u_x", "u_y", "v_x", "v_y", "p_x", "p_y", "u_xx", "u_yy", "v_xx", "v_yy")
    return PDEProblem(
        "kovasznay", "Steady incompressible Navier-Stokes, Kovasznay flow behind a grid (u, v, p)",
        ("u", "v", "p"), {"x": (-0.5, 1.0), "y": (-0.5, 1.5)}, {"Re": Re}, exact, keys, terms,
        "Standard CFD/PINN verification case for incompressible flow solvers; coupled multi-output "
        "residuals (continuity + momentum).", "Kovasznay (1948)", _kovasznay)


# --------------------------------------------------- damped oscillator (ODE)
def _damped_oscillator(m=1.0, c=0.4, k=4.0, x0=1.0, t_end=10.0):
    if m <= 0 or k <= 0 or c < 0:
        raise ValueError("need m>0, k>0, c>=0")
    w0 = math.sqrt(k / m)
    zeta = c / (2 * math.sqrt(m * k))
    if zeta >= 1:
        raise ValueError(f"damping ratio zeta={zeta:.3f} >= 1: only the underdamped case is provided")
    wd = w0 * math.sqrt(1 - zeta**2)

    def exact(t):
        (t,) = _arr(t)
        return x0 * np.exp(-zeta * w0 * t) * (np.cos(wd * t) + zeta * w0 / wd * np.sin(wd * t))

    def terms(d, co, p):
        return {"ode": p["m"] * d["x_tt"] + p["c"] * d["x_t"] + p["k"] * d["x"]}

    pr = PDEProblem(
        "damped_oscillator", "m x'' + c x' + k x = 0, x(0)=x0, x'(0)=0 (underdamped)",
        ("x",), {"t": (0.0, t_end)}, {"m": m, "c": c, "k": k, "x0": x0, "t_end": t_end}, exact,
        ("x", "x_t", "x_tt"), terms,
        "Suspensions, structural dynamics, RLC analogues; system identification of (m, c, k) from "
        "noisy sensor data; conservation tests (c=0) for learned dynamics.",
        "Classical mechanics, e.g. Taylor 'Classical Mechanics' ch.5", _damped_oscillator)
    return pr


_FACTORIES = {"heat_1d": _heat_1d, "wave_1d": _wave_1d, "burgers_1d": _burgers_1d,
              "poisson_2d": _poisson_2d, "kovasznay": _kovasznay,
              "damped_oscillator": _damped_oscillator}


def list_problems() -> list:
    return sorted(_FACTORIES)


def get_problem(name: str, **params) -> PDEProblem:
    if name not in _FACTORIES:
        raise KeyError(f"unknown problem {name!r}; available: {list_problems()}")
    return _FACTORIES[name](**params)


def heat_slab_from_material(k: float, rho: float, cp: float, length: float) -> PDEProblem:
    """Heat-conduction benchmark for a real slab: alpha = k / (rho cp) [m^2/s].

    k [W/(m K)], rho [kg/m^3], cp [J/(kg K)], length [m].  Time is in seconds.
    Example: copper PCB layer, aluminium heat spreader, Li-ion cell stack.
    """
    for nme, v in (("k", k), ("rho", rho), ("cp", cp), ("length", length)):
        if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
            raise ValueError(f"{nme} must be a positive finite number")
    return _heat_1d(alpha=k / (rho * cp), length=length)


def verify_problem(problem: PDEProblem, n: int = 400, h: float = 1e-3, tol: float = 1e-4,
                   seed: int = 0) -> dict:
    """Self-check: the closed-form solution must satisfy its own residual.

    Run this on a *new* problem definition before trusting any PINN results
    against it; it catches sign errors and missing factors in either the
    residual or the reference solution.
    """
    from .sampling import sample_interior

    dom = {k: (lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo)) for k, (lo, hi) in problem.domain.items()}
    pts = sample_interior(dom, n, "uniform", seed)

    def model(*args):
        val = problem.exact(**dict(zip(problem.domain, args)))
        return val

    res = problem.residual(model, pts, h)
    rms = {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in res.items()}
    return {"problem": problem.name, "rms": rms, "passed": all(v < tol for v in rms.values()),
            "tol": tol}
