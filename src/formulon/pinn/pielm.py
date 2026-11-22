"""NumPy-only physics-informed solver: PIELM (physics-informed extreme learning machine).

A one-hidden-layer tanh network with *fixed random* hidden weights; only the output
weights are unknown.  For a linear PDE the residual, boundary and initial conditions are
linear in those weights, so training is one regularised least-squares solve -- no
autograd framework, no epochs, deterministic with a seed.  Works for the linear
benchmarks (heat, wave, Poisson, damped oscillator); nonlinear problems need the torch
backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import lstsq

from .derivatives import parse_key
from .sampling import sample_boundary, sample_interior

__all__ = ["PIELMModel", "pielm_fit"]


@dataclass
class PIELMModel:
    names: tuple
    center: np.ndarray
    half: np.ndarray
    W: np.ndarray
    b: np.ndarray
    beta: np.ndarray
    history: dict = field(default_factory=dict)

    def _phi(self, coords):
        z = (np.stack([np.asarray(c, dtype=float) for c in coords], axis=-1) - self.center) / self.half
        return np.tanh(z @ self.W.T + self.b)

    def __call__(self, *coords):
        shape = np.asarray(coords[0]).shape
        return (self._phi(coords).reshape(-1, self.W.shape[0]) @ self.beta).reshape(shape)


def _features(model_parts, pts, key):
    """Feature matrix of the derivative named by ``key`` (order <= 2) at points ``pts``."""
    names, center, half, W, b = model_parts
    z = (np.stack([np.asarray(pts[n], dtype=float) for n in names], axis=-1) - center) / half
    phi = np.tanh(z @ W.T + b)                       # (N, K)
    _, var = parse_key(key)
    if not var:
        return phi
    idx = [names.index(v) for v in var]
    sech2 = 1.0 - phi**2
    if len(idx) == 1:
        i = idx[0]
        return sech2 * W[:, i] / half[i]
    if len(idx) == 2:
        i, j = idx
        return -2.0 * phi * sech2 * W[:, i] * W[:, j] / (half[i] * half[j])
    raise NotImplementedError("PIELM supports derivatives up to second order")


def pielm_fit(problem, n_features: int = 400, n_collocation: int = 3000, n_boundary: int = 600,
              scale: float = 2.0, seed: int = 0, w_bc: float = 10.0, ridge: float = 1e-10) -> PIELMModel:
    """Solve a linear problem by one least-squares solve.  Returns a NumPy callable model."""
    lin = problem.linear_operator()
    if lin is None:
        raise NotImplementedError(f"{problem.name} is nonlinear: use backend='torch' (gradient-based PINN)")
    coeffs, source = lin
    out = problem.outputs[0]
    names = tuple(problem.domain)
    lo = np.array([problem.domain[n][0] for n in names], dtype=float)
    hi = np.array([problem.domain[n][1] for n in names], dtype=float)
    center, half = (lo + hi) / 2, (hi - lo) / 2
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, scale, size=(n_features, len(names)))
    b = rng.uniform(-scale, scale, size=n_features)
    parts = (names, center, half, W, b)

    rows, rhs = [], []
    col = sample_interior(problem.domain, n_collocation, "sobol", seed)
    A = sum(c * _features(parts, col, k) for k, c in coeffs.items())
    rows.append(A)
    rhs.append(np.zeros(len(A)) if source is None else np.asarray(source(col), dtype=float))

    def add_conditions(kind, pts):
        for key, target in problem.conditions(kind, pts).items():
            rows.append(w_bc * _features(parts, pts, key))
            rhs.append(w_bc * np.asarray(target, dtype=float))

    if any(n != "t" for n in names):
        add_conditions("boundary", sample_boundary(problem.domain, n_boundary, seed + 1))
    if "t" in names:
        ini = sample_boundary(problem.domain, n_boundary, seed + 2, fixed={"t": problem.domain["t"][0]})
        add_conditions("initial", ini)

    M, y = np.vstack(rows), np.concatenate(rhs)
    # Tikhonov via augmented system keeps lstsq robust for ill-conditioned feature matrices
    M_aug = np.vstack([M, np.sqrt(ridge) * np.eye(n_features)])
    y_aug = np.concatenate([y, np.zeros(n_features)])
    beta = lstsq(M_aug, y_aug, lapack_driver="gelsy")[0]
    res = M @ beta - y
    return PIELMModel(names, center, half, W, b, beta,
                      {"rms_fit_residual": float(np.sqrt(np.mean(res**2))), "output": out})
