"""Collocation / boundary sampling and noisy-observation datasets."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import qmc

__all__ = ["sample_interior", "sample_boundary", "make_dataset", "Dataset"]


def sample_interior(domain: dict, n: int, method: str = "sobol", seed: int = 0) -> dict:
    """Points inside a box domain ``{'x': (0,1), 't': (0,1)}``.

    method: 'sobol' (low-discrepancy, default), 'lhs', or 'uniform'.
    """
    names = list(domain)
    lo = np.array([domain[k][0] for k in names], dtype=float)
    hi = np.array([domain[k][1] for k in names], dtype=float)
    d = len(names)
    if method == "sobol":
        m = max(1, int(np.ceil(np.log2(max(n, 2)))))
        u = qmc.Sobol(d, scramble=True, seed=seed).random_base2(m)[:n]
    elif method == "lhs":
        u = qmc.LatinHypercube(d, seed=seed).random(n)
    elif method == "uniform":
        u = np.random.default_rng(seed).random((n, d))
    else:
        raise ValueError("method must be 'sobol', 'lhs' or 'uniform'")
    pts = lo + u * (hi - lo)
    return {k: pts[:, i] for i, k in enumerate(names)}


def sample_boundary(domain: dict, n: int, seed: int = 0, fixed: dict | None = None) -> dict:
    """Points on the faces of the box.  ``fixed={'t': 0.0}`` pins a coordinate
    (e.g. initial-condition points); otherwise faces of all *spatial* axes
    (every axis not named 't') are sampled uniformly."""
    rng = np.random.default_rng(seed)
    pts = sample_interior(domain, n, "uniform", seed)
    if fixed:
        for k, v in fixed.items():
            pts[k] = np.full(n, float(v))
        return pts
    spatial = [k for k in domain if k != "t"]
    face_axis = rng.integers(0, len(spatial), n)
    face_side = rng.integers(0, 2, n)
    for i, k in enumerate(spatial):
        m = face_axis == i
        pts[k][m] = np.where(face_side[m] == 0, domain[k][0], domain[k][1])
    return pts


@dataclass
class Dataset:
    collocation: dict
    boundary: dict | None   # None for pure ODEs (no spatial boundary)
    initial: dict | None
    observations: dict        # coords + 'value' (noisy) + 'clean'
    meta: dict = field(default_factory=dict)


def make_dataset(problem, n_collocation=2000, n_boundary=200, n_observations=100,
                 noise_std: float = 0.0, seed: int = 0, method: str = "sobol") -> Dataset:
    """Ready-to-train data for a :class:`PDEProblem`.

    Observations are the exact solution plus Gaussian noise (relative to the
    solution's RMS), emulating sensor data for inverse problems / data
    assimilation.  Everything is seeded and reproducible.
    """
    dom = problem.domain
    col = sample_interior(dom, n_collocation, method, seed)
    has_space = any(k != "t" for k in dom)
    bnd = sample_boundary(dom, n_boundary, seed + 1) if has_space else None
    ini = sample_boundary(dom, n_boundary, seed + 2, fixed={"t": dom["t"][0]}) if "t" in dom else None
    obs_pts = sample_interior(dom, n_observations, "uniform", seed + 3)
    exact = problem.exact(**obs_pts)
    exact = exact if isinstance(exact, tuple) else (exact,)
    rng = np.random.default_rng(seed + 4)
    obs = dict(obs_pts)
    for name, val in zip(problem.outputs, exact):
        scale = np.sqrt(np.mean(np.square(val))) or 1.0
        obs[name + "_clean"] = val
        obs[name] = val + noise_std * scale * rng.standard_normal(val.shape)
    return Dataset(col, bnd, ini, obs,
                   {"problem": problem.name, "noise_std": noise_std, "seed": seed,
                    "params": dict(problem.params)})
