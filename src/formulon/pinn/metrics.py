"""Standard PINN error metrics (definitions follow PINN benchmark practice)."""
from __future__ import annotations

import numpy as np

__all__ = ["relative_l2", "max_abs_error", "rmse", "residual_report",
           "parameter_error", "energy_drift"]


def _pair(pred, ref):
    p, r = np.asarray(pred, dtype=float), np.asarray(ref, dtype=float)
    if p.shape != r.shape:
        raise ValueError(f"shape mismatch: prediction {p.shape} vs reference {r.shape}")
    return p, r


def relative_l2(pred, ref, eps: float = 1e-12) -> float:
    """||pred-ref||_2 / ||ref||_2  (the headline PINN metric)."""
    p, r = _pair(pred, ref)
    return float(np.linalg.norm(p - r) / max(np.linalg.norm(r), eps))


def max_abs_error(pred, ref) -> float:
    p, r = _pair(pred, ref)
    return float(np.max(np.abs(p - r)))


def rmse(pred, ref) -> float:
    p, r = _pair(pred, ref)
    return float(np.sqrt(np.mean((p - r) ** 2)))


def residual_report(residuals: dict) -> dict:
    """Summarise PDE residual arrays: per-equation RMS, max and a pass flag basis."""
    rep = {}
    for name, r in residuals.items():
        r = np.asarray(r, dtype=float)
        rep[name] = {"rms": float(np.sqrt(np.mean(r**2))), "max": float(np.max(np.abs(r)))}
    return rep


def parameter_error(estimated: dict, true: dict) -> dict:
    """Relative error per identified parameter (inverse problems)."""
    out = {}
    for k, v in true.items():
        if k in estimated:
            out[k] = abs(float(estimated[k]) - float(v)) / max(abs(float(v)), 1e-12)
    return out


def energy_drift(energy) -> float:
    """Max relative deviation of a conserved quantity along a trajectory.

    Physics-consistency check for learned dynamics: a model of a conservative
    system should keep this near zero over the rollout.
    """
    e = np.asarray(energy, dtype=float)
    e0 = e[0] if abs(e[0]) > 1e-12 else 1.0
    return float(np.max(np.abs(e - e[0])) / abs(e0))
