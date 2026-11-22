"""Derivative providers.  A residual only needs a dict of named derivatives,
so the same residual runs on NumPy (finite differences) or torch (autograd)."""
from __future__ import annotations

import numpy as np

__all__ = ["fd_derivatives", "parse_key", "split_outputs"]


def parse_key(key: str):
    """'u_xx' -> ('u', 'xx'); 'p' -> ('p', '')."""
    if "_" in key:
        out, var = key.split("_", 1)
        return out, var
    return key, ""


def split_outputs(out, outputs) -> dict:
    """Model output (tuple, (N,) array, or (N, n_out) array; NumPy/torch/JAX) -> {name: array}."""
    if isinstance(out, (tuple, list)):
        return dict(zip(outputs, out))
    if len(outputs) == 1:
        o = outputs[0]
        return {o: out[..., 0] if (out.ndim > 1 and out.shape[-1] == 1) else out}
    return {o: out[..., i] for i, o in enumerate(outputs)}


def _eval(model, coords: dict, names):
    res = model(*[coords[n] for n in names])
    if not isinstance(res, (tuple, list)):
        res = (res,)
    return [np.asarray(r, dtype=float) for r in res]


def fd_derivatives(model, coords: dict, outputs, keys, h: float = 1e-3) -> dict:
    """Central finite-difference derivatives of a NumPy callable.

    Parameters
    ----------
    model : callable ``model(*coords_in_order) -> array or tuple of arrays``.
    coords : ``{'x': arr, 't': arr, ...}`` (order = dict order).
    outputs : output names matching the model's return order, e.g. ``('u',)``.
    keys : requested derivative keys such as ``'u_t'``, ``'u_xx'``, ``'v_y'``.
    h : step.  Second derivatives have O(h^2) truncation and O(eps/h^2)
        round-off; the default 1e-3 gives ~1e-6 accuracy in float64.
    """
    names = list(coords)
    idx = {o: i for i, o in enumerate(outputs)}
    cache: dict = {}

    def value(shift: tuple):
        if shift not in cache:
            c = {n: coords[n] + dict(shift).get(n, 0.0) for n in names}
            cache[shift] = _eval(model, c, names)
        return cache[shift]

    def d(out, var, shift=()):
        """derivative of output ``out`` w.r.t. string ``var`` evaluated at ``shift``."""
        if not var:
            return value(shift)[idx[out]]
        v0 = var[0]
        if len(var) == 2 and var[0] == var[1]:
            sp = tuple(sorted(dict(shift, **{v0: dict(shift).get(v0, 0.0) + h}).items()))
            sm = tuple(sorted(dict(shift, **{v0: dict(shift).get(v0, 0.0) - h}).items()))
            return (d(out, "", sp) - 2 * d(out, "", shift) + d(out, "", sm)) / h**2
        sp = tuple(sorted(dict(shift, **{v0: dict(shift).get(v0, 0.0) + h}).items()))
        sm = tuple(sorted(dict(shift, **{v0: dict(shift).get(v0, 0.0) - h}).items()))
        return (d(out, var[1:], sp) - d(out, var[1:], sm)) / (2 * h)

    result = {}
    for k in keys:
        out, var = parse_key(k)
        if out not in idx:
            raise KeyError(f"derivative {k!r}: unknown output {out!r}")
        for v in var:
            if v not in coords:
                raise KeyError(f"derivative {k!r}: unknown coordinate {v!r}")
        result[k] = d(out, var)
    return result
