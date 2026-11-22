"""Array-valued access to Formulon's scalar formulas.

Every formula in Formulon is scalar-only: the validator rejects NumPy arrays.
That is safe, but unusable for ML work (batches, grids, Monte-Carlo samples).
``vectorize`` lifts any validated formula to broadcastable arrays while
keeping the *same* domain rules, checked element-wise, and reports which
rows are out of domain instead of failing on the first one when asked.
"""
from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..jit import _resolve, numba_vectorize, _is_numba_error, _get_numba, _warn_once

__all__ = ["vectorize", "vectorized_catalog", "ValidationReport"]


@dataclass(frozen=True)
class ValidationReport:
    """Result of ``vectorize(f, on_invalid='mask')`` calls (``.report``)."""
    n_total: int
    n_invalid: int
    invalid_mask: np.ndarray

    @property
    def valid_fraction(self) -> float:
        return 1.0 - self.n_invalid / max(self.n_total, 1)


def _broadcast(bound_args: dict, names):
    arrays = {k: np.asarray(bound_args[k]) for k in names}
    shape = np.broadcast_shapes(*[a.shape for a in arrays.values()]) if arrays else ()
    return {k: np.broadcast_to(a, shape) for k, a in arrays.items()}, shape


def vectorize(func: Callable, on_invalid: str = "raise", jit=None) -> Callable:
    """Return an array-aware version of a Formulon formula.

    Parameters
    ----------
    func : a formula from ``formulon`` (decorated with ``@validate``).
    on_invalid : ``"raise"`` (default) raises ``ValueError`` naming the first
        offending parameter; ``"mask"`` returns NaN for out-of-domain rows and
        stores a :class:`ValidationReport` on ``wrapper.report`` — handy when
        screening randomly sampled design spaces.

    jit : None (follow ``formulon.set_jit`` / ``FORMULON_JIT``), True or False.
        When on and numba is installed, the formula body is compiled to a numba
        ufunc (fast on large batches).  It can also be overridden per call:
        ``f(x, ..., jit=True)``.  If numba is missing or compilation fails the
        NumPy path is used and a warning is issued once.

    Scalar-only parameters (callables, lists meant as vectors) are passed
    through untouched; only parameters that have a validation rule are
    broadcast.
    """
    if on_invalid not in ("raise", "mask"):
        raise ValueError("on_invalid must be 'raise' or 'mask'")
    rules = getattr(func, "__formulon_rules__", None)
    if rules is None:
        raise TypeError(f"{getattr(func, '__name__', func)!r} is not a validated Formulon formula")
    raw = getattr(func, "__wrapped__", func)
    sig = inspect.signature(raw)
    jit_ok = set(sig.parameters) == set(rules) and "jit" not in sig.parameters
    box = {"kernel": None, "failed": False}

    @functools.wraps(raw)
    def wrapper(*args, **kwargs):
        call_jit = kwargs.pop("jit", jit) if "jit" not in sig.parameters else jit
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        values, shape = _broadcast(bound.arguments, rules)
        invalid = np.zeros(shape, dtype=bool)
        clean = {}
        for name, rule in rules.items():
            arr = np.asarray(values[name])
            try:
                clean[name] = rule.check_array(arr, name)
            except ValueError:
                if on_invalid == "raise":
                    raise
                # element-wise fallback to locate invalid rows
                a = arr.astype(float)
                ok = np.isfinite(a)
                if rule.min_val is not None:
                    ok &= (a > rule.min_val) if rule.strict_min else (a >= rule.min_val)
                if rule.max_val is not None:
                    ok &= (a < rule.max_val) if rule.strict_max else (a <= rule.max_val)
                invalid |= ~ok
                clean[name] = a
        call_args = dict(bound.arguments)
        call_args.update(clean)

        out = None
        if not invalid.any():
            try:  # fast path: formula body works on arrays directly
                if jit_ok and _resolve(call_jit) and not box["failed"]:
                    try:
                        if box["kernel"] is None:
                            box["kernel"] = numba_vectorize(raw)
                            if box["kernel"] is None:
                                box["failed"] = True
                        if box["kernel"] is not None:
                            out = box["kernel"](*[np.asarray(call_args[n], dtype=float) for n in sig.parameters])
                    except Exception as exc:  # noqa: BLE001
                        if not _is_numba_error(_get_numba(), exc):
                            raise
                        box["failed"] = True
                        _warn_once("vfail:" + raw.__name__, f"formulon: numba could not compile {raw.__name__}; using NumPy path")
                        out = None
                if out is None:
                    with np.errstate(all="raise"):
                        out = raw(**call_args)
                out = np.asarray(out, dtype=float)
                if out.shape != shape and out.ndim == 0:
                    out = np.broadcast_to(out, shape).copy()
                if out.shape != shape:
                    out = None
            except (TypeError, ValueError, FloatingPointError, ZeroDivisionError):
                out = None
        if out is None:  # slow path: element-wise on valid rows
            out = np.full(shape, np.nan, dtype=float)
            flat_idx = np.ndindex(*shape) if shape else [()]
            for idx in flat_idx:
                if invalid[idx]:
                    continue
                kw = dict(call_args)
                for n in rules:
                    kw[n] = float(clean[n][idx])
                out[idx] = raw(**kw)
        if on_invalid == "mask":
            out = np.where(invalid, np.nan, out)
            wrapper.report = ValidationReport(int(invalid.size), int(invalid.sum()), invalid)
        elif not np.all(np.isfinite(out)):
            raise FloatingPointError(f"{raw.__name__} returned non-finite values")
        return out

    wrapper.report = None
    return wrapper


def vectorized_catalog(names=None, on_invalid: str = "raise") -> dict:
    """Map formula name -> vectorised callable for the whole package (or a subset)."""
    import formulon
    from formulon.formula_catalog import FORMULAS

    wanted = set(names) if names is not None else None
    out = {}
    for info in FORMULAS:
        if wanted is not None and info.name not in wanted:
            continue
        fn = getattr(formulon, info.name, None)
        if fn is not None and hasattr(fn, "__formulon_rules__"):
            out[info.name] = vectorize(fn, on_invalid=on_invalid)
    return out
