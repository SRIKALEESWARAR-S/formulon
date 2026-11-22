"""Optional, dynamic Numba JIT for Formulon.

Numba is *not* required.  JIT is off by default and can be switched on at
four levels (highest priority first):

1. per call:        ``legendre_polynomial(10, 0.3, jit=True)``
2. temporary scope: ``with formulon.jit_context(True): ...``
3. process-wide:    ``formulon.set_jit(True)``
4. environment:     ``FORMULON_JIT=1``

If numba is missing, or a function cannot be compiled, Formulon warns once and
silently runs the pure-Python/NumPy path, so results never depend on JIT.
Install with ``pip install "formulon-physics[jit]"``.
"""
from __future__ import annotations

import contextlib
import functools
import os
import threading
import warnings

__all__ = ["set_jit", "get_jit", "jit_context", "numba_available", "lazy_njit", "numba_vectorize"]

_state = threading.local()
_global = {"flag": None}
_warned: set = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)


def _get_numba():
    try:
        import numba  # noqa: WPS433
        return numba
    except ImportError:
        return None


def numba_available() -> bool:
    return _get_numba() is not None


def set_jit(flag) -> None:
    """Process-wide default: True, False, or None (defer to FORMULON_JIT)."""
    if flag not in (True, False, None):
        raise ValueError("flag must be True, False or None")
    _global["flag"] = flag


def get_jit() -> bool:
    """Effective setting when no per-call override is given."""
    local = getattr(_state, "flag", None)
    if local is not None:
        return local
    if _global["flag"] is not None:
        return _global["flag"]
    return os.environ.get("FORMULON_JIT", "").strip().lower() in ("1", "true", "yes", "on")


@contextlib.contextmanager
def jit_context(flag: bool = True):
    """Temporarily enable/disable JIT for the current thread."""
    prev = getattr(_state, "flag", None)
    _state.flag = bool(flag)
    try:
        yield
    finally:
        _state.flag = prev


def _resolve(jit) -> bool:
    return get_jit() if jit is None else bool(jit)


def _is_numba_error(nb, exc) -> bool:
    errs = getattr(nb, "errors", None)
    base = getattr(errs, "NumbaError", None)
    return base is not None and isinstance(exc, base)


def lazy_njit(fn=None, *, nopython: bool = True, cache: bool = False):
    """Decorator: a function that is compiled by numba only when JIT is requested.

    The wrapper accepts an extra ``jit=None`` keyword.  Errors raised by the
    function itself propagate unchanged; only numba *compilation* errors trigger
    the Python fallback.
    """
    def deco(f):
        box = {"compiled": None, "failed": False}

        @functools.wraps(f)
        def wrapper(*args, jit=None, **kwargs):
            if _resolve(jit) and not box["failed"]:
                nb = _get_numba()
                if nb is None:
                    _warn_once("nonumba", "formulon: JIT requested but numba is not installed; "
                                          "running pure Python. pip install 'formulon-physics[jit]'")
                else:
                    try:
                        if box["compiled"] is None:
                            box["compiled"] = (nb.njit(cache=cache)(f) if nopython
                                               else nb.jit(nopython=False, cache=cache)(f))
                        return box["compiled"](*args, **kwargs)
                    except Exception as exc:  # noqa: BLE001
                        if not _is_numba_error(nb, exc):
                            raise
                        box["failed"] = True
                        _warn_once("fail:" + f.__name__,
                                   f"formulon: numba could not compile {f.__name__} ({type(exc).__name__}); "
                                   "using pure Python")
            return f(*args, **kwargs)

        wrapper.__wrapped_python__ = f
        return wrapper

    return deco(fn) if callable(fn) else deco


def numba_vectorize(raw):
    """Return a numba DUFunc for scalar function ``raw`` or None (numba missing)."""
    nb = _get_numba()
    if nb is None:
        _warn_once("nonumba", "formulon: JIT requested but numba is not installed; "
                              "running pure Python. pip install 'formulon-physics[jit]'")
        return None
    return nb.vectorize(nopython=True)(raw)
