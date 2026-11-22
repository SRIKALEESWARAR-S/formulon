"""Single entry point for all framework switches.

    import formulon
    formulon.configure(quantum="pennylane", autodiff="torch", jit=True)
    formulon.backend_status()

Domains
-------
quantum   pennylane | qiskit | numpy     (circuits, adders, Grover, Legendre encoder)
autodiff  torch | jax | numpy            (PINN derivatives, losses, training)
jit       numba on/off                   (numerical kernels, vectorize)

Each domain also has its own commands, e.g. ``quantum_computing.set_backend`` and
``pinn.set_autodiff``; environment variables FORMULON_QUANTUM_BACKEND,
FORMULON_AUTODIFF_BACKEND and FORMULON_JIT set the defaults.
"""
from __future__ import annotations

from . import jit as _jit

__all__ = ["configure", "backend_status", "reset_backends"]


def _domains():
    from .quantum_computing.backend import _switch as q
    from .pinn.autodiff import _switch as a
    return {"quantum": q, "autodiff": a}


def configure(quantum: str | None = None, autodiff: str | None = None, jit: bool | None = None) -> dict:
    """Set any combination of backends in one call; returns :func:`backend_status`.

    The arguments are validated *before* anything is changed, so a bad value
    leaves the previous configuration intact.
    """
    doms = _domains()
    new = {"quantum": quantum, "autodiff": autodiff}
    for k, v in new.items():
        if v is not None:
            doms[k]._check(v)
    if jit not in (None, True, False):
        raise ValueError("jit must be True, False or None")
    for k, v in new.items():
        if v is not None:
            doms[k].set(v)
    if jit is not None:
        _jit.set_jit(jit)
    return backend_status()


def backend_status() -> dict:
    """What is configured and what is actually installed, per domain."""
    doms = _domains()
    out = {}
    for name, sw in doms.items():
        pref = sw.get()
        out[name] = {"preferred": pref, "resolves_to": _safe(sw, pref), "available": sw.available()}
    out["jit"] = {"enabled": _jit.get_jit(), "numba_installed": _jit.numba_available()}
    return out


def _safe(sw, pref):
    try:
        return sw.resolve(pref)
    except ImportError as exc:
        return f"unavailable ({exc})"


def reset_backends() -> None:
    """Back to defaults (env vars / auto)."""
    for sw in _domains().values():
        sw.reset()
    _jit.set_jit(None)
