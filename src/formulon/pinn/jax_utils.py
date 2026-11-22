"""Optional JAX bridge (autodiff derivatives + composite PINN loss).

Requires ``pip install "formulon-physics[jax]"``.  Enable float64 for PDE work:
``jax.config.update("jax_enable_x64", True)``.  The model takes batched coordinate arrays
``model(x, t, ...)`` and returns an array (N,), (N, n_out) or a tuple of arrays.
"""
from __future__ import annotations

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover
    raise ImportError('JAX is required: pip install "formulon-physics[jax]"') from exc

from .derivatives import parse_key, split_outputs

__all__ = ["jax_derivatives", "pinn_loss"]


def jax_derivatives(model, coords: dict, outputs, keys) -> dict:
    names = list(coords)
    arrs = [jnp.asarray(coords[n]) for n in names]
    pos = {o: i for i, o in enumerate(outputs)}

    def scalar_fn(out_name):
        def f(*xs):
            res = split_outputs(model(*[x[None] for x in xs]), outputs)
            return jnp.reshape(res[out_name], ())
        return f

    result = {}
    for key in keys:
        out, var = parse_key(key)
        if out not in pos:
            raise KeyError(f"derivative {key!r}: unknown output {out!r}")
        f = scalar_fn(out)
        for v in var:
            f = jax.grad(f, argnums=names.index(v))
        result[key] = jax.vmap(f)(*arrs)
    return result


def pinn_loss(problem, model, collocation, boundary=None, initial=None,
              w_pde: float = 1.0, w_bc: float = 10.0, w_ic: float = 10.0) -> dict:
    parts = {}
    col = {k: jnp.asarray(v) for k, v in collocation.items()}
    d = jax_derivatives(model, col, problem.outputs, problem.derivative_keys)
    res = problem.residual_from_derivs(d, col)
    parts["pde"] = sum(jnp.mean(r**2) for r in res.values())
    total = w_pde * parts["pde"]
    for label, kind, pts, w in (("bc", "boundary", boundary, w_bc), ("ic", "initial", initial, w_ic)):
        if pts is None:
            continue
        p = {k: jnp.asarray(v) for k, v in pts.items()}
        targets = problem.conditions(kind, pts)
        dd = jax_derivatives(model, p, problem.outputs, list(targets))
        parts[label] = sum(jnp.mean((dd[k] - jnp.asarray(v)) ** 2) for k, v in targets.items())
        total = total + w * parts[label]
    parts["total"] = total
    return parts
