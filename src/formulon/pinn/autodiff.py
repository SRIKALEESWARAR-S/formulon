"""Autodiff backend switch for PINNs: torch | jax | numpy.

    from formulon.pinn import set_autodiff, fit, evaluate, get_problem
    set_autodiff("torch")                       # or "jax", "numpy", "auto"
    model, hist = fit(get_problem("heat_1d"))   # gradient-based PINN on torch
    evaluate(get_problem("heat_1d"), model)     # rel-L2 vs the exact solution

numpy  : finite-difference residuals; ``fit`` uses the PIELM least-squares solver (linear PDEs)
torch  : autograd derivatives, Adam-trained MLP PINN
jax    : autodiff derivatives and losses (bring your own optimiser, e.g. optax)

Priority: per-call ``backend=`` > ``use_autodiff`` > ``set_autodiff`` > FORMULON_AUTODIFF_BACKEND > auto
(torch, then jax, then numpy).  Explicit choices never fall back silently.
"""
from __future__ import annotations

import numpy as np

from .._switch import BackendSwitch
from .derivatives import fd_derivatives, split_outputs

__all__ = ["set_autodiff", "get_autodiff", "use_autodiff", "available_autodiff", "resolve_autodiff",
           "derivatives", "residuals", "loss", "fit", "evaluate"]

_switch = BackendSwitch(
    "autodiff", ("torch", "jax", "numpy"), "FORMULON_AUTODIFF_BACKEND", ("torch", "jax", "numpy"),
    {"torch": "torch", "jax": "jax", "numpy": None},
    {"torch": 'pip install "formulon-physics[ml]"', "jax": 'pip install "formulon-physics[jax]"'})

set_autodiff = _switch.set
get_autodiff = _switch.get
use_autodiff = _switch.use
available_autodiff = _switch.available
resolve_autodiff = _switch.resolve


def derivatives(model, coords: dict, outputs, keys, backend: str | None = None, h: float = 1e-3) -> dict:
    """Named derivatives of ``model`` at ``coords`` using the selected backend.

    Returns arrays of the backend's own type (NumPy / torch tensors / JAX arrays).
    """
    b = resolve_autodiff(backend)
    if b == "numpy":
        return fd_derivatives(model, coords, outputs, keys, h)
    if b == "torch":
        from .torch_utils import as_tensors, autograd_derivatives
        return autograd_derivatives(model, as_tensors(coords), outputs, keys)
    from .jax_utils import jax_derivatives
    return jax_derivatives(model, coords, outputs, keys)


def residuals(problem, model, coords: dict, backend: str | None = None, h: float = 1e-3) -> dict:
    """PDE residuals of ``model`` on the selected backend."""
    b = resolve_autodiff(backend)
    d = derivatives(model, coords, problem.outputs, problem.derivative_keys, b, h)
    c = coords
    if b == "torch":
        from .torch_utils import as_tensors
        c = as_tensors(coords)
    return problem.residual_from_derivs(d, c)


def _numpy_loss(problem, model, collocation, boundary, initial, w_pde, w_bc, w_ic, h):
    res = problem.residual(model, collocation, h)
    parts = {"pde": float(sum(np.mean(np.square(r)) for r in res.values()))}
    total = w_pde * parts["pde"]
    for label, kind, pts, w in (("bc", "boundary", boundary, w_bc), ("ic", "initial", initial, w_ic)):
        if pts is None:
            continue
        targets = problem.conditions(kind, pts)
        d = fd_derivatives(model, pts, problem.outputs, list(targets), h)
        parts[label] = float(sum(np.mean((d[k] - np.asarray(v)) ** 2) for k, v in targets.items()))
        total += w * parts[label]
    parts["total"] = total
    return parts


def loss(problem, model, collocation: dict, boundary: dict | None = None, initial: dict | None = None,
         backend: str | None = None, w_pde: float = 1.0, w_bc: float = 10.0, w_ic: float = 10.0,
         h: float = 1e-3) -> dict:
    """Composite PINN loss (PDE residual + boundary + initial/velocity terms)."""
    b = resolve_autodiff(backend)
    if b == "numpy":
        return _numpy_loss(problem, model, collocation, boundary, initial, w_pde, w_bc, w_ic, h)
    if b == "torch":
        from .torch_utils import pinn_loss
        return pinn_loss(problem, model, collocation, boundary, initial, w_pde, w_bc, w_ic)
    from .jax_utils import pinn_loss as jax_loss
    return jax_loss(problem, model, collocation, boundary, initial, w_pde, w_bc, w_ic)


def fit(problem, backend: str | None = None, **kwargs):
    """Train a model for ``problem`` on the selected backend.

    numpy -> ``(PIELMModel, history)`` (linear problems; kwargs of ``pielm_fit``)
    torch -> ``(MLP, history)``       (kwargs of ``torch_utils.train_pinn``)
    jax   -> NotImplementedError: use ``loss(..., backend='jax')`` with your own optimiser.
    """
    b = resolve_autodiff(backend)
    if b == "numpy":
        from .pielm import pielm_fit
        m = pielm_fit(problem, **kwargs)
        return m, m.history
    if b == "torch":
        from .torch_utils import train_pinn
        return train_pinn(problem, **kwargs)
    raise NotImplementedError("jax backend provides derivatives and losses; plug them into optax/your optimiser")


def evaluate(problem, model, n: int = 2000, seed: int = 123, backend: str | None = None) -> dict:
    """Relative L2 error of ``model`` against the exact solution on random interior points."""
    from .metrics import relative_l2
    from .sampling import sample_interior

    b = resolve_autodiff(backend)
    pts = sample_interior(problem.domain, n, "uniform", seed)
    exact = problem.exact_dict(**pts)
    names = list(problem.domain)
    if b == "torch":
        import torch
        with torch.no_grad():
            out = model(*[torch.as_tensor(pts[k], dtype=next(model.parameters()).dtype) for k in names])
        pred = {k: np.asarray(v.detach().cpu().numpy(), dtype=float) for k, v in split_outputs(out, problem.outputs).items()}
    else:
        out = model(*[pts[k] for k in names])
        pred = {k: np.asarray(v, dtype=float) for k, v in split_outputs(out, problem.outputs).items()}
    per = {o: relative_l2(pred[o], exact[o]) for o in problem.outputs}
    allp = np.concatenate([pred[o] for o in problem.outputs])
    alle = np.concatenate([exact[o] for o in problem.outputs])
    return {"rel_l2": relative_l2(allp, alle), "per_output": per}
