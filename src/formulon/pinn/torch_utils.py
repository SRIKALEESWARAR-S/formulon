"""Optional PyTorch bridge: autograd derivatives, composite PINN loss, MLP and a trainer.

Requires ``pip install "formulon-physics[ml]"``.  The residual formulas are the *same*
functions used by the NumPy path, so a trained PINN can be checked against the NumPy
finite-difference residual and the exact solution (see ``autodiff.evaluate``).
"""
from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError('PyTorch is required: pip install "formulon-physics[ml]"') from exc

from .derivatives import parse_key, split_outputs

__all__ = ["autograd_derivatives", "pinn_loss", "as_tensors", "MLP", "train_pinn"]


class MLP(torch.nn.Module):
    """tanh MLP mapping coordinates -> outputs.  If ``domain`` is given, inputs are
    scaled to [-1, 1] inside the network (strongly recommended for training)."""

    def __init__(self, n_in: int, n_out: int, width: int = 64, depth: int = 4, domain: dict | None = None):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.Tanh()]
            d = width
        layers.append(torch.nn.Linear(d, n_out))
        self.net = torch.nn.Sequential(*layers)
        if domain is not None:
            lo = torch.tensor([v[0] for v in domain.values()], dtype=torch.float64)
            hi = torch.tensor([v[1] for v in domain.values()], dtype=torch.float64)
            self.register_buffer("_c", (lo + hi) / 2)
            self.register_buffer("_h", (hi - lo) / 2)
        else:
            self._c = self._h = None

    def forward(self, *coords):
        z = torch.stack(coords, dim=-1)
        if self._c is not None:
            z = (z - self._c.to(z.dtype)) / self._h.to(z.dtype)
        return self.net(z)


def as_tensors(coords: dict, requires_grad: bool = True, dtype=torch.float64, device=None) -> dict:
    return {k: torch.as_tensor(np.asarray(v), dtype=dtype, device=device).clone().requires_grad_(requires_grad)
            for k, v in coords.items()}


def autograd_derivatives(model, coords: dict, outputs, keys) -> dict:
    """Exact derivatives through autograd. ``coords`` tensors need requires_grad=True."""
    base = split_outputs(model(*coords.values()), outputs)
    result = {}
    for key in keys:
        out, var = parse_key(key)
        g = base[out]
        for v in var:
            g = torch.autograd.grad(g, coords[v], torch.ones_like(g), create_graph=True)[0]
        result[key] = g
    return result


def pinn_loss(problem, model, collocation: dict, boundary: dict | None = None,
              initial: dict | None = None, w_pde: float = 1.0, w_bc: float = 10.0,
              w_ic: float = 10.0) -> dict:
    """PDE residual + boundary + initial (value and, if needed, velocity) loss.  Returns a dict
    with 'total' (differentiable) and the individual parts."""
    parts = {}
    col = as_tensors(collocation)
    d = autograd_derivatives(model, col, problem.outputs, problem.derivative_keys)
    res = problem.residual_from_derivs(d, col)
    parts["pde"] = sum(torch.mean(r**2) for r in res.values())
    total = w_pde * parts["pde"]
    for label, kind, pts, w in (("bc", "boundary", boundary, w_bc), ("ic", "initial", initial, w_ic)):
        if pts is None:
            continue
        targets = problem.conditions(kind, pts)
        t = as_tensors(pts)
        dd = autograd_derivatives(model, t, problem.outputs, list(targets))
        parts[label] = sum(torch.mean((dd[k] - torch.as_tensor(np.asarray(v), dtype=dd[k].dtype)) ** 2)
                           for k, v in targets.items())
        total = total + w * parts[label]
    parts["total"] = total
    return parts


def train_pinn(problem, epochs: int = 3000, lr: float = 1e-3, width: int = 64, depth: int = 4,
               n_collocation: int = 2048, n_boundary: int = 256, seed: int = 0, w_pde: float = 1.0,
               w_bc: float = 10.0, w_ic: float = 10.0, model=None, log_every: int = 0):
    """Adam-train an MLP PINN.  Returns ``(model, history)`` with the loss per epoch."""
    from .sampling import make_dataset

    torch.manual_seed(seed)
    data = make_dataset(problem, n_collocation=n_collocation, n_boundary=n_boundary,
                        n_observations=8, seed=seed)
    if model is None:
        model = MLP(len(problem.domain), len(problem.outputs), width, depth, problem.domain).double()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for ep in range(epochs):
        opt.zero_grad()
        parts = pinn_loss(problem, model, data.collocation, data.boundary, data.initial, w_pde, w_bc, w_ic)
        parts["total"].backward()
        opt.step()
        hist.append(float(parts["total"].detach()))
        if log_every and ep % log_every == 0:
            print(f"epoch {ep:5d}  loss {hist[-1]:.3e}")
    return model, hist
