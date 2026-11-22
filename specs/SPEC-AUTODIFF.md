# SPEC-AUTODIFF: PINN autodiff backends
- **REQ-AD-001** `derivatives`, `residuals`, `loss`, `fit`, `evaluate` take `backend=` and follow the switch rules of SPEC-BACKENDS.
- **REQ-AD-002** On `numpy`, the loss of the exact solution is ~0, a wrong-physics model has a large loss, and the initial-velocity term is enforced for second-order-in-time problems.
- **REQ-AD-003** `fit(problem, backend="numpy")` (PIELM) recovers heat, wave, Poisson and damped-oscillator solutions to relative L2 < 1e-4 deterministically; nonlinear problems raise `NotImplementedError` pointing to torch.
- **REQ-AD-004** On `torch`, autograd derivatives equal finite-difference derivatives, the loss is differentiable, and `fit` reduces the loss (requires torch).
- **REQ-AD-005** On `jax`, derivatives and loss agree with the numpy backend (requires jax).
- **REQ-AD-006** The same problem object yields the same residual of the exact solution (~0) on every available backend.
