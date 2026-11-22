# Formulon for AI / ML / PINNs

## Problem statements addressed
| # | Existing problem | What Formulon 1.1 provides |
|---|---|---|
| 1 | PINN results are not comparable: each paper re-implements the PDE, exact solution and metric; residual bugs go unnoticed | `formulon.pinn` problems with closed-form solutions, one shared residual, `verify_problem()` self-check, standard metrics (rel-L2, max, RMSE) |
| 2 | Inverse problems / system identification need reproducible noisy data with known ground truth | `make_dataset(noise_std, seed)` + `parameter_error` (see `examples/identify_damped_oscillator.py`) |
| 3 | Surrogates must respect physics, but formula libraries are scalar-only and unusable on batches | `formulon.ml.vectorize` (element-wise domain checks, NaN-mask mode) and `FormulaFeatures` for physics-derived inputs |
| 4 | Real materials/geometry need to enter benchmarks | `heat_slab_from_material(k, rho, cp, L)` |

## Autodiff backend switch (torch | jax | numpy)
```python
from formulon.pinn import set_autodiff, use_autodiff, fit, evaluate, loss, residuals, derivatives, get_problem
set_autodiff("torch")                       # or "jax" / "numpy" / "auto"
model, hist = fit(get_problem("heat_1d"))   # Adam-trained MLP PINN (torch)
evaluate(get_problem("heat_1d"), model)     # relative L2 vs the exact solution
with use_autodiff("numpy"):                 # PIELM: one least-squares solve, linear PDEs, no framework needed
    model, info = fit(get_problem("wave_1d"))
residuals(problem, my_model, coords, backend="jax")   # per-call override
```
| backend | derivatives | `fit` |
|---|---|---|
| numpy | finite differences | PIELM (heat, wave, Poisson, damped oscillator; nonlinear -> NotImplementedError) |
| torch | autograd | Adam MLP PINN (`train_pinn`) incl. Burgers and Kovasznay |
| jax | `jax.grad`/`vmap` | not provided: use `loss(..., backend="jax")` with optax or your optimiser; enable `jax_enable_x64` |

Second-order-in-time problems (wave, oscillator) automatically get an initial-velocity condition.
torch/jax residuals are CPU-tested paths; GPU tensors are not supported for the Poisson source term.

## Problems
`heat_1d`, `wave_1d`, `burgers_1d`, `poisson_2d`, `kovasznay` (u, v, p), `damped_oscillator`.
Each reports `residual(model, coords)` (NumPy finite differences, h=1e-3) or
`residual_from_derivs(derivs, coords)` (torch autograd tensors via `formulon.pinn.torch_utils`).

## Verification status
- Exact solutions satisfy their residuals to RMS 1e-10 ... 1e-5 (`verify_problem`, tol 1e-4).
- Negative control: wrong diffusivity gives residual RMS ~0.57 — the check does detect wrong physics.
- Inverse example recovers c, k within 0.1% from 3% noise.
- PIELM recovers heat, wave, Poisson and oscillator to relative L2 ~1e-7 (test threshold 1e-4); the numpy loss is ~0
  for exact solutions and large for wrong physics, including a wrong initial velocity.
- **Not verified in the build sandbox:** torch and JAX code paths (`torch_utils`, `jax_utils`, `train_pinn`) and the QML engine -
  the packages were not installable offline. Their tests (`test_autodiff.py`) compare autograd/JAX residuals with the
  NumPy finite-difference reference and skip when absent: run `pytest` with `[all,dev]` installed before releasing.

## Limits
- Finite-difference residuals are for *verification*; they carry O(h^2) error and are not for training.
- Benchmarks are smooth analytic cases; they do not replace PDEBench/PINNacle-style stiff or 3-D cases.
- Educational/research software, not certified for safety-critical design.
