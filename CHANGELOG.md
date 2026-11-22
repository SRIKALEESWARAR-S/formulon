# Changelog

## [1.1.0] - 2026-11-08
### Added
- **One backend-switch mechanism for every framework** (`formulon._switch.BackendSwitch`): per-call > scope >
  global > environment > auto, explicit choices never fall back silently. Domains: `quantum`
  (pennylane|qiskit|numpy), `autodiff` (torch|jax|numpy), `jit` (numba). `formulon.configure(...)` sets all at once
  (validated atomically); `formulon.backend_status()` reports them.
- `formulon.pinn` autodiff switch: `derivatives / residuals / loss / fit / evaluate` with `backend=`; torch trainer
  (`train_pinn`), JAX derivatives and loss (`jax_utils`), NumPy PIELM least-squares solver (no framework needed).
- `PDEProblem.conditions()` (exact boundary/initial targets incl. initial velocity), `linear_operator()`.
- Legendre encoder/decoder now backend-switchable (previously PennyLane-only); `legendre_feature_expectation`;
  `Circuit.state()` on every backend.
- **Spec-driven development**: `specs/` (38 requirements), spec-tagged tests, `scripts/spec_trace.py` gate in CI,
  generated `docs/TRACEABILITY.md`.
### Changed
- README/docs rewritten; `[jax]` extra; QML engine gives a clear install hint and is documented as intentionally pinned.
### Fixed
- Poisson residual source term worked only with NumPy arrays (broke torch tensors with grad).
- Old `tests/test_quantum.py` required PennyLane for backend-neutral features; replaced by `test_legendre.py`.

### Added (earlier in 1.1.0)
- `formulon.pinn`: six benchmark problems with closed-form solutions (heat 1-D, wave 1-D,
  viscous Burgers via Cole-Hopf, Poisson 2-D, Kovasznay Navier-Stokes, damped oscillator),
  backend-agnostic residuals (NumPy finite differences / PyTorch autograd), `verify_problem`,
  Sobol/LHS sampling, noisy-observation datasets for inverse problems, standard metrics.
- `formulon.pinn.torch_utils` (optional `[ml]` extra): autograd derivatives, composite PINN loss, baseline MLP.
- `formulon.ml`: `vectorize()` lifts validated formulas to broadcast arrays with element-wise domain
  checks (`raise` or `mask` mode); `FormulaFeatures` for physics-derived feature columns.
- `PhysicalBounds.check_array`; `@validate` now exposes `__formulon_rules__`.
- CI workflow (Python 3.10-3.12), `examples/identify_damped_oscillator.py`, `docs/ML_PINN.md`.
- **Optional JIT**: `formulon.set_jit / jit_context / FORMULON_JIT`, per-call `jit=True` on JIT-capable
  functions and on `ml.vectorize`; numba is now an optional `[jit]` extra (it was a hard dependency and
  imported at module top-level), with automatic fallback to pure Python.
- **Quantum backend switch**: `set_backend / use_backend / available_backends`, `Circuit` IR running on
  PennyLane, Qiskit or a built-in NumPy simulator; 29 gates; half/full/ripple-carry/Draper-QFT adders;
  native PennyLane modules `pennylane_gates` and `pennylane_adders`; docs in `docs/QUANTUM_JIT.md`.
### Changed
- License is Apache-2.0 everywhere (LICENSE, pyproject, classifiers, SPDX headers); stale `PKG-INFO` removed.
- `formulon.quantum_computing` no longer imports PennyLane/Qiskit on import (lazy submodules).
### Fixed
- `grover_search` in `grover.py` lacked `success_probability`/`target_binary` and defaulted to 1 iteration
  (README and tests expected the optimal count); now backend-neutral with optimal iterations.
- Removed `Modernphysics.py` (case-colliding duplicate of `modernphysics.py` that imported a
  non-existent `validators` module; breaks checkouts on macOS/Windows).
- Removed stray file `tall -e .[quantum]` (pasted git diff) and shipped `.pytest_cache`.
- `tests/test_imports.py` asserted a stale version string.
- Removed duplicate publish workflow; README no longer contains leftover citation markers.
- QML dependencies (torch, scikit-learn, qiskit-aer) declared as the `[qml]` extra; `QML/` is a real package.

## [1.0.0] - 2026-09-24
- Initial release: 234 formula functions, catalog, quantum-computing demos, QML engine.
