# SPEC-PKG: packaging
- **REQ-PKG-001** `import formulon`, `formulon.ml`, `formulon.pinn` and `formulon.quantum_computing` succeed with only NumPy/SciPy(+pint) installed; torch, jax, numba, pennylane, qiskit are optional.
- **REQ-PKG-002** The license is Apache-2.0 in `LICENSE`, `pyproject.toml` (field + classifier) and every SPDX header; "MIT" appears nowhere in source or metadata.
- **REQ-PKG-003** No two files under `src/` differ only by letter case (breaks macOS/Windows checkouts).
- **REQ-PKG-004** The version string has a single source (`__about__.py`) and `formulon.__version__` equals it.
