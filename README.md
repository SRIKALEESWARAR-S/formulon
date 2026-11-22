# Formulon Physics 1.1
## What's new in Formulon 1.0

Formulon 1.0 introduces the Quantum-Informed Physical AI Engine, a
hybrid quantum-classical research module for learning mappings between
physical feature representations and noisy quantum measurements.

### Core capabilities

- Classical physics computation
- Quantum-computing modules
- Physics-informed machine learning
- Quantum-informed physical AI
- Noisy quantum simulation
- Physical feature extraction
- Reconstruction and classification
- Interactive visualizations
- Modular Python API

Formulon Physics is a modular computational-physics library containing **234 public scientific formula functions** across classical mechanics, electromagnetism, waves/optics, thermodynamics, fluids, modern physics, mathematical physics and series. It also provides optional PennyLane/Qiskit quantum-computing examples.

## One switch for every framework (new in 1.1)

Every optional framework is selected the same way: **per call > scope > global setting > environment variable > auto**.
An explicit choice never silently falls back; a missing package raises `ImportError` with the install command.

| Domain | Choices | Used for | Commands |
|---|---|---|---|
| quantum | `pennylane` `qiskit` `numpy` | circuits, gates, adders, Grover, Legendre encoder | `set_backend`, `use_backend`, `backend=` |
| autodiff | `torch` `jax` `numpy` | PINN derivatives, losses, training | `set_autodiff`, `use_autodiff`, `backend=` |
| jit | numba on/off | numerical kernels, `vectorize` | `set_jit`, `jit_context`, `jit=` |

```python
import formulon
formulon.configure(quantum="pennylane", autodiff="torch", jit=True)   # all at once, validated atomically
formulon.backend_status()      # what is configured, what it resolves to, what is installed
```
Environment defaults: `FORMULON_QUANTUM_BACKEND`, `FORMULON_AUTODIFF_BACKEND`, `FORMULON_JIT`.
The QML engine (`formulon.QML`) is the one deliberate exception: it is pinned to Qiskit-Aer noise + PennyLane/torch.

## AI / ML / PINN work

```python
from formulon.pinn import get_problem, fit, evaluate, set_autodiff

p = get_problem("heat_1d")                    # closed-form reference solution included
model, info = fit(p, backend="numpy")        # NumPy-only physics-informed solver (no torch needed)
print(evaluate(p, model, backend="numpy"))    # {'rel_l2': ~1e-8, ...}

set_autodiff("torch")                         # same API, gradient-based PINN
model, loss_history = fit(get_problem("burgers_1d"), epochs=5000)
```
Six benchmarks (heat, wave, Burgers, Poisson, Kovasznay Navier-Stokes, damped oscillator), `verify_problem()`,
Sobol/LHS sampling, noisy datasets for inverse problems, standard metrics, `vectorize()` for array-valued formulas and
`FormulaFeatures` for physics-derived ML features.  See `docs/ML_PINN.md` and `examples/`.

## Quantum backends, gates, adders

```python
from formulon.quantum_computing import set_backend, add, Circuit
set_backend("pennylane")                    # or "qiskit" / "numpy" / "auto"
print(add(5, 9, n_bits=4))                  # 14, quantum ripple-carry adder
Circuit(2).h(0).cx(0, 1).run(backend="qiskit")
```
Details: `docs/QUANTUM_JIT.md`.

## Optional Numba JIT (off by default)

```python
formulon.legendre_polynomial(20, 0.2, jit=True)      # pip install "formulon-physics[jit]"
```

## Development process: Claude + spec-driven development

This package is developed with **Claude (Anthropic's AI assistant) used under spec-driven development**.
Requirements are written first as testable specs with IDs (`specs/`), tests are written against them and tagged with
the IDs, Claude implements against those tests, and `scripts/spec_trace.py` (run in CI) fails if any requirement lacks a
test. `docs/TRACEABILITY.md` is the generated requirement-to-test matrix. The maintainer owns goals and decisions;
the AI proposes, implements and tests. Anything that could not be verified in the build environment is stated as
such in the docs rather than claimed.

## Design principles

1. **Formula first:** functions implement familiar textbook/engineering relations.
2. **Boundary-aware:** inputs are checked for mathematical validity and common model assumptions.
3. **No fake universal limits:** classical quantities such as velocity and gravity are not capped at arbitrary values. Relativistic limits are applied only where the model requires them.
4. **Units are explicit:** numeric APIs use SI values; optional Pint helpers provide dimensionality and conversion.
5. **Documented assumptions:** formulas state their model and intended use.
5. **Optional frameworks:** torch, JAX, numba, PennyLane and Qiskit are never required by the core package.
6. **One switch:** every optional framework is chosen with the same rules (see above).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install .
```

Extras: `[units]` `[pennylane]` `[qiskit]` `[quantum]` `[jit]` (numba) `[ml]` (torch) `[jax]` `[qml]` (QML engine) `[dev]` (pytest) `[all]`.

```bash
pip install '.[all]'          # everything
pip install '.[ml,jit]'       # e.g. torch PINNs with numba
```

## Unit-aware workflow

```python
from formulon.units import quantity, to_si

length = quantity(5, 'ft')
time = quantity(2, 's')
speed = length / time
print(to_si(speed))
```

Pint is used for explicit unit conversion/dimensionality rather than silently changing the numeric API. Pint supports quantities, conversions and dimensionality checks.

## Quantum examples

### Grover search

```python
from formulon.quantum_computing import grover_search

result = grover_search(3456, num_qubits=12)
print(result.target_binary, result.found_state, result.success_probability)
```

### Legendre quantum encoder/decoder

The encoder maps x to the Legendre modes P_0(x), ..., P_{d-1}(x) and uses RY(arccos(P_n)) rotations, so ideally <Z> = P_n. Finite-shot computational-basis measurement gives sampled outcomes; the decoder reconstructs the classical coordinate from the estimated modes. Both functions accept `backend=` (PennyLane, Qiskit or NumPy).

PennyLane is an open-source platform for quantum computing and QML; Formulon's quantum layer is intentionally optional.

## Boundary philosophy

A bound belongs in the validator only when it is a mathematical requirement or a clearly stated model assumption. Examples:

- mass > 0 for formulas that divide by mass;
- radius > 0 where \(1/r\) occurs;
- absolute temperature \(T \ge 0\) K;
- volume > 0 for ideal-gas relations;
- emissivity \(0\le\epsilon\le1\);
- angles restricted only where the formula's geometry requires it;
- relativistic speed \(|v|<c\) only for relativistic formulas.

Classical velocity is **not** globally capped at \(c\), because the classical formula itself is not a relativistic model.

## Formula catalog

`formulon.formula_catalog` exposes machine-readable metadata for all 234 public formula functions:

```python
from formulon.formula_catalog import FORMULA_COUNT, get_formula
print(FORMULA_COUNT)
print(get_formula('projectile_range'))
```

The catalog records the module, signature, validation rules, description and primary real-world use case.

## Testing

```bash
pip install -e '.[dev]' && pytest && python scripts/spec_trace.py
```
The suite covers every spec requirement. Tests that need torch, JAX, numba, PennyLane or Qiskit skip automatically when the package is absent, so run the full suite in an environment with all extras (`pip install -e '.[all,dev]'`) before publishing a release.

## Scope

This is scientific/educational software, not a substitute for validated engineering codes, safety-critical design, medical decisions or certified numerical simulation. Every formula should be used within its documented assumptions.
