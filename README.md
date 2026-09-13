# Formulon Physics 0.6.6

Formulon Physics is a modular computational-physics library containing **234 public scientific formula functions** across classical mechanics, electromagnetism, waves/optics, thermodynamics, fluids, modern physics, mathematical physics and series. It also provides optional PennyLane/Qiskit quantum-computing examples.

## Design principles

1. **Formula first:** functions implement familiar textbook/engineering relations.
2. **Boundary-aware:** inputs are checked for mathematical validity and common model assumptions.
3. **No fake universal limits:** classical quantities such as velocity and gravity are not capped at arbitrary values. Relativistic limits are applied only where the model requires them.
4. **Units are explicit:** numeric APIs use SI values; optional Pint helpers provide dimensionality and conversion.
5. **Documented assumptions:** formulas state their model and intended use.
6. **Optional quantum stack:** PennyLane and Qiskit are not imported by the core package.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install .
```

With units and quantum examples:

```bash
pip install '.[all]'
```

Or individually:

```bash
pip install '.[units]'
pip install '.[pennylane]'
```

## Unit-aware workflow

```python
from formulon.units import quantity, to_si

length = quantity(5, 'ft')
time = quantity(2, 's')
speed = length / time
print(to_si(speed))
```

Pint is used for explicit unit conversion/dimensionality rather than silently changing the numeric API. Pint supports quantities, conversions and dimensionality checks. citeturn0search0turn0search2

## Quantum examples

### Grover search

```python
from formulon.quantum_computing.grover import grover_search

result = grover_search(3456, num_qubits=12)
print(result.target_binary, result.found_state, result.success_probability)
```

### Legendre quantum encoder/decoder

The encoder maps \(x\) to \(P_0(x),...,P_{d-1}(x)\), uses \(RY(rccos(P_n))\), and therefore has \(\langle Z
angle=P_n\) ideally. Finite-shot computational-basis measurement gives sampled outcomes; the decoder reconstructs the classical coordinate from estimated Legendre features.

PennyLane is an open-source platform for quantum computing and QML; Formulon's quantum layer is intentionally optional. citeturn0search5

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

The release includes import tests, formula-catalog tests and optional quantum tests. A release should be tested in a clean virtual environment on the target Linux/Python versions before publication.

## Scope

This is scientific/educational software, not a substitute for validated engineering codes, safety-critical design, medical decisions or certified numerical simulation. Every formula should be used within its documented assumptions.
