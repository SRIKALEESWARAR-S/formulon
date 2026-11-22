# Quantum backends, gates, adders and optional JIT

## Choose PennyLane or Qiskit in one line
```python
from formulon.quantum_computing import set_backend, use_backend, available_backends, add, Circuit

available_backends()          # e.g. ['pennylane', 'qiskit', 'numpy']
set_backend("pennylane")      # 'pennylane' | 'qiskit' | 'numpy' | 'auto'  (auto = first installed)
add(5, 9, n_bits=4)           # 14, ripple-carry adder on PennyLane
add(5, 9, n_bits=4, method="qft")        # Draper QFT adder (mod 2**4)
with use_backend("qiskit"):   # temporary switch
    add(5, 9, n_bits=4)
Circuit(2).h(0).cx(0, 1).run(backend="qiskit")   # per-call override
```
Priority: per-call > `use_backend` > `set_backend` > env `FORMULON_QUANTUM_BACKEND` > auto.
An explicit choice never silently falls back; a missing package raises `ImportError` with the install command.

## Everything at once
```python
import formulon
formulon.configure(quantum="qiskit", autodiff="torch", jit=True)
formulon.backend_status()
```
The Legendre encoder/decoder (`measure_and_decode`, `encode_legendre`, `legendre_feature_expectation`) and Grover
follow the same switch. `formulon.QML` (the hybrid Qiskit-Aer-noise + PennyLane/torch engine) is intentionally pinned.

## Gates
`Circuit` methods (params first, then wires): `i x y z h s sdg t tdg sx sxdg rx ry rz p u cx cy cz swap iswap cp crx cry crz ccx ccz cswap mcz`.
`circuit.to_pennylane()` returns a function to call inside your own `qml.qnode`; `circuit.to_qiskit()` returns a `QuantumCircuit`.
Pure-PennyLane versions of every gate are in `formulon.quantum_computing.pennylane_gates`.

## Adders
| Circuit | Qubits | Result |
|---|---|---|
| `half_adder()` | 4 | sum, carry |
| `full_adder()` | 5 | sum, carry-out (inputs restored) |
| `ripple_carry_adder(n)` | 3n+1 | b <- a+b, exact (n+1 bits) |
| `qft_adder(n)` | 2n | b <- (a+b) mod 2^n, no ancilla/Toffoli |

Native PennyLane code: `formulon.quantum_computing.pennylane_adders` (`ripple_carry_adder`, `draper_adder`, `add_pennylane`).

## Optional Numba JIT (off by default)
```python
import formulon
formulon.legendre_polynomial(20, 0.2, jit=True)        # per call
with formulon.jit_context(True): ...                   # scope
formulon.set_jit(True)                                 # process-wide   (or FORMULON_JIT=1)
from formulon.ml import vectorize
vectorize(formulon.projectile_range)(v0_array, 45, 9.81, jit=True)   # numba ufunc on big batches
```
Install with `pip install "formulon-physics[jit]"`. Without numba, or if a function will not compile,
Formulon warns once and uses the normal Python/NumPy path; results are identical.
Note: compilation has a one-off cost, so JIT only pays off on repeated calls or large arrays.

## Verification status
Verified in the build sandbox (NumPy simulator only): all gate matrices unitary, gate identities
(S^2=Z, T^2=S, SX^2=X, HXH=Z), half/full adder truth tables, ripple and QFT adders exhaustively for n<=3,
QFT = DFT matrix, Grover (12 qubits: 99.995%), inverse circuits.
**Not run here:** the PennyLane and Qiskit translations and `pennylane_*` modules (packages not installable
offline). `tests/test_quantum_backends.py` cross-checks both against the NumPy reference and skips if absent —
run it first in a full environment. Same for real-numba compilation (`test_real_numba_matches_python`).
