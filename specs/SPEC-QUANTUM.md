# SPEC-QUANTUM: circuits, gates, adders
- **REQ-QC-001** `Circuit` runs on `pennylane`, `qiskit` or `numpy` with identical probabilities (wire 0 = left-most bit).
- **REQ-QC-002** All gate matrices are unitary and satisfy S^2=Z, T^2=S, SX^2=X, HXH=Z.
- **REQ-QC-003** Circuit construction validates wires/gate names; `inverse()` undoes the circuit.
- **REQ-QC-004** Half/full adders match truth tables; ripple-carry gives exactly a+b and the QFT adder (a+b) mod 2^n for all inputs up to 3 bits; inputs are validated.
- **REQ-QC-005** `qft_circuit(n)` equals the n-qubit DFT matrix.
- **REQ-QC-006** `grover_search` finds the target with probability > 0.99 using the optimal iteration count (2 and 12 qubits).
- **REQ-QC-007** The native PennyLane gate/adder modules agree with the NumPy reference (requires PennyLane).
- **REQ-QC-008** The Legendre encoder/decoder run on the backend switch; `<ZZ..Z>` equals prod P_k(x); a fixed seed gives identical samples; decoding recovers x within tolerance.
- **REQ-QC-009** Importing `formulon.quantum_computing` requires neither PennyLane nor Qiskit.
