"""spec: REQ-QC-001 REQ-QC-007"""
"""Cross-checks PennyLane / Qiskit against the NumPy reference (skipped if not installed)."""
import numpy as np
import pytest
from formulon.quantum_computing import Circuit, GATES, add, grover_search, gate_matrix


def _rich_circuit():
    c = Circuit(3)
    c.h(0).x(1).y(2).s(0).sdg(1).t(2).tdg(0).sx(1).rx(0.3, 0).ry(0.5, 1).rz(0.7, 2).p(0.4, 0).u(0.3, 0.5, 0.9, 1)
    c.cx(0, 1).cy(1, 2).cz(0, 2).swap(0, 2).iswap(0, 1).cp(0.8, 1, 2).crx(0.3, 0, 1).cry(0.4, 1, 2).crz(0.5, 2, 0)
    c.ccx(0, 1, 2).cswap(0, 1, 2).ccz(0, 1, 2).mcz(0, 1, 2)
    return c


@pytest.mark.parametrize("backend", ["pennylane", "qiskit"])
def test_backend_matches_numpy_reference(backend):
    pytest.importorskip({"pennylane": "pennylane", "qiskit": "qiskit"}[backend])
    c = _rich_circuit()
    ref = c.run(backend="numpy").probabilities
    got = c.run(backend=backend).probabilities
    for k in set(ref) | set(got):
        assert got.get(k, 0.0) == pytest.approx(ref.get(k, 0.0), abs=1e-8)


@pytest.mark.parametrize("backend", ["pennylane", "qiskit"])
def test_adder_and_grover_on_backend(backend):
    pytest.importorskip(backend)
    assert add(5, 9, 4, backend=backend) == 14
    assert add(5, 9, 4, method="qft", backend=backend) == 14
    assert grover_search(5, 3, backend=backend).found_state == 5


def test_native_pennylane_adders_and_gates():
    qml = pytest.importorskip("pennylane")
    from formulon.quantum_computing import pennylane_adders as pa, pennylane_gates as pg
    for a, b in [(0, 0), (3, 5), (7, 7), (5, 2)]:
        assert pa.add_pennylane(a, b, 3) == a + b
        assert pa.add_pennylane(a, b, 3, method="qft") == (a + b) % 8
    assert np.allclose(pg.gate_unitary(pg.cnot, 2), gate_matrix("CX"))
    assert np.allclose(pg.gate_unitary(pg.hadamard, 1), gate_matrix("H"))
    assert np.allclose(pg.gate_unitary(pg.toffoli, 3), gate_matrix("CCX"))
    p = pg.run_probs(pg.bell_state, 2)
    assert p[0] == pytest.approx(0.5) and p[3] == pytest.approx(0.5)
