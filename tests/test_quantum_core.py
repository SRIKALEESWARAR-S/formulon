"""spec: REQ-QC-001 REQ-QC-002 REQ-QC-003 REQ-QC-004 REQ-QC-005 REQ-QC-006"""
import itertools
import numpy as np
import pytest
from formulon.quantum_computing import (Circuit, GATES, gate_matrix, half_adder, full_adder, add,
                                        qft_circuit, grover_search, set_backend, get_backend,
                                        use_backend, available_backends, resolve_backend)


def test_bell_state():
    p = Circuit(2).h(0).cx(0, 1).run(backend="numpy").probabilities
    assert set(p) == {"00", "11"} and p["00"] == pytest.approx(0.5)


def test_all_gate_matrices_unitary():
    for name, spec in GATES.items():
        if spec.n_wires is None:
            continue
        m = gate_matrix(name, *[0.7, 0.3, 1.1][:spec.n_params])
        assert np.allclose(m @ m.conj().T, np.eye(len(m))), name


def test_known_gate_identities():
    assert np.allclose(gate_matrix("S") @ gate_matrix("S"), gate_matrix("Z"))
    assert np.allclose(gate_matrix("T") @ gate_matrix("T"), gate_matrix("S"))
    assert np.allclose(gate_matrix("SX") @ gate_matrix("SX"), gate_matrix("X"))
    assert np.allclose(gate_matrix("H") @ gate_matrix("X") @ gate_matrix("H"), gate_matrix("Z"))
    assert np.allclose(gate_matrix("RZ", np.pi), -1j * gate_matrix("Z"))


def test_circuit_validation():
    with pytest.raises(ValueError):
        Circuit(2).cx(0, 0)
    with pytest.raises(ValueError):
        Circuit(2).h(5)
    with pytest.raises(KeyError):
        Circuit(2).gate("NOPE", 0)


def test_inverse_restores_state():
    c = Circuit(2).h(0).t(0).cp(0.4, 0, 1).u(0.3, 0.5, 0.9, 1).sx(0).rx(0.2, 1)
    back = c.inverse().statevector(initial=c.statevector())
    assert abs(back[0]) == pytest.approx(1.0)


def test_half_and_full_adder_truth_tables():
    for a, b in itertools.product([0, 1], repeat=2):
        r = half_adder().run(backend="numpy", initial=f"{a}{b}00").most_likely()
        assert (int(r[2]), int(r[3])) == (a ^ b, a & b)
    for a, b, c in itertools.product([0, 1], repeat=3):
        r = full_adder().run(backend="numpy", initial=f"{a}{b}{c}00").most_likely()
        assert r[:3] == f"{a}{b}{c}"
        assert (int(r[3]), int(r[4])) == (a ^ b ^ c, int(a + b + c >= 2))


def test_ripple_and_qft_adders_exhaustive():
    for n in (1, 2, 3):
        for a in range(2 ** n):
            for b in range(2 ** n):
                assert add(a, b, n, backend="numpy") == a + b
                assert add(a, b, n, method="qft", backend="numpy") == (a + b) % 2 ** n


def test_add_input_validation():
    with pytest.raises(ValueError):
        add(16, 1, 4, backend="numpy")
    with pytest.raises(ValueError):
        add(1, 1, 4, method="x", backend="numpy")


def test_qft_equals_dft():
    n = 3
    q = qft_circuit(n)
    cols = []
    for k in range(2 ** n):
        st = np.zeros(2 ** n, complex); st[k] = 1
        cols.append(q.statevector(initial=st))
    w = np.exp(2j * np.pi / 2 ** n)
    dft = np.array([[w ** (j * k) for k in range(2 ** n)] for j in range(2 ** n)]) / np.sqrt(2 ** n)
    assert np.allclose(np.column_stack(cols), dft)


def test_grover():
    r = grover_search(3, num_qubits=2, iterations=1, backend="numpy")
    assert r.found_state == 3 and r.success_probability > 0.99 and r.target_binary == "11"
    r = grover_search(3456, num_qubits=12, backend="numpy")
    assert r.found_state == 3456 and r.success_probability > 0.99


def test_backend_selection():
    assert "numpy" in available_backends()
    with use_backend("numpy"):
        assert get_backend() == "numpy"
        assert Circuit(1).x(0).run().backend == "numpy"
    with pytest.raises(ValueError):
        set_backend("cirq")
    if "pennylane" not in available_backends():       # explicit choice must not silently fall back
        with pytest.raises(ImportError):
            resolve_backend("pennylane")
    if "qiskit" not in available_backends():
        with pytest.raises(ImportError):
            resolve_backend("qiskit")
