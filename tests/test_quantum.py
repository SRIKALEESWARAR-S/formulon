import pytest
qml = pytest.importorskip("pennylane")

from formulon.quantum_computing.grover import grover_search
from formulon.quantum_computing.legendre_encoder import (
    legendre_features, measure_and_decode
)


def test_grover_two_qubit():
    result = grover_search(3, num_qubits=2, iterations=1)
    assert result.found_state == 3
    assert result.success_probability > 0.99


def test_grover_12_qubit_target():
    result = grover_search(3456, num_qubits=12)
    assert result.found_state == 3456
    assert result.success_probability > 0.99


def test_legendre_features_endpoints():
    f = legendre_features(-1.0, 4)
    assert f[0] == pytest.approx(1.0)
    assert f[1] == pytest.approx(-1.0)


def test_measure_decode():
    result = measure_and_decode(0.35, degree=6, shots=2000, seed=7)
    assert abs(result.decoded_x - 0.35) < 0.08
    assert len(result.samples) == 2000
