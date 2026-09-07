import math

import pytest

qiskit_mod = pytest.importorskip(
    "formulon.quantum_computing.qiskit_circuits",
    reason="qiskit not installed; run: pip install formulon-physics[qiskit]",
)
pennylane_mod = pytest.importorskip(
    "formulon.quantum_computing.pennylane_circuits",
    reason="pennylane not installed; run: pip install formulon-physics[pennylane]",
)


def test_bell_pair_probabilities():
    probs = qiskit_mod.bell_pair_probabilities()
    assert set(probs.keys()) <= {"00", "11"}
    assert math.isclose(sum(probs.values()), 1.0, rel_tol=1e-9)
    assert math.isclose(probs.get("00", 0.0), 0.5, abs_tol=1e-9)
    assert math.isclose(probs.get("11", 0.0), 0.5, abs_tol=1e-9)


def test_legendre_kernel_self_similarity_qiskit():
    # k(x, x) must be 1.0: a state has perfect fidelity with itself
    k = qiskit_mod.legendre_quantum_kernel(0.3, 0.3, degree=3)
    assert math.isclose(k, 1.0, abs_tol=1e-9)


def test_legendre_kernel_bounded_qiskit():
    k = qiskit_mod.legendre_quantum_kernel(-0.8, 0.9, degree=4)
    assert 0.0 <= k <= 1.0 + 1e-9


def test_quantum_fourier_transform_qubit_count():
    qc = qiskit_mod.quantum_fourier_transform(3)
    assert qc.num_qubits == 3


def test_grover_two_qubit_search_finds_marked_state():
    counts = qiskit_mod.grover_two_qubit_search("11", shots=256)
    # Single-iteration Grover on 2 qubits is exact: "11" should dominate
    assert max(counts, key=counts.get) == "11"


def test_legendre_kernel_self_similarity_pennylane():
    k = pennylane_mod.legendre_quantum_kernel(0.5, 0.5, degree=3)
    assert math.isclose(k, 1.0, abs_tol=1e-6)


def test_variational_ansatz_expectation_range():
    val = pennylane_mod.variational_ansatz_expectation([0.0, 0.0], n_qubits=2)
    assert math.isclose(val, 1.0, abs_tol=1e-6)  # RY(0) leaves |0> -> <Z> = 1


def test_two_level_oscillator_ground_energy_at_theta_zero():
    # theta=0 keeps the qubit in |0> (the "ground" Fock state) -> E = 0.5
    e0 = pennylane_mod.two_level_oscillator_ground_energy(0.0)
    assert math.isclose(e0, 0.5, abs_tol=1e-6)


def test_parameter_shift_gradient_shape():
    grad = pennylane_mod.parameter_shift_gradient(0.4, degree=2)
    assert len(grad) == 2
    assert all(isinstance(g, float) for g in grad)
