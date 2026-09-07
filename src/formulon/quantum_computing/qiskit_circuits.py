# SPDX-FileCopyrightText: 2026-present SRIKALEESWARAR-S <srikaleeswarar675@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0
"""
qiskit_circuits.py
===================
Small, local-simulator-only quantum circuits built with Qiskit, exposed as
an optional extension of Project Formulon-Physics.

Nothing here touches real quantum hardware or IBM Quantum's cloud service --
every function runs against Qiskit's built-in statevector primitives, so no
API token and no network access are required.

Requires: pip install "formulon-physics[qiskit]"

Project: Project Formulon-Physics
License: Apache-2.0
"""

from __future__ import annotations

import math

try:
    from qiskit import QuantumCircuit
    from qiskit.primitives import StatevectorSampler
    from qiskit.quantum_info import Statevector
except ImportError as exc:  # pragma: no cover - exercised only when qiskit is absent
    raise ImportError(
        "formulon.quantum_computing.qiskit_circuits requires Qiskit. "
        'Install it with: pip install "formulon-physics[qiskit]"'
    ) from exc

from ..mathematicalphysics import legendre_polynomial

__all__ = [
    "bell_pair_probabilities",
    "legendre_feature_map",
    "legendre_quantum_kernel",
    "quantum_fourier_transform",
    "grover_two_qubit_search",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. BASIC ENTANGLEMENT DEMO
# ══════════════════════════════════════════════════════════════════════════════

def bell_pair_probabilities() -> dict[str, float]:
    """
    Build a 2-qubit Bell pair (H + CNOT) and return its exact measurement
    probability distribution via statevector simulation.

    Expected result: {"00": 0.5, "11": 0.5} -- the textbook entanglement
    signature (no "01"/"10" outcomes).

    Project: Project Formulon-Physics
    """
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    sv = Statevector.from_instruction(qc)
    return sv.probabilities_dict()


# ══════════════════════════════════════════════════════════════════════════════
# 2. LEGENDRE FEATURE MAP + QUANTUM KERNEL
# ══════════════════════════════════════════════════════════════════════════════

def legendre_feature_map(x: float, degree: int) -> "QuantumCircuit":
    """
    Encode a classical scalar x into a `degree`-qubit quantum state using
    formulon's own `legendre_polynomial` as the feature generator.

    For qubit k (k = 0 .. degree-1), apply RY(theta_k) where theta_k is the
    value of the k-th Legendre polynomial P_k(x), rescaled into [0, pi] so it
    is a valid rotation angle (Legendre polynomials are bounded in [-1, 1]
    for x in [-1, 1]).

    This is an orthogonal-polynomial feature map in the same spirit as
    Chebyshev/Legendre feature maps studied in quantum kernel methods --
    it turns a 1D classical variable into a `degree`-dimensional quantum
    feature vector, one Legendre "mode" per qubit.

    Parameters
    ----------
    x : float
        Input value, expected in [-1, 1] (the natural domain of P_n(x)).
    degree : int
        Number of Legendre modes / qubits to use (degree >= 1).

    Project: Project Formulon-Physics
    """
    if degree < 1:
        raise ValueError("degree must be a positive integer (at least 1 qubit/mode).")

    qc = QuantumCircuit(degree)
    for k in range(degree):
        p_k = legendre_polynomial(k, x)
        # Rescale P_k(x) in [-1, 1] -> angle in [0, pi]
        theta_k = (p_k + 1.0) * (math.pi / 2.0)
        qc.ry(theta_k, k)
    return qc


def legendre_quantum_kernel(x1: float, x2: float, degree: int) -> float:
    """
    Compute a fidelity-based quantum kernel k(x1, x2) = |<psi(x1)|psi(x2)>|^2
    between two Legendre-feature-mapped states -- the "Legendre dot product"
    computed via quantum state overlap rather than a classical inner product.

    Implementation: build both feature-map states with `Statevector`, then
    take the squared magnitude of their inner product directly (exact
    statevector simulation -- no shot noise, since this is a teaching/
    prototyping tool, not a hardware run).

    Returns a value in [0, 1]: 1.0 means identical feature states, 0.0 means
    orthogonal (maximally distinguishable) feature states.

    Project: Project Formulon-Physics
    """
    sv1 = Statevector.from_instruction(legendre_feature_map(x1, degree))
    sv2 = Statevector.from_instruction(legendre_feature_map(x2, degree))
    overlap = sv1.inner(sv2)
    return float(abs(overlap) ** 2)


# ══════════════════════════════════════════════════════════════════════════════
# 3. QUANTUM FOURIER TRANSFORM (built from primitives, for pedagogy)
# ══════════════════════════════════════════════════════════════════════════════

def quantum_fourier_transform(num_qubits: int) -> "QuantumCircuit":
    """
    Build a textbook Quantum Fourier Transform circuit (Hadamards + controlled
    phase rotations + final qubit-order swaps) from primitive gates.

    Built explicitly (rather than imported from qiskit.circuit.library) so the
    construction is transparent and matches the "show the physics/math"
    philosophy of the rest of formulon.

    Project: Project Formulon-Physics
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be a positive integer.")

    qc = QuantumCircuit(num_qubits, name=f"QFT({num_qubits})")
    for j in range(num_qubits):
        qc.h(j)
        for k in range(j + 1, num_qubits):
            angle = math.pi / (2 ** (k - j))
            qc.cp(angle, k, j)
    # Reverse qubit order to match the standard QFT output convention
    for i in range(num_qubits // 2):
        qc.swap(i, num_qubits - i - 1)
    return qc


# ══════════════════════════════════════════════════════════════════════════════
# 4. GROVER'S SEARCH (2-qubit exact case)
# ══════════════════════════════════════════════════════════════════════════════

def grover_two_qubit_search(marked_state: str, shots: int = 1024) -> dict[str, int]:
    """
    Run a single-iteration 2-qubit Grover search for `marked_state`
    (one of "00", "01", "10", "11") and return sampled measurement counts.

    A single Grover iteration is *exact* for a 2-qubit (4-item) search space,
    so the marked state should appear with probability 1 (up to simulator
    rounding), unlike larger searches which need multiple iterations.

    Project: Project Formulon-Physics
    """
    if marked_state not in ("00", "01", "10", "11"):
        raise ValueError('marked_state must be one of "00", "01", "10", "11".')

    qc = QuantumCircuit(2)
    qc.h([0, 1])

    # Oracle: flip the phase of |marked_state> by sandwiching an X on any
    # qubit that should be '0' in the target, around a controlled-Z.
    flip_qubits = [i for i, bit in enumerate(reversed(marked_state)) if bit == "0"]
    for q in flip_qubits:
        qc.x(q)
    qc.cz(0, 1)
    for q in flip_qubits:
        qc.x(q)

    # Diffusion operator (inversion about the mean)
    qc.h([0, 1])
    qc.x([0, 1])
    qc.cz(0, 1)
    qc.x([0, 1])
    qc.h([0, 1])
    qc.measure_all()

    sampler = StatevectorSampler()
    job = sampler.run([qc], shots=shots)
    result = job.result()[0]
    # measure_all() creates a classical register named "meas" by convention
    counts = result.data.meas.get_counts()
    return dict(counts)
