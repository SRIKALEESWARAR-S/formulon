# SPDX-FileCopyrightText: 2026-present SRIKALEESWARAR-S <srikaleeswarar675@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0
"""
pennylane_circuits.py
=======================
Small, local-simulator-only quantum circuits built with PennyLane, exposed as
an optional extension of Project Formulon-Physics.

Everything here runs on PennyLane's built-in ``default.qubit`` simulator --
no hardware access, no external accounts, no network calls.

Requires: pip install "formulon-physics[pennylane]"

Project: Project Formulon-Physics
License: Apache-2.0
"""

from __future__ import annotations

import math

try:
    import pennylane as qml
    from pennylane import numpy as pnp
except ImportError as exc:  # pragma: no cover - exercised only when pennylane is absent
    raise ImportError(
        "formulon.quantum_computing.pennylane_circuits requires PennyLane. "
        'Install it with: pip install "formulon-physics[pennylane]"'
    ) from exc

from ..mathematicalphysics import legendre_polynomial

__all__ = [
    "legendre_feature_expectation",
    "legendre_quantum_kernel",
    "variational_ansatz_expectation",
    "parameter_shift_gradient",
    "two_level_oscillator_ground_energy",
]


def _legendre_angles(x: float, degree: int) -> list[float]:
    """Rescale P_0(x)..P_{degree-1}(x) from [-1, 1] into rotation angles [0, pi]."""
    return [(legendre_polynomial(k, x) + 1.0) * (math.pi / 2.0) for k in range(degree)]


# ══════════════════════════════════════════════════════════════════════════════
# 1. LEGENDRE FEATURE MAP
# ══════════════════════════════════════════════════════════════════════════════

def legendre_feature_expectation(x: float, degree: int) -> float:
    """
    Encode x into `degree` qubits via formulon's legendre_polynomial (one
    Legendre mode per qubit, as an RY rotation angle), then return
    <Z_0 Z_1 ... Z_{degree-1}> for the resulting state.

    This is the PennyLane counterpart of
    formulon.quantum_computing.qiskit_circuits.legendre_feature_map -- same
    encoding idea, expressed as a differentiable QNode so it can be used
    inside gradient-based (variational / quantum-ML) workflows.

    Project: Project Formulon-Physics
    """
    if degree < 1:
        raise ValueError("degree must be a positive integer (at least 1 qubit/mode).")

    dev = qml.device("default.qubit", wires=degree)

    @qml.qnode(dev)
    def circuit(x_val):
        angles = _legendre_angles(x_val, degree)
        for wire, theta in enumerate(angles):
            qml.RY(theta, wires=wire)
        return qml.expval(qml.prod(*[qml.PauliZ(w) for w in range(degree)]))

    return float(circuit(x))


def legendre_quantum_kernel(x1: float, x2: float, degree: int) -> float:
    """
    Fidelity-based quantum kernel k(x1, x2) = |<psi(x1)|psi(x2)>|^2 between
    two Legendre-feature-mapped states, computed the PennyLane way: apply the
    feature map for x1, then the *adjoint* feature map for x2, and measure
    the probability of returning to |0...0>. That return probability is
    exactly the fidelity between the two states.

    Project: Project Formulon-Physics
    """
    if degree < 1:
        raise ValueError("degree must be a positive integer (at least 1 qubit/mode).")

    dev = qml.device("default.qubit", wires=degree)

    def _feature_map(x_val):
        for wire, theta in enumerate(_legendre_angles(x_val, degree)):
            qml.RY(theta, wires=wire)

    @qml.qnode(dev)
    def circuit(a, b):
        _feature_map(a)
        qml.adjoint(_feature_map)(b)
        return qml.probs(wires=range(degree))

    probs = circuit(x1, x2)
    return float(probs[0])  # probability of measuring |0...0>


# ══════════════════════════════════════════════════════════════════════════════
# 2. VARIATIONAL ANSATZ / VQE-STYLE BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

def variational_ansatz_expectation(params: list[float], n_qubits: int = 2) -> float:
    """
    Evaluate <Z_0> for a single-layer hardware-efficient ansatz
    (RY rotations + a ring of CNOTs), the basic building block of VQE-style
    variational circuits.

    `params` must have length `n_qubits` (one rotation angle per qubit).

    Project: Project Formulon-Physics
    """
    if len(params) != n_qubits:
        raise ValueError(f"params must have length {n_qubits} (one angle per qubit).")

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(weights):
        for wire in range(n_qubits):
            qml.RY(weights[wire], wires=wire)
        for wire in range(n_qubits - 1):
            qml.CNOT(wires=[wire, wire + 1])
        return qml.expval(qml.PauliZ(0))

    return float(circuit(pnp.array(params)))


def parameter_shift_gradient(x: float, degree: int) -> list[float]:
    """
    Compute d/dx of `legendre_feature_expectation` at x, per Legendre-mode
    angle, using PennyLane's automatic differentiation (parameter-shift rule
    under the hood for hardware-compatible gradients).

    Returns the gradient of the observable with respect to each RY angle
    (one entry per qubit/Legendre mode).

    Project: Project Formulon-Physics
    """
    if degree < 1:
        raise ValueError("degree must be a positive integer (at least 1 qubit/mode).")

    dev = qml.device("default.qubit", wires=degree)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(angles):
        for wire in range(degree):
            qml.RY(angles[wire], wires=wire)
        return qml.expval(qml.prod(*[qml.PauliZ(w) for w in range(degree)]))

    angles = pnp.array(_legendre_angles(x, degree), requires_grad=True)
    grad_fn = qml.grad(circuit)
    return [float(g) for g in grad_fn(angles)]


# ══════════════════════════════════════════════════════════════════════════════
# 3. TOY VARIATIONAL EIGENSOLVER: TRUNCATED HARMONIC OSCILLATOR
# ══════════════════════════════════════════════════════════════════════════════

def two_level_oscillator_ground_energy(theta: float) -> float:
    """
    Toy 1-qubit VQE: estimate the ground-state energy of a quantum harmonic
    oscillator truncated to its lowest two Fock states {|0>, |1>}, using a
    single-parameter RY ansatz.

    In this truncated basis the Hamiltonian is H = hbar*omega * (N + 1/2),
    which (in units hbar = omega = 1) reduces to diag(0.5, 1.5) -- so the
    variational minimum over theta should approach 0.5 (the true QHO
    ground-state energy in these units), reached at theta = 0.

    This pairs naturally with formulon's own `hermite_polynomial`, which
    generates the *classical* QHO eigenfunctions -- this function estimates
    the same physical system's ground energy variationally instead.

    Project: Project Formulon-Physics
    """
    dev = qml.device("default.qubit", wires=1)

    # H = 0.5*I + 0.5*Z in the {|0>, |1>} truncated Fock basis
    hamiltonian = qml.Hamiltonian([0.5, 0.5], [qml.Identity(0), qml.PauliZ(0)])

    @qml.qnode(dev)
    def circuit(t):
        qml.RY(t, wires=0)
        return qml.expval(hamiltonian)

    return float(circuit(theta))
