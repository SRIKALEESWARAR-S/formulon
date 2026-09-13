# SPDX-FileCopyrightText: 2026-present SRIKALEESWARAR-S <srikaleeswarar675@gmail.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import math
from dataclasses import dataclass
import pennylane as qml
from pennylane import numpy as pnp
from ..mathematicalphysics import legendre_polynomial

__all__ = [
    "legendre_feature_expectation",
    "legendre_quantum_kernel",
    "GroverResult",
    "grover_search"
]

def _legendre_angles(x: float, degree: int) -> list[float]:
    """Correctly map P_n(x) to RY angles so that <Z> = P_n(x)."""
    return [float(pnp.arccos(pnp.clip(legendre_polynomial(k, x), -1.0, 1.0))) for k in range(degree)]

def legendre_feature_expectation(x: float, degree: int) -> float:
    if degree < 1:
        raise ValueError("degree must be at least 1.")
    
    dev = qml.device("default.qubit", wires=degree)

    @qml.qnode(dev)
    def circuit(x_val):
        angles = _legendre_angles(x_val, degree)
        for wire, theta in enumerate(angles):
            qml.RY(theta, wires=wire)
        obs = qml.PauliZ(0)
        for w in range(1, degree):
            obs = obs @ qml.PauliZ(w)
        return qml.expval(obs)

    return float(circuit(x))

@dataclass(frozen=True)
class GroverResult:
    target: int
    found_state: int
    success_probability: float
    iterations: int

def grover_search(target: int, num_qubits: int, iterations: int | None = None) -> GroverResult:
    if iterations is None:
        iterations = max(1, int(math.floor(math.pi / 4 * math.sqrt(2 ** num_qubits))))

    dev = qml.device("default.qubit", wires=num_qubits)

    def _phase_oracle(tgt: int):
        bits = f"{tgt:0{num_qubits}b}"
        for wire, bit in enumerate(bits):
            if bit == "0":
                qml.PauliX(wires=wire)
        # Apply correct phase flip via Multi-Controlled Z
        qml.ctrl(qml.PauliZ(wires=num_qubits - 1), control=list(range(num_qubits - 1)))
        for wire, bit in enumerate(bits):
            if bit == "0":
                qml.PauliX(wires=wire)

    def _diffusion():
        for wire in range(num_qubits):
            qml.Hadamard(wires=wire)
            qml.PauliX(wires=wire)
        qml.ctrl(qml.PauliZ(wires=num_qubits - 1), control=list(range(num_qubits - 1)))
        for wire in range(num_qubits):
            qml.PauliX(wires=wire)
            qml.Hadamard(wires=wire)

    @qml.qnode(dev)
    def circuit():
        for wire in range(num_qubits):
            qml.Hadamard(wires=wire)
        for _ in range(iterations):
            _phase_oracle(target)
            _diffusion()
        return qml.probs(wires=range(num_qubits))

    probs = circuit()
    found = int(pnp.argmax(probs))
    return GroverResult(target, found, float(probs[found]), iterations)
