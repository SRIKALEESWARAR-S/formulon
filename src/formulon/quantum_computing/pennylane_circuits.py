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

from .grover import GroverResult, grover_search  # backend-neutral implementation
