"""PennyLane Grover search on the local ``default.qubit`` simulator.

This module is an educational/reference implementation.  It marks exactly
one computational-basis state and uses the standard Grover iterate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

try:
    import pennylane as qml
    from pennylane import numpy as np
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        'PennyLane is required. Install with: pip install "formulon-physics[pennylane]"'
    ) from exc

__all__ = ["GroverResult", "grover_search", "optimal_grover_iterations"]


@dataclass(frozen=True)
class GroverResult:
    """Result of a statevector Grover search."""
    target: int
    target_binary: str
    found_state: int
    success_probability: float
    iterations: int
    probabilities: tuple[float, ...]


def optimal_grover_iterations(num_qubits: int) -> int:
    """Return the nearest useful Grover iteration count for one marked state."""
    if not isinstance(num_qubits, int) or isinstance(num_qubits, bool) or num_qubits < 1:
        raise ValueError("num_qubits must be a positive integer")
    n = 2 ** num_qubits
    return max(1, int(math.floor(math.pi / 4 * math.sqrt(n))))


def _validate_target(target: int, num_qubits: int) -> None:
    if not isinstance(target, int) or isinstance(target, bool):
        raise TypeError("target must be an integer")
    if target < 0 or target >= 2 ** num_qubits:
        raise ValueError(f"target must be in [0, {2 ** num_qubits - 1}]")


def _oracle(target: int, num_qubits: int) -> None:
    """Phase-flip exactly ``|target>`` using X-conjugated MCX."""
    bits = f"{target:0{num_qubits}b}"
    for wire, bit in enumerate(bits):
        if bit == "0":
            qml.PauliX(wires=wire)
    qml.MultiControlledX(
        control_wires=list(range(num_qubits - 1)),
        wires=num_qubits - 1,
        control_values=[1] * (num_qubits - 1),
    )
    for wire, bit in enumerate(bits):
        if bit == "0":
            qml.PauliX(wires=wire)


def _diffusion(num_qubits: int) -> None:
    """Reflection about the uniform superposition."""
    for wire in range(num_qubits):
        qml.Hadamard(wires=wire)
        qml.PauliX(wires=wire)
    qml.MultiControlledX(
        control_wires=list(range(num_qubits - 1)),
        wires=num_qubits - 1,
        control_values=[1] * (num_qubits - 1),
    )
    for wire in range(num_qubits):
        qml.PauliX(wires=wire)
        qml.Hadamard(wires=wire)


def grover_search(target: int, num_qubits: int | None = None, iterations: int | None = None) -> GroverResult:
    """Search for one marked integer using Grover's algorithm.

    Parameters
    ----------
    target:
        Integer represented by the marked computational-basis state.
    num_qubits:
        Number of qubits.  Defaults to the minimum required for ``target``.
    iterations:
        Grover iterations.  Defaults to the standard single-solution estimate.

    Returns
    -------
    GroverResult
        Statevector probability distribution and most-probable state.
    """
    if num_qubits is None:
        num_qubits = max(1, target.bit_length()) if isinstance(target, int) else 1
    if not isinstance(num_qubits, int) or isinstance(num_qubits, bool) or num_qubits < 1:
        raise ValueError("num_qubits must be a positive integer")
    _validate_target(target, num_qubits)
    if iterations is None:
        iterations = optimal_grover_iterations(num_qubits)
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 0:
        raise ValueError("iterations must be a non-negative integer")

    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def circuit():
        for wire in range(num_qubits):
            qml.Hadamard(wires=wire)
        for _ in range(iterations):
            _oracle(target, num_qubits)
            _diffusion(num_qubits)
        return qml.probs(wires=range(num_qubits))

    probs = tuple(float(x) for x in circuit())
    found = int(np.argmax(np.asarray(probs)))
    return GroverResult(
        target=target,
        target_binary=f"{target:0{num_qubits}b}",
        found_state=found,
        success_probability=probs[found],
        iterations=iterations,
        probabilities=probs,
    )
