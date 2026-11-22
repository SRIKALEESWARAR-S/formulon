"""Grover search on any backend (PennyLane, Qiskit or the NumPy simulator)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from .circuit import Circuit

__all__ = ["GroverResult", "grover_circuit", "grover_search", "optimal_iterations"]


@dataclass(frozen=True)
class GroverResult:
    target: int
    target_binary: str
    found_state: int
    success_probability: float      # probability of the *target* state
    iterations: int
    probabilities: Tuple[float, ...]
    backend: str


def optimal_iterations(num_qubits: int) -> int:
    return max(1, int(math.floor(math.pi / 4 * math.sqrt(2 ** num_qubits))))


def grover_circuit(target: int, num_qubits: int, iterations: int | None = None) -> Circuit:
    if not isinstance(num_qubits, int) or num_qubits < 1:
        raise ValueError("num_qubits must be a positive integer")
    if not isinstance(target, int) or isinstance(target, bool) or not 0 <= target < 2 ** num_qubits:
        raise ValueError(f"target must be an integer in [0, 2**{num_qubits})")
    if iterations is None:
        iterations = optimal_iterations(num_qubits)
    if iterations < 0:
        raise ValueError("iterations must be >= 0")
    bits = f"{target:0{num_qubits}b}"
    wires = list(range(num_qubits))
    c = Circuit(num_qubits)
    for w in wires:
        c.h(w)
    for _ in range(iterations):
        for w, b in enumerate(bits):          # oracle: phase-flip |target>
            if b == "0":
                c.x(w)
        c.mcz(*wires)
        for w, b in enumerate(bits):
            if b == "0":
                c.x(w)
        for w in wires:                       # diffusion: 2|s><s| - I
            c.h(w); c.x(w)
        c.mcz(*wires)
        for w in wires:
            c.x(w); c.h(w)
    return c


def grover_search(target: int, num_qubits: int = 2, iterations: int | None = None,
                  backend: str | None = None) -> GroverResult:
    """Run Grover's algorithm.  ``iterations=None`` uses floor(pi/4 * sqrt(2**n))."""
    iters = optimal_iterations(num_qubits) if iterations is None else iterations
    res = grover_circuit(target, num_qubits, iters).run(backend=backend)
    probs = [res.probabilities.get(format(i, f"0{num_qubits}b"), 0.0) for i in range(2 ** num_qubits)]
    found = max(range(len(probs)), key=probs.__getitem__)
    return GroverResult(target, f"{target:0{num_qubits}b}", found, probs[target], iters,
                        tuple(probs), res.backend)
