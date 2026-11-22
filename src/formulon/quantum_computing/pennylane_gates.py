"""Basic quantum gates written directly in PennyLane (reference / teaching module).

Every gate is exposed as a plain function that queues the PennyLane operation,
plus ``gate_unitary`` to get its matrix, and small demo circuits.
"""
from __future__ import annotations

import numpy as np

try:
    import pennylane as qml
except ImportError as exc:  # pragma: no cover
    raise ImportError('PennyLane is required: pip install "formulon-physics[pennylane]"') from exc

__all__ = [
    "identity", "pauli_x", "pauli_y", "pauli_z", "hadamard", "s_gate", "s_dagger", "t_gate",
    "t_dagger", "sx_gate", "rx", "ry", "rz", "phase", "u3",
    "cnot", "cy", "cz", "swap", "iswap", "controlled_phase", "crx", "cry", "crz",
    "toffoli", "ccz", "fredkin", "multi_controlled_z",
    "gate_unitary", "bell_state", "ghz_state", "run_probs",
]

# ---- single-qubit --------------------------------------------------------------
def identity(w): return qml.Identity(wires=w)
def pauli_x(w): return qml.PauliX(wires=w)
def pauli_y(w): return qml.PauliY(wires=w)
def pauli_z(w): return qml.PauliZ(wires=w)
def hadamard(w): return qml.Hadamard(wires=w)
def s_gate(w): return qml.S(wires=w)
def s_dagger(w): return qml.adjoint(qml.S)(wires=w)
def t_gate(w): return qml.T(wires=w)
def t_dagger(w): return qml.adjoint(qml.T)(wires=w)
def sx_gate(w): return qml.SX(wires=w)
def rx(theta, w): return qml.RX(theta, wires=w)
def ry(theta, w): return qml.RY(theta, wires=w)
def rz(theta, w): return qml.RZ(theta, wires=w)
def phase(phi, w): return qml.PhaseShift(phi, wires=w)
def u3(theta, phi, lam, w): return qml.U3(theta, phi, lam, wires=w)

# ---- two / three qubit ---------------------------------------------------------
def cnot(c, t): return qml.CNOT(wires=[c, t])
def cy(c, t): return qml.CY(wires=[c, t])
def cz(c, t): return qml.CZ(wires=[c, t])
def swap(a, b): return qml.SWAP(wires=[a, b])
def iswap(a, b): return qml.ISWAP(wires=[a, b])
def controlled_phase(phi, c, t): return qml.ControlledPhaseShift(phi, wires=[c, t])
def crx(theta, c, t): return qml.CRX(theta, wires=[c, t])
def cry(theta, c, t): return qml.CRY(theta, wires=[c, t])
def crz(theta, c, t): return qml.CRZ(theta, wires=[c, t])
def toffoli(c1, c2, t): return qml.Toffoli(wires=[c1, c2, t])
def ccz(c1, c2, t): return qml.CCZ(wires=[c1, c2, t])
def fredkin(c, a, b): return qml.CSWAP(wires=[c, a, b])


def multi_controlled_z(wires):
    """Z on wires[-1] controlled on all other wires (Grover oracle building block)."""
    wires = list(wires)
    if len(wires) == 1:
        return qml.PauliZ(wires=wires[0])
    return qml.ctrl(qml.PauliZ(wires=wires[-1]), control=wires[:-1])


def gate_unitary(op_factory, n_wires: int, *params) -> np.ndarray:
    """Matrix of a gate function from this module, e.g. ``gate_unitary(cnot, 2)``."""
    wires = list(range(n_wires))
    return np.asarray(qml.matrix(lambda: op_factory(*params, *wires), wire_order=wires)())


# ---- demo circuits ---------------------------------------------------------------
def run_probs(circuit_fn, n_qubits: int) -> np.ndarray:
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def f():
        circuit_fn()
        return qml.probs(wires=list(range(n_qubits)))
    return np.asarray(f())


def bell_state():
    """(|00> + |11>)/sqrt2."""
    hadamard(0); cnot(0, 1)


def ghz_state(n: int):
    """(|0..0> + |1..1>)/sqrt2 on ``n`` qubits."""
    hadamard(0)
    for i in range(n - 1):
        cnot(i, i + 1)
