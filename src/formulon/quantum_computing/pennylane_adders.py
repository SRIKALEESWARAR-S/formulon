"""Quantum adders written natively in PennyLane.

* ``half_adder`` / ``full_adder``  - 1-bit adders (Toffoli + CNOT)
* ``ripple_carry_adder``           - n-bit Vedral-Barenco-Ekert adder, b <- a + b
* ``draper_adder``                 - n-bit QFT adder, b <- (a + b) mod 2**n

Registers are LSB-first lists of wires.  The functions queue operations, so call
them inside a ``qml.qnode`` or use the ``*_probs`` / ``add_pennylane`` helpers.
The backend-neutral versions live in :mod:`formulon.quantum_computing.adders`
and are cross-checked against these in the test-suite.
"""
from __future__ import annotations

import math

import numpy as np

try:
    import pennylane as qml
except ImportError as exc:  # pragma: no cover
    raise ImportError('PennyLane is required: pip install "formulon-physics[pennylane]"') from exc

__all__ = ["half_adder", "full_adder", "ripple_carry_adder", "draper_adder",
           "ripple_carry_layout", "add_pennylane"]


def half_adder(a, b, s, carry):
    """s = a xor b, carry = a and b  (s, carry start in |0>)."""
    qml.Toffoli(wires=[a, b, carry])
    qml.CNOT(wires=[a, s])
    qml.CNOT(wires=[b, s])


def full_adder(a, b, cin, s, cout):
    """s = a xor b xor cin, cout = majority(a, b, cin); a, b, cin are restored."""
    qml.Toffoli(wires=[a, b, cout])
    qml.CNOT(wires=[a, b])
    qml.Toffoli(wires=[b, cin, cout])
    qml.CNOT(wires=[b, s])
    qml.CNOT(wires=[cin, s])
    qml.CNOT(wires=[a, b])


def _carry(c, a, b, c_next):
    qml.Toffoli(wires=[a, b, c_next]); qml.CNOT(wires=[a, b]); qml.Toffoli(wires=[c, b, c_next])


def _carry_inv(c, a, b, c_next):
    qml.Toffoli(wires=[c, b, c_next]); qml.CNOT(wires=[a, b]); qml.Toffoli(wires=[a, b, c_next])


def _sum(c, a, b):
    qml.CNOT(wires=[a, b]); qml.CNOT(wires=[c, b])


def ripple_carry_layout(n: int):
    """(carry wires, a wires, b wires) for an n-bit adder: 3n+1 qubits, b has n+1 bits."""
    return list(range(n)), list(range(n, 2 * n)), list(range(2 * n, 3 * n + 1))


def ripple_carry_adder(n: int):
    """Queue the VBE adder on the layout from :func:`ripple_carry_layout`."""
    c, a, b = ripple_carry_layout(n)
    for i in range(n - 1):
        _carry(c[i], a[i], b[i], c[i + 1])
    _carry(c[n - 1], a[n - 1], b[n - 1], b[n])
    qml.CNOT(wires=[a[n - 1], b[n - 1]])
    _sum(c[n - 1], a[n - 1], b[n - 1])
    for i in range(n - 2, -1, -1):
        _carry_inv(c[i], a[i], b[i], c[i + 1])
        _sum(c[i], a[i], b[i])


def draper_adder(a_wires, b_wires):
    """b <- (a + b) mod 2**n via phase addition in the Fourier basis (LSB-first registers)."""
    n = len(a_wires)
    b_msb = list(b_wires)[::-1]
    qml.QFT(wires=b_msb)
    for j in range(n):
        for i in range(n):
            e = (j + 1) - i
            if e >= 1:
                qml.ControlledPhaseShift(2 * math.pi / 2 ** e, wires=[a_wires[i], b_msb[j]])
    qml.adjoint(qml.QFT)(wires=b_msb)


def add_pennylane(a: int, b: int, n_bits: int, method: str = "ripple") -> int:
    """Run an addition on PennyLane's default.qubit and return the integer result."""
    if method == "ripple":
        _, aw, bw = ripple_carry_layout(n_bits)
        n_q = 3 * n_bits + 1
    elif method == "qft":
        aw, bw = list(range(n_bits)), list(range(n_bits, 2 * n_bits))
        n_q = 2 * n_bits
    else:
        raise ValueError("method must be 'ripple' or 'qft'")
    for name, v in (("a", a), ("b", b)):
        if not 0 <= v < 2 ** n_bits:
            raise ValueError(f"{name} must be in [0, 2**{n_bits})")
    dev = qml.device("default.qubit", wires=n_q)

    @qml.qnode(dev)
    def circuit():
        for i, w in enumerate(aw):
            if (a >> i) & 1:
                qml.PauliX(wires=w)
        for i, w in enumerate(bw[:n_bits]):
            if (b >> i) & 1:
                qml.PauliX(wires=w)
        if method == "ripple":
            ripple_carry_adder(n_bits)
        else:
            draper_adder(aw, bw)
        return qml.probs(wires=list(range(n_q)))

    probs = np.asarray(circuit())
    idx = int(np.argmax(probs))
    bits = format(idx, f"0{n_q}b")
    return sum(int(bits[w]) << i for i, w in enumerate(bw))
