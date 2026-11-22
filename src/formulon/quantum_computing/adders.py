"""Quantum adders: half adder, full adder, ripple-carry (VBE) and Draper QFT adder.

All builders return :class:`Circuit` objects (run on PennyLane, Qiskit or the
NumPy simulator).  Registers are listed least-significant-bit first, i.e.
``a_wires[i]`` carries bit i (weight 2**i).  Quick use::

    from formulon.quantum_computing import set_backend, add
    set_backend("pennylane")
    add(5, 9, n_bits=4)                     # -> 14   (ripple-carry)
    add(5, 9, n_bits=4, method="qft")       # -> 14   (Draper, result mod 2**4)
    add(5, 9, n_bits=4, backend="qiskit")   # per-call backend override
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .circuit import Circuit

__all__ = ["AdderCircuit", "half_adder", "full_adder", "ripple_carry_adder", "qft_adder",
           "qft_circuit", "add"]


@dataclass
class AdderCircuit:
    circuit: Circuit
    a: list          # input register (unchanged), LSB first
    b: list          # accumulates the sum, LSB first (ripple: n+1 bits incl. carry-out)
    ancilla: list    # workspace wires, returned to |0>

    @property
    def n_qubits(self) -> int:
        return self.circuit.n_qubits


# ------------------------------------------------------------------ 1-bit adders
def half_adder() -> Circuit:
    """Wires: 0=a, 1=b, 2=sum (start 0), 3=carry (start 0).  sum=a xor b, carry=a and b."""
    c = Circuit(4)
    c.ccx(0, 1, 3).cx(0, 2).cx(1, 2)
    return c


def full_adder() -> Circuit:
    """Wires: 0=a, 1=b, 2=cin, 3=sum (start 0), 4=cout (start 0).  Inputs are restored."""
    c = Circuit(5)
    c.ccx(0, 1, 4)          # cout  = a.b
    c.cx(0, 1)              # b -> a xor b
    c.ccx(1, 2, 4)          # cout ^= (a xor b).cin   => majority(a,b,cin)
    c.cx(1, 3).cx(2, 3)     # sum   = a xor b xor cin
    c.cx(0, 1)              # restore b
    return c


# -------------------------------------------------------------- ripple-carry (VBE)
def _carry(c: Circuit, cin, a, b, cout):
    c.ccx(a, b, cout).cx(a, b).ccx(cin, b, cout)


def _carry_inv(c: Circuit, cin, a, b, cout):
    c.ccx(cin, b, cout).cx(a, b).ccx(a, b, cout)


def _sum(c: Circuit, cin, a, b):
    c.cx(a, b).cx(cin, b)


def ripple_carry_adder(n_bits: int) -> AdderCircuit:
    """Vedral-Barenco-Ekert adder: b <- a + b (n+1 bits, top bit = carry-out).

    Layout: carries 0..n-1, a n..2n-1, b 2n..3n (b[n] is the carry-out). 3n+1 qubits,
    Toffoli/CNOT only.  Carry wires start and end in |0>.
    """
    if not isinstance(n_bits, int) or n_bits < 1:
        raise ValueError("n_bits must be a positive integer")
    n = n_bits
    cw, aw, bw = list(range(n)), list(range(n, 2 * n)), list(range(2 * n, 3 * n + 1))
    c = Circuit(3 * n + 1)
    for i in range(n - 1):
        _carry(c, cw[i], aw[i], bw[i], cw[i + 1])
    _carry(c, cw[n - 1], aw[n - 1], bw[n - 1], bw[n])
    c.cx(aw[n - 1], bw[n - 1])
    _sum(c, cw[n - 1], aw[n - 1], bw[n - 1])
    for i in range(n - 2, -1, -1):
        _carry_inv(c, cw[i], aw[i], bw[i], cw[i + 1])
        _sum(c, cw[i], aw[i], bw[i])
    return AdderCircuit(c, aw, bw, cw)


# ------------------------------------------------------------------ Draper adder
def qft_circuit(n: int, wires=None, swaps: bool = True) -> Circuit:
    """Standard QFT on ``wires`` (MSB first), equal to the n-qubit DFT matrix."""
    wires = list(range(n)) if wires is None else list(wires)
    c = Circuit(max(wires) + 1)
    m = len(wires)
    for j in range(m):
        c.h(wires[j])
        for k in range(j + 1, m):
            c.cp(math.pi / 2 ** (k - j), wires[k], wires[j])
    if swaps:
        for j in range(m // 2):
            c.swap(wires[j], wires[m - 1 - j])
    return c


def qft_adder(n_bits: int) -> AdderCircuit:
    """Draper adder: b <- (a + b) mod 2**n using phase rotations in the Fourier basis.

    Layout: a 0..n-1, b n..2n-1 (both LSB first).  2n qubits, no ancilla, no Toffoli.
    """
    if not isinstance(n_bits, int) or n_bits < 1:
        raise ValueError("n_bits must be a positive integer")
    n = n_bits
    aw, bw = list(range(n)), list(range(n, 2 * n))
    b_msb = bw[::-1]                      # QFT works MSB first
    c = Circuit(2 * n)
    qft = qft_circuit(n, b_msb)
    c.ops.extend(qft.ops)
    for j in range(n):                    # j-th qubit of b (MSB first) carries phase 2*pi*x/2**(j+1)
        for i in range(n):                # bit i of a has weight 2**i
            e = (j + 1) - i
            if e >= 1:
                c.cp(2 * math.pi / 2 ** e, aw[i], b_msb[j])
    c.ops.extend(qft.inverse().ops)
    return AdderCircuit(c, aw, bw, [])


# --------------------------------------------------------------------- front door
def _bits(value: int, n: int) -> list:
    return [(value >> i) & 1 for i in range(n)]


def add(a: int, b: int, n_bits: int, method: str = "ripple", backend: str | None = None) -> int:
    """Add two non-negative integers < 2**n_bits on a quantum circuit.

    method 'ripple': exact result a+b (up to n_bits+1 bits).
    method 'qft'   : (a+b) mod 2**n_bits.
    """
    for name, v in (("a", a), ("b", b)):
        if not isinstance(v, int) or isinstance(v, bool) or not 0 <= v < 2 ** n_bits:
            raise ValueError(f"{name} must be an integer in [0, 2**{n_bits})")
    if method == "ripple":
        adder = ripple_carry_adder(n_bits)
    elif method == "qft":
        adder = qft_adder(n_bits)
    else:
        raise ValueError("method must be 'ripple' or 'qft'")
    init = ["0"] * adder.n_qubits
    for w, bit in zip(adder.a, _bits(a, n_bits)):
        init[w] = str(bit)
    for w, bit in zip(adder.b[:n_bits], _bits(b, n_bits)):
        init[w] = str(bit)
    res = adder.circuit.run(backend=backend, initial="".join(init))
    top = res.most_likely()
    return sum(int(top[w]) << i for i, w in enumerate(adder.b))
