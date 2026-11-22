"""Basic quantum gates: reference matrices + backend name tables.

The NumPy matrices are the ground truth used by the built-in simulator and by
the test-suite to verify the PennyLane and Qiskit translations.
Wire order convention everywhere in Formulon: wire 0 is the left-most bit of a
bitstring (PennyLane convention); Qiskit results are re-ordered to match.
"""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

import numpy as np

__all__ = ["GateSpec", "GATES", "gate_matrix", "inverse_gate"]

_s2 = 1 / math.sqrt(2)
I2 = np.eye(2, dtype=complex)


def _ctrl(u: np.ndarray, n_ctrl: int = 1) -> np.ndarray:
    d = 2 ** n_ctrl * u.shape[0]
    m = np.eye(d, dtype=complex)
    m[-u.shape[0]:, -u.shape[0]:] = u
    return m


X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = _s2 * np.array([[1, 1], [1, -1]], dtype=complex)
S = np.diag([1, 1j]).astype(complex)
T = np.diag([1, cmath.exp(1j * math.pi / 4)]).astype(complex)
SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex)


def _rx(t): return np.array([[math.cos(t / 2), -1j * math.sin(t / 2)], [-1j * math.sin(t / 2), math.cos(t / 2)]], dtype=complex)
def _ry(t): return np.array([[math.cos(t / 2), -math.sin(t / 2)], [math.sin(t / 2), math.cos(t / 2)]], dtype=complex)
def _rz(t): return np.diag([cmath.exp(-1j * t / 2), cmath.exp(1j * t / 2)]).astype(complex)
def _p(phi): return np.diag([1, cmath.exp(1j * phi)]).astype(complex)
def _u(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -cmath.exp(1j * lam) * s],
                     [cmath.exp(1j * phi) * s, cmath.exp(1j * (phi + lam)) * c]], dtype=complex)


SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
ISWAP = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex)


@dataclass(frozen=True)
class GateSpec:
    name: str
    n_wires: int | None          # None = variable (MCZ)
    n_params: int
    matrix: object               # ndarray or callable(*params) -> ndarray
    pennylane: str               # attribute name in `pennylane` (or special-cased)
    qiskit: str                  # QuantumCircuit method name


def _g(name, n_w, n_p, m, pl, qk): return GateSpec(name, n_w, n_p, m, pl, qk)


GATES = {g.name: g for g in [
    # single-qubit, fixed
    _g("I", 1, 0, I2, "Identity", "id"), _g("X", 1, 0, X, "PauliX", "x"),
    _g("Y", 1, 0, Y, "PauliY", "y"), _g("Z", 1, 0, Z, "PauliZ", "z"),
    _g("H", 1, 0, H, "Hadamard", "h"), _g("S", 1, 0, S, "S", "s"),
    _g("SDG", 1, 0, S.conj().T, "adjoint:S", "sdg"), _g("T", 1, 0, T, "T", "t"),
    _g("TDG", 1, 0, T.conj().T, "adjoint:T", "tdg"), _g("SX", 1, 0, SX, "SX", "sx"),
    _g("SXDG", 1, 0, SX.conj().T, "adjoint:SX", "sxdg"),
    # single-qubit, parametrised
    _g("RX", 1, 1, _rx, "RX", "rx"), _g("RY", 1, 1, _ry, "RY", "ry"), _g("RZ", 1, 1, _rz, "RZ", "rz"),
    _g("P", 1, 1, _p, "PhaseShift", "p"), _g("U", 1, 3, _u, "U3", "u"),
    # two-qubit
    _g("CX", 2, 0, _ctrl(X), "CNOT", "cx"), _g("CY", 2, 0, _ctrl(Y), "CY", "cy"),
    _g("CZ", 2, 0, _ctrl(Z), "CZ", "cz"), _g("SWAP", 2, 0, SWAP, "SWAP", "swap"),
    _g("ISWAP", 2, 0, ISWAP, "ISWAP", "iswap"),
    _g("CP", 2, 1, lambda a: _ctrl(_p(a)), "ControlledPhaseShift", "cp"),
    _g("CRX", 2, 1, lambda a: _ctrl(_rx(a)), "CRX", "crx"),
    _g("CRY", 2, 1, lambda a: _ctrl(_ry(a)), "CRY", "cry"),
    _g("CRZ", 2, 1, lambda a: _ctrl(_rz(a)), "CRZ", "crz"),
    # three-qubit
    _g("CCX", 3, 0, _ctrl(X, 2), "Toffoli", "ccx"), _g("CCZ", 3, 0, _ctrl(Z, 2), "CCZ", "ccz"),
    _g("CSWAP", 3, 0, _ctrl(SWAP), "CSWAP", "cswap"),
    # variable arity: Z on last wire controlled on all the others
    _g("MCZ", None, 0, None, "mcz", "mcz"),
]}

_SELF_INVERSE = {"I", "X", "Y", "Z", "H", "CX", "CY", "CZ", "SWAP", "CCX", "CCZ", "CSWAP", "MCZ"}
_PAIRS = {"S": "SDG", "SDG": "S", "T": "TDG", "TDG": "T", "SX": "SXDG", "SXDG": "SX"}
_NEG = {"RX", "RY", "RZ", "P", "CP", "CRX", "CRY", "CRZ"}


def gate_matrix(name: str, *params: float) -> np.ndarray:
    spec = GATES[name]
    if spec.n_wires is None:
        raise ValueError("MCZ has no fixed-size matrix")
    if len(params) != spec.n_params:
        raise ValueError(f"{name} takes {spec.n_params} parameter(s), got {len(params)}")
    m = spec.matrix(*params) if callable(spec.matrix) else spec.matrix
    return np.asarray(m, dtype=complex)


def inverse_gate(name: str, params: tuple):
    """(name, params) of the inverse gate."""
    if name in _SELF_INVERSE:
        return name, params
    if name in _PAIRS:
        return _PAIRS[name], params
    if name in _NEG:
        return name, tuple(-p for p in params)
    if name == "U":
        t, p, l = params
        return "U", (-t, -l, -p)
    raise NotImplementedError(f"inverse of {name} is not provided")
