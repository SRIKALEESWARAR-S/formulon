"""Backend-neutral quantum circuit (gate list) with three executors.

Build once, run on the built-in NumPy simulator, PennyLane or Qiskit::

    c = Circuit(2).h(0).cx(0, 1)
    c.run()                       # preferred backend (see set_backend)
    c.run(backend="qiskit")       # explicit
    c.to_pennylane()              # callable to use inside your own qnode
    c.to_qiskit()                 # qiskit.QuantumCircuit

Gate methods take parameters first, then wires, like Qiskit: ``c.cp(phi, ctrl, tgt)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backend import resolve_backend
from .gates import GATES, gate_matrix, inverse_gate

__all__ = ["Circuit", "Result"]


@dataclass
class Result:
    probabilities: dict          # {'0101': p, ...}  wire 0 = left-most bit
    backend: str
    n_qubits: int
    statevector: np.ndarray | None = None   # only the numpy backend provides amplitudes

    def most_likely(self) -> str:
        return max(self.probabilities, key=self.probabilities.get)

    def sample(self, shots: int = 1024, seed: int | None = None) -> dict:
        rng = np.random.default_rng(seed)
        keys = list(self.probabilities)
        p = np.array([self.probabilities[k] for k in keys])
        counts = rng.multinomial(shots, p / p.sum())
        return {k: int(c) for k, c in zip(keys, counts) if c}

    def marginal(self, wires) -> dict:
        out: dict = {}
        for bits, p in self.probabilities.items():
            key = "".join(bits[w] for w in wires)
            out[key] = out.get(key, 0.0) + p
        return out


class Circuit:
    def __init__(self, n_qubits: int):
        if not isinstance(n_qubits, int) or isinstance(n_qubits, bool) or n_qubits < 1:
            raise ValueError("n_qubits must be a positive integer")
        self.n_qubits = n_qubits
        self.ops: list = []          # (name, wires tuple, params tuple)

    # ---------------------------------------------------------------- building
    def gate(self, name: str, *args):
        name = name.upper()
        if name not in GATES:
            raise KeyError(f"unknown gate {name!r}; available: {sorted(GATES)}")
        spec = GATES[name]
        params, wires = tuple(float(a) for a in args[:spec.n_params]), tuple(args[spec.n_params:])
        if spec.n_wires is not None and len(wires) != spec.n_wires:
            raise ValueError(f"{name} needs {spec.n_wires} wire(s), got {len(wires)}")
        if not wires:
            raise ValueError(f"{name} needs at least one wire")
        for w in wires:
            if not isinstance(w, (int, np.integer)) or isinstance(w, bool) or not 0 <= w < self.n_qubits:
                raise ValueError(f"wire {w!r} outside 0..{self.n_qubits - 1}")
        if len(set(wires)) != len(wires):
            raise ValueError(f"{name}: wires must be distinct, got {wires}")
        self.ops.append((name, tuple(int(w) for w in wires), params))
        return self

    def extend(self, other: "Circuit", wire_map=None):
        """Append another circuit, optionally remapping its wires."""
        wire_map = list(wire_map) if wire_map is not None else list(range(other.n_qubits))
        for name, wires, params in other.ops:
            self.gate(name, *params, *[wire_map[w] for w in wires])
        return self

    def inverse(self) -> "Circuit":
        inv = Circuit(self.n_qubits)
        for name, wires, params in reversed(self.ops):
            n2, p2 = inverse_gate(name, params)
            inv.ops.append((n2, wires, p2))
        return inv

    def with_basis_state(self, bits: str) -> "Circuit":
        """Copy of this circuit preceded by X gates preparing basis state ``bits`` (wire 0 first)."""
        if len(bits) != self.n_qubits or set(bits) - {"0", "1"}:
            raise ValueError(f"bits must be a {self.n_qubits}-char 0/1 string")
        c = Circuit(self.n_qubits)
        for w, b in enumerate(bits):
            if b == "1":
                c.gate("X", w)
        c.ops.extend(self.ops)
        return c

    def count_ops(self) -> dict:
        out: dict = {}
        for name, _w, _p in self.ops:
            out[name] = out.get(name, 0) + 1
        return out

    def draw(self) -> str:
        return "\n".join(f"{n}{'(' + ', '.join(f'{p:.4g}' for p in pr) + ')' if pr else ''} {list(w)}"
                         for n, w, pr in self.ops)

    # ------------------------------------------------------------- numpy backend
    def statevector(self, initial: np.ndarray | None = None) -> np.ndarray:
        n = self.n_qubits
        if initial is None:
            state = np.zeros((2,) * n, dtype=complex)
            state[(0,) * n] = 1.0
        else:
            state = np.asarray(initial, dtype=complex).reshape((2,) * n).copy()
        for name, wires, params in self.ops:
            if name == "MCZ":
                idx = [slice(None)] * n
                for w in wires:
                    idx[w] = 1
                state[tuple(idx)] *= -1
                continue
            k = len(wires)
            g = gate_matrix(name, *params).reshape((2,) * (2 * k))
            state = np.tensordot(g, state, axes=(list(range(k, 2 * k)), list(wires)))
            state = np.moveaxis(state, list(range(k)), list(wires))
        return state.reshape(-1)

    # --------------------------------------------------------- pennylane backend
    def to_pennylane(self):
        """Return ``apply()`` that queues this circuit's operations (use inside a qnode)."""
        try:
            import pennylane as qml
        except ImportError as exc:
            raise ImportError('PennyLane is required: pip install "formulon-physics[pennylane]"') from exc
        ops = list(self.ops)

        def apply():
            for name, wires, params in ops:
                spec = GATES[name]
                if name == "MCZ":
                    if len(wires) == 1:
                        qml.PauliZ(wires=wires[0])
                    else:
                        qml.ctrl(qml.PauliZ(wires=wires[-1]), control=list(wires[:-1]))
                elif spec.pennylane.startswith("adjoint:"):
                    qml.adjoint(getattr(qml, spec.pennylane.split(":")[1]))(wires=wires[0])
                else:
                    op = getattr(qml, spec.pennylane)
                    w = wires[0] if len(wires) == 1 else list(wires)
                    op(*params, wires=w)
        return apply

    def _run_pennylane(self):
        import pennylane as qml
        dev = qml.device("default.qubit", wires=self.n_qubits)
        apply = self.to_pennylane()

        @qml.qnode(dev)
        def f():
            apply()
            return qml.probs(wires=list(range(self.n_qubits)))
        return np.asarray(f(), dtype=float)

    # ------------------------------------------------------------ qiskit backend
    def to_qiskit(self):
        try:
            from qiskit import QuantumCircuit
        except ImportError as exc:
            raise ImportError('Qiskit is required: pip install "formulon-physics[qiskit]"') from exc
        qc = QuantumCircuit(self.n_qubits)
        for name, wires, params in self.ops:
            if name == "MCZ":
                t, ctrls = wires[-1], list(wires[:-1])
                if not ctrls:
                    qc.z(t)
                elif len(ctrls) == 1:
                    qc.cz(ctrls[0], t)
                else:
                    qc.h(t); qc.mcx(ctrls, t); qc.h(t)
            else:
                getattr(qc, GATES[name].qiskit)(*params, *wires)
        return qc

    def _run_qiskit(self):
        from qiskit.quantum_info import Statevector
        p_le = Statevector.from_instruction(self.to_qiskit()).probabilities()   # little-endian index
        n = self.n_qubits
        out = np.zeros(2 ** n)
        for i, p in enumerate(p_le):
            out[int(format(i, f"0{n}b")[::-1], 2)] = p          # reverse bits -> wire-0-first order
        return out

    def state(self, backend: str | None = None) -> np.ndarray:
        """Final statevector (wire 0 = most significant bit) on the chosen backend."""
        name = resolve_backend(backend)
        if name == "numpy":
            return self.statevector()
        if name == "pennylane":
            import pennylane as qml
            dev = qml.device("default.qubit", wires=self.n_qubits)
            apply = self.to_pennylane()

            @qml.qnode(dev)
            def f():
                apply()
                return qml.state()
            return np.asarray(f(), dtype=complex)
        from qiskit.quantum_info import Statevector
        sv = np.asarray(Statevector.from_instruction(self.to_qiskit()).data, dtype=complex)
        n = self.n_qubits
        out = np.zeros_like(sv)
        for i, a in enumerate(sv):
            out[int(format(i, f"0{n}b")[::-1], 2)] = a
        return out

    # ----------------------------------------------------------------------- run
    def run(self, backend: str | None = None, initial: str | None = None) -> Result:
        """Execute exactly (no shot noise) and return probabilities, wire 0 = left-most bit.

        ``initial`` is an optional basis-state bitstring such as ``"0110"``.
        Use ``Result.sample(shots)`` for finite-shot counts.
        """
        circ = self.with_basis_state(initial) if initial else self
        name = resolve_backend(backend)
        n = self.n_qubits
        sv = None
        if name == "numpy":
            sv = circ.statevector()
            probs = np.abs(sv) ** 2
        elif name == "pennylane":
            probs = circ._run_pennylane()
        else:
            probs = circ._run_qiskit()
        d = {format(i, f"0{n}b"): float(p) for i, p in enumerate(probs) if p > 1e-12}
        return Result(d, name, n, sv)


# ---- fluent gate methods on Circuit (parameters first, then wires) ------------
def _install_methods():
    def make(name, spec):
        def method(self, *args):
            return self.gate(name, *args)
        method.__name__ = name.lower()
        method.__doc__ = f"Append {name} (params first, then wires)."
        return method

    for name, spec in GATES.items():
        attr = {"CX": "cx", "CCX": "ccx"}.get(name, name.lower())
        if not hasattr(Circuit, attr):
            setattr(Circuit, attr, make(name, spec))


_install_methods()
