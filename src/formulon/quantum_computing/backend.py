"""Choose the quantum backend with a few commands.

    from formulon.quantum_computing import set_backend, use_backend, available_backends
    set_backend("pennylane")        # or "qiskit", "numpy", "auto"
    with use_backend("qiskit"):     # temporary
        ...
    circuit.run(backend="qiskit")   # per call (highest priority)

Priority: per-call > ``use_backend`` > ``set_backend`` > ``FORMULON_QUANTUM_BACKEND`` > auto
(PennyLane, then Qiskit, then the built-in NumPy simulator).  An explicit choice never
falls back silently; a missing package raises ImportError with the install command.
"""
from __future__ import annotations

from .._switch import BackendSwitch

__all__ = ["set_backend", "get_backend", "use_backend", "available_backends", "resolve_backend",
           "BACKENDS"]

BACKENDS = ("pennylane", "qiskit", "numpy")

_switch = BackendSwitch(
    "quantum", BACKENDS, "FORMULON_QUANTUM_BACKEND", ("pennylane", "qiskit", "numpy"),
    {"pennylane": "pennylane", "qiskit": "qiskit", "numpy": None},
    {"pennylane": 'pip install "formulon-physics[pennylane]"',
     "qiskit": 'pip install "formulon-physics[qiskit]"'})

set_backend = _switch.set
get_backend = _switch.get
use_backend = _switch.use
available_backends = _switch.available
resolve_backend = _switch.resolve
