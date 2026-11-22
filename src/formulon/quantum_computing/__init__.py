"""Quantum computing tools for Formulon Physics.

Importing this package needs neither PennyLane nor Qiskit.  Pick a backend with
``set_backend("pennylane" | "qiskit" | "numpy" | "auto")``.
"""
from .backend import set_backend, get_backend, use_backend, available_backends, resolve_backend
from .circuit import Circuit, Result
from .gates import GATES, gate_matrix
from .adders import (half_adder, full_adder, ripple_carry_adder, qft_adder, qft_circuit, add,
                     AdderCircuit)
from .legendre_encoder import (legendre_features, encode_legendre, decode_legendre,
                               measure_and_decode, legendre_feature_expectation, LegendreMeasurement)
from .grover import grover_search, grover_circuit, GroverResult

__all__ = [
    "set_backend", "get_backend", "use_backend", "available_backends", "resolve_backend",
    "Circuit", "Result", "GATES", "gate_matrix",
    "half_adder", "full_adder", "ripple_carry_adder", "qft_adder", "qft_circuit", "add", "AdderCircuit",
    "grover_search", "grover_circuit", "GroverResult",
    "legendre_features", "encode_legendre", "decode_legendre", "measure_and_decode",
    "legendre_feature_expectation", "LegendreMeasurement",
]


def __getattr__(name):          # lazy: these import PennyLane / Qiskit
    if name in ("pennylane_circuits", "qiskit_circuits", "pennylane_gates", "pennylane_adders"):
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(name)
