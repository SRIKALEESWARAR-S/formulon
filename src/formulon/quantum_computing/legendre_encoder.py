"""Legendre-polynomial quantum encoder/decoder (backend-switchable).

The encoder maps a classical scalar x in [xmin, xmax] to Legendre modes
P_0(x'), ..., P_{d-1}(x') (x' = x mapped linearly to [-1, 1]); each mode sets an RY rotation
so that <Z> = P_n.  The decoder estimates the modes from Z expectations and recovers x' by a
bounded least-squares search.  Every function takes ``backend=None`` (PennyLane, Qiskit or the
NumPy simulator; see ``formulon.quantum_computing.set_backend``).

``measure_and_decode`` samples the computational basis; each sample is a concrete
measurement outcome (the usual measurement-induced projection of the state).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .circuit import Circuit

__all__ = [
    "LegendreMeasurement",
    "legendre_features",
    "encode_legendre",
    "decode_legendre",
    "measure_and_decode",
    "legendre_feature_expectation",
]


def _check_degree(degree: int) -> None:
    if not isinstance(degree, int) or isinstance(degree, bool) or degree < 1:
        raise ValueError("degree must be a positive integer")


def _scale_to_legendre(x: float, xmin: float, xmax: float) -> float:
    if not all(math.isfinite(float(v)) for v in (x, xmin, xmax)):
        raise ValueError("x, xmin and xmax must be finite")
    if xmax <= xmin:
        raise ValueError("xmax must be greater than xmin")
    if not xmin <= x <= xmax:
        raise ValueError(f"x must lie in [{xmin}, {xmax}]")
    return 2.0 * (x - xmin) / (xmax - xmin) - 1.0


def _unscale_from_legendre(xp: float, xmin: float, xmax: float) -> float:
    return xmin + (xp + 1.0) * 0.5 * (xmax - xmin)


def legendre_features(x: float, degree: int, xmin: float = -1.0, xmax: float = 1.0) -> np.ndarray:
    """Return [P0(x'), ..., P(degree-1)(x')] for scaled x'."""
    _check_degree(degree)
    xp = _scale_to_legendre(float(x), xmin, xmax)
    values = [1.0, xp] if degree > 1 else [1.0]
    for n in range(2, degree):
        values.append(((2*n - 1) * xp * values[-1] - (n - 1) * values[-2]) / n)
    return np.asarray(values[:degree], dtype=float)


def _angles_from_features(features: np.ndarray) -> np.ndarray:
    # Choose theta=arccos(P_n), so <Z>=cos(theta)=P_n for RY|0>.
    return np.arccos(np.clip(features, -1.0, 1.0))


def _encoder_circuit(angles) -> Circuit:
    c = Circuit(len(angles))
    for wire, theta in enumerate(angles):
        c.ry(float(theta), wire)
    return c


def encode_legendre(x: float, degree: int, xmin: float = -1.0, xmax: float = 1.0,
                    backend: str | None = None):
    """Statevector of the circuit that encodes the Legendre modes of x (wire 0 = MSB)."""
    _check_degree(degree)
    angles = _angles_from_features(legendre_features(x, degree, xmin, xmax))
    return _encoder_circuit(angles).state(backend)


def legendre_feature_expectation(x: float, degree: int, backend: str | None = None) -> float:
    """<Z x Z x ... x Z> of the encoded state, which equals prod_k P_k(x) for x in [-1, 1]."""
    _check_degree(degree)
    angles = _angles_from_features(legendre_features(x, degree))
    probs = _encoder_circuit(angles).run(backend=backend).probabilities
    return float(sum(p * (-1) ** bits.count("1") for bits, p in probs.items()))


def _expectation_decode(xmin: float, xmax: float, features: np.ndarray) -> float:
    """Recover x' from noisy estimates of Legendre features."""
    degree = len(features)
    grid = np.linspace(-1.0, 1.0, 2001)
    basis = np.empty((grid.size, degree))
    basis[:, 0] = 1.0
    if degree > 1:
        basis[:, 1] = grid
    for n in range(2, degree):
        basis[:, n] = ((2*n - 1) * grid * basis[:, n-1] - (n - 1) * basis[:, n-2]) / n
    # Least-squares projection onto the one-dimensional Legendre curve.
    errors = np.sum((basis - np.asarray(features)[None, :]) ** 2, axis=1)
    return _unscale_from_legendre(float(grid[int(np.argmin(errors))]), xmin, xmax)


def decode_legendre(measured_features, xmin: float = -1.0, xmax: float = 1.0) -> float:
    """Decode an array of estimated P_n values into x."""
    values = np.asarray(measured_features, dtype=float)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("measured_features must be a one-dimensional non-empty sequence")
    if not np.all(np.isfinite(values)):
        raise ValueError("measured_features must be finite")
    if np.any(values < -1.000001) or np.any(values > 1.000001):
        raise ValueError("Legendre feature estimates must lie in [-1, 1]")
    return _expectation_decode(xmin, xmax, np.clip(values, -1.0, 1.0))


@dataclass(frozen=True)
class LegendreMeasurement:
    """Finite-shot measurement record and decoded estimate."""
    x_input: float
    decoded_x: float
    samples: tuple[tuple[int, ...], ...]
    z_expectations: tuple[float, ...]


def measure_and_decode(x: float, degree: int, shots: int = 1000,
                       xmin: float = -1.0, xmax: float = 1.0,
                       seed: int | None = None, backend: str | None = None) -> LegendreMeasurement:
    """Encode x, measure the qubits ``shots`` times, and decode x from Z expectations.

    Outcome probabilities come from the chosen backend; shots are drawn with a seeded NumPy
    generator, so a given ``seed`` gives identical samples on every backend.
    """
    _check_degree(degree)
    if not isinstance(shots, int) or isinstance(shots, bool) or shots < 1:
        raise ValueError("shots must be a positive integer")
    features = legendre_features(x, degree, xmin, xmax)
    angles = _angles_from_features(features)
    result = _encoder_circuit(angles).run(backend=backend)
    counts = result.sample(shots, seed)
    rows = [tuple(int(ch) for ch in bits) for bits, n in sorted(counts.items()) for _ in range(n)]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    raw = np.asarray([rows[i] for i in order], dtype=int)
    # For computational bit b, Z expectation is 1-2*P(b=1).
    z = tuple(float(1.0 - 2.0 * np.mean(raw[:, wire])) for wire in range(degree))
    decoded = decode_legendre(z, xmin, xmax)
    samples = tuple(tuple(int(v) for v in row) for row in raw)
    return LegendreMeasurement(float(x), float(decoded), samples, z)
