"""Legendre-polynomial quantum encoder/decoder using PennyLane.

The encoder maps a classical scalar x in [xmin, xmax] to a sequence of
Legendre modes P_0(x'), ..., P_{d-1}(x'), where x' is linearly mapped to
[-1, 1]. Each mode controls an RY rotation.  The decoder estimates the modes
from Z expectations and reconstructs x' by bounded least squares.

``measure_and_decode`` also samples the computational basis.  Measurement is
where a quantum state is projected onto an observed basis outcome (often
informally called wave-function collapse).  The implementation does not claim
that collapse is an additional computational operation beyond measurement.
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

__all__ = [
    "LegendreMeasurement",
    "legendre_features",
    "encode_legendre",
    "decode_legendre",
    "measure_and_decode",
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


def encode_legendre(x: float, degree: int, xmin: float = -1.0, xmax: float = 1.0):
    """Prepare a differentiable PennyLane circuit that encodes Legendre modes."""
    _check_degree(degree)
    features = legendre_features(x, degree, xmin, xmax)
    angles = _angles_from_features(features)
    dev = qml.device("default.qubit", wires=degree)

    @qml.qnode(dev)
    def circuit():
        for wire, theta in enumerate(angles):
            qml.RY(theta, wires=wire)
        return qml.state()

    return circuit()


def _expectation_decode(xmin: float, xmax: float, features: np.ndarray) -> float:
    """Recover x' from noisy estimates of Legendre features."""
    degree = len(features)
    grid = np.linspace(-1.0, 1.0, 2001)
    design = np.stack([np.asarray([1.0] if n == 0 else []) for n in range(degree)]) if False else None
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
                       seed: int | None = None) -> LegendreMeasurement:
    """Encode x, measure the qubits, and decode x from Z expectations.

    The returned samples are computational-basis outcomes; each sample is a
    concrete measurement result, representing the usual measurement-induced
    projection of the pre-measurement state.
    """
    _check_degree(degree)
    if not isinstance(shots, int) or isinstance(shots, bool) or shots < 1:
        raise ValueError("shots must be a positive integer")
    features = legendre_features(x, degree, xmin, xmax)
    angles = _angles_from_features(features)
    dev = qml.device("default.qubit", wires=degree, shots=shots, seed=seed)

    @qml.qnode(dev)
    def circuit():
        for wire, theta in enumerate(angles):
            qml.RY(theta, wires=wire)
        return qml.sample(wires=range(degree))

    raw = np.asarray(circuit(), dtype=int)
    # For computational bit b, Z expectation is 1-2*P(b=1).
    z = tuple(float(1.0 - 2.0 * np.mean(raw[:, wire])) for wire in range(degree))
    decoded = decode_legendre(z, xmin, xmax)
    samples = tuple(tuple(int(v) for v in row) for row in raw)
    return LegendreMeasurement(float(x), float(decoded), samples, z)
