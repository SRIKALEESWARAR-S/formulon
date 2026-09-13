# SPDX-FileCopyrightText: 2026-present SRIKALEESWARAR-S <srikaleeswarar675@gmail.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
from ..mathematicalphysics import legendre_polynomial

def legendre_feature_map(x: float, degree: int) -> QuantumCircuit:
    if degree < 1:
        raise ValueError("degree must be at least 1.")

    qc = QuantumCircuit(degree)
    for k in range(degree):
        p_k = np.clip(legendre_polynomial(k, x), -1.0, 1.0)
        theta_k = np.arccos(p_k)
        qc.ry(theta_k, k)
    return qc

def quantum_fourier_transform(num_qubits: int) -> QuantumCircuit:
    if num_qubits < 1:
        raise ValueError("num_qubits must be at least 1.")

    qc = QuantumCircuit(num_qubits, name=f"QFT({num_qubits})")
    for j in range(num_qubits):
        qc.h(j)
        for k in range(j + 1, num_qubits):
            angle = math.pi / (2 ** (k - j))
            qc.cp(angle, k, j)
            
    for i in range(num_qubits // 2):
        qc.swap(i, num_qubits - i - 1)
    return qc

def grover_two_qubit_search(marked_state: str, shots: int = 1024) -> dict[str, int]:
    if marked_state not in ("00", "01", "10", "11"):
        raise ValueError('marked_state must be a 2-bit string.')

    qc = QuantumCircuit(2)
    qc.h([0, 1])

    # Oracle
    flip_qubits = [i for i, bit in enumerate(reversed(marked_state)) if bit == "0"]
    if flip_qubits:
        qc.x(flip_qubits)
    qc.cz(0, 1)
    if flip_qubits:
        qc.x(flip_qubits)

    # Diffusion (Inversion about mean)
    qc.h([0, 1])
    qc.z([0, 1])
    qc.cz(0, 1)
    qc.h([0, 1])
    
    qc.measure_all()
    sampler = StatevectorSampler()
    job = sampler.run([qc], shots=shots)
    return dict(job.result()[0].data.meas.get_counts())
