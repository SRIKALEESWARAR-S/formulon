# SPDX-FileCopyrightText: 2026-present SRIKALEESWARAR-S <srikaleeswarar675@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0
"""
formulon.quantum_computing
===========================
Optional quantum-computing extension for Project Formulon-Physics.

This subpackage is intentionally NOT imported by formulon's top-level
``__init__.py`` and is NOT installed by default -- it depends on Qiskit
and/or PennyLane, which are heavy, fast-moving SDKs that most users of the
classical formula library will never need.

Install what you need:

    pip install "formulon-physics[qiskit]"      # Qiskit-backed circuits only
    pip install "formulon-physics[pennylane]"   # PennyLane-backed circuits only
    pip install "formulon-physics[quantum]"     # both

Then import explicitly:

    from formulon.quantum_computing import qiskit_circuits
    from formulon.quantum_computing import pennylane_circuits

Design notes
------------
* Everything here runs on LOCAL SIMULATORS ONLY (statevector simulation).
  No QPU/hardware access, no API keys, no network calls.
* The headline feature is a small, original teaching demo: a "Legendre
  feature map" that encodes a classical Legendre-polynomial expansion
  (formulon.legendre_polynomial) into qubit rotation angles, plus a
  fidelity-based quantum kernel built on top of it. This mirrors real
  quantum-kernel-methods literature (orthogonal-polynomial feature maps)
  and gives a concrete bridge between formulon's existing classical
  special-function code and quantum machine learning.
* Each submodule fails fast with a clear pip-install message if its SDK
  isn't installed, rather than a confusing ImportError deep in some
  unrelated stack trace.

Project: Project Formulon-Physics
License: Apache-2.0
"""

__all__: list[str] = []
