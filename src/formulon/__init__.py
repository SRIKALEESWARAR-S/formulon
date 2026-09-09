# SPDX-FileCopyrightText: 2026-present SRIKALEESWARAR-S <srikaleeswarar675@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0
"""
formulon
========

An open-source Python library providing physics formulas with
automatic input validation and NumPy support.

Designed for educational, scientific, and engineering use.

An optional quantum-computing extension (Qiskit- and PennyLane-backed
circuit demos, including a Legendre-polynomial quantum feature map/kernel)
lives in `formulon.quantum_computing` and is NOT imported here -- see that
subpackage's docstring for installation instructions.
"""

from .__about__ import __version__

# --- Public physics formula modules ---
from .classicalmechanics import *
from .wavedynamics import *
from .thermodynamics import *
from .modernphysics import *
from .mathematicalphysics import *
from .electromagnetism import *
from .fluidmechanics import *
from .mathematicalseries import *

# --- Public API ---
# Module-level star imports above populate the package namespace.
# Keep __version__ explicitly public; imported formula names remain available
# as attributes of the package.
__all__ = [name for name in globals() if not name.startswith("_")]


from .formula_catalog import FORMULA_COUNT, FORMULAS, get_formula
