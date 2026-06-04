# SPDX-FileCopyrightText: 2026-present SRIKALEESWARAR-S <srikaleeswarar675@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0
"""
formulon
========

An open-source Python library providing physics formulas with
automatic input validation and NumPy support.

Designed for educational, scientific, and engineering use.

Beta version count =2
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
__all__ = [
    "__version__",
]

