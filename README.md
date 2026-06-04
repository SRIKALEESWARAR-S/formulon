# Formulon

[![PyPI version](https://img.shields.io/pypi/v/formulon.svg)](https://pypi.org/project/formulon/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Support](https://img.shields.io/pypi/pyversions/formulon.svg)](https://pypi.org/project/formulon/)
[![GSoC Eligible](https://img.shields.io/badge/GSoC-Ready-orange.svg)](#)

An optimized, high-performance open-source computational library for mathematical physics, classical mechanics, electronics, wave optics, electromagnetism, and modern quantum frameworks. 

`formulon` bridges the gap between pure theoretical formulations and industry-grade numerical execution. By leveraging hardware-accelerated vectors via **NumPy**, specialized integrations via **SciPy**, and runtime machine-code serialization via **Numba JIT**, `formulon` delivers execution speeds comparable to compiled C/C++ while exposing a clean, Pythonic API.

---

## 🚀 Key Features

* **Advanced Mathematical Physics:** Integrated solvers for Fourier transforms, tensor contractions, vector calculus (Stokes, Green, Divergence theorems), and partial differential equations (Heat, Wave equations).
* **High-Performance Execution:** Performance-critical execution paths are accelerated via `numba` Just-In-Time (`@njit`) compilation, minimizing operational overhead for dense loops and data-intensive pipelines.
* **Production-Grade Verification Layer:** Decoupled functional boundary assertion utilizing a decorator-driven `validators` layer ensuring parameter dimensions never breach strict physical constants.
* **Unified API Design:** Clean, direct physical nomenclature (omitting redundant prefixes like `calculate_` or structural suffixes like `_law`) with absolute public registration catalogs (`__all__`).

---

## 📦 Project Architecture

The codebase is engineered modularly to segregate distinct physics domains into dedicated computational engines:

```text
src/formulon/
├── __init__.py                            # Package namespace constructor
├── __about__.py                           # Dynamic version registry (Hatch)
├── validators.py                          # Bound assertions & boundary constraints
├── mathematical_physics.py                # Vectors, tensors, Fourier, and PDE solvers
├── numerical_series_modules.py            # Hardware-accelerated progression limits
├── waves_oscillations_optics.py           # Wave mechanics, acoustics, and ray optics
├── thermodynamics_statistical_mechanics.py # Thermal expansion, kinetic theory, and cycles
├── electromagnetism.py                    # Electrostatics, magnetostatics, and Maxwell equations
└── modern_physics_quantum_relativity.py  # Relativistic dynamics and quantum mechanics
