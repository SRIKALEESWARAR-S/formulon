# Changelog

## 0.6.6

- Removed the broken legacy `classical_physics.py` module from the distribution.
- Added PennyLane Grover search with validated integer targets and automatic iteration count.
- Added a Legendre-polynomial quantum encoder/decoder using `RY` feature rotations.
- Added finite-shot measurement and measurement-induced projection documentation.
- Fixed optional `quantum` dependency metadata to declare Qiskit and PennyLane directly.
- Added clean import and quantum tests that skip gracefully when PennyLane is absent.
