# Changelog

## [1.0.0] - 2026-09-24

### Added

- Quantum-Informed Physical AI Engine
- Hybrid quantum-classical physical feature pipeline
- Noisy quantum simulation
- Physics-feature reconstruction
- Quantum-assisted classification
- Validation and early stopping
- Quantum/noise evaluation framework

### Improved

- Modular quantum computation architecture
- Documentation
- Package organization
- Error handling and reproducibility

### Research note

The quantum components are simulated on classical hardware.
This release does not claim quantum computational advantage.









# Changelog

## 0.6.6

- Removed the broken legacy `classical_physics.py` module from the distribution.
- Added PennyLane Grover search with validated integer targets and automatic iteration count.
- Added a Legendre-polynomial quantum encoder/decoder using `RY` feature rotations.
- Added finite-shot measurement and measurement-induced projection documentation.
- Fixed optional `quantum` dependency metadata to declare Qiskit and PennyLane directly.
- Added clean import and quantum tests that skip gracefully when PennyLane is absent.
