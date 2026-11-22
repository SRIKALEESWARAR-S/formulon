# SPEC-ML: machine-learning helpers
- **REQ-ML-001** `vectorize(f)` on arrays equals the scalar formula element-wise and supports broadcasting.
- **REQ-ML-002** Out-of-domain input raises `ValueError` (default) or yields NaN plus a `ValidationReport` (`on_invalid="mask"`); non-formulas raise `TypeError`.
- **REQ-ML-003** `FormulaFeatures` appends formula-derived columns, masks invalid rows with NaN and reports the invalid fraction.
