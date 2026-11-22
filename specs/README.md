# Specifications (spec-driven development)

Formulon Physics is developed **spec-first**, with Claude (Anthropic) as the implementation assistant:

1. A requirement is written here as a testable statement with an ID (`REQ-<AREA>-<NNN>`) and acceptance criteria.
2. Tests are written against the acceptance criteria and tagged with the ID (`# spec: REQ-...` or in the docstring).
3. Claude implements until the tests pass; anything that cannot be verified in the build environment is
   stated explicitly under "Verification" in the spec instead of being claimed.
4. `python scripts/spec_trace.py` fails CI if a requirement has no test or a test cites an unknown requirement.
   `docs/TRACEABILITY.md` is the generated requirement -> test matrix.

The maintainer (Sri Kaleeswarar S) owns goals and decisions (e.g. Apache-2.0 license, which frameworks to support);
the AI proposes, implements and tests; nothing ships without the test-suite and the trace check passing.

| File | Area |
|---|---|
| `SPEC-PKG.md` | packaging, license, imports |
| `SPEC-ML.md` | array-valued formulas, feature maps |
| `SPEC-PINN.md` | benchmark problems, conditions, sampling, metrics |
| `SPEC-BACKENDS.md` | the backend-switch mechanism, quantum / autodiff / JIT / unified config |
| `SPEC-QUANTUM.md` | circuits, gates, adders, Grover, Legendre encoder |
| `SPEC-AUTODIFF.md` | derivatives, losses, training on torch / jax / numpy |
