# SPEC-PINN: PINN benchmark problems
- **REQ-PINN-001** Each benchmark's closed-form solution satisfies its own residual (RMS < 1e-4, finite differences).
- **REQ-PINN-002** A wrong parameter (negative control) produces a clearly non-zero residual.
- **REQ-PINN-003** `conditions()` returns exact Dirichlet targets for boundary/initial points and the initial velocity for problems second-order in time.
- **REQ-PINN-004** Sampling is reproducible for a seed, inside the domain, boundary points lie on faces; datasets add seeded noise; pure ODEs have no boundary set.
- **REQ-PINN-005** Metrics (rel-L2, max, RMSE, parameter error, energy drift) are correct and reject shape mismatches.
- **REQ-PINN-006** Bad parameters (negative diffusivity, overdamped oscillator, unknown name) raise clear errors; `heat_slab_from_material` computes alpha = k/(rho*cp).
