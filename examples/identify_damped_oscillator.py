"""Inverse problem: identify (c, k) of a suspension from noisy sensor data.

Uses only NumPy/SciPy -- no deep-learning framework needed.  The same
``PDEProblem`` object supplies the reference model, the noisy dataset and the
physics residual, so the identification is checked against ground truth.
"""
import numpy as np
from scipy.optimize import least_squares
from formulon.pinn import get_problem, make_dataset, parameter_error, relative_l2

true = get_problem("damped_oscillator", m=1.0, c=0.4, k=4.0)
data = make_dataset(true, n_collocation=64, n_observations=200, noise_std=0.03, seed=7)
t, y = data.observations["t"], data.observations["x"]


def misfit(theta):
    c, k = theta
    try:
        model = true.with_params(c=c, k=k)
    except ValueError:        # overdamped / non-physical guess
        return np.full_like(y, 1e3)
    return model.exact(t=t) - y


sol = least_squares(misfit, x0=[0.1, 2.0], bounds=([0.0, 0.5], [3.0, 10.0]))
est = {"c": sol.x[0], "k": sol.x[1]}
print("estimated:", {k: round(v, 3) for k, v in est.items()})
print("relative parameter error:", {k: round(v, 3) for k, v in parameter_error(est, true.params).items()})
print("fit rel-L2 vs clean signal:", round(relative_l2(true.with_params(**est).exact(t=t), data.observations["x_clean"]), 4))
