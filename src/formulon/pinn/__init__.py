"""PINN toolkit: verified benchmark problems, residuals, sampling and metrics.

Problem it addresses
--------------------
PINN papers and in-house models are rarely comparable: every group re-codes
the PDE, the exact solution and the error metric, and silent mistakes in the
residual (wrong sign, missing factor) are common.  This subpackage provides
PDE problems that ship with **closed-form reference solutions**, residuals
written once in a backend-agnostic way (NumPy finite differences or PyTorch
autograd), standard error metrics, and sampling / noisy-data generators for
forward and inverse (parameter identification) studies.

Core is NumPy/SciPy only; PyTorch is optional (``formulon.pinn.torch_utils``).
"""
from .problems import (PDEProblem, get_problem, list_problems, heat_slab_from_material,
                       verify_problem)
from .derivatives import fd_derivatives
from .metrics import (relative_l2, max_abs_error, rmse, residual_report, parameter_error,
                      energy_drift)
from .autodiff import (set_autodiff, get_autodiff, use_autodiff, available_autodiff, resolve_autodiff,
                       derivatives, residuals, loss, fit, evaluate)
from .pielm import pielm_fit, PIELMModel
from .sampling import sample_interior, sample_boundary, make_dataset, Dataset

__all__ = [
    "PDEProblem", "get_problem", "list_problems", "heat_slab_from_material", "verify_problem",
    "fd_derivatives", "relative_l2", "max_abs_error", "rmse", "residual_report",
    "parameter_error", "energy_drift", "sample_interior", "sample_boundary",
    "make_dataset", "Dataset",
    "set_autodiff", "get_autodiff", "use_autodiff", "available_autodiff", "resolve_autodiff",
    "derivatives", "residuals", "loss", "fit", "evaluate", "pielm_fit", "PIELMModel",
]
