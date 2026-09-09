"""
Mathematical_physics.py
=======================
Advanced mathematical physics computation module providing optimized routines 
for fourier analysis, tensor algebra, vector calculus theorems, differential 
equations, special functions, and boundary value problems.

All calculations are accelerated via Numba's Just-In-Time (JIT) compilation 
and wrapped with robust engineering safety hooks.

Project: Project Formulon-Physics
License: Apache-2.0
"""

import re
from functools import wraps
import numba
from numba import jit
import numpy as np
import scipy as si

# Expose the clean public API for the PyPI package distribution
__all__ = [
    "validate_values",
    "validate_string",
    "validate_zeros_negatives",
    "safe_int_float",
    "fourier_series",
    "complex_fourier_series",
    "fourier_transform",
    "inverse_fourier_transform",
    "parsevals_identity",
    "convolution",
    "legendre_polynomial",
    "laguerre_polynomial",
    "hermite_polynomial",
    "bessel_function",
    "spherical_bessel",
    "heat_equation",
    "wave_equation",
    "solve_ode",
    "linear_algebra",
    "gauss_divergence",
    "stokes",
    "greens",
    "tensor",
]

MAX_NUMBER = 100000000
MAX_STR_LENGTH = 1280


# ══════════════════════════════════════════════════════════════════════════════
# 0. DECORATORS & CORE INTERNALS
# ═══════════════════════════════════════════════════════════════════════════

def safe_compute_jit(use_jit: bool = True, nopython: bool = False):
    """
    Combined decorator for global engineering error handling and numba JIT compilation.

    Project: Project Formulon-Physics

    Parameters
    ----------
    use_jit : bool, default=True
        Whether to apply JIT compilation to the targeted calculation loop.
    nopython : bool, default=False
        Whether to strictly force nopython mode for JIT execution paths.
    """
    def decorator(func):
        if use_jit:
            try:
                jitted_func = jit(func, nopython=nopython, cache=True)
            except Exception:
                # Fallback: use JIT with nopython=False for robust compatibility
                jitted_func = jit(func, nopython=False, cache=True)
        else:
            jitted_func = func
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return jitted_func(*args, **kwargs)
            except ValueError as e:
                raise ValueError(f"Validation error in {func.__name__}: {str(e)}")
            except (ZeroDivisionError, FloatingPointError) as e:
                raise ArithmeticError(f"Arithmetic error in {func.__name__}: {str(e)}")
            except Exception as e:
                raise RuntimeError(f"Unknown error in {func.__name__}: {type(e).__name__}: {str(e)}")
        return wrapper
    return decorator


def safe_compute(func):
    """Legacy architectural decorator - maps directly to the unified safe_compute_jit pipeline."""
    return safe_compute_jit(use_jit=True, nopython=False)(func)


# ══════════════════════════════════════════════════════════════════════════════
# 1. VALIDATION MODULES
# ══════════════════════════════════════════════════════════════════════════════

@safe_compute
def validate_values(values) -> bool:
    """
    Validate numerical inputs to ensure boundary compatibility with physics compute limits.

    Project: Project Formulon-Physics
    """
    if isinstance(values, (int, float)):
        return -MAX_NUMBER <= values <= MAX_NUMBER
    elif isinstance(values, (list, np.ndarray)):
        return all(validate_values(v) for v in values)
    else:
        return False


@safe_compute
def validate_string(s: str) -> str:
    """
    Sanitize input expressions or structural strings against processing injection boundaries.

    Project: Project Formulon-Physics
    """
    text = s.strip()
    if not text or len(text) > MAX_STR_LENGTH:
        raise ValueError("Invalid string: out of length limits.")
    if any(ord(ch) < 32 for ch in text):
        raise ValueError("Invalid string: contains unprintable control codes.")
    if any(ch in text for ch in '{}[]()<>$`\\'):
        raise ValueError("Invalid string: contains forbidden mathematical meta-characters.")
    return text


@safe_compute
def validate_zeros_negatives(values) -> bool:
    """
    Verify input parameters are strictly positive to preserve matrix and division stability.

    Project: Project Formulon-Physics
    """
    if isinstance(values, (int, float)):
        return values > 0
    elif isinstance(values, (list, np.ndarray)):
        return all(validate_zeros_negatives(v) for v in values)
    else:
        return False


@safe_compute
def safe_int_float(value):
    """
    Safely convert type configurations into standardized physics numerical types.

    Project: Project Formulon-Physics
    """
    if isinstance(value, (int, float)):
        return value
    elif isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                raise ValueError("Invalid input: not a recognizable real number configuration.")
    else:
        raise ValueError("Invalid input type: expected int, float, or string representation.")


# ══════════════════════════════════════════════════════════════════════════════
# 2. FOURIER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

@safe_compute
def fourier_series(x: float) -> float:
    """
    Compute the real Fourier series evaluation approximation of a target function at point x.

    Project: Project Formulon-Physics
    """
    n_terms = 100
    a0 = 1 / (2 * np.pi) * si.integrate.quad(lambda t: t, 0, 2 * np.pi)[0]
    an = lambda n: 1 / np.pi * si.integrate.quad(lambda t: t * np.cos(n * t), 0, 2 * np.pi)[0]
    bn = lambda n: 1 / np.pi * si.integrate.quad(lambda t: t * np.sin(n * t), 0, 2 * np.pi)[0]
    
    fourier_sum = a0
    for n in range(1, n_terms + 1):
        fourier_sum += an(n) * np.cos(n * x) + bn(n) * np.sin(n * x)
    
    return fourier_sum


@safe_compute
def complex_fourier_series(x: float) -> complex:
    """
    Evaluate the complex exponential form of the Fourier series representation at point x.

    Project: Project Formulon-Physics
    """
    n_terms = 100
    cn = lambda n: 1 / (2 * np.pi) * si.integrate.quad(lambda t: t * np.exp(-1j * n * t), 0, 2 * np.pi)[0]
    
    fourier_sum = 0j
    for n in range(-n_terms, n_terms + 1):
        fourier_sum += cn(n) * np.exp(1j * n * x)
    
    return fourier_sum


@safe_compute
def fourier_transform(f: callable, x: float) -> complex:
    """
    Compute the integral Fourier transform of the continuous function f evaluated at point x.

    Project: Project Formulon-Physics
    """
    real = si.integrate.quad(lambda t: np.real(f(t) * np.exp(-1j * x * t)), -np.inf, np.inf)[0]
    imag = si.integrate.quad(lambda t: np.imag(f(t) * np.exp(-1j * x * t)), -np.inf, np.inf)[0]
    return complex(real, imag)


@safe_compute
def inverse_fourier_transform(F: callable, t: float) -> complex:
    """
    Map a frequency spectrum function back to the coordinate spacetime domain at localized point t.

    Project: Project Formulon-Physics
    """
    real = si.integrate.quad(lambda x: np.real(F(x) * np.exp(1j * x * t)), -np.inf, np.inf)[0]
    imag = si.integrate.quad(lambda x: np.imag(F(x) * np.exp(1j * x * t)), -np.inf, np.inf)[0]
    return complex(real, imag) / (2 * np.pi)


@safe_compute
def parsevals_identity(f: callable, x: float) -> float:
    """
    Compute energy conservation parameters via Parseval's identity integration spaces.

    Project: Project Formulon-Physics
    """
    return si.integrate.quad(lambda t: abs(f(t))**2, -np.inf, np.inf)[0]


@safe_compute
def convolution(f: callable, g: callable, x: float) -> float:
    """
    Compute the mathematical spatial overlap convolution of two field profiles at point x.

    Project: Project Formulon-Physics
    """
    return si.integrate.quad(lambda t: f(t) * g(x - t), -np.inf, np.inf)[0]


# ══════════════════════════════════════════════════════════════════════════════
# 3. SPECIAL FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

@safe_compute_jit(use_jit=True, nopython=True)
def legendre_polynomial(n: int, x: float) -> float:
    """
    Evaluate the Legendre Polynomial $P_n(x)$ mapping angular momentum solutions.

    Project: Project Formulon-Physics
    """
    if n == 0:
        return 1.0
    elif n == 1:
        return x
    else:
        Pn_minus_2 = 1.0
        Pn_minus_1 = x
        for k in range(2, n + 1):
            Pn = ((2 * k - 1) * x * Pn_minus_1 - (k - 1) * Pn_minus_2) / k
            Pn_minus_2, Pn_minus_1 = Pn_minus_1, Pn
        return Pn


@safe_compute_jit(use_jit=True, nopython=True)
def laguerre_polynomial(n: int, x: float) -> float:
    """
    Evaluate the Laguerre Polynomial $L_n(x)$ tracking radial quantum wavefunctions.

    Project: Project Formulon-Physics
    """
    if n == 0:
        return 1.0
    elif n == 1:
        return 1.0 - x
    else:
        Ln_minus_2 = 1.0
        Ln_minus_1 = 1.0 - x
        for k in range(2, n + 1):
            Ln = ((2 * k - 1 - x) * Ln_minus_1 - (k - 1) * Ln_minus_2) / k
            Ln_minus_2, Ln_minus_1 = Ln_minus_1, Ln
        return Ln


@safe_compute_jit(use_jit=True, nopython=True)
def hermite_polynomial(n: int, x: float) -> float:
    """
    Evaluate the Hermite Polynomial $H_n(x)$ resolving quantum harmonic oscillators.

    Project: Project Formulon-Physics
    """
    if n == 0:
        return 1.0
    elif n == 1:
        return 2.0 * x
    else:
        Hn_minus_2 = 1.0
        Hn_minus_1 = 2.0 * x
        for k in range(2, n + 1):
            Hn = 2 * x * Hn_minus_1 - 2 * (k - 1) * Hn_minus_2
            Hn_minus_2, Hn_minus_1 = Hn_minus_1, Hn
        return Hn


def bessel_function(n: int, x: float) -> float:
    """
    Evaluate the Cylindrical Bessel function of the first kind J_n(x).

    Note: intentionally NOT numba-JIT'd. JIT'ing a thin SciPy wrapper only
    compiles to object-mode with zero speed benefit and adds call overhead.
    Also uses scipy.special.jv (jn is a deprecated alias, removed in newer SciPy).

    Project: Project Formulon-Physics
    """
    return si.special.jv(n, x)


def spherical_bessel(n: int, x: float) -> float:
    """
    Evaluate the Spherical Bessel function of the first kind j_n(x).

    Note: intentionally NOT numba-JIT'd for the same reason as bessel_function
    above -- it's a direct SciPy delegation, so JIT adds no value.

    Project: Project Formulon-Physics
    """
    return si.special.spherical_jn(n, x)


# ══════════════════════════════════════════════════════════════════════════════
# 4. PARTIAL & ORDINARY DIFFERENTIAL EQUATIONS
# ══════════════════════════════════════════════════════════════════════════════

@safe_compute
def heat_equation(initial_condition: callable, x: float, t: float) -> float:
    """
    Resolve temporal diffusion distributions by computing solutions to the classic Heat Equation.

    Project: Project Formulon-Physics
    """
    n_terms = 100
    L = 1
    solution = 0
    for n in range(1, n_terms + 1):
        An = 2 / L * si.integrate.quad(lambda s: initial_condition(s) * np.sin(n * np.pi * s / L), 0, L)[0]
        solution += An * np.exp(-n**2 * np.pi**2 * t / L**2) * np.sin(n * np.pi * x / L)
    return solution


@safe_compute
def wave_equation(initial_displacement: callable, initial_velocity: callable, x: float, t: float) -> float:
    """
    Resolve localized scalar wave field propagation solutions under the second-order Wave Equation.

    Project: Project Formulon-Physics
    """
    n_terms = 100
    L = 1
    solution = 0
    for n in range(1, n_terms + 1):
        An = 2 / L * si.integrate.quad(lambda s: initial_displacement(s) * np.sin(n * np.pi * s / L), 0, L)[0]
        Bn = 2 / (n * np.pi) * si.integrate.quad(lambda s: initial_velocity(s) * np.sin(n * np.pi * s / L), 0, L)[0]
        solution += (An * np.cos(n * np.pi * t / L) + Bn * np.sin(n * np.pi * t / L)) * np.sin(n * np.pi * x / L)
    return solution


@safe_compute
def solve_ode(equation: callable, initial_conditions, x: float):
    """
    Solve systems of ordinary differential equations tracking multi-state initial value conditions via RK45.

    Project: Project Formulon-Physics
    """
    from scipy.integrate import solve_ivp
    
    def system(t, y):
        return equation(t, y)
    
    t_span = (0, x)
    sol = solve_ivp(system, t_span, initial_conditions, method='RK45')
    return sol.y[:, -1]


# ══════════════════════════════════════════════════════════════════════════════
# 5. MATRICES, TENSORS, & VECTOR CALCULUS
# ══════════════════════════════════════════════════════════════════════════════

@safe_compute
def linear_algebra(matrix) -> dict:
    """
    Execute eigenvalue systems decompose mapping and singular value structural decomposition profiles.

    Project: Project Formulon-Physics
    """
    results = {}
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    results['eigenvalues'] = eigenvalues
    results['eigenvectors'] = eigenvectors
    
    U, S, Vh = np.linalg.svd(matrix)
    results['U'] = U
    results['S'] = S
    results['Vh'] = Vh
    
    if np.linalg.det(matrix) != 0:
        results['inverse'] = np.linalg.inv(matrix)
    
    return results


@safe_compute
def gauss_divergence(vector_field: callable, surface: callable, limits: list) -> float:
    """
    Compute vector flux indicators through an enclosed boundary space using Gauss's Divergence Theorem.

    Project: Project Formulon-Physics
    """
    from scipy.integrate import nquad
    
    def integrand(x, y, z):
        return np.dot(vector_field(x, y, z), surface(x, y, z))
    
    return nquad(integrand, limits)[0]


@safe_compute
def stokes(vector_field: callable, curve: callable, surface: callable, limits: list) -> float:
    """
    Compute closed pathway vector loop circulation bounds utilizing Stokes' Theorem tracking.

    Project: Project Formulon-Physics
    """
    from scipy.integrate import nquad
    
    def integrand(x, y, z):
        return np.dot(vector_field(x, y, z), np.cross(surface(x, y, z), [0, 0, 1]))
    
    return nquad(integrand, limits)[0]


@safe_compute
def greens(vector_field: callable, curve: callable, limits: list) -> float:
    """
    Evaluate planar vector boundaries flow conditions using Green's Theorem.

    Project: Project Formulon-Physics
    """
    from scipy.integrate import nquad
    
    def integrand(x, y):
        return np.dot(vector_field(x, y), [0, 0, 1])
    
    return nquad(integrand, limits)[0]


@safe_compute
def tensor(tensor1, tensor2) -> dict:
    """
    Perform multidimensional array contractions, additions, and metric inner tensor operations.

    Project: Project Formulon-Physics
    """
    results = {
        'addition': np.add(tensor1, tensor2),
        'dot_product': np.tensordot(tensor1, tensor2, axes=([0], [0])),
        'contraction': np.tensordot(tensor1, tensor2, axes=([0, 1], [0, 1]))
    }
    return results