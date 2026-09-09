"""
numerical_series_modules.py
===========================
Mathematical Series Computation Module optimized for rapid sequential evaluations,
summation sequences, and discrete progressions.

All algorithmic engines are accelerated via Numba's Just-In-Time (JIT) compilation
and protected against resource exhaustion limits.

Project: Project Formulon-Physics
License: Apache-2.0
"""

import re
import numba
from numba import njit

__all__ = [
    "arithmetic_nterm",
    "arithmetic_sum",
    "geometric_nterm",
    "infinite_geometric_sum",
    "natural_numbers_sum",
    "squares_sum",
    "cubes_sum",
    "odd_numbers_sum",
    "fibonacci_nterm",
]

# Structural safety boundaries to shield runtime environments from memory/overflow traps
MAX_N = 10_000_000  
MAX_STR_LEN = 128   


def _sanitize_string(value) -> str:
    """
    Sanitize string arguments to safely convert textual inputs into numerical payloads.

    Project: Project Formulon-Physics
    """
    if not isinstance(value, str):
        raise TypeError("Expected input value representation in string format.")
    
    clean_val = value.strip()
    if len(clean_val) > MAX_STR_LEN:
        raise ValueError("Input text bounds exceed maximum structural string allocation thresholds.")
        
    if any(ord(char) < 32 for char in clean_val):
        raise ValueError("String payload contains illegal hidden escape or control sequences.")
        
    if re.search(r"[{}<>\[\]$`;\\|&]", clean_val):
        raise ValueError("Forbidden syntax injection patterns encountered within parameter data.")
        
    return clean_val


def _safe_int(value) -> int:
    """
    Safely process and cast parameter types down to native mathematical integer configurations.

    Project: Project Formulon-Physics
    """
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not value.is_integer():
            raise ValueError("Configuration expects a discrete integer boundary value.")
        return int(value)
        
    sanitized = _sanitize_string(value)
    if not re.match(r"^[+-]?\d+$", sanitized):
        raise ValueError("Text representation cannot be cast safely into a clear signed integer layout.")
        
    return int(sanitized)


def _safe_float(value) -> float:
    """
    Safely process and cast parameter elements into real-number float format representations.

    Project: Project Formulon-Physics
    """
    if isinstance(value, (int, float)):
        return float(value)
        
    sanitized = _sanitize_string(value)
    if not re.match(r"^[+-]?\d*(\.\d+)?([eE][+-]?\d+)?$", sanitized):
        raise ValueError("Text translation contains non-numeric components that corrupt floating point conversion.")
        
    return float(sanitized)


# ══════════════════════════════════════════════════════════════════════════════
# JIT COMPILATION ENGINES
# ══════════════════════════════════════════════════════════════════════════════

@njit
def _arithmetic_nterm(a1: float, d: float, n: int) -> float:
    return a1 + (n - 1) * d


@njit
def _arithmetic_sum(a1: float, d: float, n: int) -> float:
    return (n / 2.0) * (2.0 * a1 + (n - 1) * d)


@njit
def _geometric_nterm(a1: float, r: float, n: int) -> float:
    return a1 * (r ** (n - 1))


@njit
def _infinite_geometric_sum(a1: float, r: float) -> float:
    if abs(r) >= 1.0:
        raise ValueError("Infinite geometric series fails to converge for ratio parameters where |r| >= 1.")
    return a1 / (1.0 - r)


@njit
def _natural_numbers_sum(n: int) -> int:
    return (n * (n + 1)) // 2


@njit
def _squares_sum(n: int) -> int:
    return (n * (n + 1) * (2 * n + 1)) // 6


@njit
def _cubes_sum(n: int) -> int:
    term = (n * (n + 1)) // 2
    return term * term


@njit
def _odd_numbers_sum(n: int) -> int:
    return n * n


@njit
def _fibonacci_nterm(n: int) -> int:
    if n == 1:
        return 0
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC INTERFACE API LAYER
# ══════════════════════════════════════════════════════════════════════════════

def arithmetic_nterm(a1, d, n):
    """
    Determine the value of the specific n-th term within an arithmetic progression progression matrix.

    Project: Project Formulon-Physics
    """
    try:
        n = _safe_int(n)
        if n <= 0 or n > MAX_N:
            return "Invalid input"
        return _arithmetic_nterm(_safe_float(a1), _safe_float(d), n)
    except KeyboardInterrupt:
        return "Execution interrupted by user"
    except (TypeError, ValueError):
        return "Invalid input"
    except Exception:
        return "I'm not dead, I'm on rest"


def arithmetic_sum(a1, d, n):
    """
    Calculate the mathematical accumulation sum up to the target n-th term inside an arithmetic progression.

    Project: Project Formulon-Physics
    """
    try:
        n = _safe_int(n)
        if n <= 0 or n > MAX_N:
            return "Invalid input"
        return _arithmetic_sum(_safe_float(a1), _safe_float(d), n)
    except KeyboardInterrupt:
        return "Execution interrupted by user"
    except (TypeError, ValueError):
        return "Invalid input"
    except Exception:
        return "I'm not dead, I'm on rest"


def geometric_nterm(a1, r, n):
    """
    Determine the value of the specific n-th term within an isolated geometric sequence context.

    Project: Project Formulon-Physics
    """
    try:
        n = _safe_int(n)
        if n <= 0 or n > MAX_N:
            return "Invalid input"
        return _geometric_nterm(_safe_float(a1), _safe_float(r), n)
    except KeyboardInterrupt:
        return "Execution interrupted by user"
    except (TypeError, ValueError):
        return "Invalid input"
    except Exception:
        return "I'm not dead, I'm on rest"


def infinite_geometric_sum(a1, r):
    """
    Evaluate convergent limit spaces to map the sum of an infinite geometric progression string.

    Project: Project Formulon-Physics
    """
    try:
        return _infinite_geometric_sum(_safe_float(a1), _safe_float(r))
    except KeyboardInterrupt:
        return "Execution interrupted by user"
    except (TypeError, ValueError):
        return "Invalid input"
    except Exception:
        return "I'm not dead, I'm on rest"


def natural_numbers_sum(n):
    """
    Sum all continuous integers starting from unity up to the targeted parameter marker count n.

    Project: Project Formulon-Physics
    """
    try:
        n = _safe_int(n)
        if n <= 0 or n > MAX_N:
            return "Invalid input"
        return _natural_numbers_sum(n)
    except KeyboardInterrupt:
        return "Execution interrupted by user"
    except (TypeError, ValueError):
        return "Invalid input"
    except Exception:
        return "I'm not dead, I'm on rest"


def squares_sum(n):
    """
    Evaluate cumulative sum structures across consecutive integer squares tracking up to index position n.

    Project: Project Formulon-Physics
    """
    try:
        n = _safe_int(n)
        if n <= 0 or n > MAX_N:
            return "Invalid input"
        return _squares_sum(n)
    except KeyboardInterrupt:
        return "Execution interrupted by user"
    except (TypeError, ValueError):
        return "Invalid input"
    except Exception:
        return "I'm not dead, I'm on rest"


def cubes_sum(n):
    """
    Evaluate cumulative sum structures across consecutive integer cubic parameters tracking up to index position n.

    Project: Project Formulon-Physics
    """
    try:
        n = _safe_int(n)
        if n <= 0 or n > MAX_N:
            return "Invalid input"
        return _cubes_sum(n)
    except KeyboardInterrupt:
        return "Execution interrupted by user"
    except (TypeError, ValueError):
        return "Invalid input"
    except Exception:
        return "I'm not dead, I'm on rest"


def odd_numbers_sum(n):
    """
    Accumulate sequential mathematical evaluations across the initial sequence set of n explicit odd number profiles.

    Project: Project Formulon-Physics
    """
    try:
        n = _safe_int(n)
        if n <= 0 or n > MAX_N:
            return "Invalid input"
        return _odd_numbers_sum(n)
    except KeyboardInterrupt:
        return "Execution interrupted by user"
    except (TypeError, ValueError):
        return "Invalid input"
    except Exception:
        return "I'm not dead, I'm on rest"


def fibonacci_nterm(n):
    """
    Solve for and isolate the structural n-th term positional value inside the standard Fibonacci recurrence chain.

    Project: Project Formulon-Physics
    """
    try:
        n = _safe_int(n)
        if n <= 0 or n > MAX_N:
            return "Invalid input"
        return _fibonacci_nterm(n)
    except KeyboardInterrupt:
        return "Execution interrupted by user"
    except (TypeError, ValueError):
        return "Invalid input"
    except Exception:
        return "I'm not dead, I'm on rest"