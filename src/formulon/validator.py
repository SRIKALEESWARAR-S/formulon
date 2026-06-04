"""
validators.py
=============
Universal physics validation module for Project Formulon-Physics.

Architecture
------------
  PhysicalBounds   — Declarative container mapping parameters to physical laws.
  PhysicsCatalog   — Comprehensive namespace containing real-world physical boundaries.
  @validate        — Decorator factory that applies boundaries and acts as a runtime safety net.
  validate_* — Independent post-computation and array-structure utilities.

Project: Project Formulon-Physics
License: MIT License ~ Open Source Project
"""

import functools
import inspect
import math
import sys
from typing import Any, Callable
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# 1. Physical Quantity Structural Model
# ══════════════════════════════════════════════════════════════════════════════

class PhysicalBounds:
    """
    A declarative boundary mapping physical properties directly to mathematical laws.

    Project: Project Formulon-Physics
    """

    def __init__(
        self, 
        min_val: float | None = None, 
        max_val: float | None = None, 
        unit: str = "", 
        reason: str = ""
    ) -> None:
        self.min_val = min_val
        self.max_val = max_val
        self.unit = unit
        self.reason = reason

    def check(self, value: Any, name: str) -> None:
        """Type-guards and bounds-tests the incoming numerical input against physical constants."""
        # Type Guard: Explicitly reject non-numbers and booleans (which inherit from int)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"'{name}' must be a real number (int or float). Got {type(value).__name__}.")

        # Lower Boundary Constraint Check
        if self.min_val is not None and value < self.min_val:
            unit_str = f" {self.unit}" if self.unit else ""
            raise ValueError(
                f"Physical boundary violation for '{name}': {value}{unit_str} is below permissible limits. "
                f"{self.reason} (Minimum: {self.min_val}{unit_str})"
            )

        # Upper Boundary Constraint Check
        if self.max_val is not None and value > self.max_val:
            unit_str = f" {self.unit}" if self.unit else ""
            raise ValueError(
                f"Physical boundary violation for '{name}': {value}{unit_str} exceeds cosmic limits. "
                f"{self.reason} (Maximum: {self.max_val}{unit_str})"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 2. Complete Physics Rule Catalog
# ══════════════════════════════════════════════════════════════════════════════

class PhysicsCatalog:
    """
    A structural inventory of boundaries across every major physics sub-domain.

    Project: Project Formulon-Physics
    """
    
    # Universal Constants
    C_LIGHT = 2.998e8       # Speed of light in a vacuum (m/s)
    G_SUN_MAX = 274.0       # Surface gravity of the Sun (m/s²)
    T_MIN = 1e-6            # Numerical stability limit for time fractions (s)
    T_MAX = 1e100           # Simulation overflow threshold (s)

    # --- CORE MATHEMATICAL SCALARS ---
    REAL = PhysicalBounds()
    POSITIVE = PhysicalBounds(min_val=1e-30, reason="Value must be strictly positive.")
    NON_NEGATIVE = PhysicalBounds(min_val=0.0, reason="Value cannot fall below zero.")
    NEGATIVE = PhysicalBounds(max_val=-1e-30, reason="Value must be strictly negative.")
    FRACTION = PhysicalBounds(min_val=0.0, max_val=1.0, reason="Ratio must be constrained between 0 and 1.")

    # --- TIME SYSTEMS ---
    NON_NEG_TIME = PhysicalBounds(min_val=0.0, unit="s", reason="Time arrow must move forward.")
    POSITIVE_TIME = PhysicalBounds(min_val=T_MIN, max_val=T_MAX, unit="s", reason="Interval triggers numerical edge risks.")
    PERIOD = PhysicalBounds(min_val=T_MIN, unit="s", reason="Rotational/wave periods must be positive.")

    # --- MECHANICS (KINEMATICS & DYNAMICS) ---
    MASS = PhysicalBounds(min_val=1e-31, unit="kg", reason="Mass must be strictly positive in classical mechanics.")
    VELOCITY = PhysicalBounds(min_val=-C_LIGHT, max_val=C_LIGHT, unit="m/s", reason="Velocity magnitude cannot violate relativity.")
    SPEED = PhysicalBounds(min_val=0.0, max_val=C_LIGHT, unit="m/s", reason="Scalar speed is restricted by the speed of light.")
    ACCELERATION = PhysicalBounds()
    FORCE = PhysicalBounds()
    DISPLACEMENT = PhysicalBounds()
    DISTANCE = PhysicalBounds(min_val=0.0, unit="m", reason="Spatial distance cannot be negative.")
    MOMENTUM = PhysicalBounds()
    IMPULSE = PhysicalBounds()
    COEFF_FRICTION = PhysicalBounds(min_val=0.0, max_val=1.0, reason="Friction coefficients exist within [0, 1].")
    HEIGHT = PhysicalBounds(min_val=0.0, unit="m", reason="Altitude/Height relative to coordinate base must be non-negative.")
    NORMAL_FORCE = PhysicalBounds(min_val=0.0, unit="N", reason="Surface contact reaction forces must be non-negative.")
    APPARENT_WEIGHT = PhysicalBounds(min_val=0.0, unit="N", reason="Perceived gravitational weight cannot drop below zero.")
    CM_COORD = PhysicalBounds()

    # --- GRAVITATIONAL FIELD SYSTEMS ---
    GRAVITY = PhysicalBounds(min_val=0.0, max_val=G_SUN_MAX, unit="m/s²", reason="Local acceleration bounds exceed stellar profiles.")

    # --- ANGLES & ROTATIONAL GEOMETRIES ---
    ANGLE_DEG = PhysicalBounds(min_val=0.0, max_val=360.0, unit="°", reason="Angle must reside in a standard continuous circle.")
    ANGLE_DEG_90 = PhysicalBounds(min_val=0.0, max_val=90.0, unit="°", reason="Angle bounded by the horizon and local vertical.")
    CROSS_ANGLE_DEG = PhysicalBounds(min_val=0.0, max_val=180.0, unit="°", reason="Vector separation angles resolve inside [0, 180].")
    ANGLE_RAD = PhysicalBounds()
    INCIDENCE_ANGLE = PhysicalBounds(min_val=0.0, max_val=90.0, unit="°", reason="Optical incidence boundaries must not track parallel to interface.")
    WORK_ANGLE_DEG = PhysicalBounds(min_val=0.0, max_val=180.0, unit="°", reason="Vector displacement dot products are bounded by standard directions.")

    # --- CIRCULAR MOTION & MECHANICS ---
    RADIUS = PhysicalBounds(min_val=1e-12, unit="m", reason="Geometric coordinate radius cannot form a zero-singularity.")
    ANGULAR_VEL = PhysicalBounds()
    ANGULAR_ACC = PhysicalBounds()
    MOMENT_OF_INERTIA = PhysicalBounds(min_val=1e-30, unit="kg·m²", reason="Rotational mass must be strictly positive.")
    TORQUE = PhysicalBounds()
    ANGULAR_MOMENTUM = PhysicalBounds()
    PARALLEL_AXIS_D = PhysicalBounds(min_val=0.0, unit="m", reason="Axis displacement distance cannot be a negative length.")
    SPRING_CONST = PhysicalBounds(min_val=1e-5, unit="N/m", reason="Hooke elastic constants must exert restoring mechanics.")
    SPRING_DISPLACEMENT = PhysicalBounds()

    # --- ENERGY, WORK & POWER SYSTEMS ---
    ENERGY = PhysicalBounds()
    KINETIC_ENERGY = PhysicalBounds(min_val=0.0, unit="J", reason="Kinetic work scalars are inherently non-negative.")
    POTENTIAL_ENERGY = PhysicalBounds()
    WORK = PhysicalBounds()
    WORK_IO = PhysicalBounds(min_val=1e-12, unit="J", reason="Thermodynamic or electrical work tracking cannot be zero.")
    POWER = PhysicalBounds()
    EFFICIENCY = PhysicalBounds(min_val=1e-12, max_val=100.0, unit="%", reason="Standard efficiency values must reside in (0, 100].")

    # --- WAVES & OPTICS ---
    FREQUENCY = PhysicalBounds(min_val=1e-3, unit="Hz", reason="Spectral oscillations require positive frequency metrics.")
    WAVELENGTH = PhysicalBounds(min_val=1e-18, unit="m", reason="Physical waves must possess valid spatial extensions.")
    WAVE_SPEED = PhysicalBounds(min_val=1e-3, unit="m/s", reason="Wave velocity vectors require positive magnitudes.")
    REFRACTIVE_INDEX = PhysicalBounds(min_val=1.0, reason="Media velocities cannot exceed vacuum speeds (n >= 1).")
    AMPLITUDE = PhysicalBounds(min_val=0.0, reason="Wave envelope maximum bounds must remain non-negative.")
    INTENSITY = PhysicalBounds(min_val=0.0, unit="W/m²", reason="Energy flux surface distributions must be non-negative.")
    WAVE_NUMBER = PhysicalBounds(min_val=1e-5, unit="rad/m", reason="Spatial frequencies must be strictly positive.")

    # --- THERMODYNAMICS ---
    TEMPERATURE_K = PhysicalBounds(min_val=0.0, unit="K", reason="Thermal kinetic energy cannot breach Absolute Zero.")
    TEMPERATURE_C = PhysicalBounds(min_val=-273.15, unit="°C", reason="Thermal kinetic energy cannot breach Absolute Zero.")
    TEMPERATURE_F = PhysicalBounds(min_val=-459.67, unit="°F", reason="Thermal kinetic energy cannot breach Absolute Zero.")
    DELTA_TEMP = PhysicalBounds()
    PRESSURE = PhysicalBounds(min_val=1e-10, unit="Pa", reason="Kinetic collisions dictate strictly positive fluid pressures.")
    VOLUME = PhysicalBounds(min_val=1e-30, unit="m³", reason="Physical space parameters must be positive.")
    MOLES = PhysicalBounds(min_val=1e-24, unit="mol", reason="Matter parameters must possess real positive entities.")
    HEAT = PhysicalBounds()
    SPECIFIC_HEAT = PhysicalBounds(min_val=1e-3, unit="J/(kg·K)", reason="Thermal absorption behaviors must be positive.")
    LATENT_HEAT = PhysicalBounds(min_val=1e-3, unit="J/kg", reason="Enthalpy changes during structural transition require positive energy scales.")
    ENTROPY = PhysicalBounds()
    THERMAL_EFFICIENCY = PhysicalBounds(min_val=1e-12, max_val=1.0, reason="Thermodynamic cycle benchmarks span across (0, 1].")

    # --- ELECTROSTATICS & ELECTROMAGNETISM ---
    CHARGE = PhysicalBounds()
    ELECTRIC_FIELD = PhysicalBounds(min_val=0.0, unit="N/C", reason="Field strength tensor magnitudes must be non-negative.")
    ELECTRIC_POTENTIAL = PhysicalBounds()
    PERMITTIVITY = PhysicalBounds(min_val=1e-12, unit="F/m", reason="Dielectric responses remain strictly positive.")
    MAGNETIC_FIELD = PhysicalBounds(min_val=0.0, unit="T", reason="Induction field vector lengths are non-negative properties.")
    MAGNETIC_FLUX = PhysicalBounds()
    PERMEABILITY = PhysicalBounds(min_val=1e-7, unit="H/m", reason="Magnetic medium interactions remain strictly positive.")

    # --- CIRCUITS ---
    RESISTANCE = PhysicalBounds(min_val=0.0, unit="Ω", reason="Superconducting mechanics protect non-negative resistances.")
    RESISTIVITY = PhysicalBounds(min_val=1e-15, unit="Ω·m", reason="Intrinsic medium electrical damping requires positive constraints.")
    CURRENT = PhysicalBounds()
    VOLTAGE = PhysicalBounds()
    CAPACITANCE = PhysicalBounds(min_val=1e-20, unit="F", reason="Dielectric layout storage metrics must remain strictly positive.")
    INDUCTANCE = PhysicalBounds(min_val=1e-20, unit="H", reason="Flux-current configuration balances must be strictly positive.")
    EMF = PhysicalBounds()
    IMPEDANCE = PhysicalBounds(min_val=0.0, unit="Ω", reason="Complex load total magnitudes must equal or cross zero.")
    POWER_FACTOR = PhysicalBounds(min_val=0.0, max_val=1.0, reason="Phase angle cosine values remain inside [0, 1].")
    TURNS = PhysicalBounds(min_val=1.0, reason="Inductor loops must feature positive integers.")

    # --- MODERN PHYSICS ---
    BETA = PhysicalBounds(min_val=0.0, max_val=1.0, reason="Relativistic speed fractions must span inside (0, 1).")
    LORENTZ_FACTOR = PhysicalBounds(min_val=1.0, reason="Dilation scale multipliers evaluate to 1 or higher.")
    PHOTON_ENERGY = PhysicalBounds(min_val=1e-34, unit="J", reason="Quantum packets must register positive energetic footprints.")
    DE_BROGLIE_WL = PhysicalBounds(min_val=1e-20, unit="m", reason="Matter mechanics demand positive wavelength metrics.")
    PLANCK_H = PhysicalBounds(min_val=1e-35, unit="J·s", reason="Fundamental actions require real, finite limits.")
    REST_MASS = PhysicalBounds(min_val=1e-31, unit="kg", reason="Inertial rest profiles require positive masses.")

    # --- FLUID MECHANICS ---
    DENSITY = PhysicalBounds(min_val=1e-15, unit="kg/m³", reason="Matter distributions necessitate positive density layers.")
    VISCOSITY = PhysicalBounds(min_val=1e-7, unit="Pa·s", reason="Internal shear losses demand valid friction levels.")
    FLOW_RATE = PhysicalBounds(min_val=1e-20, unit="m³/s", reason="Volumetric dynamic movements remain strictly positive metrics.")
    AREA = PhysicalBounds(min_val=1e-20, unit="m²", reason="Planar boundaries must span positive dimensions.")
    REYNOLDS = PhysicalBounds(min_val=1e-5, reason="Dimensionless inertia/viscous balances require positive tracking.")
    BULK_MODULUS = PhysicalBounds(min_val=1e-2, unit="Pa", reason="Incompressibility elastic properties remain positive.")


# Global system singleton reference
RULES = PhysicsCatalog()


# ══════════════════════════════════════════════════════════════════════════════
# 3. Comprehensive Intercept Validation Decorator
# ══════════════════════════════════════════════════════════════════════════════

def validate(**param_rules: PhysicalBounds) -> Callable:
    """
    Decorator factory: maps structural rules to parameter inputs, and
    intercepts execution-level mathematical faults gracefully.

    Project: Project Formulon-Physics
    """
    # Validation constraint checkpoint at load-time
    for key, rule in param_rules.items():
        if not isinstance(rule, PhysicalBounds):
            raise TypeError(f"@validate config error: Key '{key}' needs a valid PhysicalBounds instance.")

    def decorator(func: Callable) -> Callable:
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())

        # Signature mismatch alignment validation
        for key in param_rules:
            if key not in param_names:
                raise NameError(f"@validate tracking error on '{func.__name__}': Parameter '{key}' does not exist.")

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Match incoming execution arguments to signature keys
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Execute explicit input validations
            for param, rule in param_rules.items():
                if param in bound.arguments:
                    rule.check(bound.arguments[param], param)

            # Execution Protection Phase
            try:
                result = func(*args, **kwargs)
                
                # Intercept invalid numerical calculations before passing downstream
                if isinstance(result, (int, float)) and not math.isfinite(result):
                    raise ValueError("Calculation generated an Infinite or NaN value.")
                return result

            except ZeroDivisionError as exc:
                raise RuntimeError(
                    f"Physical Singularity in '{func.__name__}': Calculation caused a division by zero. "
                    f"Ensure denominator components (time intervals, distance steps, mass components) are not zero."
                ) from exc

            except OverflowError as exc:
                raise RuntimeError(
                    f"Numerical Overflow in '{func.__name__}': System values reached scales python cannot resolve. "
                    f"Check scientific exponential terms or units."
                ) from exc

            except ValueError as exc:
                raise ValueError(
                    f"Domain Matrix Error in '{func.__name__}': {exc}. "
                    f"Verify math consistency rules (e.g., negative parameters passed into square root transformations)."
                ) from exc

            except KeyboardInterrupt:
                print(f"\n[Terminated] Computation loop inside '{func.__name__}' intercepted by hardware interrupt request.")
                sys.exit(130)

            except Exception as exc:
                raise RuntimeError(
                    f"Unhandled calculation exception trapped inside execution layer of '{func.__name__}': {exc}"
                ) from exc

        return wrapper
    return decorator


# ══════════════════════════════════════════════════════════════════════════════
# 4. Standalone Multi-Parameter and Vector Trackers
# ══════════════════════════════════════════════════════════════════════════════

def validate_array_lengths(a: Any, b: Any, name_a: str, name_b: str) -> None:
    """
    Validates dimensional consistency across matching metric lengths.

    Project: Project Formulon-Physics
    """
    if len(a) != len(b):
        raise ValueError(f"Structural Mismatch: '{name_a}' (len={len(a)}) must match '{name_b}' (len={len(b)}).")


def validate_strictly_increasing(arr: Any, name: str = "arr") -> None:
    """
    Ensures sequential metrics (like time timelines) increase steadily.

    Project: Project Formulon-Physics
    """
    parsed = np.asarray(arr, dtype=float)
    if np.any(np.diff(parsed) <= 0):
        raise ValueError(f"Sequence configuration fault: Array tracking vector '{name}' must be strictly increasing.")


def validate_time_array(t: Any, name: str = "t") -> np.ndarray:
    """
    Analyzes, typeguards, and structural-maps a continuous time array.

    Project: Project Formulon-Physics
    """
    try:
        parsed = np.asarray(t, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Array '{name}' could not be compiled into real physical float structures. Trace: {exc}") from exc

    if parsed.ndim != 1:
        raise ValueError(f"Dimensional fault: '{name}' needs to map to a 1D timeline array (received shape {parsed.shape}).")
    if len(parsed) < 2:
        raise ValueError(f"Data density error: Array timeline '{name}' must provide at least 2 distinct metrics.")
    if np.any(parsed <= 0):
        raise ValueError(f"Temporal arrow violation: Tracked matrix segments inside '{name}' must exceed absolute zero (t > 0).")
    if parsed[-1] == parsed[0]:
        raise ValueError(f"Static timeline fault: End coordinates match starting indices inside vector reference '{name}'.")
    if np.any(parsed < PhysicsCatalog.T_MIN) or np.any(parsed > PhysicsCatalog.T_MAX):
        raise ValueError(f"Numeric simulation limits exceeded: Ensure array data values stay within standard resolution limits.")
    return parsed


def validate_positive_array(arr: Any, name: str) -> np.ndarray:
    """
    Ensures scalar entries across tracked arrays remain explicitly greater than zero.

    Project: Project Formulon-Physics
    """
    try:
        parsed = np.asarray(arr, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Matrix parsing error for '{name}': Couldn't map targets to a float schema. Detail: {exc}") from exc

    if np.any(parsed <= 0):
        invalid_items = parsed[parsed <= 0]
        raise ValueError(f"Physical constraint error inside matrix '{name}': Elements must match positive scales. Infractions: {invalid_items}")
    return parsed


def validate_non_negative_array(arr: Any, name: str) -> np.ndarray:
    """
    Verifies metric variables inside collection tracks never fall below zero.

    Project: Project Formulon-Physics
    """
    try:
        parsed = np.asarray(arr, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Matrix parsing error for '{name}': Couldn't convert indices into standard floats. Detail: {exc}") from exc

    if np.any(parsed < 0):
        invalid_items = parsed[parsed < 0]
        raise ValueError(f"Boundary fault on element array '{name}': Coordinates cannot be negative indices. Infractions: {invalid_items}")
    return parsed


def validate_same_shape(*arrays_names: tuple[Any, str]) -> None:
    """
    Verifies multi-dimensional geometry states match across arrays.

    Project: Project Formulon-Physics
    """
    converted = [(np.asarray(a), n) for a, n in arrays_names]
    base_shape = converted[0][0].shape
    base_name = converted[0][1]
    
    for current_arr, current_name in converted[1:]:
        if current_arr.shape != base_shape:
            raise ValueError(f"Geometry coordinate tracking fault: Parameter '{base_name}' features shape {base_shape}, but variable '{current_name}' yields shape {current_arr.shape}.")


def validate_finite(value: float, name: str) -> None:
    """
    Final computation checkpoint ensuring scalar indicators are valid numbers.

    Project: Project Formulon-Physics
    """
    if not math.isfinite(value):
        raise ValueError(f"Computed computational error for '{name}': Generated metrics are not finite values ({value}). Check input dimensions.")


def validate_cross_param_le(a: float, b: float, name_a: str, name_b: str) -> None:
    """
    Calculates relational constraints across interdependent values (e.g., T_cold <= T_hot).

    Project: Project Formulon-Physics
    """
    if a > b:
        raise ValueError(f"Interdependent rule breach: Component parameter '{name_a}' ({a}) can't exceed comparative baseline '{name_b}' ({b}).")