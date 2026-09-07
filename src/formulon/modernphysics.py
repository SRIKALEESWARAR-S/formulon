"""
modern_physics_quantum_relativity.py
====================================
Comprehensive engineering library covering Special Relativity, Quantum Mechanics,
Atomic Structures, and Nuclear Decay mechanics.

All formulations are validated prior to execution using physical boundary constraints
and protected against runtime anomalies via the external validators framework.

Project: Project Formulon-Physics
License: Apache-2.0
"""

import math
from .validator import validate, RULES, validate_cross_param_le

__all__ = [
    # 1. Special Relativity
    "lorentz_factor",
    "time_dilation",
    "length_contraction",
    "relativistic_momentum",
    "total_relativistic_energy",
    "rest_mass_energy",
    "energy_momentum_invariant",
    "relativistic_velocity_addition",
    # 2. Quantum Mechanics & Atomic Physics
    "photon_energy",
    "photoelectric_max_kinetic_energy",
    "de_broglie_wavelength",
    "heisenberg_uncertainty_minimum_momentum",
    "bohr_orbit_radius_angular_momentum",
    "bohr_hydrogen_energy_level",
    "compton_scattering_wavelength_shift",
    # 3. Nuclear Physics & Radioactive Decay
    "radioactive_decay_remaining_nuclei",
    "half_life_from_decay_constant",
    "binding_energy_from_mass_defect",
]

# Physical Invariant Constants
C_SPEED_LIGHT = 299792458
H_PLANCK = 6.62607015e-34
H_BAR_PLANCK = H_PLANCK / (2.0 * math.pi)
M_ELECTRON = 9.1093837e-31


# ══════════════════════════════════════════════════════════════════════════════
# 1. SPECIAL RELATIVITY
# ══════════════════════════════════════════════════════════════════════════════

@validate(v=RULES.SPEED)
def lorentz_factor(v: float) -> float:
    r"""
    Calculate the dimensionless Lorentz factor scaling property ($\gamma = 1 / \sqrt{1 - v^2/c^2}$).

    Project: Project Formulon-Physics
    """
    validate_cross_param_le(v, C_SPEED_LIGHT, "v", "C_SPEED_LIGHT")
    if v == C_SPEED_LIGHT:
        raise ZeroDivisionError("System frames cannot evaluate precisely at the speed of light asymptote.")
    return 1.0 / math.sqrt(1.0 - ((v / C_SPEED_LIGHT) ** 2))


@validate(t0_proper=RULES.NON_NEG_TIME, v=RULES.SPEED)
def time_dilation(t0_proper: float, v: float) -> float:
    r"""
    Calculate dilated timeline horizons observed from a moving frame ($\Delta t = \gamma \Delta t_0$).

    Project: Project Formulon-Physics
    """
    return t0_proper * lorentz_factor(v)


@validate(l0_proper=RULES.DISTANCE, v=RULES.SPEED)
def length_contraction(l0_proper: float, v: float) -> float:
    r"""
    Evaluate dimensional length contraction scales along the axis of relative motion ($L = L_0 / \gamma$).

    Project: Project Formulon-Physics
    """
    return l0_proper / lorentz_factor(v)


@validate(m0_rest=RULES.MASS, v=RULES.SPEED)
def relativistic_momentum(m0_rest: float, v: float) -> float:
    r"""
    Evaluate net relativistic momentum thresholds across high velocity states ($p = \gamma m_0 v$).

    Project: Project Formulon-Physics
    """
    return lorentz_factor(v) * m0_rest * v


@validate(m0_rest=RULES.MASS, v=RULES.SPEED)
def total_relativistic_energy(m0_rest: float, v: float) -> float:
    r"""
    Determine the comprehensive total relativistic energy profile ($E = \gamma m_0 c^2$).

    Project: Project Formulon-Physics
    """
    return lorentz_factor(v) * m0_rest * (C_SPEED_LIGHT ** 2)


@validate(m0_rest=RULES.MASS)
def rest_mass_energy(m0_rest: float) -> float:
    """
    Determine localized rest mass state equivalence footprints ($E_0 = m_0 c^2$).

    Project: Project Formulon-Physics
    """
    return m0_rest * (C_SPEED_LIGHT ** 2)


@validate(momentum_p=RULES.REAL, m0_rest=RULES.MASS)
def energy_momentum_invariant(momentum_p: float, m0_rest: float) -> float:
    r"""
    Resolve total energy values using the relativistic energy-momentum invariant framework ($E = \sqrt{(pc)^2 + (m_0c^2)^2}$).

    Project: Project Formulon-Physics
    """
    term_pc = momentum_p * C_SPEED_LIGHT
    term_mc2 = m0_rest * (C_SPEED_LIGHT ** 2)
    return math.sqrt((term_pc ** 2) + (term_mc2 ** 2))


@validate(u_velocity=RULES.REAL, v_frame=RULES.REAL)
def relativistic_velocity_addition(u_velocity: float, v_frame: float) -> float:
    """
    Calculate the combined frame transformations for relativistic velocities ($u' = (u - v) / (1 - uv/c^2)$).

    Project: Project Formulon-Physics
    """
    denominator = 1.0 - ((u_velocity * v_frame) / (C_SPEED_LIGHT ** 2))
    if denominator == 0:
        raise ZeroDivisionError("Velocity addition maps onto a singular frame calculation horizon anomaly.")
    return (u_velocity - v_frame) / denominator


# ══════════════════════════════════════════════════════════════════════════════
# 2. QUANTUM MECHANICS & ATOMIC PHYSICS
# ══════════════════════════════════════════════════════════════════════════════

@validate(wavelength=RULES.WAVELENGTH)
def photon_energy(wavelength: float) -> float:
    r"""
    Determine single photon operational energy capacities using spectral parameters ($E = hc / \lambda$).

    Project: Project Formulon-Physics
    """
    return (H_PLANCK * C_SPEED_LIGHT) / wavelength


@validate(wavelength=RULES.WAVELENGTH, work_function_ev=RULES.REAL)
def photoelectric_max_kinetic_energy(wavelength: float, work_function_ev: float) -> float:
    r"""
    Calculate peak electron emission kinetic energy fields tracking the photoelectric effect.

    $K_{max} = \frac{hc}{\lambda} - \Phi$

    Project: Project Formulon-Physics
    """
    work_function_joules = work_function_ev * 1.602176634e-19
    incident_energy = photon_energy(wavelength)
    
    # Note: If negative, it implies the photon energy lacks the capacity to cross the material threshold barrier
    return incident_energy - work_function_joules


@validate(momentum_p=RULES.REAL)
def de_broglie_wavelength(momentum_p: float) -> float:
    r"""
    Calculate corresponding de Broglie spatial wavelengths matching matter momentum states ($\lambda = h / p$).

    Project: Project Formulon-Physics
    """
    if momentum_p == 0:
        raise ZeroDivisionError("Stationary rest configurations yield undefined infinite wave scaling footprints.")
    return H_PLANCK / abs(momentum_p)


@validate(delta_x_position=RULES.DISTANCE)
def heisenberg_uncertainty_minimum_momentum(delta_x_position: float) -> float:
    r"""
    Isolate standard minimal momentum constraint profiles via Heisenberg's uncertainty relationship ($\Delta p \ge \hbar / (2 \Delta x)$).

    Project: Project Formulon-Physics
    """
    return H_BAR_PLANCK / (2.0 * delta_x_position)


@validate(principal_n=RULES.DEGREES_FREEDOM)
def bohr_orbit_radius_angular_momentum(principal_n: int) -> float:
    r"""
    Determine quantized angular momentum thresholds across circular orbital systems ($L = n \hbar$).

    Project: Project Formulon-Physics
    """
    if principal_n <= 0:
        raise ValueError("Principal quantum numbers must evaluate inside real positive integer matrices.")
    return float(principal_n) * H_BAR_PLANCK


@validate(principal_n=RULES.DEGREES_FREEDOM)
def bohr_hydrogen_energy_level(principal_n: int) -> float:
    """
    Predict structural energy level tracks in Hydrogen atoms under the Bohr approximation ($E_n = -13.6\text{ eV} / n^2$).

    Project: Project Formulon-Physics
    """
    if principal_n <= 0:
        raise ValueError("Principal quantum numbers must correspond to stable positive shell levels.")
    return -13.6 / (principal_n ** 2)


@validate(scattering_theta_deg=RULES.ANGLE_DEG)
def compton_scattering_wavelength_shift(scattering_theta_deg: float) -> float:
    r"""
    Calculate electromagnetic wavelength shift profiles under localized particle collisions.

    $\Delta\lambda = \frac{h}{m_e c}(1 - \cos\theta)$

    Project: Project Formulon-Physics
    """
    compton_constant = H_PLANCK / (M_ELECTRON * C_SPEED_LIGHT)
    return compton_constant * (1.0 - math.cos(math.radians(scattering_theta_deg)))


# ══════════════════════════════════════════════════════════════════════════════
# 3. NUCLEAR PHYSICS & RADIOACTIVE DECAY
# ══════════════════════════════════════════════════════════════════════════════

@validate(n0_initial=RULES.REAL, decay_constant_lambda=RULES.NON_NEGATIVE, t_elapsed=RULES.NON_NEG_TIME)
def radioactive_decay_remaining_nuclei(n0_initial: float, decay_constant_lambda: float, t_elapsed: float) -> float:
    r"""
    Calculate remaining structural nuclei records tracking continuous exponential decay sequences ($N(t) = N_0 e^{-\lambda t}$).

    Project: Project Formulon-Physics
    """
    if n0_initial < 0:
        raise ValueError("Initial sample structural footprints cannot incorporate negative counts.")
    return n0_initial * math.exp(-decay_constant_lambda * t_elapsed)


@validate(decay_constant_lambda=RULES.NON_NEGATIVE)
def half_life_from_decay_constant(decay_constant_lambda: float) -> float:
    r"""
    Map characteristic structural system half-lives using raw isotopic decay parameters ($T_{1/2} = \ln(2) / \lambda$).

    Project: Project Formulon-Physics
    """
    if decay_constant_lambda == 0:
        raise ZeroDivisionError("Stable isotopic targets incorporate a decay profile tracking to zero (infinite lifetime limits).")
    return math.log(2.0) / decay_constant_lambda


@validate(mass_defect_kg=RULES.REAL)
def binding_energy_from_mass_defect(mass_defect_kg: float) -> float:
    r"""
    Translate raw structural mass missing differentials directly into nuclear binding holding forces ($E_{be} = \Delta m \cdot c^2$).

    Project: Project Formulon-Physics
    """
    return mass_defect_kg * (C_SPEED_LIGHT ** 2)