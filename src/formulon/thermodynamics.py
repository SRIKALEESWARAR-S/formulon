"""
thermodynamics_statistical_mechanics.py
=======================================
Comprehensive engineering library covering thermal expansion, calorimetry, kinetic 
theory, non-ideal gases, classical laws of thermodynamics, and transport mechanisms.

All formulations are validated prior to execution using physical boundary constraints
and protected against runtime anomalies via the external validators framework.

Project: Project Formulon-Physics
License: MIT License
"""

import math
from typing import Callable
import scipy.integrate as integrate
from validators import validate, RULES, validate_cross_param_le

__all__ = [
    # 1. Thermal Expansion & Calorimetry
    "linear_expansion",
    "volume_expansion",
    "sensible_heat",
    "latent_heat",
    # 2. Kinetic Theory & Gases
    "ideal_gas_pressure",
    "van_der_waals_pressure",
    "average_molecular_kinetic_energy",
    "rms_velocity",
    "mean_free_path",
    # 3. Laws of Thermodynamics & Cycles
    "first_law_internal_energy",
    "work_variable_volume",
    "work_isothermal",
    "adiabatic_process_constant",
    "work_adiabatic",
    "mayers_relation_cp",
    "adiabatic_index_from_degrees_freedom",
    "carnot_efficiency",
    "entropy_change",
    "boltzmann_entropy",
    # 4. Heat Transfer & Transport Phenomena
    "thermal_conduction_rate",
    "stefan_boltzmann_radiation",
    "wiens_displacement_wavelength",
    "newtons_cooling_rate",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. THERMAL EXPANSION & CALORIMETRIC HEAT (4 Formulas)
# ══════════════════════════════════════════════════════════════════════════════

@validate(alpha=RULES.REAL, l0=RULES.DISTANCE, delta_t=RULES.REAL)
def linear_expansion(alpha: float, l0: float, delta_t: float) -> float:
    """
    Calculate linear spatial elongation under changing temperature states ($\Delta L = \alpha L_0 \Delta T$).

    Project: Project Formulon-Physics
    """
    return alpha * l0 * delta_t


@validate(beta=RULES.REAL, v0=RULES.VOLUME, delta_t=RULES.REAL)
def volume_expansion(beta: float, v0: float, delta_t: float) -> float:
    """
    Calculate volumetric expansion matching continuous thermal adjustments ($\Delta V = \beta V_0 \Delta T$).

    Project: Project Formulon-Physics
    """
    return beta * v0 * delta_t


@validate(mass=RULES.MASS, specific_heat=RULES.SPECIFIC_HEAT, delta_t=RULES.REAL)
def sensible_heat(mass: float, specific_heat: float, delta_t: float) -> float:
    """
    Evaluate heat energy exchanges driving single-phase thermal delta shifts ($Q = m c \Delta T$).

    Project: Project Formulon-Physics
    """
    return mass * specific_heat * delta_t


@validate(mass=RULES.MASS, specific_latent_heat=RULES.LATENT_HEAT)
def latent_heat(mass: float, specific_latent_heat: float) -> float:
    """
    Evaluate enthalpy configurations driving physical pure isothermal phase variations ($Q = m L$).

    Project: Project Formulon-Physics
    """
    return mass * specific_latent_heat


# ══════════════════════════════════════════════════════════════════════════════
# 2. KINETIC THEORY & GAS DYNAMICS (5 Formulas)
# ══════════════════════════════════════════════════════════════════════════════

@validate(moles=RULES.MOLES, temperature=RULES.TEMPERATURE, volume=RULES.VOLUME)
def ideal_gas_pressure(moles: float, temperature: float, volume: float) -> float:
    """
    Solve for ideal gas state pressures using thermodynamic parameter matching ($P = nRT / V$).

    Project: Project Formulon-Physics
    """
    r_gas = 8.314462618
    return (moles * r_gas * temperature) / volume


@validate(moles=RULES.MOLES, volume=RULES.VOLUME, temperature=RULES.TEMPERATURE, a_factor=RULES.NON_NEGATIVE, b_factor=RULES.NON_NEGATIVE)
def van_der_waals_pressure(moles: float, volume: float, temperature: float, a_factor: float, b_factor: float) -> float:
    """
    Predict non-ideal real gas state pressures accounting for intermolecular attractions and co-volumes.

    $P = \\frac{nRT}{V - nb} - \\frac{an^2}{V^2}$

    Project: Project Formulon-Physics
    """
    r_gas = 8.314462618
    co_volume_limit = moles * b_factor
    validate_cross_param_le(co_volume_limit, volume, "moles * b_factor", "volume")
    
    if volume == co_volume_limit:
        raise ZeroDivisionError("System boundaries compressed to absolute minimum molecular structural limit thresholds.")
        
    repulsive_term = (moles * r_gas * temperature) / (volume - co_volume_limit)
    attractive_term = (a_factor * (moles ** 2)) / (volume ** 2)
    return repulsive_term - attractive_term


@validate(temperature=RULES.TEMPERATURE)
def average_molecular_kinetic_energy(temperature: float) -> float:
    """
    Determine specific mean localized molecular translation kinetic energy footprints ($K_{avg} = \\frac{3}{2}k_B T$).

    Project: Project Formulon-Physics
    """
    k_b = 1.380649e-23
    return 1.5 * k_b * temperature


@validate(temperature=RULES.TEMPERATURE, molar_mass=RULES.MOLAR_MASS)
def rms_velocity(temperature: float, molar_mass: float) -> float:
    """
    Calculate root-mean-square thermal velocities tracking particle metrics ($v_{rms} = \\sqrt{3RT / M}$).

    Project: Project Formulon-Physics
    """
    r_gas = 8.314462618
    return math.sqrt((3.0 * r_gas * temperature) / molar_mass)


@validate(molecular_diameter=RULES.DISTANCE, number_density=RULES.NUMBER_DENSITY)
def mean_free_path(molecular_diameter: float, number_density: float) -> float:
    """
    Calculate average displacement tracking scales between successive collision events ($\lambda = \\frac{1}{\\sqrt{2}\pi d^2 n}$).

    Project: Project Formulon-Physics
    """
    denominator = math.sqrt(2.0) * math.pi * (molecular_diameter ** 2) * number_density
    return 1.0 / denominator


# ══════════════════════════════════════════════════════════════════════════════
# 3. LAWS OF THERMODYNAMICS & CYCLES (10 Formulas)
# ══════════════════════════════════════════════════════════════════════════════

@validate(heat_added=RULES.REAL, work_done=RULES.REAL)
def first_law_internal_energy(heat_added: float, work_done: float) -> float:
    """
    Express energy balance tracking matching classical First Law limits ($dU = dQ - dW$).

    Project: Project Formulon-Physics
    """
    return heat_added - work_done


def work_variable_volume(pressure_func: Callable[[float], float], v_start: float, v_end: float) -> float:
    """
    Integrate dynamic boundary variations determining work distributions ($W = \\int P(V) dV$).

    Project: Project Formulon-Physics
    """
    if v_start <= 0 or v_end <= 0:
        raise ValueError("Volumetric boundaries must map across positive physical spaces.")
    validate_cross_param_le(v_start, v_end, "v_start", "v_end")
    val, _ = integrate.quad(pressure_func, v_start, v_end)
    return val


@validate(moles=RULES.MOLES, temperature=RULES.TEMPERATURE, v1=RULES.VOLUME, v2=RULES.VOLUME)
def work_isothermal(moles: float, temperature: float, v1: float, v2: float) -> float:
    """
    Calculate mechanical work deliveries completed along pure isothermal limits ($W = nRT \\ln(V_2 / V_1)$).

    Project: Project Formulon-Physics
    """
    r_gas = 8.314462618
    return moles * r_gas * temperature * math.log(v2 / v1)


@validate(pressure=RULES.PRESSURE, volume=RULES.VOLUME, gamma=RULES.ADIABATIC_INDEX)
def adiabatic_process_constant(pressure: float, volume: float, gamma: float) -> float:
    """
    Isolate invariant pathway tracking indexes matching zero-heat transfer horizons ($C = P V^\\gamma$).

    Project: Project Formulon-Physics
    """
    return pressure * (volume ** gamma)


@validate(p1=RULES.PRESSURE, v1=RULES.VOLUME, p2=RULES.PRESSURE, v2=RULES.VOLUME, gamma=RULES.ADIABATIC_INDEX)
def work_adiabatic(p1: float, v1: float, p2: float, v2: float, gamma: float) -> float:
    """
    Solve for energetic work translations performed inside non-isentropic expansion boundaries.

    $W = \\frac{P_1 V_1 - P_2 V_2}{\\gamma - 1}$

    Project: Project Formulon-Physics
    """
    if gamma == 1.0:
        raise ZeroDivisionError("Adiabatic limits cannot evaluate over unity indexes (isothermal boundary break).")
    numerator = (p1 * v1) - (p2 * v2)
    return numerator / (gamma - 1.0)


@validate(cv_molar=RULES.SPECIFIC_HEAT)
def mayers_relation_cp(cv_molar: float) -> float:
    """
    Map constant pressure specific capacities using Mayer's ideal relation ($C_p = C_v + R$).

    Project: Project Formulon-Physics
    """
    r_gas = 8.314462618
    return cv_molar + r_gas


@validate(degrees_freedom=RULES.DEGREES_FREEDOM)
def adiabatic_index_from_degrees_freedom(degrees_freedom: int) -> float:
    """
    Evaluate structural molecular heat capacity ratio properties tracking degrees of freedom ($\\gamma = 1 + 2/f$).

    Project: Project Formulon-Physics
    """
    return 1.0 + (2.0 / degrees_freedom)


@validate(t_cold=RULES.TEMPERATURE, t_hot=RULES.TEMPERATURE)
def carnot_efficiency(t_cold: float, t_hot: float) -> float:
    """
    Measure absolute theoretical thermodynamic efficiency conversion limits across heat engines ($\\eta = 1 - T_C / T_H$).

    Project: Project Formulon-Physics
    """
    validate_cross_param_le(t_cold, t_hot, "t_cold", "t_hot")
    return 1.0 - (t_cold / t_hot)


@validate(heat_reversible=RULES.REAL, temperature=RULES.TEMPERATURE)
def entropy_change(heat_reversible: float, temperature: float) -> float:
    """
    Measure state degradation parameter mutations matching specific heat additions ($dS = dQ_{rev} / T$).

    Project: Project Formulon-Physics
    """
    return heat_reversible / temperature


@validate(microstates_w=RULES.MICROSTATES)
def boltzmann_entropy(microstates_w: float) -> float:
    """
    Determine statistical entropy values from microscopic configuration multiplicity bounds ($S = k_B \\ln W$).

    Project: Project Formulon-Physics
    """
    k_b = 1.380649e-23
    return k_b * math.log(microstates_w)


# ══════════════════════════════════════════════════════════════════════════════
# 4. HEAT TRANSFER & TRANSPORT PHENOMENA (4 Formulas)
# ══════════════════════════════════════════════════════════════════════════════

@validate(k_conduct=RULES.THERMAL_CONDUCTIVITY, area=RULES.AREA, delta_t=RULES.REAL, thickness_dx=RULES.DISTANCE)
def thermal_conduction_rate(k_conduct: float, area: float, delta_t: float, thickness_dx: float) -> float:
    """
    Measure directional linear conductive heat energy flow fluxes utilizing Fourier's framework ($dQ/dt = -k A \\frac{\\Delta T}{\\Delta x}$).

    Project: Project Formulon-Physics
    """
    return -(k_conduct * area * delta_t) / thickness_dx


@validate(emissivity=RULES.EMISSIVITY, area=RULES.AREA, temperature=RULES.TEMPERATURE)
def stefan_boltzmann_radiation(emissivity: float, area: float, temperature: float) -> float:
    """
    Calculate macroscopic ideal blackbody electromagnetic energetic discharge radiation power ($P = \\epsilon \\sigma A T^4$).

    Project: Project Formulon-Physics
    """
    sigma = 5.670374419e-8
    return emissivity * sigma * area * (temperature ** 4)


@validate(temperature=RULES.TEMPERATURE)
def wiens_displacement_wavelength(temperature: float) -> float:
    """
    Isolate peak spectral emission wavelengths matching specific temperature states via Wien's relation ($\\lambda_{max} = b / T$).

    Project: Project Formulon-Physics
    """
    wien_b = 2.897771955e-3
    return wien_b / temperature


@validate(cooling_constant_k=RULES.NON_NEGATIVE, t_body=RULES.TEMPERATURE, t_ambient=RULES.TEMPERATURE)
def newtons_cooling_rate(cooling_constant_k: float, t_body: float, t_ambient: float) -> float:
    """
    Predict rate tracking parameters governing convective temperature drop states ($dT/dt = -K(T_{body} - T_{ambient})$).

    Project: Project Formulon-Physics
    """
    return -cooling_constant_k * (t_body - t_ambient)