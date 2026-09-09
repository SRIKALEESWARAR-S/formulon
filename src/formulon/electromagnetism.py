"""
electromagnetism.py
===================
Comprehensive engineering library covering electrostatics, capacitors, dielectrics,
magnetostatics, DC circuits, electrodynamics, AC networks, and Maxwell relationships.

All formulations are validated prior to execution using physical boundary constraints
and protected against runtime anomalies via the external validators framework.

Project: Project Formulon-Physics
License: Apache-2.0
"""

import math
from typing import Callable, Tuple
import scipy.integrate as integrate
from .validator import validate, RULES, validate_cross_param_le

__all__ = [
    # 1. Electrostatics
    "coulomb_force",
    "electric_field_strength",
    "point_charge_electric_field",
    "electric_flux",
    "gauss_law_enclosed_charge",
    "electric_potential_energy",
    "electric_potential",
    "electric_potential_from_field",
    "electric_dipole_moment",
    "torque_on_electric_dipole",
    "potential_energy_of_electric_dipole",
    # 2. Capacitors & Dielectrics
    "capacitance",
    "parallel_plate_capacitance",
    "dielectric_capacitance",
    "capacitors_series_equivalent",
    "capacitors_parallel_equivalent",
    "capacitor_stored_energy",
    "electric_field_energy_density",
    # 3. Magnetostatics
    "lorentz_magnetic_force",
    "wire_magnetic_force",
    "biot_savart_magnitude",
    "amperes_law_enclosed_current",
    "straight_wire_magnetic_field",
    "solenoid_magnetic_field",
    "magnetic_dipole_moment",
    "torque_on_magnetic_dipole",
    # 4. DC Circuits
    "electric_current",
    "current_density_drift",
    "ohms_law_voltage",
    "resistance_from_resistivity",
    "temperature_dependent_resistance",
    "resistors_series_equivalent",
    "resistors_parallel_equivalent",
    "joule_heating_power",
    # 5. Electrodynamics & AC Circuits
    "magnetic_flux",
    "faradays_induction_emf",
    "motional_emf",
    "self_inductance_flux",
    "inductor_induced_emf",
    "inductor_stored_energy",
    "magnetic_field_energy_density",
    "rlc_series_impedance",
    "inductive_reactance",
    "capacitive_reactance",
    "resonant_frequency",
    "resonant_angular_frequency",
    # 6. Maxwell's Equations & EM Waves
    "maxwell_ampere_displacement_current",
    "em_wave_speed_vacuum",
    "poynting_vector_magnitude",
]

# Physical Constants
EPSILON_0 = 8.8541878128e-12
MU_0 = 1.25663706212e-6
COULOMB_CONSTANT = 1.0 / (4.0 * math.pi * EPSILON_0)
C_SPEED_LIGHT = 299792458


# ══════════════════════════════════════════════════════════════════════════════
# 1. ELECTROSTATICS
# ══════════════════════════════════════════════════════════════════════════════

@validate(q1=RULES.REAL, q2=RULES.REAL, r=RULES.RADIUS)
def coulomb_force(q1: float, q2: float, r: float) -> float:
    r"""
    Calculate electrostatic attraction or repulsion force between localized charges ($F = \\frac{1}{4\\pi\\epsilon_0}\\frac{q_1 q_2}{r^2}$).

    Project: Project Formulon-Physics
    """
    return COULOMB_CONSTANT * (q1 * q2) / (r ** 2)


@validate(force=RULES.FORCE, charge=RULES.CHARGE)
def electric_field_strength(force: float, charge: float) -> float:
    """
    Evaluate structural electric field vectors experienced by localized probe charges ($E = F / q$).

    Project: Project Formulon-Physics
    """
    return force / charge


@validate(q=RULES.REAL, r=RULES.RADIUS)
def point_charge_electric_field(q: float, r: float) -> float:
    r"""
    Evaluate point source electric fields matching radial isolation tracking scales ($E = \\frac{1}{4\\pi\\epsilon_0}\\frac{q}{r^2}$).

    Project: Project Formulon-Physics
    """
    return COULOMB_CONSTANT * q / (r ** 2)


@validate(area=RULES.AREA, theta_deg=RULES.ANGLE_DEG)
def electric_flux(field_func: Callable[[float, float], float], area: float, theta_deg: float) -> float:
    r"""
    Evaluate static electric surface field flux metrics across simple planar window bounds ($\Phi_E = E A \\cos\\theta$).

    Project: Project Formulon-Physics
    """
    # Evaluating average field boundary conditions placeholder using standard functional parameters
    avg_field = field_func(0.0, 0.0)
    return avg_field * area * math.cos(math.radians(theta_deg))


@validate(q_encl=RULES.REAL)
def gauss_law_enclosed_charge(q_encl: float) -> float:
    r"""
    Determine net surface field flux matching enclosed boundary charge profiles via Gauss's model ($\Phi_E = Q_{encl}/\\epsilon_0$).

    Project: Project Formulon-Physics
    """
    return q_encl / EPSILON_0


@validate(q1=RULES.REAL, q2=RULES.REAL, r=RULES.RADIUS)
def electric_potential_energy(q1: float, q2: float, r: float) -> float:
    r"""
    Evaluate conservation electrostatic binding potential energy metrics ($U = \\frac{1}{4\\pi\\epsilon_0}\\frac{q_1 q_2}{r}$).

    Project: Project Formulon-Physics
    """
    return COULOMB_CONSTANT * (q1 * q2) / r


@validate(energy=RULES.REAL, charge=RULES.CHARGE)
def electric_potential(energy: float, charge: float) -> float:
    """
    Determine local work potentials matching specific metric positions ($V = U / q$).

    Project: Project Formulon-Physics
    """
    return energy / charge


def electric_potential_from_field(field_func: Callable[[float], float], l_start: float, l_end: float) -> float:
    r"""
    Integrate electric vector fields determining potential transitions down line paths ($V = -\\int E \\cdot dl$).

    Project: Project Formulon-Physics
    """
    validate_cross_param_le(l_start, l_end, "l_start", "l_end")
    val, _ = integrate.quad(field_func, l_start, l_end)
    return -val


@validate(charge=RULES.CHARGE, separation_d=RULES.DISTANCE)
def electric_dipole_moment(charge: float, separation_d: float) -> float:
    """
    Calculate primary electrostatic physical dipole distribution metrics ($p = q d$).

    Project: Project Formulon-Physics
    """
    return charge * separation_d


@validate(moment_p=RULES.REAL, field_e=RULES.REAL, theta_deg=RULES.ANGLE_DEG)
def torque_on_electric_dipole(moment_p: float, field_e: float, theta_deg: float) -> float:
    r"""
    Calculate rotational mechanical moments applied across charge pairs inside static tracking fields ($\\tau = p E \\sin\\theta$).

    Project: Project Formulon-Physics
    """
    return moment_p * field_e * math.sin(math.radians(theta_deg))


@validate(moment_p=RULES.REAL, field_e=RULES.REAL, theta_deg=RULES.ANGLE_DEG)
def potential_energy_of_electric_dipole(moment_p: float, field_e: float, theta_deg: float) -> float:
    r"""
    Evaluate configuration orientation energetic metrics for electrical dipoles ($U = -p E \\cos\\theta$).

    Project: Project Formulon-Physics
    """
    return -moment_p * field_e * math.cos(math.radians(theta_deg))


# ══════════════════════════════════════════════════════════════════════════════
# 2. CAPACITORS & DIELECTRICS
# ══════════════════════════════════════════════════════════════════════════════

@validate(charge=RULES.REAL, voltage=RULES.VOLTAGE)
def capacitance(charge: float, voltage: float) -> float:
    """
    Measure standard operational electrostatic capacity bounds ($C = Q / V$).

    Project: Project Formulon-Physics
    """
    return charge / voltage


@validate(plate_area_a=RULES.AREA, separation_d=RULES.DISTANCE)
def parallel_plate_capacitance(plate_area_a: float, separation_d: float) -> float:
    r"""
    Evaluate structural metrics calculating capacities across geometric sheets ($C = \\epsilon_0 A / d$).

    Project: Project Formulon-Physics
    """
    return (EPSILON_0 * plate_area_a) / separation_d


@validate(dielectric_kappa=RULES.DIELECTRIC_CONSTANT, c_vacuum=RULES.REAL)
def dielectric_capacitance(dielectric_kappa: float, c_vacuum: float) -> float:
    r"""
    Scale baseline charge storage capacities using material dielectric properties ($C = \\kappa C_0$).

    Project: Project Formulon-Physics
    """
    return dielectric_kappa * c_vacuum


def capacitors_series_equivalent(capacitances: list[float]) -> float:
    r"""
    Calculate composite series capacity equivalence parameters ($\\frac{1}{C_{eq}} = \\sum \\frac{1}{C_i}$).

    Project: Project Formulon-Physics
    """
    if len(capacitances) == 0:
        raise ValueError("Capacitor configuration sequences must incorporate operational nodes.")
    inverse_sum = 0.0
    for c in capacitances:
        if c <= 0:
            raise ValueError("Capacitance tracking values must maintain strictly positive properties.")
        inverse_sum += 1.0 / c
    return 1.0 / inverse_sum


def capacitors_parallel_equivalent(capacitances: list[float]) -> float:
    r"""
    Sum capacity configurations matching parallel electrical pathways ($C_{eq} = \\sum C_i$).

    Project: Project Formulon-Physics
    """
    if len(capacitances) == 0:
        raise ValueError("Capacitor configuration sets must contain real computational tracking nodes.")
    total_c = 0.0
    for c in capacitances:
        if c <= 0:
            raise ValueError("Capacitance tracking metrics require positive operational dimensions.")
        total_c += c
    return total_c


@validate(c=RULES.REAL, voltage=RULES.VOLTAGE)
def capacitor_stored_energy(c: float, voltage: float) -> float:
    r"""
    Measure conservation electrostatic work thresholds stacked inside charged fields ($U = \\frac{1}{2}C V^2$).

    Project: Project Formulon-Physics
    """
    return 0.5 * c * (voltage ** 2)


@validate(field_e=RULES.REAL)
def electric_field_energy_density(field_e: float) -> float:
    r"""
    Evaluate localized specific electric potential energy density metrics ($u_E = \\frac{1}{2}\\epsilon_0 E^2$).

    Project: Project Formulon-Physics
    """
    return 0.5 * EPSILON_0 * (field_e ** 2)


# ══════════════════════════════════════════════════════════════════════════════
# 3. MAGNETOSTATICS
# ══════════════════════════════════════════════════════════════════════════════

@validate(charge=RULES.REAL, velocity=RULES.SPEED, field_b=RULES.MAGNETIC_FIELD, theta_deg=RULES.ANGLE_DEG)
def lorentz_magnetic_force(charge: float, velocity: float, field_b: float, theta_deg: float) -> float:
    r"""
    Measure structural magnetic deflection vectors handling moving points via Lorentz limits ($F = q v B \\sin\\theta$).

    Project: Project Formulon-Physics
    """
    return charge * velocity * field_b * math.sin(math.radians(theta_deg))


@validate(current=RULES.CURRENT, length=RULES.DISTANCE, field_b=RULES.MAGNETIC_FIELD, theta_deg=RULES.ANGLE_DEG)
def wire_magnetic_force(current: float, length: float, field_b: float, theta_deg: float) -> float:
    r"""
    Calculate composite deflection load distributions sustained over conductive wires ($F = I L B \\sin\\theta$).

    Project: Project Formulon-Physics
    """
    return current * length * field_b * math.sin(math.radians(theta_deg))


@validate(current=RULES.CURRENT, length_dl=RULES.DISTANCE, r=RULES.RADIUS, theta_deg=RULES.ANGLE_DEG)
def biot_savart_magnitude(current: float, length_dl: float, r: float, theta_deg: float) -> float:
    r"""
    Predict differential flux density tracking metrics utilizing the Biot-Savart equation ($dB = \\frac{\\mu_0}{4\\pi}\\frac{I dl \\sin\\theta}{r^2}$).

    Project: Project Formulon-Physics
    """
    factor = MU_0 / (4.0 * math.pi)
    return factor * (current * length_dl * math.sin(math.radians(theta_deg))) / (r ** 2)


@validate(i_encl=RULES.CURRENT)
def amperes_law_enclosed_current(i_encl: float) -> float:
    r"""
    Evaluate circulation loop limits given enclosed structural currents via Ampere's law ($\oint B \\cdot dl = \\mu_0 I_{encl}$).

    Project: Project Formulon-Physics
    """
    return MU_0 * i_encl


@validate(current=RULES.CURRENT, r=RULES.RADIUS)
def straight_wire_magnetic_field(current: float, r: float) -> float:
    r"""
    Evaluate induced field vectors matching infinite linear current lines ($B = \\frac{\\mu_0 I}{2\\pi r}$).

    Project: Project Formulon-Physics
    """
    return (MU_0 * current) / (2.0 * math.pi * r)


@validate(turns_per_meter_n=RULES.TURNS_PER_LENGTH, current=RULES.CURRENT)
def solenoid_magnetic_field(turns_per_meter_n: float, current: float) -> float:
    r"""
    Predict internal uniform magnetic field layers inside long structural solenoids ($B = \\mu_0 n I$).

    Project: Project Formulon-Physics
    """
    return MU_0 * turns_per_meter_n * current


@validate(turns_n=RULES.NON_NEGATIVE, current=RULES.CURRENT, area=RULES.AREA)
def magnetic_dipole_moment(turns_n: int, current: float, area: float) -> float:
    r"""
    Determine tracking loop magnetic vector moments ($\mu = N I A$).

    Project: Project Formulon-Physics
    """
    return float(turns_n) * current * area


@validate(moment_mu=RULES.REAL, field_b=RULES.MAGNETIC_FIELD, theta_deg=RULES.ANGLE_DEG)
def torque_on_magnetic_dipole(moment_mu: float, field_b: float, theta_deg: float) -> float:
    r"""
    Measure twisting mechanics acting across loop moment distributions inside field parameters ($\\tau = \\mu B \\sin\\theta$).

    Project: Project Formulon-Physics
    """
    return moment_mu * field_b * math.sin(math.radians(theta_deg))


# ══════════════════════════════════════════════════════════════════════════════
# 4. DC CIRCUITS
# ══════════════════════════════════════════════════════════════════════════════

@validate(delta_q=RULES.REAL, delta_t=RULES.POSITIVE_TIME)
def electric_current(delta_q: float, delta_t: float) -> float:
    r"""
    Calculate standard macroscopic charge transit flow rates ($I = \\Delta q / \\Delta t$).

    Project: Project Formulon-Physics
    """
    return delta_q / delta_t


@validate(carrier_density_n=RULES.NUMBER_DENSITY, unit_charge_q=RULES.REAL, drift_velocity_vd=RULES.SPEED)
def current_density_drift(carrier_density_n: float, unit_charge_q: float, drift_velocity_vd: float) -> float:
    """
    Measure localized charge vector flux concentrations mapping particle kinetics ($J = n q v_d$).

    Project: Project Formulon-Physics
    """
    return carrier_density_n * unit_charge_q * drift_velocity_vd


@validate(current=RULES.CURRENT, resistance=RULES.RESISTANCE)
def ohms_law_voltage(current: float, resistance: float) -> float:
    """
    Solve for single tracking loop operational potential drops via Ohm's relation ($V = I R$).

    Project: Project Formulon-Physics
    """
    return current * resistance


@validate(resistivity_rho=RULES.RESISTIVITY, length=RULES.DISTANCE, area=RULES.AREA)
def resistance_from_resistivity(resistivity_rho: float, length: float, area: float) -> float:
    r"""
    Translate uniform material property parameters into absolute component resistances ($R = \\rho L / A$).

    Project: Project Formulon-Physics
    """
    return (resistivity_rho * length) / area


@validate(r0=RULES.RESISTANCE, alpha_coefficient=RULES.REAL, delta_t=RULES.REAL)
def temperature_dependent_resistance(r0: float, alpha_coefficient: float, delta_t: float) -> float:
    r"""
    Evaluate linear tracking thermal mutations across component resistance steps ($R(T) = R_0[1 + \\alpha \\Delta T]$).

    Project: Project Formulon-Physics
    """
    factor = 1.0 + (alpha_coefficient * delta_t)
    if factor < 0:
        raise ValueError("Physical thermal configurations force component metrics below absolute zero boundaries.")
    return r0 * factor


def resistors_series_equivalent(resistances: list[float]) -> float:
    r"""
    Sum resistances across continuous series circuit paths ($R_{eq} = \\sum R_i$).

    Project: Project Formulon-Physics
    """
    if len(resistances) == 0:
        raise ValueError("Resistor tracking records must contain operating line layout entries.")
    total_r = 0.0
    for r in resistances:
        if r <= 0:
            raise ValueError("Resistance component parameters require positive engineering dimensions.")
        total_r += r
    return total_r


def resistors_parallel_equivalent(resistances: list[float]) -> float:
    r"""
    Calculate composite parallel pathway network resistance footprints ($\\frac{1}{R_{eq}} = \\sum \\frac{1}{R_i}$).

    Project: Project Formulon-Physics
    """
    if len(resistances) == 0:
        raise ValueError("Circuit node collections require valid structural resistor data entries.")
    inverse_sum = 0.0
    for r in resistances:
        if r <= 0:
            raise ValueError("Component resistance parameters must maintain positive metric entries.")
        inverse_sum += 1.0 / r
    return 1.0 / inverse_sum


@validate(voltage=RULES.VOLTAGE, current=RULES.CURRENT)
def joule_heating_power(voltage: float, current: float) -> float:
    """
    Measure electrical energy dissipation rates crossing circuit pathways via Joule's law ($P = V I$).

    Project: Project Formulon-Physics
    """
    return voltage * current


# ══════════════════════════════════════════════════════════════════════════════
# 5. ELECTRODYNAMICS & AC CIRCUITS
# ══════════════════════════════════════════════════════════════════════════════

@validate(area=RULES.AREA, theta_deg=RULES.ANGLE_DEG)
def magnetic_flux(field_func: Callable[[float, float], float], area: float, theta_deg: float) -> float:
    r"""
    Measure surface induction flux profiles scaling vector intersections ($\Phi_B = B A \\cos\\theta$).

    Project: Project Formulon-Physics
    """
    avg_field = field_func(0.0, 0.0)
    return avg_field * area * math.cos(math.radians(theta_deg))


@validate(delta_flux_b=RULES.REAL, delta_t=RULES.POSITIVE_TIME)
def faradays_induction_emf(delta_flux_b: float, delta_t: float) -> float:
    r"""
    Calculate induced electromotive potentials tracking flux changes via Faraday's framework ($\\mathcal{E} = -\\Delta\Phi_B / \\Delta t$).

    Project: Project Formulon-Physics
    """
    return -delta_flux_b / delta_t


@validate(field_b=RULES.MAGNETIC_FIELD, length=RULES.DISTANCE, velocity=RULES.SPEED)
def motional_emf(field_b: float, length: float, velocity: float) -> float:
    r"""
    Calculate kinetic potential distributions driven along conductors cutting tracking fields ($\\mathcal{E} = B L v$).

    Project: Project Formulon-Physics
    """
    return field_b * length * velocity


@validate(flux_b=RULES.REAL, current=RULES.CURRENT)
def self_inductance_flux(flux_b: float, current: float) -> float:
    r"""
    Evaluate structural induction scale invariants matching loop configurations ($L = \Phi_B / I$).

    Project: Project Formulon-Physics
    """
    return flux_b / current


@validate(inductance_l=RULES.REAL, delta_i=RULES.REAL, delta_t=RULES.POSITIVE_TIME)
def inductor_induced_emf(inductance_l: float, delta_i: float, delta_t: float) -> float:
    r"""
    Calculate inductive reverse potential boundaries bucking active current changes ($\\mathcal{E} = -L \\frac{\\Delta I}{\\Delta t}$).

    Project: Project Formulon-Physics
    """
    if inductance_l < 0:
        raise ValueError("Inductance track invariants cannot process negative physical properties.")
    return -inductance_l * (delta_i / delta_t)


@validate(inductance_l=RULES.REAL, current=RULES.CURRENT)
def inductor_stored_energy(inductance_l: float, current: float) -> float:
    r"""
    Measure dynamic potential fields stored inside inductive vector arrays ($U = \\frac{1}{2}L I^2$).

    Project: Project Formulon-Physics
    """
    if inductance_l < 0:
        raise ValueError("Self-inductance components require strictly non-negative definitions.")
    return 0.5 * inductance_l * (current ** 2)


@validate(field_b=RULES.MAGNETIC_FIELD)
def magnetic_field_energy_density(field_b: float) -> float:
    r"""
    Evaluate localized specific potential energy density fields bound within induction arrays ($u_B = B^2 / (2\\mu_0)$).

    Project: Project Formulon-Physics
    """
    return (field_b ** 2) / (2.0 * MU_0)


@validate(resistance=RULES.RESISTANCE, x_l=RULES.REAL, x_c=RULES.REAL)
def rlc_series_impedance(resistance: float, x_l: float, x_c: float) -> float:
    r"""
    Calculate net alternate phase vector network impedance tracking limits across AC systems.

    $Z = \\sqrt{R^2 + (X_L - X_C)^2}$

    Project: Project Formulon-Physics
    """
    return math.sqrt((resistance ** 2) + ((x_l - x_c) ** 2))


@validate(angular_frequency_omega=RULES.ANGULAR_FREQUENCY, inductance_l=RULES.REAL)
def inductive_reactance(angular_frequency_omega: float, inductance_l: float) -> float:
    r"""
    Measure inductive AC tracking loop current phase block metrics ($X_L = \omega L$).

    Project: Project Formulon-Physics
    """
    if inductance_l < 0:
        raise ValueError("Inductive tracking properties require non-negative constraints.")
    return angular_frequency_omega * inductance_l


@validate(angular_frequency_omega=RULES.ANGULAR_FREQUENCY, capacitance_c=RULES.REAL)
def capacitive_reactance(angular_frequency_omega: float, capacitance_c: float) -> float:
    r"""
    Measure capacitive AC tracking phase expansion block metrics ($X_C = 1 / (\omega C)$).

    Project: Project Formulon-Physics
    """
    if capacitance_c <= 0:
        raise ValueError("Capacitive storage footprints require valid positive definitions.")
    return 1.0 / (angular_frequency_omega * capacitance_c)


@validate(inductance_l=RULES.REAL, capacitance_c=RULES.REAL)
def resonant_angular_frequency(inductance_l: float, capacitance_c: float) -> float:
    """Return the LC resonant angular frequency, omega_0 = 1/sqrt(LC), in rad/s."""
    if inductance_l <= 0 or capacitance_c <= 0:
        raise ValueError("Inductance and capacitance must be positive.")
    return 1.0 / math.sqrt(inductance_l * capacitance_c)


def resonant_frequency(inductance_l: float, capacitance_c: float) -> float:
    """Return the LC resonant frequency f_0 = 1/(2*pi*sqrt(LC)), in Hz."""
    return resonant_angular_frequency(inductance_l, capacitance_c) / (2.0 * math.pi)


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAXWELL'S EQUATIONS & EM WAVES
# ══════════════════════════════════════════════════════════════════════════════

@validate(i_encl=RULES.CURRENT, delta_flux_e=RULES.REAL, delta_t=RULES.POSITIVE_TIME)
def maxwell_ampere_displacement_current(i_encl: float, delta_flux_e: float, delta_t: float) -> float:
    r"""
    Evaluate continuous loop field circulations driving combined real conduction and dynamic displacement paths.

    $\oint B \\cdot dl = \\mu_0 \\left(I_{encl} + \\epsilon_0 \\frac{\\Delta\\Phi_E}{\\Delta t}\\right)$

    Project: Project Formulon-Physics
    """
    displacement_term = EPSILON_0 * (delta_flux_e / delta_t)
    return MU_0 * (i_encl + displacement_term)


def em_wave_speed_vacuum() -> float:
    r"""
    Return baseline structural phase propagation velocities evaluated for vacuum waves via continuum constants ($c = 1 / \\sqrt{\\mu_0 \\epsilon_0}$).

    Project: Project Formulon-Physics
    """
    return 1.0 / math.sqrt(MU_0 * EPSILON_0)


@validate(field_e=RULES.REAL, field_b=RULES.MAGNETIC_FIELD)
def poynting_vector_magnitude(field_e: float, field_b: float) -> float:
    r"""
    Measure directional electromagnetic energy flux power transfer throughput dimensions ($S = E B / \\mu_0$).

    Project: Project Formulon-Physics
    """
    return (field_e * field_b) / MU_0