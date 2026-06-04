"""
waves_oscillations_optics.py
============================
Comprehensive engineering library covering simple harmonic motion, wave mechanics,
acoustics, geometric optics, and wave optics formulations.

All formulations are validated prior to execution using physical boundary constraints
and protected against runtime anomalies via the external validators framework.

Project: Project Formulon-Physics
License: MIT License
"""

import math
from typing import Tuple
from validators import validate, RULES, validate_cross_param_le

__all__ = [
    # 1. Simple Harmonic Motion (SHM)
    "shm_acceleration_from_displacement",
    "shm_displacement",
    "shm_velocity",
    "shm_velocity_from_displacement",
    "shm_acceleration",
    "spring_angular_frequency",
    "spring_period",
    "pendulum_period",
    "physical_pendulum_period",
    "shm_total_energy",
    # 2. Wave Motion & Sound
    "wave_speed",
    "wave_number",
    "string_wave_speed",
    "gas_sound_speed",
    "sound_intensity_level",
    "doppler_effect",
    "beat_frequency",
    "open_pipe_fundamental",
    "closed_pipe_fundamental",
    # 3. Geometric & Physical Optics
    "refractive_index",
    "snells_law_angle",
    "critical_angle",
    "mirror_focal_relation",
    "thin_lens_relation",
    "lens_makers_equation",
    "lens_power",
    "mirror_magnification",
    "lens_magnification",
    "malus_law",
    "brewsters_angle",
    "double_slit_fringe_width",
    "single_slit_diffraction_angle",
    "rayleigh_criterion",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. SIMPLE HARMONIC MOTION (SHM)
# ══════════════════════════════════════════════════════════════════════════════

@validate(omega=RULES.ANGULAR_FREQUENCY, x=RULES.DISPLACEMENT)
def shm_acceleration_from_displacement(omega: float, x: float) -> float:
    """
    Calculate SHM acceleration directly from the differential equation constraint ($a = -\omega^2 x$).

    Project: Project Formulon-Physics
    """
    return -(omega ** 2) * x


@validate(amplitude=RULES.AMPLITUDE, omega=RULES.ANGULAR_FREQUENCY, t=RULES.NON_NEG_TIME, phi_rad=RULES.REAL)
def shm_displacement(amplitude: float, omega: float, t: float, phi_rad: float) -> float:
    """
    Calculate system displacement tracking position state in SHM ($x(t) = A \cos(\omega t + \phi)$).

    Project: Project Formulon-Physics
    """
    return amplitude * math.cos((omega * t) + phi_rad)


@validate(amplitude=RULES.AMPLITUDE, omega=RULES.ANGULAR_FREQUENCY, t=RULES.NON_NEG_TIME, phi_rad=RULES.REAL)
def shm_velocity(amplitude: float, omega: float, t: float, phi_rad: float) -> float:
    """
    Calculate structural phase space velocity tracking in SHM ($v(t) = -A \omega \sin(\omega t + \phi)$).

    Project: Project Formulon-Physics
    """
    return -amplitude * omega * math.sin((omega * t) + phi_rad)


@validate(omega=RULES.ANGULAR_FREQUENCY, amplitude=RULES.AMPLITUDE, x=RULES.DISPLACEMENT)
def shm_velocity_from_displacement(omega: float, amplitude: float, x: float) -> float:
    """
    Calculate the velocity magnitude given a localized spatial displacement coordinate ($v = \pm\omega\sqrt{A^2 - x^2}$).

    Project: Project Formulon-Physics
    """
    validate_cross_param_le(abs(x), amplitude, "abs(x)", "amplitude")
    return omega * math.sqrt((amplitude ** 2) - (x ** 2))


@validate(amplitude=RULES.AMPLITUDE, omega=RULES.ANGULAR_FREQUENCY, t=RULES.NON_NEG_TIME, phi_rad=RULES.REAL)
def shm_acceleration(amplitude: float, omega: float, t: float, phi_rad: float) -> float:
    """
    Calculate dynamic structural linear acceleration tracking in SHM ($a(t) = -A \omega^2 \cos(\omega t + \phi)$).

    Project: Project Formulon-Physics
    """
    return -amplitude * (omega ** 2) * math.cos((omega * t) + phi_rad)


@validate(k=RULES.SPRING_CONST, mass=RULES.MASS)
def spring_angular_frequency(k: float, mass: float) -> float:
    """
    Evaluate structural natural angular frequency for ideal elastic spring systems ($\omega = \sqrt{k/m}$).

    Project: Project Formulon-Physics
    """
    return math.sqrt(k / mass)


@validate(mass=RULES.MASS, k=RULES.SPRING_CONST)
def spring_period(mass: float, k: float) -> float:
    """
    Calculate natural structural oscillation time period of an elastic spring model ($T = 2\pi\sqrt{m/k}$).

    Project: Project Formulon-Physics
    """
    return 2.0 * math.pi * math.sqrt(mass / k)


@validate(length=RULES.DISTANCE, g=RULES.GRAVITY)
def pendulum_period(length: float, g: float) -> float:
    """
    Calculate the small-angle time oscillation period of an idealized simple pendulum ($T = 2\pi\sqrt{L/g}$).

    Project: Project Formulon-Physics
    """
    return 2.0 * math.pi * math.sqrt(length / g)


@validate(i=RULES.MOMENT_OF_INERTIA, mass=RULES.MASS, g=RULES.GRAVITY, d=RULES.DISTANCE)
def physical_pendulum_period(i: float, mass: float, g: float, d: float) -> float:
    """
    Evaluate mechanical oscillation periods for rigid distributed physical pendulum systems ($T = 2\pi\sqrt{I / (mgd)}$).

    Project: Project Formulon-Physics
    """
    return 2.0 * math.pi * math.sqrt(i / (mass * g * d))


@validate(k=RULES.SPRING_CONST, amplitude=RULES.AMPLITUDE)
def shm_total_energy(k: float, amplitude: float) -> float:
    """
    Determine invariant consolidated total mechanical energy bound within an active SHM system ($E = \frac{1}{2}k A^2$).

    Project: Project Formulon-Physics
    """
    return 0.5 * k * (amplitude ** 2)


# ══════════════════════════════════════════════════════════════════════════════
# 2. WAVE MOTION & SOUND
# ══════════════════════════════════════════════════════════════════════════════

@validate(frequency=RULES.FREQUENCY, wavelength=RULES.WAVELENGTH)
def wave_speed(frequency: float, wavelength: float) -> float:
    """
    Calculate characteristic phase velocity propagation speeds ($v = f \lambda$).

    Project: Project Formulon-Physics
    """
    return frequency * wavelength


@validate(wavelength=RULES.WAVELENGTH)
def wave_number(wavelength: float) -> float:
    """
    Calculate spatial propagation angular wavenumber constraints ($k = 2\pi / \lambda$).

    Project: Project Formulon-Physics
    """
    return (2.0 * math.pi) / wavelength


@validate(tension=RULES.FORCE, linear_density=RULES.LINEAR_DENSITY)
def string_wave_speed(tension: float, linear_density: float) -> float:
    """
    Evaluate structural transverse elastic wave transmission speeds sustained across ideal strings ($v = \sqrt{T/\mu}$).

    Project: Project Formulon-Physics
    """
    return math.sqrt(tension / linear_density)


@validate(gamma=RULES.ADIABATIC_INDEX, r_gas=RULES.GAS_CONSTANT, temperature=RULES.TEMPERATURE, molar_mass=RULES.MOLAR_MASS)
def gas_sound_speed(gamma: float, r_gas: float, temperature: float, molar_mass: float) -> float:
    """
    Predict acoustic compression wave velocities inside gases using Laplace's formulation ($v = \sqrt{\gamma R T / M}$).

    Project: Project Formulon-Physics
    """
    return math.sqrt((gamma * r_gas * temperature) / molar_mass)


@validate(intensity=RULES.SOUND_INTENSITY, base_intensity=RULES.SOUND_INTENSITY_BASE)
def sound_intensity_level(intensity: float, base_intensity: float) -> float:
    """
    Translate pure surface vector acoustic fluxes into standard decibel scaling metrics ($\beta = 10 \log_{10}(I / I_0)$).

    Project: Project Formulon-Physics
    """
    return 10.0 * math.log10(intensity / base_intensity)


@validate(f_source=RULES.FREQUENCY, v_medium=RULES.SPEED, v_observer=RULES.SPEED, v_source=RULES.SPEED)
def doppler_effect(f_source: float, v_medium: float, v_observer: float, v_source: float, 
                   observer_moving_towards: bool, source_moving_towards: bool) -> float:
    """
    Calculate observed shifting frequencies matching relative motion kinematics under the Doppler phenomenon.

    $f' = f \left(\frac{v \pm v_o}{v \mp v_s}\right)$

    Project: Project Formulon-Physics
    """
    top_sign = 1.0 if observer_moving_towards else -1.0
    bottom_sign = -1.0 if source_moving_towards else 1.0
    
    numerator = v_medium + (top_sign * v_observer)
    denominator = v_medium + (bottom_sign * v_source)
    
    if denominator <= 0:
        raise ZeroDivisionError("System configuration breaks continuity limit boundaries; sonic boom or singular division.")
        
    return f_source * (numerator / denominator)


@validate(f1=RULES.FREQUENCY, f2=RULES.FREQUENCY)
def beat_frequency(f1: float, f2: float) -> float:
    """
    Isolate envelope modulations generated by dual superposed wave interference grids ($f_{beat} = |f_1 - f_2|$).

    Project: Project Formulon-Physics
    """
    return abs(f1 - f2)


@validate(v_sound=RULES.SPEED, length=RULES.DISTANCE)
def open_pipe_fundamental(v_sound: float, length: float) -> float:
    """
    Calculate the fundamental acoustic resonance frequency matching symmetric open acoustic boundaries ($f = v / 2L$).

    Project: Project Formulon-Physics
    """
    return v_sound / (2.0 * length)


@validate(v_sound=RULES.SPEED, length=RULES.DISTANCE)
def closed_pipe_fundamental(v_sound: float, length: float) -> float:
    """
    Calculate fundamental resonance modes inside asymmetric closed-ended acoustic boundary lines ($f = v / 4L$).

    Project: Project Formulon-Physics
    """
    return v_sound / (4.0 * length)


# ══════════════════════════════════════════════════════════════════════════════
# 3. GEOMETRIC & PHYSICAL OPTICS
# ══════════════════════════════════════════════════════════════════════════════

@validate(c_vacuum=RULES.SPEED, v_medium=RULES.SPEED)
def refractive_index(c_vacuum: float, v_medium: float) -> float:
    """
    Measure electromagnetic spatial phase restriction indices across media boundaries ($n = c / v$).

    Project: Project Formulon-Physics
    """
    validate_cross_param_le(v_medium, c_vacuum, "v_medium", "c_vacuum")
    return c_vacuum / v_medium


@validate(n1=RULES.REFRACTIVE_INDEX, n2=RULES.REFRACTIVE_INDEX, theta1_deg=RULES.ANGLE_DEG)
def snells_law_angle(n1: float, n2: float, theta1_deg: float) -> float:
    """
    Determine refract pathway angles across optical sheet discontinuities via Snell's framework.

    Project: Project Formulon-Physics
    """
    sin_theta2 = (n1 * math.sin(math.radians(theta1_deg))) / n2
    if not (-1.0 <= sin_theta2 <= 1.0):
        raise ValueError("System configurations trigger internal total reflection bounds; domain break.")
    return math.degrees(math.asin(sin_theta2))


@validate(n1=RULES.REFRACTIVE_INDEX, n2=RULES.REFRACTIVE_INDEX)
def critical_angle(n1: float, n2: float) -> float:
    """
    Isolate exact total internal reflection threshold angle points across interface boundaries ($\theta_c = \sin^{-1}(n_2 / n_1)$).

    Project: Project Formulon-Physics
    """
    validate_cross_param_le(n2, n1, "n2", "n1")
    return math.degrees(math.asin(n2 / n1))


@validate(u_obj=RULES.REAL, v_img=RULES.REAL)
def mirror_focal_relation(u_obj: float, v_img: float) -> float:
    """
    Solve for reflective focal lengths matching coordinate object mapping surfaces ($\frac{1}{f} = \frac{1}{v} + \frac{1}{u}$).

    Project: Project Formulon-Physics
    """
    if u_obj == 0 or v_img == 0:
        raise ZeroDivisionError("Optical point source tracking indices cannot reside on reference vertex intersections.")
    inverse_f = (1.0 / v_img) + (1.0 / u_obj)
    if inverse_f == 0:
        raise ZeroDivisionError("Focal length approaches infinite tracking constraints (planar reflection model).")
    return 1.0 / inverse_f


@validate(u_obj=RULES.REAL, v_img=RULES.REAL)
def thin_lens_relation(u_obj: float, v_img: float) -> float:
    """
    Solve for single thin-refractive focal properties under paraxial approximations ($\frac{1}{f} = \frac{1}{v} - \frac{1}{u}$).

    Project: Project Formulon-Physics
    """
    if u_obj == 0 or v_img == 0:
        raise ZeroDivisionError("Optical tracking points cannot intersect lens node core boundaries.")
    inverse_f = (1.0 / v_img) - (1.0 / u_obj)
    if inverse_f == 0:
        raise ZeroDivisionError("Infinite focal constraint limits encountered across zero-power refractive sheets.")
    return 1.0 / inverse_f


@validate(n_lens=RULES.REFRACTIVE_INDEX, n_medium=RULES.REFRACTIVE_INDEX, r1=RULES.REAL, r2=RULES.REAL)
def lens_makers_equation(n_lens: float, n_medium: float, r1: float, r2: float) -> float:
    """
    Evaluate structural focal parameters matching lens surface curvature indices via the Lens Maker formulation.

    $\frac{1}{f} = \left(\frac{n_{lens}}{n_{medium}} - 1\right)\left(\frac{1}{R_1} - \frac{1}{R_2}\right)$

    Project: Project Formulon-Physics
    """
    if r1 == 0 or r2 == 0:
        raise ZeroDivisionError("Curvature metrics cannot vanish across valid structural tracking horizons.")
    rel_index = (n_lens / n_medium) - 1.0
    geometry = (1.0 / r1) - (1.0 / r2)
    inverse_f = rel_index * geometry
    if inverse_f == 0:
        raise ZeroDivisionError("Composite physical geometries generate a net zero refractive performance footprint.")
    return 1.0 / inverse_f


@validate(f_meters=RULES.REAL)
def lens_power(f_meters: float) -> float:
    """
    Calculate optical focal convergence capacity parameters in Diopters ($P = 1 / f$).

    Project: Project Formulon-Physics
    """
    if f_meters == 0:
        raise ZeroDivisionError("Power calculation models approach infinity over planar tracking layers.")
    return 1.0 / f_meters


@validate(u_obj=RULES.REAL, v_img=RULES.REAL)
def mirror_magnification(u_obj: float, v_img: float) -> float:
    """
    Evaluate lateral magnification coefficients generated across mirror tracking spaces ($m = -v / u$).

    Project: Project Formulon-Physics
    """
    if u_obj == 0:
        raise ZeroDivisionError("Object spatial placement index cannot rest on mirror center planes.")
    return -v_img / u_obj


@validate(u_obj=RULES.REAL, v_img=RULES.REAL)
def lens_magnification(u_obj: float, v_img: float) -> float:
    """
    Evaluate lateral magnification tracking dimensions matching refractive lens components ($m = v / u$).

    Project: Project Formulon-Physics
    """
    if u_obj == 0:
        raise ZeroDivisionError("Object structural baseline tracking position cannot match lens central indices.")
    return v_img / u_obj


@validate(i0=RULES.SOUND_INTENSITY, theta_deg=RULES.ANGLE_DEG)
def malus_law(i0: float, theta_deg: float) -> float:
    """
    Measure transmission flux reduction parameters crossing polarization planes via Malus's Law ($I = I_0 \cos^2\theta$).

    Project: Project Formulon-Physics
    """
    return i0 * (math.cos(math.radians(theta_deg)) ** 2)


@validate(n1=RULES.REFRACTIVE_INDEX, n2=RULES.REFRACTIVE_INDEX)
def brewsters_angle(n1: float, n2: float) -> float:
    """
    Isolate polarizing brewster reflection coordinate angle configurations ($\tan\theta_p = n_2 / n_1$).

    Project: Project Formulon-Physics
    """
    return math.degrees(math.atan(n2 / n1))


@validate(wavelength=RULES.WAVELENGTH, d_screen=RULES.DISTANCE, d_slits=RULES.DISTANCE)
def double_slit_fringe_width(wavelength: float, d_screen: float, d_slits: float) -> float:
    """
    Calculate spatial interference fringe track separation width in Young's interference models ($\beta = \lambda D / d$).

    Project: Project Formulon-Physics
    """
    return (wavelength * d_screen) / d_slits


@validate(slit_width_a=RULES.DISTANCE, order_m=RULES.REAL, wavelength=RULES.WAVELENGTH)
def single_slit_diffraction_angle(slit_width_a: float, order_m: float, wavelength: float) -> float:
    """
    Evaluate structural minimum diffraction dispersion vectors under single slit footprints ($a \sin\theta = m\lambda$).

    Project: Project Formulon-Physics
    """
    sin_theta = (order_m * wavelength) / slit_width_a
    if not (-1.0 <= sin_theta <= 1.0):
        raise ValueError("Target structural parameter configurations fall outside physical diffraction windows.")
    return math.degrees(math.asin(sin_theta))


@validate(wavelength=RULES.WAVELENGTH, aperture_diameter_d=RULES.DISTANCE)
def rayleigh_criterion(wavelength: float, aperture_diameter_d: float) -> float:
    """
    Determine maximum angular resolution limits bounded under diffraction metrics ($\theta_{min} = 1.22 \lambda / D$).

    Project: Project Formulon-Physics
    """
    return (1.22 * wavelength) / aperture_diameter_d