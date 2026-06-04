"""
classical_mechanics.py
======================
Comprehensive engineering library for classical mechanics covering kinematics, 
dynamics, work-energy, rotational systems, orbital mechanics, and gravity.

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
    # 1. Linear Kinematics & Free Fall
    "average_velocity",
    "average_acceleration",
    "displacement_constant_a",
    "final_velocity_constant_a",
    "velocity_displacement_relation",
    "displacement_from_velocities",
    "freefall_velocity",
    "freefall_height",
    "time_of_flight_vertical",
    # 2. Projectile Motion
    "projectile_time_of_flight",
    "projectile_max_height",
    "projectile_range",
    "projectile_trajectory_y",
    # 3. Linear Dynamics & Forces
    "newtons_second_law",
    "weight",
    "linear_momentum",
    "newtons_second_law_derivative",
    "impulse_constant_force",
    "impulse_variable_force",
    "static_friction_max",
    "kinetic_friction",
    "hookes_law",
    "apparent_weight_upward",
    "apparent_weight_downward",
    # 4. Rotational Kinematics & Centripetal Systems
    "angular_velocity",
    "angular_acceleration",
    "linear_displacement_from_angular",
    "linear_velocity_from_angular",
    "tangential_acceleration",
    "centripetal_acceleration",
    "centripetal_force",
    "rotational_final_velocity",
    "rotational_displacement",
    "rotational_velocity_displacement_relation",
    # 5. Rotational Dynamics & Inertia
    "torque",
    "discrete_moment_of_inertia",
    "continuous_moment_of_inertia",
    "parallel_axis_theorem",
    "perpendicular_axis_theorem",
    "angular_momentum_particle",
    "angular_momentum_rigid_body",
    "torque_angular_momentum_relation",
    # 6. Work, Energy & Power
    "work_constant_force",
    "work_variable_force",
    "translational_kinetic_energy",
    "rotational_kinetic_energy",
    "gravitational_potential_energy",
    "elastic_potential_energy",
    "work_energy_theorem_translational",
    "average_power",
    "instantaneous_power",
    "mechanical_efficiency",
    # 7. Systems of Particles & Center of Mass
    "center_mass_discrete",
    # 8. Gravitation & Orbital Mechanics
    "newtons_law_of_gravitation",
    "gravitational_field_strength",
    "gravitational_potential",
    "gravitational_potential_energy_cosmic",
    "orbital_velocity",
    "escape_velocity",
    "keplers_third_law_period",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. LINEAR KINEMATICS & FREE FALL (9 Formulas)
# ══════════════════════════════════════════════════════════════════════════════

@validate(delta_x=RULES.DISPLACEMENT, delta_t=RULES.POSITIVE_TIME)
def average_velocity(delta_x: float, delta_t: float) -> float:
    """
    Calculate the average velocity over a given spatial displacement and time interval ($v_{avg} = \\Delta x / \\Delta t$).

    Project: Project Formulon-Physics
    """
    return delta_x / delta_t


@validate(delta_v=RULES.VELOCITY, delta_t=RULES.POSITIVE_TIME)
def average_acceleration(delta_v: float, delta_t: float) -> float:
    """
    Calculate the average linear acceleration ($a_{avg} = \\Delta v / \\Delta t$).

    Project: Project Formulon-Physics
    """
    return delta_v / delta_t


@validate(v0=RULES.VELOCITY, t=RULES.NON_NEG_TIME, a=RULES.ACCELERATION)
def displacement_constant_a(v0: float, t: float, a: float) -> float:
    """
    Calculate displacement under constant acceleration ($x = v_0 t + \\frac{1}{2}a t^2$).

    Project: Project Formulon-Physics
    """
    return (v0 * t) + (0.5 * a * (t ** 2))


@validate(v0=RULES.VELOCITY, a=RULES.ACCELERATION, t=RULES.NON_NEG_TIME)
def final_velocity_constant_a(v0: float, a: float, t: float) -> float:
    """
    Calculate final velocity under constant acceleration ($v = v_0 + a t$).

    Project: Project Formulon-Physics
    """
    return v0 + (a * t)


@validate(v0=RULES.VELOCITY, a=RULES.ACCELERATION, delta_x=RULES.DISPLACEMENT)
def velocity_displacement_relation(v0: float, a: float, delta_x: float) -> float:
    """
    Calculate final velocity magnitude squared relation ($v = \\sqrt{v_0^2 + 2 a \\Delta x}$).

    Project: Project Formulon-Physics
    """
    radicand = (v0 ** 2) + (2.0 * a * delta_x)
    if radicand < 0:
        raise ValueError("Physical configuration yields a negative radical; velocity domain violation.")
    return math.sqrt(radicand)


@validate(v0=RULES.VELOCITY, v=RULES.VELOCITY, t=RULES.NON_NEG_TIME)
def displacement_from_velocities(v0: float, v: float, t: float) -> float:
    """
    Calculate displacement using initial and final velocities under uniform acceleration ($x = \\frac{v_0 + v}{2} t$).

    Project: Project Formulon-Physics
    """
    return 0.5 * (v0 + v) * t


@validate(g=RULES.GRAVITY, t=RULES.NON_NEG_TIME)
def freefall_velocity(g: float, t: float) -> float:
    """
    Calculate the speed of an object dropped from rest under standard freefall conditions ($v = g t$).

    Project: Project Formulon-Physics
    """
    return g * t


@validate(g=RULES.GRAVITY, t=RULES.NON_NEG_TIME)
def freefall_height(g: float, t: float) -> float:
    """
    Calculate the vertical drop distance of an object from rest under freefall ($h = \\frac{1}{2}g t^2$).

    Project: Project Formulon-Physics
    """
    return 0.5 * g * (t ** 2)


@validate(h=RULES.HEIGHT, g=RULES.GRAVITY)
def time_of_flight_vertical(h: float, g: float) -> float:
    """
    Calculate time to reach baseline floor when dropped vertically from height h ($t = \\sqrt{2h/g}$).

    Project: Project Formulon-Physics
    """
    return math.sqrt((2.0 * h) / g)


# ══════════════════════════════════════════════════════════════════════════════
# 2. PROJECTILE MOTION (4 Formulas)
# ══════════════════════════════════════════════════════════════════════════════

@validate(v0=RULES.SPEED, theta_deg=RULES.ANGLE_DEG_90, g=RULES.GRAVITY)
def projectile_time_of_flight(v0: float, theta_deg: float, g: float) -> float:
    """
    Calculate total projectile time of flight over symmetric plane surfaces ($T = \\frac{2 v_0 \\sin\\theta}{g}$).

    Project: Project Formulon-Physics
    """
    return (2.0 * v0 * math.sin(math.radians(theta_deg))) / g


@validate(v0=RULES.SPEED, theta_deg=RULES.ANGLE_DEG_90, g=RULES.GRAVITY)
def projectile_max_height(v0: float, theta_deg: float, g: float) -> float:
    """
    Calculate maximum trajectory vertex elevation height reached by a projectile ($H = \\frac{v_0^2 \\sin^2\\theta}{2g}$).

    Project: Project Formulon-Physics
    """
    return ((v0 * math.sin(math.radians(theta_deg))) ** 2) / (2.0 * g)


@validate(v0=RULES.SPEED, theta_deg=RULES.ANGLE_DEG_90, g=RULES.GRAVITY)
def projectile_range(v0: float, theta_deg: float, g: float) -> float:
    """
    Calculate horizontal ground downrange footprint displacement achieved ($R = \\frac{v_0^2 \\sin(2\\theta)}{g}$).

    Project: Project Formulon-Physics
    """
    return ((v0 ** 2) * math.sin(2.0 * math.radians(theta_deg))) / g


@validate(x=RULES.DISTANCE, v0=RULES.SPEED, theta_deg=RULES.ANGLE_DEG_90, g=RULES.GRAVITY)
def projectile_trajectory_y(x: float, v0: float, theta_deg: float, g: float) -> float:
    """
    Solve for trajectory altitude coordinate y given horizontal downrange index position x via the trajectory equation.

    Project: Project Formulon-Physics
    """
    rad = math.radians(theta_deg)
    tan_term = x * math.tan(rad)
    cos_term = v0 * math.cos(rad)
    if cos_term == 0:
        raise ZeroDivisionError("Horizontal speed vector configuration vanishes; vertical singularity.")
    return tan_term - ((g * (x ** 2)) / (2.0 * (cos_term ** 2)))


# ══════════════════════════════════════════════════════════════════════════════
# 3. LINEAR DYNAMICS & FORCES (11 Formulas)
# ══════════════════════════════════════════════════════════════════════════════

@validate(mass=RULES.MASS, acceleration=RULES.ACCELERATION)
def newtons_second_law(mass: float, acceleration: float) -> float:
    """
    Calculate net active force acting on an inertial system ($F = m a$).

    Project: Project Formulon-Physics
    """
    return mass * acceleration


@validate(mass=RULES.MASS, g=RULES.GRAVITY)
def weight(mass: float, g: float) -> float:
    """
    Calculate localized static weight force vector magnitude ($W = m g$).

    Project: Project Formulon-Physics
    """
    return mass * g


@validate(mass=RULES.MASS, velocity=RULES.VELOCITY)
def linear_momentum(mass: float, velocity: float) -> float:
    """
    Calculate linear momentum of a translation system coordinate ($p = m v$).

    Project: Project Formulon-Physics
    """
    return mass * velocity


@validate(delta_p=RULES.MOMENTUM, delta_t=RULES.POSITIVE_TIME)
def newtons_second_law_derivative(delta_p: float, delta_t: float) -> float:
    """
    Calculate force acting as a direct time rate change of linear momentum system state ($F = \\Delta p / \\Delta t$).

    Project: Project Formulon-Physics
    """
    return delta_p / delta_t


@validate(force=RULES.FORCE, delta_t=RULES.POSITIVE_TIME)
def impulse_constant_force(force: float, delta_t: float) -> float:
    """
    Calculate structural linear impulse under a constant force payload window ($J = F \\Delta t$).

    Project: Project Formulon-Physics
    """
    return force * delta_t


def impulse_variable_force(force_func: Callable[[float], float], t_start: float, t_end: float) -> float:
    """
    Calculate impulse delivered across time tracking profiles under fluctuating variable load windows ($J = \\int F(t) dt$).

    Project: Project Formulon-Physics
    """
    if t_start < 0 or t_end < 0:
        raise ValueError("Time bounds cannot fall beneath structural absolute metrics.")
    validate_cross_param_le(t_start, t_end, "t_start", "t_end")
    val, _ = integrate.quad(force_func, t_start, t_end)
    return val


@validate(mu_s=RULES.COEFF_FRICTION, normal_force=RULES.NORMAL_FORCE)
def static_friction_max(mu_s: float, normal_force: float) -> float:
    """
    Calculate absolute limit threshold value before structural boundary interface slippage occurs ($f_{s,max} = \\mu_s F_n$).

    Project: Project Formulon-Physics
    """
    return mu_s * normal_force


@validate(mu_k=RULES.COEFF_FRICTION, normal_force=RULES.NORMAL_FORCE)
def kinetic_friction(mu_k: float, normal_force: float) -> float:
    """
    Calculate uniform interface dynamic friction mechanical constraint value during slippage ($f_k = \\mu_k F_n$).

    Project: Project Formulon-Physics
    """
    return mu_k * normal_force


@validate(k=RULES.SPRING_CONST, x=RULES.SPRING_DISPLACEMENT)
def hookes_law(k: float, x: float) -> float:
    """
    Calculate structural physical linear restoring force vector magnitude of an elastic spring model ($F = -k x$).

    Project: Project Formulon-Physics
    """
    return -k * x


@validate(mass=RULES.MASS, g=RULES.GRAVITY, a_up=RULES.NON_NEGATIVE)
def apparent_weight_upward(mass: float, g: float, a_up: float) -> float:
    """
    Calculate internal apparent weight inside an upward accelerated frame system container ($N = m(g + a)$).

    Project: Project Formulon-Physics
    """
    return mass * (g + a_up)


@validate(mass=RULES.MASS, g=RULES.GRAVITY, a_down=RULES.NON_NEGATIVE)
def apparent_weight_downward(mass: float, g: float, a_down: float) -> float:
    """
    Calculate internal apparent weight inside a downward accelerated frame environment container ($N = m(g - a)$).

    Project: Project Formulon-Physics
    """
    validate_cross_param_le(a_down, g, "a_down", "g")
    return mass * (g - a_down)


# ══════════════════════════════════════════════════════════════════════════════
# 4. ROTATIONAL KINEMATICS & CENTRIPETAL SYSTEMS (10 Formulas)
# ══════════════════════════════════════════════════════════════════════════════

@validate(delta_theta=RULES.REAL, delta_t=RULES.POSITIVE_TIME)
def angular_velocity(delta_theta: float, delta_t: float) -> float:
    """
    Calculate average scalar angular velocity metric ($\omega = \\Delta \\theta / \\Delta t$).

    Project: Project Formulon-Physics
    """
    return delta_theta / delta_t


@validate(delta_omega=RULES.REAL, delta_t=RULES.POSITIVE_TIME)
def angular_acceleration(delta_omega: float, delta_t: float) -> float:
    """
    Calculate average scalar angular acceleration tracking indicator ($\\alpha = \\Delta \omega / \\Delta t$).

    Project: Project Formulon-Physics
    """
    return delta_omega / delta_t


@validate(radius=RULES.RADIUS, theta_rad=RULES.REAL)
def linear_displacement_from_angular(radius: float, theta_rad: float) -> float:
    """
    Translate pure localized angular metric tracking dimensions directly into linear arc lengths ($s = r \\theta$).

    Project: Project Formulon-Physics
    """
    return radius * theta_rad


@validate(radius=RULES.RADIUS, omega_rad_s=RULES.REAL)
def linear_velocity_from_angular(radius: float, omega_rad_s: float) -> float:
    """
    Determine tangential edge translational velocity component vector magnitude scales ($v = r \omega$).

    Project: Project Formulon-Physics
    """
    return radius * omega_rad_s


@validate(radius=RULES.RADIUS, alpha_rad_s2=RULES.REAL)
def tangential_acceleration(radius: float, alpha_rad_s2: float) -> float:
    """
    Evaluate translation edge boundary tangential structural acceleration properties ($a_t = r \\alpha$).

    Project: Project Formulon-Physics
    """
    return radius * alpha_rad_s2


@validate(v=RULES.SPEED, radius=RULES.RADIUS)
def centripetal_acceleration(v: float, radius: float) -> float:
    """
    Calculate inbound center directed uniform centripetal coordinate acceleration components ($a_c = v^2 / r$).

    Project: Project Formulon-Physics
    """
    return (v ** 2) / radius


@validate(mass=RULES.MASS, v=RULES.SPEED, radius=RULES.RADIUS)
def centripetal_force(mass: float, v: float, radius: float) -> float:
    """
    Calculate active vector constraint forces required to sustain stable curvature trajectory loops ($F_c = m v^2 / r$).

    Project: Project Formulon-Physics
    """
    return (mass * (v ** 2)) / radius


@validate(omega0=RULES.REAL, alpha=RULES.REAL, t=RULES.NON_NEG_TIME)
def rotational_final_velocity(omega0: float, alpha: float, t: float) -> float:
    """
    Calculate final state angular velocity under uniform coordinate angular acceleration ($\omega = \omega_0 + \\alpha t$).

    Project: Project Formulon-Physics
    """
    return omega0 + (alpha * t)


@validate(omega0=RULES.REAL, t=RULES.NON_NEG_TIME, alpha=RULES.REAL)
def rotational_displacement(omega0: float, t: float, alpha: float) -> float:
    """
    Calculate angular rotation displacement tracking index coordinates ($\\theta = \omega_0 t + \\frac{1}{2}\\alpha t^2$).

    Project: Project Formulon-Physics
    """
    return (omega0 * t) + (0.5 * alpha * (t ** 2))


@validate(omega0=RULES.REAL, alpha=RULES.REAL, delta_theta=RULES.REAL)
def rotational_velocity_displacement_relation(omega0: float, alpha: float, delta_theta: float) -> float:
    """
    Calculate rotational angular velocity relationship thresholds ($\omega = \\sqrt{\omega_0^2 + 2 \\alpha \\Delta \\theta}$).

    Project: Project Formulon-Physics
    """
    radicand = (omega0 ** 2) + (2.0 * alpha * delta_theta)
    if radicand < 0:
        raise ValueError("Rotational kinetic constraint calculation fields drop below zero; imaginary domain break.")
    return math.sqrt(radicand)


# ══════════════════════════════════════════════════════════════════════════════
# 5. ROTATIONAL DYNAMICS & INERTIA (8 Formulas)
# ══════════════════════════════════════════════════════════════════════════════

@validate(radius=RULES.RADIUS, force=RULES.FORCE, theta_deg=RULES.ANGLE_DEG)
def torque(radius: float, force: float, theta_deg: float) -> float:
    """
    Calculate structural directional lever arm twisting torque force moments ($\\tau = r F \\sin\\theta$).

    Project: Project Formulon-Physics
    """
    return radius * force * math.sin(math.radians(theta_deg))


def discrete_moment_of_inertia(masses: list[float], radii: list[float]) -> float:
    """
    Calculate total rotational moment of inertia across collections of point mass distribution nodes ($I = \\sum m_i r_i^2$).

    Project: Project Formulon-Physics
    """
    if len(masses) != len(radii):
        raise ValueError("Mass node collections must align with spatial coordinate placement radii.")
    total_i = 0.0
    for m, r in zip(masses, radii):
        if m < 0 or r < 0:
            raise ValueError("Mass matrix metrics and relative coordinate radiuses must remain non-negative properties.")
        total_i += m * (r ** 2)
    if total_i <= 0:
        raise ValueError("Physical inertia profiles require valid space distribution tracking boundaries.")
    return total_i


def continuous_moment_of_inertia(density_func: Callable[[float], float], r_min: float, r_max: float) -> float:
    """
    Integrate rotational system profiles to determine continuous volume mass inertia metrics ($I = \\int r^2 dm$).

    Project: Project Formulon-Physics
    """
    if r_min < 0 or r_max < 0:
        raise ValueError("Continuous limits cannot map beneath absolute metric positions.")
    validate_cross_param_le(r_min, r_max, "r_min", "r_max")
    # Integrand corresponds to r^2 * dm(r) -> r^2 * density_func(r)
    val, _ = integrate.quad(lambda r: (r ** 2) * density_func(r), r_min, r_max)
    return val


@validate(i_cm=RULES.MOMENT_OF_INERTIA, mass=RULES.MASS, d=RULES.PARALLEL_AXIS_D)
def parallel_axis_theorem(i_cm: float, mass: float, d: float) -> float:
    """
    Shift tracking center axes references utilizing the parallel axis translation equation ($I = I_{cm} + m d^2$).

    Project: Project Formulon-Physics
    """
    return i_cm + (mass * (d ** 2))


@validate(i_x=RULES.MOMENT_OF_INERTIA, i_y=RULES.MOMENT_OF_INERTIA)
def perpendicular_axis_theorem(i_x: float, i_y: float) -> float:
    """
    Evaluate structural planar reference sheet rotational inertia layers via perpendicular coordinate axes ($I_z = I_x + I_y$).

    Project: Project Formulon-Physics
    """
    return i_x + i_y


@validate(radius=RULES.RADIUS, linear_momentum_val=RULES.REAL, theta_deg=RULES.ANGLE_DEG)
def angular_momentum_particle(radius: float, linear_momentum_val: float, theta_deg: float) -> float:
    """
    Determine specific localized vector orbital angular momentum values tracking dynamic particles ($L = r p \\sin\\theta$).

    Project: Project Formulon-Physics
    """
    return radius * linear_momentum_val * math.sin(math.radians(theta_deg))


@validate(i=RULES.MOMENT_OF_INERTIA, omega_rad_s=RULES.REAL)
def angular_momentum_rigid_body(i: float, omega_rad_s: float) -> float:
    """
    Determine consolidated macroscopic angular momentum vectors across rigid bodies ($L = I \omega$).

    Project: Project Formulon-Physics
    """
    return i * omega_rad_s


@validate(delta_l=RULES.REAL, delta_t=RULES.POSITIVE_TIME)
def torque_angular_momentum_relation(delta_l: float, delta_t: float) -> float:
    """
    Calculate dynamic angular torque as direct continuous time rates of angular momentum mutations ($\\tau = \\Delta L / \\Delta t$).

    Project: Project Formulon-Physics
    """
    return delta_l / delta_t


# ══════════════════════════════════════════════════════════════════════════════
# 6. WORK, ENERGY & POWER (10 Formulas)
# ══════════════════════════════════════════════════════════════════════════════

@validate(force=RULES.FORCE, displacement=RULES.DISPLACEMENT, theta_deg=RULES.WORK_ANGLE_DEG)
def work_constant_force(force: float, displacement: float, theta_deg: float) -> float:
    """
    Calculate mechanical work translations completed by absolute uniform constant directional loads ($W = F d \\cos\\theta$).

    Project: Project Formulon-Physics
    """
    return force * displacement * math.cos(math.radians(theta_deg))


def work_variable_force(force_func: Callable[[float], float], x_start: float, x_end: float) -> float:
    """
    Evaluate energetic integrals determining work distributions matching continuous changing forces ($W = \\int F(x) dx$).

    Project: Project Formulon-Physics
    """
    validate_cross_param_le(x_start, x_end, "x_start", "x_end")
    val, _ = integrate.quad(force_func, x_start, x_end)
    return val


@validate(mass=RULES.MASS, velocity=RULES.VELOCITY)
def translational_kinetic_energy(mass: float, velocity: float) -> float:
    """
    Calculate active macroscopic translational kinetic state work limits ($K = \\frac{1}{2}m v^2$).

    Project: Project Formulon-Physics
    """
    return 0.5 * mass * (velocity ** 2)


@validate(i=RULES.MOMENT_OF_INERTIA, omega_rad_s=RULES.REAL)
def rotational_kinetic_energy(i: float, omega_rad_s: float) -> float:
    """
    Calculate internal spin-state energy metrics across rotating bodies ($K_{rot} = \\frac{1}{2}I \omega^2$).

    Project: Project Formulon-Physics
    """
    return 0.5 * i * (omega_rad_s ** 2)


@validate(mass=RULES.MASS, g=RULES.GRAVITY, height=RULES.REAL)
def gravitational_potential_energy(mass: float, g: float, height: float) -> float:
    """
    Determine localized configuration potential energy balances within regular constant fields ($U_g = m g h$).

    Project: Project Formulon-Physics
    """
    return mass * g * height


@validate(k=RULES.SPRING_CONST, delta_x=RULES.SPRING_DISPLACEMENT)
def elastic_potential_energy(k: float, delta_x: float) -> float:
    """
    Evaluate structural internal conservation grid power loaded into compressed springs ($U_s = \\frac{1}{2}k x^2$).

    Project: Project Formulon-Physics
    """
    return 0.5 * k * (delta_x ** 2)


@validate(w_net=RULES.REAL)
def work_energy_theorem_translational(w_net: float) -> float:
    """
    Express net external spatial load allocations as direct mutations across total kinetic fields ($\\Delta K = W_{net}$).

    Project: Project Formulon-Physics
    """
    return w_net


@validate(work=RULES.REAL, delta_t=RULES.POSITIVE_TIME)
def average_power(work: float, delta_t: float) -> float:
    """
    Measure gross average energetic dissipation throughput velocities ($P_{avg} = W / \\Delta t$).

    Project: Project Formulon-Physics
    """
    return work / delta_t


@validate(force=RULES.FORCE, velocity=RULES.SPEED)
def instantaneous_power(force: float, velocity: float) -> float:
    """
    Map point-in-time work injection velocities based on load configurations ($P = F v$).

    Project: Project Formulon-Physics
    """
    return force * velocity


@validate(work_out=RULES.WORK_IO, work_in=RULES.WORK_IO)
def mechanical_efficiency(work_out: float, work_in: float) -> float:
    """
    Measure systemic mechanical efficiency percentages comparing load footprints ($\\eta = [W_{out} / W_{in}] \\times 100$).

    Project: Project Formulon-Physics
    """
    validate_cross_param_le(work_out, work_in, "work_out", "work_in")
    return (work_out / work_in) * 100.0


# ══════════════════════════════════════════════════════════════════════════════
# 7. SYSTEMS OF PARTICLES & CENTER OF MASS (1 Formula)
# ══════════════════════════════════════════════════════════════════════════════

def center_mass_discrete(masses: list[float], coordinates: list[float]) -> float:
    """
    Determine the composite structural mean balance point across discrete distributed particle systems ($x_{cm} = \\frac{\\sum m_i x_i}{\\sum m_i}$).

    Project: Project Formulon-Physics
    """
    if len(masses) != len(coordinates) or len(masses) == 0:
        raise ValueError("Particle tracking mass arrays must match geometric positioning indices.")
    
    weighted_sum = 0.0
    total_mass = 0.0
    for m, x in zip(masses, coordinates):
        if m < 1e-31:
            raise ValueError("Matter distribution properties require strictly valid positive mass boundaries.")
        weighted_sum += m * x
        total_mass += m
        
    return weighted_sum / total_mass


# ══════════════════════════════════════════════════════════════════════════════
# 8. GRAVITATION & ORBITAL MECHANICS (7 Formulas)
# ══════════════════════════════════════════════════════════════════════════════

@validate(m1=RULES.MASS, m2=RULES.MASS, r=RULES.RADIUS)
def newtons_law_of_gravitation(m1: float, m2: float, r: float) -> float:
    """
    Calculate cosmic gravitational attraction forces shared by isolated physical systems ($F = G \\frac{m_1 m_2}{r^2}$).

    Project: Project Formulon-Physics
    """
    big_g = 6.6743e-11
    return big_g * m1 * m2 / (r ** 2)


@validate(m_source=RULES.MASS, r=RULES.RADIUS)
def gravitational_field_strength(m_source: float, r: float) -> float:
    """
    Evaluate structural cosmic pull fields active at external radial separation indexes ($g = G M / r^2$).

    Project: Project Formulon-Physics
    """
    big_g = 6.6743e-11
    return (big_g * m_source) / (r ** 2)


@validate(m_source=RULES.MASS, r=RULES.RADIUS)
def gravitational_potential(m_source: float, r: float) -> float:
    """
    Evaluate specific potential metrics representing mechanical position states within deep spaces ($V = -G M / r$).

    Project: Project Formulon-Physics
    """
    big_g = 6.6743e-11
    return -(big_g * m_source) / r


@validate(m1=RULES.MASS, m2=RULES.MASS, r=RULES.RADIUS)
def gravitational_potential_energy_cosmic(m1: float, m2: float, r: float) -> float:
    """
    Evaluate absolute continuous macroscopic mechanical binding energies spanning orbital scales ($U = -G \\frac{m_1 m_2}{r}$).

    Project: Project Formulon-Physics
    """
    big_g = 6.6743e-11
    return -(big_g * m1 * m2) / r


@validate(m_central=RULES.MASS, r=RULES.RADIUS)
def orbital_velocity(m_central: float, r: float) -> float:
    """
    Calculate stable horizontal trajectory velocities required to maintain circular paths ($v = \\sqrt{G M / r}$).

    Project: Project Formulon-Physics
    """
    big_g = 6.6743e-11
    return math.sqrt((big_g * m_central) / r)


@validate(m_central=RULES.MASS, r=RULES.RADIUS)
def escape_velocity(m_central: float, r: float) -> float:
    """
    Calculate entry launch thresholds needed to exit localized gravitational field wells ($v_{esc} = \\sqrt{2 G M / r}$).

    Project: Project Formulon-Physics
    """
    big_g = 6.6743e-11
    return math.sqrt((2.0 * big_g * m_central) / r)


@validate(m_central=RULES.MASS, semi_major_axis_r=RULES.RADIUS)
def keplers_third_law_period(m_central: float, semi_major_axis_r: float) -> float:
    """
    Determine structural periodic orbital cycle parameters via Kepler's harmonic matching ($T = \\sqrt{\\frac{4\\pi^2 r^3}{G M}}$).

    Project: Project Formulon-Physics
    """
    big_g = 6.6743e-11
    numerator = 4.0 * (math.pi ** 2) * (semi_major_axis_r ** 3)
    denominator = big_g * m_central
    return math.sqrt(numerator / denominator)