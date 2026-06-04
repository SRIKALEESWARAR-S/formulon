"""
fluid_mechanics.py
==================
Comprehensive library covering fluid statics, fluid dynamics, elasticity, 
surface phenomena, and viscous transport phenomena.

All formulations are validated prior to execution using physical boundary constraints 
and protected against runtime anomalies (singularities, domain errors).

Project: Project Formulon-Physics
License: MIT License
"""

import math
from validators import validate, RULES, validate_cross_param_le

# Expose the public API for the PyPI package distribution
__all__ = [
    "density",
    "pressure",
    "hydrostatic_pressure",
    "buoyant_force",
    "continuity_equation",
    "bernoullis_equation",
    "torricellis_law",
    "reynolds_number",
    "stress",
    "strain",
    "youngs_modulus",
    "shear_modulus",
    "bulk_modulus",
    "surface_tension",
    "capillary_rise",
    "stokes_law",
    "poiseuilles_law",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. FLUID STATICS
# ══════════════════════════════════════════════════════════════════════════════

@validate(mass=RULES.MASS, volume=RULES.VOLUME)
def density(mass: float, volume: float) -> float:
    """
    Calculate the mass density of a uniform substance ($\rho = m/V$).

    Project: Project Formulon-Physics

    Parameters
    ----------
    mass : float
        The total mass of the body or fluid sample in kilograms (kg).
    volume : float
        The physical volume occupied by the matter in cubic meters (m³).

    Returns
    -------
    float
        The mass density of the object in kilograms per cubic meter (kg/m³).
    """
    return mass / volume


@validate(force=RULES.FORCE, area=RULES.AREA)
def pressure(force: float, area: float) -> float:
    """
    Calculate the average perpendicular pressure acting on a surface ($P = F_\perp/A$).

    Project: Project Formulon-Physics

    Parameters
    ----------
    force : float
        The magnitude of the normal force exerted on the surface area in Newtons (N).
    area : float
        The surface area across which the normal force is distributed in square meters (m²).

    Returns
    -------
    float
        The distributed pressure in Pascals (Pa).
    """
    return force / area


@validate(density=RULES.DENSITY, g=RULES.GRAVITY, depth=RULES.HEIGHT)
def hydrostatic_pressure(density: float, g: float, depth: float) -> float:
    """
    Calculate the gauge hydrostatic pressure at a specific depth within a fluid ($P = \rho g h$).

    Project: Project Formulon-Physics

    Parameters
    ----------
    density : float
        The mass density of the stagnant fluid in kilograms per cubic meter (kg/m³).
    g : float
        The local structural acceleration due to gravity in meters per second squared (m/s²).
    depth : float
        The vertical depth below the fluid's free surface layer in meters (m).

    Returns
    -------
    float
        The fluid hydrostatic gauge pressure at the target depth in Pascals (Pa).
    """
    return density * g * depth


@validate(fluid_density=RULES.DENSITY, volume_submerged=RULES.VOLUME, g=RULES.GRAVITY)
def buoyant_force(fluid_density: float, volume_submerged: float, g: float) -> float:
    """
    Calculate the upward buoyant force acting on a submerged body ($F_b = \rho_f V_{sub} g$).

    Project: Project Formulon-Physics

    Parameters
    ----------
    fluid_density : float
        The mass density of the surrounding displaced fluid in kilograms per cubic meter (kg/m³).
    volume_submerged : float
        The exact localized portion of the object volume submerged under the fluid line in cubic meters (m³).
    g : float
        The local gravitational acceleration parameter in meters per second squared (m/s²).

    Returns
    -------
    float
        The upward buoyant force vector magnitude in Newtons (N).
    """
    return fluid_density * volume_submerged * g


# ══════════════════════════════════════════════════════════════════════════════
# 2. FLUID DYNAMICS
# ══════════════════════════════════════════════════════════════════════════════

@validate(area1=RULES.AREA, velocity1=RULES.SPEED, area2=RULES.AREA)
def continuity_equation(area1: float, velocity1: float, area2: float) -> float:
    """
    Solve for the output velocity using the Continuity Equation ($A_1 v_1 = A_2 v_2$).

    Assumes steady, continuous, incompressible, non-viscous fluid stream tracking.

    Project: Project Formulon-Physics

    Parameters
    ----------
    area1 : float
        Cross-sectional area of the first conduit region in square meters (m²).
    velocity1 : float
        Average fluid flow scalar speed at the first conduit cross-section in meters per second (m/s).
    area2 : float
        Cross-sectional boundary area of the target exit section in square meters (m²).

    Returns
    -------
    float
        The corresponding downstream flow velocity vector magnitude in meters per second (m/s).
    """
    return (area1 * velocity1) / area2


@validate(
    p1=RULES.PRESSURE, density=RULES.DENSITY, v1=RULES.SPEED, y1=RULES.HEIGHT,
    v2=RULES.SPEED, y2=RULES.HEIGHT, g=RULES.GRAVITY
)
def bernoullis_equation(
    p1: float, density: float, v1: float, y1: float, v2: float, y2: float, g: float
) -> float:
    """
    Determine the local pressure at a downstream state using Bernoulli's Theorem.

    $P_2 = P_1 + \\frac{1}{2}\\rho(v_1^2 - v_2^2) + \\rho g(y_1 - y_2)$

    Project: Project Formulon-Physics

    Parameters
    ----------
    p1 : float
        Static fluid pressure at the baseline starting cross-section in Pascals (Pa).
    density : float
        Mass density of the moving continuous fluid matrix in kilograms per cubic meter (kg/m³).
    v1 : float
        Fluid flow velocity tracking baseline at state 1 in meters per second (m/s).
    y1 : float
        Elevation head altitude coordinate relative to reference at state 1 in meters (m).
    v2 : float
        Downstream flow velocity magnitude profile at state 2 in meters per second (m/s).
    y2 : float
        Elevation head altitude coordinate relative to reference at state 2 in meters (m).
    g : float
        Local gravitational field acceleration magnitude in meters per second squared (m/s²).

    Returns
    -------
    float
        The local downstream static pressure ($P_2$) evaluated at state 2 in Pascals (Pa).
    """
    kinetic_term = 0.5 * density * (v1**2 - v2**2)
    potential_term = density * g * (y1 - y2)
    return p1 + kinetic_term + potential_term


@validate(g=RULES.GRAVITY, liquid_height=RULES.HEIGHT)
def torricellis_law(g: float, liquid_height: float) -> float:
    """
    Calculate the theoretical speed of efflux from an open sharp-edged orifice via Torricelli's Law ($v = \sqrt{2gh}$).

    Project: Project Formulon-Physics

    Parameters
    ----------
    g : float
        The local environmental gravitational field strength acceleration in meters per second squared (m/s²).
    liquid_height : float
        The vertical height of the liquid fluid level surface above the center axis of the orifice hole in meters (m).

    Returns
    -------
    float
        The ideal fluid efflux escape speed leaving the opening point in meters per second (m/s).
    """
    return math.sqrt(2.0 * g * liquid_height)


@validate(density=RULES.DENSITY, flow_speed=RULES.SPEED, characteristic_length=RULES.DISTANCE, viscosity=RULES.VISCOSITY)
def reynolds_number(density: float, flow_speed: float, characteristic_length: float, viscosity: float) -> float:
    """
    Calculate the dimensionless Reynolds number evaluating fluid turbulence scales ($Re = \frac{\rho v L}{\mu}$).

    Project: Project Formulon-Physics

    Parameters
    ----------
    density : float
        The fluid mass matrix density profile tracking value in kilograms per cubic meter (kg/m³).
    flow_speed : float
        The average localized relative bulk movement velocity scale in meters per second (m/s).
    characteristic_length : float
        The defining spatial scale dimension geometry (e.g. pipe inner diameter, chord length) in meters (m).
    viscosity : float
        The fluid molecular dynamic shear viscosity coefficient parameter in Pascal-seconds (Pa·s).

    Returns
    -------
    float
        The dimensionless Reynolds number indicator tracking dynamic stability.
    """
    return (density * flow_speed * characteristic_length) / viscosity


# ══════════════════════════════════════════════════════════════════════════════
# 3. ELASTICITY & RHEOLOGY
# ══════════════════════════════════════════════════════════════════════════════

@validate(force=RULES.FORCE, cross_sectional_area=RULES.AREA)
def stress(force: float, cross_sectional_area: float) -> float:
    """
    Calculate the internal restorative mechanical stress distribution ($\sigma = F/A$).

    Project: Project Formulon-Physics

    Parameters
    ----------
    force : float
        The deforming force payload load vector magnitude acting across the profile boundary in Newtons (N).
    cross_sectional_area : float
        The structural continuous target material area perpendicular or parallel to loading in square meters (m²).

    Returns
    -------
    float
        The structural internal stress magnitude metric in Pascals (Pa).
    """
    return force / cross_sectional_area


@validate(delta_l=RULES.DISTANCE, initial_l=RULES.DISTANCE)
def strain(delta_l: float, initial_l: float) -> float:
    """
    Calculate the structural dimensionless linear deformation strain value ($\epsilon = \Delta L / L_0$).

    Project: Project Formulon-Physics

    Parameters
    ----------
    delta_l : float
        The absolute magnitude of spatial elongation or compressive displacement length change in meters (m).
    initial_l : float
        The original structural un-deformed resting dimension length of the element segment in meters (m).

    Returns
    -------
    float
        The dimensionless linear strain scalar value tracker.
    """
    if initial_l == 0:
        raise ZeroDivisionError("Original component mechanical length cannot evaluate to exactly zero.")
    return delta_l / initial_l


@validate(stress=RULES.REAL, strain=RULES.REAL)
def youngs_modulus(stress: float, strain: float) -> float:
    """
    Calculate the intrinsic Young's Modulus of elasticity tracking tensile rigidity ($E = \sigma / \epsilon$).

    Project: Project Formulon-Physics

    Parameters
    ----------
    stress : float
        The applied tensile or compressive mechanical stress loading metrics in Pascals (Pa).
    strain : float
        The corresponding recorded linear structural physical strain dimension ratio.

    Returns
    -------
    float
        The characteristic material Young's Elastic Modulus scale in Pascals (Pa).
    """
    if strain == 0:
        raise ZeroDivisionError("Cannot compute Young's Modulus when strain is precisely zero.")
    return stress / strain


@validate(shear_stress=RULES.REAL, shear_strain=RULES.REAL)
def shear_modulus(shear_stress: float, shear_strain: float) -> float:
    """
    Calculate the material Shear Modulus evaluation index ($G = \tau / \gamma$).

    Project: Project Formulon-Physics

    Parameters
    ----------
    shear_stress : float
        The applied planar or tangential structural slicing shear stress matrix scale in Pascals (Pa).
    shear_strain : float
        The structural angular deformation deflection angle strain response profile.

    Returns
    -------
    float
        The characteristic material Shear Rigidity Modulus indicator in Pascals (Pa).
    """
    if shear_strain == 0:
        raise ZeroDivisionError("Cannot compute Shear Modulus when shear strain is zero.")
    return shear_stress / shear_strain


@validate(delta_pressure=RULES.REAL, volumetric_strain=RULES.REAL)
def bulk_modulus(delta_pressure: float, volumetric_strain: float) -> float:
    """
    Calculate the intrinsic compression Bulk Modulus measuring volume elasticity resistances ($K = - \Delta P / (\Delta V / V_0)$).

    Project: Project Formulon-Physics

    Parameters
    ----------
    delta_pressure : float
        The uniform hydrostatic ambient pressure step increase parameter change in Pascals (Pa).
    volumetric_strain : float
        The dimensionless compression factor representing fractional volume structural change ($\Delta V / V_0$).

    Returns
    -------
    float
        The bulk structural resistance elasticity modulus factor in Pascals (Pa).
    """
    if volumetric_strain == 0:
        raise ZeroDivisionError("Cannot compute Bulk Modulus when volumetric strain is zero.")
    return -delta_pressure / volumetric_strain


# ══════════════════════════════════════════════════════════════════════════════
# 4. SURFACE PHENOMENA & INTERFACES
# ══════════════════════════════════════════════════════════════════════════════

@validate(force=RULES.FORCE, perimeter_length=RULES.DISTANCE)
def surface_tension(force: float, perimeter_length: float) -> float:
    """
    Determine the acting boundary layer liquid surface tension property ($\gamma = F / L$).

    Project: Project Formulon-Physics

    Parameters
    ----------
    force : float
        The active contractive parallel boundary layer force pull in Newtons (N).
    perimeter_length : float
        The continuous perimeter edge interface line boundary layer tracing distance in meters (m).

    Returns
    -------
    float
        The evaluated film surface tension index scale in Newtons per meter (N/m).
    """
    return force / perimeter_length


@validate(
    surface_tension=RULES.POSITIVE, contact_angle_deg=RULES.ANGLE_DEG_90,
    tube_radius=RULES.RADIUS, fluid_density=RULES.DENSITY, g=RULES.GRAVITY
)
def capillary_rise(
    surface_tension: float, contact_angle_deg: float, tube_radius: float, fluid_density: float, g: float
) -> float:
    """
    Predict the static capillary equilibrium tube height rise or depression level.

    $h = \\frac{2 \gamma \cos(\\theta)}{\\rho g r}$

    Project: Project Formulon-Physics

    Parameters
    ----------
    surface_tension : float
        The continuous interfacial parameter constant of the working liquid phase in Newtons per meter (N/m).
    contact_angle_deg : float
        The contact wetting boundary layer envelope angle recorded in degrees (°). Bounded between [0, 90].
    tube_radius : float
        The interior core coordinate radius size metric of the capillary channel in meters (m).
    fluid_density : float
        The target mass density configuration parameter of the fluid matrix in kilograms per cubic meter (kg/m³).
    g : float
        The acceleration context generated due to the gravitational background field in meters per second squared (m/s²).

    Returns
    -------
    float
        The equilibrium capillary meniscus baseline vertical displacement coordinate step height in meters (m).
    """
    angle_rad = math.radians(contact_angle_deg)
    numerator = 2.0 * surface_tension * math.cos(angle_rad)
    denominator = fluid_density * g * tube_radius
    return numerator / denominator


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRANSPORT PHENOMENA & VISCOUS DRAG
# ══════════════════════════════════════════════════════════════════════════════

@validate(viscosity=RULES.VISCOSITY, radius=RULES.RADIUS, terminal_velocity=RULES.SPEED)
def stokes_law(viscosity: float, radius: float, terminal_velocity: float) -> float:
    """
    Calculate the viscous drag resistance profile acting over an isolated sphere via Stokes' Law ($F_d = 6 \pi \mu r v$).

    Valid inside laminar regimes with very low particle Reynolds limits ($Re \ll 1$).

    Project: Project Formulon-Physics

    Parameters
    ----------
    viscosity : float
        The continuous baseline ambient medium dynamic fluid molecular viscosity in Pascal-seconds (Pa·s).
    radius : float
        The exact physical sphere radius boundary metric tracking size properties in meters (m).
    terminal_velocity : float
        The steady relative translation velocity displacement speed matching flow vectors in meters per second (m/s).

    Returns
    -------
    float
        The total resistive viscous fluid friction drag force metric output in Newtons (N).
    """
    return 6.0 * math.pi * viscosity * radius * terminal_velocity


@validate(
    pressure_drop=RULES.REAL, pipe_radius=RULES.RADIUS,
    pipe_length=RULES.DISTANCE, fluid_viscosity=RULES.VISCOSITY
)
def poiseuilles_law(
    pressure_drop: float, pipe_radius: float, pipe_length: float, fluid_viscosity: float
) -> float:
    """
    Calculate the laminar volumetric flow rate inside a cylindrical pipe channel via Poiseuille's Law.

    $Q = \\frac{\pi \Delta P r^4}{8 \mu L}$

    Project: Project Formulon-Physics

    Parameters
    ----------
    pressure_drop : float
        The total linear static pressure difference drop across the pipe channel bounds ($\Delta P$) in Pascals (Pa).
    pipe_radius : float
        The internal radius geometric boundary profile of the channel conduit line in meters (m).
    pipe_length : float
        The target linear segment travel spatial separation length of the pipe conduit line in meters (m).
    fluid_viscosity : float
        The intrinsic medium dynamic shear molecular viscosity constant parameter in Pascal-seconds (Pa·s).

    Returns
    -------
    float
        The continuous volumetric stream movement tracking flow rate scale ($Q$) in cubic meters per second (m³/s).
    """
    numerator = math.pi * pressure_drop * (pipe_radius**4)
    denominator = 8.0 * fluid_viscosity * pipe_length
    return numerator / denominator