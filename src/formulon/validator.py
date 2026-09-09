"""Validation primitives for Formulon Physics.

The validator distinguishes mathematical/numerical constraints from optional
physical assumptions.  Bounds are deliberately conservative: they reject
undefined mathematical inputs (zero denominators, negative absolute
quantities, invalid angles) without pretending that textbook ranges are
universal laws.  Dimensional/unit checking is provided separately by
:mod:`formulon.units` using Pint.
"""
from __future__ import annotations
import functools, inspect, math
from typing import Any, Callable
import numpy as np

class PhysicalBounds:
    """Declarative numeric domain rule.

    Parameters are inclusive unless ``strict_min``/``strict_max`` is set.
    ``None`` means no bound.  Bounds are model assumptions, not universal
    statements about every physical system.
    """
    def __init__(self, min_val=None, max_val=None, unit="", reason="", *, strict_min=False, strict_max=False):
        self.min_val=min_val; self.max_val=max_val; self.unit=unit; self.reason=reason
        self.strict_min=strict_min; self.strict_max=strict_max
    def check(self, value: Any, name: str) -> None:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name!r} must be numeric, not boolean")
        if isinstance(value, np.ndarray):
            if value.ndim != 0:
                raise TypeError(f"{name!r} expects a scalar; use a vectorized API for arrays")
            value=value.item()
        if not isinstance(value, (int,float,np.integer,np.floating)):
            raise TypeError(f"{name!r} must be a real scalar, got {type(value).__name__}")
        value=float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name!r} must be finite")
        if self.min_val is not None and ((value <= self.min_val) if self.strict_min else (value < self.min_val)):
            op='>' if self.strict_min else '>='
            raise ValueError(f"{name}={value} violates lower bound {op} {self.min_val}. {self.reason}")
        if self.max_val is not None and ((value >= self.max_val) if self.strict_max else (value > self.max_val)):
            op='<' if self.strict_max else '<='
            raise ValueError(f"{name}={value} violates upper bound {op} {self.max_val}. {self.reason}")

class PhysicsCatalog:
    """Shared domain rules used by the formula modules.

    The rules are intentionally about mathematical validity and common model
    assumptions.  They do not impose arbitrary cosmic limits on classical
    quantities such as velocity or gravity.
    """
    C_LIGHT=299_792_458.0
    REAL=PhysicalBounds()
    POSITIVE=PhysicalBounds(min_val=0.0, strict_min=True, reason='must be strictly positive')
    NON_NEGATIVE=PhysicalBounds(min_val=0.0, reason='must be non-negative')
    NEGATIVE=PhysicalBounds(max_val=0.0, strict_max=True, reason='must be strictly negative')
    FRACTION=PhysicalBounds(min_val=0.0,max_val=1.0,reason='dimensionless fraction')
    NON_NEG_TIME=PhysicalBounds(min_val=0.0,unit='s')
    POSITIVE_TIME=PhysicalBounds(min_val=0.0,unit='s',strict_min=True)
    PERIOD=PhysicalBounds(min_val=0.0,unit='s',strict_min=True)
    MASS=PhysicalBounds(min_val=0.0,unit='kg',strict_min=True)
    VELOCITY=PhysicalBounds(unit='m/s')
    SPEED=PhysicalBounds(min_val=0.0,unit='m/s')
    RELATIVISTIC_SPEED=PhysicalBounds(min_val=0.0,max_val=C_LIGHT,unit='m/s',strict_max=True)
    ACCELERATION=PhysicalBounds(unit='m/s^2')
    FORCE=PhysicalBounds(unit='N')
    DISPLACEMENT=PhysicalBounds(unit='m')
    DISTANCE=PhysicalBounds(min_val=0.0,unit='m',strict_min=True)
    MOMENTUM=PhysicalBounds(unit='kg m/s')
    IMPULSE=PhysicalBounds(unit='N s')
    COEFF_FRICTION=PhysicalBounds(min_val=0.0,reason='coefficient of friction cannot be negative')
    HEIGHT=PhysicalBounds(min_val=0.0,unit='m')
    NORMAL_FORCE=PhysicalBounds(min_val=0.0,unit='N')
    APPARENT_WEIGHT=PhysicalBounds(min_val=0.0,unit='N')
    CM_COORD=PhysicalBounds(unit='m')
    GRAVITY=PhysicalBounds(min_val=0.0,unit='m/s^2')
    ANGLE_DEG=PhysicalBounds(min_val=0.0,max_val=360.0,unit='deg')
    ANGLE_DEG_90=PhysicalBounds(min_val=0.0,max_val=90.0,unit='deg')
    ANGLE_DEG_180=PhysicalBounds(min_val=0.0,max_val=180.0,unit='deg')
    ANGULAR_FREQUENCY=PhysicalBounds(min_val=0.0,unit='rad/s')
    CROSS_ANGLE_DEG=PhysicalBounds(min_val=0.0,max_val=180.0,unit='deg')
    ANGLE_RAD=PhysicalBounds(unit='rad')
    INCIDENCE_ANGLE=PhysicalBounds(min_val=0.0,max_val=90.0,unit='deg')
    WORK_ANGLE_DEG=PhysicalBounds(min_val=0.0,max_val=180.0,unit='deg')
    RADIUS=PhysicalBounds(min_val=0.0,unit='m',strict_min=True)
    ANGULAR_VEL=PhysicalBounds(unit='rad/s')
    ANGULAR_ACC=PhysicalBounds(unit='rad/s^2')
    MOMENT_OF_INERTIA=PhysicalBounds(min_val=0.0,unit='kg m^2')
    TORQUE=PhysicalBounds(unit='N m')
    ANGULAR_MOMENTUM=PhysicalBounds(unit='kg m^2/s')
    PARALLEL_AXIS_D=PhysicalBounds(min_val=0.0,unit='m')
    SPRING_CONST=PhysicalBounds(min_val=0.0,unit='N/m',strict_min=True)
    SPRING_DISPLACEMENT=PhysicalBounds(unit='m')
    ENERGY=PhysicalBounds(unit='J')
    KINETIC_ENERGY=PhysicalBounds(min_val=0.0,unit='J')
    POTENTIAL_ENERGY=PhysicalBounds(unit='J')
    WORK=PhysicalBounds(unit='J')
    WORK_IO=PhysicalBounds(min_val=0.0,unit='J',strict_min=True)
    POWER=PhysicalBounds(unit='W')
    EFFICIENCY=PhysicalBounds(min_val=0.0,max_val=100.0,unit='%')
    FREQUENCY=PhysicalBounds(min_val=0.0,unit='Hz')
    WAVELENGTH=PhysicalBounds(min_val=0.0,unit='m',strict_min=True)
    WAVE_SPEED=PhysicalBounds(min_val=0.0,unit='m/s',strict_min=True)
    REFRACTIVE_INDEX=PhysicalBounds(min_val=0.0,reason='index must be non-negative; effective/metamaterial indices may be below 1')
    AMPLITUDE=PhysicalBounds(min_val=0.0)
    INTENSITY=PhysicalBounds(min_val=0.0,unit='W/m^2')
    WAVE_NUMBER=PhysicalBounds(min_val=0.0,unit='rad/m',strict_min=True)
    TEMPERATURE_K=PhysicalBounds(min_val=0.0,unit='K')
    TEMPERATURE_C=PhysicalBounds(min_val=-273.15,unit='degC')
    TEMPERATURE_F=PhysicalBounds(min_val=-459.67,unit='degF')
    DELTA_TEMP=PhysicalBounds(unit='K')
    PRESSURE=PhysicalBounds(min_val=0.0,unit='Pa',strict_min=True)
    VOLUME=PhysicalBounds(min_val=0.0,unit='m^3',strict_min=True)
    MOLES=PhysicalBounds(min_val=0.0,unit='mol',strict_min=True)
    HEAT=PhysicalBounds(unit='J')
    SPECIFIC_HEAT=PhysicalBounds(min_val=0.0,unit='J/(kg K)',strict_min=True)
    LATENT_HEAT=PhysicalBounds(min_val=0.0,unit='J/kg',strict_min=True)
    ENTROPY=PhysicalBounds(unit='J/K')
    THERMAL_EFFICIENCY=PhysicalBounds(min_val=0.0,max_val=1.0)
    CHARGE=PhysicalBounds(unit='C')
    ELECTRIC_FIELD=PhysicalBounds(min_val=0.0,unit='N/C')
    ELECTRIC_POTENTIAL=PhysicalBounds(unit='V')
    PERMITTIVITY=PhysicalBounds(min_val=0.0,unit='F/m',strict_min=True)
    MAGNETIC_FIELD=PhysicalBounds(min_val=0.0,unit='T')
    MAGNETIC_FLUX=PhysicalBounds(unit='Wb')
    RESISTANCE=PhysicalBounds(min_val=0.0,unit='ohm',strict_min=True)
    RESISTIVITY=PhysicalBounds(min_val=0.0,unit='ohm m',strict_min=True)
    VOLTAGE=PhysicalBounds(unit='V')
    CURRENT=PhysicalBounds(unit='A')
    CAPACITANCE=PhysicalBounds(min_val=0.0,unit='F',strict_min=True)
    INDUCTANCE=PhysicalBounds(min_val=0.0,unit='H',strict_min=True)
    CONDUCTIVITY=PhysicalBounds(min_val=0.0,unit='S/m')
    RELATIVE_PERMITTIVITY=PhysicalBounds(min_val=0.0)
    VISCOSITY=PhysicalBounds(min_val=0.0,unit='Pa s',strict_min=True)
    DENSITY=PhysicalBounds(min_val=0.0,unit='kg/m^3',strict_min=True)
    AREA=PhysicalBounds(min_val=0.0,unit='m^2',strict_min=True)
    DYNAMIC_PRESSURE=PhysicalBounds(min_val=0.0,unit='Pa')
    BULK_MODULUS=PhysicalBounds(min_val=0.0,unit='Pa',strict_min=True)
    MOLAR_MASS=PhysicalBounds(min_val=0.0,unit='kg/mol',strict_min=True)
    TEMPERATURE=PhysicalBounds(min_val=0.0,unit='K')
    ADIABATIC_INDEX=PhysicalBounds(min_val=1.0)
    GAS_CONSTANT=PhysicalBounds(min_val=0.0,unit='J/(mol K)',strict_min=True)
    LINEAR_DENSITY=PhysicalBounds(min_val=0.0,unit='kg/m',strict_min=True)
    NUMBER_DENSITY=PhysicalBounds(min_val=0.0,unit='1/m^3',strict_min=True)
    SOUND_INTENSITY=PhysicalBounds(min_val=0.0,unit='W/m^2')
    SOUND_INTENSITY_BASE=PhysicalBounds(min_val=0.0,unit='W/m^2',strict_min=True)
    THERMAL_CONDUCTIVITY=PhysicalBounds(min_val=0.0,unit='W/(m K)')
    EMISSIVITY=PhysicalBounds(min_val=0.0,max_val=1.0)
    DIELECTRIC_CONSTANT=PhysicalBounds(min_val=0.0)
    TURNS_PER_LENGTH=PhysicalBounds(min_val=0.0,unit='1/m')
    MICROSTATES=PhysicalBounds(min_val=1.0)
    DEGREES_FREEDOM=PhysicalBounds(min_val=1.0)

RULES=PhysicsCatalog()

def validate(**param_rules: PhysicalBounds) -> Callable:
    """Validate scalar inputs, then preserve the original scientific exception."""
    for key,rule in param_rules.items():
        if not isinstance(rule,PhysicalBounds): raise TypeError(f"Invalid validation rule for {key}")
    def decorator(func):
        sig=inspect.signature(func)
        for key in param_rules:
            if key not in sig.parameters: raise NameError(f"{func.__name__}: unknown parameter {key}")
        @functools.wraps(func)
        def wrapper(*args,**kwargs):
            bound=sig.bind(*args,**kwargs); bound.apply_defaults()
            for param,rule in param_rules.items(): rule.check(bound.arguments[param],param)
            result=func(*args,**kwargs)
            if isinstance(result,(int,float,np.integer,np.floating)) and not math.isfinite(float(result)):
                raise FloatingPointError(f"{func.__name__} returned a non-finite result")
            return result
        return wrapper
    return decorator

def validate_array_lengths(a,b,name_a='a',name_b='b'):
    """Require two sequences to have identical lengths."""
    if len(a)!=len(b): raise ValueError(f"{name_a} and {name_b} must have equal length")
def validate_strictly_increasing(arr,name='arr'):
    """Require a finite one-dimensional sequence to increase strictly."""
    x=np.asarray(arr,dtype=float)
    if x.ndim!=1 or x.size<2 or np.any(~np.isfinite(x)) or np.any(np.diff(x)<=0): raise ValueError(f"{name} must be finite, 1-D, and strictly increasing")
def validate_time_array(t,name='t'):
    """Validate a finite, strictly increasing one-dimensional time grid."""
    x=np.asarray(t,dtype=float)
    if x.ndim!=1 or x.size<2 or np.any(~np.isfinite(x)) or np.any(np.diff(x)<=0): raise ValueError(f"{name} must be a finite strictly increasing 1-D time grid")
    return x
def validate_positive_array(arr,name='arr'):
    """Require every array element to be finite and strictly positive."""
    x=np.asarray(arr,dtype=float)
    if np.any(~np.isfinite(x)) or np.any(x<=0): raise ValueError(f"{name} must contain finite positive values")
    return x
def validate_non_negative_array(arr,name='arr'):
    """Require every array element to be finite and non-negative."""
    x=np.asarray(arr,dtype=float)
    if np.any(~np.isfinite(x)) or np.any(x<0): raise ValueError(f"{name} must contain finite non-negative values")
    return x
def validate_same_shape(*arrays_names):
    """Require all named arrays to have identical shapes."""
    if not arrays_names: return
    shapes=[(np.asarray(a).shape,n) for a,n in arrays_names]
    if any(s!=shapes[0][0] for s,_ in shapes[1:]): raise ValueError('arrays must have identical shapes')
def validate_finite(value,name='value'):
    """Reject NaN and infinite scalar values."""
    if not math.isfinite(float(value)): raise ValueError(f"{name} must be finite")
def validate_cross_param_le(a,b,name_a='a',name_b='b'):
    """Require parameter ``a`` not to exceed comparison parameter ``b``."""
    if a>b: raise ValueError(f"{name_a} must be <= {name_b}")

__all__=['PhysicalBounds','PhysicsCatalog','RULES','validate','validate_array_lengths','validate_strictly_increasing','validate_time_array','validate_positive_array','validate_non_negative_array','validate_same_shape','validate_finite','validate_cross_param_le']
