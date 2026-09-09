"""Optional unit-aware helpers powered by Pint.

Formulon formula functions accept plain numeric SI values for speed and
simplicity.  This module provides explicit unit conversion and dimensionality
checking when engineering workflows need units attached to values.
"""
from __future__ import annotations
try:
    from pint import UnitRegistry
except ImportError as exc:  # pragma: no cover
    raise ImportError('Install Pint with: pip install "formulon-physics[units]"') from exc

ureg=UnitRegistry()
Q_=ureg.Quantity

def quantity(value, unit):
    """Create a Pint quantity, e.g. ``quantity(9.81, 'm/s^2')``."""
    return Q_(value, unit)

def to_si(value):
    """Convert a Pint quantity to SI base units."""
    return value.to_base_units()

def require_dimension(value, dimension, name='value'):
    """Require a quantity to have a specified Pint dimensionality."""
    if not hasattr(value,'dimensionality'):
        raise TypeError(f'{name} must be a Pint Quantity')
    expected=ureg.get_dimensionality(dimension)
    if value.dimensionality != expected:
        raise ValueError(f'{name} has dimensionality {value.dimensionality}, expected {expected}')
    return value

__all__=['ureg','Q_','quantity','to_si','require_dimension']
