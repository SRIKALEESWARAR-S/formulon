import math
import formulon
from formulon.classicalmechanics import projectile_range, newtons_second_law
from formulon.thermodynamics import ideal_gas_pressure

def test_core_smoke():
    assert math.isclose(newtons_second_law(2,3),6)
    assert math.isclose(projectile_range(10,45,9.81), 10**2/9.81, rel_tol=1e-12)
    assert ideal_gas_pressure(1,300,0.024) > 0
    assert formulon.FORMULA_COUNT >= 200
