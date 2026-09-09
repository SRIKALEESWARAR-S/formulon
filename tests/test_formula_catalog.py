from formulon.formula_catalog import FORMULA_COUNT, FORMULAS, get_formula

def test_formula_count():
    assert FORMULA_COUNT >= 200
    assert len(FORMULAS) == FORMULA_COUNT

def test_formula_names_unique():
    names=[x.name for x in FORMULAS]
    assert len(names)==len(set(names))

def test_catalog_lookup():
    item=get_formula('projectile_range')
    assert item.module=='classicalmechanics'
    assert item.use_case
