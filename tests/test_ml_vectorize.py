"""spec: REQ-ML-001 REQ-ML-002 REQ-ML-003"""
import numpy as np
import pytest
import formulon
from formulon.ml import vectorize, FormulaFeatures, vectorized_catalog


def test_array_matches_scalar():
    f = vectorize(formulon.projectile_range)
    v0 = np.array([5.0, 10.0, 20.0])
    out = f(v0, 45, 9.81)
    ref = [formulon.projectile_range(float(v), 45, 9.81) for v in v0]
    assert np.allclose(out, ref)


def test_broadcast_2d():
    f = vectorize(formulon.projectile_range)
    out = f(np.linspace(1, 10, 4)[:, None], np.linspace(10, 80, 3)[None, :], 9.81)
    assert out.shape == (4, 3)


def test_invalid_raises_and_masks():
    with pytest.raises(ValueError):
        vectorize(formulon.projectile_range)(10, np.array([45, 95.0]), 9.81)
    g = vectorize(formulon.projectile_range, on_invalid="mask")
    out = g(10, np.array([45, 95.0]), 9.81)
    assert np.isfinite(out[0]) and np.isnan(out[1])
    assert g.report.n_invalid == 1


def test_rejects_non_formula():
    with pytest.raises(TypeError):
        vectorize(lambda x: x)


def test_catalog_nonempty():
    assert len(vectorized_catalog()) > 150


def test_formula_features():
    ff = FormulaFeatures({"re": ("reynolds_number", dict(density="rho", flow_speed="v",
                                                         characteristic_length="L", viscosity="mu"))})
    out = ff.fit_transform({"rho": [1000.0, -1.0], "v": [2.0, 2.0], "L": [0.05, 0.05], "mu": [1e-3, 1e-3]})
    assert out["re"][0] == pytest.approx(1e5)
    assert np.isnan(out["re"][1]) and ff.invalid_fraction_["re"] == 0.5
    assert ff.to_matrix(out).shape == (2, 1)
