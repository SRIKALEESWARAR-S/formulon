"""spec: REQ-JIT-001 REQ-JIT-002 REQ-JIT-003 REQ-JIT-004"""
import warnings
import numpy as np
import pytest
import formulon
from formulon import jit as fj
from formulon.mathematicalphysics import legendre_polynomial
from formulon.ml import vectorize


def test_default_is_off_and_results_identical():
    assert formulon.get_jit() is False
    assert legendre_polynomial(6, 0.3) == pytest.approx(legendre_polynomial(6, 0.3, jit=True))


def test_scope_and_global_switch():
    with formulon.jit_context(True):
        assert formulon.get_jit() is True
        with formulon.jit_context(False):
            assert formulon.get_jit() is False
    assert formulon.get_jit() is False
    formulon.set_jit(True)
    try:
        assert formulon.get_jit() is True
    finally:
        formulon.set_jit(None)
    with pytest.raises(ValueError):
        formulon.set_jit("yes")


def test_missing_numba_falls_back_with_warning():
    if formulon.numba_available():
        pytest.skip("numba installed")
    fj._warned.discard("nonumba")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = vectorize(formulon.projectile_range)(np.array([10.0, 20.0]), 45, 9.81, jit=True)
    assert np.allclose(out, [10.19367992, 40.77471967])
    assert any("numba" in str(x.message) for x in w)


def test_jit_wiring_with_fake_numba():
    """A fake numba proves the dispatch/fallback logic even where numba is absent."""
    import types
    calls = {"njit": 0, "vec": 0}
    fake = types.ModuleType("numba")
    fake.njit = lambda cache=False: (lambda f: (calls.__setitem__("njit", calls["njit"] + 1), f)[1])
    fake.vectorize = lambda nopython=True: (lambda f: (calls.__setitem__("vec", calls["vec"] + 1), np.vectorize(f))[1])
    orig = fj._get_numba
    fj._get_numba = lambda: fake
    try:
        @fj.lazy_njit
        def twice(x):
            return 2 * x

        assert twice(3, jit=False) == 6 and calls["njit"] == 0     # off -> never compiled
        assert twice(3, jit=True) == 6 and calls["njit"] == 1      # on  -> compiled once
        assert twice(4, jit=True) == 8 and calls["njit"] == 1      # cached
        out = vectorize(formulon.projectile_range)(np.array([10.0, 20.0]), 45, 9.81, jit=True)
        assert calls["vec"] == 1 and np.allclose(out, [10.19367992, 40.77471967])
    finally:
        fj._get_numba = orig


def test_real_numba_matches_python():
    pytest.importorskip("numba")
    f = vectorize(formulon.projectile_range)
    x = np.linspace(1, 50, 1000)
    assert np.allclose(f(x, 30.0, 9.81, jit=True), f(x, 30.0, 9.81, jit=False))
    assert legendre_polynomial(20, 0.2, jit=True) == pytest.approx(legendre_polynomial(20, 0.2))


def test_req_jit_002_environment_variable_and_priority():
    # spec: REQ-JIT-002
    import os
    formulon.set_jit(None)
    os.environ["FORMULON_JIT"] = "1"
    try:
        assert formulon.get_jit() is True
        formulon.set_jit(False)                          # global beats env
        assert formulon.get_jit() is False
        with formulon.jit_context(True):                 # scope beats global
            assert formulon.get_jit() is True
    finally:
        os.environ.pop("FORMULON_JIT", None)
        formulon.set_jit(None)
    assert formulon.get_jit() is False
