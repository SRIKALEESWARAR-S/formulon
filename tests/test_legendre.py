"""spec: REQ-QC-008"""
from math import prod
import pytest
from formulon.quantum_computing import (legendre_features, measure_and_decode, legendre_feature_expectation,
                                        encode_legendre, decode_legendre, available_backends)


def test_features_endpoints_and_validation():
    # spec: REQ-QC-008
    f = legendre_features(-1.0, 4)
    assert f[0] == pytest.approx(1.0) and f[1] == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        legendre_features(2.0, 3)
    with pytest.raises(ValueError):
        legendre_features(0.0, 0)


def test_expectation_equals_product_of_modes_on_every_backend():
    # spec: REQ-QC-008
    for b in available_backends():
        for x in (-0.7, 0.0, 0.4):
            assert legendre_feature_expectation(x, 4, backend=b) == pytest.approx(prod(legendre_features(x, 4)), abs=1e-9)


def test_measure_and_decode_seeded_and_accurate():
    # spec: REQ-QC-008
    r = measure_and_decode(0.35, degree=6, shots=2000, seed=7, backend="numpy")
    assert abs(r.decoded_x - 0.35) < 0.08 and len(r.samples) == 2000
    again = measure_and_decode(0.35, degree=6, shots=2000, seed=7, backend="numpy")
    assert r.samples == again.samples
    for b in available_backends():                       # same seed => same samples on every backend
        assert measure_and_decode(0.35, degree=6, shots=500, seed=7, backend=b).z_expectations == pytest.approx(
            measure_and_decode(0.35, degree=6, shots=500, seed=7, backend="numpy").z_expectations)


def test_encode_state_and_decode_roundtrip():
    # spec: REQ-QC-008
    st = encode_legendre(0.3, 3, backend="numpy")
    assert st.shape == (8,) and abs((abs(st) ** 2).sum() - 1) < 1e-12
    assert abs(decode_legendre(legendre_features(0.3, 5)) - 0.3) < 0.01
    for b in available_backends():
        assert abs(encode_legendre(0.3, 3, backend=b)) .shape == (8,)
