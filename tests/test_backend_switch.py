"""spec: REQ-BE-001 REQ-BE-002 REQ-BE-003 REQ-BE-004 REQ-BE-005 REQ-BE-006 REQ-QC-001 REQ-QC-009"""
import os
import pytest
import formulon
from formulon._switch import BackendSwitch
from formulon.quantum_computing import (set_backend, get_backend, use_backend, available_backends,
                                        resolve_backend, Circuit)
from formulon.pinn import (set_autodiff, get_autodiff, use_autodiff, available_autodiff,
                           resolve_autodiff)


def _fake_switch():
    return BackendSwitch("demo", ("a", "b", "c"), "FORMULON_DEMO_BACKEND", ("a", "b", "c"),
                         {"a": "definitely_not_installed_pkg", "b": "also_missing_pkg", "c": None},
                         {"a": "pip install a"})


def test_req_be_001_priority_order():
    # spec: REQ-BE-001
    sw = _fake_switch()
    os.environ["FORMULON_DEMO_BACKEND"] = "b"
    try:
        assert sw.get() == "b"                      # env
        sw.set("c")
        assert sw.get() == "c"                      # global beats env
        with sw.use("auto"):
            assert sw.get() == "auto"               # scope beats global
            assert sw.resolve("c") == "c"           # per-call beats scope
        assert sw.get() == "c"
    finally:
        del os.environ["FORMULON_DEMO_BACKEND"]
        sw.reset()
    assert sw.get() == "auto"


def test_req_be_002_explicit_choice_never_falls_back():
    # spec: REQ-BE-002
    sw = _fake_switch()
    assert sw.resolve("auto") == "c"               # auto picks first installed
    with pytest.raises(ImportError) as e:
        sw.resolve("a")
    assert "pip install a" in str(e.value)


def test_req_be_003_unknown_name_rejected():
    # spec: REQ-BE-003
    sw = _fake_switch()
    with pytest.raises(ValueError):
        sw.set("zzz")
    with pytest.raises(ValueError):
        set_backend("cirq")
    with pytest.raises(ValueError):
        set_autodiff("tensorflow")


def test_req_be_004_configure_is_atomic():
    # spec: REQ-BE-004
    formulon.reset_backends()
    try:
        st = formulon.configure(quantum="numpy", autodiff="numpy", jit=False)
        assert st["quantum"]["preferred"] == "numpy" and st["autodiff"]["preferred"] == "numpy"
        with pytest.raises(ValueError):
            formulon.configure(quantum="auto", autodiff="bogus")
        assert get_backend() == "numpy"             # unchanged: validation ran before any mutation
        with pytest.raises(ValueError):
            formulon.configure(jit="maybe")
    finally:
        formulon.reset_backends()
    assert get_backend() == "auto" and get_autodiff() == "auto"


def test_req_be_005_status_report():
    # spec: REQ-BE-005
    st = formulon.backend_status()
    assert set(st) == {"quantum", "autodiff", "jit"}
    assert "numpy" in st["quantum"]["available"] and "numpy" in st["autodiff"]["available"]
    assert st["quantum"]["resolves_to"] in ("pennylane", "qiskit", "numpy")
    assert set(st["jit"]) == {"enabled", "numba_installed"}


def test_req_be_006_environment_defaults():
    # spec: REQ-BE-006
    formulon.reset_backends()
    os.environ["FORMULON_QUANTUM_BACKEND"] = "numpy"
    os.environ["FORMULON_AUTODIFF_BACKEND"] = "numpy"
    os.environ["FORMULON_JIT"] = "1"
    try:
        assert get_backend() == "numpy" and get_autodiff() == "numpy" and formulon.get_jit() is True
        os.environ["FORMULON_QUANTUM_BACKEND"] = "nonsense"
        with pytest.raises(ValueError):
            get_backend()
    finally:
        for k in ("FORMULON_QUANTUM_BACKEND", "FORMULON_AUTODIFF_BACKEND", "FORMULON_JIT"):
            os.environ.pop(k, None)
    assert formulon.get_jit() is False


def test_req_qc_001_resolution_for_quantum():
    # spec: REQ-QC-001
    assert "numpy" in available_backends()
    with use_backend("numpy"):
        assert Circuit(1).x(0).run().backend == "numpy"
    for name in ("pennylane", "qiskit"):
        if name not in available_backends():
            with pytest.raises(ImportError):
                resolve_backend(name)


def test_req_qc_009_quantum_package_imports_without_frameworks():
    # spec: REQ-QC-009
    import importlib
    import formulon.quantum_computing as qc
    importlib.reload(qc)
    assert hasattr(qc, "add") and hasattr(qc, "Circuit")


def test_autodiff_resolution_matches_switch_rules():
    # spec: REQ-BE-001 REQ-BE-002
    assert "numpy" in available_autodiff()
    with use_autodiff("numpy"):
        assert resolve_autodiff() == "numpy"
    for name in ("torch", "jax"):
        if name not in available_autodiff():
            with pytest.raises(ImportError):
                resolve_autodiff(name)
