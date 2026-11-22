"""spec: REQ-PKG-001 REQ-PKG-002 REQ-PKG-003 REQ-PKG-004"""
import importlib
import importlib.util
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).parents[1]


def test_req_pkg_001_imports_without_optional_frameworks():
    # spec: REQ-PKG-001
    for mod in ("formulon", "formulon.ml", "formulon.pinn", "formulon.quantum_computing",
                "formulon.backends", "formulon.jit"):
        importlib.import_module(mod)
    for heavy in ("torch", "jax", "numba", "pennylane", "qiskit"):
        if importlib.util.find_spec(heavy) is None:
            assert heavy not in sys.modules


def test_req_pkg_002_license_is_apache_everywhere():
    # spec: REQ-PKG-002
    assert "Apache License" in (ROOT / "LICENSE").read_text()
    assert "Version 2.0" in (ROOT / "LICENSE").read_text()
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert 'license = { text = "Apache-2.0" }' in pyproject
    assert "License :: OSI Approved :: Apache Software License" in pyproject
    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "License: MIT" not in text and "SPDX-License-Identifier: MIT" not in text, path


def test_req_pkg_003_no_case_colliding_files():
    # spec: REQ-PKG-003
    seen = {}
    for p in (ROOT / "src").rglob("*"):
        key = str(p.relative_to(ROOT)).lower()
        assert key not in seen, f"case collision: {p} vs {seen[key]}"
        seen[key] = p


def test_req_pkg_004_single_version_source():
    # spec: REQ-PKG-004
    import formulon
    from formulon.__about__ import __version__
    assert formulon.__version__ == __version__ and re.fullmatch(r"\d+\.\d+\.\d+", __version__)
