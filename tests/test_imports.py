import re


def test_core_import():
    import formulon
    from formulon.__about__ import __version__
    assert formulon.__version__ == __version__
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)


def test_subpackages_import_without_optional_deps():
    import formulon.ml, formulon.pinn  # noqa: F401  (NumPy/SciPy only)
