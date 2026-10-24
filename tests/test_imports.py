from importlib.metadata import version

def test_core_import():
    import formulon

    assert formulon.__version__ == version("formulon-physics")
