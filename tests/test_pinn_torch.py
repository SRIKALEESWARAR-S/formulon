"""spec: REQ-AD-004 (torch-specific extras live in test_autodiff.py; kept for the importorskip gate)"""
import pytest


def test_torch_module_imports_when_installed():
    # spec: REQ-AD-004
    pytest.importorskip("torch")
    import formulon.pinn.torch_utils as tu
    assert hasattr(tu, "train_pinn") and hasattr(tu, "MLP")
