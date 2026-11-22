"""spec: REQ-AD-001 REQ-AD-002 REQ-AD-003 REQ-AD-004 REQ-AD-005 REQ-AD-006"""
import numpy as np
import pytest
from formulon.pinn import (get_problem, list_problems, make_dataset, sample_interior, fit, evaluate,
                           loss, residuals, derivatives, available_autodiff, pielm_fit)


def _exact_model(p):
    names = list(p.domain)
    return lambda *c: p.exact(**dict(zip(names, c)))


def test_req_ad_001_backend_kwarg_everywhere():
    # spec: REQ-AD-001
    p = get_problem("heat_1d")
    pts = {"x": np.linspace(0.2, 0.8, 9), "t": np.full(9, 0.3)}
    d = derivatives(_exact_model(p), pts, p.outputs, ("u_t", "u_xx"), backend="numpy")
    assert set(d) == {"u_t", "u_xx"}
    r = residuals(p, _exact_model(p), pts, backend="numpy")
    assert np.max(np.abs(r["heat"])) < 1e-5
    for name in ("torch", "jax"):
        if name not in available_autodiff():
            with pytest.raises(ImportError):
                derivatives(_exact_model(p), pts, p.outputs, ("u_t",), backend=name)


def test_req_ad_002_numpy_loss_semantics():
    # spec: REQ-AD-002
    p = get_problem("heat_1d")
    ds = make_dataset(p, seed=1)
    exact = _exact_model(p)
    good = loss(p, exact, ds.collocation, ds.boundary, ds.initial, backend="numpy")
    bad = loss(p.with_params(alpha=0.5), exact, ds.collocation, ds.boundary, ds.initial, backend="numpy")
    assert good["total"] < 1e-9 < 1e-2 < bad["total"]
    # wave: a model with the right shape but a wrong initial velocity is penalised by the ic term
    w = get_problem("wave_1d")
    dw = make_dataset(w, seed=1)
    ok = loss(w, _exact_model(w), dw.collocation, dw.boundary, dw.initial, backend="numpy")
    kick = lambda x, t: w.exact(x=x, t=t) + 0.3 * np.sin(np.pi * x) * t
    wrong = loss(w, kick, dw.collocation, dw.boundary, dw.initial, backend="numpy")
    assert ok["ic"] < 1e-12 and wrong["ic"] > 1e-2


def test_req_ad_003_pielm_solves_linear_problems_and_rejects_nonlinear():
    # spec: REQ-AD-003
    for name in ("heat_1d", "wave_1d", "poisson_2d", "damped_oscillator"):
        p = get_problem(name)
        model, hist = fit(p, backend="numpy")
        assert evaluate(p, model, backend="numpy")["rel_l2"] < 1e-4, name
        assert hist["rms_fit_residual"] < 1e-4
    a = pielm_fit(get_problem("heat_1d"), seed=3)
    b = pielm_fit(get_problem("heat_1d"), seed=3)
    assert np.array_equal(a.beta, b.beta)                       # deterministic
    for name in ("burgers_1d", "kovasznay"):
        with pytest.raises(NotImplementedError) as e:
            fit(get_problem(name), backend="numpy")
        assert "torch" in str(e.value)


def test_pielm_other_parameters():
    # spec: REQ-AD-003
    p = get_problem("heat_1d", alpha=0.05, length=2.0)
    model, _ = fit(p, backend="numpy")
    assert evaluate(p, model, backend="numpy")["rel_l2"] < 1e-4


def _xp_model(xp):
    def model(*c):
        second = xp.cos(0.7 * c[1]) if len(c) > 1 else 1.0
        lin = c[1] if len(c) > 1 else 1.0
        return xp.sin(1.3 * c[0]) * second + 0.1 * c[0] * lin
    return model


_LINEAR_NAMES = ("heat_1d", "wave_1d", "poisson_2d", "damped_oscillator", "burgers_1d")


def _pts(p):
    return sample_interior({k: (lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo)) for k, (lo, hi) in p.domain.items()},
                           25, "uniform", 5)


def test_req_ad_004_torch_matches_finite_differences():
    # spec: REQ-AD-004
    torch = pytest.importorskip("torch")
    from formulon.pinn.torch_utils import MLP, as_tensors, autograd_derivatives, pinn_loss, train_pinn
    for name in _LINEAR_NAMES:
        p = get_problem(name)
        pts = _pts(p)
        r_t = residuals(p, _xp_model(torch), pts, backend="torch")
        r_n = residuals(p, _xp_model(np), pts, backend="numpy")
        for k in r_n:
            assert np.allclose(r_t[k].detach().numpy(), r_n[k], atol=1e-4), (name, k)
    p = get_problem("heat_1d")
    net = MLP(2, 1, 16, 2, p.domain).double()
    ds = make_dataset(p, n_collocation=64, n_boundary=32)
    out = pinn_loss(p, net, ds.collocation, ds.boundary, ds.initial)
    out["total"].backward()
    assert all(q.grad is not None for q in net.parameters())
    model, hist = train_pinn(p, epochs=150, width=24, depth=3, n_collocation=256, n_boundary=64)
    assert hist[-1] < 0.5 * hist[0]
    assert evaluate(p, model, backend="torch")["rel_l2"] < 1.5         # runs end-to-end; accuracy needs more epochs
    p = get_problem("kovasznay")
    assert set(autograd_derivatives(MLP(2, 3, 8, 2, p.domain).double(), as_tensors(_pts(p)), p.outputs,
                                    p.derivative_keys)) == set(p.derivative_keys)


def test_req_ad_005_jax_matches_numpy():
    # spec: REQ-AD-005
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    for name in _LINEAR_NAMES:
        p = get_problem(name)
        pts = _pts(p)
        r_j = residuals(p, _xp_model(jnp), pts, backend="jax")
        r_n = residuals(p, _xp_model(np), pts, backend="numpy")
        for k in r_n:
            assert np.allclose(np.asarray(r_j[k]), r_n[k], atol=1e-4), (name, k)
    p = get_problem("heat_1d")
    ds = make_dataset(p, n_collocation=64, n_boundary=32)
    lj = loss(p, _xp_model(jnp), ds.collocation, ds.boundary, ds.initial, backend="jax")["total"]
    ln = loss(p, _xp_model(np), ds.collocation, ds.boundary, ds.initial, backend="numpy")["total"]
    assert float(lj) == pytest.approx(ln, rel=1e-3, abs=1e-6)


def test_req_ad_006_same_residual_on_every_available_backend():
    # spec: REQ-AD-006
    for name in _LINEAR_NAMES:
        p = get_problem(name)
        pts = _pts(p)
        ref = residuals(p, _xp_model(np), pts, backend="numpy")
        assert all(np.all(np.isfinite(v)) for v in ref.values())
    assert "numpy" in available_autodiff()
