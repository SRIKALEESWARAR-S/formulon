"""spec: REQ-PINN-001 REQ-PINN-002 REQ-PINN-003 REQ-PINN-004 REQ-PINN-005 REQ-PINN-006"""
import numpy as np
import pytest
from formulon.pinn import (get_problem, list_problems, verify_problem, make_dataset, relative_l2,
                           max_abs_error, parameter_error, energy_drift, residual_report,
                           heat_slab_from_material, sample_interior, sample_boundary)


@pytest.mark.parametrize("name", list_problems())
def test_exact_solution_satisfies_residual(name):
    assert verify_problem(get_problem(name))["passed"]


def test_wrong_physics_is_detected():
    p = get_problem("heat_1d", alpha=0.1)
    wrong = p.with_params(alpha=0.2)
    pts = {"x": np.linspace(0.1, 0.9, 40), "t": np.full(40, 0.3)}
    res = wrong.residual(lambda x, t: p.exact(x=x, t=t), pts)
    assert residual_report(res)["heat"]["rms"] > 0.1


def test_boundary_and_initial_conditions():
    p = get_problem("heat_1d")
    L = p.params["length"]
    assert np.allclose(p.exact(x=np.array([0.0, L]), t=np.array([0.3, 0.3])), 0, atol=1e-12)
    x = np.linspace(0, L, 5)
    assert np.allclose(p.exact(x=x, t=np.zeros(5)), np.sin(np.pi * x / L))


def test_kovasznay_is_divergence_free_and_multi_output():
    p = get_problem("kovasznay")
    assert p.outputs == ("u", "v", "p")
    assert verify_problem(p)["rms"]["continuity"] < 1e-4


def test_oscillator_energy_conserved_without_damping():
    p = get_problem("damped_oscillator", c=0.0)
    t = np.linspace(0, p.params["t_end"], 4001)
    x = p.exact(t=t)
    w0 = np.sqrt(p.params["k"] / p.params["m"])
    v = -p.params["x0"] * w0 * np.sin(w0 * t)          # analytic velocity
    e = 0.5 * p.params["m"] * v**2 + 0.5 * p.params["k"] * x**2
    assert energy_drift(e) < 1e-3


def test_oscillator_rejects_overdamped():
    with pytest.raises(ValueError):
        get_problem("damped_oscillator", c=10.0)


def test_unknown_problem_and_bad_params():
    with pytest.raises(KeyError):
        get_problem("nope")
    with pytest.raises(ValueError):
        get_problem("heat_1d", alpha=-1)
    with pytest.raises(ValueError):
        heat_slab_from_material(0, 1, 1, 1)


def test_material_diffusivity_copper():
    p = heat_slab_from_material(k=401, rho=8960, cp=385, length=0.01)
    assert p.params["alpha"] == pytest.approx(401 / (8960 * 385))


def test_sampling_reproducible_and_in_domain():
    dom = {"x": (0, 2), "t": (0, 1)}
    a = sample_interior(dom, 100, "sobol", seed=3)
    b = sample_interior(dom, 100, "sobol", seed=3)
    assert np.array_equal(a["x"], b["x"]) and len(a["x"]) == 100
    assert a["x"].min() >= 0 and a["x"].max() <= 2
    bd = sample_boundary(dom, 200, seed=1)
    assert np.all(np.isin(bd["x"], [0.0, 2.0]))


def test_dataset_noise_and_ode_has_no_boundary():
    ds = make_dataset(get_problem("damped_oscillator"), noise_std=0.05, seed=1)
    assert ds.boundary is None
    err = relative_l2(ds.observations["x"], ds.observations["x_clean"])
    assert 0.01 < err < 0.15
    clean = make_dataset(get_problem("heat_1d"), noise_std=0.0)
    assert max_abs_error(clean.observations["u"], clean.observations["u_clean"]) == 0.0


def test_metrics():
    assert relative_l2([1, 2, 3], [1, 2, 3]) == 0
    with pytest.raises(ValueError):
        relative_l2([1, 2], [1, 2, 3])
    assert parameter_error({"a": 1.1}, {"a": 1.0})["a"] == pytest.approx(0.1)


def test_req_pinn_003_conditions_targets():
    # spec: REQ-PINN-003
    w = get_problem("wave_1d")
    pts = {"x": np.array([0.3, 0.5]), "t": np.zeros(2)}
    ini = w.conditions("initial", pts)
    assert np.allclose(ini["u"], np.sin(np.pi * pts["x"])) and np.allclose(ini["u_t"], 0.0, atol=1e-8)
    assert set(w.conditions("boundary", pts)) == {"u"}
    assert "u_t" not in get_problem("heat_1d").conditions("initial", {"x": pts["x"], "t": pts["t"]})
    k = get_problem("kovasznay").conditions("boundary", {"x": pts["x"], "y": pts["x"]})
    assert set(k) == {"u", "v", "p"}
    with pytest.raises(ValueError):
        w.conditions("interior", pts)
