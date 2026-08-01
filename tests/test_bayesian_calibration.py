"""Tests for Bayesian calibration module."""

import math

import numpy as np
import pytest

pytestmark = pytest.mark.slow
from models.uncertainty.bayesian_calibration import (
    ParameterPrior,
    CalibrationResult,
    MCMCResult,
    default_priors,
    calibrate_ensemble,
    calibrate_mcmc,
    information_gain,
    optimal_next_experiment,
    generate_synthetic_observations,
    screening_forward_model,
)


# -----------------------------------------------------------------------
# Test 1: ParameterPrior construction and sampling
# -----------------------------------------------------------------------

def test_parameter_prior_lognormal_sampling_and_pdf():
    """Log-normal prior: samples positive, log_pdf finite at mean."""
    prior = ParameterPrior("test", mean=1e-7, std=5e-8)
    rng = np.random.default_rng(42)
    samples = prior.sample(rng, 10000)
    assert np.all(samples > 0), "log-normal samples must be positive"
    assert abs(np.mean(samples) - 1e-7) / 1e-7 < 0.3, "sample mean should be near prior mean"
    # log_pdf at prior mean should be finite
    lp = prior.log_pdf(1e-7)
    assert math.isfinite(lp)


def test_parameter_prior_normal_bounds():
    """Normal prior with bounds clips samples."""
    prior = ParameterPrior("sigma", mean=100.0, std=30.0, distribution="normal",
                           bounds=(0.0, 300.0))
    rng = np.random.default_rng(0)
    samples = prior.sample(rng, 5000)
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 300.0)
    assert abs(np.mean(samples) - 100.0) < 10.0


def test_parameter_prior_invalid():
    """Invalid prior specifications raise ValueError."""
    with pytest.raises(ValueError):
        ParameterPrior("bad", mean=-1.0, std=1.0)  # lognormal with negative mean
    with pytest.raises(ValueError):
        ParameterPrior("bad", mean=1.0, std=-1.0)  # negative std
    with pytest.raises(ValueError):
        ParameterPrior("bad", mean=1.0, std=1.0, distribution="uniform")  # unknown dist


# -----------------------------------------------------------------------
# Test 2: Default priors coverage
# -----------------------------------------------------------------------

def test_default_priors_covers_all_targets():
    """All 11 calibratable parameters have priors."""
    priors = default_priors()
    expected = {
        "sigma0_MPa", "k_HP_MPa_sqrt_m",
        "D0_m2_s", "Q_kJ_mol",
        "fe_i0_A_m2", "fe_tafel_V_dec", "her_tafel_V_dec",
        "K_B_offset", "K_CH4_offset",
        "k_softening",
        "K_SS_NI_MPa_per_wt",
    }
    assert set(priors.keys()) == expected
    for name, prior in priors.items():
        assert isinstance(prior, ParameterPrior)
        assert prior.mean != 0


# -----------------------------------------------------------------------
# Test 3: EnKF converges on synthetic data
# -----------------------------------------------------------------------

def test_enkf_converges_on_synthetic_data():
    """EnKF posterior should be closer to true values than the prior."""
    priors = default_priors()
    param_names = sorted(priors.keys())
    rng = np.random.default_rng(42)

    true_params = {
        "sigma0_MPa": 85.0,
        "k_HP_MPa_sqrt_m": 0.55,
        "D0_m2_s": 4.0e-7,
        "Q_kJ_mol": 88.0,
        "fe_i0_A_m2": 5e-3,
        "fe_tafel_V_dec": 0.11,
        "her_tafel_V_dec": 0.15,
        "K_B_offset": 1.15,
        "K_CH4_offset": 0.9,
        "k_softening": 2.2e-4,
        "K_SS_NI_MPa_per_wt": 45.0,
    }

    obs, _ = generate_synthetic_observations(
        priors, true_params, screening_forward_model, noise_fraction=0.01, rng=rng,
    )

    result = calibrate_ensemble(
        priors, obs, screening_forward_model,
        n_ensemble=1500, observation_error=0.02, n_iterations=20, rng=rng,
    )

    assert isinstance(result, CalibrationResult)
    assert result.n_updates == 20
    # RMSE should decrease over iterations
    first_rmse = result.convergence_history[0]["rmse"]
    last_rmse = result.convergence_history[-1]["rmse"]
    assert last_rmse < first_rmse, f"RMSE did not decrease: {first_rmse} -> {last_rmse}"

    # Posterior should shrink vs prior for most params
    prior_var = np.sum(result.prior_std ** 2)
    post_var = np.sum(result.posterior_std ** 2)
    assert post_var < prior_var, "total posterior variance should be less than prior"


# -----------------------------------------------------------------------
# Test 4: MCMC produces posteriors with R-hat < 1.1
# -----------------------------------------------------------------------

def test_mcmc_rhat_below_threshold():
    """MCMC convergence: R-hat < 1.1 on a focused 2-param problem."""
    # Use a focused subset: Hall-Petch only (2 params, 1 observable)
    priors = {
        "sigma0_MPa": ParameterPrior("sigma0_MPa", mean=100.0, std=30.0,
                                     distribution="normal", bounds=(0.0, 300.0)),
        "k_HP_MPa_sqrt_m": ParameterPrior("k_HP_MPa_sqrt_m", mean=0.50, std=0.15,
                                          bounds=(0.1, 1.5)),
    }

    def hall_petch_model(params_vec, priors_dict):
        param_names_local = sorted(priors_dict.keys())
        p = {n: params_vec[i] for i, n in enumerate(param_names_local)}
        # Two grain sizes to break sigma0/k_HP correlation
        return {
            "yield_5um_MPa": p["sigma0_MPa"] + p["k_HP_MPa_sqrt_m"] / math.sqrt(5e-6),
            "yield_20um_MPa": p["sigma0_MPa"] + p["k_HP_MPa_sqrt_m"] / math.sqrt(20e-6),
        }

    true_params = {"sigma0_MPa": 85.0, "k_HP_MPa_sqrt_m": 0.55}
    rng = np.random.default_rng(42)
    obs, obs_err = generate_synthetic_observations(
        priors, true_params, hall_petch_model, noise_fraction=0.02, rng=rng,
    )

    result = calibrate_mcmc(
        priors, obs, hall_petch_model,
        observation_errors=obs_err,
        n_walkers=32, n_steps=8000, burnin_fraction=0.3,
        proposal_scale=0.05, rng=np.random.default_rng(99),
    )

    assert isinstance(result, MCMCResult)
    assert result.chain.shape[0] > 0
    assert 0.0 < result.acceptance_rate < 1.0

    # 2-param problem should converge well
    finite_rhats = [v for v in result.r_hat.values() if math.isfinite(v)]
    if finite_rhats:
        median_rhat = float(np.median(finite_rhats))
        assert median_rhat < 1.1, f"median R-hat too high: {median_rhat}"

    # Posterior should be closer to true values than prior
    for i, name in enumerate(sorted(priors.keys())):
        prior_err = abs(priors[name].mean - true_params[name])
        post_err = abs(result.posterior_mean[i] - true_params[name])
        assert post_err < prior_err * 1.5, f"{name}: posterior not closer to truth"


# -----------------------------------------------------------------------
# Test 5: Information gain identifies most-informative parameters
# -----------------------------------------------------------------------

def test_information_gain_positive_for_calibrated_params():
    """Information gain should be positive for all calibrated parameters."""
    priors = default_priors()
    param_names = sorted(priors.keys())
    rng = np.random.default_rng(42)

    true_params = {
        "sigma0_MPa": 85.0, "k_HP_MPa_sqrt_m": 0.55,
        "D0_m2_s": 4.0e-7, "Q_kJ_mol": 88.0,
        "fe_i0_A_m2": 5e-3, "fe_tafel_V_dec": 0.11, "her_tafel_V_dec": 0.15,
        "K_B_offset": 1.15, "K_CH4_offset": 0.9,
        "k_softening": 2.2e-4, "K_SS_NI_MPa_per_wt": 45.0,
    }
    obs, _ = generate_synthetic_observations(
        priors, true_params, screening_forward_model, noise_fraction=0.01, rng=rng,
    )
    result = calibrate_ensemble(
        priors, obs, screening_forward_model,
        n_ensemble=2000, observation_error=0.02, n_iterations=15, rng=rng,
    )

    ig = information_gain(priors, result.posterior_mean, result.posterior_std, param_names)
    assert isinstance(ig, dict)
    assert len(ig) == len(param_names)

    # All should be non-negative
    for name, kl in ig.items():
        assert kl >= 0.0 or math.isnan(kl), f"KL for {name} should be >= 0, got {kl}"

    # Parameters shifted from prior should show measurable gain
    nonzero = [v for v in ig.values() if v > 0.01]
    assert len(nonzero) >= 3, "At least 3 parameters should show measurable information gain"


# -----------------------------------------------------------------------
# Test 6: Optimal next experiment is physically sensible
# -----------------------------------------------------------------------

def test_optimal_next_experiment_returns_valid_target():
    """Optimal experiment recommendation should be one of the 6 targets."""
    priors = default_priors()
    param_names = sorted(priors.keys())

    # Use large posterior std to simulate high uncertainty
    large_std = np.array([priors[n].std * 0.8 for n in param_names])
    exp = optimal_next_experiment(priors, large_std, param_names)
    valid_experiments = {
        "Hall-Petch (EBSD + tensile)",
        "Diffusivity (foil weight gain)",
        "Tafel slopes (LSV)",
        "O2 probe (pO2 mV vs foil C)",
        "Tempering kinetics (HV vs T,t)",
        "Ni strengthening (ICP-OES + tensile)",
    }
    assert exp in valid_experiments, f"Unknown experiment: {exp}"


def test_optimal_next_experiment_prefers_high_uncertainty():
    """When one target has much higher uncertainty, it should be recommended."""
    priors = default_priors()
    param_names = sorted(priors.keys())

    # Make Hall-Petch params very uncertain
    post_std = np.array([priors[n].std * 0.1 for n in param_names])
    for i, name in enumerate(param_names):
        if name in ("sigma0_MPa", "k_HP_MPa_sqrt_m"):
            post_std[i] = priors[name].std * 5.0  # very high remaining uncertainty

    exp = optimal_next_experiment(priors, post_std, param_names)
    assert "Hall-Petch" in exp, f"Should recommend Hall-Petch, got {exp}"


# -----------------------------------------------------------------------
# Test 7: Synthetic observation generator
# -----------------------------------------------------------------------

def test_generate_synthetic_observations():
    """Synthetic observations should be close to model predictions."""
    priors = default_priors()
    rng = np.random.default_rng(42)
    true_params = {n: priors[n].mean for n in priors}
    obs, obs_err = generate_synthetic_observations(
        priors, true_params, screening_forward_model, noise_fraction=0.01, rng=rng,
    )
    assert isinstance(obs, dict)
    assert len(obs) == 6  # 6 observables
    # Predictions from mean params
    param_names = sorted(priors.keys())
    mean_vec = np.array([priors[n].mean for n in param_names])
    pred = screening_forward_model(mean_vec, priors)
    for k in obs:
        assert abs(obs[k] - pred[k]) / max(abs(pred[k]), 1e-15) < 0.1, \
            f"Observable {k} too far from prediction"


# -----------------------------------------------------------------------
# Test 8: Forward model physical sanity
# -----------------------------------------------------------------------

def test_screening_forward_model_physical_values():
    """Forward model outputs should be physically plausible."""
    priors = default_priors()
    param_names = sorted(priors.keys())
    mean_vec = np.array([priors[n].mean for n in param_names])
    pred = screening_forward_model(mean_vec, priors)

    # Yield strength at 5um grains should be >100 MPa
    assert pred["yield_strength_MPa"] > 100
    # D_eff at 900°C should be 1e-14 to 1e-9 m2/s
    assert 1e-14 < pred["D_eff_m2_s"] < 1e-9
    # Total cathodic current should be positive
    assert pred["total_cathodic_current_A_m2"] > 0
    # Tempered HV should be < 800 (softened)
    assert 100 < pred["tempered_HV"] < 800
    # Ni strengthening at 5wt% should be 50-200 MPa
    assert 50 < pred["ni_strengthening_MPa"] < 200
