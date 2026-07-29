"""
Driver for Bayesian calibration of electrowinning screening parameters.

Produces 4 output plots:
  bayesian_prior_posterior.png  — prior vs posterior distributions per parameter
  bayesian_information_gain.png — KL divergence per parameter
  bayesian_convergence.png     — EnKF convergence + MCMC trace
  bayesian_next_experiment.png — remaining uncertainty by experiment target

Usage:
  python -m models.run_bayesian_calibration [--output-dir outputs/]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .uncertainty.bayesian_calibration import (
    ParameterPrior,
    default_priors,
    calibrate_ensemble,
    calibrate_mcmc,
    information_gain,
    optimal_next_experiment,
    screening_forward_model,
    generate_synthetic_observations,
)


def _plot_prior_posterior(priors, param_names, enkf_result, mcmc_result, ax):
    """Bar chart: prior mean±std vs EnKF posterior vs MCMC posterior."""
    n = len(param_names)
    x = np.arange(n)
    width = 0.25

    prior_means = [priors[name].mean for name in param_names]
    prior_stds = [priors[name].std for name in param_names]
    enkf_means = enkf_result.posterior_mean
    enkf_stds = enkf_result.posterior_std
    mcmc_means = mcmc_result.posterior_mean
    mcmc_stds = mcmc_result.posterior_std

    # Normalize to prior mean for visual comparison
    norm = np.array([max(abs(m), 1e-15) for m in prior_means])

    ax.bar(x - width, prior_means / norm, width, yerr=prior_stds / norm,
           label="Prior", color="steelblue", alpha=0.7, capsize=3)
    ax.bar(x, enkf_means / norm, width, yerr=enkf_stds / norm,
           label="EnKF posterior", color="coral", alpha=0.7, capsize=3)
    ax.bar(x + width, mcmc_means / norm, width, yerr=mcmc_stds / norm,
           label="MCMC posterior", color="seagreen", alpha=0.7, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in param_names], fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Normalized value (prior mean = 1)")
    ax.set_title("Prior vs Posterior Distributions")
    ax.legend(fontsize=8)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.3)


def _plot_information_gain(ig_dict, ax):
    """Bar chart of KL divergence per parameter."""
    names = sorted(ig_dict.keys())
    values = [ig_dict[n] for n in names]
    colors = plt.cm.YlOrRd(np.linspace(0.2, 0.9, len(names)))
    ax.barh(names, values, color=colors)
    ax.set_xlabel("KL divergence (nats)")
    ax.set_title("Information Gain per Parameter")
    ax.tick_params(axis="y", labelsize=7)


def _plot_convergence(enkf_result, mcmc_result, axes):
    """EnKF convergence and MCMC trace."""
    # EnKF convergence
    ax1 = axes[0]
    iters = [c["iteration"] for c in enkf_result.convergence_history]
    rmses = [c["rmse"] for c in enkf_result.convergence_history]
    ax1.plot(iters, rmses, "o-", color="coral", markersize=4)
    ax1.set_xlabel("EnKF iteration")
    ax1.set_ylabel("RMSE (model vs obs)")
    ax1.set_title("EnKF Convergence")
    ax1.grid(True, alpha=0.3)

    # MCMC trace (first 3 params, thin for visibility)
    ax2 = axes[1]
    chain = mcmc_result.chain
    thin = max(1, chain.shape[0] // 500)
    n_show = min(3, chain.shape[1])
    for i in range(n_show):
        ax2.plot(chain[::thin, i], alpha=0.7, label=mcmc_result.param_names[i], linewidth=0.5)
    ax2.set_xlabel("Step (post-burnin)")
    ax2.set_ylabel("Parameter value")
    ax2.set_title("MCMC Trace")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)


def _plot_next_experiment(priors, posterior_std, param_names, ax):
    """Horizontal bars showing remaining normalized uncertainty per experiment target."""
    experiment_map = {
        "Hall-Petch": ["sigma0_MPa", "k_HP_MPa_sqrt_m"],
        "Diffusivity": ["D0_m2_s", "Q_kJ_mol"],
        "Tafel slopes": ["fe_i0_A_m2", "fe_tafel_V_dec", "her_tafel_V_dec"],
        "O2 probe": ["K_B_offset", "K_CH4_offset"],
        "Tempering": ["k_softening"],
        "Ni strengthening": ["K_SS_NI_MPa_per_wt"],
    }
    param_to_idx = {name: i for i, name in enumerate(param_names)}
    exp_names = []
    uncertainties = []
    for exp_name, params in experiment_map.items():
        total = 0.0
        count = 0
        for p in params:
            if p in param_to_idx:
                idx = param_to_idx[p]
                total += posterior_std[idx] / max(abs(priors[p].mean), 1e-15)
                count += 1
        exp_names.append(exp_name)
        uncertainties.append(total / max(count, 1))

    # Sort by uncertainty descending (most uncertain first = most informative)
    order = np.argsort(uncertainties)[::-1]
    exp_names = [exp_names[i] for i in order]
    uncertainties = [uncertainties[i] for i in order]
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(exp_names)))
    ax.barh(exp_names, uncertainties, color=colors)
    ax.set_xlabel("Normalized remaining uncertainty")
    ax.set_title("Optimal Next Experiment\n(higher = more informative)")
    ax.tick_params(axis="y", labelsize=8)


def run(output_dir: str = "outputs") -> dict:
    """Run full Bayesian calibration pipeline on synthetic data.

    Returns dict with all results and paths to generated plots.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Setup
    priors = default_priors()
    param_names = sorted(priors.keys())
    rng = np.random.default_rng(42)

    # True parameters (slightly shifted from prior means)
    true_params = {
        "sigma0_MPa": 85.0,         # prior 100
        "k_HP_MPa_sqrt_m": 0.55,    # prior 0.50
        "D0_m2_s": 4.0e-7,          # prior 6.2e-7
        "Q_kJ_mol": 88.0,           # prior 80
        "fe_i0_A_m2": 5e-3,         # prior 1e-2
        "fe_tafel_V_dec": 0.11,     # prior 0.12
        "her_tafel_V_dec": 0.15,    # prior 0.14
        "K_B_offset": 1.15,         # prior 1.0
        "K_CH4_offset": 0.9,        # prior 1.0
        "k_softening": 2.2e-4,      # prior 1.8e-4
        "K_SS_NI_MPa_per_wt": 45.0, # prior 38
    }

    # Generate synthetic observations
    obs, obs_err = generate_synthetic_observations(
        priors, true_params, screening_forward_model, noise_fraction=0.02, rng=rng,
    )

    # 1. EnKF calibration
    enkf_result = calibrate_ensemble(
        priors, obs, screening_forward_model,
        n_ensemble=2000, observation_error=0.03, n_iterations=15, rng=rng,
    )

    # 2. MCMC calibration (with reduced steps for speed)
    mcmc_result = calibrate_mcmc(
        priors, obs, screening_forward_model,
        observation_errors=obs_err,
        n_walkers=24, n_steps=3000, burnin_fraction=0.3,
        proposal_scale=0.03, rng=np.random.default_rng(123),
    )

    # 3. Information gain (from EnKF posterior)
    ig = information_gain(priors, enkf_result.posterior_mean, enkf_result.posterior_std, param_names)

    # 4. Optimal next experiment
    next_exp = optimal_next_experiment(priors, enkf_result.posterior_std, param_names)

    # Generate plots
    # Plot 1: Prior vs Posterior
    fig, ax = plt.subplots(figsize=(14, 5))
    _plot_prior_posterior(priors, param_names, enkf_result, mcmc_result, ax)
    fig.tight_layout()
    fig.savefig(out / "bayesian_prior_posterior.png", dpi=150)
    plt.close(fig)

    # Plot 2: Information gain
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_information_gain(ig, ax)
    fig.tight_layout()
    fig.savefig(out / "bayesian_information_gain.png", dpi=150)
    plt.close(fig)

    # Plot 3: Convergence
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _plot_convergence(enkf_result, mcmc_result, axes)
    fig.tight_layout()
    fig.savefig(out / "bayesian_convergence.png", dpi=150)
    plt.close(fig)

    # Plot 4: Next experiment
    fig, ax = plt.subplots(figsize=(8, 4))
    _plot_next_experiment(priors, enkf_result.posterior_std, param_names, ax)
    fig.tight_layout()
    fig.savefig(out / "bayesian_next_experiment.png", dpi=150)
    plt.close(fig)

    # Summary
    summary = {
        "true_params": true_params,
        "observations": obs,
        "enkf": {
            "posterior_mean": {n: float(enkf_result.posterior_mean[i]) for i, n in enumerate(param_names)},
            "posterior_std": {n: float(enkf_result.posterior_std[i]) for i, n in enumerate(param_names)},
            "n_updates": enkf_result.n_updates,
            "final_rmse": enkf_result.convergence_history[-1]["rmse"],
        },
        "mcmc": {
            "posterior_mean": {n: float(mcmc_result.posterior_mean[i]) for i, n in enumerate(param_names)},
            "posterior_std": {n: float(mcmc_result.posterior_std[i]) for i, n in enumerate(param_names)},
            "r_hat": mcmc_result.r_hat,
            "acceptance_rate": mcmc_result.acceptance_rate,
        },
        "information_gain": ig,
        "optimal_next_experiment": next_exp,
        "plots": [
            str(out / "bayesian_prior_posterior.png"),
            str(out / "bayesian_information_gain.png"),
            str(out / "bayesian_convergence.png"),
            str(out / "bayesian_next_experiment.png"),
        ],
    }

    # Write summary JSON
    (out / "bayesian_calibration_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    return summary


def main():
    parser = argparse.ArgumentParser(description="Bayesian calibration of screening parameters")
    parser.add_argument("--output-dir", default="outputs", help="Directory for plots and JSON")
    args = parser.parse_args()
    summary = run(args.output_dir)
    print(f"Calibration complete. {len(summary['plots'])} plots written to {args.output_dir}/")
    print(f"  EnKF final RMSE: {summary['enkf']['final_rmse']:.4e}")
    print(f"  MCMC acceptance: {summary['mcmc']['acceptance_rate']:.2%}")
    max_rhat = max(v for v in summary['mcmc']['r_hat'].values() if not math.isnan(v))
    print(f"  MCMC max R-hat:  {max_rhat:.3f}")
    print(f"  Recommended next experiment: {summary['optimal_next_experiment']}")


if __name__ == "__main__":
    main()
