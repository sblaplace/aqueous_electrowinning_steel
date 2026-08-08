"""
Bayesian calibration for aqueous electrowinning screening parameters.

Provides two calibration engines:
* Ensemble Kalman Filter (EnKF) — fast sequential updating, good for online use.
* Markov-Chain Monte Carlo (MCMC) — Metropolis-Hastings, full posterior exploration.

Both take a ParameterPrior (screening literature values) and experimental
data/model predictions, and produce calibrated posteriors.

Calibration targets (ordered by information value):
  1. Hall-Petch (EBSD + tensile -> sigma0, k_HP)
  2. Diffusivity (foil weight gain -> D0, Q)
  3. Tafel slopes (LSV -> j0, beta_a, beta_c)
  4. O2 probe (pO2 mV vs foil C -> K_B, K_CH4)
  5. Tempering kinetics (HV vs T,t -> k_softening)
  6. Ni strengthening (ICP-OES + tensile -> K_SS_NI)

All distributions are log-normal for positive-definite physical quantities
(except sigma0 which uses a normal prior since it can be near zero).

References:
* Evensen, G. (2009). Data Assimilation — The Ensemble Kalman Filter.
* Foreman-Mackey et al. (2013). emcee: the MCMC hammer. PASP 125:306.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import math
import numpy as np


# ---------------------------------------------------------------------------
# Parameter prior specification
# ---------------------------------------------------------------------------

@dataclass
class ParameterPrior:
    """Prior distribution for a single parameter.

    For log-normal: mean and std are in *natural* (linear) space;
    internally stored as log-space mu, sigma.
    For normal: mean and std are in linear space.
    """

    name: str
    mean: float
    std: float
    distribution: str = "lognormal"  # "lognormal" or "normal"
    bounds: Optional[Tuple[float, float]] = None  # hard bounds for MCMC

    def __post_init__(self):
        if self.distribution == "lognormal":
            if self.mean <= 0:
                raise ValueError(f"lognormal prior '{self.name}' requires mean > 0")
            if self.std <= 0:
                raise ValueError(f"lognormal prior '{self.name}' requires std > 0")
            # Convert to log-space
            var = self.std ** 2
            mu2 = self.mean ** 2
            self._log_mu = math.log(mu2 / math.sqrt(var + mu2))
            self._log_sigma = math.sqrt(math.log(1.0 + var / mu2))
        elif self.distribution == "normal":
            if self.std <= 0:
                raise ValueError(f"normal prior '{self.name}' requires std > 0")
            self._log_mu = self.mean
            self._log_sigma = self.std
        else:
            raise ValueError(f"unknown distribution '{self.distribution}'")

    @property
    def log_mu(self) -> float:
        return self._log_mu

    @property
    def log_sigma(self) -> float:
        return self._log_sigma

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        """Draw n samples from the prior."""
        if self.distribution == "lognormal":
            samples = rng.lognormal(self._log_mu, self._log_sigma, n)
        else:
            samples = rng.normal(self._log_mu, self._log_sigma, n)
        if self.bounds is not None:
            samples = np.clip(samples, self.bounds[0], self.bounds[1])
        return samples

    def log_pdf(self, x: float) -> float:
        """Evaluate log-probability density at x."""
        if self.distribution == "lognormal":
            if x <= 0:
                return -math.inf
            logx = math.log(x)
            return (
                -0.5 * ((logx - self._log_mu) / self._log_sigma) ** 2
                - math.log(x * self._log_sigma * math.sqrt(2 * math.pi))
            )
        else:
            return (
                -0.5 * ((x - self._log_mu) / self._log_sigma) ** 2
                - math.log(self._log_sigma * math.sqrt(2 * math.pi))
            )


# ---------------------------------------------------------------------------
# Default screening priors for the 6 calibration targets
# ---------------------------------------------------------------------------

def default_priors() -> Dict[str, ParameterPrior]:
    """Return screening-level priors for all calibratable parameters.

    Values from literature means documented in the respective model modules.
    """
    return {
        # 1. Hall-Petch (mechanical_properties.py: SIGMA0=100, K_HP=0.50)
        "sigma0_MPa": ParameterPrior("sigma0_MPa", mean=100.0, std=30.0, distribution="normal",
                                     bounds=(0.0, 300.0)),
        "k_HP_MPa_sqrt_m": ParameterPrior("k_HP_MPa_sqrt_m", mean=0.50, std=0.15,
                                          bounds=(0.1, 1.5)),
        # 2. Diffusivity (carburization.py: D0_FERRITE=6.2e-7, Q_FERRITE=80 kJ/mol)
        "D0_m2_s": ParameterPrior("D0_m2_s", mean=6.2e-7, std=4.0e-7, bounds=(1e-9, 1e-5)),
        "Q_kJ_mol": ParameterPrior("Q_kJ_mol", mean=80.0, std=20.0, distribution="normal",
                                   bounds=(20.0, 200.0)),
        # 3. Tafel (kinetics.py: fe_i0 ~1e-2, fe_tafel ~0.12, her_tafel ~0.14)
        "fe_i0_A_m2": ParameterPrior("fe_i0_A_m2", mean=1e-2, std=5e-2, bounds=(1e-8, 1e2)),
        "fe_tafel_V_dec": ParameterPrior("fe_tafel_V_dec", mean=0.12, std=0.05,
                                         bounds=(0.02, 0.50)),
        "her_tafel_V_dec": ParameterPrior("her_tafel_V_dec", mean=0.14, std=0.05,
                                          bounds=(0.02, 0.50)),
        # 4. O2 probe / Boudouard (carbon_potential.py: K_B via dG=170700-174.5T)
        "K_B_offset": ParameterPrior("K_B_offset", mean=1.0, std=0.3, distribution="normal",
                                     bounds=(0.1, 5.0)),
        "K_CH4_offset": ParameterPrior("K_CH4_offset", mean=1.0, std=0.3, distribution="normal",
                                       bounds=(0.1, 5.0)),
        # 5. Tempering (tempering.py / foil_calibration.py: k ~0.00018)
        "k_softening": ParameterPrior("k_softening", mean=1.8e-4, std=1.0e-4,
                                      bounds=(1e-6, 1e-2)),
        # 6. Ni strengthening (mechanical_properties.py: K_SS_NI=38 MPa/wt%)
        "K_SS_NI_MPa_per_wt": ParameterPrior("K_SS_NI_MPa_per_wt", mean=38.0, std=15.0,
                                              distribution="normal", bounds=(5.0, 100.0)),
    }


# ---------------------------------------------------------------------------
# Calibration result containers
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """Result from Ensemble Kalman Filter calibration."""

    param_names: List[str]
    prior_mean: np.ndarray
    prior_std: np.ndarray
    posterior_mean: np.ndarray
    posterior_std: np.ndarray
    posterior_ensemble: np.ndarray  # (n_ensemble, n_params)
    n_updates: int
    convergence_history: List[Dict[str, float]]  # per-update RMSE


@dataclass
class MCMCResult:
    """Result from MCMC calibration."""

    param_names: List[str]
    chain: np.ndarray  # (n_steps, n_params)
    log_prob: np.ndarray  # (n_steps,)
    acceptance_rate: float
    r_hat: Dict[str, float]  # Gelman-Rubin per param
    burnin: int
    posterior_mean: np.ndarray
    posterior_std: np.ndarray
    n_walkers: int
    n_steps: int


# ---------------------------------------------------------------------------
# Ensemble Kalman Filter
# ---------------------------------------------------------------------------

def calibrate_ensemble(
    priors: Dict[str, ParameterPrior],
    observations: Dict[str, float],
    model_fn: Callable[[np.ndarray, Dict[str, ParameterPrior]], Dict[str, float]],
    n_ensemble: int = 1000,
    observation_error: float = 0.05,
    n_iterations: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> CalibrationResult:
    """Calibrate parameters using the Ensemble Kalman Filter (EnKF).

    Parameters
    ----------
    priors : dict
        Mapping of parameter name -> ParameterPrior.
    observations : dict
        Mapping of observable name -> measured value.
    model_fn : callable
        Forward model: takes (param_vector, priors_dict) and returns dict of
        predicted observables. The param_vector is ordered by sorted param names.
    n_ensemble : int
        Number of ensemble members (default 1000).
    observation_error : float
        Relative std for observation noise (fraction of observed value).
    n_iterations : int
        Number of EnKF update cycles (iterative EnKF).
    rng : numpy Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    CalibrationResult
    """
    if rng is None:
        rng = np.random.default_rng(42)

    param_names = sorted(priors.keys())
    n_params = len(param_names)

    # Build initial ensemble from priors
    ensemble = np.zeros((n_ensemble, n_params))
    for i, name in enumerate(param_names):
        ensemble[:, i] = priors[name].sample(rng, n_ensemble)

    prior_mean = np.mean(ensemble, axis=0)
    prior_std = np.std(ensemble, axis=0)

    obs_names = sorted(observations.keys())
    n_obs = len(obs_names)
    obs_values = np.array([observations[k] for k in obs_names])

    convergence = []

    for iteration in range(n_iterations):
        # Forward model for each ensemble member
        H_ensemble = np.zeros((n_ensemble, n_obs))
        for j in range(n_ensemble):
            pred = model_fn(ensemble[j, :], priors)
            for k, obs_name in enumerate(obs_names):
                H_ensemble[j, k] = pred.get(obs_name, 0.0)

        # Ensemble covariance
        X_mean = np.mean(ensemble, axis=0)
        H_mean = np.mean(H_ensemble, axis=0)
        X_dev = ensemble - X_mean  # (N, n_params)
        H_dev = H_ensemble - H_mean  # (N, n_obs)

        # Cross-covariance P_xy = X^T H / (N-1)
        P_xy = (X_dev.T @ H_dev) / (n_ensemble - 1)  # (n_params, n_obs)

        # Innovation covariance P_yy = H^T H / (N-1) + R
        P_yy = (H_dev.T @ H_dev) / (n_ensemble - 1)  # (n_obs, n_obs)
        R = np.diag((observation_error * np.maximum(np.abs(obs_values), 1e-10)) ** 2)
        P_yy += R

        # Kalman gain K = P_xy @ inv(P_yy)
        try:
            K = P_xy @ np.linalg.inv(P_yy)
        except np.linalg.LinAlgError:
            K = P_xy @ np.linalg.pinv(P_yy)

        # Perturbed observations
        obs_noise = rng.normal(0, observation_error * np.maximum(np.abs(obs_values), 1e-10),
                               (n_ensemble, n_obs))
        Y_obs = obs_values[np.newaxis, :] + obs_noise

        # Update ensemble
        innovation = Y_obs - H_ensemble  # (N, n_obs)
        ensemble = ensemble + (innovation @ K.T)

        # Enforce bounds
        for i, name in enumerate(param_names):
            if priors[name].bounds is not None:
                lo, hi = priors[name].bounds
                ensemble[:, i] = np.clip(ensemble[:, i], lo, hi)

        # Track convergence
        rmse = float(np.sqrt(np.mean((np.mean(H_ensemble, axis=0) - obs_values) ** 2)))
        convergence.append({"iteration": iteration, "rmse": rmse})

    posterior_mean = np.mean(ensemble, axis=0)
    posterior_std = np.std(ensemble, axis=0)

    return CalibrationResult(
        param_names=param_names,
        prior_mean=prior_mean,
        prior_std=prior_std,
        posterior_mean=posterior_mean,
        posterior_std=posterior_std,
        posterior_ensemble=ensemble,
        n_updates=n_iterations,
        convergence_history=convergence,
    )


# ---------------------------------------------------------------------------
# MCMC (Metropolis-Hastings)
# ---------------------------------------------------------------------------

def calibrate_mcmc(
    priors: Dict[str, ParameterPrior],
    observations: Dict[str, float],
    model_fn: Callable[[np.ndarray, Dict[str, ParameterPrior]], Dict[str, float]],
    observation_errors: Optional[Dict[str, float]] = None,
    n_walkers: int = 32,
    n_steps: int = 5000,
    burnin_fraction: float = 0.3,
    proposal_scale: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> MCMCResult:
    """Calibrate parameters using Metropolis-Hastings MCMC.

    Each walker runs an independent chain. The proposal is a Gaussian
    perturbation in log-space for lognormal params and linear-space for normal.
    Convergence diagnosed via Gelman-Rubin R-hat across walkers split in half.

    Parameters
    ----------
    priors : dict
        Mapping of parameter name -> ParameterPrior.
    observations : dict
        Observable name -> measured value.
    model_fn : callable
        Forward model: (param_vector, priors) -> dict of predicted observables.
    observation_errors : dict, optional
        Observable name -> absolute error. Defaults to 5% of observed value.
    n_walkers : int
        Number of independent MCMC walkers (default 32).
    n_steps : int
        Steps per walker (default 5000).
    burnin_fraction : float
        Fraction of chain to discard as burn-in (default 0.3).
    proposal_scale : float
        Scale for Gaussian proposal (fraction of prior sigma).
    rng : numpy Generator, optional
        Random number generator.

    Returns
    -------
    MCMCResult
    """
    if rng is None:
        rng = np.random.default_rng(42)

    param_names = sorted(priors.keys())
    n_params = len(param_names)
    obs_names = sorted(observations.keys())
    obs_values = np.array([observations[k] for k in obs_names])

    if observation_errors is None:
        observation_errors = {k: 0.05 * abs(v) for k, v in observations.items()}
    obs_sigma = np.array([max(observation_errors.get(k, 0.05 * abs(observations[k])), 1e-15)
                          for k in obs_names])

    def log_likelihood(params_vec: np.ndarray) -> float:
        """Gaussian log-likelihood."""
        try:
            pred = model_fn(params_vec, priors)
        except Exception:
            return -math.inf
        pred_vals = np.array([pred.get(k, 0.0) for k in obs_names])
        if not np.all(np.isfinite(pred_vals)):
            return -math.inf
        return float(-0.5 * np.sum(((pred_vals - obs_values) / obs_sigma) ** 2))

    def log_prior(params_vec: np.ndarray) -> float:
        """Sum of individual log-priors."""
        lp = 0.0
        for i, name in enumerate(param_names):
            lp += priors[name].log_pdf(params_vec[i])
        return lp

    def log_posterior(params_vec: np.ndarray) -> float:
        lp = log_prior(params_vec)
        if not np.isfinite(lp):
            return -math.inf
        ll = log_likelihood(params_vec)
        return lp + ll

    # Initialize walkers near prior mean
    init_ensemble = np.zeros((n_walkers, n_params))
    for i, name in enumerate(param_names):
        init_ensemble[:, i] = priors[name].sample(rng, n_walkers)

    # Pre-compute proposal scales (additive in parameter space, scaled by prior std)
    proposal_stds = np.array([proposal_scale * priors[n].std for n in param_names])

    # Run chains
    chain = np.zeros((n_steps, n_walkers, n_params))
    log_p = np.zeros((n_steps, n_walkers))
    chain[0] = init_ensemble
    log_p[0] = np.array([log_posterior(init_ensemble[w]) for w in range(n_walkers)])
    accepts = np.zeros(n_walkers)

    for step in range(1, n_steps):
        # Vectorized proposal for all walkers at once
        current = chain[step - 1].copy()  # (n_walkers, n_params)
        current_lp = log_p[step - 1].copy()  # (n_walkers,)

        proposals = current + rng.normal(0, proposal_stds, (n_walkers, n_params))

        # Enforce bounds
        for i, name in enumerate(param_names):
            if priors[name].bounds is not None:
                lo, hi = priors[name].bounds
                proposals[:, i] = np.clip(proposals[:, i], lo, hi)

        # Evaluate log-posterior for all proposals
        proposal_lp = np.array([log_posterior(proposals[w]) for w in range(n_walkers)])

        # Metropolis acceptance (vectorized)
        log_alpha = proposal_lp - current_lp
        accept = (log_alpha > 0) | ((log_alpha > -math.inf) & (np.log(rng.uniform(size=n_walkers)) < log_alpha))

        # Update accepted walkers
        for w in range(n_walkers):
            if accept[w]:
                chain[step, w] = proposals[w]
                log_p[step, w] = proposal_lp[w]
                accepts[w] += 1
            else:
                chain[step, w] = current[w]
                log_p[step, w] = current_lp[w]

    # Burn-in
    burnin = int(n_steps * burnin_fraction)

    # Flatten walkers post-burnin for posterior stats
    flat_chain = chain[burnin:].reshape(-1, n_params)
    posterior_mean = np.mean(flat_chain, axis=0)
    posterior_std = np.std(flat_chain, axis=0)

    # Gelman-Rubin R-hat (split each walker's post-burnin chain in half)
    r_hat = {}
    post = chain[burnin:]  # (n_post, n_walkers, n_params)
    n_post = post.shape[0]
    if n_post >= 4 and n_walkers >= 2:
        # Split into 2 halves per walker -> 2*n_walkers chains of length n_post//2
        half = n_post // 2
        if half >= 2:
            m = 2 * n_walkers
            n = half
            all_chains = np.zeros((m, n, n_params))
            for w in range(n_walkers):
                all_chains[2 * w] = post[:half, w]
                all_chains[2 * w + 1] = post[half:2 * half, w]

            for i, name in enumerate(param_names):
                chain_means = np.mean(all_chains[:, :, i], axis=1)  # (m,)
                chain_vars = np.var(all_chains[:, :, i], axis=1, ddof=1)  # (m,)
                B = n * np.var(chain_means, ddof=1)  # between-chain variance
                W = np.mean(chain_vars)  # within-chain variance
                var_hat = (1.0 - 1.0 / n) * W + B / n
                r_hat[name] = float(np.sqrt(max(var_hat / max(W, 1e-30), 1.0)))
        else:
            for i, name in enumerate(param_names):
                r_hat[name] = float("nan")
    else:
        for i, name in enumerate(param_names):
            r_hat[name] = float("nan")

    acceptance_rate = float(np.mean(accepts / n_steps))

    return MCMCResult(
        param_names=param_names,
        chain=flat_chain,
        log_prob=log_p[burnin:].reshape(-1),
        acceptance_rate=acceptance_rate,
        r_hat=r_hat,
        burnin=burnin,
        posterior_mean=posterior_mean,
        posterior_std=posterior_std,
        n_walkers=n_walkers,
        n_steps=n_steps,
    )


# ---------------------------------------------------------------------------
# Information gain (KL divergence)
# ---------------------------------------------------------------------------

def information_gain(
    priors: Dict[str, ParameterPrior],
    posterior_mean: np.ndarray,
    posterior_std: np.ndarray,
    param_names: List[str],
) -> Dict[str, float]:
    """Compute KL divergence D_KL(posterior || prior) for each parameter.

    Both distributions approximated as log-normal (or normal for normal priors).
    Positive values = information gained from calibration.

    Returns dict of param_name -> KL divergence in nats.
    """
    result = {}
    for i, name in enumerate(param_names):
        prior = priors[name]
        post_mean = posterior_mean[i]
        post_std = max(posterior_std[i], 1e-30)

        if prior.distribution == "lognormal":
            # KL between two log-normals
            if post_mean <= 0:
                result[name] = float("nan")
                continue
            var_post = post_std ** 2
            mu2_post = post_mean ** 2
            log_mu_post = math.log(mu2_post / math.sqrt(var_post + mu2_post))
            log_sigma_post = math.sqrt(math.log(1.0 + var_post / mu2_post))

            mu1, sig1 = prior._log_mu, prior._log_sigma
            mu2, sig2 = log_mu_post, max(log_sigma_post, 1e-15)

            kl = (math.log(sig2 / sig1)
                  + (sig1 ** 2 + (mu1 - mu2) ** 2) / (2 * sig2 ** 2)
                  - 0.5)
        else:
            # KL between two normals
            mu1, sig1 = prior._log_mu, prior._log_sigma
            mu2, sig2 = post_mean, max(post_std, 1e-15)
            kl = (math.log(sig2 / sig1)
                  + (sig1 ** 2 + (mu1 - mu2) ** 2) / (2 * sig2 ** 2)
                  - 0.5)

        result[name] = float(max(kl, 0.0))
    return result


# ---------------------------------------------------------------------------
# Optimal next experiment
# ---------------------------------------------------------------------------

def optimal_next_experiment(
    priors: Dict[str, ParameterPrior],
    posterior_std: np.ndarray,
    param_names: List[str],
    experiment_map: Optional[Dict[str, List[str]]] = None,
) -> str:
    """Recommend the most informative next experiment based on remaining uncertainty.

    Selects the experiment target whose associated parameters have the highest
    total normalized remaining variance (posterior_std / prior_mean).

    Parameters
    ----------
    priors : dict
        Current priors/posteriors.
    posterior_std : array
        Posterior standard deviations (from calibration result).
    param_names : list
        Ordered parameter names matching posterior_std.
    experiment_map : dict, optional
        Maps experiment name -> list of parameter names it constrains.
        Defaults to the 6 calibration targets.

    Returns
    -------
    str
        Name of the recommended experiment.
    """
    if experiment_map is None:
        experiment_map = {
            "Hall-Petch (EBSD + tensile)": ["sigma0_MPa", "k_HP_MPa_sqrt_m"],
            "Diffusivity (foil weight gain)": ["D0_m2_s", "Q_kJ_mol"],
            "Tafel slopes (LSV)": ["fe_i0_A_m2", "fe_tafel_V_dec", "her_tafel_V_dec"],
            "O2 probe (pO2 mV vs foil C)": ["K_B_offset", "K_CH4_offset"],
            "Tempering kinetics (HV vs T,t)": ["k_softening"],
            "Ni strengthening (ICP-OES + tensile)": ["K_SS_NI_MPa_per_wt"],
        }

    param_to_idx = {name: i for i, name in enumerate(param_names)}

    scores = {}
    for exp_name, params in experiment_map.items():
        total_score = 0.0
        count = 0
        for p in params:
            if p in param_to_idx:
                idx = param_to_idx[p]
                # Normalized uncertainty: posterior_std / prior_mean
                prior_mean = abs(priors[p].mean)
                norm_uncert = posterior_std[idx] / max(prior_mean, 1e-15)
                total_score += norm_uncert
                count += 1
        scores[exp_name] = total_score / max(count, 1)

    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# Synthetic data generators (for testing and driver)
# ---------------------------------------------------------------------------

def generate_synthetic_observations(
    priors: Dict[str, ParameterPrior],
    true_params: Dict[str, float],
    model_fn: Callable[[np.ndarray, Dict[str, ParameterPrior]], Dict[str, float]],
    noise_fraction: float = 0.03,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Generate synthetic noisy observations from a forward model.

    Returns (observations, observation_errors).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    param_names = sorted(priors.keys())
    true_vec = np.array([true_params.get(n, priors[n].mean) for n in param_names])
    pred = model_fn(true_vec, priors)

    observations = {}
    obs_errors = {}
    for k, v in pred.items():
        noise = rng.normal(0, noise_fraction * abs(v))
        observations[k] = float(v + noise)
        obs_errors[k] = float(noise_fraction * abs(v))

    return observations, obs_errors


def screening_forward_model(
    params_vec: np.ndarray,
    priors: Dict[str, ParameterPrior],
) -> Dict[str, float]:
    """Simplified forward model mapping parameters to observable predictions.

    For each calibration target, produces a synthetic observable that depends
    on the relevant parameters. Used for synthetic testing and as a template
    for real-data calibration.

    Observables returned:
    - yield_strength_MPa: from sigma0 + k_HP / sqrt(d) for d=5um
    - D_eff_m2_s: from D0 * exp(-Q/RT) at 900°C
    - total_cathodic_current_A_m2: from Tafel params at eta=0.2V
    - o2_probe_offset: from K_B_offset, K_CH4_offset
    - tempered_HV: from k_softening at P=10000
    - ni_strengthening_MPa: from K_SS_NI at 5 wt% Ni
    """
    param_names = sorted(priors.keys())
    p = {name: params_vec[i] for i, name in enumerate(param_names)}

    result = {}

    # 1. Hall-Petch: sigma_y = sigma0 + k_HP / sqrt(d) with d=5e-6 m
    d_m = 5e-6
    result["yield_strength_MPa"] = p.get("sigma0_MPa", 100) + p.get("k_HP_MPa_sqrt_m", 0.5) / math.sqrt(d_m)

    # 2. Diffusivity: D = D0 * exp(-Q*1000 / (R*T)) at T=1173K (900°C)
    R = 8.314462
    T = 1173.15
    D0 = p.get("D0_m2_s", 6.2e-7)
    Q = p.get("Q_kJ_mol", 80.0)
    result["D_eff_m2_s"] = D0 * math.exp(-Q * 1000.0 / (R * T))

    # 3. Tafel: total current at eta=0.2V from two branches
    eta = 0.2
    i0_fe = p.get("fe_i0_A_m2", 1e-2)
    beta_fe = p.get("fe_tafel_V_dec", 0.12)
    beta_her = p.get("her_tafel_V_dec", 0.14)
    i_fe = i0_fe * 10 ** (eta / max(beta_fe, 0.01))
    i_her = 1e-3 * 10 ** (eta / max(beta_her, 0.01))  # HER i0 fixed at 1e-3
    result["total_cathodic_current_A_m2"] = i_fe + i_her

    # 4. O2 probe: simple offset
    result["o2_probe_offset"] = (p.get("K_B_offset", 1.0) + p.get("K_CH4_offset", 1.0)) / 2.0

    # 5. Tempering: HV = 800 * exp(-k * max(P-8000, 0)) at P=10000
    P = 10000.0
    k = p.get("k_softening", 1.8e-4)
    result["tempered_HV"] = 800.0 * math.exp(-k * max(P - 8000.0, 0.0))

    # 6. Ni strengthening: delta_sigma = K_SS_NI * wt%^0.75
    K = p.get("K_SS_NI_MPa_per_wt", 38.0)
    result["ni_strengthening_MPa"] = K * (5.0 ** 0.75)

    return result
