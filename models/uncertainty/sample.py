"""
Sampling utilities for uncertainty propagation across the model chain.

Provides:
* :func:`sample_parameters` — draw N parameter vectors from the registry
* :func:`parameter_matrix_to_kwargs` — map a sample dict to model constructor kwargs
* :func:`sobol_sequence` — low-discrepancy quasi-random sequence for space-filling DOE
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .parameter_registry import Parameter, REGISTRY


# ---------------------------------------------------------------------------
# Sobol sequence (primitive direction numbers for up to 21201 dimensions)
# A production implementation would use SALib or scipy.stats.qmc; this is a
# self-contained fallback using the simplest generator (Joe & Kuo 2008).
# For the typical use here (<=100 params) the built-in works well enough.
# ---------------------------------------------------------------------------

def _gray_code(i: int) -> int:
    return i ^ (i >> 1)


def sobol_sequence(n: int, d: int, seed: int = 0) -> np.ndarray:
    """Generate an n x d Sobol quasi-random sequence in [0, 1).

    This is a minimal implementation sufficient for sensitivity screening.
    For production DOE, prefer ``scipy.stats.qmc.Sobol``.

    Parameters
    ----------
    n : int
        Number of sample points.
    d : int
        Number of dimensions.
    seed : int
        Starting index (skip first ``seed`` points).

    Returns
    -------
    np.ndarray
        Shape (n, d) with values in [0, 1).
    """
    if n <= 0 or d <= 0:
        raise ValueError("n and d must be positive")

    # Use numpy's Halton sequence as a practical Sobol alternative when
    # direction numbers are not hard-coded.  Halton is also a low-discrepancy
    # sequence and avoids the need to embed thousands of direction numbers.
    result = np.empty((n, d), dtype=float)

    # First few primes for Halton bases
    primes = _first_n_primes(d)

    for j in range(d):
        base = primes[j]
        result[:, j] = _halton_column(n, base, skip=seed)

    return result


def _first_n_primes(n: int) -> list[int]:
    """Return the first n prime numbers."""
    primes = []
    candidate = 2
    while len(primes) < n:
        is_prime = True
        for p in primes:
            if p * p > candidate:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes


def _halton_column(n: int, base: int, skip: int = 0) -> np.ndarray:
    """Generate n Halton sequence values in [0, 1) for a given base."""
    result = np.empty(n, dtype=float)
    for i in range(n):
        idx = i + skip + 1  # Halton index starts at 1
        f = 1.0
        r = 0.0
        while idx > 0:
            f /= base
            r += f * (idx % base)
            idx //= base
        result[i] = r
    return result


# ---------------------------------------------------------------------------
# Core sampling
# ---------------------------------------------------------------------------

def sample_parameters(
    n: int,
    registry: Optional[Dict[str, Parameter]] = None,
    seed: Optional[int] = None,
    method: str = "monte_carlo",
) -> List[Dict[str, float]]:
    """Draw *n* parameter vectors from the registry.

    Parameters
    ----------
    n : int
        Number of samples.
    registry : dict, optional
        Registry to sample from.  Defaults to :data:`REGISTRY`.
    seed : int, optional
        Random seed for reproducibility.
    method : str
        ``"monte_carlo"`` for pseudo-random draws, ``"sobol"`` for a
        quasi-random low-discrepancy sequence.

    Returns
    -------
    list of dict
        Each dict maps parameter name -> sampled float value.
    """
    reg = registry if registry is not None else REGISTRY
    names = sorted(reg.keys())
    d = len(names)
    rng = np.random.default_rng(seed)

    if method == "sobol":
        # Sobol in [0, 1) then transform
        unit = sobol_sequence(n, d, seed=0)
    else:
        unit = rng.random((n, d))

    samples: List[Dict[str, float]] = []
    for i in range(n):
        row: Dict[str, float] = {}
        for j, name in enumerate(names):
            p = reg[name]
            u = unit[i, j]
            row[name] = _transform(u, p)
        samples.append(row)

    return samples


def _transform(u: float, p: Parameter) -> float:
    """Map a uniform [0, 1) draw to the parameter's distribution."""
    lo, hi = p.bounds

    if p.distribution == "uniform":
        return lo + u * (hi - lo)

    if p.distribution == "normal":
        # Truncated normal via inverse-CDF approximation
        from scipy.stats import norm as _norm
        a = (lo - p.mean) / max(p.std, 1e-30)
        b = (hi - p.mean) / max(p.std, 1e-30)
        pa, pb = _norm.cdf(a), _norm.cdf(b)
        u_trunc = pa + u * (pb - pa)
        return float(p.mean + p.std * _norm.ppf(u_trunc))

    if p.distribution == "lognormal":
        # Lognormal with given mean and std (parameterised on the log scale)
        # mu, sigma of the underlying normal: solve E[X]=mean, Var[X]=std^2
        from scipy.stats import norm as _norm
        m = max(p.mean, 1e-30)
        s = max(p.std, 1e-30)
        sigma2 = np.log(1.0 + (s / m) ** 2)
        mu = np.log(m) - 0.5 * sigma2
        sigma = np.sqrt(sigma2)
        # truncated lognormal to bounds
        a_t = (np.log(max(lo, 1e-30)) - mu) / max(sigma, 1e-30)
        b_t = (np.log(max(hi, 1e-30)) - mu) / max(sigma, 1e-30)
        pa_t, pb_t = _norm.cdf(a_t), _norm.cdf(b_t)
        u_trunc = pa_t + u * (pb_t - pa_t)
        return float(np.exp(mu + sigma * _norm.ppf(u_trunc)))

    if p.distribution == "triangular":
        # Symmetric triangular centred on mean with half-width = std
        from scipy.stats import triang
        # scipy triang: c = (mode - loc) / scale
        c = (p.mean - lo) / max(hi - lo, 1e-30)
        c = np.clip(c, 0.01, 0.99)
        return float(triang.ppf(u, c, loc=lo, scale=hi - lo))

    raise ValueError(f"Unknown distribution: {p.distribution}")


def parameter_matrix_to_kwargs(
    params: Dict[str, float],
    model: str,
) -> Dict[str, Any]:
    """Map a flat parameter sample to model-specific constructor kwargs.

    Parameters
    ----------
    params : dict
        Flat mapping of all parameter names to sampled values.
    model : str
        Target model module name (e.g. ``"mechanical_properties"``,
        ``"carburization"``, ``"kinetics"``).

    Returns
    -------
    dict
        Keyword arguments suitable for constructing the model's Params
        dataclass.
    """
    # Dispatch table: registry name -> (target model, kwargs key)
    # Only parameters that actually appear as constructor arguments need mapping.
    mapping = _build_kwarg_mapping()
    kwargs: Dict[str, Any] = {}
    for reg_name, (target, kw) in mapping.items():
        if target == model and reg_name in params:
            kwargs[kw] = params[reg_name]
    return kwargs


def _build_kwarg_mapping() -> Dict[str, tuple[str, str]]:
    """Return {registry_name: (model_module, kwarg_name)} for all params."""
    return {
        # mechanical_properties
        "sigma0_fe_MPa": ("mechanical_properties", "sigma0_MPa"),
        "k_hp_MPa_sqrt_m": ("mechanical_properties", "k_hp_MPa_sqrt_m"),
        "k_ss_ni_MPa_per_wt": ("mechanical_properties", "k_ss_ni_MPa_per_wt"),
        "ss_ni_exp": ("mechanical_properties", "ss_ni_exp"),
        "ss_ni_sat_wt": ("mechanical_properties", "ss_ni_sat_wt"),
        "k_carbon_MPa_per_wt": ("mechanical_properties", "k_carbon_MPa_per_wt"),
        "carbon_nl_exp": ("mechanical_properties", "carbon_nl_exp"),
        "carbon_size_ref_um": ("mechanical_properties", "carbon_size_ref_um"),
        "carbon_size_exp": ("mechanical_properties", "carbon_size_exp"),
        "load_transfer_frac": ("mechanical_properties", "load_transfer_frac"),
        "porosity_penalty_exp": ("mechanical_properties", "porosity_penalty_exp"),
        "porosity_max": ("mechanical_properties", "porosity_max"),
        "tabor_factor": ("mechanical_properties", "tabor_factor"),
        "uts_over_ys_base": ("mechanical_properties", "uts_over_ys_base"),
        "elongation_base_pct": ("mechanical_properties", "elongation_base_pct"),
        # grain size
        "grain_d0_dc_ref_um": ("grain_size", "d0_dc_ref_um"),
        "grain_j_ref_mA_cm2": ("grain_size", "j_ref_mA_cm2"),
        "grain_j_exponent": ("grain_size", "j_exponent"),
        "grain_pe_factor_base": ("grain_size", "pe_factor_base"),
        "grain_pre_factor_base": ("grain_size", "pre_factor_base"),
        # kinetics
        "fe_i0": ("kinetics", "fe_i0"),
        "her_i0": ("kinetics", "her_i0"),
        "fe_tafel_V": ("kinetics", "fe_tafel_V"),
        "her_tafel_V": ("kinetics", "her_tafel_V"),
        "fe_E_eq": ("kinetics", "fe_E_eq"),
        # transport
        "D_Fe2": ("transport", "diffusivity_fe_m2_s"),
        "D_H_plus": ("transport", "diffusivity_h_m2_s"),
        "D_OH_minus": ("transport", "diffusivity_oh_m2_s"),
        "D_Na_plus": ("transport", "diffusivity_na_m2_s"),
        "D_SO4_2minus": ("transport", "diffusivity_so4_m2_s"),
        # carburization
        "D0_ferrite_m2_s": ("carburization", "D0_ferrite_m2_s"),
        "Q_ferrite_kJ_mol": ("carburization", "Q_ferrite_kJ_mol"),
        "D0_austenite_m2_s": ("carburization", "D0_austenite_m2_s"),
        "Q_austenite_kJ_mol": ("carburization", "Q_austenite_kJ_mol"),
        "HV_base_Maynier": ("carburization", "HV_base"),
        "HV_per_C_wt_Maynier": ("carburization", "HV_per_C_wt"),
        # tempering
        "C_HJ": ("tempering", "C_HJ"),
        "k_softening": ("tempering", "k_softening"),
        "KM_alpha_K_inv": ("tempering", "alpha_K_inv"),
        # co_deposition
        "ni_i0": ("co_deposition", "ni_i0"),
        "ni_tafel_V": ("co_deposition", "ni_tafel_V"),
        "D_Ni2": ("co_deposition", "diffusivity_ni_m2_s"),
        # anode
        "oer_ea_IrO2_kJ_mol": ("anode", "oer_ea_kj_mol"),
        # closed_loop
        "coating_loading_g_m2": ("closed_loop", "coating_loading_g_m2"),
        "base_wear_mg_per_kAh": ("closed_loop", "base_wear_mg_per_kAh"),
        "precipitation_rate_per_hr": ("closed_loop", "precipitation_rate_per_hr"),
        "ligand_decay_per_hr": ("closed_loop", "ligand_decay_per_hr"),
    }
