"""Time-dependent internal-stress relaxation for long runs (winding / defect).

Closes the gap named in ``CHEM_PHYS_REVIEW.md`` Tier 3.5:

    "The program runs at 100-300 mA/cm² for hours. The internal-stress
    model returns a snapshot; in reality stress relaxes by: dislocation
    glide, GB diffusion (Coble), and H-enhanced localised plasticity.
    The same stress relaxation is what determines whether the deposit
    can survive being wound on a drum at thickness. A simple
    ``stress_relaxation.py`` with a log-linear σ(t) = σ₀(1 - A·ln(1 +
    t/τ)) closure, coupled to the temperature and H fields, would let
    ``closed_loop.py`` report a *defect rate* with a stress mechanism
    attached."

``internal_stress.py`` already produces the *snapshot* residual stress
(std::sigma0) from plating conditions.  This module turns that snapshot into
a time evolution:

* **Log-linear closure.** ``σ(t) = σ₀ (1 − A·ln(1 + t/τ))`` — a logarithmic
  (primary-creep-like) decay, the standard screening form for low-temperature
  stress relaxation via dislocation glide and GB (Coble) diffusion.

* **Temperature coupling.** The relaxation time is Arrhenius-activated with a
  GB-diffusion-scale activation energy: ``τ(T) = τ_ref·exp(Q/R·(1/T − 1/T_ref))``.
  Higher temperature → smaller τ → faster decay onto the floor (tested).

* **Hydrogen coupling.** H-enhanced localised plasticity shortens the effective
  relaxation time: ``τ(C_H) ∝ exp(−C_H/C_H_ref)``.  More diffusible hydrogen →
  faster relaxation (tested).  This is the *same* physical channel the
  (separate) ``stress_hydrogen_coupling`` module relaxes through — here it acts
  on the screening closure time constant rather than on a fixed-point strength.

* **Floor.** Stress never relaxes below a residual floor ``σ_floor`` (the
  mechanically as-locked component cannot diffuse away at these temperatures),
  so the closure saturates rather than decaying to zero or negative.

* **Defect rate.** A stress-mechanism defect rate ``k(t) = k0·(σ(t)/σ₀)ⁿ``,
  which ``closed_loop.py`` reports per time step.  Because peel-driving energy
  scales as σ², the exponent defaults to 2; a deposit that relaxes onto the
  floor has its stress-driven defect rate collapse by ``(σ_floor/σ₀)ⁿ``.

Additive and opt-in: the module is standalone (the ``closed_loop`` hook is a
new method that does not alter ``simulate``), so the default code path is
byte-identical to the snapshot behaviour.  All constants are screening
estimates to be recovered from coupon-curvature / in-situ stress
measurements.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .internal_stress import deposit_stress_from_conditions  # stress seed

# ℞ = 8.314 J/(mol·K)
R_GAS_J_MOL_K = 8.31446261815324

# Screening reference: 60 °C operating temperature.
T_REF_K = 273.15 + 60.0


@dataclass(frozen=True)
class StressRelaxationParams:
    """Screening parameters for the logarithmic stress-relaxation closure.

    All are literature-order engineering estimates — the same credibility
    (L0 screening) as the rest of ``internal_stress``.  Recovered from
    coupon-curvature / in-situ bent-strip stress measurement.
    """

    # Log-linear amplitude: total fractional relaxation eventually available.
    A: float = 0.30
    # Relaxation time (hours) at the reference temperature and zero hydrogen.
    tau_ref_hr: float = 25.0
    # Arrhenius activation energy for GB (Coble) diffusion, J/mol.
    Q_relax_J_mol: float = 60.0e3
    # Reference temperature for tau_ref, K.
    T_ref_K: float = T_REF_K
    # Diffusible hydrogen at which tau is reduced by a factor of ~e
    # (H-enhanced localised plasticity channel).
    C_H_ref_ppm: float = 200.0
    # Residual stress floor (MPa): the mechanically as-locked component that
    # cannot relax at operating temperature. sigma decays toward this, never
    # below it.
    sigma_floor_MPa: float = 40.0
    # Defect-rate base (per hour) at the full unrelaxed stress.
    defect_rate_ref_per_hr: float = 0.02
    # Exponent linking retained stress to the defect rate (peel G ∝ σ² → 2).
    defect_exponent: float = 2.0

    def __post_init__(self) -> None:
        if not 0.0 < self.A < 1.0:
            raise ValueError("A (log-linear amplitude) must lie in (0, 1)")
        if self.tau_ref_hr <= 0:
            raise ValueError("tau_ref_hr must be positive")
        if self.Q_relax_J_mol < 0:
            raise ValueError("Q_relax_J_mol cannot be negative")
        if self.T_ref_K <= 0:
            raise ValueError("T_ref_K must be positive")
        if self.C_H_ref_ppm <= 0:
            raise ValueError("C_H_ref_ppm must be positive")
        if self.sigma_floor_MPa < 0:
            raise ValueError("sigma_floor_MPa cannot be negative")
        if self.defect_rate_ref_per_hr < 0:
            raise ValueError("defect_rate_ref_per_hr cannot be negative")


def relaxation_tau_hr(
    temperature_C: float,
    C_H_ppm: float = 0.0,
    params: StressRelaxationParams | None = None,
) -> float:
    """Effective relaxation time τ (hours) coupled to T and H fields.

    τ(T, C_H) = τ_ref · exp(Q/R·(1/T − 1/T_ref)) · exp(−C_H / C_H_ref)

    The Arrhenius term makes higher temperature relax faster (smaller τ); the
    hydrogen term makes higher diffusible hydrogen relax faster via
    H-enhanced localised plasticity.  Both directions are tested.
    """
    p = params or StressRelaxationParams()
    T_K = temperature_C + 273.15
    if T_K <= 0:
        raise ValueError("temperature_C must be above absolute zero")
    arrhenius = np.exp(p.Q_relax_J_mol / R_GAS_J_MOL_K * (1.0 / T_K - 1.0 / p.T_ref_K))
    hydrogen = np.exp(-max(C_H_ppm, 0.0) / p.C_H_ref_ppm)
    return float(p.tau_ref_hr * arrhenius * hydrogen)


def sigma_relaxed(
    sigma0_MPa: float,
    elapsed_hr: float,
    temperature_C: float,
    C_H_ppm: float = 0.0,
    params: StressRelaxationParams | None = None,
) -> dict:
    """Log-linear closure: σ(t) = σ₀ (1 − A·ln(1 + t/τ)), floored at σ_floor.

    Returns the relaxed stress (never below the residual floor), the retained
    fraction, the effective τ, and whether the floor has been reached.
    """
    p = params or StressRelaxationParams()
    if sigma0_MPa < 0:
        raise ValueError("sigma0_MPa cannot be negative")
    if elapsed_hr < 0:
        raise ValueError("elapsed_hr cannot be negative")
    tau = relaxation_tau_hr(temperature_C, C_H_ppm, p)
    if elapsed_hr == 0.0:
        sigma = sigma0_MPa * 1.0
    else:
        sigma_unfloored = sigma0_MPa * (1.0 - p.A * np.log1p(elapsed_hr / tau))
        sigma = max(sigma_unfloored, min(p.sigma_floor_MPa, sigma0_MPa))
    floor_reached = bool(sigma <= p.sigma_floor_MPa + 1e-9)
    return {
        "sigma_MPa": float(sigma),
        "retained_fraction": float((sigma / sigma0_MPa) if sigma0_MPa > 0 else 1.0),
        "tau_hr": float(tau),
        "floor_reached": floor_reached,
        "sigma_floor_MPa": float(min(p.sigma_floor_MPa, sigma0_MPa)),
    }


def sigma_relaxation_series(
    sigma0_MPa: float,
    time_hr: np.ndarray,
    temperature_C: float,
    C_H_ppm: float = 0.0,
    params: StressRelaxationParams | None = None,
) -> np.ndarray:
    """Vectorised σ(t) over a time array (same closure as :func:`sigma_relaxed`)."""
    p = params or StressRelaxationParams()
    t = np.asarray(time_hr, dtype=float)
    with np.errstate(divide="ignore"):
        sigma_unfloored = sigma0_MPa * (
            1.0 - p.A * np.log1p(t / relaxation_tau_hr(temperature_C, C_H_ppm, p))
        )
    floor = min(p.sigma_floor_MPa, sigma0_MPa)
    return np.maximum(sigma_unfloored, floor)


def stress_defect_rate(
    sigma_MPa: float,
    sigma0_MPa: float,
    params: StressRelaxationParams | None = None,
) -> dict:
    """Stress-mechanism defect rate k(t) = k0·(σ(t)/σ₀)ⁿ (per hour).

    The retained-stress fraction drives a peel-type defect rate with exponent
    ``defect_exponent`` (default 2, matching peel energy G ∝ σ²).  As the
    deposit relaxes onto the floor its defect rate collapses by
    ``(σ_floor/σ₀)ⁿ``.  Returns the rate plus the mechanism attribution.
    """
    p = params or StressRelaxationParams()
    if sigma0_MPa <= 0:
        raise ValueError("sigma0_MPa must be positive")
    retention = max(sigma_MPa, 0.0) / sigma0_MPa
    rate = p.defect_rate_ref_per_hr * retention**p.defect_exponent
    return {
        "defect_rate_per_hr": float(rate),
        "retention_fraction": float(retention),
        "mechanism": "stress-driven peel (relaxation-modulated)",
        "stress_MPa": float(sigma_MPa),
        "stress_relaxed_fraction": float(1.0 - retention),
    }


def survives_drum_winding(
    sigma_MPa: float,
    sigma_survival_threshold_MPa: float = 150.0,
    params: StressRelaxationParams | None = None,
) -> dict:
    """Screening verdict: does the relaxed deposition survive winding?

    ``internal_stress``'s peel/curl machinery already decides the adhesion
    verdict at a snapshot; this is the *time* complement — for a deposit that
    survived deposition, has relaxation brought its retained stress below the
    threshold at which winding-on-a-drum self-releases?  Screening-level,
    additive, and orthogonal to ``adhesion_peel``.
    """
    p = params or StressRelaxationParams()
    floor = min(p.sigma_floor_MPa, sigma_MPa) if sigma_MPa >= 0 else sigma_MPa
    ok = sigma_MPa <= sigma_survival_threshold_MPa
    return {
        "survives_winding": bool(ok),
        "sigma_MPa": float(sigma_MPa),
        "sigma_survival_threshold_MPa": float(sigma_survival_threshold_MPa),
        "relaxed_to_floor": bool(sigma_MPa <= floor + 1e-9),
        "margin_MPa": float(sigma_survival_threshold_MPa - sigma_MPa),
    }


def seed_stress_snapshot_Mpa(
    j_mA_cm2: float = 100.0,
    current_efficiency_percent: float = 85.0,
    deposition_time_s: float = 900.0,
    bath_pH: float = 3.0,
    temperature_C: float = 60.0,
    substrate: str = "ti_passive_tio2",
    saccharin_g_L: float = 0.0,
    chloride_bath: bool = False,
) -> dict:
    """Stress snapshot from plating conditions (reuses ``internal_stress``).

    This is the re-derivation boundary the CHEM_PHYS_REVIEW Tier 3.5 text
    demands: do **not** recompute the residual stress here — consume
    ``internal_stress.deposit_stress_from_conditions`` and relax *its*
    snapshot over time.  Returns the total residual stress and its components.
    """
    sd = deposit_stress_from_conditions(
        j_mA_cm2=j_mA_cm2,
        current_efficiency_percent=current_efficiency_percent,
        deposition_time_s=deposition_time_s,
        bath_pH=bath_pH,
        temperature_C=temperature_C,
        substrate=substrate,
        saccharin_g_L=saccharin_g_L,
        chloride_bath=chloride_bath,
    )
    comp = sd["components"]
    return {
        "sigma0_MPa": float(comp["total_MPa"]),
        "components_MPa": {k: float(v) for k, v in comp.items()},
        "derived": {k: float(v) for k, v in sd["derived"].items()},
    }


def model_scope() -> dict:
    """What this module does and does not compute (house scope contract)."""
    return {
        "computes": [
            "Log-linear stress relaxation σ(t) = σ₀(1 − A·ln(1 + t/τ))",
            "Arrhenius temperature coupling of the relaxation time (GB diffusion)",
            "H-enhanced-plasticity shortening of the relaxation time (C_H field)",
            "Residual stress floor: relaxation saturates, never drives σ negative",
            "Stress-mechanism defect rate k = k0·(σ/σ₀)ⁿ for closed_loop.py",
            "Drum-winding survival screen at a retained-stress threshold",
        ],
        "does_not_compute": [
            "The residual-stress snapshot itself (reused from internal_stress)",
            "Spatially resolved stress through the deposit thickness",
            "Anelastic recovery / reverse relaxation on unloading",
            "Grain-size-dependent Coble creep (single GB-diffusion activation)",
            "Recrystallisation or hydrogen bake-out kinetics",
        ],
        "reuses_without_duplicating": [
            "internal_stress.deposit_stress_from_conditions (the σ₀ snapshot)",
            "internal_stress residual-stress components and derived quantities",
        ],
        "key_uncertainty": (
            "All screening constants: A, tau_ref, Q_relax, C_H_ref, sigma_floor "
            "and k0. Literature-order estimates; recovered from coupon-curvature "
            "/ in-situ bent-strip stress vs time measurements."
        ),
        "limitations": (
            "Screening creep-with-time closure. No measured stress-relaxation "
            "data exists in this repository; the closure makes the snapshot "
            "time-resolved so closed_loop.py can report a defect rate, and "
            "the coupon protocol in internal_stress is the intended recovery "
            "route for every constant."
        ),
    }
