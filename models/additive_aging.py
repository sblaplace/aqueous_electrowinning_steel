"""
Electrolyte organic addititive aging / degradation over long campaigns.

Not on board A/B — connects queued ``leveler_kinetics.py`` (fresh-state
Langmuir) to ``bath_dynamics.py`` / ``closed_loop.py`` (long-run CSTR).

Physics / Chemistry
-------------------
Real baths run 100–300 h.  Additives (saccharin, thiourea, PEG, coumarin,
clavulanic-type organics) degrade by:

- **Anodic oxidation** (DSA / OER surface):  R–S–R' → sulfoxides / sulfonates;
  thiourea → formamidine disulfide; PEG → carbonyl / carboxylate fragments.
- **Cathodic reduction / H-radical attack**:  C=S bonds hydrogenolyze;
  aromatics partially reduce.
- **Hydrolysis at pH > 3.5**:  Ester / amide groups in some brighteners
  hydrolyze with t½ dependent on temperature.

Current screening assumption: ``leveler_kinetics.py`` treats Γ_organic as
constant (fresh bath).  This module provides a first-order decay envelope
so ``bath_dynamics.py`` can update C_add at each timestep.

Modules to wire into when built: ``bath_dynamics.py``, ``leveler_kinetics.py``,
``closed_loop.py``, ``calibration_pipeline.py`` (for replenishment policy).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

# Screening rate constants (first-order, 60 °C, ~200 mA/cm², acidic sulfate)
# Source: literature on additive degradation in Fe-plating baths (e.g.
# thiourea oxidation by OER intermediates; saccharin cathodic reduction).
# These are SCREENING values; calibration against long-run FE drift is needed.
K_ANODIC_60C = 2.5e-6     # s⁻¹  (approx 1 / (4.6 days))
K_CATHODIC_60C = 8.0e-7   # s⁻¹  (slower; H-attack selective)
K_HYDROLYSIS_60C = 1.0e-7  # s⁻¹  (pH-dependent; activates above pH ~3)

# Temperature correction (Arrhenius, Ea ~ 55 kJ/mol screening)
EA_DEGRAD_J = 55_000

# Replenishment / feed policies (molar per L per day)
DEFAULT_REPL_RATE_MOL_L_DAY = 0.005  # ~0.8 g/L/day saccharin equivalent

# Additive identity mapping for light-weight tracking
ADDITIVE_SPEC = {
    "saccharin": {"M_fwd_g_mol": 183.18, "primary_loss": "cathodic"},
    "thiourea": {"M_fwd_g_mol": 76.12, "primary_loss": "anodic"},
    "peg": {"M_fwd_g_mol": 400.0, "primary_loss": "anodic"},  # average MW
    "coumarin": {"M_fwd_g_mol": 146.15, "primary_loss": "hydrolysis"},
}


def _arrhenius_factor(Ea_J: float, T_C: float) -> float:
    T = T_C + 273.15
    T_ref = 333.15  # 60 °C reference
    return math.exp(-Ea_J / 8.314 * (1 / T - 1 / T_ref))


def decay_rate_per_hour(
    T_C: float = 60.0,
    j_local_A_cm2: float = 0.2,
    pH_surf: float = 2.5,
    additive_id: str = "saccharin",
) -> Dict[str, float]:
    """
    Screening first-order decay rates (h⁻¹) for a given operating point.
    Returns rates for anodic, cathodic, hydrolytic loss channels.
    """
    f_T = _arrhenius_factor(EA_DEGRAD_J, T_C)
    # Light current-dependence: anodic rate scales with OER current fraction
    # (simplified: linear with j above 50 mA/cm²)
    j_factor = min(3.0, 1.0 + 10.0 * max(0.0, j_local_A_cm2 - 0.05))

    # Hydrolysis activates above ~pH 3.5 (screening threshold)
    hyd_factor = 1.0 if pH_surf > 3.5 else 0.1

    spec = ADDITIVE_SPEC.get(additive_id, ADDITIVE_SPEC["saccharin"])
    primary = spec["primary_loss"]

    k_anodic = K_ANODIC_60C * f_T * j_factor * 3600.0  # s⁻¹ → h⁻¹
    k_cathodic = K_CATHODIC_60C * f_T * 3600.0
    k_hydro = K_HYDROLYSIS_60C * f_T * hyd_factor * 3600.0

    # If primary loss is anodic, weight that channel; others are background
    if primary == "anodic":
        k_eff = k_anodic + 0.2 * k_cathodic + 0.1 * k_hydro
    elif primary == "cathodic":
        k_eff = k_cathodic + 0.2 * k_anodic + 0.1 * k_hydro
    else:
        k_eff = k_hydro + 0.3 * (k_anodic + k_cathodic)

    return {
        "k_anodic_h": k_anodic,
        "k_cathodic_h": k_cathodic,
        "k_hydro_h": k_hydro,
        "k_eff_h": k_eff,
        "primary_channel": primary,
        "half_life_h": math.log(2) / max(k_eff, 1e-9),
        "note": "additive_aging v0 — untracked #2; connects queued leveler_kinetics to bath_dynamics",
    }


def effective_leveler_coverage(
    C_initial_M: float,
    t_hours: float,
    decay_dict: Optional[Dict[str, float]] = None,
    replenishment_M_per_hour: float = DEFAULT_REPL_RATE_MOL_L_DAY / 24.0,
) -> float:
    """
    Effective surface concentration after decay + replenishment.
    Returns C_eff (M) — feed to ``models/leveler_kinetics.py`` Langmuir.
    Simple 1st-order with continuous feed (screening).
    """
    if decay_dict is None:
        decay_dict = decay_rate_per_hour()
    k_eff = decay_dict.get("k_eff_h", 1e-6)
    # Exact 1st-order with constant source: C(t) = C_ss + (C0 - C_ss) * exp(-k t)
    # C_ss = replenishment / k_eff  (steady-state with feed)
    C_ss = replenishment_M_per_hour / max(k_eff, 1e-12)
    C_eff = C_ss + (C_initial_M - C_ss) * math.exp(-k_eff * t_hours)
    return float(C_eff)


def feed_to_bath_dynamics(
    C_eff_M: float,
    additive_id: str = "saccharin",
    C_initial_M: float = 0.002,
    t_hours: float = 24.0,
) -> Dict[str, float]:
    """Output dictionary for ``bath_dynamics.py`` / ``closed_loop.py`` update."""
    decay = decay_rate_per_hour()
    return {
        "additive_id": additive_id,
        "C_initial_M": C_initial_M,
        "C_eff_M": C_eff_M,
        "half_life_h": decay["half_life_h"],
        "replenishment_rate_M_h": DEFAULT_REPL_RATE_MOL_L_DAY / 24.0,
        "note": "additive_aging v0 — untracked #2; connects queued leveler_kinetics to CSTR",
    }
