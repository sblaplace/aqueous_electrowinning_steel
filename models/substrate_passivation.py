"""
Ti drum / substrate passivation (TiO₂ growth) over campaign life.

Not on board A/B — connects B-running ``adhesion_peel.py`` (static interface)
to long-run campaign failure (peel fails at hour 800, not hour 8).

Physics / Chemistry
-------------------
Electrodeposition drums are typically Ti or Ti-alloy substrates.
In acidic sulfate + O₂ (anolyte drift, air ingress, idle periods), Ti forms
a protective TiO₂ film.  Over 1000+ h the film thickens by parabolic
kinetics (Wagner / Mott–Cabrera), eventually becoming brittle and
reducing the interfacial fracture energy G_c required for clean peel.

Key relationships (screening):
- Parabolic growth: δ_ox² = k_p · t
- k_p varies with T, potential (anodic vs cathodic exposure during idle),
  pH, and dissolved O₂.
- Critical peel thickness h_c decreases as δ_ox increases because oxide
  introduces stress concentration and lowers effective W_ad.

Module to wire in: ``adhesion_peel.py``, ``internal_stress.py``,
``closed_loop.py`` (campaign time tracking), ``bath_startup.py`` / shutdown.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

# Screening parabolic rate constants (m²/s) at 60 °C, anodic exposure
# Source: Ti passivation literature; highly variable with electrolyte.
K_P_60C_ANODIC = 4.5e-20  # m²/s  →  δ ≈ 15 nm after 1000 h
K_P_60C_CATHODIC = 5.0e-22  # cathodic protection suppresses growth

# Temperature dependence (Ea ~ 70 kJ/mol for TiO₂ growth)
EA_KP_J = 70_000

# Potential dependence: anodic potential accelerates; cathodic suppresses
# Approximate linear in (E - E_corr) with E_corr ≈ 0.0 V vs SHE for Ti in acid
E_CORR_V = 0.0

# Critical peel thickness reduction from oxide
# Screening: h_c(δ) = h_c0 · (1 - α · δ / δ_crit)  with δ_crit ≈ 100 nm
H_C0_NM = 15.0  # baseline critical thickness (nm) for clean Ti/Fe interface
ALPHA_SUBSTRATE = 0.8
DELTA_CRIT_NM = 100.0


# Reference interfacial fracture energies (J/m²) — from adhesion_peel literature
G_C_CLEAN_TI = 12.0
G_C_OXIDIZED = 3.5  # brittle oxide reduces by ~70 %


def _arrhenius_kp(T_C: float, k_ref: float, T_ref_C: float = 60.0) -> float:
    T = T_C + 273.15
    T_ref = T_ref_C + 273.15
    return k_ref * math.exp(-EA_KP_J / 8.314 * (1 / T - 1 / T_ref))


def oxide_growth_parabolic(
    t_hours: float,
    T_C: float = 60.0,
    E_V_vs_SHE: float = 1.2,  # anodic exposure / open-circuit drift
    pH: float = 1.5,
    pO2_bar: float = 0.21,
    exposure_mode: str = "anodic",  # "anodic", "cathodic", "idle"
) -> Tuple[float, float, float]:
    """
    Return:
      delta_ox_m (oxide thickness), k_p_m2_s, growth_rate_nm_per_1000h
    """
    # Base k_p at 60 °C
    if exposure_mode == "cathodic":
        k_p0 = K_P_60C_CATHODIC
    else:
        k_p0 = K_P_60C_ANODIC

    # Temperature correction
    k_p = _arrhenius_kp(T_C, k_p0)

    # Potential factor: more anodic → faster; cathodic → suppressed
    # Screening linear factor near corrosion potential
    delta_E = E_V_vs_SHE - E_CORR_V
    if delta_E > 0.5:
        pot_factor = 1.0 + 2.0 * (delta_E - 0.5)
    elif delta_E < -0.2:
        pot_factor = 0.3  # cathodic protection
    else:
        pot_factor = 1.0 + 0.5 * delta_E

    # pH factor: lower pH (more acid) can accelerate or suppress depending on
    # oxide stability; screening: mild acceleration below pH 2
    if pH < 2.0:
        ph_factor = 1.1 - 0.05 * pH  # ~1.0 at pH 2, ~1.1 at pH 0.5
    else:
        ph_factor = 1.0

    # O₂ partial pressure: more O₂ → faster growth
    o2_factor = 0.7 + 1.5 * pO2_bar  # ~1.0 at air, ~0.7 under N2

    k_p *= pot_factor * ph_factor * o2_factor

    # Parabolic: δ = sqrt(k_p * t)  [t in seconds]
    t_s = t_hours * 3600.0
    delta_ox = math.sqrt(k_p * t_s)
    delta_ox = max(delta_ox, 0.0)

    rate_1000h = math.sqrt(k_p * 3_600_000.0)  # nm/1000h approx (convert m→nm after)
    rate_1000h_nm = rate_1000h * 1e9

    return delta_ox, k_p, rate_1000h_nm


def critical_peel_thickness(
    delta_ox_m: float,
    h_c0_nm: float = H_C0_NM,
) -> float:
    """
    Critical peel thickness (nm) decreases with oxide growth.
    Returns nm; if δ_ox > δ_crit, peel is essentially impossible.
    """
    delta_ox_nm = delta_ox_m * 1e9
    if delta_ox_nm >= DELTA_CRIT_NM:
        return 0.0
    decay = ALPHA_SUBSTRATE * (delta_ox_nm / DELTA_CRIT_NM)
    return float(max(0.0, h_c0_nm * (1.0 - decay)))


def interfacial_fracture_energy(
    delta_ox_m: float,
    G_c_clean: float = G_C_CLEAN_TI,
    G_c_ox: float = G_C_OXIDIZED,
) -> float:
    """Approximate G_c (J/m²) as oxide fraction increases."""
    delta_ox_nm = delta_ox_m * 1e9
    frac_ox = min(1.0, delta_ox_nm / DELTA_CRIT_NM)
    return float(G_c_clean - (G_c_clean - G_c_ox) * frac_ox)


def feed_to_adhesion_peel(
    delta_ox_m: float,
    h_c0_nm: float = H_C0_NM,
    G_c_J_m2: Optional[float] = None,
) -> Dict[str, float]:
    """
    Return correction dictionary for ``adhesion_peel.py``.
    If G_c_J_m2 not given, compute from oxide thickness.
    """
    if G_c_J_m2 is None:
        G_c_J_m2 = interfacial_fracture_energy(delta_ox_m)
    h_c = critical_peel_thickness(delta_ox_m, h_c0_nm)
    return {
        "delta_ox_m": delta_ox_m,
        "delta_ox_nm": delta_ox_m * 1e9,
        "critical_peel_thickness_nm": h_c,
        "interfacial_fracture_energy_J_m2": G_c_J_m2,
        "peel_viable": h_c > 3.0,
        "note": "substrate_passivation v0 — untracked #5; connects adhesion_peel + campaign life",
    }
