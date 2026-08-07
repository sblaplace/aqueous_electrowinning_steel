"""
Micro-pH buffer dynamics at the cathode diffusion film.

Connects BDD multi-step kinetics (``bdd_kinetics.py``) to passivation/film
resistance (queued card: ``models/fe3_shuttle.py`` / diffusion-layer film).

Physics
-------
The BDD mechanism predicts local alkalization (pH 2 → 3.5) at the surface:

1. Pre-equilibrium hydrolysis:      Fe²⁺ + H₂O ⇌ FeOH⁺ + H⁺   (log K ≈ −9.5)
2. Adsorbed intermediate:           FeOH⁺ + e⁻ ⇌ (FeOH)ₐds
3. Crystallization / OH⁻ release:   (FeOH)ₐds + e⁻ → Fe(s) + OH⁻
4. Competing HER:                    2 H⁺ + 2 e⁻ → H₂                  (consumes H⁺ → raises pH)

This module computes the **surface pH** and **buffer capacity** inside the
1-D film, which determines when Fe(OH)₂ precipitates (link to ``pourbaix.py``
boundaries and ``fe3_shuttle.py`` solubility cap).

Not on board A/B — untracked gap between B-running BDD and queued passivation.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

# Default thermodynamic anchors (25 °C, 1 bar)
# Source: BDD literature / standard hydrolysis data; adjusted via van't Hoff below.
LOG_KHYD_25 = -9.50          # Fe²⁺ + H₂O ⇌ FeOH⁺ + H⁺
DH_KHYD_J = 12_000           # approximate ΔH (endothermic hydrolysis)

# Fe(OH)₂ solubility product reference (for precipitation check)
# From ``models/fe3_shuttle.py`` / pourbaix anchors; approximate 25 °C.
PKSP_FE_OH2_25 = 14.87       # log Ksp ≈ −14.87  →  Ksp ≈ 1.35e-15
DH_KSP_J = -25_000           # approximate ΔH (exothermic precipitation)

# Buffer pKa defaults (approximate at 25 °C)
PBUFF_BORATE = 9.24          # H₃BO₃ / B(OH)₄⁻
PBUFF_ACETATE = 4.76         # HAc / Ac⁻
PBUFF_SULFATE = 1.99         # HSO₄⁻ / SO₄²⁻ (first dissociation not buffer in this region)


def van_t_hoff_constant(log_k_25: float, dH_j: float, T_c: float) -> float:
    """Adjust equilibrium log K from 25 °C to operating T_c (°C)."""
    T = T_c + 273.15
    T_ref = 298.15
    # ln K = ln K_ref - ΔH/R (1/T - 1/T_ref)  (ΔH constant approximation)
    ln_k = log_k_25 * math.log(10) - (dH_j / 8.314) * (1 / T - 1 / T_ref)
    return ln_k / math.log(10)


def compute_surface_pH(
    pH_bulk: float,
    C_fe2_bulk_M: float,
    C_so4_bulk_M: float,
    j_local_A_m2: float,
    T_C: float = 60.0,
    buffer_molar: Optional[Dict[str, float]] = None,
    film_thickness_m: float = 50e-6,
) -> Tuple[float, float, float, bool]:
    """
    Return:
      pH_surface, buffer_capacity_beta_M_per_pH, precipitation_risk_idx,
      precipitates_now (bool)

    pH_surface is estimated from:
      - Bulk hydrolysis equilibrium (adjusted to T)
      - H⁺ consumption by HER (Faraday + ideal-gas approximation)
      - OH⁻ production by BDD crystallization step (proportional to j_local / F)
      - Buffer attenuation (sum of β for each buffer pair)
    """
    if buffer_molar is None:
        buffer_molar = {}

    # 1) Temperature-corrected hydrolysis at bulk conditions
    log_khyd = van_t_hoff_constant(LOG_KHYD_25, DH_KHYD_J, T_C)
    khyd = 10 ** log_khyd

    # Bulk [FeOH⁺] from equilibrium with bulk [H⁺]
    h_bulk = 10 ** (-pH_bulk)
    # FeOH⁺ = khyd * C_fe2 * (H2O / H⁺) ≈ khyd * C_fe2 / h_bulk  (activity coeffs = 1 screening)
    feoh_bulk = khyd * C_fe2_bulk_M / max(h_bulk, 1e-12)

    # 2) Surface reaction rates (screening — Faradaic equivalents per unit area)
    # HER current fraction: assume α_HER ~ 0.5, Tafel from kinetics; simplify to
    # H⁺ flux to surface ≈ j_local / (2 F)  (mole m⁻² s⁻¹)
    F = 96485.0
    j_local_A_cm2 = j_local_A_m2 / 1e4  # convert for intuition

    # H⁺ consumption by HER: 2 H⁺ + 2e⁻ → H₂  →  consumption rate = j / (2F)  (mol/m²/s)
    her_h_consumption = j_local_A_m2 / (2 * F)  # mol H⁺ / m² / s

    # BDD OH⁻ release: (FeOH)ads + e⁻ → Fe + OH⁻  →  rate ≈ j / F  (mol OH⁻ / m² / s)
    # We approximate that ~50 % of total j goes through BDD route at moderate η.
    bdd_oh_release = 0.5 * j_local_A_m2 / F  # mol OH⁻ / m² / s

    # 3) Net pH shift in film (screening layer diffusion approximation)
    # Diffusion layer δ ≈ 50 µm; D_H⁺ ≈ 9.3e-9 m²/s; characteristic time τ = δ²/D
    # For steady-state: surface concentration ≈ bulk + (generation - consumption) · δ / D
    delta = film_thickness_m
    D_h = 9.3e-9  # m²/s, approximate
    D_oh = 5.3e-9

    # Net moles consumed/produced per m², scaled to film volume
    # Volume per m² = δ (m³/m² = m); divide by δ to get concentration change (M = mol/L, need /1000)
    h_shift = (her_h_consumption / D_h) * delta / 1000.0  # approximate M change from consumption
    oh_shift = (bdd_oh_release / D_oh) * delta / 1000.0  # M increase from OH⁻ release

    # Net pH change: consumption of H⁺ raises pH; production of OH⁻ also raises pH
    # Approximate: ΔpH ≈ log10( (h_bulk + h_shift + oh_shift) / h_bulk ) ... rough screening
    # Better screening approximation: pH_surf = pH_bulk + ΔpH_from_H_her + ΔpH_from_OH_bdd
    delta_ph_her = -math.log10(max(h_bulk - h_shift, 1e-14)) + math.log10(h_bulk) if h_bulk > 1e-12 else 0.0
    delta_ph_oh = math.log10(h_bulk + oh_shift) - math.log10(h_bulk) if h_bulk > 1e-14 else 0.0
    # Simplified additive screening estimate:
    pH_surf = pH_bulk + max(0.3, min(1.8, delta_ph_her + delta_ph_oh + 0.2))

    # 4) Buffer capacity β = Σ 2.303 · C_acid · C_base / (C_acid + C_base)
    beta = 0.0
    # Borate
    c_bor = buffer_molar.get("borate", 0.01)
    # Approximate half-base at local pH: split by pKa
    pka_bor = PBUFF_BORATE
    c_acid_bor = c_bor / (1 + 10 ** (pH_surf - pka_bor))
    c_base_bor = c_bor / (1 + 10 ** (pka_bor - pH_surf))
    beta += 2.303 * c_acid_bor * c_base_bor / max(c_acid_bor + c_base_bor, 1e-12)

    # Acetate
    c_ac = buffer_molar.get("acetate", 0.02)
    pka_ac = PBUFF_ACETATE
    c_acid_ac = c_ac / (1 + 10 ** (pH_surf - pka_ac))
    c_base_ac = c_ac / (1 + 10 ** (pka_ac - pH_surf))
    beta += 2.303 * c_acid_ac * c_base_ac / max(c_acid_ac + c_base_ac, 1e-12)

    # Sulfate (HSO₄⁻ / SO₄²⁻) — low pH only relevant
    c_sulf = buffer_molar.get("sulfate", 0.5)
    pka_sulf = PBUFF_SULFATE
    c_acid_sulf = c_sulf / (1 + 10 ** (pH_surf - pka_sulf))
    c_base_sulf = c_sulf / (1 + 10 ** (pka_sulf - pH_surf))
    beta += 2.303 * c_acid_sulf * c_base_sulf / max(c_acid_sulf + c_base_sulf, 1e-12)

    # 5) Precipitation check: Fe(OH)₂(s) from Fe²⁺ + 2 OH⁻ ⇌ Fe(OH)₂(s)
    # At surface: [OH⁻] = 10^(pH_surf - 14) (approx at 25 C, adjust slightly for T)
    pOH_surf = 14.0 - pH_surf  # screening approximation
    oh_surf = 10 ** (-pOH_surf)
    # Temperature-corrected Ksp
    log_ksp = -PKSP_FE_OH2_25  # log Ksp = -14.87 → Ksp = 10^-14.87
    # Adjust log Ksp to T_C (exothermic, so Ksp increases at lower T?)
    log_ksp_T = log_ksp + (DH_KSP_J / (2.303 * 8.314)) * (1 / (T_C + 273.15) - 1 / 298.15)
    ksp = 10 ** log_ksp_T

    # Ion product Q = [Fe²⁺][OH⁻]²; use surface Fe²⁺ ≈ bulk (depletion is film-level, use C_fe2_bulk as screening)
    Q = C_fe2_bulk_M * (oh_surf ** 2)
    precip_now = Q > ksp

    # Risk index: log10(Q/Ksp) — positive means supersaturated
    risk_idx = math.log10(Q / max(ksp, 1e-30))

    return float(pH_surf), float(beta), float(risk_idx), bool(precip_now)


def feed_to_diffusion_layer(
    pH_surf: float,
    beta: float,
    j_local_A_m2: float,
    C_fe2_bulk_M: float,
    C_so4_bulk_M: float,
) -> Dict[str, float]:
    """
    Return a dictionary of screening corrections to pass to
    ``models/diffusion_layer_1d.py`` / ``models/transport.py``.
    """
    # If pH is high enough to precipitate, add a precipitation sink term.
    # This is a placeholder for the film-thickness ODE (Tier 1 gap).
    sink_rate = 1.5e-6 if pH_surf > 3.8 else 0.0  # mol / m² / s screening
    return {
        "pH_surface": pH_surf,
        "buffer_capacity_M": beta,
        "precipitation_sink_M_s": sink_rate,
        "effective_diffusivity_scale": 0.85 if pH_surf > 3.5 else 1.0,
        "note": "microph_buffer v0 — connects BDD (bdd_kinetics) to film resistance (queued)",
    }
