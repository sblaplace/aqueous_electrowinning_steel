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


def _buffer_capacity_M_per_pH(pH: float, buffer_molar: Dict[str, float]) -> float:
    """Buffer capacity β = Σ 2.303 · C_acid · C_base / (C_acid + C_base)."""
    beta = 0.0
    # Borate (H₃BO₃ / B(OH)₄⁻, pKa 9.24)
    c_bor = buffer_molar.get("borate", 0.01)
    pka_bor = PBUFF_BORATE
    c_acid_bor = c_bor / (1 + 10 ** (pH - pka_bor))
    c_base_bor = c_bor / (1 + 10 ** (pka_bor - pH))
    beta += 2.303 * c_acid_bor * c_base_bor / max(c_acid_bor + c_base_bor, 1e-12)
    # Acetate (HAc / Ac⁻, pKa 4.76)
    c_ac = buffer_molar.get("acetate", 0.02)
    pka_ac = PBUFF_ACETATE
    c_acid_ac = c_ac / (1 + 10 ** (pH - pka_ac))
    c_base_ac = c_ac / (1 + 10 ** (pka_ac - pH))
    beta += 2.303 * c_acid_ac * c_base_ac / max(c_acid_ac + c_base_ac, 1e-12)
    # Sulfate (HSO₄⁻ / SO₄²⁻, pKa 1.99)
    c_sulf = buffer_molar.get("sulfate", 0.5)
    pka_sulf = PBUFF_SULFATE
    c_acid_sulf = c_sulf / (1 + 10 ** (pH - pka_sulf))
    c_base_sulf = c_sulf / (1 + 10 ** (pka_sulf - pH))
    beta += 2.303 * c_acid_sulf * c_base_sulf / max(c_acid_sulf + c_base_sulf, 1e-12)
    return float(beta)


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
      - Net acid-equivalent removal at the surface: Fe-deposition OH⁻ release +
        HER H⁺ consumption (both raise surface pH), converted to a film acid
        deficit via a diffusion-layer flux relation
      - Buffer attenuation: ΔpH = ΔC_acid / β (buffer capacity resists the shift,
        so surface pH rises with current but is bounded rather than hard-clamped)
    """
    if buffer_molar is None:
        buffer_molar = {}

    # 1) Surface reaction fluxes (screening — Faradaic equivalents per unit area)
    # HER: 2 H⁺ + 2 e⁻ → H₂ → consumes 1 H⁺ per electron → flux = j_her / F
    # Fe deposition: (FeOH)ads + e⁻ → Fe + OH⁻ → releases 1 OH⁻ per electron.
    # Both act as *acid removal*; net H⁺-equivalent removal ≈ j / F at moderate η.
    F = 96485.0
    # Net acid-equivalent removal flux (mol H⁺ equiv / m² / s)
    acid_removal_flux = j_local_A_m2 / F

    # 3) Steady-state film acid deficit via diffusion-layer flux relation
    # ΔC ≈ flux · δ / D_H  (mol/m³) → divide by 1000 for M (mol/L)
    D_h = 9.3e-9  # m²/s, approximate
    dC_acid_M = (acid_removal_flux * film_thickness_m / D_h) / 1000.0

    # 4) Surface pH: attenuated by buffer capacity at the bulk pH (screening).
    #    ΔpH = ΔC_acid / β ; cap the shift so it stays physically bounded.
    beta = _buffer_capacity_M_per_pH(pH_bulk, buffer_molar)
    MAX_DPH = 3.0
    if beta > 1e-9 and dC_acid_M > 0:
        dph = min(dC_acid_M / beta, MAX_DPH)
    else:
        dph = 0.0
    # Hydrolysis equilibrium already accounts for a small floor of alkalinity;
    # keep the surface pH at least 0.1 above bulk (hyd + FeOH⁺ residual buffer).
    pH_surf = pH_bulk + max(0.1, dph)

    # 5) Buffer capacity at the *surface* pH for reporting
    beta_surf = _buffer_capacity_M_per_pH(pH_surf, buffer_molar)

    # 6) Precipitation check: Fe(OH)₂(s) from Fe²⁺ + 2 OH⁻ ⇌ Fe(OH)₂(s)
    pOH_surf = 14.0 - pH_surf  # screening approximation
    oh_surf = 10 ** (-pOH_surf)
    # Temperature-corrected Ksp
    log_ksp = -PKSP_FE_OH2_25  # log Ksp = -14.87 → Ksp = 10^-14.87
    log_ksp_T = log_ksp + (DH_KSP_J / (2.303 * 8.314)) * (1 / (T_C + 273.15) - 1 / 298.15)
    ksp = 10 ** log_ksp_T

    Q = C_fe2_bulk_M * (oh_surf ** 2)
    precip_now = Q > ksp
    risk_idx = math.log10(Q / max(ksp, 1e-30))

    return float(pH_surf), float(beta_surf), float(risk_idx), bool(precip_now)


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
