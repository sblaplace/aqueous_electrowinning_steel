"""
Tramp-element (Cu, Sn) surface hot-shortness ceiling for recycled steel feed.

Why this module exists
----------------------
``impurity_codeposition.py`` / ``purification.py`` treat Cu and Sn as bath trace
impurities, but nothing models their **effect in the finished steel** — the real
ceiling for recycled/waste feedstocks (pickle liquor, steel-mill dust,
scrap-adjacent feeds), which the program targets as its beachhead.

The physics (Round 5, E1): on reheating for hot rolling, tramp Cu (with Sn)
segregates to the steel/scale interface; above the Cu–Fe eutectic (~1094 °C)
the Cu-rich phase is *liquid* and wets/penetrates austenite grain boundaries,
causing **surface hot-shortness** (alligatoring / edge cracking) in the rolled
sheet. Sn lowers the effective melting temperature and worsens the effect.

This module computes a Cu/Sn surface-enrichment index, a hot-shortness risk,
and a rolling-temperature ceiling + allowable residual Cu+Sn for a target grade.

Screening flag
--------------
L1. Enrichment factors and risk thresholds are screening proxies for the
literature (Seetharaman; the classic Cu/Sn hot-shortness work); calibrate
against trial-rolling data on electrolytic-Fe + tramp feeds.

References
----------
* Classic tramp-element hot-shortness literature on Cu/Sn segregation during
  reheating (e.g. review in "Recycled steel: hot shortness" — Seetharaman et al.).
* Cu–Fe eutectic ~1094 °C; Sn addition lowers the liquidus and widens the
  hot-shortness temperature window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

SCREENING_FLAG = "unvalidated (L1)"

# Cu–Fe eutectic temperature (C); above this, a Cu-rich liquid phase can form
# at austenite grain boundaries.
CU_FE_EUTECTIC_C = 1094.0


@dataclass
class HotShortnessParams:
    """Screening parameters for tramp Cu/Sn surface hot-shortness."""

    # Surface enrichment factor during scale formation / reheating: the Cu/Sn
    # concentration at the metal/scale interface is higher than bulk because
    # iron oxidizes away preferentially (Cu/Sn do not oxidize as readily).
    cu_enrichment_factor: float = 8.0
    sn_enrichment_factor: float = 5.0
    # Effective melting-point depression by Sn (C per wt% Sn) — lowers the
    # liquidus of the Cu-rich phase.
    sn_liquidus_depression_C_per_wt: float = 30.0
    # Reference bulk residual (wt%) and risk exponents.
    cu_risk_ref_wt: float = 0.25
    cu_risk_exp: float = 1.5
    sn_risk_ref_wt: float = 0.05
    sn_risk_exp: float = 1.2
    # Rolling-temperature reference for a given grade (C); hot-shortness risk
    # scales with how far above the (Sn-depressed) liquidus you roll.
    roll_temp_ref_C: float = 1150.0
    # Allowable residual Cu+Sn ceiling for "clean" deep-drawing grades (wt%).
    allowable_residual_deep_draw_wt: float = 0.10


def cu_sn_surface_enrichment(
    cu_wt_percent: float,
    sn_wt_percent: float,
    params: Optional[HotShortnessParams] = None,
) -> dict:
    """Cu/Sn surface enrichment during reheating / scale formation (wt%)."""
    p = params or HotShortnessParams()
    cu = max(float(cu_wt_percent), 0.0)
    sn = max(float(sn_wt_percent), 0.0)
    cu_surf = cu * p.cu_enrichment_factor
    sn_surf = sn * p.sn_enrichment_factor
    return {
        "cu_surface_wt": cu_surf,
        "sn_surface_wt": sn_surf,
        "cu_sn_surface_wt": cu_surf + sn_surf,
    }


def effective_liquidus_C(
    sn_surface_wt_percent: float,
    params: Optional[HotShortnessParams] = None,
) -> float:
    """Effective Cu-rich liquidus temperature, depressed by surface Sn."""
    p = params or HotShortnessParams()
    return max(CU_FE_EUTECTIC_C - p.sn_liquidus_depression_C_per_wt * max(sn_surface_wt_percent, 0.0),
               CU_FE_EUTECTIC_C - 200.0)


def hot_shortness_risk(
    cu_wt_percent: float,
    sn_wt_percent: float,
    roll_temperature_C: float = 1150.0,
    params: Optional[HotShortnessParams] = None,
) -> dict:
    """
    Surface hot-shortness risk for hot rolling of a tramp-bearing feed.

    Returns
    -------
    dict with surface enrichments, effective liquidus, a risk index (0..1),
    a recommended rolling-temperature ceiling, and an eligibility verdict for
    deep-drawing / clean grades.
    """
    p = params or HotShortnessParams()
    enrich = cu_sn_surface_enrichment(cu_wt_percent, sn_wt_percent, p)
    cu_surf = enrich["cu_surface_wt"]
    sn_surf = enrich["sn_surface_wt"]
    liquidus = effective_liquidus_C(sn_surf, p)

    # Risk rises with surface Cu and Sn, and with rolling above the liquidus.
    risk_cu = (cu_surf / p.cu_risk_ref_wt) ** p.cu_risk_exp
    risk_sn = (sn_surf / p.sn_risk_ref_wt) ** p.sn_risk_exp
    over_liq = max(roll_temperature_C - liquidus, 0.0)
    risk_temp = 1.0 + over_liq / 300.0
    risk = min(risk_cu * risk_sn * risk_temp, 1.0)

    # Rolling ceiling: max roll temp that keeps the Cu-rich phase solid-ish.
    roll_ceiling_C = liquidus - 50.0  # safety margin below the liquidus

    residual = cu_wt_percent + sn_wt_percent
    eligible_deep_draw = residual <= p.allowable_residual_deep_draw_wt

    return {
        **enrich,
        "effective_liquidus_C": float(liquidus),
        "risk_index": float(risk),
        "roll_temp_ceiling_C": float(roll_ceiling_C),
        "rolling_below_ceiling": bool(roll_temperature_C <= roll_ceiling_C),
        "residual_cu_sn_wt": float(residual),
        "eligible_deep_draw": bool(eligible_deep_draw),
    }


def main() -> None:
    """CLI entrypoint for tramp-element hot-shortness analysis."""
    print("=" * 70)
    print(" Tramp Cu/Sn Hot-Shortness Ceiling (Round 5, E1)")
    print("=" * 70)
    print(f" Screening flag : {SCREENING_FLAG}")
    print(f" Cu-Fe eutectic : {CU_FE_EUTECTIC_C:.0f} °C")
    for cu, sn in ((0.02, 0.005), (0.10, 0.02), (0.35, 0.10)):
        res = hot_shortness_risk(cu, sn, roll_temperature_C=1150.0)
        print(f"  Cu={cu:.3f} Sn={sn:.3f} -> liquidus={res['effective_liquidus_C']:.0f}°C "
              f"risk={res['risk_index']:.2f} ceiling={res['roll_temp_ceiling_C']:.0f}°C "
              f"deep-draw={'yes' if res['eligible_deep_draw'] else 'no'}")


if __name__ == "__main__":
    main()
