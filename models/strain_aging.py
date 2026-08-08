"""Nitrogen from ammoniacal baths → Cottrell strain aging → Lüders bands (V6 §7.1).

Gap / physics / impact / implementation
----------------------------------------
Round 4's ``ammonium_buffer.py`` added NH₄⁺/NH₃ buffering and Fe–ammine
complexation as a boric-free path; Round 5's ``carbon_electrodeposition``
pilots interstitial C into the deposit.  The ammonium route's **free
interstitial nitrogen** channel is nobody's: at cathodic potentials
adsorbed NHₓ decomposes and co-deposits N, and C+N interstitials
classically **strain-age** recrystallised ferrite — Cottrell atmospheres
lock dislocations, the yield point returns, and the sheet Lüders-bands
at the customer's press.  The JMAK/thermomechanical chain can report
\"AISI 1018-like, recrystallised\" while the product is not shippable
deep-drawing stock.

Physics (screening chain)
--------------------------
 ::

     bath → deposit (screening uptake):
       [N]_dep [ppm wt] ≈ k_N · a_NH₃(free) · g(j,FE)  +  baseline
       a_NH₃ from ammonium_buffer.AmmoniumBufferModel(pH,T,Fe,NH₄,tot)
       (or total ammonium proxy when speciation unavailable)

     metal → Cottrell return (Baird / Cottrell–Bilby, n=2/3):
       D_N(T) = D₀·exp(−Q_N/RT)                     (anchored, live T)
       t★(T,C_N) = t_ref·(C_ref/C_N)·exp[Q_N/R·(1/T−1/T_ref)]
                         faster at high T and high N
       yield-point return:
         Δσ_y(t) = Δσ_sat(C_N)·[1−exp(−(t/t★)^{2/3})]
         Δσ_sat = min(σ_cap , s_ppm·C_N)            (linear to ~60 MPa)

     Lüders strain (grain-size scaled screening):
       ε_L [%] ≈ k_L·Δσ_y·√(d/d_ref)                1–3 % at 30–60 MPa
       grain size d from mechanical_properties.estimate_grain_size_um
       or 20 µm fallback

     decision:
       ε_L < 0.5 % and Δσ < 12 MPa → clears
       ε_L < 2.0 % → conditional (1–2 % skin-pass temper rolling, V6 §7.1 lever)
       otherwise → fails  + bake/degas note (N diffuses ≫ slower than H)

Live derivations
----------------
* ``ammonium_buffer.AmmoniumBufferModel`` at call time → free NH₃ activity
  (fallback: total ammonium proxy).
* ``mechanical_properties.estimate_grain_size_um`` → d for ε_L scaling.
* ``hydrogen_trapping.default_trap_hierarchy`` dislocation density proxy
  for t★ extensibility (currently as a comment-grade coupling; ρ enters
  only via the anchored t_ref).
* Anchor fallbacks are explicit so the module stays importable.

Screening flag
--------------
L1.  N uptake coefficient, D₀/Q_N, t_ref, σ-per-ppm, σ-cap and k_L are
anchored screening proxies with band uncertainties (the ammonium Fe–N
uptake literature is thin — flagged SPECULATIVE).  The Cottrell 2/3-law,
Arrhenius and Lüders verdict structure are exact.  Scavenger nitride
formation (TiN/BN/AlN), bake-out kinetics beyond the note, and
specimen-constraint effects are out of scope.

References
----------
*docs/CHEM_PHYS_IMPROVEMENTS_V6.md §7.1*, Cottrell & Bilby (1949) Proc.
Phys. Soc. A 62; Baird strain-aging reviews; Wert–Zener interstitial
diffusion in bcc Fe; ammoniacal iron plating literature.

Anchors: STRAIN_AGING_* family (references §28).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .anchors import get_anchor
from .electrochemistry import R_GAS

SCREENING_FLAG = "unvalidated (L1)"

T_REF_K = 293.15  # 20 °C — t_ref convention
C_REF_PPM = 20.0
D_REF_UM = 20.0


def _a(key: str) -> float:
    return float(get_anchor(key).value)


# ─── N uptake (bath → deposit) ───────────────────────────────────────


def _free_nh3_m(
    total_ammonium_M: float,
    pH: float,
    temperature_C: float,
    total_fe_M: float = 1.0,
) -> float:
    """Live free NH₃ via ammonium_buffer speciation, fallback to proxy."""
    if total_ammonium_M <= 0:
        return 0.0
    try:
        from .ammonium_buffer import AmmoniumBufferModel  # live

        model = AmmoniumBufferModel(temperature_C=temperature_C)
        spec = model.solve_speciation(pH=pH, total_Fe_M=total_fe_M, total_ammonia_M=total_ammonium_M)
        return float(spec.free_nh3_M)
    except Exception:
        # Henderson–Hasselbalch proxy: NH₄⁺ ⇌ NH₃ + H⁺, pKa ~9.25 at 25°C, ΔH~52 kJ
        pka_25 = 9.245
        dh = 52.2e3
        tk = temperature_C + 273.15
        ka = 10 ** (-pka_25) * math.exp(-(dh / R_GAS) * (1.0 / tk - 1.0 / 298.15))
        h = 10 ** (-max(float(pH), 0.0))
        # free fraction = Ka/(Ka+h)
        frac = ka / (ka + h) if (ka + h) > 0 else 0.0
        return float(max(total_ammonium_M * frac, 0.0))


def deposit_nitrogen_ppm(
    total_ammonium_M: float = 0.0,
    pH: float = 5.5,
    temperature_C: float = 60.0,
    total_fe_M: float = 1.0,
    current_density_A_m2: float = 3000.0,
    baseline_ppm: float = 6.0,
) -> float:
    """Screening deposit [N] (ppm wt) from bath ammonia.

    Baseline 5–10 ppm is the boric-route impurity floor; ammoniacal baths
    add k_N·a_NH₃ with a mild current/efficiency lever (higher overpotential
    → more NHₓ decomposition).  The coefficient k_N is the SPECULATIVE band.
    """
    if total_ammonium_M <= 0:
        return float(baseline_ppm)
    free_nh3 = _free_nh3_m(total_ammonium_M, pH, temperature_C, total_fe_M)
    # current lever: +15 % per doubling above 3000 A/m² (screening, weak)
    j_ref = 3000.0
    j_factor = 1.0 + 0.15 * math.log1p(max(current_density_A_m2, 0.0) / j_ref)
    k = _a("STRAIN_AGING_N_PPM_PER_M_NH3")
    ppm = baseline_ppm + k * free_nh3 * j_factor
    return float(max(ppm, baseline_ppm))


# ─── N diffusivity and Cottrell time ──────────────────────────────────


def nitrogen_diffusivity_m2_s(temperature_C: float) -> float:
    """D_N(T) for N in bcc ferrite (Wert–Zener, anchored D₀/Q)."""
    d0 = _a("STRAIN_AGING_D0_N_M2_S")
    q = _a("STRAIN_AGING_Q_N_KJ_MOL") * 1000.0
    tk = max(float(temperature_C) + 273.15, 200.0)
    return float(d0 * math.exp(-q / (R_GAS * tk)))


def cottrell_time_hours(
    c_n_ppm: float,
    temperature_C: float = 20.0,
) -> float:
    """Cottrell half-return time t★(T,C_N) [h].  Smaller → faster aging."""
    t_ref = _a("STRAIN_AGING_TAU_REF_H")
    q = _a("STRAIN_AGING_Q_N_KJ_MOL") * 1000.0
    # Arrhenius factor about T_ref; t scales as 1/D → exp(+Q/R·(1/T−1/T_ref))
    tk = max(float(temperature_C) + 273.15, 200.0)
    arr = math.exp(q / R_GAS * (1.0 / tk - 1.0 / T_REF_K))
    # concentration factor: higher N → more Atmospheres → faster return
    # screened as inverse-linear in C, anchored at C_ref
    c = max(float(c_n_ppm), 1.0)
    c_factor = C_REF_PPM / c
    return float(t_ref * arr * c_factor)


# ─── Δσ and Lüders ─────────────────────────────────────────────────────


def _delta_sigma_sat_mpa(c_n_ppm: float) -> float:
    s_per_ppm = _a("STRAIN_AGING_SIGMA_PER_PPM_MPA")
    cap = _a("STRAIN_AGING_SIGMA_CAP_MPA")
    return float(min(s_per_ppm * max(float(c_n_ppm), 0.0), cap))


def yield_return_mpa(
    c_n_ppm: float,
    aging_hours: float,
    temperature_C: float = 20.0,
    pre_strain_pct: float = 0.0,
) -> float:
    """Yield-point return Δσ_y after storage time t at temperature T.

    Pre-strain (temper rolling) partially pre-locks dislocations, reducing
    the available return: Δσ_eff = Δσ(t)·(1 − pre_strain/2%).  The 2 % scale
    is the standard industrial skin-pass window (screening).
    """
    if aging_hours <= 0 or c_n_ppm <= 0:
        return 0.0
    t_star = cottrell_time_hours(c_n_ppm, temperature_C)
    if t_star <= 0:
        return 0.0
    sat = _delta_sigma_sat_mpa(c_n_ppm)
    # Cottrell–Bilby 2/3 law
    f = 1.0 - math.exp(-((aging_hours / t_star) ** (2.0 / 3.0)))
    d_sigma = sat * f
    if pre_strain_pct > 0:
        # 2 % skin-pass erases return; linear proxy inside that window
        erase = min(max(float(pre_strain_pct), 0.0) / 2.0, 1.0)
        d_sigma *= max(1.0 - erase, 0.0)
    return float(max(d_sigma, 0.0))


def _grain_size_um(
    current_density_A_m2: Optional[float] = None,
    bath_temperature_C: Optional[float] = None,
) -> float:
    """Live grain size via mechanical_properties, fallback 20 µm."""
    try:
        from .mechanical_properties import estimate_grain_size_um  # live

        # estimate_grain_size_um signature varies across repo history;
        # try the two-argument form, fall back to defaults.
        try:
            return float(estimate_grain_size_um(current_density_A_m2 or 3000.0, bath_temperature_C or 60.0))
        except TypeError:
            return float(estimate_grain_size_um())
    except Exception:
        return float(D_REF_UM)


def luders_strain_pct(
    delta_sigma_mpa: float,
    grain_size_um: Optional[float] = None,
    current_density_A_m2: Optional[float] = None,
    bath_temperature_C: Optional[float] = None,
) -> float:
    """Lüders strain [%] scaled by grain size (screening, Hall–Petch family)."""
    if delta_sigma_mpa <= 0:
        return 0.0
    d = grain_size_um if grain_size_um is not None else _grain_size_um(current_density_A_m2, bath_temperature_C)
    k_l = _a("STRAIN_AGING_LUDERS_PCT_PER_MPA")
    # ε_L ∝ Δσ·√(d/d_ref)  — coarser annealed sheet Lüders-bands harder
    scale = math.sqrt(max(float(d), 1.0) / D_REF_UM)
    return float(k_l * float(delta_sigma_mpa) * scale)


@dataclass(frozen=True)
class StrainAgingResult:
    """One storage condition (bath + age + T)."""

    total_ammonium_M: float
    free_nh3_M: float
    c_n_ppm: float
    delta_sigma_sat_mpa: float
    storage_temperature_C: float
    storage_hours: float
    pre_strain_pct: float
    t_star_h: float
    delta_sigma_mpa: float
    grain_size_um: float
    luders_strain_pct: float
    verdict: str  # clears | conditional (skin-pass) | fails
    prescription: str


def evaluate_strain_aging(
    total_ammonium_M: float = 0.0,
    pH: float = 5.5,
    bath_temperature_C: float = 60.0,
    total_fe_M: float = 1.0,
    current_density_A_m2: float = 3000.0,
    storage_hours: float = 24.0,
    storage_temperature_C: float = 20.0,
    pre_strain_pct: float = 0.0,
    grain_size_um: Optional[float] = None,
    c_n_ppm_override: Optional[float] = None,
) -> StrainAgingResult:
    """Evaluate N uptake → Cottrell return → Lüders risk at one condition."""
    if c_n_ppm_override is not None:
        c_n = float(c_n_ppm_override)
        free_nh3 = _free_nh3_m(total_ammonium_M, pH, bath_temperature_C, total_fe_M)
    else:
        free_nh3 = _free_nh3_m(total_ammonium_M, pH, bath_temperature_C, total_fe_M)
        c_n = deposit_nitrogen_ppm(total_ammonium_M, pH, bath_temperature_C, total_fe_M, current_density_A_m2)
    sat = _delta_sigma_sat_mpa(c_n)
    t_star = cottrell_time_hours(c_n, storage_temperature_C)
    d_sigma = yield_return_mpa(c_n, storage_hours, storage_temperature_C, pre_strain_pct)
    d = float(grain_size_um) if grain_size_um is not None else _grain_size_um(current_density_A_m2, bath_temperature_C)
    eps_l = luders_strain_pct(d_sigma, d)
    # verdict thresholds (screening): 0.5 % Lüders or 12 MPa is where
    # automotive press shops start rejecting; 2 % is the skin-pass window.
    if eps_l < 0.5 and d_sigma < 12.0:
        verdict = "clears"
        pres = "no temper rolling required at this [N], storage time and grain size"
    elif eps_l < 2.0 and d_sigma < 30.0:
        verdict = "conditional"
        pres = "1–2 % skin-pass (temper rolling) suppresses YPE; verify on temper mill"
    else:
        verdict = "fails"
        pres = "bake/degas not effective for N (D_N ≪ D_H); reduce [N] via lower NH₃, higher pH, or Ti/B scavenger"
    return StrainAgingResult(
        total_ammonium_M=float(total_ammonium_M),
        free_nh3_M=float(free_nh3),
        c_n_ppm=float(c_n),
        delta_sigma_sat_mpa=float(sat),
        storage_temperature_C=float(storage_temperature_C),
        storage_hours=float(storage_hours),
        pre_strain_pct=float(pre_strain_pct),
        t_star_h=float(t_star),
        delta_sigma_mpa=float(d_sigma),
        grain_size_um=float(d),
        luders_strain_pct=float(eps_l),
        verdict=verdict,
        prescription=pres,
    )


def sweep_ammonium(
    ammonium_list: Sequence[float] = (0.0, 0.5, 1.0, 2.0),
    storage_hours: float = 24.0,
    storage_temperature_C: float = 20.0,
) -> List[Dict[str, float]]:
    """Tabulate [N] and Lüders risk across ammonium levels."""
    rows: List[Dict[str, float]] = []
    for nh4 in ammonium_list:
        r = evaluate_strain_aging(total_ammonium_M=float(nh4), storage_hours=storage_hours, storage_temperature_C=storage_temperature_C)
        rows.append(
            {
                "total_ammonium_M": r.total_ammonium_M,
                "free_nh3_M": r.free_nh3_M,
                "c_n_ppm": r.c_n_ppm,
                "t_star_h": r.t_star_h,
                "delta_sigma_mpa": r.delta_sigma_mpa,
                "luders_pct": r.luders_strain_pct,
                "verdict": r.verdict,
            }
        )
    return rows


def model_scope() -> Dict[str, Any]:
    return {
        "screening_flag": SCREENING_FLAG,
        "live_derivations": [
            "ammonium_buffer.AmmoniumBufferModel(pH,T,Fe,NH₄) → free NH₃ for N-uptake (Henderson proxy fallback)",
            "mechanical_properties.estimate_grain_size_um(j,T) → d for Lüders √d scaling (20 µm fallback)",
            "electrochemistry.R_GAS sets D_N(T) and t★(T) Arrhenius",
        ],
        "screening_proxies_anchored": [
            "N uptake k_N (ppm per M free NH₃, SPECULATIVE, decade band)",
            "N diffusivity D₀ and Q_N in bcc Fe (Wert–Zener, 60–85 kJ/mol band)",
            "Cottrell reference return time t_ref at 20 °C, 20 ppm",
            "yield-point saturation σ_per_ppm and σ_cap (Cottrell atmosphere limit)",
            "Lüders strain per MPa (grain-size scaled, 0.5 %/2 % verdict thresholds)",
        ],
        "out_of_scope": [
            "ammine electrochemistry and NHₓ decomposition kinetics at the cathode",
            "Ti/B/AlN scavenger nitride precipitation (requires impurity codeposition plus nitride thermodynamics)",
            "bake-out kinetics for N (D_N ≪ D_H — noted but not sized)",
            "constraint and specimen-geometry effects on Lüders band nucleation",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> None:  # pragma: no cover
    p = argparse.ArgumentParser(description="Strain aging (N + Cottrell → Lüders) screen (V6 §7.1).")
    p.add_argument("--ammonium", type=float, default=1.0, help="total ammonium salt [M]")
    p.add_argument("--pH", type=float, default=5.5)
    p.add_argument("--storage-hours", type=float, default=24.0)
    p.add_argument("--storage-temp", type=float, default=20.0)
    p.add_argument("--pre-strain", type=float, default=0.0, help="skin-pass pre-strain [%]")
    p.add_argument("--c-n-ppm", type=float, default=None, help="override deposit [N] ppm")
    args = p.parse_args(argv)

    r = evaluate_strain_aging(
        total_ammonium_M=args.ammonium,
        pH=args.pH,
        storage_hours=args.storage_hours,
        storage_temperature_C=args.storage_temp,
        pre_strain_pct=args.pre_strain,
        c_n_ppm_override=args.c_n_ppm,
    )
    print(f"strain_aging — N-interstitial Cottrell → Lüders  [{SCREENING_FLAG}]")
    print(f"  bath  NH₄,tot {r.total_ammonium_M:g} M → free NH₃ {r.free_nh3_M:.2e} M → [N]_dep {r.c_n_ppm:.1f} ppm  (sat Δσ {r.delta_sigma_sat_mpa:.0f} MPa)")
    print(f"  D_N({r.storage_temperature_C:.0f}°C) {nitrogen_diffusivity_m2_s(r.storage_temperature_C):.2e} m²/s  t★ {r.t_star_h:.1f} h  grain {r.grain_size_um:.1f} µm")
    print(f"  storage {r.storage_hours:.0f} h @ {r.storage_temperature_C:.0f} °C  pre-strain {r.pre_strain_pct:.1f} %  → Δσ_y {r.delta_sigma_mpa:.1f} MPa  ε_L {r.luders_strain_pct:.2f} %")
    print(f"  verdict {r.verdict}  — {r.prescription}")
    print()
    print("  sweep vs ammonium (24 h @ 20 °C):")
    for row in sweep_ammonium(storage_hours=24.0):
        print(f"    NH₄ {row['total_ammonium_M']:4.1f} M  [N] {row['c_n_ppm']:5.1f} ppm  Δσ {row['delta_sigma_mpa']:4.1f} MPa  ε_L {row['luders_pct']:4.2f} %  {row['verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
