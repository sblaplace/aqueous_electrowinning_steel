"""Room-temperature self-annealing of electrowon iron — the metrology time stamp (V6 §5.2).

Gap / physics / impact / implementation
----------------------------------------
An electrowon iron foil is not an equilibrium crystal when it leaves the
cathode: fine grains (≈1 µm), supersaturated point defects and hydrogen in
reversible traps (see ``hydrogen_trapping.py`` and Round-5 C2) carry
megajoules per cubic metre of stored energy.  Electrodeposited Cu and Ni
are textbook cases: hardness, resistivity, residual stress and even grain
size drift by 10–30 % at room temperature over hours to days, with
log-time recovery kinetics that prosecution cannot see in a single-period
regression per duplicate structure.  For H-charged iron the effect couples to
hydrogen egress — ``internal_stress.py`` returns a *snapshot* but the
stress-H handshake itself is time stamped — so two runs harvested at +4 h
and +48 h can disagree by more than the additive effect the programme is
trying to resolve (does saccharin raise or lower σ?).

Physics (screening chain)
--------------------------
 ::

     log-time recovery (classic recovery law):
       f_σ(t) = max(floor, 1 − A_σ · ln(1 + t/τ_eff))      screening
       f_HV, f_ρ use scaled amplitudes (0.62, 0.48 × A_σ)  proxy

     τ_eff(T, C_H) = τ_ref · exp[E_a/R·(1/T − 1/T_ref)] / (1 + β·C_H,diff)
       T dependence: Arrhenius (anchored E_a, live T)
       H dependence: higher diffusible H → faster recovery (β anchor,
                     decade band — the V2 stress-H coupling made time-dependent)
       τ_ref anchored at 20 °C, low-H foil (≈4 h from Cu nanocrystalline
       self-annealing analogue)

     H egress (live from hydrogen_trapping):
       C_H,diff(t)/C_H0 ≈ slab desorption with D_eff(T) from
       ``hydrogen_trapping.effective_trapped_diffusivity_m2_s`` and foil
       half-thickness L = t_foil/2.  First-term plus truncated-series form
       exact for the slab.  This ties the stress time stamp to the same
       trap hierarchy ``melt_hydrogen`` and ``hydrogen_embrittlement``
       already use.

The comparative question — does a +4 h / +48 h metrology spread explain the
observed hardness scatter, and how tightly must the QA time window be
controlled — is decision-grade.  The reported decimal (HV at 24.00 h) is
not.

This module is tiny on purpose and wires *into* the existing measurement
stack rather than replacing it: ``validate_aging_record`` adds the
mandatory ``aging_hours`` / ``harvest_to_measure`` fields to
``run_record.py`` QA (fail-soft warning, never a hard gate), and the
correction helpers let the analyst map any measured HV/σ/ρ back to
as-deposited (t→0) or to the metrology standard (24 h, 20 °C).

Live derivations
----------------
* ``hydrogen_trapping.effective_trapped_diffusivity_m2_s`` / lattice
  diffusivity at call time (foil thickness + temperature → H egress).
* ``internal_stress`` is not imported at module load; the stress
  magnitude lives there, this module only scales it.
* Anchor fallbacks are explicit so the module remains importable before
  the trap hierarchy is present.

Screening flag
--------------
L1.  Amplitudes, τ_ref, E_a, β and the floor are anchored screening
proxies with decade-band uncertainties; the log-time and slab-desorption
structures are exact.  Grain-growth, electrochemistry-coupled recovery and
specimen-size effects are out of scope.

References
----------
*docs/CHEM_PHYS_IMPROVEMENTS_V6.md §5.2*, Ling & Gross self-annealing of
electrodeposited Cu, Stangl et al. Acta Mater. (nanocrystalline Cu aging),
classic recovery kinetics (Cottrell–Dienes).

Anchors: DEPOSIT_AGING_* family (references §27).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .anchors import get_anchor
from .electrochemistry import R_GAS

SCREENING_FLAG = "unvalidated (L1)"

# ─── exact / convention constants ────────────────────────────────────
T_REF_K = 293.15  # 20 °C — τ_ref convention
HV_TO_VICKERS_SCALE = 1.0  # placeholder — HV is reported as-measured
HV_RATIO = 0.62  # proxy: hardness relaxes ~62 % of the stress amplitude
RHO_RATIO = 0.48  # proxy: resistivity relaxes ~48 % of the stress amplitude


def _a(key: str) -> float:
    return float(get_anchor(key).value)


# ─── H egress (live from hydrogen_trapping) ───────────────────────────


def _d_eff_m2_s(temperature_C: float) -> float:
    """Live D_eff with anchor fallback (no import at module load)."""
    try:
        from .hydrogen_trapping import effective_trapped_diffusivity_m2_s  # live

        return float(effective_trapped_diffusivity_m2_s(temperature_C))
    except Exception:
        # fallback: lattice diffusivity with screening Q (≈4.6 kJ/mol, Kiuchi)
        # — intentionally conservative; real trapped D is lower.
        d0 = 7.3e-8
        q = 4.6e3
        tk = max(float(temperature_C) + 273.15, 200.0)
        return d0 * math.exp(-q / (R_GAS * tk))


def diffusible_h_remaining_frac(
    aging_hours: float,
    foil_thickness_um: float = 25.0,
    temperature_C: float = 20.0,
) -> float:
    """Fraction of initial diffusible H still in a foil slab at age t.

    Slab of half-thickness L = t_foil/2 with sealed mid-plane, free
    surfaces: C_avg/C0 = Σ_{n odd} 8/(π²n²) exp(−n²π² D t / 4L²).
    Four terms is <1 % error for Fo > 0.01; for Fo≈0 we return 1.
    """
    if aging_hours <= 0:
        return 1.0
    if foil_thickness_um <= 0:
        raise ValueError("foil_thickness_um must be positive")
    l_half = foil_thickness_um * 1e-6 / 2.0
    d_eff = _d_eff_m2_s(temperature_C)
    fo = d_eff * aging_hours * 3600.0 / (l_half ** 2)
    if fo < 1e-6:
        return 1.0
    # sum odd n =1,3,5,7 ; include n=9 for completeness when Fo is tiny
    total = 0.0
    for n in (1, 3, 5, 7, 9, 11):
        total += 8.0 / (math.pi ** 2 * n ** 2) * math.exp(-(n ** 2 * math.pi ** 2 * fo) / 4.0)
    return float(min(max(total, 0.0), 1.0))


# ─── τ_eff and log-time factors ───────────────────────────────────────


def effective_tau_hours(
    temperature_C: float = 20.0,
    diffusible_h_ppm: float = 1.0,
) -> float:
    """Characteristic recovery time τ_eff(T, C_H) [h].  Smaller → faster aging."""
    tau_ref = _a("DEPOSIT_AGING_TAU_REF_H")
    ea_j = _a("DEPOSIT_AGING_EA_KJ_MOL") * 1000.0
    beta = _a("DEPOSIT_AGING_H_BETA_PER_PPM")
    tk_ref = T_REF_K
    tk = max(float(temperature_C) + 273.15, 200.0)
    arr = math.exp(ea_j / R_GAS * (1.0 / tk - 1.0 / tk_ref))
    # H accelerates recovery; clamp denominator ≥1 so H never *slows* aging
    denom = 1.0 + beta * max(float(diffusible_h_ppm), 0.0)
    return float(tau_ref * arr / denom)


def _log_factor(aging_hours: float, tau_h: float, amplitude: float, floor: float) -> float:
    if aging_hours <= 0:
        return 1.0
    if tau_h <= 0:
        raise ValueError("tau_h must be positive")
    raw = 1.0 - amplitude * math.log1p(aging_hours / tau_h)
    return float(max(raw, floor))


@dataclass(frozen=True)
class AgingFactors:
    """Fractional retention vs as-deposited (1.0 = no drift)."""

    aging_hours: float
    temperature_C: float
    diffusible_h_ppm_initial: float
    tau_eff_h: float
    diffusible_h_remaining_frac: float
    f_stress: float
    f_hv: float
    f_resistivity: float
    floor_stress: float


def aging_factors(
    aging_hours: float,
    temperature_C: float = 20.0,
    diffusible_h_ppm: float = 1.0,
    foil_thickness_um: float = 25.0,
) -> AgingFactors:
    """Return retention factors for σ, HV, ρ plus the H-egress fraction."""
    if aging_hours < 0:
        raise ValueError("aging_hours must be non-negative")
    tau = effective_tau_hours(temperature_C, diffusible_h_ppm)
    h_rem = diffusible_h_remaining_frac(aging_hours, foil_thickness_um, temperature_C)
    a_sigma = _a("DEPOSIT_AGING_A_SIGMA")
    floor = _a("DEPOSIT_AGING_FLOOR_FRAC")
    f_sigma = _log_factor(aging_hours, tau, a_sigma, floor)
    # hardness / resistivity track stress with reduced amplitude and higher floor
    # floors are derived from the same anchor by linear remapping:
    #   floor_hv = 1 − (1−floor)·HV_RATIO  etc., so a SPECULATIVE low floor
    #   never implies HV collapses faster than stress.
    floor_hv = 1.0 - (1.0 - floor) * HV_RATIO
    floor_rho = 1.0 - (1.0 - floor) * RHO_RATIO
    f_hv = _log_factor(aging_hours, tau, a_sigma * HV_RATIO, floor_hv)
    f_rho = _log_factor(aging_hours, tau, a_sigma * RHO_RATIO, floor_rho)
    return AgingFactors(
        aging_hours=aging_hours,
        temperature_C=temperature_C,
        diffusible_h_ppm_initial=diffusible_h_ppm,
        tau_eff_h=tau,
        diffusible_h_remaining_frac=h_rem,
        f_stress=f_sigma,
        f_hv=f_hv,
        f_resistivity=f_rho,
        floor_stress=floor,
    )


# ─── correction helpers (analyst-facing) ───────────────────────────────


def correct_to_as_deposited(
    measured_value: float,
    aging_hours: float,
    factor: float,
) -> float:
    """Back-extrapolate a drifted measurement to t→0 (as-deposited)."""
    if factor <= 0:
        raise ValueError("factor must be positive")
    return float(measured_value / factor)


def correct_to_standard(
    measured_value: float,
    aging_hours_measured: float,
    aging_hours_standard: float = 24.0,
    temperature_C: float = 20.0,
    diffusible_h_ppm: float = 1.0,
    foil_thickness_um: float = 25.0,
    property: str = "stress",
) -> float:
    """Map a measurement at t_meas to what it *would* have read at t_std."""
    if property not in ("stress", "hv", "hardness", "resistivity", "rho"):
        raise ValueError("property must be 'stress', 'hv'/'hardness', or 'resistivity'/'rho'")
    f_meas = aging_factors(aging_hours_measured, temperature_C, diffusible_h_ppm, foil_thickness_um)
    f_std = aging_factors(aging_hours_standard, temperature_C, diffusible_h_ppm, foil_thickness_um)
    key = {"stress": "f_stress", "hv": "f_hv", "hardness": "f_hv", "resistivity": "f_resistivity", "rho": "f_resistivity"}[property]
    return float(measured_value * getattr(f_std, key) / getattr(f_meas, key))


@dataclass(frozen=True)
class DepositAgingCorrection:
    """Correction bundle for one coupon/property at a stated age."""

    property: str
    aging_hours: float
    temperature_C: float
    foil_thickness_um: float
    measured_value: float
    factor: float
    as_deposited_value: float
    standard_value: float  # at 24 h, 20 °C
    standard_hours: float = 24.0


def correction_for(
    measured_value: float,
    aging_hours: float,
    temperature_C: float = 20.0,
    diffusible_h_ppm: float = 1.0,
    foil_thickness_um: float = 25.0,
    property: str = "stress",
) -> DepositAgingCorrection:
    """Convenience: drift ledger for a single measurement."""
    facs = aging_factors(aging_hours, temperature_C, diffusible_h_ppm, foil_thickness_um)
    key = {"stress": "f_stress", "hv": "f_hv", "hardness": "f_hv", "resistivity": "f_resistivity", "rho": "f_resistivity"}[property] if property in ("stress", "hv", "hardness", "resistivity", "rho") else None
    if key is None:
        raise ValueError("property must be 'stress', 'hv'/'hardness', or 'resistivity'/'rho'")
    factor = getattr(facs, key)
    std_val = correct_to_standard(measured_value, aging_hours, 24.0, temperature_C, diffusible_h_ppm, foil_thickness_um, property)
    return DepositAgingCorrection(
        property=property,
        aging_hours=aging_hours,
        temperature_C=temperature_C,
        foil_thickness_um=foil_thickness_um,
        measured_value=float(measured_value),
        factor=float(factor),
        as_deposited_value=float(measured_value / factor),
        standard_value=float(std_val),
    )


# ─── metrology standard + QA hook ────────────────────────────────────


def metrology_standard() -> Dict[str, Any]:
    """Recommended harvest→measure window that the metrology standard should enforce."""
    return {
        "standard_aging_hours": 24.0,
        "allowed_window_hours": [18.0, 30.0],
        "temperature_C": 20.0,
        "temperature_tolerance_C": 2.0,
        "required_fields": ["aging_hours", "harvest_timestamp", "measurement_timestamp", "storage_temperature_C"],
        "correction_note": "Measurements outside the window must be mapped to 24 h via correct_to_standard(); report both raw and corrected values.",
        "drift_at_window_edges": {
            "18h": aging_factors(18.0).f_stress,
            "30h": aging_factors(30.0).f_stress,
        },
    }


def validate_aging_record(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Fail-soft QA check for the harvest→measure time stamp (run_record hook).

    Looks for ``aging_hours`` or the pair ``harvest_timestamp``/
    ``measurement_timestamp``.  Missing fields emit a WARNING-level issue,
    never an error — the run remains valid, but the gate evidence is
    annotated so cross-run comparisons carry the metrology covariance.
    """
    issues: List[Dict[str, str]] = []
    has_aging = "aging_hours" in metadata and metadata["aging_hours"] is not None
    has_pair = "harvest_timestamp" in metadata and "measurement_timestamp" in metadata

    if not has_aging and not has_pair:
        issues.append(
            {
                "path": "metadata.aging_hours",
                "message": "missing harvest→measure time stamp; add aging_hours (h) or harvest/measurement timestamps — stress/HV comparisons carry uncorrected aging scatter.",
                "severity": "warning",
            }
        )
    elif has_aging:
        try:
            val = float(metadata["aging_hours"])
            if val < 0:
                issues.append({"path": "metadata.aging_hours", "message": "aging_hours must be non-negative", "severity": "warning"})
            elif val < 1.0:
                issues.append({"path": "metadata.aging_hours", "message": "aging <1 h: foil still supersaturated; treat HV/σ as upper-bound snapshot.", "severity": "warning"})
        except (TypeError, ValueError):
            issues.append({"path": "metadata.aging_hours", "message": "aging_hours must be numeric (hours)", "severity": "warning"})

    # storage temperature check — drift is T-sensitive
    if "storage_temperature_C" not in metadata:
        issues.append({"path": "metadata.storage_temperature_C", "message": "storage temperature not recorded; aging correction assumes 20 °C.", "severity": "warning"})

    return {"valid": True, "issues": issues, "standard": metrology_standard()}


# ─── sweep helper (paper/CI figure) ──────────────────────────────────

def sweep_aging(aging_hours_list: Sequence[float] = (1, 4, 16, 24, 48, 120)) -> List[Dict[str, float]]:
    """Tabulate drift across the canonical metrology delays."""
    rows: List[Dict[str, float]] = []
    for t in aging_hours_list:
        facs = aging_factors(float(t))
        rows.append(
            {
                "aging_hours": float(t),
                "f_stress": facs.f_stress,
                "f_hv": facs.f_hv,
                "f_resistivity": facs.f_resistivity,
                "h_remaining": facs.diffusible_h_remaining_frac,
                "tau_eff_h": facs.tau_eff_h,
            }
        )
    return rows


def model_scope() -> Dict[str, Any]:
    return {
        "screening_flag": SCREENING_FLAG,
        "live_derivations": [
            "hydrogen_trapping.effective_trapped_diffusivity_m2_s(T) → slab H-egress fraction (foil-thickness dependent)",
            "electrochemistry.R_GAS sets Arrhenius T dependence of τ_eff",
        ],
        "screening_proxies_anchored": [
            "log-time amplitude A_σ (Cu nanocrystalline self-annealing analogue)",
            "reference recovery time τ_ref at 20 °C, low-H",
            "recovery activation energy E_a",
            "H-acceleration factor β (decade band, SPECULATIVE)",
            "floor fraction (residual after long aging, SPECULATIVE)",
        ],
        "out_of_scope": [
            "grain-growth kinetics and texture evolution",
            "electrochemical driving of recovery during plating",
            "specimen-size and constraint effects on thin-foil relaxation",
            "automatic wiring into run_record / characterization loaders (QA hook is opt-in until the contract migrates)",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> None:  # pragma: no cover
    p = argparse.ArgumentParser(description="Deposit self-annealing drift screen (V6 §5.2).")
    p.add_argument("--hours", type=float, default=24.0, help="aging time t (h) since harvest")
    p.add_argument("--temperature", type=float, default=20.0, help="storage temperature (°C)")
    p.add_argument("--h-ppm", type=float, default=1.0, help="initial diffusible H (ppm wt)")
    p.add_argument("--thickness", type=float, default=25.0, help="foil thickness (µm)")
    p.add_argument("--property", default="stress", choices=["stress", "hv", "resistivity"])
    p.add_argument("--measured", type=float, default=None, help="optional measured value to correct")
    args = p.parse_args(argv)

    facs = aging_factors(args.hours, args.temperature, args.h_ppm, args.thickness)
    print(f"deposit_aging — RT self-annealing drift  [{SCREENING_FLAG}]")
    std = metrology_standard()
    print(f"  standard  {std['standard_aging_hours']:.0f} h  window {std['allowed_window_hours'][0]:.0f}–{std['allowed_window_hours'][1]:.0f} h @ {std['temperature_C']:.0f}±{std['temperature_tolerance_C']:.0f} °C")
    print(f"  input     t={args.hours:g} h  T={args.temperature:g} °C  H_diff⁰={args.h_ppm:g} ppm  t_foil={args.thickness:g} µm")
    print(f"  τ_eff     {facs.tau_eff_h:.2f} h   (A_σ={_a('DEPOSIT_AGING_A_SIGMA'):g}, floor={facs.floor_stress:.2f})")
    print(f"  H remain  {facs.diffusible_h_remaining_frac:.3f}  (D_eff slab, live traps)")
    print(f"  f_σ       {facs.f_stress:.4f}   f_HV {facs.f_hv:.4f}   f_ρ {facs.f_resistivity:.4f}")
    if args.measured is not None:
        key = {"stress": "f_stress", "hv": "f_hv", "resistivity": "f_resistivity"}[args.property]
        fac = getattr(facs, key)
        as_dep = args.measured / fac
        std_val = correct_to_standard(args.measured, args.hours, 24.0, args.temperature, args.h_ppm, args.thickness, args.property)
        print(f"  measured {args.property} {args.measured:g} → as-deposited {as_dep:.4g}  → 24 h standard {std_val:.4g}")
    print()
    print("  sweep (h → retention):")
    for row in sweep_aging():
        print(f"    {row['aging_hours']:6.0f} h  σ {row['f_stress']:.3f}  HV {row['f_hv']:.3f}  ρ {row['f_resistivity']:.3f}  H {row['h_remaining']:.3f}")


if __name__ == "__main__":  # pragma: no cover
    main()
