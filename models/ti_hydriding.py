"""Titanium drum hydriding — the reductive counterpart to oxide passivation (V6 §4.1).

Why this module exists
----------------------
``substrate_passivation.py`` models the **oxidative** drum-life mechanism
(TiO₂ parabolic growth lowers G_c until "peel fails at hour 800, not hour
8").  But a drum in service is a *cathode*.  At the hydrogen-adjacent
potentials of an iron cathode in acid, titanium takes up hydrogen, and the
Ti–H system is unforgiving: solid-solution α-Ti crosses a terminal solid
solubility (TSS, anchored; ≲100 wt-ppm at service temperatures) into
δ-TiH₍₂₋ₓ₎ with ~20–25 % atomic-volume expansion.  The classic
copper-foil-drum service experience is that hydriding sets drum
re-skinning intervals — the coupon peels fine on day one and the drum is
dead in month two.  Nothing in the repository priced that clock.

The physics (screening chain)
-----------------------------
::

    HER partial current (j_op × (1−FE), LIVE from the architecture screen)
      → absorbed H flux  J = k_entry × f_shield × j_HER / F
        k_entry : fraction of cathodic H that enters the metal (anchor,
                  decade band — cathodic-charging literature)
        f_shield: fraction of drum area that actually sees bath H
                  (pinholes, strip edges, the peel front); the iron deposit
                  is itself an H sink and shields the rest (anchor)
      → diffusion depth x_d = √(D_H·t), D_H Arrhenius in α-Ti (anchor)
      → dissolved inventory ≤ TSS × x_d (terminal solid solubility, anchor)
      → EXCESS inventory precipitates δ-hydride in a near-surface *case*
          δ_case = N_excess / (c_hyd − c_TSS),  TiH~1.7 (anchor)
      → scale spall when δ_case ≥ δ_crit (anchor, foil-drum practice)
      → spalled/patchy oxide multiplies substrate_passivation's G_c drift
        (this module's feed_to_adhesion_peel)

Two honesty notes.  (1) The elastic-energy number behind a spall
criterion is formally enormous (a constrained ~7 % linear strain is GPa)
and useless: real case failures involve plasticity, hydride fracture
(K_IC ~ 1–2 MPa√m, noted but not modeled), and oxide buckling; the module
therefore uses the service-practice critical case depth and says so.
(2) k_entry × f_shield is the least-constrained product in the chain
(three-decade band); the module reports the verdict **across that band**
and returns the *design target* — keep the product below the value that
makes campaign life exceed the reskin budget — rather than pretending a
central estimate is validated.

Decision relevance
------------------
``cell_architecture``'s drum-and-strip route is the only continuous
coherent-foil path in the architecture screen, and its gating unknown is
already "does iron peel from titanium?" (`adhesion_peel`, coupon test
pending).  This module makes that answer **time-dependent**: it emits the
campaign-life hours at which hydriding would kill the peel window even if
the day-one coupon clears, the G_c drift multiplier vs campaign hours for
``adhesion_peel.py``, and a campaign ledger row for ``closed_loop.py``.

Anchors: Ti–H reviews (McQuillan-classic Sieverts data; Patton & Zur
Megede drum-service reviews), foil-machine drum practice (references §25).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from . import cell_architecture
from .anchors import get_anchor
from .electrochemistry import FARADAY

SCREENING_FLAG = "unvalidated (L1)"

# ─── Exact physical constants (module-level, like electrochemistry.M_FE) ──
M_TI = 47.867e-3           # kg/mol — titanium molar mass
M_H_G = 1.008              # g/mol — hydrogen molar mass
RHO_TI_KG_M3 = 4510.0      # kg/m³ — CP titanium density (grade 2)
RHO_TIH2_KG_M3 = 3900.0    # kg/m³ — δ-TiH₂−x density (approx)

SECONDS_PER_HOUR = 3600.0

# Verdict vocabulary (matches the rest of the program's screening modules)
VERDICT_CLEARS = "clears"
VERDICT_CONDITIONAL = "conditional"
VERDICT_FAILS = "fails"


# ─── Anchor accessors (every literature number lives in anchors.py) ───────

def _a(key: str) -> float:
    return get_anchor(key).value


def h_diffusivity_m2_s(T_C: float) -> float:
    """H diffusivity in α-Ti (m²/s), Arrhenius about the 60 °C anchor."""
    D_ref = _a("TI_H_D_60C_M2_S")
    Ea = _a("TI_H_D_EA_J_MOL")
    T = T_C + 273.15
    T_ref = 60.0 + 273.15
    return D_ref * math.exp(-Ea / 8.3144626181532 * (1.0 / T - 1.0 / T_ref))


def tss_h_mol_m3() -> float:
    """Terminal solid solubility of H in α-Ti (mol H/m³ Ti), from wt-ppm."""
    wppm = _a("TI_H_TSS_WT_PPM_60C")           # g H per 1e6 g Ti
    g_h_per_m3 = RHO_TI_KG_M3 * 1000.0 * wppm * 1e-6
    return g_h_per_m3 / M_H_G


def hydride_h_mol_m3() -> float:
    """H inventory of the δ-hydride case TiH~1.7 (mol H/m³ hydride)."""
    x_over_ti = _a("TI_HYD_H_PER_TI")          # H/Ti atomic ratio ≈ 1.7
    formula_mass_g = M_TI * 1000.0 + x_over_ti * M_H_G
    mol_formula_per_m3 = RHO_TIH2_KG_M3 * 1000.0 / formula_mass_g
    return mol_formula_per_m3 * x_over_ti


def her_partial_current_A_m2(j_op_A_m2: float, faradaic_efficiency: float) -> float:
    """Parasitic HER current density on the drum (A/m²)."""
    return j_op_A_m2 * max(0.0, 1.0 - faradaic_efficiency)


def absorbed_h_flux_mol_m2_s(
    j_op_A_m2: float,
    faradaic_efficiency: float,
    entry_frac: Optional[float] = None,
    shield_frac: Optional[float] = None,
) -> float:
    """H flux actually absorbed into the Ti drum (mol/m²/s).

    J = k_entry × f_shield × j_HER / F.  Both factors are anchored with
    wide bands — they are the least-constrained link in the chain, and the
    caller-facing APIs expose them explicitly for exactly that reason.
    """
    k_entry = _a("TI_H_ENTRY_FRAC") if entry_frac is None else entry_frac
    f_shield = _a("TI_H_SHIELD_FRAC") if shield_frac is None else shield_frac
    return k_entry * f_shield * her_partial_current_A_m2(j_op_A_m2, faradaic_efficiency) / FARADAY


def diffusion_depth_m(t_hours: float, T_C: float) -> float:
    """Classical penetration depth √(D·t) of H into the drum (m)."""
    return math.sqrt(h_diffusivity_m2_s(T_C) * t_hours * SECONDS_PER_HOUR)


def excess_inventory_mol_m2(
    t_hours: float,
    j_op_A_m2: float,
    faradaic_efficiency: float,
    T_C: float,
    entry_frac: Optional[float] = None,
    shield_frac: Optional[float] = None,
) -> float:
    """Absorbed H minus what the α-case can dissolve to the diffusion depth.

    Positive excess → δ-hydride must exist in the near-surface case.  The
    comparison is the physical onset condition for hydriding damage.
    """
    absorbed = absorbed_h_flux_mol_m2_s(
        j_op_A_m2, faradaic_efficiency, entry_frac, shield_frac
    ) * t_hours * SECONDS_PER_HOUR
    capacity = tss_h_mol_m3() * diffusion_depth_m(t_hours, T_C)
    return absorbed - capacity


def hydride_case_depth_m(
    t_hours: float,
    j_op_A_m2: float,
    faradaic_efficiency: float,
    T_C: float,
    entry_frac: Optional[float] = None,
    shield_frac: Optional[float] = None,
) -> float:
    """Near-surface δ-hydride case thickness (m) from the excess inventory."""
    n_ex = excess_inventory_mol_m2(
        t_hours, j_op_A_m2, faradaic_efficiency, T_C, entry_frac, shield_frac
    )
    if n_ex <= 0.0:
        return 0.0
    return n_ex / max(hydride_h_mol_m3() - tss_h_mol_m3(), 1.0)


def case_growth_rate_um_per_1000h(
    j_op_A_m2: float,
    faradaic_efficiency: float,
    T_C: float,
    entry_frac: Optional[float] = None,
    shield_frac: Optional[float] = None,
) -> float:
    """Steady case growth rate (µm/1000 h) once the case exists."""
    delta = hydride_case_depth_m(
        2000.0, j_op_A_m2, faradaic_efficiency, T_C, entry_frac, shield_frac
    ) - hydride_case_depth_m(
        1000.0, j_op_A_m2, faradaic_efficiency, T_C, entry_frac, shield_frac
    )
    return delta * 1e6


def hydride_onset_hours(
    j_op_A_m2: float,
    faradaic_efficiency: float,
    T_C: float,
    entry_frac: Optional[float] = None,
    shield_frac: Optional[float] = None,
) -> float:
    """Campaign hours at which the absorbed inventory first exceeds the
    α-case dissolved capacity (hydride nucleation condition).  Bisection on
    the monotone excess function; ``inf`` if it never happens by 1e6 h.
    """
    def excess(t_h: float) -> float:
        return excess_inventory_mol_m2(
            t_h, j_op_A_m2, faradaic_efficiency, T_C, entry_frac, shield_frac
        )

    if excess(1.0e-3) > 0.0:
        return 1.0e-3            # hydride from the first seconds (super-saturated)
    if excess(1.0e6) <= 0.0:
        return math.inf
    lo, hi = 1.0e-3, 1.0e6
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if excess(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return hi


def campaign_life_hours(
    j_op_A_m2: float,
    faradaic_efficiency: float,
    T_C: float,
    entry_frac: Optional[float] = None,
    shield_frac: Optional[float] = None,
) -> float:
    """Hours until the hydride case reaches the critical spall depth
    (anchor ``TI_HYD_CRIT_CASE_UM``).  ``inf`` if never — the drum then
    dies by the oxidative mechanism instead (substrate_passivation).
    """
    delta_crit_m = _a("TI_HYD_CRIT_CASE_UM") * 1e-6

    def case_m(t_h: float) -> float:
        return hydride_case_depth_m(
            t_h, j_op_A_m2, faradaic_efficiency, T_C, entry_frac, shield_frac
        )

    if case_m(1.0e-3) >= delta_crit_m:
        return 1.0e-3
    if case_m(1.0e6) < delta_crit_m:
        return math.inf
    lo, hi = 1.0e-3, 1.0e6
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if case_m(mid) >= delta_crit_m:
            hi = mid
        else:
            lo = mid
    return hi


def gc_drift_multiplier(
    t_hours: float,
    j_op_A_m2: float,
    faradaic_efficiency: float,
    T_C: float,
    entry_frac: Optional[float] = None,
    shield_frac: Optional[float] = None,
) -> float:
    """Multiplier on ``substrate_passivation.interfacial_fracture_energy``.

    1.0 while the surface stays hydride-free; ramps linearly to the
    anchored floor as the case approaches δ_crit.  Hydride lifts then
    cracks the oxide scale, and the spall patches are the new peel-front
    morphology — this factor is what makes "peel passes day one, fails
    month two" concrete.  Composited multiplicatively with the oxidative
    δ_ox drift (different mechanism, same interface).
    """
    delta_m = hydride_case_depth_m(
        t_hours, j_op_A_m2, faradaic_efficiency, T_C, entry_frac, shield_frac
    )
    if delta_m <= 0.0:
        return 1.0
    delta_crit = _a("TI_HYD_CRIT_CASE_UM") * 1e-6
    floor = _a("TI_HYD_GC_FLOOR_FRAC")
    ramp = min(1.0, delta_m / delta_crit)
    return 1.0 - (1.0 - floor) * ramp


def campaign_verdict(
    life_h: float,
    target_h: float,
    safety_multiple: float = 3.0,
) -> str:
    """clears / conditional / fails for one campaign-life estimate.

    ``safety_multiple`` is a structural screening convention (the same
    shape as the melt-balance margin bands), not a measured number:
    hydriding is locally patchy, and the band between "expected life"
    and "design target" must absorb the patch factor.
    """
    if life_h >= safety_multiple * target_h:
        return VERDICT_CLEARS
    if life_h >= target_h:
        return VERDICT_CONDITIONAL
    return VERDICT_FAILS


# ════════════════════════════════════════════════════════════════════════
#  Evaluation record
# ════════════════════════════════════════════════════════════════════════

@dataclass
class DrumHydridingResult:
    """Screening verdict for the Ti drum under one operating point."""

    architecture_id: str
    j_op_A_m2: float
    faradaic_efficiency: float
    temperature_C: float
    entry_frac: float
    shield_frac: float

    her_partial_A_m2: float
    absorbed_flux_mol_m2_s: float
    h_diffusivity_m2_s: float
    tss_h_wt_ppm: float

    hydride_onset_h: float
    case_rate_um_per_1000h: float
    campaign_life_h: float
    crit_case_um: float

    verdict_1000h: str
    verdict_4000h: str
    dominant_at_1000h: str    # "hydriding" | "oxidation (TiO2)" | "hydride-free"

    note: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ("hydride_onset_h", "campaign_life_h"):
            if math.isinf(d[k]):
                d[k] = "inf"
        return d


def evaluate_drum_hydriding(
    architecture_id: str = "drum_and_strip",
    conditions: Optional[cell_architecture.OperatingConditions] = None,
    entry_frac: Optional[float] = None,
    shield_frac: Optional[float] = None,
    j_op_override: Optional[float] = None,
) -> DrumHydridingResult:
    """Hydriding screen for the drum at its *live* operating point.

    The HER partial current derives from the architecture evaluation (j_op)
    and the shared OperatingConditions FE — perturbing either rederives
    this module's verdict, as with every ladder-facing module.
    """
    cond = conditions or cell_architecture.OperatingConditions()
    arch = cell_architecture.evaluate_architecture(
        cell_architecture.ARCHITECTURES[architecture_id], cond
    )
    j_op = arch.j_operating_A_m2 if j_op_override is None else j_op_override
    fe = cond.faradaic_efficiency
    T = cond.temperature_C
    k = _a("TI_H_ENTRY_FRAC") if entry_frac is None else entry_frac
    s = _a("TI_H_SHIELD_FRAC") if shield_frac is None else shield_frac

    life = campaign_life_hours(j_op, fe, T, k, s)
    onset = hydride_onset_hours(j_op, fe, T, k, s)

    if math.isinf(onset):
        dominant = "hydride-free"
    elif life < 1000.0:
        dominant = "hydriding"
    else:
        dominant = "oxidation (TiO2)"

    return DrumHydridingResult(
        architecture_id=architecture_id,
        j_op_A_m2=j_op,
        faradaic_efficiency=fe,
        temperature_C=T,
        entry_frac=k,
        shield_frac=s,
        her_partial_A_m2=her_partial_current_A_m2(j_op, fe),
        absorbed_flux_mol_m2_s=absorbed_h_flux_mol_m2_s(j_op, fe, k, s),
        h_diffusivity_m2_s=h_diffusivity_m2_s(T),
        tss_h_wt_ppm=_a("TI_H_TSS_WT_PPM_60C"),
        hydride_onset_h=onset,
        case_rate_um_per_1000h=case_growth_rate_um_per_1000h(j_op, fe, T, k, s),
        campaign_life_h=life,
        crit_case_um=_a("TI_HYD_CRIT_CASE_UM"),
        verdict_1000h=campaign_verdict(life, 1000.0),
        verdict_4000h=campaign_verdict(life, 4000.0),
        dominant_at_1000h=dominant,
        note=(
            "k_entry × f_shield carries a ~3-decade band: interrogate "
            "sweep_entry_shield() before trusting any single verdict. "
            "Plasticity / hydride K_IC / oxide buckling are named and "
            "shelved, not modelled (L1)."
        ),
    )


def sweep_entry_shield(
    architecture_id: str = "drum_and_strip",
    conditions: Optional[cell_architecture.OperatingConditions] = None,
    entry_fracs: Optional[List[float]] = None,
    shield_fracs: Optional[List[float]] = None,
    target_h: float = 1000.0,
) -> List[Dict[str, Any]]:
    """Verdict grid over the honest uncertainty band of k_entry × f_shield.

    This is the module's answer to its own weakest link: instead of a point
    estimate, the design question is *which quadrant of the band* keeps the
    peel window alive for one campaign.
    """
    entries = entry_fracs or [1e-3, 1e-2, 5e-2, 2e-1]
    shields = shield_fracs or [1e-4, 1e-3, 1e-2, 1e-1]
    cond = conditions or cell_architecture.OperatingConditions()
    arch = cell_architecture.evaluate_architecture(
        cell_architecture.ARCHITECTURES[architecture_id], cond
    )
    rows: List[Dict[str, Any]] = []
    for k in entries:
        for s in shields:
            life = campaign_life_hours(
                arch.j_operating_A_m2, cond.faradaic_efficiency,
                cond.temperature_C, k, s,
            )
            rows.append({
                "entry_frac": k,
                "shield_frac": s,
                "campaign_life_h": life if not math.isinf(life) else "inf",
                "verdict": campaign_verdict(
                    life if not math.isinf(life) else math.inf, target_h
                ),
            })
    return rows


def design_target_entry_shield(
    architecture_id: str = "drum_and_strip",
    conditions: Optional[cell_architecture.OperatingConditions] = None,
    target_h: float = 3000.0,
    safety_multiple: float = 3.0,
) -> float:
    """Max k_entry × f_shield whose campaign life clears target × safety.

    Bisection on the product (verdicts are monotone in it) — this is the
    number a drum-treatment programme (re-passivation, pinhole audit,
    low-entry coatings) must hold.
    """
    cond = conditions or cell_architecture.OperatingConditions()
    arch = cell_architecture.evaluate_architecture(
        cell_architecture.ARCHITECTURES[architecture_id], cond
    )
    floor_life = safety_multiple * target_h

    def life_at(product: float) -> float:
        return campaign_life_hours(
            arch.j_operating_A_m2, cond.faradaic_efficiency,
            cond.temperature_C, product, 1.0,
        )

    hi = 1.0
    if life_at(hi) >= floor_life:
        return hi
    lo = 1.0e-9
    if life_at(lo) < floor_life:
        return 0.0           # unreachable at any band value — say so
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if life_at(mid) >= floor_life:
            lo = mid
        else:
            hi = mid
    return lo


# ─── Feeds to sister modules ──────────────────────────────────────────────

def feed_to_adhesion_peel(
    t_hours: float,
    architecture_id: str = "drum_and_strip",
    conditions: Optional[cell_architecture.OperatingConditions] = None,
    entry_frac: Optional[float] = None,
    shield_frac: Optional[float] = None,
) -> Dict[str, float]:
    """G_c drift for ``adhesion_peel`` — composite with substrate_passivation.

    The caller multiplies: G_c(t) = G_c,oxide(t) × gc_drift_multiplier(t).
    """
    cond = conditions or cell_architecture.OperatingConditions()
    arch = cell_architecture.evaluate_architecture(
        cell_architecture.ARCHITECTURES[architecture_id], cond
    )
    mult = gc_drift_multiplier(
        t_hours, arch.j_operating_A_m2, cond.faradaic_efficiency,
        cond.temperature_C, entry_frac, shield_frac,
    )
    return {
        "t_hours": t_hours,
        "gc_hydride_multiplier": mult,
        "hydride_case_um": hydride_case_depth_m(
            t_hours, arch.j_operating_A_m2, cond.faradaic_efficiency,
            cond.temperature_C, entry_frac, shield_frac,
        ) * 1e6,
        "note": "ti_hydriding v0 — multiply into substrate_passivation G_c; "
                "composite of two mechanisms on one interface",
    }


def campaign_ledger_row(
    t_hours: float,
    architecture_id: str = "drum_and_strip",
    conditions: Optional[cell_architecture.OperatingConditions] = None,
) -> Dict[str, Any]:
    """One row for ``closed_loop.py``'s campaign drum-life ledger."""
    res = evaluate_drum_hydriding(architecture_id, conditions)
    return {
        "campaign_h": t_hours,
        "hydride_case_um": hydride_case_depth_m(
            t_hours, res.j_op_A_m2, res.faradaic_efficiency,
            res.temperature_C, res.entry_frac, res.shield_frac,
        ) * 1e6,
        "gc_hydride_multiplier": gc_drift_multiplier(
            t_hours, res.j_op_A_m2, res.faradaic_efficiency,
            res.temperature_C, res.entry_frac, res.shield_frac,
        ),
        "campaign_life_h": (
            res.campaign_life_h if not math.isinf(res.campaign_life_h) else "inf"
        ),
        "crit_case_um": res.crit_case_um,
        "verdict_1000h": res.verdict_1000h,
    }


def model_scope() -> Dict[str, Any]:
    return {
        "screening_flag": SCREENING_FLAG,
        "live_derivations": [
            "cell_architecture.evaluate_architecture(drum_and_strip) j_op "
            "(HER partial current = j_op × (1 − FE) at call time)",
            "substrate_passivation G_c chain (composited by caller via "
            "feed_to_adhesion_peel)",
        ],
        "screening_proxies_anchored": [
            "H entry fraction k_entry (decade band)",
            "drum-area shield fraction f_shield (deposit is an H sink)",
            "α-Ti H diffusivity + activation energy",
            "terminal solid solubility (TSS) at 60 °C",
            "hydride stoichiometry TiH~1.7, critical case depth",
        ],
        "out_of_scope": [
            "elastic/plastic case stress evolution (formally GPa, shelved)",
            "hydride fracture toughness K_IC and oxide buckling mechanics",
            "θ_H surface-state coupling (surface_state.py; Fe-side only today)",
            "transient T profiles along the drum circumference",
        ],
    }


def main(argv: Optional[List[str]] = None) -> None:  # pragma: no cover - CLI
    p = argparse.ArgumentParser(
        description="Ti drum hydriding & campaign-life screen (V6 §4.1)."
    )
    p.add_argument("--architecture", default="drum_and_strip")
    p.add_argument("--entry", type=float, default=None,
                   help="override k_entry (H entry fraction)")
    p.add_argument("--shield", type=float, default=None,
                   help="override f_shield (exposed drum-area fraction)")
    args = p.parse_args(argv)

    res = evaluate_drum_hydriding(
        args.architecture, entry_frac=args.entry, shield_frac=args.shield,
    )
    print(f"ti_hydriding — Ti drum campaign life  [{SCREENING_FLAG}]")
    print(f"  architecture        {res.architecture_id}")
    print(f"  j_op                {res.j_op_A_m2:,.0f} A/m²  "
          f"(HER partial {res.her_partial_A_m2:,.0f} A/m² @ FE "
          f"{res.faradaic_efficiency:.2f}, {res.temperature_C:.0f} °C)")
    print(f"  absorbed H flux     {res.absorbed_flux_mol_m2_s:.3e} mol/m²/s  "
          f"(k_entry {res.entry_frac:g} × f_shield {res.shield_frac:g})")
    print(f"  D_H(α-Ti)           {res.h_diffusivity_m2_s:.2e} m²/s   "
          f"TSS {res.tss_h_wt_ppm:.0f} wt-ppm")
    onset = "inf" if math.isinf(res.hydride_onset_h) else f"{res.hydride_onset_h:,.0f} h"
    life = "inf" if math.isinf(res.campaign_life_h) else f"{res.campaign_life_h:,.0f} h"
    print(f"  hydride onset       {onset}")
    print(f"  case growth         {res.case_rate_um_per_1000h:.1f} µm/1000 h")
    print(f"  campaign life       {life}   (crit case {res.crit_case_um:.0f} µm)")
    print(f"  verdict @1000 h     {res.verdict_1000h}")
    print(f"  verdict @4000 h     {res.verdict_4000h}")
    print(f"  dominant @1000 h    {res.dominant_at_1000h}")
    target = design_target_entry_shield()
    print(f"  design target       k_entry × f_shield ≤ {target:.2e} "
          f"(clears 3,000 h ×3 safety at live j_op)")
    print()
    print("  verdict grid over the honest k_entry × f_shield band:")
    for row in sweep_entry_shield():
        print(f"    k={row['entry_frac']:>6g}  f={row['shield_frac']:>6g}  →  "
              f"life {row['campaign_life_h']:>10}   {row['verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
