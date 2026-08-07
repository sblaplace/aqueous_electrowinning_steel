"""
Q5 combined harvestability coupon: adhesion/release + residual stress + hydrogen.

Why this module exists
----------------------
``docs/NEXT_STEPS.md`` closes the two highest-value physical unknowns with one
coordinated coupon campaign on candidate *titanium release surfaces*:

* **#2.5 Adhesion/release coupon** — plate iron on candidate titanium release
  surfaces and measure peel/strip force.  "This is a branch decision, not a
  material-properties project."
* **#2.6 Stress and hydrogen** — bent-strip curvature (residual stress) plus
  hydrogen uptake/bake-out on representative deposits.

Those two items protect the program against building a machine that makes
**unharvestable** (a deposit that will not come off the drum, or comes off in
fragments) or **unsafe** (a deposit whose hydrogen cannot be baked out to a
non-embrittling level) material.

The pieces already exist separately — ``adhesion_peel`` (interface, peel
force), ``internal_stress`` (film, bent-strip/Stoney curvature) and
``hydrogen_embrittlement`` (H uptake + bake-out).  Each exposes its own coupon
protocol but they are scattered and speak different conditions.  This module is
the **Q5 wrapper that runs all three measurements against the same candidate Ti
coupon matrix and a single operating point**, returns one combined
harvestability + safety verdict per surface, and specifies the physical coupon
set that would replace the estimates with measurements.

It deliberately adds **no new physics**.  Everything here is composition and
reporting on top of the three existing screening models; ``model_scope()``
states that explicitly.

Scope and honesty
-----------------
This is L0 screening scaffold, exactly like ``adhesion_peel`` and
``internal_stress``.  No wet-lab iron data exists in this repository and none
is invented here: the numbers are what the three existing *screening* models
predict.  The purpose of the dry-run is to (a) prove the three measurements can
share one coupon matrix without contradiction and (b) turn NEXT_STEPS #2.5/#2.6
into an executable protocol with named decision rules — *not* to produce a
peel-strength or bake-out prediction.  See :func:`coupon_spec` for the physical
instructions and :func:`model_scope` for the is / is-not.

References
----------
See the composing modules: ``adhesion_peel.py`` (Kendall, Hutchinson–Suo,
Rice–Wang, Weil), ``internal_stress.py`` (Stoney, Hoffman), and
``hydrogen_embrittlement.py`` (IPZ H-entry, Fickian bake-out, Troiano
susceptibility).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from .adhesion_peel import (
    MIN_CONTROLLABLE_TENSION_N_PER_M,
    SUBSTRATES,
    PeelConditions,
    PeelResult,
    evaluate_peel,
)
from .hydrogen_embrittlement import (
    bakeout_schedule,
    he_susceptibility_index,
    hydrogen_uptake_from_electrolysis,
)
from .internal_stress import deposit_stress_from_conditions

#: Reference operating point — the Day-1 reference bath, shared with the
#: adhesion/stress coupon sets (docs/FIRST_LAB_DAY.md bath B0).
REFERENCE_J_MA_CM2 = 100.0
REFERENCE_FE_PERCENT = 85.0
REFERENCE_TIME_S = 1800.0          # 30 min → ~38 µm at 85% FE (c.f. ~50 µm@60min)
REFERENCE_T_C = 60.0
REFERENCE_PH = 3.0

#: Bake-out gate: temperature and target diffusible H treated as "safe".
#: These match hydrogen_embrittlement.bakeout_time_hr defaults so the dry-run
#: and the standalone bake-out tool agree.
BAKEOUT_T_C = 170.0
BAKEOUT_TARGET_PPM = 0.1

#: Bake-out is "fast"/safe below this many hours at the gate temperature.  A
#: deposit whose H can be baked to the target well within a shift is *safe
#: after a cheap bake*; one that needs > 24 hr is effectively not removable in
#: a production cadence and is a hard stop (NEXT_STEPS #2.6 "unsafe material").
#: 25 µm foil bakes out in minutes at 170 °C; the 24 hr gate separates that
#: from a deposit where traps or thickness make bake infeasible.
BAKEOUT_FEASIBLE_HR = 24.0

#: Peel-machine window (from adhesion_peel) reused verbatim for the verdicts.
WINDER_FLOOR_N_PER_M = MIN_CONTROLLABLE_TENSION_N_PER_M

HarvestOutcome = Literal[
    "harvestable_foil",
    "harvestable_flake",
    "unharvestable",
    "bakeout_required",
]

#: Candidate titanium release surfaces, in coupon-spec order.  The reference is
#: Ti / passive TiO₂ (copper-foil drum practice).  Etched Ti is the
#: de-passivation failure mode of that same surface, not a design choice.
TI_SURFACE_IDS = ["ti_passive_tio2", "ti_bare_etched"]

#: Full coupon matrix = Ti candidates + the discrimination/control set from the
#: adhesion module (conductive only; PTFE is a non-cathode and excluded on
#: physics).  The copper substrate is the deliberate negative control that
#: proves the set measures adhesion at all.
COUPON_SURFACE_IDS = [
    "ti_passive_tio2",     # reference drum surface
    "ti_bare_etched",      # depassivation failure mode of the reference
    "stainless_316_passive",  # known-good industrial batch-strip blank
    "chromium_plated",     # electroforming release-mandrel alternative
    "copper_substrate",    # negative control (expect strong bond)
]


@dataclass(frozen=True)
class OperatingPoint:
    """One shared plating condition for every coupon in the set.

    The purpose of holding one point is the cross-modal integrity check: the
    same (j, FE, t, T, pH) must consistently drive peel force, film stress and
    hydrogen content.  If the three measurements disagree about the *same*
    operating point, the models are internally inconsistent and the coupon data
    will arbitrate.
    """

    j_mA_cm2: float = REFERENCE_J_MA_CM2
    current_efficiency_percent: float = REFERENCE_FE_PERCENT
    deposition_time_s: float = REFERENCE_TIME_S
    temperature_C: float = REFERENCE_T_C
    bath_pH: float = REFERENCE_PH

    def to_peel_conditions(self, C_H_ppm: float, thickness_um: float) -> PeelConditions:
        """Build the PeelConditions that match this shared operating point."""
        return PeelConditions(
            thickness_um=thickness_um,
            C_H_ppm=C_H_ppm,
            bath_temperature_C=self.temperature_C,
        )


def _derived_deposit(op: OperatingPoint) -> Dict[str, Any]:
    """Deposit thickness + diffusible H for the shared operating point.

    Reuses ``internal_stress.deposit_stress_from_conditions`` for thickness
    (Faraday at j, FE, t) and the IPZ hydrogen uptake — so the peel module's
    stress decomposition and this module agree on what the same run deposits.
    """
    sd = deposit_stress_from_conditions(
        j_mA_cm2=op.j_mA_cm2,
        current_efficiency_percent=op.current_efficiency_percent,
        deposition_time_s=op.deposition_time_s,
        bath_pH=op.bath_pH,
        temperature_C=op.temperature_C,
        substrate="ti_passive_tio2",  # thickness/H do not depend on substrate
    )
    derived = sd["derived"]
    return {
        "thickness_um": float(derived["thickness_um"]),
        "C_H_diffusible_ppm": float(derived["C_H_diffusible_ppm"]),
        "grain_size_um": float(derived["grain_size_um"]),
    }


def assess_surface(
    substrate_id: str,
    op: Optional[OperatingPoint] = None,
    sigma_y_MPa: float = 400.0,
) -> Dict[str, Any]:
    """Run all three measurements for one candidate surface at the shared point.

    Composes, for one substrate id:

    * **Adhesion/release** (NEXT_STEPS #2.5): ``adhesion_peel.evaluate_peel``
      => peel/strip force + interface outcome.
    * **Residual stress** (NEXT_STEPS #2.6a): ``internal_stress``
      deposit-from-conditions + bent-strip curvature measurability.
    * **Hydrogen** (NEXT_STEPS #2.6b): ``hydrogen_embrittlement`` uptake +
      bake-out schedule + Troiano susceptibility.

    Returns a JSON-serializable dict with the per-surface verdict, its
    supporting quantities, and the decision-rules mapping (see :func:`coupon_spec`).
    """
    op = op or OperatingPoint()
    sub = SUBSTRATES.get(substrate_id)
    if sub is None:
        raise ValueError(f"unknown surface: {substrate_id!r} (not in adhesion_peel.SUBSTRATES)")
    if not sub.electrically_conductive:
        raise ValueError(
            f"{substrate_id} is not electrically conductive and cannot be a "
            "cathode; excluded from the coupon matrix on physics."
        )

    derived = _derived_deposit(op)
    thickness_um = derived["thickness_um"]
    C_H_ppm = derived["C_H_diffusible_ppm"]

    peel: PeelResult = evaluate_peel(
        sub, op.to_peel_conditions(C_H_ppm, thickness_um)
    )

    stress = deposit_stress_from_conditions(
        j_mA_cm2=op.j_mA_cm2,
        current_efficiency_percent=op.current_efficiency_percent,
        deposition_time_s=op.deposition_time_s,
        bath_pH=op.bath_pH,
        temperature_C=op.temperature_C,
        substrate=sub,
    )

    h_uptake = hydrogen_uptake_from_electrolysis(
        current_density_mA_cm2=op.j_mA_cm2,
        deposition_time_s=op.deposition_time_s,
        her_efficiency=max(1.0 - op.current_efficiency_percent / 100.0, 1e-4),
        bath_pH=op.bath_pH,
        temperature_C=op.temperature_C,
    )

    bake = bakeout_schedule(
        deposit_thickness_um=thickness_um,
        initial_C_H_ppm=C_H_ppm,
        target_C_H_ppm=BAKEOUT_TARGET_PPM,
        grain_size_um=derived["grain_size_um"],
    )
    # bakeout_schedule returns one dict per temperature (matching standalone tool)
    bo = {row["temperature_C"]: row for row in bake}
    bake_at_gate = bo.get(BAKEOUT_T_C, bake[-1])

    he = he_susceptibility_index(sigma_y_MPa=sigma_y_MPa, C_H_diffusible_ppm=C_H_ppm,
                                 temperature_C=op.temperature_C)

    verdict, reasons = _combined_verdict(
        peel=peel,
        C_H_ppm=C_H_ppm,
        bakeout_time_hr=float(bake_at_gate["bakeout_time_hr"]),
        he_risk=str(he["risk_level"]),
    )

    return {
        "substrate_id": substrate_id,
        "substrate_name": sub.name,
        "evidence_level": sub.evidence_level,
        "role": _role(substrate_id),
        "adhesion_release": peel.to_dict(),
        "residual_stress": {
            "total_MPa": round(stress["components"]["total_MPa"], 1),
            "breakdown_MPa": {
                k: round(v, 1) for k, v in stress["components"].items()
            },
            "dominant_mechanism": stress["dominant_mechanism"],
            "sign": stress["sign"],
            "sources": stress["sources"],
        },
        "hydrogen": {
            "C_H_diffusible_ppm": round(C_H_ppm, 2),
            "uptake_model": h_uptake["model"],
            "absorption_fraction": round(float(h_uptake["absorption_fraction"]), 4),
            "bakeout": {
                "gate_temperature_C": BAKEOUT_T_C,
                "target_ppm": BAKEOUT_TARGET_PPM,
                "time_hr": round(float(bake_at_gate["bakeout_time_hr"]), 1),
                "residual_ppm": round(float(bake_at_gate["residual_C_H_ppm"]), 4),
                "full_schedule": [
                    {
                        "temperature_C": row["temperature_C"],
                        "time_hr": round(float(row["bakeout_time_hr"]), 1),
                    }
                    for row in bake
                ],
            },
            "susceptibility": {
                "I_HE": round(float(he["I_HE"]), 2),
                "risk_level": he["risk_level"],
            },
        },
        "verdict": verdict,
        "reasons": reasons,
        "operating_point": {
            "j_mA_cm2": op.j_mA_cm2,
            "current_efficiency_percent": op.current_efficiency_percent,
            "deposition_time_s": op.deposition_time_s,
            "temperature_C": op.temperature_C,
            "bath_pH": op.bath_pH,
            "derived_thickness_um": round(thickness_um, 1),
        },
    }


def _role(substrate_id: str) -> str:
    if substrate_id == "ti_passive_tio2":
        return "reference drum surface"
    if substrate_id == "ti_bare_etched":
        return "de-passivation failure mode (covers the reference's drift)"
    if substrate_id == "copper_substrate":
        return "negative control (expect strong bond)"
    return "alternative release surface"


def _combined_verdict(
    peel: PeelResult,
    C_H_ppm: float,
    bakeout_time_hr: float,
    he_risk: str,
) -> tuple[str, list[str]]:
    """Merge the peel outcome, bake-out feasibility and H susceptibility.

    Decision order (each rule maps to NEXT_STEPS #2.5/#2.6):

    1. **Hard safety stop** — the deposit's H *cannot* be baked to the safe
       target in a production cadence or even the as-plated Troiano index is
       critical: the material is unsafe however it releases.  (At the reference
       point H bakes out in < 1 hr, so this is *not* hit; it protects the
       thick/trap-heavy / un-baked corner NEXT_STEPS #2.6 names "unsafe
       material".)
    2. **Unharvestable** — the peel outcome says the deposit will not come
       off the drum cleanly on this surface at this point (bonded,
       cohesive-failure into the film, or tears before peel).  This kills the
       foil branch *for this surface* regardless of hydrogen.  (Redirection to
       flake is scored separately via ``good_for_flake_harvest``.)
    3. **harvestable_* / bakeout_required** — otherwise, from the peel
       outcome plus whether a feasible bake is still owed.
    """
    reasons: list[str] = peel.reasons

    # Any as-plated H hazard that a *feasible* bake removes is recorded, not
    # fatal; the hard stop is bakeout infeasibility alone.
    bake_infeasible = bakeout_time_hr > BAKEOUT_FEASIBLE_HR
    if bake_infeasible:
        reasons.append(
            f"Bake-out to {BAKEOUT_TARGET_PPM} ppm at {BAKEOUT_T_C} °C needs "
            f"{bakeout_time_hr:.1f} hr (> {BAKEOUT_FEASIBLE_HR:g} hr gate); "
            f"Troiano index {he_risk}. Unsafe: H cannot be removed in a "
            f"production cadence."
        )
        return "bakeout_required", reasons

    if peel.outcome == "bonded_no_release":
        return "unharvestable", reasons
    if peel.outcome == "cohesive_failure_in_film":
        return "unharvestable", reasons
    if peel.outcome == "tears_before_peel":
        return "unharvestable", reasons

    bake_owed = bakeout_time_hr > 1.0
    if bake_owed:
        reasons.append(
            f"Diffusible H {C_H_ppm:.0f} ppm (as-plated Troiano {he_risk}) "
            f"bakes to {BAKEOUT_TARGET_PPM} ppm in {bakeout_time_hr:.1f} hr at "
            f"{BAKEOUT_T_C} °C — safe after a bake step."
        )
        return "bakeout_required", reasons

    if he_risk in ("high", "critical"):
        # Fast bake does not erase the as-plated hazard; record it so the
        # report never silently converts a hydrogen-heavy deposit into
        # "harvestable" without the bake step being visible.
        reasons.append(
            f"As-plated Troiano index {he_risk} at {C_H_ppm:.0f} ppm — retires "
            f"to {BAKEOUT_TARGET_PPM} ppm in {bakeout_time_hr:.1f} hr bake at "
            f"{BAKEOUT_T_C} °C. A bake step is mandatory before melt/ship."
        )

    if peel.good_for_flake_harvest:   # spontaneous_delamination
        return "harvestable_flake", reasons
    if peel.peelable:                 # clean_peel / marginal_peel
        return "harvestable_foil", reasons

    return "unharvestable", reasons


def coupon_spec(op: Optional[OperatingPoint] = None) -> Dict[str, Any]:
    """Return the executable physical coupon protocol for NEXT_STEPS #2.5/#2.6.

    This is the *instructions* half of Q5: which coupons, which three
    measurements, what each replaces in the model set, and the decision rules
    that turn the measured numbers into a branch + safety call.  It is the
    combined analogue of ``adhesion_peel.coupon_test_protocol`` and
    ``internal_stress.coupon_curvature_protocol``, run against one shared
    candidate-Ti coupon matrix.
    """
    op = op or OperatingPoint()
    derived = _derived_deposit(op)

    return {
        "title": "Q5 combined harvestability coupon (NEXT_STEPS #2.5 + #2.6)",
        "gates": (
            "docs/NEXT_STEPS.md §2 items 5 & 6; "
            "docs/PROGRAM_SUMMARY.md gate 2 (architecture); "
            "adhesion_peel.coupon_test_protocol; "
            "internal_stress.coupon_curvature_protocol"
        ),
        "runs_alongside": (
            "The Day-1 Hull cell + divided-cell sets (docs/FIRST_LAB_DAY.md) — "
            "same bath B0, same rectifier, same session."
        ),
        "operating_point": {
            "j_mA_cm2": op.j_mA_cm2,
            "current_efficiency_percent": op.current_efficiency_percent,
            "deposition_time_s": op.deposition_time_s,
            "temperature_C": op.temperature_C,
            "bath_pH": op.bath_pH,
            "expected_thickness_um": round(derived["thickness_um"], 1),
            "expected_C_H_diffusible_ppm": round(derived["C_H_diffusible_ppm"], 2),
        },
        "coupons": [
            {
                "substrate_id": sid,
                "substrate": SUBSTRATES[sid].name,
                "role": _role(sid),
                "n_replicates": 3,
                "evidence_level": SUBSTRATES[sid].evidence_level,
            }
            for sid in COUPON_SURFACE_IDS
        ],
        "measurements": [
            {
                "measurement": "90° peel test, ASTM B571 / D6862 geometry",
                "instrument": "load cell on a motorised stage, 10 mm strip",
                "yields": "peel/strip force per width -> interfacial toughness",
                "replaces_in_model": (
                    "adhesion_peel interfacical_toughness.plastic_amplification"
                ),
                "tag": "adhesion_release",
            },
            {
                "measurement": "bent-strip / coupon curvature before & after",
                "instrument": (
                    "dial gauge (10 µm) or stylus profilometer (1 µm) over "
                    "coupon gauge length on 0.2/0.4 mm Ti shim"
                ),
                "yields": "residual stress via Stoney (film-side)",
                "replaces_in_model": (
                    "internal_stress HOFFMAN_DELTA_M and the forward intrinsic "
                    "estimate"
                ),
                "tag": "residual_stress",
            },
            {
                "measurement": "deposit thickness by mass & cross-section",
                "instrument": "analytical balance + metallographic mount",
                "yields": "h for G=(1−ν)σ²h/E and a Faraday cross-check",
                "replaces_in_model": "Faraday-law thickness assumption",
                "tag": "geometry",
            },
            {
                "measurement": "diffusible H by inert-gas fusion / TDS",
                "instrument": "hot-extraction analyser (outsourced); Devanathan-"
                               "Stachurski permeation cell to calibrate IPZ",
                "yields": "C_H (ppm) and the H entry fraction",
                "replaces_in_model": (
                    "hydrogen_embrittlement IPZ absorption constants"
                ),
                "tag": "hydrogen_uptake",
            },
            {
                "measurement": "bake-out schedule (temperature x time to target)",
                "instrument": "laboratory oven + repeated H analysis",
                "yields": "measured bake-out curve",
                "replaces_in_model": "bakeout_time_hr Fickian estimate",
                "tag": "hydrogen_bakeout",
            },
        ],
        "decision_rules": [
            {
                "if": "iron on passive TiO₂ requires > available winder tension "
                      "to peel, or the strip fractures at any practical thickness",
                "then": "foil branch dead on Ti — move to flake/feedstock or an "
                        "alternative release surface (NEXT_STEPS #2.5)",
                "class": "unharvestable",
            },
            {
                "if": "peel force is controllable and the strip stays intact "
                      f"({WINDER_FLOOR_N_PER_M:,.0f}+ N/m, ≥2× margin to web "
                      "yield)",
                "then": "foil branch survives on that surface",
                "class": "harvestable_foil",
            },
            {
                "if": "deposit self-releases below the foil target thickness",
                "then": "adopt flake/powder path and delete the winder",
                "class": "harvestable_flake",
            },
            {
                "if": "diffusible H cannot reach the safe gate in a short bake, "
                      "or Troiano index is high/critical",
                "then": "the electrolytically-fine deposit is unsafe — bake-out "
                        "or suppress H (waveform/temp/additive) before use "
                        "(NEXT_STEPS #2.6)",
                "class": "bakeout_required",
            },
            {
                "if": "measured σ(h) or H disagrees with the mechanism "
                      "decomposition of the shared operating point",
                "then": "recalibrate intrinsic & hydrogen terms against measured "
                        "C_H and grain size before trusting the peel window",
                "class": "cross-modal check",
            },
        ],
        "why_now": (
            "NEXT_STEPS #2.5/#2.6 are exactly the physical unknowns that "
            "protect against building a machine that makes unharvestable or "
            "unsafe material. One shared coupon matrix lets the same runs "
            "close the interface (peel), the film (stress) and the "
            "hydrogen/safety question simultaneously — the cheapest set that "
            "can delete an architecture branch or flag a bake-out requirement."
        ),
    }


def run_harvestability_dryrun(
    op: Optional[OperatingPoint] = None,
    sigma_y_MPa: float = 400.0,
) -> Dict[str, Any]:
    """Deterministic dry-run of the Q5 coupon across the candidate-Ti matrix.

    Runs :func:`assess_surface` for every coupon in :data:`COUPON_SURFACE_IDS`
    at the shared operating point and returns a JSON-serializable report.
    Single-source-of-truth for the CLI wrapper.
    """
    op = op or OperatingPoint()
    surfaces = [assess_surface(sid, op, sigma_y_MPa=sigma_y_MPa)
                for sid in COUPON_SURFACE_IDS]

    counts: Dict[str, int] = {}
    for s in surfaces:
        counts[s["verdict"]] = counts.get(s["verdict"], 0) + 1

    ref = next(s for s in surfaces if s["substrate_id"] == "ti_passive_tio2")
    return {
        "campaign": "q5-harvestability-coupon-dry-run",
        "credibility_note": (
            "Synthetic L0 dry-run. Composes three existing scoring models; no "
            "wet-lab iron data exists in this repository. Decision rules, not "
            "predictions."
        ),
        "operating_point": {
            "j_mA_cm2": op.j_mA_cm2,
            "current_efficiency_percent": op.current_efficiency_percent,
            "deposition_time_s": op.deposition_time_s,
            "temperature_C": op.temperature_C,
            "bath_pH": op.bath_pH,
        },
        "verdict_counts": counts,
        "reference_ti_surface": ref,
        "surfaces": surfaces,
        "spec": coupon_spec(op),
    }


def main() -> Dict[str, Any]:
    """CLI entry point: run the dry-run and print the summary table."""
    report = run_harvestability_dryrun()

    print("Q5 combined harvestability coupon — dry run (L0, synthetic)\n")
    print(f"{'Surface':<28} {'Verdict':<18} {'σ_tot':>7} {'C_H':>6} "
          f"{'bake@170':>9} {'peel':>8}")
    print("─" * 80)
    for s in report["surfaces"]:
        p = s["adhesion_release"]["peel_force_N_per_m"]
        p_str = "∞" if isinstance(p, str) or p is None else f"{p:,.0f}"
        print(
            f"{s['substrate_name'][:27]:<28} "
            f"{s['verdict']:<18} "
            f"{s['residual_stress']['total_MPa']:>7,.0f} "
            f"{s['hydrogen']['C_H_diffusible_ppm']:>6.2f} "
            f"{s['hydrogen']['bakeout']['time_hr']:>9.1f} "
            f"{p_str:>8}"
        )

    print("\nVerdict counts:", report["verdict_counts"])
    ref = report["reference_ti_surface"]
    print(f"\nReference Ti (passive TiO₂): {ref['verdict']}")
    for r in ref["reasons"]:
        print(f"  - {r}")

    print("\nPhysical coupon spec (what the lab runs):")
    for c in report["spec"]["coupons"]:
        print(f"  - {c['substrate_id']} ({c['role']})")
    return report


def model_scope() -> Dict[str, Any]:
    """Machine-readable statement of exactly what this module is and is not."""
    return {
        "provenance": (
            "Screening wrapper. Adds no physics — composes adhesion_peel, "
            "internal_stress and hydrogen_embrittlement at one operating point "
            "against a shared candidate-Ti coupon matrix. No wet-lab data."
        ),
        "computes": [
            "per-surface verdict: harvestable_foil / harvestable_flake / "
            "unharvestable / bakeout_required",
            "combined decision rules from NEXT_STEPS #2.5 (release) + "
            "#2.6 (stress + hydrogen)",
            "the executable physical coupon protocol (coupon_spec)",
        ],
        "does_not_compute": [
            "any new physics (delegated to the composing modules)",
            "a peel-strength or bake-out prediction (all L0 estimates)",
            "substrate passivation drift over service life",
        ],
        "calibration_required": [
            "90° peel toughness on the actual drum surface",
            "residual stress by coupon curvature (Stoney) in the actual bath",
            "diffusible H by thermal desorption",
            "a measured bake-out curve",
            "IPZ uptake constants from a Devanathan-Stachurski cell",
        ],
        "key_uncertainty": (
            "inherited: adhesion_peel plastic_amplification and the "
            "hydrogen_embrittlement IPZ constants dominate the verdicts; see "
            "those modules' model_scope()."
        ),
    }
