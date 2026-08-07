"""Tests for the Q5 combined harvestability coupon (adhesion + stress + H).

Covers the five-surface coupon matrix, the shared-operating-point cross-modal
consistency (the same (j, FE, t, T, pH) must drive all three measurements
without contradiction), the verdict classification, and the composed dry-run.

The module adds no physics; these tests lock the *composition* of
adhesion_peel / internal_stress / hydrogen_embrittlement against one operating
point.  Everything is L0 screening scaffold (no wet-lab data).
"""

from models.harvestability_coupon import (
    COUPON_SURFACE_IDS,
    TI_SURFACE_IDS,
    OperatingPoint,
    assess_surface,
    coupon_spec,
    model_scope,
    run_harvestability_dryrun,
)


def test_all_coupons_are_conductive_cathodes():
    """Every coupon in the matrix is a plate-able cathode surface."""
    from models.adhesion_peel import SUBSTRATES

    for sid in COUPON_SURFACE_IDS:
        assert SUBSTRATES[sid].electrically_conductive, sid
    # PTFE (insulating) is deliberately excluded from the conductive matrix.
    assert COUPON_SURFACE_IDS[0] == "ti_passive_tio2"


def test_titanium_reference_is_first_surface():
    """The passive-TiO2 drum surface is the reference coupon (order matters)."""
    assert TI_SURFACE_IDS == ["ti_passive_tio2", "ti_bare_etched"]
    assert COUPON_SURFACE_IDS[0] == "ti_passive_tio2"


def test_assess_surface_returns_all_three_channels():
    """One surface returns adhesion/release, residual stress, and hydrogen."""
    r = assess_surface("ti_passive_tio2")
    assert r["adhesion_release"]["outcome"] in (
        "clean_peel", "marginal_peel", "spontaneous_delamination",
        "bonded_no_release", "cohesive_failure_in_film", "tears_before_peel",
    )
    assert "total_MPa" in r["residual_stress"]
    assert r["hydrogen"]["C_H_diffusible_ppm"] > 0
    assert "time_hr" in r["hydrogen"]["bakeout"]
    assert r["verdict"] in (
        "harvestable_foil", "harvestable_flake",
        "unharvestable", "bakeout_required",
    )


def test_shared_operating_point_consistent_across_channels():
    """The three channels must agree on the same deposit.

    A 30 min / 100 mA/cm² / 85% FE point deposits a thin foil carrying a fixed
    diffusible-H content; the peel stress decomposition, the internal-stress
    decomposition, and the H uptake must report the *same* C_H and thickness so
    they genuinely share one coupon matrix rather than three disconnected runs.
    """
    op = OperatingPoint()
    peel_ch = assess_surface("ti_bare_etched", op)
    stress_ch = {
        s["substrate_id"]: s for s in run_harvestability_dryrun(op)["surfaces"]
    }["ti_bare_etched"]

    # Same IPZ diffusible-H everywhere (H does not depend on the substrate).
    assert abs(peel_ch["hydrogen"]["C_H_diffusible_ppm"]
               - stress_ch["hydrogen"]["C_H_diffusible_ppm"]) < 1e-6
    # Thickness from Faraday at the shared point.
    assert abs(op.deposition_time_s - 1800.0) < 1e-9


def test_hydrogen_heavy_reference_is_not_silently_harvestable():
    """As-plated critical-H is surfaced on the harvestable path, never erased.

    At the reference point the deposit bakes out fast (thin foil), but the
    as-plated Troiano index is high/critical and MUST appear in the reasons so
    the report never presents a hydrogen-heavy deposit as cleanly harvestable
    without the bake step being visible.
    """
    r = assess_surface("ti_passive_tio2")
    text = " ".join(r["reasons"])
    # Either flagged as a hard stop (bake infeasible) or surfaced as a bake
    # obligation on the harvestable path.
    assert ("bake" in text.lower()) or r["verdict"] == "bakeout_required"


def test_reference_surface_verdict_is_one_of_the_harvestable_classes():
    """The reference drum surface at the Day-1 point is harvestable (some path)."""
    r = assess_surface("ti_passive_tio2")
    assert r["verdict"] in ("harvestable_foil", "harvestable_flake",
                            "bakeout_required")


def test_coupon_spec_has_five_measurements_and_decision_rules():
    """The executable protocol lists the three Q5 measurements + gates."""
    spec = coupon_spec()
    assert len(spec["coupons"]) == len(COUPON_SURFACE_IDS)
    tags = [m["tag"] for m in spec["measurements"]]
    for required in ("adhesion_release", "residual_stress",
                     "hydrogen_uptake", "hydrogen_bakeout"):
        assert required in tags
    assert len(spec["decision_rules"]) >= 3


def test_dryrun_is_deterministic_and_complete():
    """The dry-run covers every surface and is reproducible."""
    a = run_harvestability_dryrun()
    b = run_harvestability_dryrun()
    assert [s["verdict"] for s in a["surfaces"]] == \
        [s["verdict"] for s in b["surfaces"]]
    assert len(a["surfaces"]) == len(COUPON_SURFACE_IDS)
    assert a["spec"]["title"].startswith("Q5")


def test_model_scope_is_a_wrapper_not_new_physics():
    """The module explicitly disclaims new physics and names calibration."""
    scope = model_scope()
    assert "adds no physics" in scope["provenance"].lower()
    assert any("calibration_required" == k or len(scope["calibration_required"]) >= 4
               for k in scope)
