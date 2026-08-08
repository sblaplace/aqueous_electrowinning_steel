"""Tests for models/ti_hydriding.py — Ti drum hydriding & campaign life (V6 §4.1).

The physical contract: absorbed H flux = k_entry × f_shield × j_HER/F;
hydride exists iff absorbed inventory > TSS × √(D·t); case depth linear in
excess inventory; verdicts monotone in every damage-driving parameter.  The
engineering contract: the weakest link (k_entry × f_shield) is carried as an
explicit band, the module returns a *design target* for it, and the G_c drift
feed composites multiplicatively with substrate_passivation.
"""

from __future__ import annotations

import math

import pytest

from models import cell_architecture
from models.ti_hydriding import (
    SCREENING_FLAG,
    absorbed_h_flux_mol_m2_s,
    campaign_life_hours,
    campaign_verdict,
    case_growth_rate_um_per_1000h,
    diffusion_depth_m,
    evaluate_drum_hydriding,
    excess_inventory_mol_m2,
    feed_to_adhesion_peel,
    gc_drift_multiplier,
    h_diffusivity_m2_s,
    her_partial_current_A_m2,
    hydride_case_depth_m,
    hydride_h_mol_m3,
    hydride_onset_hours,
    main,
    model_scope,
    sweep_entry_shield,
    tss_h_mol_m3,
    design_target_entry_shield,
    campaign_ledger_row,
    VERDICT_CLEARS,
    VERDICT_CONDITIONAL,
    VERDICT_FAILS,
)

J_OP = 9000.0   # A/m² — drum-class operating point
FE = 0.85
T = 50.0


@pytest.fixture(scope="module")
def result():
    return evaluate_drum_hydriding()


# ── flux chain identities ────────────────────────────────────────────

def test_her_partial_current_identity():
    assert her_partial_current_A_m2(1000.0, 0.85) == pytest.approx(150.0)
    assert her_partial_current_A_m2(1000.0, 1.0) == 0.0


def test_absorbed_flux_factors_multiplicative():
    full = absorbed_h_flux_mol_m2_s(J_OP, FE, entry_frac=0.1, shield_frac=0.1)
    half_entry = absorbed_h_flux_mol_m2_s(J_OP, FE, entry_frac=0.05, shield_frac=0.1)
    half_shield = absorbed_h_flux_mol_m2_s(J_OP, FE, entry_frac=0.1, shield_frac=0.05)
    assert half_entry == pytest.approx(full / 2.0, rel=1e-12)
    assert half_shield == pytest.approx(full / 2.0, rel=1e-12)


def test_absorbed_flux_is_her_over_faraday():
    j = absorbed_h_flux_mol_m2_s(2000.0, 0.9, entry_frac=1.0, shield_frac=1.0)
    from models.electrochemistry import FARADAY
    assert j == pytest.approx(2000.0 * 0.1 / FARADAY, rel=1e-12)


# ── diffusion / solubility ───────────────────────────────────────────

def test_diffusivity_arrhenius_decreases_with_T_drop():
    assert h_diffusivity_m2_s(25.0) < h_diffusivity_m2_s(60.0)
    assert h_diffusivity_m2_s(90.0) > h_diffusivity_m2_s(60.0)


def test_diffusion_depth_root_t():
    d4 = diffusion_depth_m(4.0, T)
    d1 = diffusion_depth_m(1.0, T)
    assert d4 == pytest.approx(2.0 * d1, rel=1e-12)


def test_hydride_sinks_more_h_than_solid_solution():
    assert hydride_h_mol_m3() > 10.0 * tss_h_mol_m3()


# ── onset & case growth ──────────────────────────────────────────────

def test_excess_inventory_grows_linearly_at_long_time():
    e1 = excess_inventory_mol_m2(1000.0, J_OP, FE, T, 0.1, 0.5)
    e2 = excess_inventory_mol_m2(2000.0, J_OP, FE, T, 0.1, 0.5)
    # absorbed term dominates over √t capacity here → monotone
    assert e2 > e1


def test_no_case_below_onset():
    # negligible flux → absorbed never beats capacity → no case, onset inf
    case = hydride_case_depth_m(1e5, J_OP, FE, T, 1e-9, 1e-9)
    assert case == 0.0
    assert math.isinf(hydride_onset_hours(J_OP, FE, T, 1e-9, 1e-9))
    assert math.isinf(campaign_life_hours(J_OP, FE, T, 1e-9, 1e-9))


def test_case_appears_after_onset():
    # interior onset (flux small enough that absorbed and dissolved inventory
    # cross at t > 1e-3 h, so the bisector is not pinned to its floor)
    onset = hydride_onset_hours(J_OP, FE, T, 0.01, 0.1)
    assert math.isfinite(onset) and onset > 1.0e-3
    assert hydride_case_depth_m(0.5 * onset, J_OP, FE, T, 0.01, 0.1) == 0.0
    assert hydride_case_depth_m(4.0 * onset, J_OP, FE, T, 0.01, 0.1) > 0.0


def test_campaign_life_exceeds_onset_when_case_required():
    onset = hydride_onset_hours(J_OP, FE, T, 0.1, 0.5)
    life = campaign_life_hours(J_OP, FE, T, 0.1, 0.5)
    assert math.isfinite(life)
    assert life > onset
    # at life, the case must equal the crit depth
    from models.anchors import get_anchor
    crit = get_anchor("TI_HYD_CRIT_CASE_UM").value
    case_um = hydride_case_depth_m(life, J_OP, FE, T, 0.1, 0.5) * 1e6
    assert case_um == pytest.approx(crit, rel=1e-6)


def test_more_hydrogen_shorter_life():
    life_low = campaign_life_hours(J_OP, FE, T, 0.01, 0.1)
    life_high = campaign_life_hours(J_OP, FE, T, 0.2, 0.1)
    assert life_high < life_low
    life_fe = campaign_life_hours(J_OP, 0.95, T, 0.1, 0.1)
    assert life_fe > campaign_life_hours(J_OP, FE, T, 0.1, 0.1)


def test_case_rate_positive_when_hydriding():
    rate = case_growth_rate_um_per_1000h(J_OP, FE, T, 0.1, 0.5)
    assert rate > 0.0


# ── verdict bands ────────────────────────────────────────────────────

@pytest.mark.parametrize("life,target,want", [
    (math.inf, 1000.0, VERDICT_CLEARS),
    (3000.0, 1000.0, VERDICT_CLEARS),
    (2999.9, 1000.0, VERDICT_CONDITIONAL),
    (1000.0, 1000.0, VERDICT_CONDITIONAL),
    (999.9, 1000.0, VERDICT_FAILS),
])
def test_campaign_verdict_bands(life, target, want):
    assert campaign_verdict(life, target) == want


# ── G_c drift multiplier ─────────────────────────────────────────────

def test_gc_multiplier_untouched_before_onset():
    assert gc_drift_multiplier(1.0, J_OP, FE, T, 1e-9, 1e-9) == 1.0


def test_gc_multiplier_between_floor_and_one():
    from models.anchors import get_anchor
    floor = get_anchor("TI_HYD_GC_FLOOR_FRAC").value
    vals = [
        gc_drift_multiplier(t, J_OP, FE, T, 0.1, 0.5)
        for t in (10.0, 100.0, 500.0, 1000.0, 2000.0, 5000.0)
    ]
    for v in vals:
        assert floor - 1e-12 <= v <= 1.0 + 1e-12
    assert vals[-1] <= vals[0]          # monotone non-increasing with damage


def test_gc_multiplier_reaches_floor_at_crit():
    from models.anchors import get_anchor
    life = campaign_life_hours(J_OP, FE, T, 0.1, 0.5)
    assert math.isfinite(life)
    assert gc_drift_multiplier(life, J_OP, FE, T, 0.1, 0.5) == pytest.approx(
        get_anchor("TI_HYD_GC_FLOOR_FRAC").value, rel=1e-6
    )


# ── live rederivation hooks ──────────────────────────────────────────

def test_result_tracks_live_conditions(result):
    cond = cell_architecture.OperatingConditions(faradaic_efficiency=0.99)
    hi_fe = evaluate_drum_hydriding(conditions=cond)
    base_fe = result.faradaic_efficiency
    assert hi_fe.her_partial_A_m2 < result.her_partial_A_m2
    # campaign life at higher FE must be no shorter
    l_hi = hi_fe.campaign_life_h
    l_base = result.campaign_life_h
    assert (
        (math.isinf(l_hi) or math.isinf(l_base))
        or l_hi >= l_base
    )
    assert base_fe != hi_fe.faradaic_efficiency


def test_temperature_feeds_diffusivity(result):
    assert result.h_diffusivity_m2_s == pytest.approx(
        h_diffusivity_m2_s(result.temperature_C)
    )


# ── sweep + design-target contract ───────────────────────────────────

def test_sweep_grid_shape_and_monotonicity():
    rows = sweep_entry_shield(
        entry_fracs=[1e-3, 1e-1], shield_fracs=[1e-4, 1e-1]
    )
    assert len(rows) == 4
    by_key = {(r["entry_frac"], r["shield_frac"]): r for r in rows}
    life = lambda k, s: by_key[(k, s)]["campaign_life_h"]
    num = lambda v: math.inf if v == "inf" else v
    assert num(life(1e-1, 1e-1)) <= num(life(1e-1, 1e-4))
    assert num(life(1e-1, 1e-1)) <= num(life(1e-3, 1e-1))


def test_design_target_scales_with_required_life():
    easy = design_target_entry_shield(target_h=100.0, safety_multiple=1.0)
    hard = design_target_entry_shield(target_h=3000.0, safety_multiple=3.0)
    assert 0.0 <= hard <= easy
    # at the returned product the life must actually clear target×safety
    if hard > 0.0:
        life = campaign_life_hours(1.0, 0.85, 50.0, hard, 1.0)
        assert math.isinf(life) or life >= 3.0 * 3000.0


# ── feeds / CLI / flags ──────────────────────────────────────────────

def test_feed_to_adhesion_peel_shape():
    feed = feed_to_adhesion_peel(500.0)
    assert 0.0 < feed["gc_hydride_multiplier"] <= 1.0
    assert feed["hydride_case_um"] >= 0.0
    assert feed["t_hours"] == 500.0


def test_campaign_ledger_row_shape():
    row = campaign_ledger_row(500.0)
    for k in ("campaign_h", "hydride_case_um", "gc_hydride_multiplier",
              "campaign_life_h", "crit_case_um", "verdict_1000h"):
        assert k in row


def test_screening_flag_and_scope():
    assert SCREENING_FLAG == "unvalidated (L1)"
    scope = model_scope()
    assert any("cell_architecture" in s for s in scope["live_derivations"])
    assert "θ_H surface-state coupling" in " ".join(scope["out_of_scope"])


def test_cli_runs(capsys):
    main(["--entry", "0.1", "--shield", "0.1"])
    out = capsys.readouterr().out
    assert "ti_hydriding" in out
    assert "campaign life" in out
    assert "design target" in out
