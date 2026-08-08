"""Tests for models/ladder_tea.py — ladder × full-TEA wiring.

The scientific contract under test: the ladder's contribution-margin screen
excludes a known set of cost lines; the full TEA includes them.  The
screening gap must (a) itemise exactly to the excluded lines, (b) leave the
rung ranking auditable (flip count exposed), and (c) recover the ladder's
margin when the excluded lines are zeroed — proving the two models close on
the same cost stack rather than merely being correlated.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from models import cell_architecture, technoeconomic
from models.ladder_tea import (
    DEFAULT_PLANT_CAPACITY_T_YR,
    SCREENING_FLAG,
    comparison_table,
    evaluate_ladder_tea,
    evaluate_rung_tea,
    gap_table,
    main,
    model_scope,
    provenance,
    render_markdown,
)
from models.product_ladder import RUNGS, evaluate_rung


@pytest.fixture(scope="module")
def result():
    return evaluate_ladder_tea()


@pytest.fixture(scope="module")
def flake(result):
    return next(r for r in result.rungs if r.rung_id == "flake_feed")


# ── structural sanity ────────────────────────────────────────────────

def test_all_rungs_evaluated(result):
    assert {r.rung_id for r in result.rungs} == set(RUNGS.keys())


def test_screening_flag_and_scope():
    assert SCREENING_FLAG == "unvalidated (L1)"
    scope = model_scope()
    assert scope["live_derivations"]
    assert "construction schedule, working capital, tax, financing" in " ".join(
        scope["out_of_scope"]
    )


def test_capacity_scenario_default(result):
    assert result.capacity_t_yr == DEFAULT_PLANT_CAPACITY_T_YR


# ── the screening-gap identity ──────────────────────────────────────

def test_tea_margin_never_above_ladder_margin(result):
    for r in result.rungs:
        assert r.margin_per_t <= r.ladder_margin_per_t + 1e-6
        assert r.screening_gap_per_t >= -1e-6


def test_gap_itemises_to_excluded_lines(flake):
    """gap == Σ excluded lines (basis residual absorbs ladder rounding only)."""
    total = sum(flake.excluded_lines_per_t.values())
    assert flake.screening_gap_per_t == pytest.approx(total, rel=1e-9, abs=1e-9)
    assert abs(flake.excluded_lines_per_t["basis residual"]) < 1.0  # $/t


def test_zeroed_exclusions_recover_ladder_margin(flake):
    """Knob test: zero every excluded line → TEA closes on the ladder cost."""
    om = technoeconomic.OPEXModel(
        electrolyte_makeup_per_t_Fe=0.0,
        anode_replacement_cost_per_m2_yr=0.0,
        ore_cost_per_t_Fe=0.0,
        grinding_energy_kWh_per_t=0.0,
        water_cost_per_t_Fe=0.0,
        maintenance_pct_capex=0.0,
        labor_cost_per_yr=0.0,
        insurance_pct_capex=0.0,
        overhead_pct=0.0,
    )
    cm = technoeconomic.CAPEXModel(
        leaching_cost_per_tpy=0.0, grinding_cost_per_tpy=0.0,
    )
    rung = RUNGS[flake.rung_id]
    r = evaluate_rung_tea(rung, opex_model=om, capex_model=cm)
    ladder = evaluate_rung(rung)
    assert r.full_cost_per_t == pytest.approx(ladder.total_cost_per_t, rel=1e-9)
    assert r.margin_per_t == pytest.approx(ladder.margin_per_t, rel=1e-9)
    assert r.screening_gap_per_t == pytest.approx(0.0, abs=1e-6)


# ── live rederivation (no constants frozen into this module) ─────────

def test_electricity_price_flows_through_all_rungs():
    lo = evaluate_ladder_tea(electricity_price_kWh=0.02)
    hi = evaluate_ladder_tea(electricity_price_kWh=0.08)
    overhead = technoeconomic.OPEXModel().overhead_pct  # 10% rides on variable
    for a, b in zip(lo.rungs, hi.rungs):
        delta_kwh = (
            a.dc_energy_kWh_per_t + a.postop_energy_kWh_per_t
            + a.grinding_energy_kWh_per_t
        )
        assert b.margin_per_t - a.margin_per_t == pytest.approx(
            -(0.08 - 0.02) * delta_kwh * (1.0 + overhead), rel=1e-9
        )


def test_architecture_cost_perturbation_propagates(monkeypatch, flake):
    spec = cell_architecture.ARCHITECTURES[flake.architecture_id]
    bumped = dataclasses.replace(spec, electrode_cost_per_m2=spec.electrode_cost_per_m2 + 1000.0)
    monkeypatch.setitem(cell_architecture.ARCHITECTURES, flake.architecture_id, bumped)
    r = evaluate_rung_tea(RUNGS[flake.rung_id])
    assert r.installed_cost_per_m2 > flake.installed_cost_per_m2
    assert r.capital_charge_per_t > flake.capital_charge_per_t


def test_labour_dilutes_with_capacity():
    small = evaluate_ladder_tea(capacity_t_yr=10_000.0)
    large = evaluate_ladder_tea(capacity_t_yr=1_000_000.0)
    for a, b in zip(small.rungs, large.rungs):
        assert b.labor_usd_per_t < a.labor_usd_per_t
        assert b.full_cost_per_t < a.full_cost_per_t


# ── economics coherence ─────────────────────────────────────────────

def test_price_band_ordering(result):
    for r in result.rungs:
        assert r.margin_at_low <= r.margin_per_t <= r.margin_at_high
        assert r.price_low_per_t <= r.price_mid_per_t <= r.price_high_per_t


def test_npv_sign_matches_margin_sign(result):
    for r in result.rungs:
        if r.margin_per_t > 0:
            assert r.npv_usd is not None and r.npv_usd > 0
            assert r.payback_yr is not None and r.payback_yr > 0
        else:
            assert r.npv_usd is not None and r.npv_usd < 0
            assert r.payback_yr is None
            assert r.irr is None


def test_irr_within_solver_bounds(result):
    for r in result.rungs:
        if r.irr is not None:
            assert -0.9 < r.irr <= 10.0


def test_verdict_vocabulary(result):
    for r in result.rungs:
        assert r.verdict in {"clears", "marginal", "stalls"}
        if r.margin_per_t <= 0:
            assert r.verdict == "stalls"


# ── rank-stability contract ─────────────────────────────────────────

def test_rank_orders_and_flip_bounds(result):
    n = len(result.rungs)
    assert sorted(result.rank_by_ladder_margin) == sorted(set(RUNGS))
    assert sorted(result.rank_by_tea_margin) == sorted(set(RUNGS))
    assert 0 <= result.n_pairwise_flips <= n * (n - 1) // 2
    assert result.ranking_preserved == (result.n_pairwise_flips == 0)


# ── dark-mill site wiring ────────────────────────────────────────────

def test_site_overrides_price_and_labour(flake):
    from models.dark_mill import EXAMPLE_SITES  # lazy: heavy import

    site = next(iter(EXAMPLE_SITES.values()))
    r = evaluate_rung_tea(
        RUNGS[flake.rung_id], site=site, capacity_t_yr=50_000.0
    )
    assert r.electricity_price_kWh == pytest.approx(site.grid.effective_price_kWh)
    assert r.labor_cost_per_yr == pytest.approx(site.labor_cost_per_yr)
    assert r.labor_usd_per_t == pytest.approx(site.labor_cost_per_yr / 50_000.0)


# ── reporting / artifacts ────────────────────────────────────────────

def test_tables_render(result):
    txt = comparison_table(result)
    assert "flake_feed" in txt and "magnetic_foil" in txt
    gaps = gap_table(result)
    assert "labour" in gaps or "ore feedstock" in gaps


def test_markdown_contains_decision_sections(result):
    md = render_markdown(result)
    assert "Ladder × TEA" in md
    assert "## 5. Ranking verdict" in md
    assert "pairwise order flips" in md.lower() or "Pairwise order flips" in md


def test_provenance_hashes_match_live_sources():
    prov = provenance()
    import hashlib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name, h in prov["source_hashes"].items():
        live = hashlib.sha256((root / name).read_bytes()).hexdigest()[:16]
        assert live == h, f"provenance stale for {name}"
    assert prov["mode"] == "full-grade"


def test_json_roundtrip(result, tmp_path):
    data = result.to_dict()
    data["_provenance"] = provenance()
    out = tmp_path / "r.json"
    out.write_text(json.dumps(data, indent=2) + "\n")
    back = json.loads(out.read_text())
    assert back["rungs"][0]["rung_id"] == data["rungs"][0]["rung_id"]
    assert back["_provenance"]["artifact"] == "ladder_tea"


def test_cli_runs_and_writes_json(tmp_path, capsys):
    out = tmp_path / "cli.json"
    main(["--capacity", "50000", "--rung", "flake_feed", "--json-out", str(out)])
    captured = capsys.readouterr()
    assert "ladder_tea" in captured.out
    payload = json.loads(out.read_text())
    assert len(payload["rungs"]) == 1
    assert payload["rungs"][0]["rung_id"] == "flake_feed"
