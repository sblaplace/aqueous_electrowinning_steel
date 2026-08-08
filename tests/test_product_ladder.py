"""Tests for the product value ladder (models/product_ladder.py).

The suite's defining property is *rederivation liveness*: the ladder must
move when the models underneath it move.  The tests therefore perturb the
underlying modules (architecture cost defaults, shared operating conditions,
anchor values, gate-module presence) and assert the ladder responds — a
ladder with frozen numbers would fail each of these.
"""

import dataclasses
import json
import types
from pathlib import Path

import pytest

from models import cell_architecture, electrochemistry, technoeconomic
from models.anchors import get_anchor
from models.product_ladder import (
    GATE_REGISTRY,
    RUNGS,
    comparison_table,
    evaluate_ladder,
    evaluate_rung,
    gate_status,
    model_scope,
    provenance,
    render_markdown,
    write_artifacts,
)


# ── static sanity ───────────────────────────────────────────────────────

def test_rungs_reference_real_architectures_and_gates():
    for rung in RUNGS.values():
        assert rung.architecture_id in cell_architecture.ARCHITECTURES
        for gid in rung.gates:
            assert gid in GATE_REGISTRY, f"{rung.id}: unknown gate {gid}"
    # every registered gate is used by at least one rung (no orphan registry)
    used = {g for r in RUNGS.values() for g in r.gates}
    assert used == set(GATE_REGISTRY), (
        f"orphan gates in registry: {set(GATE_REGISTRY) - used}"
    )


def test_price_anchors_resolve_and_band_is_monotonic():
    for rung in RUNGS.values():
        a = get_anchor(rung.price.anchor_key)
        assert a.uncertainty > 0.0
        assert rung.price.low < rung.price.mid < rung.price.high
        assert rung.price.mid == pytest.approx(a.value)
        # the band is exactly the anchor uncertainty by construction
        assert rung.price.high - rung.price.low == pytest.approx(2 * a.uncertainty)


def test_melt_feed_is_cheapest_rung_and_foils_price_higher():
    assert RUNGS["flake_feed"].price.mid < RUNGS["annealed_foil"].price.mid
    assert RUNGS["flake_feed"].price.mid < RUNGS["own_melt_bar"].price.mid


# ── economics sanity at defaults ─────────────────────────────────────────

def test_default_ladder_is_finite_and_cost_stack_consistent():
    result = evaluate_ladder()
    for r in result.rungs:
        assert r.areal_productivity_t_m2_yr > 0
        assert r.installed_cost_per_m2 > 0
        assert r.margin_share_of_price == pytest.approx(
            r.margin_per_t / r.price_mid_per_t
        )
        # cost stack closes
        assert r.total_cost_per_t == pytest.approx(
            r.energy_cost_per_t + r.cash_other_per_t + r.capital_charge_per_t
        )
        # margin per m²-year is margin per t times productivity
        assert r.margin_per_m2_yr == pytest.approx(
            r.margin_per_t * r.areal_productivity_t_m2_yr
        )
        # banded margins bracket the mid
        assert r.margin_at_low <= r.margin_per_t <= r.margin_at_high
        # verdict is one of the declared values
        assert r.verdict.split("-")[0] in ("clears", "marginal", "stalls")


def test_dc_energy_matches_live_electrochemistry_call():
    cond = cell_architecture.OperatingConditions()
    expected = electrochemistry.specific_energy_kWh_per_t(
        cond.cell_voltage_V, cond.faradaic_efficiency
    )
    r = evaluate_rung(RUNGS["flake_feed"], cond)
    assert r.dc_energy_kWh_per_t == pytest.approx(expected)
    # and it is bounded by the program kill-criterion framing
    assert r.dc_energy_kWh_per_t < 4500.0


def test_electricity_default_comes_from_technoeconomic():
    live = technoeconomic.OPEXModel().electricity_price_kWh
    r = evaluate_rung(RUNGS["flake_feed"])
    expected_energy_cost = (
        (r.dc_energy_kWh_per_t + r.aux_energy_kWh_per_t) * live
    )
    assert r.energy_cost_per_t == pytest.approx(expected_energy_cost)


# ── rederivation liveness: the core user requirement ─────────────────────

def test_ladder_moves_when_architecture_costs_move():
    """Double the rotating-cylinder cost lines → flake economics move."""
    spec = cell_architecture.ARCHITECTURES["rotating_cylinder"]
    try:
        before = evaluate_rung(RUNGS["flake_feed"])
        cell_architecture.ARCHITECTURES["rotating_cylinder"] = dataclasses.replace(
            spec,
            electrode_cost_per_m2=spec.electrode_cost_per_m2 * 2.0,
            separator_cost_per_m2=spec.separator_cost_per_m2 * 2.0,
            hardware_cost_per_m2=spec.hardware_cost_per_m2 * 2.0,
            harvesting_cost_per_m2=spec.harvesting_cost_per_m2 * 2.0,
        )
        after = evaluate_rung(RUNGS["flake_feed"])
        assert after.capital_charge_per_t == pytest.approx(
            2.0 * before.capital_charge_per_t, rel=1e-6
        )
        assert after.margin_per_t < before.margin_per_t
        assert after.required_zinc_multiple > before.required_zinc_multiple
    finally:
        cell_architecture.ARCHITECTURES["rotating_cylinder"] = spec


def test_ladder_moves_when_shared_conditions_move():
    """A worse shared cell-voltage/FE state must cost more on every rung."""
    base = evaluate_ladder()
    worse = cell_architecture.OperatingConditions(
        cell_voltage_V=4.0, faradaic_efficiency=0.60
    )
    hot = evaluate_ladder(worse)
    for b, h in zip(base.rungs, hot.rungs):
        assert h.dc_energy_kWh_per_t > b.dc_energy_kWh_per_t
        assert h.energy_cost_per_t > b.energy_cost_per_t
        assert h.total_cost_per_t > b.total_cost_per_t
        assert h.margin_per_t < b.margin_per_t
        # productivity drops with FE (Faraday arithmetic, live)
        assert h.areal_productivity_t_m2_yr < b.areal_productivity_t_m2_yr


def test_price_artifact_claim_holds_at_extremes():
    """The headline: the README's '~5×' is flake-economics, not cell physics.

    Three live, constant-free claims (they harden if the architecture screen
    is recalibrated — that is the point of rederiving them):

    1. At commodity flake price the drum's capital share is ~5× the rotating
       cylinder's — this *is* the program's 5× imperative, stated precisely.
    2. The same drum's capital share falls with price inverse-linearly, into
       single digits at the top of the ladder.
    3. required_zinc_multiple is exactly ∝ 1/price for a fixed architecture.
    """
    result = evaluate_ladder()
    flake_price = RUNGS["flake_feed"].price.mid
    rc = next(r for r in result.rungs
              if r.architecture_id == "rotating_cylinder")
    drums = [r for r in result.rungs if r.architecture_id == "drum_and_strip"]
    # claim 1: drum capital charge / RC capital charge *priced at flake* —
    # the price cancels, leaving the pure architecture ratio (the "~5×").
    ratio = ((drums[0].capital_charge_per_t / flake_price)
             / (rc.capital_charge_per_t / flake_price))
    assert 2.0 < ratio < 8.0  # the "~5×" imperative, recovered
    # claim 2: every drum rung's capital share is inverse-linear in price and
    # in single digits (or below) once off commodity iron
    shares = sorted((r.capital_share_of_price, r.price_mid_per_t) for r in drums)
    assert shares[0][0] < 0.025
    for (s1, p1), (s2, p2) in zip(shares, shares[1:]):
        assert s1 / s2 == pytest.approx(p2 / p1, rel=1e-6)
    # claim 3: exact inverse-proportionality of the required multiple
    demo = result.price_artifact_demo
    drum_rows = {d["price_per_t"]: d for d in demo
                 if d["architecture"] == "drum_and_strip"}
    p_lo, p_hi = min(drum_rows), max(drum_rows)
    assert (drum_rows[p_lo]["required_zinc_multiple"]
            / drum_rows[p_hi]["required_zinc_multiple"]) == pytest.approx(
        p_hi / p_lo, rel=0.05  # demo values are rounded to 2 decimals
    )


def test_verdict_and_capital_share_consistency():
    result = evaluate_ladder()
    for r in result.rungs:
        assert r.capital_share_of_price == pytest.approx(
            r.capital_charge_per_t / r.price_mid_per_t
        )
        assert r.min_price_for_budget_per_t == pytest.approx(
            r.capital_charge_per_t / result.capital_budget_fraction
        )


def test_price_override_flips_verdict_monotonically():
    foil = RUNGS["annealed_foil"]
    cheap = evaluate_rung(foil, price_override=RUNGS["flake_feed"].price.mid)
    rich = evaluate_rung(foil, price_override=foil.price.high)
    # margins rise with price; the breakeven stays below the band
    assert cheap.margin_per_t < rich.margin_per_t
    assert rich.verdict.split("-")[0] == "clears"
    assert cheap.breakeven_product_price_per_t < foil.price.low
    # a deep price cut must eventually stall the rung
    floored = evaluate_rung(foil, price_override=cheap.total_cost_per_t - 1.0)
    assert floored.verdict.startswith("stalls")


def test_anneal_energy_is_live_derived_from_thermomechanical():
    """Anneal op energy must equal the thermomechanical model's number."""
    from models.thermomechanical import ThermomechanicalModel

    live = ThermomechanicalModel().anneal_energy_kWh_per_kg() * 1000.0
    from models.product_ladder import POST_OPS

    assert POST_OPS["op_anneal"].energy_kWh_per_t == pytest.approx(live)
    assert "live" in POST_OPS["op_anneal"].notes


# ── gate resolution (the V6-proposal tracker) ───────────────────────────

def test_gate_status_probes_live_module_tree():
    # an existing L1 module resolves as modelled/unvalidated
    peel = gate_status("g_peel")
    assert peel["exists"] is True
    assert peel["state"].startswith("modelled")
    # V6 §1.1 landed: the idle-corrosion gate flipped unmodelled → modelled
    corr = gate_status("g_oc_corrosion")
    assert corr["exists"] is True
    assert corr["state"] == "modelled (L1)"
    assert corr["module"] == "deposit_corrosion"
    # V6 §1.4 landed: the densification gate flipped the same way
    briq = gate_status("g_briquet")
    assert briq["exists"] is True
    assert briq["state"] == "modelled (L1)"
    assert briq["module"] == "briquetting"
    # landed V6 modules resolve as modelled (L1) — strain aging flips here
    assert gate_status("g_deposit_aging")["state"] == "modelled (L1)"
    assert gate_status("g_strain_aging")["state"] == "modelled (L1)"
    still = gate_status("g_magnetic")
    assert still["state"] == "unmodelled"
    assert still["module"] == "magnetic_properties"


def test_gate_status_flips_when_module_lands():
    """Simulate implementing a V6 proposal: status flips, ladder follows."""
    from models import product_ladder

    fake = types.ModuleType("models.magnetic_properties")
    fake.SCREENING_FLAG = "unvalidated (L1)"
    real_import = product_ladder.importlib.import_module

    def fake_import(name, *a, **k):
        if name == "models.magnetic_properties":
            return fake
        return real_import(name, *a, **k)

    try:
        product_ladder.importlib.import_module = fake_import
        flipped = gate_status("g_magnetic")
        assert flipped["exists"] is True
        assert flipped["state"] == "modelled (L1)"
        r = evaluate_rung(RUNGS["magnetic_foil"])
        names_unmodelled = [g["name"] for g in r.gate_rows
                            if g["state"] == "unmodelled"]
        assert all("magnetic" not in n.lower() for n in names_unmodelled)
    finally:
        product_ladder.importlib.import_module = real_import


# ── artifacts & provenance ──────────────────────────────────────────────

def test_write_artifacts_roundtrip(tmp_path: Path):
    out_json = tmp_path / "report.json"
    out_doc = tmp_path / "LADDER.md"
    paths = write_artifacts(out_json, out_doc)
    payload = json.loads(Path(paths["json"]).read_text())
    assert payload["screening_flag"].startswith("unvalidated")
    assert payload["_provenance"]["mode"] == "full-grade"
    assert set(payload["_provenance"]["source_hashes"]) >= {
        "models/product_ladder.py", "models/cell_architecture.py"
    }
    assert len(payload["rungs"]) == len(RUNGS)
    doc = Path(paths["doc"]).read_text()
    assert "provenance" in doc
    for rung_id in RUNGS:
        assert rung_id in doc


def test_generated_markdown_tracks_model_changes(tmp_path: Path):
    base = evaluate_ladder()
    md1 = render_markdown(base)
    shifted = evaluate_ladder(
        cell_architecture.OperatingConditions(cell_voltage_V=5.0)
    )
    md2 = render_markdown(shifted)
    assert md1 != md2  # the decision document is regenerated from the result
    assert "5.0" in md2


def test_provenance_hashes_present_and_typed():
    prov = provenance()
    assert prov["mode"] == "full-grade"
    for _, h in prov["source_hashes"].items():
        assert isinstance(h, str) and len(h) == 16


def test_comparison_table_and_scope_declared():
    table = comparison_table(evaluate_ladder())
    for rung_id in RUNGS:
        assert rung_id in table
    scope = model_scope()
    assert scope["screening_flag"].startswith("unvalidated")
    assert scope["live_derivations"] and scope["explicitly_out_of_scope"]
