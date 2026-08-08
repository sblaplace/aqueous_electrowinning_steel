"""Product value ladder — *which* product should the cell make, recomputed live.

Why this module exists
----------------------
The program's page-1 decision (``docs/RESEARCH_PROGRAM.md``) is "feedstock
first, product later" — Option A (melt-shop feedstock) vs. Option B (direct
steel sheet/foil).  That decision has so far been argued **textually**.  This
module makes it **computed and rederiving**: it prices each candidate product
rung (flake feed → own-melt bar → annealed foil → structural sheet → specialty
powders) against:

* the cell's *live* physics and cost state — productivity, capital charge and
  DC energy are imported from ``cell_architecture`` / ``electrochemistry`` /
  ``thermomechanical`` / ``technoeconomic`` at call time, so the ladder
  re-derives automatically whenever those models are updated (calibrated
  constants, new architectures, changed defaults — the ladder moves);
* the *live* modelling state of each rung's gating physics — gates resolve
  against the module tree at call time, so a V6-proposed module
  (``docs/CHEM_PHYS_IMPROVEMENTS_V6.md``) flips from ``unmodelled`` to
  ``modelled (L1)`` the day it lands; and
* product-price anchors — the one genuinely *new* input — which live in
  ``models/anchors.py`` with the repo's usual provenance rule.

What this module is NOT
-----------------------
* A plant TEA.  ``technoeconomic.py`` remains the full-plant model.  The
  ladder computes **contribution-margin screening** per tonne and per m²·yr
  of cell: product price − (DC electricity + post-cell unit-operation energy
  and cash + installed cell capital charge).  Feedstock, labour, BOP and
  working capital live in the TEA; here they are deliberately held constant
  *across* rungs so the product-form comparison is apples-to-apples.
* A substitute for customer qualification.  Each rung carries a
  ``qualification`` text field noting those (e.g. structural certification
  takes years); the economics are screened *before* qualification cost.

The headline computation
------------------------
The README's "~5× a zinc tankhouse" imperative assumes output priced at
commodity iron ($400–600/t).  This module re-derives that imperative per rung
as ``required_zinc_multiple`` = the areal-productivity multiple of the zinc
tankhouse benchmark needed for the installed cell capital to stay within a
fixed budget *fraction of product price*.  Commodity-price rungs reproduce
the 5× imperative; higher-value rungs dissolve it.  The claim "the
productivity panic is a price artefact" is thus a number the suite recomputes
on every model update, in ``price_artifact_demo()`` / the generated
``docs/PRODUCT_VALUE_LADDER.md``.

Screening flag
--------------
L1.  Product prices are screening bands anchored to public benchmarks
(Fastmarkets-style reportage, USGS, vendor list prices) with wide
uncertainties; replace with quoted offtakes before any investment decision.
Unit-operation energies/cash are screening anchors (L1).  Verdict
fractions are screening thresholds, exposed as keyword arguments.

References
----------
* docs/RESEARCH_PROGRAM.md — Option A / Option B decision text.
* docs/CHEM_PHYS_IMPROVEMENTS_V6.md — the unmodelled gate modules this ladder
  tracks (deposit_corrosion, melt_balance, briquetting, …).
* models/cell_architecture.py — kill criterion #3 and the zinc benchmark.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import cell_architecture, electrochemistry, technoeconomic
from .anchors import get_anchor

SCREENING_FLAG = "unvalidated (L1)"

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON_PATH = ROOT / "experiments" / "data" / "product_ladder_report.json"
DEFAULT_DOC_PATH = ROOT / "docs" / "PRODUCT_VALUE_LADDER.md"

# Provenance set: the files whose current state defines the ladder's numbers.
_STAMPED_SOURCES = (
    "models/product_ladder.py",
    "models/cell_architecture.py",
    "models/electrochemistry.py",
    "models/technoeconomic.py",
    "models/anchors.py",
    "models/thermomechanical.py",
)

# Screening threshold: installed-cell capital charge must stay within this
# fraction of product price for the rung to "clear" kill criterion #3 at that
# price.  Screening central value 10%: after non-cell costs at commodity
# prices (electricity ~25%, feedstock/labour/BOP per technoeconomic.py), a
# ~10% capital-share headroom is the conventional allowance.  Note the
# README's "~5×" imperative is recoverable from this module *without* this
# constant (see render_markdown §4): it is the cross-architecture ratio of
# capital shares at commodity price, computed live.
CAPITAL_BUDGET_FRACTION = 0.10

# Screening threshold for the rung verdict: rung "clears" only if the
# contribution margin is at least this fraction of product price.
MARGIN_VERDICT_FRACTION = 0.10


# ════════════════════════════════════════════════════════════════════════
#  Gate registry — the modelling state of what each rung needs
# ════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class GateRef:
    """A physics/engineering gate and the module that owns (or should own) it.

    ``module`` is a short name under ``models.``; when the module exists its
    SCREENING_FLAG is read at call time, so gate status tracks the codebase.
    A module that does not (yet) exist resolves to ``unmodelled`` — these are
    the V6 proposals (``docs/CHEM_PHYS_IMPROVEMENTS_V6.md``).
    """

    id: str
    name: str
    module: str
    note: str = ""


GATE_REGISTRY: Dict[str, GateRef] = {
    g.id: g
    for g in (
        GateRef("g_fe_gate", "FE ≥ 70 % at j ≥ 300 mA/cm² (kill #1)",
                "diffusion_layer_1d",
                "FE engine; calibration pending (Q3/RDE campaign)."),
        GateRef("g_energy_gate", "net DC ≤ 4,000 kWh/t Fe (kill #2)",
                "cell_physics",
                "Voltage decomposition; needs divided-cell measurement."),
        GateRef("g_arch_gate", "continuous-harvest cell ≥ target productivity (kill #3)",
                "cell_architecture", "Screen only; hardware evidence absent."),
        GateRef("g_loop", "closed electrolyte loop & purge policy",
                "closed_loop", "CSTR screen; long-run drift unmeasured."),
        GateRef("g_peel", "coherent foil peels from the drum",
                "adhesion_peel", "Coupon test specified, not yet run."),
        GateRef("g_stress_h", "deposit stress & hydrogen control in-window",
                "internal_stress", "σ(h) screen + bent-strip protocol pending."),
        GateRef("g_grade", "deposit → anneal → steel grade routing",
                "as_deposited_grade", "Coupling seam built (V5/A2); L1."),
        GateRef("g_strain_aging", "N pickup / Lüders-band forming gate",
                "strain_aging", "V6 §7.1 proposal — UNMODELLED."),
        GateRef("g_deposit_aging", "RT aging between harvest & metrology",
                "deposit_aging", "V6 §5.2 proposal — UNMODELLED."),
        GateRef("g_drum_life", "drum campaign life (oxide + hydriding)",
                "ti_hydriding",
                "V6 §4.1 proposal — UNMODELLED (oxide side in substrate_passivation)."),
        GateRef("g_oc_corrosion", "idle corrosion / ferric etch of deposit",
                "deposit_corrosion",
                "V6 §1.1 implemented (L1) — mixed-potential idle corrosion + "
                "ferric etch; run_record predicted ledger terms + "
                "closed_loop campaign accounting."),
        GateRef("g_product_ox", "post-harvest oxidation & passivation spec",
                "product_oxidation", "V6 §1.2 proposal — UNMODELLED."),
        GateRef("g_rinse", "rinse carryover → charge sulfur",
                "rinse_carryover",
                "V6 §1.3 implemented (L1) — Landau–Levich film / cake "
                "liquor + counter-current cascade → charge S/Na/B budgets, "
                "conductivity endpoint; feeds melt_balance live."),
        GateRef("g_briquet", "densification / briquetting product-form gate",
                "briquetting", "V6 §1.4 proposal — UNMODELLED."),
        GateRef("g_melt_balance", "melt-shop remelt verdict (yield/boil/slag)",
                "melt_balance",
                "V6 §1.5 implemented (L1) — EAF/induction remelt verdict "
                "with live O and H inputs."),
        GateRef("g_casting", "in-house melt + solidification/casting block",
                "strip_casting",
                "Option A.5 gap beyond V6 — UNMODELLED (liquidus, segregation, "
                "cast-ability, incl. flotation)."),
        GateRef("g_purity", "bath impurity / purification chain",
                "purification", "Train screen exists; feed-fingerprint L1."),
        GateRef("g_form_factor", "porosity / surface spec for anode feed",
                "bubble_engulfment", "V5/B3 porosity model — L1."),
        GateRef("g_magnetic", "magnetic property model (coercivity/loss)",
                "magnetic_properties", "V6 short-list proposal — UNMODELLED."),
        GateRef("g_pm_finish", "PM powder sizing & fines spec",
                "pm_powder_finish", "V6 short-list proposal — UNMODELLED."),
        GateRef("g_hot_short", "tramp-element hot-shortness ceiling",
                "hot_shortness", "V5/E1 ceiling model — L1."),
    )
}


def gate_status(gate_id: str) -> Dict[str, Any]:
    """Live modelling status of one gate, probed from the module tree.

    Returns a dict with ``state`` in
    ``{"unmodelled", "modelled (L1)", "modelled (flag: ...)", "modelled (flag unstated)"}``
    and ``unvalidated_L1``/``exists`` booleans.  This is the ladder's
    rederivation mechanism for gate physics: add the module named in
    ``GATE_REGISTRY`` and every rung that gates on it flips automatically.
    """
    ref = GATE_REGISTRY[gate_id]  # KeyError on unknown id: intentional.
    try:
        module = importlib.import_module(f"models.{ref.module}")
    except ImportError:
        return {
            "gate_id": ref.id,
            "name": ref.name,
            "module": ref.module,
            "exists": False,
            "flag": None,
            "state": "unmodelled",
            "unvalidated_L1": False,
            "note": ref.note,
        }
    flag = getattr(module, "SCREENING_FLAG", None)
    if flag is None:
        state, l1 = "modelled (flag unstated)", False
    elif "L1" in str(flag) or "unvalidated" in str(flag):
        state, l1 = "modelled (L1)", True
    else:
        state, l1 = f"modelled (flag: {flag})", False
    return {
        "gate_id": ref.id,
        "name": ref.name,
        "module": ref.module,
        "exists": True,
        "flag": None if flag is None else str(flag),
        "state": state,
        "unvalidated_L1": l1,
        "note": ref.note,
    }


# ════════════════════════════════════════════════════════════════════════
#  Post-cell unit operations (the only new physics inputs, all anchored)
# ════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PostUnitOp:
    """A post-harvest unit operation: energy (kWh/t product) + cash ($/t)."""

    id: str
    name: str
    energy_kWh_per_t: float
    cash_per_t: float
    anchor_key: str
    notes: str = ""


def _op(op_id: str, name: str, e_key: str, c_key: str, notes: str = "") -> PostUnitOp:
    e = get_anchor(e_key)
    c = get_anchor(c_key)
    return PostUnitOp(op_id, name, e.value, c.value, f"{e_key}+{c_key}", notes)


def _anneal_op() -> PostUnitOp:
    """Anneal energy *derived live* from the thermomechanical model.

    Falls back to the ANNEAL_KWH_T anchor only if the model signature changes
    — the anchor row documents the screening value; the live call is the
    rederivation hook (anneal temperature/efficiency updates propagate).
    """
    cash = get_anchor("ANNEAL_CASH_T")
    try:
        from .thermomechanical import ThermomechanicalModel

        e_kwh_kg = ThermomechanicalModel().anneal_energy_kWh_per_kg()
        e_kwh_t = e_kwh_kg * 1000.0
        provenance = "live: thermomechanical.anneal_energy_kWh_per_kg"
    except Exception:  # pragma: no cover - defensive; tested via fallback band
        e_kwh_t = get_anchor("ANNEAL_KWH_T").value
        provenance = "anchor fallback: ANNEAL_KWH_T"
    return PostUnitOp(
        "op_anneal", "recrystallization anneal (batch/intercover)",
        e_kwh_t, cash.value, "ANNEAL_KWH_T(live)+ANNEAL_CASH_T", provenance,
    )


POST_OPS: Dict[str, PostUnitOp] = {
    op.id: op
    for op in (
        _op("op_rinse_dry", "counter-current rinse + dry + passivation",
            "DRY_PASSIVATE_KWH_T", "RINSE_DRY_CASH_T",
            "V6 §1.2/§1.3 — Landau–Levich film, ~10 % moisture evaporation, "
            "controlled-O₂ passivation; chemistry model pending."),
        _op("op_briquette", "briquetting press (die wear, binder-less)",
            "BRIQUETTE_KWH_T", "BRIQUETTE_CASH_T",
            "V6 §1.4 — Heckel compaction screen; energy is press-only."),
        _op("op_induction_melt", "induction melting 25→1,600 °C",
            "INDUCTION_MELT_KWH_T", "MELT_CASH_T",
            "Option A.5 core op — refractories, slag formers, melt loss cash."),
        _op("op_cast_roll", "continuous cast + hot roll to bar",
            "CAST_ROLL_KWH_T", "CAST_ROLL_CASH_T",
            "billet cast + hot-roll energy and yield loss."),
        _op("op_skinpass", "temper mill / skin-pass",
            "SKINPASS_KWH_T", "SKINPASS_CASH_T",
            "Lüders-band suppression + gauge finish (g_strain_aging lever)."),
        _op("op_carburize", "gas carburize case (Option-B C route)",
            "CARBURIZE_KWH_T", "CARBURIZE_CASH_T",
            "screens carburization.py atmosphere; C in-cell alternative is "
            "carbon_electrodeposition.py (V5/A1)."),
        _op("op_pm_finish", "PM-powder sizing/classification (inert gas)",
            "PM_FINISH_KWH_T", "PM_FINISH_CASH_T",
            "screen, blend, spec fines; N₂ blanket (pyrophoricity, V6 §1.2)."),
        _op("op_battery_finish", "battery-anode finish (porosity/size spec)",
            "BATTERY_FINISH_KWH_T", "BATTERY_FINISH_CASH_T",
            "porous-spec sizing + QA; spec is customer-owned (L1 guess)."),
        _op("op_magnetic_qa", "magnetic lamination QA + insulation coat",
            "MAGNETIC_QA_KWH_T", "MAGNETIC_QA_CASH_T",
            "interlaminar insulation coat + loss certification."),
    )
}
POST_OPS["op_anneal"] = _anneal_op()


# ════════════════════════════════════════════════════════════════════════
#  Product rungs
# ════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PriceBand:
    """A product-price band, derived from one anchor (mid ± uncertainty)."""

    anchor_key: str

    @property
    def mid(self) -> float:
        return get_anchor(self.anchor_key).value

    @property
    def low(self) -> float:
        a = get_anchor(self.anchor_key)
        return a.value - a.uncertainty

    @property
    def high(self) -> float:
        a = get_anchor(self.anchor_key)
        return a.value + a.uncertainty

    def as_dict(self) -> Dict[str, float]:
        a = get_anchor(self.anchor_key)
        return {"low": self.low, "mid": a.value, "high": self.high,
                "ref": a.ref}


@dataclass(frozen=True)
class ProductRung:
    """One step on the product value ladder."""

    id: str
    name: str
    option: str                      # "A" | "A.5" | "B-lite" | "B" | "side"
    product_form: str
    architecture_id: str             # key into cell_architecture.ARCHITECTURES
    price: PriceBand
    unit_ops: Tuple[PostUnitOp, ...]
    gates: Tuple[str, ...]           # keys into GATE_REGISTRY
    qualification: str               # what economics cannot see: spec culture
    description: str

    def gate_rows(self) -> List[Dict[str, Any]]:
        return [gate_status(g) for g in self.gates]

    @property
    def unmodelled_gates(self) -> List[str]:
        return [g["name"] for g in self.gate_rows() if g["state"] == "unmodelled"]

    @property
    def modelled_l1_gates(self) -> List[str]:
        return [g["name"] for g in self.gate_rows() if g["unvalidated_L1"]]

    @property
    def aux_energy_kWh_per_t(self) -> float:
        return sum(op.energy_kWh_per_t for op in self.unit_ops)

    @property
    def aux_cash_per_t(self) -> float:
        return sum(op.cash_per_t for op in self.unit_ops)


def _rungs() -> Dict[str, ProductRung]:
    order: List[ProductRung] = [
        ProductRung(
            id="flake_feed",
            name="Option A — passivated briquetted flake (melt-shop virgin units)",
            option="A",
            product_form="flake/powder → passivated briquette",
            architecture_id="rotating_cylinder",
            price=PriceBand("FLAKE_FEED_PRICE_T"),
            unit_ops=(POST_OPS["op_rinse_dry"], POST_OPS["op_briquette"]),
            gates=(
                "g_fe_gate", "g_energy_gate", "g_arch_gate", "g_loop",
                "g_oc_corrosion", "g_product_ox", "g_rinse", "g_briquet",
                "g_melt_balance", "g_purity",
            ),
            qualification="Buyer lab trials (yield/boil verification); months, "
            "not years. Price risks scrap/DRI substitution.",
            description="The RESEARCH_PROGRAM Option A path: friable powder is "
            "a feature, H/C/deposit-stress problems deleted by the buyer's "
            "furnace. Competes on iron-unit price against scrap/HBI — the "
            "lowest-margin rung.",
        ),
        ProductRung(
            id="own_melt_bar",
            name="Option A.5 — own-melt + cast/roll to rebar/merchant bar",
            option="A.5",
            product_form="flake → induction melt → billet → bar",
            architecture_id="rotating_cylinder",
            price=PriceBand("REBAR_PRICE_T"),
            unit_ops=(
                POST_OPS["op_rinse_dry"], POST_OPS["op_briquette"],
                POST_OPS["op_induction_melt"], POST_OPS["op_cast_roll"],
            ),
            gates=(
                "g_fe_gate", "g_energy_gate", "g_arch_gate", "g_loop",
                "g_oc_corrosion", "g_product_ox", "g_rinse", "g_briquet",
                "g_casting", "g_grade", "g_hot_short", "g_purity",
            ),
            qualification="Rebar is the lowest-certification steel SKU "
            "(ASTM A615); merchant quality achievable in 2–5 yr. Adds a "
            "metallurgical workforce the pure hydromet path avoids.",
            description="The missing middle from the product debate: keep the "
            "simple hydromet cell (powder fine, H fine), own the melt step in "
            "a commodity induction furnace, ship *steel* and capture the "
            "flake→bar margin delta yourself.",
        ),
        ProductRung(
            id="annealed_foil",
            name="Option B-lite — annealed ferritic foil / non-structural sheet",
            option="B-lite",
            product_form="drum-harvested foil, annealed, temper-rolled",
            architecture_id="drum_and_strip",
            price=PriceBand("LOWC_FOIL_PRICE_T"),
            unit_ops=(POST_OPS["op_anneal"], POST_OPS["op_skinpass"]),
            gates=(
                "g_fe_gate", "g_energy_gate", "g_arch_gate", "g_loop",
                "g_peel", "g_stress_h", "g_grade", "g_drum_life",
                "g_strain_aging", "g_deposit_aging", "g_purity",
            ),
            qualification="Non-structural niches (battery substrates, "
            "shielding, brazing foil): customer sampling, 1–2 yr. The drum "
            "coupon is the branch-defining experiment (already specified).",
            description="Near-net foil straight off the drum, annealed to "
            "ferritic low-C sheet. Keeps the six hard problems Option A "
            "deletes — but prices the product at 4–6× commodity iron.",
        ),
        ProductRung(
            id="structural_sheet",
            name="Option B — structural low-C sheet (in-cell C or carburized)",
            option="B",
            product_form="foil + in-cell or carburized C + anneal",
            architecture_id="drum_and_strip",
            price=PriceBand("HRC_STRUCTURAL_PRICE_T"),
            unit_ops=(
                POST_OPS["op_carburize"], POST_OPS["op_anneal"],
                POST_OPS["op_skinpass"],
            ),
            gates=(
                "g_fe_gate", "g_energy_gate", "g_arch_gate", "g_loop",
                "g_peel", "g_stress_h", "g_grade", "g_drum_life",
                "g_strain_aging", "g_deposit_aging", "g_hot_short", "g_purity",
            ),
            qualification="Structural certification (AISI/ASTM structural "
            "grades, welding, Charpy): multi-year spec culture. The "
            "full-academic-Option-B endpoint.",
            description="The original program name. Note the economics screen "
            "prices it at HRC parity — *lower* than B-lite foil — so within "
            "pure economics B ranks below B-lite until volume/qualification "
            "learnings dominate. Kept as the identity rung.",
        ),
        ProductRung(
            id="pm_powder",
            name="Side — electrolytic PM iron powder",
            option="side",
            product_form="powder → passivate/size/classify",
            architecture_id="rotating_cylinder",
            price=PriceBand("PM_POWDER_PRICE_T"),
            unit_ops=(POST_OPS["op_rinse_dry"], POST_OPS["op_pm_finish"]),
            gates=(
                "g_fe_gate", "g_energy_gate", "g_arch_gate", "g_loop",
                "g_oc_corrosion", "g_product_ox", "g_pm_finish", "g_purity",
            ),
            qualification="PM-spec vendor qualification + small market "
            "(kt-scale, historically electrolytic-iron-powered).",
            description="The one niche where aqueous iron EW has survived "
            "commercially (99.9 % purity electrolytic powder). Small market, "
            "proven route, real premiums over atomized powder.",
        ),
        ProductRung(
            id="battery_iron",
            name="Side — iron-air battery anode feed",
            option="side",
            product_form="powder/foam → porosity-spec finish",
            architecture_id="rotating_cylinder",
            unit_ops=(POST_OPS["op_rinse_dry"], POST_OPS["op_battery_finish"]),
            gates=(
                "g_fe_gate", "g_energy_gate", "g_arch_gate", "g_loop",
                "g_oc_corrosion", "g_product_ox", "g_form_factor", "g_purity",
            ),
            price=PriceBand("BATTERY_IRON_PRICE_T"),
            qualification="Customer-owned spec (iron-air developers); "
            "price anchor is the most speculative on the ladder — treat as "
            "parity target, not market quote.",
            description="Iron-air storage is an electrochemistry-adjacent "
            "buyer for clean, porous iron; demand could be large if storage "
            "deploys at grid scale. High option value, thin evidence.",
        ),
        ProductRung(
            id="magnetic_foil",
            name="Side — soft-magnetic laminate foil",
            option="side",
            product_form="drum foil → anneal → insulation coat → laminate",
            architecture_id="drum_and_strip",
            price=PriceBand("MAGNETIC_FOIL_PRICE_T"),
            unit_ops=(POST_OPS["op_anneal"], POST_OPS["op_magnetic_qa"]),
            gates=(
                "g_fe_gate", "g_energy_gate", "g_arch_gate", "g_loop",
                "g_peel", "g_stress_h", "g_drum_life", "g_magnetic",
                "g_deposit_aging", "g_purity",
            ),
            qualification="Core-loss certification (Epstein/SST); buyer = "
            "motor/transformer niche, not commodity steel.",
            description="The drum's 25–50 µm form factor is *already* the "
            "eddy-current-optimal lamination thickness. Needs a magnetic "
            "property model (coercivity vs grain/inclusions — unmodelled).",
        ),
    ]
    return {r.id: r for r in order}


RUNGS: Dict[str, ProductRung] = _rungs()


# ════════════════════════════════════════════════════════════════════════
#  Evaluation
# ════════════════════════════════════════════════════════════════════════

@dataclass
class RungResult:
    """Screening economics for one rung at one price point (mid band)."""

    rung_id: str
    name: str
    option: str
    architecture_id: str
    architecture_name: str

    # Live-derived cell physics
    areal_productivity_t_m2_yr: float
    installed_cost_per_m2: float
    capital_charge_per_t: float
    dc_energy_kWh_per_t: float
    limited_by: str

    # Economics at mid price
    price_mid_per_t: float
    aux_energy_kWh_per_t: float
    energy_cost_per_t: float
    cash_other_per_t: float
    total_cost_per_t: float
    margin_per_t: float
    margin_share_of_price: float
    margin_per_m2_yr: float
    breakeven_product_price_per_t: float
    breakeven_electricity_price_kWh: float

    # Kill criterion #3 at rung price (the anti-5× numbers)
    capital_share_of_price: float          # parameter-free: charge / price
    min_price_for_budget_per_t: float      # price ≥ this at declared fraction
    capital_budget_per_t: float
    required_productivity_t_m2_yr: float
    required_zinc_multiple: float
    clears_kc3_at_price: bool

    # Price-band margins
    margin_at_low: float
    margin_at_high: float

    # Gates
    gate_rows: List[Dict[str, Any]] = field(default_factory=list)
    n_unmodelled_gates: int = 0
    n_l1_gates: int = 0
    n_flag_unstated_gates: int = 0

    verdict: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        return d


def evaluate_rung(
    rung: ProductRung,
    conditions: Optional[cell_architecture.OperatingConditions] = None,
    electricity_price_kWh: Optional[float] = None,
    price_override: Optional[float] = None,
    capital_budget_fraction: float = CAPITAL_BUDGET_FRACTION,
) -> RungResult:
    """Contribution-margin screen for one product rung.

    Parameters
    ----------
    rung
        The rung to evaluate.
    conditions
        Shared electrolyte/economic context — defaults to
        ``cell_architecture.OperatingConditions()`` so the ladder tracks the
        architecture screen's defaults as they are calibrated.
    electricity_price_kWh
        Defaults to the *live* ``technoeconomic.OPEXModel`` electricity price
        (another rederivation hook).
    price_override
        Screening convenience: evaluate the rung at an arbitrary product
        price (used by the price-artifact demo and tests).
    capital_budget_fraction
        Share of product price allocatable to installed-cell capital charge.
    """
    cond = conditions or cell_architecture.OperatingConditions()
    if electricity_price_kWh is None:
        electricity_price_kWh = technoeconomic.OPEXModel().electricity_price_kWh
    price = rung.price.mid if price_override is None else price_override

    # ── live cell physics ────────────────────────────────────────────
    arch = cell_architecture.evaluate_architecture(
        cell_architecture.ARCHITECTURES[rung.architecture_id], cond
    )
    dc_kwh_t = electrochemistry.specific_energy_kWh_per_t(
        cond.cell_voltage_V, cond.faradaic_efficiency
    )

    # ── cost stack (contribution basis) ─────────────────────────────
    aux_kwh_t = rung.aux_energy_kWh_per_t
    energy_cost = (dc_kwh_t + aux_kwh_t) * electricity_price_kWh
    cash_other = rung.aux_cash_per_t
    capital_charge = arch.capital_charge_per_t_fe   # installed × CRF / P
    total_cost = energy_cost + cash_other + capital_charge

    margin_t = price - total_cost
    margin_share = margin_t / price if price > 0 else float("-inf")
    margin_m2 = margin_t * arch.areal_productivity_t_m2_yr
    breakeven_price = total_cost
    denom = dc_kwh_t + aux_kwh_t
    breakeven_elec = (
        (price - cash_other - capital_charge) / denom if denom > 0 else 0.0
    )

    # ── kill criterion #3 re-derived at rung price ──────────────────
    budget_per_t = capital_budget_fraction * price
    capital_share = capital_charge / price if price > 0 else float("inf")
    min_price_for_budget = (
        capital_charge / capital_budget_fraction
        if capital_budget_fraction > 0
        else float("inf")
    )
    required_productivity = (
        arch.installed_cost_per_m2 * cond.crf / budget_per_t
        if budget_per_t > 0
        else float("inf")
    )
    zinc = cell_architecture.zinc_tankhouse_productivity(
        faradaic_efficiency=cond.faradaic_efficiency,
        hours_per_year=cond.hours_per_year,
    )
    required_mult = required_productivity / zinc if zinc > 0 else float("inf")
    clears_kc3 = arch.areal_productivity_t_m2_yr >= required_productivity

    # price-band margins
    margin_low = rung.price.low - total_cost
    margin_high = rung.price.high - total_cost

    # gates (live statuses)
    rows = rung.gate_rows()
    n_unmodelled = sum(1 for g in rows if g["state"] == "unmodelled")
    n_l1 = sum(1 for g in rows if g["unvalidated_L1"])
    n_unstated = sum(1 for g in rows if g["state"] == "modelled (flag unstated)")

    if margin_t <= 0:
        verdict = "stalls"
    elif margin_share < MARGIN_VERDICT_FRACTION:
        verdict = "marginal"
    else:
        verdict = "clears"
    if not clears_kc3:
        verdict = "stalls-capital" if verdict == "stalls" else f"{verdict}-but-kc3-fails"

    return RungResult(
        rung_id=rung.id,
        name=rung.name,
        option=rung.option,
        architecture_id=rung.architecture_id,
        architecture_name=arch.name,
        areal_productivity_t_m2_yr=arch.areal_productivity_t_m2_yr,
        installed_cost_per_m2=arch.installed_cost_per_m2,
        capital_charge_per_t=capital_charge,
        dc_energy_kWh_per_t=dc_kwh_t,
        limited_by=arch.limited_by,
        price_mid_per_t=price,
        aux_energy_kWh_per_t=aux_kwh_t,
        energy_cost_per_t=energy_cost,
        cash_other_per_t=cash_other,
        total_cost_per_t=total_cost,
        margin_per_t=margin_t,
        margin_share_of_price=margin_share,
        margin_per_m2_yr=margin_m2,
        breakeven_product_price_per_t=breakeven_price,
        breakeven_electricity_price_kWh=breakeven_elec,
        capital_share_of_price=capital_share,
        min_price_for_budget_per_t=min_price_for_budget,
        capital_budget_per_t=budget_per_t,
        required_productivity_t_m2_yr=required_productivity,
        required_zinc_multiple=required_mult,
        clears_kc3_at_price=bool(clears_kc3),
        margin_at_low=margin_low,
        margin_at_high=margin_high,
        gate_rows=rows,
        n_unmodelled_gates=n_unmodelled,
        n_l1_gates=n_l1,
        n_flag_unstated_gates=n_unstated,
        verdict=verdict,
    )


@dataclass
class LadderResult:
    """Full ladder evaluation + context + price-artifact demo."""

    rungs: List[RungResult]
    electricity_price_kWh: float
    cell_voltage_V: float
    faradaic_efficiency: float
    zinc_tankhouse_productivity_t_m2_yr: float
    capital_budget_fraction: float
    price_artifact_demo: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rungs": [r.to_dict() for r in self.rungs],
            "context": {
                "electricity_price_kWh": self.electricity_price_kWh,
                "cell_voltage_V": self.cell_voltage_V,
                "faradaic_efficiency": self.faradaic_efficiency,
                "zinc_tankhouse_productivity_t_m2_yr":
                    self.zinc_tankhouse_productivity_t_m2_yr,
                "capital_budget_fraction": self.capital_budget_fraction,
            },
            "price_artifact_demo": self.price_artifact_demo,
        }


def price_artifact_demo(
    conditions: Optional[cell_architecture.OperatingConditions] = None,
    capital_budget_fraction: float = CAPITAL_BUDGET_FRACTION,
) -> List[Dict[str, Any]]:
    """The headline table: required zinc-multiple per architecture × price.

    Shows the "~5×" imperative is a price artefact: at commodity prices the
    drum route needs multiples above what it can deliver (so the program
    concluded 'must beat Zn tankhouse 5×'); at foil/magnetics prices the
    *same* drum needs <1× of itself.  All cells recomputed live from the
    architecture module.
    """
    cond = conditions or cell_architecture.OperatingConditions()
    zinc = cell_architecture.zinc_tankhouse_productivity(
        faradaic_efficiency=cond.faradaic_efficiency,
        hours_per_year=cond.hours_per_year,
    )
    price_points = sorted({r.price.mid for r in RUNGS.values()} | {450.0, 2000.0})
    arcs = ("rotating_cylinder", "drum_and_strip", "plate_and_frame")
    rows: List[Dict[str, Any]] = []
    for arc_id in arcs:
        spec = cell_architecture.ARCHITECTURES[arc_id]
        res = cell_architecture.evaluate_architecture(spec, cond)
        for price in price_points:
            req = (
                res.installed_cost_per_m2 * cond.crf
                / (capital_budget_fraction * price)
            )
            rows.append({
                "architecture": arc_id,
                "name": res.name,
                "price_per_t": price,
                "installed_cost_per_m2": round(res.installed_cost_per_m2, 0),
                "areal_productivity_t_m2_yr":
                    round(res.areal_productivity_t_m2_yr, 1),
                "capital_charge_per_t": round(res.capital_charge_per_t_fe, 2),
                "capital_share_of_price":
                    round(res.capital_charge_per_t_fe / price, 4),
                "required_productivity_t_m2_yr": round(req, 1),
                "required_zinc_multiple": round(req / zinc, 2),
                "delivers_zinc_multiple":
                    round(res.areal_productivity_t_m2_yr / zinc, 2),
                "clears": bool(res.areal_productivity_t_m2_yr >= req),
            })
    return rows


def evaluate_ladder(
    conditions: Optional[cell_architecture.OperatingConditions] = None,
    electricity_price_kWh: Optional[float] = None,
    capital_budget_fraction: float = CAPITAL_BUDGET_FRACTION,
    rung_ids: Optional[Sequence[str]] = None,
) -> LadderResult:
    """Evaluate all rungs (or a subset) against the live model state."""
    cond = conditions or cell_architecture.OperatingConditions()
    elec = (
        technoeconomic.OPEXModel().electricity_price_kWh
        if electricity_price_kWh is None
        else electricity_price_kWh
    )
    ids = rung_ids if rung_ids is not None else list(RUNGS.keys())
    results = [
        evaluate_rung(RUNGS[i], cond, elec,
                      capital_budget_fraction=capital_budget_fraction)
        for i in ids
    ]
    zinc = cell_architecture.zinc_tankhouse_productivity(
        faradaic_efficiency=cond.faradaic_efficiency,
        hours_per_year=cond.hours_per_year,
    )
    return LadderResult(
        rungs=results,
        electricity_price_kWh=elec,
        cell_voltage_V=cond.cell_voltage_V,
        faradaic_efficiency=cond.faradaic_efficiency,
        zinc_tankhouse_productivity_t_m2_yr=zinc,
        capital_budget_fraction=capital_budget_fraction,
        price_artifact_demo=price_artifact_demo(cond, capital_budget_fraction),
    )


# ════════════════════════════════════════════════════════════════════════
#  Provenance + artifacts (REPO_OUTPUT_POLICY compliant stamps)
# ════════════════════════════════════════════════════════════════════════

def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def provenance(root: Optional[Path] = None) -> Dict[str, Any]:
    """Publication-artifact provenance stamp (docs/REPO_OUTPUT_POLICY.md)."""
    r = root or ROOT
    return {
        "artifact": "product_ladder",
        "recipe": "python -m models.run_product_ladder  (full-grade; "
                  "screening constants are L1 by declaration, not by mode)",
        "mode": "full-grade",
        "source_hashes": {
            name: _sha16(r / name) for name in _STAMPED_SOURCES
        },
    }


def comparison_table(result: LadderResult) -> str:
    cols = (
        "rung", "option", "arch", "$/t mid", "cost $/t", "margin $/t",
        "margin $/(m²·yr)", "kc3 req ×Zn", "kc3 clears", "gates U/L1", "verdict",
    )
    lines = ["  ".join(cols), "-" * 118]
    for r in result.rungs:
        lines.append(
            f"{r.rung_id:<16}  {r.option:<6}  {r.architecture_id:<19}"
            f"{r.price_mid_per_t:>8.0f}  {r.total_cost_per_t:>8.0f}"
            f"{r.margin_per_t:>10.0f}  {r.margin_per_m2_yr:>15,.0f}"
            f"{r.required_zinc_multiple:>10.2f}"
            f"{str(r.clears_kc3_at_price):>11}"
            f"{r.n_unmodelled_gates:>6}/{r.n_l1_gates:<3}  {r.verdict}"
        )
    return "\n".join(lines)


def render_markdown(result: LadderResult, root: Optional[Path] = None) -> str:
    """Generate docs/PRODUCT_VALUE_LADDER.md from the live result."""
    prov = provenance(root)
    ctx = result.to_dict()["context"]
    L: List[str] = []
    a = L.append
    a("# Product Value Ladder — Feedstock vs. Steel, Computed Live")
    a("")
    a("> **Generated artifact — do not hand-edit.** Rebuild with "
      "`python -m models.run_product_ladder` (or `aq-steel-product-ladder`) "
      "after any change to the models it derives from (cell_architecture, "
      "electrochemistry, technoeconomic, thermomechanical, anchors, this "
      "module). Numbers re-derive on every run; the comparative structure is "
      "what is decision-grade, not the decimals.")
    a("")
    a(f"<!-- provenance:\n```json\n{json.dumps(prov, indent=2)}\n```\n-->")
    a("")
    a("**Screening flag:** unvalidated (L1). Product prices are anchored "
      "screening bands (see Appendix B); everything physical re-derives from "
      "the model suite at the moment of generation.")
    a("")
    a("---")
    a("")
    a("## 1. Why this document exists")
    a("")
    a("`docs/RESEARCH_PROGRAM.md` poses the page-1 decision — Option A "
      "(melt-shop feedstock) vs. Option B (direct steel) — as text. This "
      "ladder makes the same decision a **recomputed number**, and adds the "
      "rung the split missed: **Option A.5 (own-melt + cast to bar)**. Any "
      "model change that moves productivity, capital charge, DC energy, "
      "anneal energy, or the default electricity price moves every verdict "
      "here automatically — that is the entire design.")
    a("")
    a("Contribution-margin basis (uniform across rungs; BOP/labour/feedstock "
      "live in `technoeconomic.py`, which is the full-plant model):")
    a("")
    a("```\nmargin $/t  =  price_mid − ( (DC kWh/t + aux kWh/t)×$/kWh "
      "+ aux cash $/t + installed-cell capital charge $/t )\n"
      "margin $/(m²·yr)  =  margin $/t × areal productivity t/(m²·yr)\n"
      "required ×Zn     =  installed $/m² × CRF / (budget_frac × price) / "
      "zinc-benchmark productivity\n```")
    a("")
    a(f"Context at generation: V_cell = {ctx['cell_voltage_V']} V, "
      f"FE = {ctx['faradaic_efficiency']}, "
      f"electricity = ${ctx['electricity_price_kWh']}/kWh, "
      f"zinc benchmark = {ctx['zinc_tankhouse_productivity_t_m2_yr']:.2f} "
      f"t/(m²·yr) at this FE, capital budget fraction = "
      f"{ctx['capital_budget_fraction']} of product price.")
    a("")
    a("## 2. The ladder")
    a("")
    a("| Rung | Option | Architecture | Price band $/t | Product |")
    a("|---|---|---|---|---|")
    for r in result.rungs:
        ru = RUNGS[r.rung_id]
        a(f"| `{r.rung_id}` | {r.option} | {r.architecture_id} | "
          f"{ru.price.low:,.0f}–{ru.price.high:,.0f} "
          f"(mid {r.price_mid_per_t:,.0f}) | {ru.product_form} |")
    a("")
    a("## 3. Screening economics at mid price")
    a("")
    a("| Rung | Productivity t/(m²·yr) | DC kWh/t | Aux kWh/t | Capital $/t | "
      "Capital share | Cost $/t | Margin $/t | Margin $/(m²·yr) | Verdict |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in result.rungs:
        a(f"| {r.rung_id} | {r.areal_productivity_t_m2_yr:.1f} | "
          f"{r.dc_energy_kWh_per_t:,.0f} | {r.aux_energy_kWh_per_t:,.0f} | "
          f"{r.capital_charge_per_t:,.1f} | "
          f"{100 * r.capital_share_of_price:.1f}% | "
          f"{r.total_cost_per_t:,.0f} | "
          f"{r.margin_per_t:,.0f} | {r.margin_per_m2_yr:,.0f} | {r.verdict} |")
    a("")
    a("`min_price_for_budget` — the product price at which the cell's capital "
      f"charge is exactly {ctx['capital_budget_fraction']:.0%} of price — "
      "per rung:")
    a("")
    a("| Rung | Capital $/t | Min price for budget $/t | Price band clears? |")
    a("|---|---:|---:|:---:|")
    for r in result.rungs:
        ru = RUNGS[r.rung_id]
        clears_band = r.min_price_for_budget_per_t <= ru.price.high
        a(f"| {r.rung_id} | {r.capital_charge_per_t:,.1f} | "
          f"{r.min_price_for_budget_per_t:,.0f} | "
          f"{'✓' if clears_band else '✗ (price too low)'} |")
    a("")
    a("Price-band robustness (margin $/t at band edges). **These contribution "
      "margins are upper bounds** — at screening electricity prices and with "
      "non-cell costs held uniform, most things clear; the discriminators "
      "between rungs are the *size* of the margin per m²·yr (which buys down "
      "plant-wide risks) and the gate load (§5), not the sign:")
    a("")
    a("| Rung | Margin @ low price | Margin @ mid | Margin @ high |")
    a("|---|---:|---:|---:|")
    for r in result.rungs:
        a(f"| {r.rung_id} | {r.margin_at_low:,.0f} | {r.margin_per_t:,.0f} | "
          f"{r.margin_at_high:,.0f} |")
    a("")
    # ── the README 5×, recovered without any budget constant ───────────
    rcs = next(r for r in result.rungs if r.architecture_id == "rotating_cylinder")
    drums = [r for r in result.rungs if r.architecture_id == "drum_and_strip"]
    drum_share_at_flake = drums[0].capital_charge_per_t / RUNGS["flake_feed"].price.mid
    rc_share_at_flake = rcs.capital_charge_per_t / RUNGS["flake_feed"].price.mid
    ratio_5x = drum_share_at_flake / rc_share_at_flake

    a("## 4. The '5× imperative' is flake-economics, not physics")
    a("")
    a("Two parameter-free facts the suite now recomputes:")
    a("")
    a(f"1. **The README's '~5×' is a cross-architecture capital-share ratio at "
      f"commodity price.** At ${RUNGS['flake_feed'].price.mid:,.0f}/t, the drum's "
      f"installed-cell capital charge is ${drums[0].capital_charge_per_t:.2f}/t "
      f"({drum_share_at_flake:.1%} of price) vs. the rotating cylinder's "
      f"${rcs.capital_charge_per_t:.2f}/t ({rc_share_at_flake:.1%}) — a "
      f"{ratio_5x:.1f}× ratio. Both technically clear a "
      f"{ctx['capital_budget_fraction']:.0%} capital-share budget even at flake "
      f"price; the README's imperative corresponds to demanding ~4–6% capital "
      f"share, which is what commodity-iron contribution margins can actually "
      f"carry after non-cell costs. So the drum's real blocker at commodity "
      f"price is not its $/m² — it is that thin margins leave no room for it.")
    a("2. **The same drum's capital share collapses going up the ladder.**")
    a("")
    a("| Rung (drum architecture) | Price $/t | Capital $/t | Capital share | "
      "Min price to fit budget |")
    a("|---|---:|---:|---:|---:|")
    for r in drums:
        a(f"| {r.rung_id} | {r.price_mid_per_t:,.0f} | "
          f"{r.capital_charge_per_t:.2f} | {100 * r.capital_share_of_price:.1f}% | "
          f"{r.min_price_for_budget_per_t:,.0f} |")
    a("")
    a("Productivity and product price are the *same lever* in kill criterion "
      "#3: required zinc-benchmark multiples across architectures and prices "
      f"(budget = {ctx['capital_budget_fraction']:.0%} of price):")
    a("")
    a("| Architecture | Price $/t | Required t/(m²·yr) | Required ×Zn | "
      "Delivers ×Zn | Clears |")
    a("|---|---:|---:|---:|---:|:---:|")
    for row in result.price_artifact_demo:
        a(f"| {row['architecture']} | {row['price_per_t']:,.0f} | "
          f"{row['required_productivity_t_m2_yr']} | "
          f"{row['required_zinc_multiple']} | {row['delivers_zinc_multiple']} | "
          f"{'✓' if row['clears'] else '✗'} |")
    a("")
    a("**Read:** with a conventional capital budget, hardware affordability "
      "is *not* the discriminator between rungs — the drum is affordable "
      "everywhere. What changes across the ladder is (a) margin per m²·yr "
      "(§3), which buys down the electricity-price and productivity risks of "
      "the whole plant, and (b) the **gate load** (§5): the higher rungs "
      "hold the physics the program has spent its modelling effort on "
      "(peel, stress, grade) or has yet to build (drum life, melt balance). "
      "The drum's real cost at the high rungs is *science risk*, not capital.")
    a("")
    a("## 5. Gate status matrix (resolve live from the model tree)")
    a("")
    a("Legend: ✓ modelled, validated beyond L1 · ◐ modelled, unvalidated L1 · "
      "◑ modelled, screening flag unstated · "
      "✗ **unmodelled** (proposed in `docs/CHEM_PHYS_IMPROVEMENTS_V6.md` or "
      "here) · — n/a. The ✗ cells are the *science agenda implied by each "
      "product rung*; when a named module lands in `models/`, its ✗ flips to "
      "◐ on the next regeneration. Note: ◑ cells mark mature screening "
      "modules that pre-date the SCREENING_FLAG convention — 'modelled', "
      "not 'validated'.")
    a("")
    gate_ids = list(GATE_REGISTRY.keys())
    header = "| Gate |" + "".join(f" {r.rung_id} |" for r in result.rungs)
    a(header)
    a("|" + "---|" * (len(result.rungs) + 1))
    # Pre-fetch statuses per rung
    status_map = {
        r.rung_id: {g["gate_id"]: g for g in r.gate_rows} for r in result.rungs
    }
    for gid in gate_ids:
        ref = GATE_REGISTRY[gid]
        cells = []
        for r in result.rungs:
            g = status_map[r.rung_id].get(gid)
            if g is None:
                cells.append(" — ")
            elif g["state"] == "unmodelled":
                cells.append(" **✗** ")
            elif g["unvalidated_L1"]:
                cells.append(" ◐ ")
            elif g["state"] == "modelled (flag unstated)":
                cells.append(" ◑ ")
            else:
                cells.append(" ✓ ")
        a(f"| {ref.name} |" + "|".join(cells) + "|")
    a("")
    a("Gate notes: " + "; ".join(
        f"**{ref.id}** ({ref.module}): {ref.note}" for g in gate_ids
        for ref in (GATE_REGISTRY[g],)
    ))
    a("")
    a("## 6. What the uniform economics hide (read before acting)")
    a("")
    for r in result.rungs:
        ru = RUNGS[r.rung_id]
        a(f"- **{r.rung_id}** — {ru.qualification}")
    a("")
    a("## 7. Method, limitations, and how to rederive")
    a("")
    a("- Contribution margin, not plant TEA: feedstock, labour, BOP, "
      "maintenance, working capital and market risk are held constant across "
      "rungs via `technoeconomic.py` conventions; `technoeconomic.py` remains "
      "the decision-grade plant model for a chosen rung.")
    a("- Live derivations on every run: architecture productivity & capital "
      "(`cell_architecture.evaluate_architecture`), DC energy "
      "(`electrochemistry.specific_energy_kWh_per_t`), zinc benchmark "
      "(`cell_architecture.zinc_tankhouse_productivity`), anneal energy "
      "(`thermomechanical.anneal_energy_kWh_per_kg` — anchor fallback only "
      "on API change), default electricity price "
      "(`technoeconomic.OPEXModel`), gate states (module tree probe).")
    a("- New constants on this rung only: product price bands and post-cell "
      "unit-operation energies/cash — all in `models/anchors.py` with refs."
      )
    a("- To rederive: `aq-steel-product-ladder` rewrites this file and "
      "`experiments/data/product_ladder_report.json` with a fresh provenance "
      "stamp. CI hash-checks the stamp (docs/REPO_OUTPUT_POLICY.md).")
    a("")
    a("## Appendix A — rung descriptions")
    a("")
    for r in result.rungs:
        ru = RUNGS[r.rung_id]
        a(f"### {ru.name}")
        a("")
        a(ru.description)
        a("")
        ops = ", ".join(f"{op.name} ({op.energy_kWh_per_t:.0f} kWh/t + "
                        f"${op.cash_per_t:.0f}/t)" for op in ru.unit_ops)
        a(f"*Post-cell ops:* {ops or 'none'}.")
        a("")
    a("## Appendix B — price & unit-op anchors")
    a("")
    a("| Anchor | Value | ± | Ref | Notes |")
    a("|---|---:|---:|---|---|")
    keys = []
    for r in RUNGS.values():
        keys.append(r.price.anchor_key)
    for op in POST_OPS.values():
        keys.extend(op.anchor_key.replace("(live)", "").split("+"))
    seen = set()
    for k in keys:
        if k in seen:
            continue
        seen.add(k)
        try:
            anch = get_anchor(k)
        except KeyError:
            continue
        a(f"| {k} | {anch.value:g} | {anch.uncertainty:g} | {anch.ref} | "
          f"{anch.notes} |")
    a("")
    a("---")
    a("")
    a("*Generated by models/product_ladder.py — full-grade "
      "(screening L1 constants)*.")
    return "\n".join(L) + "\n"


def write_artifacts(
    out_json: Optional[Path] = None,
    out_doc: Optional[Path] = None,
    conditions: Optional[cell_architecture.OperatingConditions] = None,
    electricity_price_kWh: Optional[float] = None,
) -> Dict[str, str]:
    """Evaluate and write the JSON report + regenerated markdown doc."""
    result = evaluate_ladder(conditions, electricity_price_kWh)
    out_json = Path(out_json) if out_json else DEFAULT_JSON_PATH
    out_doc = Path(out_doc) if out_doc else DEFAULT_DOC_PATH
    payload = result.to_dict()
    payload["screening_flag"] = SCREENING_FLAG
    payload["_provenance"] = provenance()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    out_doc.parent.mkdir(parents=True, exist_ok=True)
    out_doc.write_text(render_markdown(result))
    return {"json": str(out_json), "doc": str(out_doc)}


def model_scope() -> Dict[str, Any]:
    """Scope declaration, mirroring the convention in cell_architecture."""
    return {
        "screening_flag": SCREENING_FLAG,
        "live_derivations": [
            "cell_architecture (productivity, installed cost, capital charge, "
            "zinc benchmark)",
            "electrochemistry.specific_energy_kWh_per_t (DC energy)",
            "technoeconomic.OPEXModel.electricity_price_kWh (default price)",
            "thermomechanical.anneal_energy_kWh_per_kg (anneal op energy)",
            "models module tree (gate statuses via SCREENING_FLAG probes)",
        ],
        "new_constants_owned_here": [
            "product price bands (anchors: *_PRICE_T)",
            "post-cell unit-operation energy/cash (anchors: *_KWH_T, *_CASH_T)",
            "verdict thresholds (CAPITAL_BUDGET_FRACTION, MARGIN_VERDICT_FRACTION)",
        ],
        "explicitly_out_of_scope": [
            "plant-level TEA (technoeconomic.py owns it)",
            "feedstock/labour/BOP (uniform across rungs by design)",
            "customer qualification cash/time (text field per rung)",
        ],
    }


def main() -> None:  # pragma: no cover - exercised via run_product_ladder
    result = evaluate_ladder()
    print(comparison_table(result))
    print(
        "\nGate coverage: see docs/PRODUCT_VALUE_LADDER.md §5 "
        "(unmodelled gates are the implied science agenda)."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
