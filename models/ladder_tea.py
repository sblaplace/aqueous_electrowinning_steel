"""Ladder × TEA — full plant costing for every product rung.

Why this module exists
----------------------
``product_ladder.py`` screens the product rungs on a **contribution-margin**
basis::

    margin = price − (DC electricity + post-cell unit-op energy/cash
                      + installed cell capital charge)

and states explicitly that feedstock, electrolyte make-up, anode wear,
labour, maintenance, insurance, overhead and the ore-side plant are *held
constant in the TEA* — correct for a screen, provided the excluded lines
are truly rung-independent.  This module tests that assumption: every rung
is run through the **full plant cost stack** at a common nameplate
capacity, sized from the same live architecture evaluation, and the module
reports

* the full-TEA cost and margin per tonne of shipped product,
* the **screening gap** (ladder contribution margin − full-TEA margin)
  itemised by the cost lines the ladder excludes, and
* whether the ladder's rung *ranking* survives complete costing
  (pairwise order flips).

If the ranking is preserved, the ladder's contribution screen is
decision-grade for choosing *which product to make*; only the absolute
dollars change.  If a flip appears, the screen was hiding a scale- or
form-dependent cost, and the flip names it.

Everything re-derives at call time (the user's standing requirement):
architecture productivity and installed cost from
``cell_architecture.evaluate_architecture``, DC energy from
``electrochemistry.specific_energy_kWh_per_t``, cost-line defaults from
``technoeconomic.CAPEXModel`` / ``OPEXModel`` class defaults, CRF and
annuity from ``cell_architecture.OperatingConditions``, product prices and
post-cell unit ops from ``product_ladder`` (itself live on
``thermomechanical`` and ``anchors``).  **No new physical or cost
constants are introduced** — the only module-owned number is the scenario
knob ``DEFAULT_PLANT_CAPACITY_T_YR`` (a CLI-overridable default, not a
literature constant).

Capital accounting (no double counting)
---------------------------------------
The cell block uses the architecture's installed $/m².  Its
``installed_cost_factor`` is declared on ``OperatingConditions`` as the
round-trip equivalent of ``CAPEXModel``'s assembly × (infrastructure +
engineering) × contingency stack *including* rectifier/electrolyte BOP, so
the ``CAPEXModel`` BOP lines are **not** re-added.  The only plant-side
capital added is the ore-side block (leaching $/tpy + grinding $/tpy ×
nameplate), which the installed factor does not claim to cover.

Dark-mill wiring
----------------
``evaluate_rung_tea(..., site=...)`` accepts a
``dark_mill.SiteDefinition`` (duck-typed): the site grid's effective
electricity price and the site's labour line replace the
``OPEXModel``/module defaults, so the same rung set can be asked
"which product should a mill *at this site* make?" without re-running the
(slower) cell-physics sizing.  Economics of the *physics-solved* operating
point at that site remain ``dark_mill.size_dark_mill``'s job.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import cell_architecture, electrochemistry, technoeconomic
from .product_ladder import (
    MARGIN_VERDICT_FRACTION,
    RUNGS,
    ProductRung,
    evaluate_rung,
)

SCREENING_FLAG = "unvalidated (L1)"

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON_PATH = ROOT / "experiments" / "data" / "ladder_tea_report.json"
DEFAULT_DOC_PATH = ROOT / "docs" / "LADDER_TEA.md"
DEFAULT_FIGURE_PATH = ROOT / "docs" / "figures" / "ladder_tea_margin.png"

# Provenance set: files whose current state defines this report's numbers.
_STAMPED_SOURCES = (
    "models/ladder_tea.py",
    "models/product_ladder.py",
    "models/cell_architecture.py",
    "models/electrochemistry.py",
    "models/technoeconomic.py",
    "models/thermomechanical.py",
    "models/anchors.py",
)

# Scenario knob — nameplate capacity for the cross-rung comparison.
# NOT an anchored constant: per-tonne variable costs are capacity-free and
# the only capacity-sensitive lines are the fixed labour line and the
# ore-side plant capital, both disclosed per tonne in the results.  100 kt/yr
# is a mid-size tankhouse scale (zinc houses run ~100–500 kt/yr); the CLI
# exposes --capacity and the tests check labour dilution across decades.
DEFAULT_PLANT_CAPACITY_T_YR = 100_000.0


# ════════════════════════════════════════════════════════════════════════
#  Result records
# ════════════════════════════════════════════════════════════════════════

@dataclass
class RungTEAResult:
    """Full plant economics for one product rung at a common nameplate."""

    rung_id: str
    name: str
    option: str
    architecture_id: str

    # Scenario context
    capacity_t_yr: float
    electricity_price_kWh: float
    labor_cost_per_yr: float

    # Cell block (live from cell_architecture)
    areal_productivity_t_m2_yr: float
    electrode_area_m2: float
    installed_cost_per_m2: float
    cell_capex_usd: float

    # Ore-side plant (live from technoeconomic.CAPEXModel defaults)
    leaching_capex_usd: float
    grinding_capex_usd: float
    total_capex_usd: float

    # Capital charge ($/t product), CRF basis
    capital_charge_per_t: float          # cell + ore-side plant
    cell_capital_charge_per_t: float     # cell only — equals the ladder's line

    # Energy build-up (kWh/t product)
    dc_energy_kWh_per_t: float
    postop_energy_kWh_per_t: float
    grinding_energy_kWh_per_t: float

    # OPEX itemisation ($/t product)
    electricity_usd_per_t: float
    ore_usd_per_t: float
    electrolyte_usd_per_t: float
    water_usd_per_t: float
    anode_usd_per_t: float
    postop_cash_usd_per_t: float
    overhead_usd_per_t: float
    maintenance_usd_per_t: float
    insurance_usd_per_t: float
    labor_usd_per_t: float
    variable_opex_per_t: float
    total_opex_per_t: float

    # Bottom line
    full_cost_per_t: float               # CRF capital + OPEX
    price_low_per_t: float
    price_mid_per_t: float
    price_high_per_t: float
    margin_per_t: float                  # at mid price
    margin_at_low: float
    margin_at_high: float
    margin_share_of_price: float
    annual_net_usd: float                # margin_per_t × capacity
    npv_usd: Optional[float]             # −CAPEX + net × annuity (r, n)
    payback_yr: Optional[float]
    irr: Optional[float]                 # bisection; None if net ≤ 0

    # The punchline: ladder screen vs full TEA
    ladder_margin_per_t: float           # live product_ladder.evaluate_rung
    ladder_verdict: str
    screening_gap_per_t: float           # ladder − TEA (≥ 0 by construction)
    excluded_lines_per_t: Dict[str, float] = field(default_factory=dict)

    verdict: str = ""                    # tea verdict (clears/marginal/stalls)

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["npv_Musd"] = None if self.npv_usd is None else round(self.npv_usd / 1e6, 2)
        return d


@dataclass
class LadderTEAResult:
    """Full-TEA evaluation of the whole ladder + rank-stability metrics."""

    rungs: List[RungTEAResult]
    capacity_t_yr: float
    electricity_price_kWh: float
    cell_voltage_V: float
    faradaic_efficiency: float
    discount_rate: float
    plant_lifetime_yr: int
    crf: float
    annuity_factor: float
    rank_by_ladder_margin: List[str]
    rank_by_tea_margin: List[str]
    n_pairwise_flips: int
    ranking_preserved: bool
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rungs": [r.to_dict() for r in self.rungs],
            "capacity_t_yr": self.capacity_t_yr,
            "electricity_price_kWh": self.electricity_price_kWh,
            "cell_voltage_V": self.cell_voltage_V,
            "faradaic_efficiency": self.faradaic_efficiency,
            "discount_rate": self.discount_rate,
            "plant_lifetime_yr": self.plant_lifetime_yr,
            "crf": self.crf,
            "annuity_factor": self.annuity_factor,
            "rank_by_ladder_margin": self.rank_by_ladder_margin,
            "rank_by_tea_margin": self.rank_by_tea_margin,
            "n_pairwise_flips": self.n_pairwise_flips,
            "ranking_preserved": self.ranking_preserved,
            "context": self.context,
        }


# ════════════════════════════════════════════════════════════════════════
#  Core evaluation
# ════════════════════════════════════════════════════════════════════════

def _annuity_factor(rate: float, n: int) -> float:
    """Capital-recovery inverse: present value of $1/yr for n years."""
    if n <= 0:
        return 0.0
    if abs(rate) < 1e-12:
        return float(n)
    return (1.0 - (1.0 + rate) ** (-n)) / rate


def _irr(annual_net: float, capex: float, n: int) -> Optional[float]:
    """Bisection IRR of −CAPEX + net × annuity(r, n); None if unrecoverable."""
    if capex <= 0.0 or n <= 0:
        return None
    if annual_net <= 0.0:
        return None

    def npv(r: float) -> float:
        return -capex + annual_net * _annuity_factor(r, n)

    lo, hi = -0.9, 10.0          # r > −1 required for the annuity; 10 = cap
    if npv(hi) > 0.0:
        return hi                # "≥1000 %": capped, disclosed by caller
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if npv(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def evaluate_rung_tea(
    rung: ProductRung,
    conditions: Optional[cell_architecture.OperatingConditions] = None,
    electricity_price_kWh: Optional[float] = None,
    capacity_t_yr: float = DEFAULT_PLANT_CAPACITY_T_YR,
    opex_model: Optional[technoeconomic.OPEXModel] = None,
    capex_model: Optional[technoeconomic.CAPEXModel] = None,
    labor_cost_per_yr: Optional[float] = None,
    site: Optional[Any] = None,
) -> RungTEAResult:
    """Full plant TEA for one product rung.

    Parameters
    ----------
    rung
        The ``product_ladder.ProductRung`` to evaluate.
    conditions
        Shared electrolyte/economic context — defaults to
        ``cell_architecture.OperatingConditions()`` (same live default the
        ladder uses, so the screening-gap comparison is apples-to-apples).
    electricity_price_kWh
        Defaults (in order): the ``site`` grid's effective price, else the
        *live* ``OPEXModel`` default — the same rederivation hook as the
        ladder.
    capacity_t_yr
        Nameplate product capacity (scenario knob).
    opex_model / capex_model
        Inject custom cost models (tests use zeroed instances to prove the
        cost stack closes on the ladder's contribution cost).  Defaults are
        the live ``technoeconomic`` class defaults.
    labor_cost_per_yr
        Defaults (in order): the ``site`` labour line, else the live
        ``OPEXModel`` labour default.
    site
        Optional duck-typed ``dark_mill.SiteDefinition``: provides grid
        price and labour.  Imported only structurally (no cell-physics run).
    """
    cond = conditions or cell_architecture.OperatingConditions()
    om = opex_model or technoeconomic.OPEXModel()
    cm = capex_model or technoeconomic.CAPEXModel()

    if electricity_price_kWh is None:
        if site is not None:
            electricity_price_kWh = float(site.grid.effective_price_kWh)
        else:
            electricity_price_kWh = om.electricity_price_kWh
    if labor_cost_per_yr is None:
        if site is not None:
            labor_cost_per_yr = float(site.labor_cost_per_yr)
        else:
            labor_cost_per_yr = om.labor_cost_per_yr

    cap = float(capacity_t_yr)
    if cap <= 0.0:
        raise ValueError("capacity_t_yr must be positive")

    # ── cell block, live ─────────────────────────────────────────────
    arch = cell_architecture.evaluate_architecture(
        cell_architecture.ARCHITECTURES[rung.architecture_id], cond
    )
    area_m2 = cap / arch.areal_productivity_t_m2_yr
    cell_capex = area_m2 * arch.installed_cost_per_m2

    # ── ore-side plant (leaching + grinding capacity lines) ──────────
    leaching_capex = cm.leaching_cost_per_tpy * cap
    grinding_capex = cm.grinding_cost_per_tpy * cap
    total_capex = cell_capex + leaching_capex + grinding_capex

    capital_per_t = cond.crf * total_capex / cap
    cell_capital_per_t = cond.crf * cell_capex / cap

    # ── energy build-up ──────────────────────────────────────────────
    dc_kwh_t = electrochemistry.specific_energy_kWh_per_t(
        cond.cell_voltage_V, cond.faradaic_efficiency
    )
    postop_kwh_t = rung.aux_energy_kWh_per_t
    grinding_kwh_t = om.grinding_energy_kWh_per_t
    electricity_per_t = (dc_kwh_t + postop_kwh_t + grinding_kwh_t) * electricity_price_kWh

    # ── variable OPEX ($/t) ──────────────────────────────────────────
    ore_per_t = om.ore_cost_per_t_Fe
    electrolyte_per_t = om.electrolyte_makeup_per_t_Fe
    water_per_t = om.water_cost_per_t_Fe
    anode_per_t = om.anode_replacement_cost_per_m2_yr * area_m2 / cap
    postop_cash_per_t = rung.aux_cash_per_t

    variable_per_t = (
        electricity_per_t + ore_per_t + electrolyte_per_t + water_per_t
        + anode_per_t + postop_cash_per_t
    )
    overhead_per_t = om.overhead_pct * variable_per_t

    # ── fixed OPEX ($/t) ─────────────────────────────────────────────
    maintenance_per_t = om.maintenance_pct_capex * total_capex / cap
    insurance_per_t = om.insurance_pct_capex * total_capex / cap
    labor_per_t = labor_cost_per_yr / cap

    total_opex_per_t = (
        variable_per_t + overhead_per_t
        + maintenance_per_t + insurance_per_t + labor_per_t
    )
    full_cost_per_t = capital_per_t + total_opex_per_t

    # ── revenue & investment metrics ─────────────────────────────────
    p_low, p_mid, p_high = rung.price.low, rung.price.mid, rung.price.high
    margin_mid = p_mid - full_cost_per_t
    margin_low = p_low - full_cost_per_t
    margin_high = p_high - full_cost_per_t
    margin_share = margin_mid / p_mid if p_mid > 0 else float("-inf")

    annual_net = margin_mid * cap
    annuity = _annuity_factor(cond.discount_rate, cond.plant_lifetime_yr)
    npv = -total_capex + annual_net * annuity
    payback = (total_capex / annual_net) if annual_net > 0 else None
    irr = _irr(annual_net, total_capex, cond.plant_lifetime_yr)

    # ── screening gap vs the contribution-margin ladder (live) ───────
    ladder = evaluate_rung(
        rung, cond, electricity_price_kWh=electricity_price_kWh
    )
    gap = ladder.margin_per_t - margin_mid
    excluded = {
        "grinding electricity": grinding_kwh_t * electricity_price_kWh,
        "ore feedstock": ore_per_t,
        "electrolyte make-up": electrolyte_per_t,
        "water": water_per_t,
        "anode wear": anode_per_t,
        "overhead (10% of variable)": overhead_per_t,
        "maintenance": maintenance_per_t,
        "insurance": insurance_per_t,
        "labour": labor_per_t,
        "ore-side plant capital": capital_per_t - cell_capital_per_t,
        # cell-capital basis must match the ladder exactly; residual is the
        # ladder's rounding — kept visible rather than forced to zero.
        "basis residual": gap - (
            grinding_kwh_t * electricity_price_kWh + ore_per_t
            + electrolyte_per_t + water_per_t + anode_per_t + overhead_per_t
            + maintenance_per_t + insurance_per_t + labor_per_t
            + (capital_per_t - cell_capital_per_t)
        ),
    }

    if margin_mid <= 0:
        verdict = "stalls"
    elif margin_share < MARGIN_VERDICT_FRACTION:
        verdict = "marginal"
    else:
        verdict = "clears"

    return RungTEAResult(
        rung_id=rung.id,
        name=rung.name,
        option=rung.option,
        architecture_id=rung.architecture_id,
        capacity_t_yr=cap,
        electricity_price_kWh=electricity_price_kWh,
        labor_cost_per_yr=labor_cost_per_yr,
        areal_productivity_t_m2_yr=arch.areal_productivity_t_m2_yr,
        electrode_area_m2=area_m2,
        installed_cost_per_m2=arch.installed_cost_per_m2,
        cell_capex_usd=cell_capex,
        leaching_capex_usd=leaching_capex,
        grinding_capex_usd=grinding_capex,
        total_capex_usd=total_capex,
        capital_charge_per_t=capital_per_t,
        cell_capital_charge_per_t=cell_capital_per_t,
        dc_energy_kWh_per_t=dc_kwh_t,
        postop_energy_kWh_per_t=postop_kwh_t,
        grinding_energy_kWh_per_t=grinding_kwh_t,
        electricity_usd_per_t=electricity_per_t,
        ore_usd_per_t=ore_per_t,
        electrolyte_usd_per_t=electrolyte_per_t,
        water_usd_per_t=water_per_t,
        anode_usd_per_t=anode_per_t,
        postop_cash_usd_per_t=postop_cash_per_t,
        overhead_usd_per_t=overhead_per_t,
        maintenance_usd_per_t=maintenance_per_t,
        insurance_usd_per_t=insurance_per_t,
        labor_usd_per_t=labor_per_t,
        variable_opex_per_t=variable_per_t,
        total_opex_per_t=total_opex_per_t,
        full_cost_per_t=full_cost_per_t,
        price_low_per_t=p_low,
        price_mid_per_t=p_mid,
        price_high_per_t=p_high,
        margin_per_t=margin_mid,
        margin_at_low=margin_low,
        margin_at_high=margin_high,
        margin_share_of_price=margin_share,
        annual_net_usd=annual_net,
        npv_usd=npv,
        payback_yr=payback,
        irr=irr,
        ladder_margin_per_t=ladder.margin_per_t,
        ladder_verdict=ladder.verdict,
        screening_gap_per_t=gap,
        excluded_lines_per_t=excluded,
        verdict=verdict,
    )


def _count_pairwise_flips(order_a: Sequence[str], order_b: Sequence[str]) -> int:
    """Number of rung pairs ranked in opposite order by two metrics."""
    pos_a = {rid: i for i, rid in enumerate(order_a)}
    pos_b = {rid: i for i, rid in enumerate(order_b)}
    ids = list(order_a)
    flips = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if (pos_a[ids[i]] - pos_a[ids[j]]) * (pos_b[ids[i]] - pos_b[ids[j]]) < 0:
                flips += 1
    return flips


def evaluate_ladder_tea(
    conditions: Optional[cell_architecture.OperatingConditions] = None,
    electricity_price_kWh: Optional[float] = None,
    capacity_t_yr: float = DEFAULT_PLANT_CAPACITY_T_YR,
    rung_ids: Optional[Sequence[str]] = None,
    site: Optional[Any] = None,
    **kwargs: Any,
) -> LadderTEAResult:
    """Evaluate all rungs (or a subset) through the full plant TEA."""
    cond = conditions or cell_architecture.OperatingConditions()
    if electricity_price_kWh is None:
        if site is not None:
            electricity_price_kWh = float(site.grid.effective_price_kWh)
        else:
            electricity_price_kWh = technoeconomic.OPEXModel().electricity_price_kWh
    ids = list(rung_ids) if rung_ids is not None else list(RUNGS.keys())
    results = [
        evaluate_rung_tea(
            RUNGS[i], cond,
            electricity_price_kWh=electricity_price_kWh,
            capacity_t_yr=capacity_t_yr, site=site, **kwargs,
        )
        for i in ids
    ]

    by_ladder = sorted(results, key=lambda r: r.ladder_margin_per_t, reverse=True)
    by_tea = sorted(results, key=lambda r: r.margin_per_t, reverse=True)
    order_ladder = [r.rung_id for r in by_ladder]
    order_tea = [r.rung_id for r in by_tea]
    flips = _count_pairwise_flips(order_ladder, order_tea)

    return LadderTEAResult(
        rungs=results,
        capacity_t_yr=float(capacity_t_yr),
        electricity_price_kWh=electricity_price_kWh,
        cell_voltage_V=cond.cell_voltage_V,
        faradaic_efficiency=cond.faradaic_efficiency,
        discount_rate=cond.discount_rate,
        plant_lifetime_yr=cond.plant_lifetime_yr,
        crf=cond.crf,
        annuity_factor=_annuity_factor(cond.discount_rate, cond.plant_lifetime_yr),
        rank_by_ladder_margin=order_ladder,
        rank_by_tea_margin=order_tea,
        n_pairwise_flips=flips,
        ranking_preserved=(flips == 0),
        context={
            "site": getattr(site, "name", None),
            "capital_accounting": (
                "cell block = cell_architecture installed $/m² (installed factor "
                "≈ CAPEXModel indirect stack incl. BOP); ore-side leaching + "
                "grinding $/tpy added; CAPEXModel BOP NOT re-added (no double "
                "count)."
            ),
            "screening_flag": SCREENING_FLAG,
        },
    )


# ════════════════════════════════════════════════════════════════════════
#  Reporting
# ════════════════════════════════════════════════════════════════════════

def comparison_table(result: LadderTEAResult) -> str:
    cols = (
        "rung", "opt", "$/t mid", "full cost", "TEA margin", "share",
        "ladder m.", "gap", "NPV M$", "IRR", "verdict",
    )
    lines = [
        "  ".join(f"{c:<10}" for c in cols),
        "-" * 124,
    ]
    for r in result.rungs:
        npv = "n/a" if r.npv_usd is None else f"{r.npv_usd / 1e6:,.0f}"
        irr = "n/a" if r.irr is None else f"{r.irr * 100:.0f}%"
        lines.append(
            f"{r.rung_id:<16}{r.option:<5}{r.price_mid_per_t:>8,.0f}"
            f"{r.full_cost_per_t:>11,.0f}{r.margin_per_t:>12,.0f}"
            f"{r.margin_share_of_price * 100:>6.0f}%"
            f"{r.ladder_margin_per_t:>11,.0f}{r.screening_gap_per_t:>7,.0f}"
            f"{npv:>9}{irr:>6}  {r.verdict}"
        )
    return "\n".join(lines)


def gap_table(result: LadderTEAResult, top_n: int = 4) -> str:
    """Itemise, per rung, which excluded lines eat the screening gap."""
    lines = ["rung            gap $/t   top excluded lines (the ladder's blind spot)"]
    lines.append("-" * 96)
    for r in result.rungs:
        items = sorted(r.excluded_lines_per_t.items(), key=lambda kv: kv[1],
                       reverse=True)
        top = ", ".join(f"{k} {v:,.0f}" for k, v in items[:top_n] if v > 0.005)
        lines.append(f"{r.rung_id:<16}{r.screening_gap_per_t:>8,.0f}   {top}")
    return "\n".join(lines)


def provenance(root: Optional[Path] = None) -> Dict[str, Any]:
    """Publication-artifact provenance stamp (docs/REPO_OUTPUT_POLICY.md)."""
    r = root or ROOT
    return {
        "artifact": "ladder_tea",
        "recipe": "python -m models.run_ladder_tea  (full-grade; "
                  "screening constants are L1 by declaration, not by mode)",
        "mode": "full-grade",
        "source_hashes": {
            name: _sha16(r / name) for name in _STAMPED_SOURCES
        },
    }


def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def render_markdown(result: LadderTEAResult, root: Optional[Path] = None) -> str:
    """Generate docs/LADDER_TEA.md from the live result."""
    prov = provenance(root)
    L: List[str] = []
    a = L.append
    a("# Ladder × TEA — Does the Product Ranking Survive Full Plant Costing?")
    a("")
    a("> **Generated artifact — do not hand-edit.** Rebuild with "
      "`python -m models.run_ladder_tea` (or `aq-steel-ladder-tea`) after any "
      "change to the models it derives from (product_ladder, cell_architecture, "
      "electrochemistry, technoeconomic, thermomechanical, anchors). Numbers "
      "re-derive on every run; the comparative structure is what is "
      "decision-grade, not the decimals.")
    a("")
    a(f"<!-- provenance:\n```json\n{json.dumps(prov, indent=2)}\n```\n-->")
    a("")
    a("**Screening flag:** unvalidated (L1). Companion to "
      "[PRODUCT_VALUE_LADDER.md](PRODUCT_VALUE_LADDER.md): the ladder screens "
      "rungs on contribution margin; this document prices **the whole plant** "
      "against the same live state.")
    a("")
    a("---")
    a("")
    a("## 1. The question")
    a("")
    a("`product_ladder.py` ranks rungs on contribution margin — price minus "
      "(DC electricity + post-cell unit ops + installed cell capital) — and "
      "deliberately holds feedstock, electrolyte make-up, anode wear, labour, "
      "maintenance, insurance, overhead and the ore-side plant constant "
      "(\"they live in the TEA\"). That is a valid screen **only if the "
      "excluded lines are rung-independent**. This module runs every rung "
      "through the full plant cost stack at a common nameplate and checks the "
      "assumption directly.")
    a("")
    a("## 2. Method and deliberate accounting choices")
    a("")
    a(f"* Nameplate: **{result.capacity_t_yr:,.0f} t product/yr** "
      "(scenario knob, `--capacity`; per-t variable costs are capacity-free, "
      "the capacity-sensitive lines — labour, ore-side plant — are disclosed "
      "per tonne).")
    a("* Cell block: area = capacity ÷ live architecture productivity; "
      "CAPEX = area × live installed $/m². The installed factor is declared "
      "by `OperatingConditions` as the round-trip equivalent of "
      "`CAPEXModel`'s indirect stack **including rectifier/electrolyte BOP**, "
      "so CAPEXModel BOP lines are *not* re-added (no double counting). "
      "Only the ore-side plant (leaching + grinding $/tpy) is added.")
    a("* OPEX lines from the live `OPEXModel` defaults: electricity "
      "(DC + post-op + grinding kWh), ore, electrolyte make-up, water, anode "
      "wear (∝ architecture area), post-op cash (rung unit ops), overhead "
      "(10 % of variable), maintenance + insurance (∝ CAPEX), labour.")
    a(f"* Money: WACC {result.discount_rate:.0%}, "
      f"{result.plant_lifetime_yr} yr → CRF {result.crf:.4f}; "
      "NPV = −CAPEX + net × annuity (construction period ignored, L1).")
    a(f"* Conditions: V_cell {result.cell_voltage_V} V, FE "
      f"{result.faradaic_efficiency}, electricity "
      f"${result.electricity_price_kWh}/kWh — identical to the ladder's live "
      "conditions so the gap is attributable to cost structure, not to "
      "physics drift.")
    a("")
    a("## 3. Full-TEA results vs the contribution screen")
    a("")
    a("```")
    a(comparison_table(result))
    a("```")
    a("")
    a("## 4. The screening gap, itemised")
    a("")
    a("Gap = ladder contribution margin − full-TEA margin = exactly the cost "
      "lines the ladder excludes. Where it is uniform across rungs it cannot "
      "change the decision; where it varies, it names the cost that could.")
    a("")
    a("```")
    a(gap_table(result))
    a("```")
    a("")
    a("## 5. Ranking verdict")
    a("")
    a(f"* Ladder order (margin $/t): `{' > '.join(result.rank_by_ladder_margin)}`")
    a(f"* Full-TEA order (margin $/t): `{' > '.join(result.rank_by_tea_margin)}`")
    a(f"* Pairwise order flips: **{result.n_pairwise_flips}**")
    a("")
    if result.ranking_preserved:
        gaps = [r.screening_gap_per_t for r in result.rungs]
        spread = max(gaps) - min(gaps)
        a(f"**The ranking is preserved under full costing.** The ladder's "
          "contribution screen is decision-grade for *which product to make*; "
          "full-TEA margins sit lower in absolute dollars by the itemised gap "
          f"above (rung-uniform to within ${spread:,.0f}/t at this capacity). "
          "The decision risk is therefore not cost structure but the "
          "anchored price bands and the L1 physics flags, in that order.")
    else:
        a("**The ranking changes under full costing — see the flipped pairs "
          "above.** Those rung pairs have a form- or scale-dependent cost the "
          "contribution screen cannot see (area-driven anode wear and "
          "maintenance, post-op cash, or labour dilution). Treat the screen's "
          "ordering of those pairs as provisional.")
    a("")
    a("## 6. Caveats (read before citing)")
    a("")
    a("* All cost lines are `technoeconomic.py` screening defaults; none are "
      "quoted projects. Product prices are anchored bands, battery-iron "
      "especially speculative.")
    a("* Construction period and working capital are ignored; IRR/NPV are "
      "ranking aids, not investment-committee numbers (IRR is capped at "
      "1,000 % by the solver, so \"1000 %\" reads as \"above cap\").")
    a("* The ladder's `clears/stalls` verdicts and this table's may differ: "
      "the ladder asks \"margin within the screen\", this asks \"margin "
      "within the plant\". Both are shown, neither is hidden.")
    a("")
    return "\n".join(L)


def write_artifacts(
    result: LadderTEAResult,
    root: Optional[Path] = None,
    doc: bool = True,
) -> Dict[str, Path]:
    """Write the JSON report (and the regenerated doc) with provenance."""
    r = root or ROOT
    data = result.to_dict()
    data["_provenance"] = provenance(r)
    json_path = r / DEFAULT_JSON_PATH.relative_to(ROOT)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(data, indent=2) + "\n")
    out = {"json": json_path}
    if doc:
        doc_path = r / DEFAULT_DOC_PATH.relative_to(ROOT)
        doc_path.write_text(render_markdown(result, r))
        out["doc"] = doc_path
    return out


def model_scope() -> Dict[str, Any]:
    return {
        "screening_flag": SCREENING_FLAG,
        "live_derivations": [
            "cell_architecture.evaluate_architecture (areal productivity, "
            "installed $/m², cell capital charge)",
            "electrochemistry.specific_energy_kWh_per_t (DC kWh/t)",
            "technoeconomic CAPEXModel/OPEXModel class defaults (every "
            "non-cell cost line)",
            "product_ladder.evaluate_rung (ladder margin for the gap)",
            "product_ladder rung prices/post-ops (anchors + live anneal)",
            "cell_architecture.OperatingConditions (CRF, WACC, lifetime)",
        ],
        "screening_proxies_anchored": [
            "product price bands (battery-iron most speculative)",
            "ore/electrolyte/water/anode/labour/indirect percentages",
        ],
        "out_of_scope": [
            "construction schedule, working capital, tax, financing",
            "site physics (dark_mill.size_dark_mill owns v_cell/FE/j solve)",
            "revenue ramps / market share",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> None:  # pragma: no cover - CLI
    p = argparse.ArgumentParser(
        description="Full plant TEA for every product-ladder rung "
                    "(ladder × technoeconomic wiring)."
    )
    p.add_argument("--capacity", type=float, default=DEFAULT_PLANT_CAPACITY_T_YR,
                   help="nameplate product capacity, t/yr")
    p.add_argument("--elec-price", type=float, default=None,
                   help="$/kWh (default: live OPEXModel price, or site grid price)")
    p.add_argument("--rung", type=str, default=None,
                   help="evaluate a single rung id (default: all)")
    p.add_argument("--site", type=str, default=None,
                   help="dark_mill EXAMPLE_SITES key: use its grid price + labour")
    p.add_argument("--json-out", type=str, default=None,
                   help="also write the JSON report to this path")
    args = p.parse_args(argv)

    site = None
    if args.site is not None:
        from .dark_mill import EXAMPLE_SITES          # lazy: heavy import
        site = EXAMPLE_SITES[args.site]

    result = evaluate_ladder_tea(
        electricity_price_kWh=args.elec_price,
        capacity_t_yr=args.capacity,
        rung_ids=[args.rung] if args.rung else None,
        site=site,
    )
    print(f"ladder_tea — full plant TEA per product rung  [{SCREENING_FLAG}]")
    print(f"capacity {result.capacity_t_yr:,.0f} t/yr · V_cell "
          f"{result.cell_voltage_V} V · FE {result.faradaic_efficiency} · "
          f"${result.electricity_price_kWh}/kWh"
          + (f" · site: {result.context['site']}" if result.context["site"] else ""))
    print()
    print(comparison_table(result))
    print()
    print(gap_table(result))
    print()
    print(f"rank by ladder margin : {' > '.join(result.rank_by_ladder_margin)}")
    print(f"rank by full-TEA margin: {' > '.join(result.rank_by_tea_margin)}")
    print(f"pairwise flips: {result.n_pairwise_flips} "
          f"({'ranking preserved' if result.ranking_preserved else 'RANKING CHANGES under full costing'})")
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = result.to_dict()
        data["_provenance"] = provenance()
        out.write_text(json.dumps(data, indent=2) + "\n")
        print(f"[wrote {out}]")


if __name__ == "__main__":  # pragma: no cover
    main()
