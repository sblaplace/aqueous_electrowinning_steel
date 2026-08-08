"""Melt-shop remelt balance — what does an electrowon charge DO in a furnace?

Why this module exists
----------------------
The near-term product path (``docs/RESEARCH_PROGRAM.md``, Option A) is a
qualified iron *feedstock* for a melt shop, and the product value ladder
(``models/product_ladder.py``, rungs ``flake_feed`` / ``own_melt_bar``) needs
this gate to say *anything quantitative* about what the buyer receives.
``melt_hydrogen.py`` answers the buyer's hydrogen question; nobody answered
the other three:

* **Iron yield** — how much of the charged iron reports to metal after its
  oxide is (or is not) carbothermically reduced, and after fines are lost
  to off-gas dust;
* **CO boil** — the reducible oxygen must become CO somewhere; mass of CO,
  carbon demand, and the endothermic penalty of $FeO + C → Fe + CO$;
* **Slag / fume** — unrecovered FeO to slag, lime practice for charge sulfur,
  and the dust stream from fines.

Physics / chemistry
-------------------
Per tonne of electrowon charge (screening, EAF and induction routes):

.. code-block:: text

    charge O (as-deposited + post-harvest pickup)
      → reducible fraction  r_route  carboreduced:  FeO + C → Fe + CO
        C   =  r·O·(12/16)            (kg C per tonne)
        CO  =  r·O·(28/16)            (kg CO per tonne — the boil)
        Q   =  r·O·ΔH/16 g·mol⁻¹ · 149 kJ/mol   (endothermic penalty)
      → unreducible fraction (1 − r) reports to slag as FeO (route-dependent)
    fines → off-gas dust at a route capture fraction (EAF ≫ induction)
    charge S → lime practice (slag addition ∝ S at fixed basicity)
    charge H → boil-off / white-spot verdict via melt_hydrogen (LIVE call)

Yield gate: the product must at minimum match the #1-scrap band (~94.5 %)
and should approach DRI/pig (97 %).  The route matters: an EAF with carbon
injection recovers ~90 % of charged oxide; an induction furnace (no carbon
boil, slag Fo attack on lining) recovers less and pays a bigger slag bill —
this is why "sell flake to an induction foundry" and "sell to an EAF" are
*different products*, and why Option A.5 (own-melt) wants the EAF line.

Live derivations
----------------
* as-deposited O comes from ``models/oxygen_in_iron.py`` at call time
  (``ChargeState.o_wt_pct=None``), the same L1 engine the deposit gates use;
  post-harvest pickup comes live from ``models/product_oxidation.py``
  (passivation-film physics per product form, V6 §1.2).
* the H verdict is a live call to ``models/melt_hydrogen.py``.
* charge S comes live from ``models/rinse_carryover.py`` (counter-current
  rinse-train carryover, V6 §1.3); anchor ``CHARGE_S_WT_PCT`` is the
  fallback.
* shipped-product fines come live from ``models/briquetting.py``
  (recommended densification line, V6 §1.4); anchor
  ``FINES_FRACTION_PASSIVATED`` is the fallback.

Screening flag
--------------
L1.  Route recovery fractions, dust captures, lime practice — anchored
screening proxies for EAF/induction practice.  Stoichiometry is exact;
recovery kinetics are not modelled (that is what customer melt trials
measure).  Baselines (scrap/DRI/pig) are anchored comparisons only.

References
----------
* docs/CHEM_PHYS_IMPROVEMENTS_V6.md §1.5 (this module's gap statement).
* Turkdogan, *Fundamentals of Steelmaking*; IISI EAF mass-balance practice.
* USBM RI-series historical iron-EW product specs (references/README.md §7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .anchors import get_anchor

SCREENING_FLAG = "unvalidated (L1)"

# atomic weights (kg/mol), consistent with thermodynamic_constants rounding
M_FE = 55.845
M_O = 16.00
M_C = 12.011
M_CO = M_C + M_O                    # CO mass closure by construction
M_S = 32.06

FE_PER_O_FEO = M_FE / M_O            # Fe bound per O in FeO-like oxide
FEO_PER_O = (M_FE + M_O) / M_O       # FeO mass per O
C_PER_O = M_C / M_O
CO_PER_O = M_CO / M_O

ROUTES = ("eaf", "induction")


@dataclass(frozen=True)
class ChargeState:
    """The electrowon product as it arrives at the furnace gate.

    ``o_wt_pct=None`` means *derive live*: as-deposited O from
    ``oxygen_in_iron.OxygenInIronModel().predict()`` at the reference
    operating point plus the post-harvest pickup computed live by
    ``product_oxidation`` for this product form (passivation film).
    """

    product_form: str = "briquette"   # briquette | flake | powder | foil
    charge_fraction: float = 1.0      # electrowon share of the furnace charge
    o_wt_pct: Optional[float] = None
    s_wt_pct: Optional[float] = None
    fines_fraction: Optional[float] = None
    c_h_ppm: Optional[float] = None
    tramp_note: str = "residual-free (Cu/Sn below spec) — ladder screening"


@dataclass(frozen=True)
class MeltRouteParams:
    """Route practice parameters, all anchored screening proxies."""

    oxide_recovery_frac: float
    dust_capture_frac: float
    dust_fe_fraction: float
    feo_reduction_dh_kJ_mol: float
    lime_per_s_kg: float
    base_gangue_slag_kg_t: float

    # verdict thresholds (module params, not anchors: they are the buyer's
    # acceptance band, stated as screening defaults)
    yield_gate_pct: float = 96.5
    co_boil_soft_max_kg_t: float = 60.0   # above: slopping risk on cold charge
    co_boil_soft_min_kg_t: float = 10.0   # below: inject carbon anyway (noted)
    h_flake_gate: bool = True             # respect melt_hydrogen verdict


def route_params(route: str) -> MeltRouteParams:
    """Anchored practice parameters per route (live from anchors registry)."""
    if route not in ROUTES:
        raise ValueError(f"route must be one of {ROUTES}, got {route!r}")
    rec_key = ("EAF_OXIDE_RECOVERY_FRAC" if route == "eaf"
               else "INDUCTION_OXIDE_RECOVERY_FRAC")
    dust_key = ("DUST_CAPTURE_FINES_EAF" if route == "eaf"
                else "DUST_CAPTURE_FINES_INDUCTION")
    return MeltRouteParams(
        oxide_recovery_frac=get_anchor(rec_key).value,
        dust_capture_frac=get_anchor(dust_key).value,
        dust_fe_fraction=get_anchor("DUST_FE_FRACTION").value,
        feo_reduction_dh_kJ_mol=get_anchor("FEO_C_REDUCTION_DH_KJ_MOL").value,
        lime_per_s_kg=get_anchor("LIME_PER_S_KG").value,
        base_gangue_slag_kg_t=get_anchor("BASE_GANGUE_SLAG_KG_T").value,
    )


def _live_as_deposited_o_wt_pct(product_form: str = "briquette") -> float:
    """As-deposited O (wt%) + live post-harvest pickup per product form.

    Both legs try/except with anchor fallback (same pattern as
    product_ladder's anneal link): as-deposited O from the oxygen engine;
    post-harvest pickup from ``product_oxidation`` passivation-film
    physics (V6 §1.2), with the anchor row documenting the screening
    band it replaced.
    """
    try:
        from .product_oxidation import postharvest_o_pickup_wt_pct

        pickup = float(postharvest_o_pickup_wt_pct(product_form))
    except Exception:  # pragma: no cover - defensive
        pickup = get_anchor("POSTHARVEST_O_PICKUP_WT_PCT").value
    try:
        from .oxygen_in_iron import OxygenInIronModel

        o_ppm = float(OxygenInIronModel().predict()["o_ppm"])
        as_deposited = o_ppm / 1.0e4  # ppm -> wt%
    except Exception:  # pragma: no cover - defensive
        as_deposited = get_anchor("AS_DEPOSITED_O_WT_PCT").value
    return as_deposited + pickup


def _live_charge_s_wt_pct() -> float:
    """Charge S (wt%) from the rinse-train physics + anchored fallback.

    try/except with anchor fallback, same pattern as the oxygen link:
    ``rinse_carryover.default_charge_s_wt_pct`` computes the well-rinsed
    counter-current train's carryover (V6 §1.3); the anchor row documents
    the screening band it replaced.
    """
    try:
        from .rinse_carryover import default_charge_s_wt_pct

        return float(default_charge_s_wt_pct())
    except Exception:  # pragma: no cover - defensive
        return get_anchor("CHARGE_S_WT_PCT").value


def _live_fines_fraction() -> float:
    """Shipped-product fines as a mass fraction from the densification line.

    ``briquetting.shipped_fines_fraction`` runs the recommended Option-A
    line (hot die-press of the column powder, V6 §1.4) live — the §1.4 →
    §1.5 dust channel; the anchor row documents the handling band it
    replaced.
    """
    try:
        from .briquetting import shipped_fines_fraction

        return float(shipped_fines_fraction())
    except Exception:  # pragma: no cover - defensive
        return get_anchor("FINES_FRACTION_PASSIVATED").value


def _resolved_state(state: ChargeState) -> Dict[str, float]:
    return {
        "o_wt_pct": (
            state.o_wt_pct if state.o_wt_pct is not None
            else _live_as_deposited_o_wt_pct(state.product_form)
        ),
        "s_wt_pct": (
            state.s_wt_pct if state.s_wt_pct is not None
            else _live_charge_s_wt_pct()
        ),
        "fines_fraction": (
            state.fines_fraction if state.fines_fraction is not None
            else _live_fines_fraction()
        ),
        "c_h_ppm": (
            state.c_h_ppm if state.c_h_ppm is not None
            else get_anchor("CHARGE_H_PPM").value
        ),
    }


@dataclass
class MeltVerdict:
    """The buyer's numbers for one charge state on one route."""

    route: str
    product_form: str
    charge_fraction: float

    # charge state (resolved)
    o_wt_pct: float
    s_wt_pct: float
    fines_fraction: float
    c_h_ppm: float

    # mass balance (per tonne electrowon)
    o_kg_t: float
    fe_in_charge_kg_t: float
    fe_recovered_from_oxide_kg_t: float
    fe_to_slag_kg_t: float
    dust_kg_t: float
    fe_to_dust_kg_t: float
    fe_to_metal_kg_t: float
    fe_yield_pct: float

    # boil & slag
    co_boil_kg_t: float
    carbon_required_kg_t: float
    thermal_penalty_kWh_t: float
    lime_kg_t: float
    gangue_kg_t: float
    slag_total_kg_t: float

    # hydrogen (live melt_hydrogen verdict, charge-fraction adjusted)
    h_budget: Dict[str, Any] = field(default_factory=dict)

    # verdict
    verdict: str = ""
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["reasons"] = list(self.reasons)
        return d


def evaluate_charge(
    state: Optional[ChargeState] = None,
    route: str = "eaf",
    params: Optional[MeltRouteParams] = None,
) -> MeltVerdict:
    """Run the remelt balance for one charge state on one route."""
    st = state or ChargeState()
    p = params or route_params(route)
    r = p.oxide_recovery_frac
    res = _resolved_state(st)

    o_kg = res["o_wt_pct"] * 10.0            # wt% -> kg per tonne
    s_kg = res["s_wt_pct"] * 10.0
    fines_kg = res["fines_fraction"] * 1000.0

    fe_in = 1000.0 - o_kg - s_kg             # metal + oxide iron, tramp-free
    fe_recovered = r * o_kg * FE_PER_O_FEO   # oxide Fe back to metal
    fe_slag = (1.0 - r) * o_kg * FE_PER_O_FEO
    dust = fines_kg * p.dust_capture_frac
    fe_dust = dust * p.dust_fe_fraction
    fe_metal = fe_in - fe_slag - fe_dust
    yield_pct = 100.0 * fe_metal / fe_in if fe_in > 0 else 0.0

    carbon = r * o_kg * C_PER_O
    co = r * o_kg * CO_PER_O
    moles_o = r * o_kg * 1000.0 / M_O        # kg -> g -> mol
    penalty = moles_o * p.feo_reduction_dh_kJ_mol / 3600.0  # kJ -> kWh

    lime = s_kg * p.lime_per_s_kg
    gangue = p.base_gangue_slag_kg_t
    slag = fe_slag / FE_PER_O_FEO * FEO_PER_O + lime + gangue  # as FeO + flux

    try:
        from .melt_hydrogen import melt_hydrogen_budget

        h_budget = melt_hydrogen_budget(
            res["c_h_ppm"], charge_fraction=st.charge_fraction
        )
    except Exception:  # pragma: no cover - defensive
        h_budget = {"error": "melt_hydrogen unavailable",
                    "needs_bake_or_degas": None}

    reasons: List[str] = []
    if yield_pct >= p.yield_gate_pct:
        verdict = "qualified"
    elif yield_pct >= p.yield_gate_pct - 1.5:
        verdict = "conditional"
        reasons.append(
            f"yield {yield_pct:.1f}% within 1.5 pp of the {p.yield_gate_pct}% "
            f"gate — buyer trial decides")
    else:
        verdict = "fails"
        reasons.append(
            f"yield {yield_pct:.1f}% below {p.yield_gate_pct}% gate")
    if co > p.co_boil_soft_max_kg_t:
        verdict = "conditional" if verdict == "qualified" else verdict
        reasons.append(
            f"CO boil {co:.0f} kg/t above {p.co_boil_soft_max_kg_t:.0f} — "
            "slopping risk on cold charge; stage the feed")
    elif co < p.co_boil_soft_min_kg_t:
        reasons.append(
            f"CO boil {co:.0f} kg/t below {p.co_boil_soft_min_kg_t:.0f} — "
            "benign; foamy-slag practice may still inject carbon")
    if h_budget.get("needs_bake_or_degas") and p.h_flake_gate:
        verdict = "conditional" if verdict == "qualified" else verdict
        reasons.append(
            "excess H above white-spot threshold — bake/degas before melt "
            "(melt_hydrogen.py); induction melt offers no boil assist")
    if route == "induction":
        reasons.append(
            "induction route: no carbon-boil assist — oxide recovery and "
            "slag Fo lining attack are the binding constraints")

    return MeltVerdict(
        route=route,
        product_form=st.product_form,
        charge_fraction=st.charge_fraction,
        o_wt_pct=res["o_wt_pct"],
        s_wt_pct=res["s_wt_pct"],
        fines_fraction=res["fines_fraction"],
        c_h_ppm=res["c_h_ppm"],
        o_kg_t=o_kg,
        fe_in_charge_kg_t=fe_in,
        fe_recovered_from_oxide_kg_t=fe_recovered,
        fe_to_slag_kg_t=fe_slag,
        dust_kg_t=dust,
        fe_to_dust_kg_t=fe_dust,
        fe_to_metal_kg_t=fe_metal,
        fe_yield_pct=yield_pct,
        co_boil_kg_t=co,
        carbon_required_kg_t=carbon,
        thermal_penalty_kWh_t=penalty,
        lime_kg_t=lime,
        gangue_kg_t=gangue,
        slag_total_kg_t=slag,
        h_budget=h_budget,
        verdict=verdict,
        reasons=reasons,
    )


# ────────────────────────────────────────────────────────────────────────
#  Baselines: the same engine run on the buyer's alternatives
# ────────────────────────────────────────────────────────────────────────

def compare_baselines(
    state: Optional[ChargeState] = None,
    route: str = "eaf",
) -> Dict[str, Dict[str, Any]]:
    """Electrowon charge vs. the buyer's alternatives on the same engine.

    Scrap and pig rows are anchors (their chemistry is long-established);
    DRI runs the same oxide-reduction engine with the anchored DRI oxygen.
    """
    st = state or ChargeState()
    ew = evaluate_charge(st, route)
    scrap_yield = get_anchor("SCRAP_YIELD_PCT")
    dri_o = get_anchor("DRI_O_WT_PCT")
    dri_state = ChargeState(
        product_form="DRI/HBI pellet",
        charge_fraction=st.charge_fraction,
        o_wt_pct=dri_o.value,
        s_wt_pct=_resolved_state(st)["s_wt_pct"],
        fines_fraction=st.fines_fraction,
        c_h_ppm=0.0,
    )
    dri = evaluate_charge(dri_state, route)
    return {
        "electrowon": ew.to_dict(),
        "no1_scrap": {
            "fe_yield_pct": scrap_yield.value,
            "source": scrap_yield.ref,
            "note": "anchored band; tramp residuals (Cu/Sn) are the real "
                    "scrap ceiling, not yield — see hot_shortness.py",
            "anchor_uncertainty_pp": scrap_yield.uncertainty,
        },
        "dri_hbi": dri.to_dict(),
    }


def to_dict_full(state: Optional[ChargeState] = None) -> Dict[str, Any]:
    """Both routes + baselines + screening flag (machine-readable report)."""
    st = state or ChargeState()
    return {
        "screening_flag": SCREENING_FLAG,
        "charge_state": _resolved_state(st) | {
            "product_form": st.product_form,
            "charge_fraction": st.charge_fraction,
        },
        "eaf": evaluate_charge(st, "eaf").to_dict(),
        "induction": evaluate_charge(st, "induction").to_dict(),
        "baselines": compare_baselines(st),
    }


def comparison_table(state: Optional[ChargeState] = None) -> str:
    st = state or ChargeState()
    lines = [
        "route      O wt%  fines%  yield %  CO kg/t  C kg/t  dust kg/t  "
        "slag kg/t  penalty kWh/t  verdict",
        "-" * 104,
    ]
    for route in ROUTES:
        v = evaluate_charge(st, route)
        lines.append(
            f"{route:<10} {v.o_wt_pct:>5.2f} {v.fines_fraction*100:>6.1f}"
            f"{v.fe_yield_pct:>8.2f}{v.co_boil_kg_t:>8.1f}{v.carbon_required_kg_t:>7.2f}"
            f"{v.dust_kg_t:>10.1f}{v.slag_total_kg_t:>10.1f}"
            f"{v.thermal_penalty_kWh_t:>14.1f}  {v.verdict}"
        )
    v = evaluate_charge(st, "eaf")
    lines.append("reasons:")
    for why in v.reasons:
        lines.append(f"  - {why}")
    return "\n".join(lines)


def model_scope() -> Dict[str, Any]:
    return {
        "screening_flag": SCREENING_FLAG,
        "live_derivations": [
            "oxygen_in_iron.OxygenInIronModel.predict (as-deposited O)",
            "melt_hydrogen.melt_hydrogen_budget (H boil-off verdict)",
            "rinse_carryover.default_charge_s_wt_pct (charge sulfur, V6 §1.3)",
            "product_oxidation.postharvest_o_pickup_wt_pct (charge O "
            "pickup, V6 §1.2)",
            "briquetting.shipped_fines_fraction (charge fines as shipped, "
            "V6 §1.4)",
        ],
        "exact": [
            "C / CO / FeO stoichiometry of carboreduction",
            "Fe mass balance closure (metal + slag Fe + dust Fe = Fe in)",
        ],
        "screening_proxies_anchored": [
            "route oxide recovery fractions",
            "fines dust capture fractions",
            "lime-per-sulfur slag practice",
        ],
        "out_of_scope": [
            "kinetics of the boil / bath stirring rates",
            "furnace operating practice (foam depth, injection schedules)",
            "tramp-element melt chemistry (hot_shortness.py owns Cu/Sn)",
        ],
    }


def main() -> None:  # pragma: no cover - CLI wrapper
    print(f"melt_balance — melt-shop remelt verdict  [{SCREENING_FLAG}]")
    print(comparison_table())
    print()
    b = compare_baselines()
    print(f"baselines: #1 scrap yield {b['no1_scrap']['fe_yield_pct']}% "
          f"| DRI/HBI yield {b['dri_hbi']['fe_yield_pct']:.2f}% (same engine)")
    print(f"H verdict (live melt_hydrogen): "
          f"{b['electrowon']['h_budget'].get('needs_bake_or_degas')}")


if __name__ == "__main__":  # pragma: no cover
    main()
