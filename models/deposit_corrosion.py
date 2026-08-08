"""Deposit corrosion at open circuit & ferric etch — the silent iron-ledger leak (V6 §1.1).

Why this module exists
----------------------
``run_record.py`` demands a closed charge/mass/electrolyte balance and
``docs/NEXT_STEPS.md`` §1 makes the iron ledger gate evidence.  But nothing
in the repository models the deposit losing mass **without any current
through the cell**:

1. **Open-circuit (mixed-potential) corrosion.**  Freshly deposited,
   nanocrystalline, hydrogen-charged iron sits at −0.4 V vs SHE or below in
   a pH ~2 bath at 40–60 °C.  At open circuit it runs a corrosion cell on
   itself — ``Fe + 2H⁺ → Fe²⁺ + H₂`` (plus O₂ reduction to whatever extent
   air reaches the bath) — at corrosion current densities ``j_corr`` of
   order 1–50 µA/cm² (Stern 1955; Kelly 1965 screening band, V6 §1.1).
   Over an 8-hour idle or a weekend shutdown this is 0.1–2 µm of deposit:
   small per night, systematic with the number of idle cycles, and exactly
   the kind of term that shows up as "the ledger never quite closes."
   ``INDEPENDENT_SHUTDOWN.md`` covers the *safety* of shutdown;
   ``kinetics.py``'s BV dissolution branch covers the deposit only when it
   is polarised anodic of ``E_eq``; neither covers the *unpowered* state.
2. **Ferric etch.**  Membrane-crossover / PRE-produced Fe³⁺ chemically
   attacks the metal: ``2Fe³⁺ + Fe → 3Fe²⁺`` — the classical
   current-efficiency killer of the USBM chloride electrowinning
   flowsheets.  It leaks cathode iron back into the bath **without
   electrons**.  ``fe3_shuttle.py`` models the *electrochemical* shuttle
   (Fe³⁺ + e⁻ → Fe²⁺ at a powered, flowing cathode); the
   homogeneous/mixed-potential etch of the unpowered deposit is a distinct
   channel with a different rate law (half-order in a_Fe3, strongly
   T-dependent, acid-catalysed).  To avoid double counting, the Fe³⁺
   channel is *only* in the etch law here — the mixed-potential part
   carries H⁺ and O₂, matching the V6 sketch
   ``corrosion_current(...) + ferric_etch_flux(...)``.

Physics / chemistry
-------------------
.. code-block:: text

    mixed potential (self-pinned to the anchored reference state
    pH 2.0, 25 °C, deaerated, clean Fe — j_corr(anchor) exactly
    reproduced there by construction):

        i_Fe(E)  = j_ref·(1−θ)·Arr(T)·10^( E    / b_a)      Fe → Fe²⁺ + 2e⁻
        i_H(E)   = j_ref·(1−θ)·(a_H/a_H,ref)·Arr(T)
                            ·10^(−E / b_c)                   2H⁺ + 2e⁻ → H₂
        i_O2,lim = 4F·(D_O2(T)/δ)·[O₂]                        transport-limited
        E_corr:   i_Fe(E_corr) = i_H(E_corr) + i_O2,lim      (bisection)
        Fe flux  = i_Fe(E_corr)/(2F);  H₂ flux = i_H(E_corr)/(2F)

    ferric etch (separate channel):

        J_etch   = K_etch(T)·√(a_Fe3,surf/a_ref)·(a_H/a_H,ref)^m

    stagnant idle: Fe³⁺ reaches the surface through a growing layer
    δ(t)=√(2·D_Fe3·t); the etch rate is the harmonic combination of the
    kinetic law and the diffusive supply (kinetic at short t, √t supply
    at long t).  Stirred: kinetic law at bulk activity, constant in t.

    mass loss = ∫ (corrosion flux + etch flux) dt, optionally capped by
    the plated thickness actually left on the drum/strip.

Ledger accounting (why the iron ledger stays closed)
----------------------------------------------------
Both channels dissolve *plated* iron back into the *bath*: initial
inventory, deposit, and post-run bath Fe are all shifted such that the
mol-scale closure of ``run_record.compute_ledgers`` still reads zero.
The signature instead lands in the **charge ledger** (gravimetric FE
biased low by the redissolved amount; unresolved charge explained) and in
the **distribution** between deposit and bath inventories.  The module
therefore exports ``predicted_idle_terms(manifest)`` — a *predicted,
not measured* L1 term that ``run_record`` attaches to both ledgers so
residuals are tested *against* the prediction rather than absorbed into
"uncertainty".  ``closed_loop.campaign_idle_accounting`` carries the same
term into campaign Fe inventory (nights and weekends).

Live derivations
----------------
* bath Fe³⁺ default comes from ``fe3_shuttle.steady_state`` (sealed
  divided-cell scenario, pH-following) at call time — the bath enters
  idle with its end-of-plating ferric level; anchor fallback
  ``IDLE_BATH_FE3_M``.
* dissolved O₂ saturation from ``bath_startup.dissolved_o2_saturation_mol_L``
  (Weiss 1970), same source as the shuttle/bath-aging modules.
* D_Fe3 from ``fe3_shuttle.D_FE3_REF_M2_S`` (existing screening constant).

Screening flag
--------------
L1.  All kinetic constants are anchored screening proxies (Stern/Kelly
Tafel family for the acid channels; USBM RI-series / FeCl₃-etch practice
for the ferric channel — the etch rate constant is SPECULATIVE, flag set
in ``models/anchors.py``).  Stoichiometry and the mixed-potential solve
are exact; the coverage factor θ, the stagnant-layer picture, and the
half-order etch law are L1 structure, not fitted physics.  Nothing here
is gate evidence.

References
----------
* docs/CHEM_PHYS_IMPROVEMENTS_V6.md §1.1 (this module's gap statement).
* Stern (1955); Kelly (1965) — iron in acid, mixed-potential corrosion.
* USBM RI-series iron-EW flowsheets — ferric etch as the CE killer.
* Weiss (1970) — O₂ solubility (via bath_startup).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .anchors import get_anchor
from .bath_startup import dissolved_o2_saturation_mol_L
from .electrochemistry import FARADAY, M_FE, R_GAS, RHO_FE, T_REF
from .fe3_shuttle import D_FE3_REF_M2_S

SCREENING_FLAG = "unvalidated (L1)"

PH_REF = 2.0                    # reference pH of the anchored state (bath target, BATH_SPEC §1.4)
MIXING_MODES = ("stagnant", "stirred")

# Verdict thresholds (µm of deposit lost per idle event) — screening
# acceptance bands, exposed as module constants like melt_balance's gates.
LOSS_NEGLIGIBLE_UM = 0.05
LOSS_MATERIAL_UM = 2.0


@dataclass(frozen=True)
class IdleBathState:
    """Bath + deposit state during an unpowered (idle/shutdown) interval.

    ``None`` fields resolve at call time: bath Fe³⁺ live from
    ``fe3_shuttle.steady_state`` (sealed divided-cell scenario at the
    stated pH), O₂ fraction / additive coverage / diffusion layer /
    j_corr reference from ``models/anchors.py``.
    """

    pH: float = PH_REF
    T_C: float = 40.0           # bath cools toward RT during a long idle
    a_Fe3_bulk_M: Optional[float] = None
    o2_fraction_of_sat: Optional[float] = None
    theta_additive: Optional[float] = None
    mixing: str = "stagnant"
    area_m2: float = 1.0
    delta_m: Optional[float] = None            # quasi-steady layer for O₂
    j_corr_ref_uA_cm2: Optional[float] = None  # override anchored reference

    def __post_init__(self) -> None:
        if self.mixing not in MIXING_MODES:
            raise ValueError(f"mixing must be one of {MIXING_MODES}, got {self.mixing!r}")
        if self.pH < 0 or self.pH > 14:
            raise ValueError("pH must lie in [0, 14]")
        if self.T_C < 0.0 or self.T_C > 95.0:
            raise ValueError("T_C outside the aqueous screening window [0, 95]")
        for name in ("o2_fraction_of_sat", "theta_additive"):
            v = getattr(self, name)
            if v is not None and not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.area_m2 <= 0:
            raise ValueError("area_m2 must be positive")
        if self.a_Fe3_bulk_M is not None and self.a_Fe3_bulk_M < 0:
            raise ValueError("a_Fe3_bulk_M must be non-negative")


def _aval(key: str) -> float:
    """Anchored screening value, read at call time (live registry)."""
    return float(get_anchor(key).value)


def _arrhenius(T_K: float, Ea_J_mol: float) -> float:
    """Arrhenius multiplier relative to T_REF (exp(−Ea/R·(1/T − 1/T_ref)))."""
    return math.exp(-Ea_J_mol / R_GAS * (1.0 / T_K - 1.0 / T_REF))


def _live_bath_fe3_M(pH: float, T_C: float) -> float:
    """Bath [Fe³⁺] entering idle, live from the shuttle steady state.

    The bath leaves plating with its operating ferric level; the sealed
    divided-cell scenario is the screening picture (no crossover fault).
    Falls back to the ``IDLE_BATH_FE3_M`` anchor if the shuttle stack is
    unavailable.
    """
    try:
        from .fe3_shuttle import ShuttleParams, sealed_divided_cell, steady_state

        ss = steady_state(
            ShuttleParams(pH=pH, temperature_C=T_C), sealed_divided_cell()
        )
        return float(ss["fe3_ss_M"])
    except Exception:  # pragma: no cover - defensive fallback
        return _aval("IDLE_BATH_FE3_M")


@dataclass(frozen=True)
class _Resolved:
    """Concrete numbers behind an IdleBathState (anchors + live sources)."""

    pH: float
    T_C: float
    T_K: float
    a_H: float
    a_Fe3_bulk_M: float
    fe3_provenance: str
    o2_fraction_of_sat: float
    theta_additive: float
    mixing: str
    area_m2: float
    delta_m: float
    j_corr_ref_A_m2: float
    b_a_mV_dec: float
    b_c_mV_dec: float


def _resolve(state: IdleBathState) -> _Resolved:
    a_Fe3, prov = (
        (state.a_Fe3_bulk_M, "caller-specified")
        if state.a_Fe3_bulk_M is not None
        else (_live_bath_fe3_M(state.pH, state.T_C),
              "live: fe3_shuttle.steady_state (sealed divided cell)")
    )
    j_ref = (state.j_corr_ref_uA_cm2
             if state.j_corr_ref_uA_cm2 is not None
             else _aval("FE_ACID_JCORR_REF_UA_CM2"))
    return _Resolved(
        pH=state.pH,
        T_C=state.T_C,
        T_K=state.T_C + 273.15,
        a_H=10.0 ** (-state.pH),
        a_Fe3_bulk_M=max(a_Fe3, 0.0),
        fe3_provenance=prov,
        o2_fraction_of_sat=(state.o2_fraction_of_sat
                            if state.o2_fraction_of_sat is not None
                            else _aval("O2_FRACTION_SAT_IDLE")),
        theta_additive=(state.theta_additive
                        if state.theta_additive is not None
                        else _aval("ADDITIVE_BLOCKING_COVERAGE")),
        mixing=state.mixing,
        area_m2=state.area_m2,
        delta_m=(state.delta_m if state.delta_m is not None
                 else _aval("DIFFUSION_LAYER_IDLE_M")),
        j_corr_ref_A_m2=j_ref * 1e-2,   # µA/cm² → A/m²
        b_a_mV_dec=_aval("FE_ACID_ANODIC_TAFEL_MV_DEC"),
        b_c_mV_dec=_aval("FE_ACID_HER_TAFEL_MV_DEC"),
    )


# ────────────────────────────────────────────────────────────────────────
#  Channel 1: mixed-potential acid corrosion (H⁺ kinetics + O₂ transport)
# ────────────────────────────────────────────────────────────────────────

def _fe_anodic_A_m2(E_mV: float, r: _Resolved) -> float:
    k_t = _arrhenius(r.T_K, _aval("FE_CORR_EA_KJ_MOL") * 1000.0)
    return (r.j_corr_ref_A_m2 * (1.0 - r.theta_additive) * k_t
            * 10.0 ** (E_mV / r.b_a_mV_dec))


def _her_current_A_m2(E_mV: float, r: _Resolved) -> float:
    k_t = _arrhenius(r.T_K, _aval("FE_CORR_EA_KJ_MOL") * 1000.0)
    a_H_ref = 10.0 ** (-PH_REF)
    return (r.j_corr_ref_A_m2 * (1.0 - r.theta_additive) * (r.a_H / a_H_ref)
            * k_t * 10.0 ** (-E_mV / r.b_c_mV_dec))


def _o2_limiting_current_A_m2(r: _Resolved) -> float:
    """Transport-limited O₂ reduction current through the quasi-steady layer."""
    c_o2 = (dissolved_o2_saturation_mol_L(r.T_C)
            * r.o2_fraction_of_sat * 1000.0)            # mol/m³
    k_t = _arrhenius(r.T_K, _aval("DIFFUSION_EA_KJ_MOL") * 1000.0)
    d_o2 = _aval("O2_DIFFUSIVITY_25C_M2_S") * k_t
    return 4.0 * FARADAY * d_o2 / r.delta_m * c_o2


def corrosion_current(
    pH: Optional[float] = None,
    T_C: Optional[float] = None,
    o2_fraction_of_sat: Optional[float] = None,
    theta_additive: Optional[float] = None,
    delta_m: Optional[float] = None,
    j_corr_ref_uA_cm2: Optional[float] = None,
) -> Dict[str, Any]:
    """Open-circuit mixed-potential corrosion pair (E_corr, j_corr).

    The solver is *self-pinned*: with all inputs at the anchored reference
    state (pH 2.0, 25 °C, deaerated, clean Fe) it returns exactly
    ``FE_ACID_JCORR_REF_UA_CM2`` by construction, so the anchored anchor
    is sovereign and the Tafel structure moves off it.

    Fe³⁺ is deliberately **not** an argument here: ferric attack at open
    circuit is the homogeneous/mixed-potential *etch* channel
    (:func:`ferric_etch_flux`), and counting it in both places would
    double-book the same iron (see module docstring; differs from the V6
    sketch signature for that reason).
    """
    state = IdleBathState(
        pH=PH_REF if pH is None else pH,
        T_C=25.0 if T_C is None else T_C,
        o2_fraction_of_sat=o2_fraction_of_sat,
        theta_additive=theta_additive,
        delta_m=delta_m,
        j_corr_ref_uA_cm2=j_corr_ref_uA_cm2,
    )
    r = _resolve(state)
    i_o2 = _o2_limiting_current_A_m2(r)

    def f(E_mV: float) -> float:
        return _fe_anodic_A_m2(E_mV, r) - _her_current_A_m2(E_mV, r) - i_o2

    lo, hi = -500.0, 500.0       # mV relative to the reference E_corr (=0)
    if f(lo) > 0.0 or f(hi) < 0.0:  # pragma: no cover - bisection guard
        raise ArithmeticError("mixed-potential bracket failed to straddle zero")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-9:
            break
    e_corr = 0.5 * (lo + hi)
    j_corr = _fe_anodic_A_m2(e_corr, r)
    j_her = _her_current_A_m2(e_corr, r)
    return {
        "e_corr_vs_ref_mV": e_corr,
        "j_corr_uA_cm2": j_corr * 100.0,          # A/m² → µA/cm²
        "j_her_uA_cm2": j_her * 100.0,
        "j_o2_lim_uA_cm2": i_o2 * 100.0,
        "fe_dissolution_mol_m2_s": j_corr / (2.0 * FARADAY),
        "h2_evolution_mol_m2_s": j_her / (2.0 * FARADAY),
        "inputs": {
            "pH": r.pH,
            "T_C": r.T_C,
            "o2_fraction_of_sat": r.o2_fraction_of_sat,
            "theta_additive": r.theta_additive,
            "delta_m": r.delta_m,
            "j_corr_ref_uA_cm2": r.j_corr_ref_A_m2 * 100.0,
        },
        "flag": SCREENING_FLAG,
    }


# ────────────────────────────────────────────────────────────────────────
#  Channel 2: ferric etch  (2Fe³⁺ + Fe → 3Fe²⁺, half-order kinetic law)
# ────────────────────────────────────────────────────────────────────────

def ferric_etch_flux(
    a_Fe3_surf_M: float,
    pH: Optional[float] = None,
    T_C: Optional[float] = None,
) -> float:
    """Fe metal consumed by ferric etch, mol/(m²·s), at the kinetic limit.

    ``K_etch·√(a_Fe3/a_ref)·(a_H/a_H,ref)^m·Arr(T)`` with the anchored
    reference state ``a_ref = 0.05 M, pH 2.0, 25 °C``.  The constant is a
    SPECULATIVE screening anchor (USBM RI-series / FeCl₃-etch practice) —
    an order-of-magnitude picture, not a fitted rate.
    """
    if a_Fe3_surf_M <= 0.0:
        return 0.0
    ph = PH_REF if pH is None else pH
    t_c = 25.0 if T_C is None else T_C
    k = _aval("FE3_ETCH_K_REF_MOL_M2_S")
    a_ref = _aval("FE3_ETCH_REF_M")
    m_acid = _aval("FE3_ETCH_H_ORDER")
    k_t = _arrhenius(t_c + 273.15, _aval("FE3_ETCH_EA_KJ_MOL") * 1000.0)
    acid_ratio = (10.0 ** (-ph)) / (10.0 ** (-PH_REF))
    return float(k * math.sqrt(a_Fe3_surf_M / a_ref)
                 * acid_ratio ** m_acid * k_t)


# ────────────────────────────────────────────────────────────────────────
#  Integration: mass lost over one idle interval
# ────────────────────────────────────────────────────────────────────────

def _um_from_mol_m2(fe_mol_m2: float) -> float:
    return fe_mol_m2 * M_FE / RHO_FE * 1.0e6


def mass_loss_over_idle(
    t_idle_s: float,
    state: Optional[IdleBathState] = None,
    deposit_thickness_um: Optional[float] = None,
    n_steps: int = 256,
) -> Dict[str, Any]:
    """Fe dissolved from the deposit during one unpowered interval.

    Integrates the (constant) mixed-potential corrosion flux and the
    ferric-etch flux over ``t_idle_s``.  With ``mixing="stagnant"`` the
    etch is supply-limited through the growing layer δ(t)=√(2·D_Fe3·t)
    and the instantaneous rate is the harmonic combination of the kinetic
    law and the diffusive supply (kinetic at short times, √t diffusion at
    long times); with ``mixing="stirred"`` the etch runs at its kinetic
    rate against the bulk activity.

    ``deposit_thickness_um`` caps the loss at the metal actually present
    (what is left on the drum/strip when the rectifier opens).  Returns a
    ledger-shaped dict: mol, g, µm, channel split, and the plating-charge
    equivalent (2F per mol) of the lost iron.
    """
    if t_idle_s <= 0:
        raise ValueError("t_idle_s must be positive")
    if n_steps < 8:
        raise ValueError("n_steps must be >= 8")
    st = state or IdleBathState()
    r = _resolve(st)
    corr = corrosion_current(
        pH=r.pH, T_C=r.T_C,
        o2_fraction_of_sat=r.o2_fraction_of_sat,
        theta_additive=r.theta_additive,
        delta_m=r.delta_m,
        j_corr_ref_uA_cm2=r.j_corr_ref_A_m2 * 100.0,
    )
    flux_acid = corr["fe_dissolution_mol_m2_s"]           # mol/m²/s, constant
    j_etch_kin = ferric_etch_flux(r.a_Fe3_bulk_M, pH=r.pH, T_C=r.T_C)
    d_fe3 = (D_FE3_REF_M2_S
             * _arrhenius(r.T_K, _aval("DIFFUSION_EA_KJ_MOL") * 1000.0))
    c_bulk_mol_m3 = r.a_Fe3_bulk_M * 1000.0

    def etch_flux(t: float) -> float:
        """Instantaneous ferric-etch metal flux at time t (mol/m²/s)."""
        if j_etch_kin <= 0.0:
            return 0.0
        if r.mixing == "stirred":
            return j_etch_kin
        # stagnant: harmonic combination of kinetic demand and the
        # diffusive Fe³⁺ supply through δ(t); 2 Fe³⁺ per Fe metal.
        delta_t = math.sqrt(2.0 * d_fe3 * t) + 1.0e-9
        supply_metal = d_fe3 * c_bulk_mol_m3 / delta_t / 2.0
        return 1.0 / (1.0 / j_etch_kin + 1.0 / supply_metal)

    cap_mol_m2 = None
    if deposit_thickness_um is not None:
        cap_mol_m2 = deposit_thickness_um * 1.0e-6 * RHO_FE / M_FE

    dt = t_idle_s / n_steps
    mol_acid = 0.0
    mol_etch = 0.0
    capped = False
    t_consumed_s = t_idle_s
    for k in range(n_steps):
        t_mid = (k + 0.5) * dt
        d_acid = flux_acid * dt
        d_etch = etch_flux(t_mid) * dt
        if cap_mol_m2 is not None and (mol_acid + mol_etch
                                       + d_acid + d_etch) > cap_mol_m2:
            remaining = cap_mol_m2 - mol_acid - mol_etch
            frac = remaining / (d_acid + d_etch) if (d_acid + d_etch) > 0 else 0.0
            mol_acid += d_acid * frac
            mol_etch += d_etch * frac
            capped = True
            t_consumed_s = (k + frac) * dt
            break
        mol_acid += d_acid
        mol_etch += d_etch

    total_mol_m2 = mol_acid + mol_etch
    total_mol = total_mol_m2 * r.area_m2
    return {
        "t_idle_h": t_idle_s / 3600.0,
        "area_m2": r.area_m2,
        "mixing": r.mixing,
        "fe_dissolved_mol_m2": total_mol_m2,
        "fe_mol": total_mol,
        "fe_g": total_mol * M_FE * 1000.0,
        "um_lost": _um_from_mol_m2(total_mol_m2),
        "um_from_acid": _um_from_mol_m2(mol_acid),
        "um_from_etch": _um_from_mol_m2(mol_etch),
        "h2_from_corrosion_mol": (corr["h2_evolution_mol_m2_s"]
                                  * t_idle_s * r.area_m2),
        "charge_equivalent_C": total_mol * 2.0 * FARADAY,
        "j_corr_uA_cm2": corr["j_corr_uA_cm2"],
        "etch_flux_kinetic_mol_m2_s": j_etch_kin,
        "a_Fe3_bulk_M": r.a_Fe3_bulk_M,
        "fe3_provenance": r.fe3_provenance,
        "o2_fraction_of_sat": r.o2_fraction_of_sat,
        "theta_additive": r.theta_additive,
        "capped_by_thickness": capped,
        "t_to_consume_h": t_consumed_s / 3600.0 if capped else None,
        "deposit_thickness_um": deposit_thickness_um,
        "inputs": {"pH": r.pH, "T_C": r.T_C},
        "flag": SCREENING_FLAG,
    }


# ────────────────────────────────────────────────────────────────────────
#  Verdict: is this idle event negligible, a ledger term, or material?
# ────────────────────────────────────────────────────────────────────────

@dataclass
class IdleVerdict:
    """Screening verdict for one idle event."""

    t_idle_h: float
    loss: Dict[str, Any]
    verdict: str                       # negligible | ledger_term | material
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t_idle_h": self.t_idle_h,
            "um_lost": self.loss["um_lost"],
            "fe_g": self.loss["fe_g"],
            "charge_equivalent_C": self.loss["charge_equivalent_C"],
            "verdict": self.verdict,
            "reasons": list(self.reasons),
            "loss": self.loss,
            "flag": SCREENING_FLAG,
        }


def evaluate_idle(
    t_idle_h: float = 8.0,
    state: Optional[IdleBathState] = None,
    deposit_thickness_um: Optional[float] = None,
) -> IdleVerdict:
    """Screening verdict for one idle/shutdown interval (default: a night)."""
    loss = mass_loss_over_idle(
        t_idle_h * 3600.0, state=state,
        deposit_thickness_um=deposit_thickness_um,
    )
    um = loss["um_lost"]
    reasons: List[str] = []
    if um < LOSS_NEGLIGIBLE_UM:
        verdict = "negligible"
        reasons.append(
            f"{um:.3f} µm below the {LOSS_NEGLIGIBLE_UM} µm floor — "
            "invisible to a 4-decimal-gram scale after harvest")
    elif um <= LOSS_MATERIAL_UM:
        verdict = "ledger_term"
        reasons.append(
            f"{um:.3f} µm in the {LOSS_NEGLIGIBLE_UM}–{LOSS_MATERIAL_UM} µm "
            "band — book as a predicted term in the iron ledger (run_record)")
    else:
        verdict = "material"
        reasons.append(
            f"{um:.3f} µm above the {LOSS_MATERIAL_UM} µm ceiling — "
            "review shutdown practice (rinse-and-drain or cooler idle)")
    acid, etch = loss["um_from_acid"], loss["um_from_etch"]
    if acid + etch > 0:
        reasons.append(
            f"channel split {100*acid/(acid+etch):.0f}% acid / "
            f"{100*etch/(acid+etch):.0f}% ferric etch")
    if loss["capped_by_thickness"]:
        reasons.append(
            f"plated thickness consumed after {loss['t_to_consume_h']:.1f} h — "
            "longer idles lose no more metal from this surface")
    if loss["h2_from_corrosion_mol"] > 1.0e-3:
        reasons.append(
            f"corrosion H₂ {loss['h2_from_corrosion_mol']:.3f} mol over the "
            "interval — the shutdown safety case covers venting, this is the "
            "inventory side of the same event")
    return IdleVerdict(t_idle_h=t_idle_h, loss=loss, verdict=verdict,
                       reasons=reasons)


# ────────────────────────────────────────────────────────────────────────
#  Wiring: run_record manifests (predicted, not measured, ledger terms)
# ────────────────────────────────────────────────────────────────────────

def predicted_idle_terms(manifest: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build the run_record predicted ledger terms from a run manifest.

    Reads the optional declared block documented in docs/DATA_CONTRACT.md::

        manifest["setup"]["idle"] = {
            "hours": 8.0,                 # required; unpowered soak time
            "pH": 2.0, "T_C": 40.0,       # optional overrides
            "a_fe3_M": 1.0e-4,            # optional (else live fe3_shuttle)
            "o2_fraction_of_sat": 0.2,
            "theta_additive": 0.5,
            "mixing": "stagnant",
            "deposit_thickness_um": 50.0,
        }

    plus ``manifest["setup"]["cathode"]["area_cm2"]`` for the area.  Returns
    ``None`` when no positive idle time is declared — runs without a
    declared idle carry no predicted term (advisory, never a QA error).
    """
    if not isinstance(manifest, dict):
        return None
    setup = manifest.get("setup")
    if not isinstance(setup, dict):
        return None
    idle = setup.get("idle")
    if not isinstance(idle, dict) or "hours" not in idle:
        return None
    try:
        hours = float(idle["hours"])
    except (TypeError, ValueError):
        return None
    if hours <= 0:
        return None
    cathode = setup.get("cathode")
    area_cm2 = None
    if isinstance(cathode, dict):
        try:
            area_cm2 = float(cathode["area_cm2"])
        except (KeyError, TypeError, ValueError):
            area_cm2 = None
    state = IdleBathState(
        pH=float(idle.get("pH", PH_REF)),
        T_C=float(idle.get("T_C", 40.0)),
        a_Fe3_bulk_M=(None if idle.get("a_fe3_M") is None
                      else float(idle["a_fe3_M"])),
        o2_fraction_of_sat=(None if idle.get("o2_fraction_of_sat") is None
                            else float(idle["o2_fraction_of_sat"])),
        theta_additive=(None if idle.get("theta_additive") is None
                        else float(idle["theta_additive"])),
        mixing=str(idle.get("mixing", "stagnant")),
        area_m2=(area_cm2 * 1.0e-4) if area_cm2 else 1.0,
    )
    th = idle.get("deposit_thickness_um")
    loss = mass_loss_over_idle(
        hours * 3600.0, state=state,
        deposit_thickness_um=None if th is None else float(th),
    )
    notes = [
        "predicted term (models/deposit_corrosion.py, L1) — not measured",
        "idle redissolution moves Fe deposit→bath: it biases gravimetric FE "
        "low and explains unresolved charge, but cannot open the mol-scale "
        "iron closure",
    ]
    if not area_cm2:
        notes.append("no declared cathode area — 1 m² screening default")
    return {
        "fe_mol": loss["fe_mol"],
        "charge_C": loss["charge_equivalent_C"],
        "um_lost": loss["um_lost"],
        "fe_g": loss["fe_g"],
        "declared_idle_hours": hours,
        "loss": loss,
        "screening_flag": SCREENING_FLAG,
        "assumptions": notes,
    }


# ────────────────────────────────────────────────────────────────────────
#  Scope declaration + CLI
# ────────────────────────────────────────────────────────────────────────

def model_scope() -> Dict[str, Any]:
    return {
        "screening_flag": SCREENING_FLAG,
        "live_derivations": [
            "fe3_shuttle.steady_state (bath Fe³⁺ entering idle; "
            "sealed divided-cell scenario, pH-following)",
            "bath_startup.dissolved_o2_saturation_mol_L (Weiss 1970 O₂)",
            "fe3_shuttle.D_FE3_REF_M2_S (Fe³⁺ diffusivity screening family)",
        ],
        "exact": [
            "mixed-potential solve (bisection on Tafel + limiting currents)",
            "Fe + 2H⁺ / 2Fe³⁺ + Fe stoichiometry and 2F charge equivalents",
            "self-pinning: anchored reference state reproduced exactly",
        ],
        "screening_proxies_anchored": [
            "reference j_corr and Tafel slopes (Stern/Kelly family)",
            "ferric-etch rate constant — SPECULATIVE half-order law "
            "(USBM RI-series / FeCl₃-etch practice)",
            "additive blocking coverage θ (pickling-inhibitor practice)",
            "stagnant growing-layer picture δ(t)=√(2·D·t)",
        ],
        "out_of_scope": [
            "powered dissolution anodic of E_eq (kinetics.py BV branch)",
            "electrochemical Fe³⁺ shuttle at a flowing cathode (fe3_shuttle.py)",
            "shutdown safety/venting (docs/INDEPENDENT_SHUTDOWN.md)",
            "depletion of bath Fe³⁺ inventory by the etch itself "
            "(flat-reservoir assumption; giant-V cells revisit)",
        ],
    }


def _fmt(x: float, sig: int = 3) -> str:
    return f"{x:.{sig}g}"


def main() -> None:  # pragma: no cover - CLI wrapper
    print(f"deposit_corrosion — idle corrosion & ferric etch  [{SCREENING_FLAG}]")
    print()
    print("j_corr (µA/cm²) — mixed-potential solve, θ & O₂ at anchored defaults")
    header = "pH \\ T_C   " + "  ".join(f"{t:>8d}" for t in (25, 40, 60))
    print(header)
    print("-" * len(header))
    for pH in (2.0, 2.5, 3.0):
        row = "  ".join(
            f"{corrosion_current(pH=pH, T_C=t)['j_corr_uA_cm2']:>8.2f}"
            for t in (25, 40, 60)
        )
        print(f"{pH:>7.1f}   {row}")
    print()
    deaerated = corrosion_current(o2_fraction_of_sat=0.0, theta_additive=0.0)
    print(f"deaerated, clean Fe, 25 °C, pH 2: j_corr = "
          f"{deaerated['j_corr_uA_cm2']:.2f} µA/cm² "
          f"(anchored {_aval('FE_ACID_JCORR_REF_UA_CM2'):g})")
    st = IdleBathState()
    print(f"live bath Fe³⁺ (fe3_shuttle, sealed cell, pH {st.pH}): "
          f"{_live_bath_fe3_M(st.pH, st.T_C):.2e} M")
    print()
    for label, hours in (("night (8 h)", 8.0), ("weekend (64 h)", 64.0)):
        v = evaluate_idle(t_idle_h=hours)
        loss = v.loss
        print(f"{label}: {loss['um_lost']:.3f} µm  "
              f"({loss['um_from_acid']:.3f} acid + {loss['um_from_etch']:.3f} etch)"
              f"  = {loss['fe_g']:.2f} g/m²  "
              f"| charge eq. {loss['charge_equivalent_C']/max(loss['area_m2'],1e-12):.0f} C/m²"
              f"  | verdict: {v.verdict}")
    print()
    try:
        from .closed_loop import campaign_idle_accounting

        camp = campaign_idle_accounting()
        print(f"campaign view (closed_loop defaults, 30 days × 16 h-on): "
              f"idle loss {camp['idle_fe_g']:.1f} g Fe = "
              f"{camp['idle_loss_pct_of_production']:.3f}% of production, "
              f"apparent FE bias {camp['apparent_fe_bias_pp']:.3f} pp")
    except Exception:  # pragma: no cover - advisory demo
        pass
    print()
    print("Wire-in: declare manifest setup.idle.hours → run_record ledgers gain "
          "predicted terms; campaign view via closed_loop.campaign_idle_accounting.")


if __name__ == "__main__":  # pragma: no cover
    main()
