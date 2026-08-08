"""Product oxidation, drying & pyrophoricity — the post-harvest oxygen budget (V6 §1.2).

Why this module exists
----------------------
``oxygen_in_iron.py`` and ``bubble_engulfment.py`` bound the oxygen the
deposit is *born* with; nothing bounded the oxygen it **gains between the
scraper and the furnace**.  For column powder from the rotating cylinder —
the architecture screen's only 5×-clearing option is *powder only* — this is
not a slow aging term: high-area, hydrogen-bearing, freshly washed iron
passivates on air (Mott–Cabrera film, ~2–4 nm at RT), and if dried hot and
fast it self-heats, because exothermic oxidation scales with **surface
area** while heat loss scales with **bulk** — the Semenov thermal-runaway
criterion.  Pyrophoric iron-powder events are routine in the PM industry.

This module computes (a) the passivation-film oxygen the product carries
into the melt balance, (b) whether the product is pyrophoric-prone under a
hot-dry/storage fault (Semenov), and (c) the **passivation protocol that
becomes the product spec** — "flake, passivated, ≤1.5 wt% O, non-pyrophoric"
is a buyable article; wet reactive powder is not.

Physics / chemistry
-------------------
.. code-block:: text

    specific surface (live geometry: powder spheres × roughness; flake/foil
    two-face):
        powder:  S_A = 6/(ρ_Fe·d50)·roughness        m²/kg
        flake:   S_A = 2/(ρ_Fe·t_foil)               m²/kg

    film growth (two anchored branches, max()ed — the log→parabolic
    crossover is explicit):
        fresh-surface log branch:  x_log(t) = x_lim·(1 − e^(−t/τ))    (RT)
        established-film parabolic hot branch:
            x_par(t) = √(x₀² + 2κ(T,p_O2)·t),
            κ = w_ox(T, x_ref, p_O2)·x_ref / (mol O per nm per m²)
        w_ox = W_REF · (p_O2/0.21) · exp(−Ea/R·(1/T − 1/T_ref)) · (x_ref/x)

    self-heating (Semenov), per kg of bed:
        q_gen(T) = w_ox·Q_ox·S_A                    [W/kg, Arrhenius]
        q_loss(T) = h·(A/V)·(T − T_amb)/ρ_bulk      [W/kg, linear]
        runaway ⇔ no stable intersection (gen > loss everywhere)
        T_crit = max T_amb admitting a stable fixed point     (bisection)

    passivation protocol (the product spec):
        controlled p_O2 / warm T / residence → grow x_lim, then RT storage
        → product O(wt%) and a pass/fail spec + pyrophoricity class.

Live derivations
----------------
* product geometry maps from ``cell_architecture`` ids (rotating_cylinder
  → powder, drum_and_strip → flake/foil) with anchored size defaults —
  the V6 "feed from cell_architecture.py (product form/area)".
* foil/flake thickness reuses the ``PRODUCT_FOIL_THICKNESS_UM`` anchor
  (single geometric source of truth shared with rinse_carryover).
* output feeds ``melt_balance`` (§1.5): ``postharvest_o_pickup_wt_pct``
  replaces the ``POSTHARVEST_O_PICKUP_WT_PCT`` anchor add-on live.

Screening flag
--------------
L1.  The parabolic rate prefactor (``OX_RATE_REF_MOL_M2_S``) is
SPECULATIVE (decade band); film limits, timescales, heats and device
geometry are anchored screening proxies.  Stoichiometry, the Semenov
balance structure and the area bookkeeping are exact.  Nothing here is
gate evidence; the numbers order the safety cases and the melt O pickup,
they do not certify either.

References
----------
* docs/CHEM_PHYS_IMPROVEMENTS_V6.md §1.2 (gap statement).
* Mott–Cabrera native-film theory (2–4 nm RT limiting film on iron).
* Semenov (1928) thermal-explosion criterion.
* NFPA / PM-industry practice for iron-powder drying, passivation and
  combustible-dust classification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .anchors import get_anchor
from .electrochemistry import R_GAS, RHO_FE

SCREENING_FLAG = "unvalidated (L1)"

T_RT_K = 298.15
CP_FE_J_KG_K = 450.0           # Fe heat capacity, screening constant
M_O_KG_MOL = 0.016             # per O atom

PRODUCT_FORMS = ("powder", "flake", "foil", "briquette")
ARCH_TO_FORM = {
    "rotating_cylinder": "powder",
    "drum_and_strip": "flake",
    "plate_and_frame": "foil",
}


def _aval(key: str) -> float:
    return float(get_anchor(key).value)


def _ox_rate_ref() -> Dict[str, float]:
    return {
        "w_ref": _aval("OX_RATE_REF_MOL_M2_S"),
        "ea_J": _aval("OX_EA_KJ_MOL") * 1000.0,
        "t_ref_K": 60.0 + 273.15,
        "film_ref_nm": _aval("PASSIV_FILM_LIM_NM"),
    }


def mol_o_per_nm_m2() -> float:
    """Mol of O atoms per nm of passive film per m² (oxide-stoichiometric)."""
    return (_aval("OXIDE_DENSITY_KG_M3") * 1.0e-9
            * _aval("OXIDE_O_MASS_FRAC") / M_O_KG_MOL)


# ────────────────────────────────────────────────────────────────────────
#  Product state & specific surface
# ────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProductState:
    """The harvested product whose post-scraper oxygen we track.

    ``kind="briquette"`` keeps the *powder* geometry: passivation happens
    before densification (process order), so the carried area is the
    powder's at passivation time; post-densification external exposure is
    second-order and noted, not modelled.
    """

    kind: str = "powder"
    architecture_id: Optional[str] = None
    d50_um: Optional[float] = None
    foil_thickness_um: Optional[float] = None
    roughness: Optional[float] = None

    def __post_init__(self) -> None:
        if self.kind not in PRODUCT_FORMS:
            raise ValueError(
                f"kind must be one of {PRODUCT_FORMS}, got {self.kind!r}")


def from_architecture(architecture_id: str) -> ProductState:
    """Map a cell_architecture id to its product form (the V6 §1.2 feed)."""
    if architecture_id not in ARCH_TO_FORM:
        raise ValueError(f"no product form mapped for {architecture_id!r}")
    kind = ARCH_TO_FORM[architecture_id]
    return ProductState(kind=kind, architecture_id=architecture_id)


def column_powder() -> ProductState:
    return ProductState("powder", architecture_id="rotating_cylinder")


def drum_flake() -> ProductState:
    return ProductState("flake", architecture_id="drum_and_strip")


def passivated_briquette() -> ProductState:
    return ProductState("briquette", architecture_id="rotating_cylinder")


def specific_surface_area_m2_kg(state: ProductState) -> Dict[str, Any]:
    """Specific surface area (m²/kg) with provenance of the geometry."""
    if state.kind in ("powder", "briquette"):
        d50 = (state.d50_um if state.d50_um is not None
               else _aval("POWDER_D50_UM"))
        rough = (state.roughness if state.roughness is not None
                 else _aval("POWDER_ROUGHNESS_FACTOR"))
        area = 6.0 / (RHO_FE * d50 * 1.0e-6) * rough
        prov = ("powder spheres 6/(ρ·d50)×roughness"
                + (" (briquette: powder area at passivation time, "
                   "pre-densification)" if state.kind == "briquette" else ""))
        return {"s_area_m2_kg": area, "d50_um": d50, "roughness": rough,
                "provenance": prov}
    t_um = (state.foil_thickness_um if state.foil_thickness_um is not None
            else _aval("PRODUCT_FOIL_THICKNESS_UM"))
    area = 2.0 / (RHO_FE * t_um * 1.0e-6)
    return {"s_area_m2_kg": area, "foil_thickness_um": t_um,
            "provenance": f"{state.kind} two-face 2/(ρ·t)"}


# ────────────────────────────────────────────────────────────────────────
#  Oxidation kinetics: log branch + parabolic hot branch
# ────────────────────────────────────────────────────────────────────────

def oxidation_rate_mol_m2_s(T_C: float, film_nm: float,
                            pO2_frac: float = 0.21) -> float:
    """O consumed per m² per s at (T, film, p_O2) — inverse-film parabolic."""
    ref = _ox_rate_ref()
    if film_nm <= 0.0:
        raise ValueError("film_nm must be positive (x=0 is the fresh-surface "
                         "log branch, not this law)")
    return (ref["w_ref"] * (pO2_frac / 0.21)
            * math.exp(-ref["ea_J"] / R_GAS
                       * (1.0 / (T_C + 273.15) - 1.0 / ref["t_ref_K"]))
            * (ref["film_ref_nm"] / film_nm))


def passive_film_after_air_nm(t_air_s: float, T_C: float = 25.0,
                              pO2_frac: float = 0.21,
                              seed_film_nm: float = 1.0) -> float:
    """Film thickness after air exposure: max of the two anchored branches.

    RT behavior is the Mott–Cabrera log law toward x_lim; hot behavior is
    the parabolic branch √–law from the seed film.  The max() IS the
    log→parabolic crossover; both branches carry decade-band anchors.
    """
    if t_air_s < 0 or seed_film_nm <= 0:
        raise ValueError("t_air_s >= 0 and seed_film_nm > 0 required")
    x_lim = _aval("PASSIV_FILM_LIM_NM")
    tau = _aval("PASSIV_TAU_S")
    x_log = max(x_lim * (1.0 - math.exp(-t_air_s / tau)), seed_film_nm)
    # parabolic hot branch from the seed film:
    #   x² = x₀² + 2κt,  κ = w_ox(T, x_ref, p_O2)·x_ref/(mol O per nm per m²)
    ref = _ox_rate_ref()
    kappa = (oxidation_rate_mol_m2_s(T_C, ref["film_ref_nm"], pO2_frac)
             * ref["film_ref_nm"] / mol_o_per_nm_m2())            # nm²/s
    x_par = math.sqrt(seed_film_nm ** 2 + 2.0 * kappa * t_air_s)
    return max(x_log, x_par)


def o_wt_pct_from_film(film_gain_nm: float, s_area_m2_kg: float) -> float:
    """Charge oxygen (wt%) from a film increment over the product's area."""
    return (film_gain_nm * mol_o_per_nm_m2() * M_O_KG_MOL
            * s_area_m2_kg * 100.0)


# ────────────────────────────────────────────────────────────────────────
#  Semenov thermal-runaway balance
# ────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BedState:
    """The lumped drying/storage bed for the Semenov loss term."""

    tray_depth_m: Optional[float] = None     # → TRAY_BED_DEPTH_M (A/V = 1/depth)
    h_W_m2K: Optional[float] = None          # → DRYER_H_W_M2K
    bulk_rho_kg_m3: Optional[float] = None   # → POWDER_BULK_DENSITY_KG_M3

    def resolved(self) -> "BedState":
        return BedState(
            tray_depth_m=(self.tray_depth_m if self.tray_depth_m is not None
                          else _aval("TRAY_BED_DEPTH_M")),
            h_W_m2K=(self.h_W_m2K if self.h_W_m2K is not None
                     else _aval("DRYER_H_W_M2K")),
            bulk_rho_kg_m3=(self.bulk_rho_kg_m3
                            if self.bulk_rho_kg_m3 is not None
                            else _aval("POWDER_BULK_DENSITY_KG_M3")),
        )


def _heat_gen_W_kg(T_C: float, s_area: float, film_nm: float,
                   pO2_frac: float) -> float:
    return (oxidation_rate_mol_m2_s(T_C, film_nm, pO2_frac)
            * _aval("OX_HEAT_KJ_MOL_O") * 1000.0 * s_area)


def _heat_loss_coeff_W_kgK(bed: BedState) -> float:
    """h·(A/V)/ρ_bulk — the Semenov conductance per kg of bed."""
    return bed.h_W_m2K * (1.0 / bed.tray_depth_m) / bed.bulk_rho_kg_m3


def _stable(T_amb_C: float, s_area: float, film_nm: float, pO2: float,
            loss_coeff: float) -> bool:
    """True iff a stable fixed point exists above T_amb (gen < loss there).

    Scans candidate bed temperatures; a stable point exists if somewhere
    in (T_amb, T_amb + 400 K) the loss exceeds generation before the
    Arrhenius branch diverges past it.
    """
    t_hi = T_amb_C + 400.0
    n = 400
    for i in range(1, n + 1):
        t = T_amb_C + (t_hi - T_amb_C) * i / n
        gen = _heat_gen_W_kg(t, s_area, film_nm, pO2)
        loss = loss_coeff * (t - T_amb_C)
        if gen < loss:
            return True
    return False


def semenov_critical_T(state: ProductState,
                       bed: Optional[BedState] = None,
                       film_nm: Optional[float] = None,
                       pO2_frac: float = 0.21) -> Dict[str, Any]:
    """Critical ambient temperature (°C) for thermal runaway.

    Bisection on T_amb using the stability test; returned with the
    generation/loss point values at the default dryer temperature so the
    margin is auditable.
    """
    b = (bed or BedState()).resolved()
    film = film_nm if film_nm is not None else _aval("PASSIV_FILM_LIM_NM")
    area = specific_surface_area_m2_kg(state)["s_area_m2_kg"]
    coeff = _heat_loss_coeff_W_kgK(b)
    lo, hi = 20.0, 500.0
    if not _stable(lo, area, film, pO2_frac, coeff):
        t_crit = lo
    elif _stable(hi, area, film, pO2_frac, coeff):
        t_crit = hi
    else:
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if _stable(mid, area, film, pO2_frac, coeff):
                lo = mid
            else:
                hi = mid
        t_crit = 0.5 * (lo + hi)
    t_dry = _aval("DRYER_AIR_T_C")
    return {
        "t_crit_ambient_C": t_crit,
        "s_area_m2_kg": area,
        "film_nm": film,
        "pO2_frac": pO2_frac,
        "loss_coeff_W_kgK": coeff,
        "dryer_air_T_C": t_dry,
        "dryer_margin_K": t_crit - t_dry,
        "dryer_subcritical": t_dry < t_crit,
        "q_gen_at_dryer_W_kg": _heat_gen_W_kg(t_dry, area, film, pO2_frac),
        "q_loss_slope_W_kgK": coeff,
        "flag": SCREENING_FLAG,
    }


# ────────────────────────────────────────────────────────────────────────
#  Verdicts: pyrophoricity class, passivation protocol, product spec
# ────────────────────────────────────────────────────────────────────────

def pyrophoricity_class(state: ProductState) -> Dict[str, Any]:
    """NFPA/PM-style classification of the as-harvested (unpassivated) form."""
    d50 = (specific_surface_area_m2_kg(state).get("d50_um"))
    boundary = _aval("COMBUSTIBLE_DUST_D50_UM")
    if state.kind in ("powder", "briquette"):
        if d50 < boundary:
            cls, note = "combustible-dust / pyrophoricity candidate", (
                f"d50 {d50:.0f} µm below the {boundary:.0f} µm "
                "classification boundary — unpassivated + H-bearing "
                "product is inert-handled")
        else:
            cls, note = "coarse powder (manageable)", (
                f"d50 {d50:.0f} µm at/above the {boundary:.0f} µm boundary")
    else:
        cls, note = "web form (area-bound but dense)", (
            "flake/foil area is film-geometry bound; edge fresh-cracking "
            "is the local risk, bulk self-heating bounded by Semenov")
    return {"class": cls, "reason": note,
            "boundary_d50_um": boundary, "flag": SCREENING_FLAG}


@dataclass
class PassivationProtocol:
    """The passivation recipe that becomes the product spec (V6 §1.2b)."""

    pO2_frac: float
    T_C: float
    residence_hr: float
    storage_hr: float
    target_film_nm: float
    achieved_film_nm: float
    o_pickup_wt_pct: float
    spec_max_o_wt_pct: float
    spec_ok: bool
    semenov: Dict[str, Any]
    pyro_class: Dict[str, Any]
    verdict: str = ""
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["reasons"] = list(self.reasons)
        d["flag"] = SCREENING_FLAG
        return d


def design_passivation(state: Optional[ProductState] = None,
                       target_o_max_wt_pct: Optional[float] = None
                       ) -> PassivationProtocol:
    """Design the controlled passivation and grade the resulting article.

    The protocol (anchored defaults): controlled p_O2 blanket at a warm,
    sub-critical T for the residence needed to grow the limiting film,
    then RT storage.  The product spec is the V6 §1.2 article:
    passivated, ≤1.5 wt% O, non-pyrophoric.
    """
    st = state or passivated_briquette()
    spec_max = (target_o_max_wt_pct if target_o_max_wt_pct is not None
                else _aval("PRODUCT_PASSIV_O_MAX_WT_PCT"))
    pO2 = _aval("PASSIV_PO2_FRAC")
    t_c = _aval("PASSIV_PROTOCOL_T_C")
    x_lim = _aval("PASSIV_FILM_LIM_NM")
    tau_eff = _aval("PASSIV_TAU_S") * (0.21 / max(pO2, 0.005)) ** 0.5
    residence_hr = 5.0 * tau_eff / 3600.0     # ~5τ to the limiting film
    storage_hr = _aval("STORAGE_HOURS")
    area = specific_surface_area_m2_kg(st)["s_area_m2_kg"]

    # film during protocol (log branch, pO2-rescaled timescale)
    film_proto = x_lim * (1.0 - math.exp(-residence_hr * 3600.0 / tau_eff))
    film_proto = max(film_proto, 1.0)
    # storage at RT on air: parabolic increment from the protocol film
    film_final = passive_film_after_air_nm(
        storage_hr * 3600.0, T_C=25.0, pO2_frac=0.21,
        seed_film_nm=film_proto)
    o_pickup = o_wt_pct_from_film(film_final, area)
    # a passivated product should be far sub-critical at storage/dryer T
    sem = semenov_critical_T(st, film_nm=film_final)
    pyro = pyrophoricity_class(st)
    spec_ok = o_pickup <= spec_max

    reasons: List[str] = [
        f"protocol: {pO2*100:.1f}% O₂ at {t_c:.0f} °C for "
        f"{residence_hr:.1f} h (5τ at the anchored log timescale), then "
        f"{storage_hr:.0f} h RT storage",
        f"film {film_final:.1f} nm → O pickup {o_pickup:.3f} wt% "
        f"(spec ceiling {spec_max} wt%)",
        f"Semenov after passivation: T_crit {sem['t_crit_ambient_C']:.0f} °C, "
        f"dryer margin {sem['dryer_margin_K']:.0f} K",
    ]
    if spec_ok and sem["dryer_subcritical"]:
        verdict = "spec-qualified"
    elif sem["dryer_subcritical"]:
        verdict = "passivatable-but-over-spec"
    else:
        verdict = "pyrophoric-risk"
    if pyro["class"].startswith("combustible-dust"):
        reasons.append(pyro["reason"])
    if not sem["dryer_subcritical"]:
        reasons.append("hot-air drying at the default dryer T is "
                       "SUPER-critical — the V6 §1.2 fault case; passivate "
                       "first, dry warm-and-slow")
    return PassivationProtocol(
        pO2_frac=pO2, T_C=t_c, residence_hr=residence_hr,
        storage_hr=storage_hr, target_film_nm=x_lim,
        achieved_film_nm=film_final, o_pickup_wt_pct=o_pickup,
        spec_max_o_wt_pct=spec_max, spec_ok=spec_ok,
        semenov=sem, pyro_class=pyro, verdict=verdict, reasons=reasons,
    )


def postharvest_o_pickup_wt_pct(product_form: str = "briquette") -> float:
    """Live post-harvest O pickup for ``melt_balance`` (V6 §1.2 → §1.5).

    Replaces the ``POSTHARVEST_O_PICKUP_WT_PCT`` anchor add-on with the
    passivated-article physics: limiting-film oxygen on the product's
    specific surface after the anchored protocol and storage.
    """
    if product_form in ("foil", "flake"):
        st = drum_flake() if product_form == "flake" else ProductState("foil")
    else:
        st = passivated_briquette()
    return design_passivation(st).o_pickup_wt_pct


# ────────────────────────────────────────────────────────────────────────
#  Scope + CLI
# ────────────────────────────────────────────────────────────────────────

def model_scope() -> Dict[str, Any]:
    return {
        "screening_flag": SCREENING_FLAG,
        "live_derivations": [
            "cell_architecture id → product form mapping (powder/flake/foil)",
            "PRODUCT_FOIL_THICKNESS_UM shared with rinse_carryover",
        ],
        "live_consumers": [
            "melt_balance post-harvest O pickup "
            "(postharvest_o_pickup_wt_pct replaces "
            "POSTHARVEST_O_PICKUP_WT_PCT live)",
        ],
        "exact": [
            "specific-surface bookkeeping (spheres × roughness; two-face web)",
            "film nm → charge O wt% stoichiometry",
            "Semenov balance structure (Arrhenius generation vs linear loss)",
        ],
        "screening_proxies_anchored": [
            "parabolic rate prefactor — SPECULATIVE decade band",
            "log-film limit & timescale (Mott–Cabrera family)",
            "bed geometry / h / bulk density (Semenov loss channel)",
            "passivation window (pO2, T) and storage residence",
            "combustible-dust d50 boundary (NFPA/PM practice)",
        ],
        "out_of_scope": [
            "humidity adsorption / FeOOH hydrate chemistry",
            "hydrogen desorption coupling (melt_hydrogen owns H)",
            "full thermal PDE inside the bed (lumped Semenov only)",
            "dryer CAPEX line items (dark_mill.py consumes the protocol "
            "duty, no wiring yet)",
        ],
    }


def self_heat_transient(state: Optional[ProductState] = None,
                        bed: Optional[BedState] = None,
                        T_amb_C: Optional[float] = None,
                        film_nm: Optional[float] = None,
                        pO2_frac: float = 0.21,
                        t_hr: float = 2.0,
                        n_steps: int = 2000) -> Dict[str, Any]:
    """Bed-temperature trajectory at fixed film (RK4) — the ignition delay.

    C_p·dT/dt = q_gen(T) − q_loss(T − T_amb) per kg (V6's ODE, solved).
    Film growth inside the transient is frozen (quasi-steady-film
    screening assumption, documented).  Returns peak T and the time the
    bed crossed T_amb + 50 K, if it did (the runaway signature).
    """
    st = state or column_powder()
    b = (bed or BedState()).resolved()
    t_amb = T_amb_C if T_amb_C is not None else _aval("DRYER_AIR_T_C")
    film = film_nm if film_nm is not None else _aval("PASSIV_FILM_LIM_NM")
    area = specific_surface_area_m2_kg(st)["s_area_m2_kg"]
    coeff = _heat_loss_coeff_W_kgK(b)
    dt = t_hr * 3600.0 / n_steps
    t = t_amb
    t_peak = t_amb
    t_cross_s: Optional[float] = None
    # Crossing T_amb + 50 K in a lumped Semenov lump IS the runaway verdict;
    # integrating beyond is physically meaningless (other sinks appear), so
    # the integration stops at the crossing.
    cross_delta_K = 50.0

    def rhs(temp: float) -> float:
        return (_heat_gen_W_kg(temp, area, film, pO2_frac)
                - coeff * (temp - t_amb)) / CP_FE_J_KG_K

    for i in range(n_steps):
        k1 = rhs(t)
        k2 = rhs(t + 0.5 * dt * k1)
        k3 = rhs(t + 0.5 * dt * k2)
        k4 = rhs(t + dt * k3)
        t += dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
        t_peak = max(t_peak, t)
        if t > t_amb + cross_delta_K:
            t_cross_s = (i + 1) * dt
            t_peak = t_amb + cross_delta_K
            break
    return {
        "t_amb_C": t_amb, "t_peak_C": t_peak,
        "runaway": t_cross_s is not None,
        "runaway_after_s": t_cross_s,
        "equilibrium_C": t_peak if t_cross_s is None else None,
        "s_area_m2_kg": area, "film_nm": film,
        "q_gen0_W_kg": _heat_gen_W_kg(t_amb, area, film, pO2_frac),
        "flag": SCREENING_FLAG,
    }


def main() -> None:  # pragma: no cover - CLI wrapper
    print(f"product_oxidation — post-harvest O budget & Semenov  [{SCREENING_FLAG}]")
    print()
    for st, label in ((column_powder(), "column powder (rotating_cylinder)"),
                      (drum_flake(), "drum flake (drum_and_strip)")):
        area = specific_surface_area_m2_kg(st)
        sem = semenov_critical_T(st)
        pyro = pyrophoricity_class(st)
        print(f"{label}: S_A {area['s_area_m2_kg']:.1f} m²/kg "
              f"({area['provenance']})")
        print(f"   pyrophoricity class: {pyro['class']} — {pyro['reason']}")
        print(f"   Semenov (3 nm film, air): T_crit "
              f"{sem['t_crit_ambient_C']:.0f} °C, dryer "
              f"{sem['dryer_air_T_C']:.0f} °C margin "
              f"{sem['dryer_margin_K']:+.0f} K → "
              f"{'sub-critical' if sem['dryer_subcritical'] else 'SUPER-CRITICAL'}")
        proto = design_passivation(st)
        print(f"   passivation spec: film {proto.achieved_film_nm:.1f} nm, "
              f"O pickup {proto.o_pickup_wt_pct:.3f} wt% "
              f"(ceiling {proto.spec_max_o_wt_pct} wt%) → {proto.verdict}")
        print()
    fine = column_powder().__class__("powder", d50_um=20.0)
    sem_fine = semenov_critical_T(fine)
    print(f"fault case — d50 20 µm unpassivated in air: T_crit "
          f"{sem_fine['t_crit_ambient_C']:.0f} °C "
          f"({'hot dryer is safe' if sem_fine['dryer_subcritical'] else 'HOT DRYER RUNAWAY — the V6 §1.2 warning'})")
    tr = self_heat_transient(fine, film_nm=1.0)
    print(f"   transient at {tr['t_amb_C']:.0f} °C dryer air, 1 nm film: "
          f"peak {tr['t_peak_C']:.0f} °C"
          + (f", runaway after {tr['runaway_after_s']:.0f} s"
             if tr["runaway"] else ", stable"))
    print()
    for form in ("briquette", "flake", "foil"):
        print(f"postharvest_o_pickup_wt_pct({form!r}) = "
              f"{postharvest_o_pickup_wt_pct(form):.4f} wt% "
              "→ melt_balance live feed")


if __name__ == "__main__":  # pragma: no cover
    main()
