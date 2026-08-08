"""Briquetting / densification — the product-form gate (V6 §1.4).

Why this module exists
----------------------
The architecture screen reports product *form* (powder / flake / foil) but
nothing in the repository models the last unit operation of Option A:
**getting it shippable**.  A melt shop will not pneumatically fine-powder
an EAF roof and will not take delivery of a product that arrives as dust;
the harvested iron must be briquetted or pelletised to a size-and-strength
spec with low fines generation in transit — fines being simultaneously the
pyrophoric fraction (``product_oxidation.py``, §1.2) and the off-gas dust
fraction (``melt_balance.py``, §1.5).  Iron-flake/powder compaction is
mechanical metallurgy the repo *already owns the inputs for* — as-deposited
yield strength, hardness and elongation live in
``mechanical_properties.py`` — but never runs.  This module runs it, and
emits the **shippable-product spec block** (density, crush strength,
fines, residual O, press kWh/t) that ``feedstock_logistics.py`` /
``dark_mill.py`` price against.

Physics
-------
.. code-block:: text

    Heckel compaction (pressure-density master curve):
        ln(1/(1 − D_rel)) = K·P + A
        A    = ln(1/(1 − D_tap))               (rearrangement branch:
                                                the tapped fill state is
                                                recovered exactly at P = 0)
        1/K  = 3·σ_y·(1/hardness-family)       (Panelli & Filho ranking:
                                                the Heckel yield pressure
                                                ≈ 3× uniaxial yield)
        K_eff = friable · K                    (porous/friable electrodeposit:
                                                particle crush assists pore
                                                closure at low P — the V6
                                                inverted design goal made
                                                quantitative)
        hot die pressing (HBI-style): σ_y(T) = σ_y(ref)·soften(T)

    green strength after ejection:
        σ_g = σ_g0 · exp(b · D_rel)            (sintered bond × factor_s)

    springback & ejection:
        ε_sb = P / E(D_rel)·100%,  E(D) = E_Fe · D^m   (porous modulus)
        F_eject/F_press ≈ μ_wall · k_radial

    press specific work (closed bookkeeping given P(D)):
        w = ∫ P(D)/(ρ_Fe·D²) dD   from D_tap → D_final   [J/kg]
        delivered kWh/t = w / (3.6·10³) / η_hydraulic

    fines (ratio-form screening, anchored at a reference strength):
        fines% = fines_ref · (σ_ref/σ_g)^n

    densification order (the §1.2 ↔ §1.5 oxygen exchange):
        passivate-first: residual O = live passivation-film pickup
                                      (product_oxidation)
        sinter-first:    H₂ sinter 600–800 °C reduces the film →
                         anchored low residual O, at a thermal duty and a
                         repassivation overhead (hot bare Fe is pyrophoric
                         until re-passivated — noted, not re-modelled)

    hopper/flow reliability (Jenike-style screen, V6 §1.4 ferromagnetic
    agglomeration term):
        powder/flake:  B_rathole = H(θ)·(σ_c + p_mag)/(ρ_bulk·g)
        briquette:     B_bridge  = rule_multiple · briquette_size

Live derivations
----------------
* as-deposited σ_y comes live from
  ``mechanical_properties.MechanicalPropertiesModel().predict()`` at the
  reference DC operating point, with an anchored fallback — the V6 §1.4
  "Heckel constants from deposit σ_y" feed.  Changing the plating recipe
  (grain size, additives, H) changes σ_y there and re-derives every
  compaction number here.
* residual O in the passivate-first branch comes live from
  ``product_oxidation.postharvest_o_pickup_wt_pct`` (the §1.2 article spec),
  anchor fallback (``POSTHARVEST_O_PICKUP_WT_PCT``).
* ``melt_balance.py`` consumes the shipped-fines fraction live
  (``shipped_fines_fraction``), replacing the ``FINES_FRACTION_PASSIVATED``
  anchor (kept as fallback) — the §1.4 → §1.5 dust channel.

Screening flag
--------------
L1.  The Heckel/green-strength master-curve *shapes* and the bookkeeping
closures (A↔D_tap, work integral, CCS = σ·face) are exact; the constant
levels (σ_g0, b, friable factor, softening factor, fines ratio-form
exponent, magnetic cohesion) are anchored screening proxies — the fines
ratio exponent and magnetic cohesion are flagged SPECULATIVE.  The numbers
order the design space (cold vs hot press, powder vs flake, sinter-first
vs passivate-first); they do not certify a briquette.

References
----------
* docs/CHEM_PHYS_IMPROVEMENTS_V6.md §1.4 (gap statement).
* Heckel, R. W. (1961), *Trans. Metall. Soc. AIME* 221 — ln(1/(1−D))
  linear in P; Panelli & Ambrosio Filho (2001) — 1/K ≈ 3σ_y.
* Höganäs / MPIF iron-powder compaction literature — tap densities,
  green-strength master curves, press bands.
* Jenike, A. W. (1964), Bull. 123 — rathole diameter & H(θ); bunker
  bridging rule (outlet ≥ 6–8 particle diameters).
* ISO 4700 family — cold-crush screening floor for shipped agglomerates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .anchors import get_anchor
from .electrochemistry import RHO_FE

SCREENING_FLAG = "unvalidated (L1)"

G_CONST = 9.80665            # m/s²
E_FE_GPA = 205.0             # Young's modulus of α-Fe, screening constant
                            # (2G(1+ν) from mechanical_properties' shear
                            # modulus family)

PRODUCT_KINDS = ("powder", "flake")
ARCH_TO_KIND = {
    "rotating_cylinder": "powder",
    "drum_and_strip": "flake",
}


def _aval(key: str) -> float:
    return float(get_anchor(key).value)


# ────────────────────────────────────────────────────────────────────────
#  Feed form (what the harvester hands the press)
# ────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FeedForm:
    """The harvested product as it arrives at the press.

    ``plate_and_frame`` foil is a dense web at birth — it is shipped as
    coil/chopped strip and has no densification line here (raising is
    intentional).
    """

    kind: str = "powder"
    architecture_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in PRODUCT_KINDS:
            raise ValueError(
                f"kind must be one of {PRODUCT_KINDS} (foil webs are "
                f"shipped dense and bypass briquetting), got {self.kind!r}")

    def tap_density_rel(self) -> float:
        key = ("TAP_DENSITY_POWDER_REL" if self.kind == "powder"
               else "TAP_DENSITY_FLAKE_REL")
        return _aval(key)

    def bulk_rho_kg_m3(self) -> float:
        return self.tap_density_rel() * RHO_FE


def from_architecture(architecture_id: str) -> FeedForm:
    """Map a cell_architecture id to its press-feed kind (the V6 §1.4 feed)."""
    if architecture_id not in ARCH_TO_KIND:
        raise ValueError(
            f"no densification line mapped for {architecture_id!r} "
            f"(powder & flake only)")
    return FeedForm(kind=ARCH_TO_KIND[architecture_id],
                    architecture_id=architecture_id)


def column_powder() -> FeedForm:
    return FeedForm("powder", architecture_id="rotating_cylinder")


def drum_flake() -> FeedForm:
    return FeedForm("flake", architecture_id="drum_and_strip")


# ────────────────────────────────────────────────────────────────────────
#  Live yield strength (V6 §1.4: Heckel constants from deposit σ_y)
# ────────────────────────────────────────────────────────────────────────

def as_deposited_yield_MPa() -> Dict[str, Any]:
    """Deposit yield strength from the live mechanical model, with fallback.

    ``mechanical_properties.MechanicalPropertiesModel().predict()`` at the
    reference DC recipe (default args) → ``sigma_y_MPa``.  Any recipe change
    in that model re-derives the Heckel constant here — the rederivation
    seam the whole module tree is built around.
    """
    try:
        from .mechanical_properties import MechanicalPropertiesModel

        res = MechanicalPropertiesModel().predict()
        return {
            "sigma_y_MPa": float(res.sigma_y_MPa),
            "vickers_hv": float(res.vickers_hv),
            "source": "live (mechanical_properties reference DC recipe)",
        }
    except Exception:
        return {
            "sigma_y_MPa": _aval("AS_DEPOSITED_SIGMA_Y_MPA"),
            "vickers_hv": None,
            "source": "anchor fallback AS_DEPOSITED_SIGMA_Y_MPA",
        }


# ────────────────────────────────────────────────────────────────────────
#  Heckel compaction law
# ────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HeckelLaw:
    """ln(1/(1−D)) = K·P + A parameterisation.

    ``k_per_Pa`` is the pressability K (1/K ≈ 3σ_y for fully dense,
    non-friable metal; the friable factor rescales it for the crushable
    electrodeposit); ``a`` encodes the rearrangement branch, pinned so the
    tapped fill state D_tap is recovered at P = 0 (exact closure).
    """

    k_per_Pa: float
    a: float
    tap_density_rel: float
    sigma_y_effective_MPa: float
    friable_factor: float
    hot: bool = False

    @classmethod
    def from_yield(cls, sigma_y_MPa: float,
                   form: Optional[FeedForm] = None,
                   hot: bool = False,
                   friable_factor: Optional[float] = None) -> "HeckelLaw":
        if sigma_y_MPa <= 0:
            raise ValueError("sigma_y_MPa must be positive")
        frm = form or column_powder()
        fri = (friable_factor if friable_factor is not None
               else _aval("HECKEL_FRIABLE_FACTOR"))
        sy = sigma_y_MPa * (_aval("HOT_PRESS_SIGMA_SOFTEN") if hot else 1.0)
        k = fri / (3.0 * sy * 1.0e6)          # 1/K = 3σ_y/friable (Pa)
        d_tap = frm.tap_density_rel()
        if not 0.0 < d_tap < 1.0:
            raise ValueError("tap density must be a relative fraction")
        a = math.log(1.0 / (1.0 - d_tap))
        return cls(k_per_Pa=k, a=a, tap_density_rel=d_tap,
                   sigma_y_effective_MPa=sy, friable_factor=fri, hot=hot)

    def heckel_yield_pressure_MPa(self) -> float:
        """1/K in MPa — the ranking yield pressure (≈ 3σ_y_eff/friable)."""
        return 1.0 / (self.k_per_Pa * 1.0e6)

    def relative_density(self, pressure_MPa: float) -> float:
        """D_rel at applied pressure — exact Heckel inversion."""
        if pressure_MPa < 0.0:
            raise ValueError("pressure must be non-negative")
        return 1.0 - math.exp(-(self.k_per_Pa * pressure_MPa * 1.0e6 + self.a))

    def pressure_MPa_for(self, d_rel: float) -> float:
        """Applied pressure to reach D_rel — exact Heckel inversion."""
        if not self.tap_density_rel <= d_rel < 1.0:
            raise ValueError(
                f"d_rel must be in [{self.tap_density_rel:.3f}, 1) — below "
                "tap density the touchless rearrangement branch applies, "
                "which this law does not carry")
        s = math.log(1.0 / (1.0 - d_rel))
        return (s - self.a) / (self.k_per_Pa * 1.0e6)


def press_work_kWh_per_t(law: HeckelLaw, d_final: float,
                         n_steps: int = 8192,
                         hydraulic_eta: Optional[float] = None
                         ) -> Dict[str, float]:
    """Specific compaction work D_tap → d_final, ideal and delivered.

    w = ∫ P(D)/(ρ_Fe·D²) dD   [J/kg]  — the platen work that goes into
    the compact (exact bookkeeping given the Heckel P(D)); delivered
    energy adds the hydraulic/mechanical efficiency anchor.
    """
    d0 = law.tap_density_rel
    if not d0 < d_final < 1.0:
        raise ValueError(f"d_final must be in ({d0:.3f}, 1)")

    def _p_pa(d: float) -> float:
        return law.pressure_MPa_for(d) * 1.0e6

    h = (d_final - d0) / n_steps
    acc = 0.5 * _p_pa(d0) / (RHO_FE * d0 ** 2)          # endpoint (∝ P(D0) = 0)
    for i in range(1, n_steps):
        d = d0 + i * h
        acc += _p_pa(d) / (RHO_FE * d ** 2)
    acc += 0.5 * _p_pa(d_final) / (RHO_FE * d_final ** 2)
    w_J_kg = acc * h                                    # Pa·(kg⁻¹·m³)=J/kg
    ideal_kWh_t = w_J_kg * 1000.0 / 3.6e6               # J/kg → kWh/t
    eta = hydraulic_eta if hydraulic_eta is not None else _aval(
        "PRESS_HYDRAULIC_ETA")
    delivered_kWh_t = ideal_kWh_t / eta
    return {
        "work_ideal_J_kg": w_J_kg,
        "energy_ideal_kWh_per_t": ideal_kWh_t,
        "energy_delivered_kWh_per_t": delivered_kWh_t,
        "hydraulic_eta": eta,
    }


# ────────────────────────────────────────────────────────────────────────
#  Post-press state: strength, springback, ejection, fines, crush
# ────────────────────────────────────────────────────────────────────────

def green_strength_MPa(d_rel: float, sintered: bool = False) -> float:
    """σ_g = σ_g0·exp(b·D_rel); a sintered bond multiplies it (screening)."""
    if not 0.0 < d_rel <= 1.0:
        raise ValueError("d_rel must be in (0, 1]")
    g = (_aval("GREEN_STRENGTH_PRE_MPA")
         * math.exp(_aval("GREEN_STRENGTH_B") * d_rel))
    if sintered:
        g *= _aval("SINTER_STRENGTH_FACTOR")
    return g


def compact_modulus_GPa(d_rel: float) -> float:
    """E(D) = E_Fe·D^m — porous-compact modulus family (Phani–Niyogi)."""
    if not 0.0 < d_rel <= 1.0:
        raise ValueError("d_rel must be in (0, 1]")
    return E_FE_GPA * d_rel ** _aval("SPRINGBACK_MODULUS_EXP")


def springback_pct(pressure_MPa: float, d_rel: float) -> float:
    """Elastic recovery strain on unload, %: ε_sb = P/E(D)·100."""
    if pressure_MPa < 0.0:
        raise ValueError("pressure must be non-negative")
    return (pressure_MPa / (compact_modulus_GPa(d_rel) * 1000.0)) * 100.0


def ejection_fraction_pct() -> float:
    """Ejection force as % of compaction force: μ_wall · k_radial · 100."""
    return _aval("DIE_WALL_MU") * _aval("RADIAL_STRESS_FRACTION") * 100.0


def fines_fraction_wt_pct(strength_MPa: float) -> float:
    """Handling fines as a ratio-form in green/crush strength.

    Anchored exactly at the reference strength — fines(σ_ref) = fines_ref
    by construction; the strength exponent is the module's flagged
    SPECULATIVE ratio-form term.
    """
    if strength_MPa <= 0.0:
        raise ValueError("strength must be positive")
    n = _aval("FINES_STRENGTH_EXP")
    return (_aval("FINES_REF_PCT")
            * (_aval("GREEN_STRENGTH_REF_MPA") / strength_MPa) ** n)


def cold_crush_strength_N(strength_MPa: float,
                          size_mm: Optional[float] = None) -> float:
    """Cold-crush force = σ × face area (MPa·mm² = N, exact bookkeeping)."""
    size = size_mm if size_mm is not None else _aval("BRIQUETTE_SIZE_MM")
    return strength_MPa * size ** 2


# ────────────────────────────────────────────────────────────────────────
#  Densification order — the §1.2 ↔ §1.5 residual-oxygen exchange
# ────────────────────────────────────────────────────────────────────────

def residual_o_wt_pct(order: str, form: Optional[FeedForm] = None
                      ) -> Dict[str, Any]:
    """Charge-borne oxygen after densification, per processing order.

    passivate-first: the §1.2 passivation-film pickup, computed *live* from
    ``product_oxidation`` (anchor fallback).
    sinter-first: H₂-reduction sinter strips the film → anchored low
    residual; the product must be re-passivated before shipment (hot bare
    iron is the §1.2 fault case — carried as a note, not re-modelled).
    """
    frm = form or column_powder()
    if order == "passivate_first":
        try:
            from .product_oxidation import postharvest_o_pickup_wt_pct

            pickup = float(postharvest_o_pickup_wt_pct(frm.kind))
            source = "live (product_oxidation.postharvest_o_pickup_wt_pct)"
        except Exception:
            pickup = _aval("POSTHARVEST_O_PICKUP_WT_PCT")
            source = "anchor fallback POSTHARVEST_O_PICKUP_WT_PCT"
        return {
            "residual_o_wt_pct": pickup, "order": order, "source": source,
            "note": "passivation film carried into the melt (§1.2 article "
                    "spec ceiling applies)",
        }
    if order == "sinter_first":
        return {
            "residual_o_wt_pct": _aval("SINTER_RESIDUAL_O_WT_PCT"),
            "order": order, "source": "anchor SINTER_RESIDUAL_O_WT_PCT",
            "note": "H₂-reduction sinter strips the passive film; "
                    "re-passivation required before shipment (hot bare Fe "
                    "is the §1.2 pyrophoric fault case)",
        }
    raise ValueError(f"order must be passivate_first | sinter_first, "
                     f"got {order!r}")


# ────────────────────────────────────────────────────────────────────────
#  The densification line → shippable-product spec block
# ────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BriquettingLine:
    """One densification process window."""

    form: FeedForm = field(default_factory=column_powder)
    press_MPa: Optional[float] = None        # → PRESS_DESIGN_MPA
    hot: bool = False                        # hot die press (HBI-style)
    sinter_first: bool = False               # reduction sinter before densify
    sigma_y_MPa: Optional[float] = None      # override the live feed

    def resolved_press_MPa(self) -> float:
        return self.press_MPa if self.press_MPa is not None else _aval(
            "PRESS_DESIGN_MPA")

    def press_T_C(self) -> float:
        return _aval("HOT_PRESS_T_C") if self.hot else 25.0


@dataclass
class ShippableSpec:
    """The V6 §1.4 shippable-product spec block (screening verdict)."""

    line: str
    form_kind: str
    press_MPa: float
    hot: bool
    sinter_first: bool
    press_T_C: float
    sigma_y_effective_MPa: float
    heckel_yield_pressure_MPa: float
    relative_density: float
    density_kg_m3: float
    green_or_sintered_strength_MPa: float
    cold_crush_strength_N: float
    springback_pct: float
    ejection_fraction_pct: float
    fines_wt_pct: float
    press_kWh_per_t: float
    sinter_kWh_per_t: float
    residual_o_wt_pct: float
    verdict: str
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["reasons"] = list(self.reasons)
        d["flag"] = SCREENING_FLAG
        return d


def evaluate_line(line: Optional[BriquettingLine] = None) -> ShippableSpec:
    """Run the densification line and grade the shipped article.

    Verdict specs (anchored screening floors): relative density ≥
    ``DENSITY_REL_SPEC`` (sink-and-size), cold crush ≥ ``CCS_SPEC_N``
    (ISO-4700-family floor), shipped fines ≤ ``FINES_SPEC_PCT``.
    """
    ln = line or BriquettingLine()
    sy = (ln.sigma_y_MPa if ln.sigma_y_MPa is not None
          else as_deposited_yield_MPa()["sigma_y_MPa"])
    law = HeckelLaw.from_yield(sy, form=ln.form, hot=ln.hot)
    press = ln.resolved_press_MPa()
    d = law.relative_density(press)
    sintered = ln.sinter_first
    strength = green_strength_MPa(d, sintered=sintered)
    ccs = cold_crush_strength_N(strength)
    sb = springback_pct(press, d)
    ej = ejection_fraction_pct()
    fines = fines_fraction_wt_pct(strength)
    work = press_work_kWh_per_t(law, d)
    sinter_kwh = _aval("SINTER_KWH_PER_T") if sintered else 0.0
    o = residual_o_wt_pct(
        "sinter_first" if sintered else "passivate_first", form=ln.form)

    d_spec = _aval("DENSITY_REL_SPEC")
    ccs_spec = _aval("CCS_SPEC_N")
    fines_spec = _aval("FINES_SPEC_PCT")
    failures: List[str] = []
    reasons: List[str] = []
    if d < d_spec:
        failures.append(f"relative density {d:.3f} below the {d_spec:.2f} "
                        "sink-and-size spec (raise P, densify hot, or "
                        "soften the feed)")
    if ccs < ccs_spec:
        failures.append(f"cold crush {ccs:.0f} N below the {ccs_spec:.0f} N "
                        "screening floor")
    if fines > fines_spec:
        failures.append(f"shipped fines {fines:.2f} wt% above the "
                        f"{fines_spec:.1f} wt% buyer ceiling — fines are the "
                        "pyrophoric + fume fractions (§1.2/§1.5)")
    reasons.extend(failures)
    reasons.append(
        f"oxygen carried: {o['residual_o_wt_pct']:.3f} wt% ({o['order']}, "
        f"{o['source']})")
    name = (f"{ln.form.kind}, {'hot' if ln.hot else 'cold'} press"
            + (", sinter-first" if sintered else ""))
    verdict = "shippable-spec" if not failures else "conditional"
    return ShippableSpec(
        line=name, form_kind=ln.form.kind, press_MPa=press, hot=ln.hot,
        sinter_first=sintered, press_T_C=ln.press_T_C(),
        sigma_y_effective_MPa=law.sigma_y_effective_MPa,
        heckel_yield_pressure_MPa=law.heckel_yield_pressure_MPa(),
        relative_density=d, density_kg_m3=d * RHO_FE,
        green_or_sintered_strength_MPa=strength,
        cold_crush_strength_N=ccs, springback_pct=sb,
        ejection_fraction_pct=ej, fines_wt_pct=fines,
        press_kWh_per_t=work["energy_delivered_kWh_per_t"],
        sinter_kWh_per_t=sinter_kwh,
        residual_o_wt_pct=o["residual_o_wt_pct"],
        verdict=verdict, reasons=reasons,
    )


# ────────────────────────────────────────────────────────────────────────
#  Hopper / flow reliability — Jenike rathole & bridging screen
# ────────────────────────────────────────────────────────────────────────

def rathole_critical_outlet_m(form: Optional[FeedForm] = None) -> Dict[str, Any]:
    """Critical rathole diameter for the loose feed (Jenike H(θ) screen).

    B = H(θ)·(σ_c + p_mag)/(ρ_bulk·g).  The p_mag term is the ferromagnetic
    agglomeration contribution of fresh iron powder — flagged SPECULATIVE,
    decade band.  This is the *feed* line's handling problem (and of any
    shipped fines); the briquette itself flows on the bridging rule below.
    """
    frm = form or column_powder()
    sc = _aval("UNCONFINED_YIELD_PA")
    pmag = _aval("MAGNETIC_COHESION_PA")
    rho_b = frm.bulk_rho_kg_m3()
    b = _aval("JENIKE_H_THETA") * (sc + pmag) / (rho_b * G_CONST)
    return {
        "brathole_m": b,
        "h_theta": _aval("JENIKE_H_THETA"),
        "sigma_c_Pa": sc,
        "p_magnetic_Pa": pmag,
        "bulk_rho_kg_m3": rho_b,
        "kind": frm.kind,
        "note": "hopper outlet must exceed B to avoid ratholing; the "
                "magnetic-cohesion term is the V6 §1.4 ferromagnetic "
                "agglomeration screening term (SPECULATIVE band)",
        "flag": SCREENING_FLAG,
    }


def bridging_min_outlet_m(size_mm: Optional[float] = None) -> Dict[str, Any]:
    """Bridging-rule outlet floor for the *briquetted* product."""
    size = size_mm if size_mm is not None else _aval("BRIQUETTE_SIZE_MM")
    mult = _aval("BRIDGING_RULE_MULTIPLE")
    return {
        "min_outlet_m": mult * size / 1000.0,
        "rule_multiple": mult,
        "briquette_size_mm": size,
        "note": "coarse free-flowing bodies bridge, not rathole — outlet "
                "≥ multiple × particle dimension (bunker-design rule)",
        "flag": SCREENING_FLAG,
    }


# ────────────────────────────────────────────────────────────────────────
#  Exports to the rest of the tree
# ────────────────────────────────────────────────────────────────────────

def default_shipped_spec() -> ShippableSpec:
    """The recommended Option-A line: hot die-press of the column powder.

    Hot pressing at the anchored 600 °C screen is to the powder briquette
    what hot briquetting is to DRI (HBI practice): the yield-softened
    particle reaches the sink-and-size and crush specs at the standard
    press band.  Cold pressing at the same force lands conditional on
    density — that contrast is the module's headline.
    """
    return evaluate_line(BriquettingLine(form=column_powder(), hot=True))


def shipped_fines_fraction() -> float:
    """Shipped-product fines as a mass *fraction* (0–1) for melt_balance.

    Live §1.4 → §1.5 wiring: replaces the ``FINES_FRACTION_PASSIVATED``
    anchor (which stays as melt_balance's fallback).
    """
    return default_shipped_spec().fines_wt_pct / 100.0


def shippable_spec_block(line: Optional[BriquettingLine] = None
                         ) -> Dict[str, Any]:
    """The export block for feedstock_logistics/dark_mill consumers (V6 §1.4)."""
    spec = evaluate_line(line) if line is not None else default_shipped_spec()
    block = spec.to_dict()
    block["consumers"] = ["feedstock_logistics.py", "dark_mill.py"]
    return block


def model_scope() -> Dict[str, Any]:
    return {
        "screening_flag": SCREENING_FLAG,
        "live_derivations": [
            "as-deposited σ_y ← mechanical_properties.MechanicalPropertiesModel"
            ".predict() (reference DC recipe; anchor fallback)",
            "passivate-first residual O ← product_oxidation."
            "postharvest_o_pickup_wt_pct (anchor fallback)",
        ],
        "live_consumers": [
            "melt_balance charge fines ← briquetting.shipped_fines_fraction "
            "(replaces FINES_FRACTION_PASSIVATED live; anchor fallback kept)",
        ],
        "exact": [
            "Heckel invariants: A ↔ D_tap at P=0; P↔D inversions",
            "specific compaction work ∫P dV bookkeeping (J/kg → kWh/t)",
            "cold crush = strength × face area (MPa·mm² = N)",
            "fines anchored exactly at the reference strength (ratio form)",
        ],
        "screening_proxies_anchored": [
            "1/K = 3σ_y Heckel–yield ranking; friable factor on K",
            "green-strength master curve σ_g0·exp(b·D)",
            "hot-press σ_y softening factor (HBI-style screen)",
            "press hydraulic efficiency; die-wall μ; radial stress ratio",
            "fines ratio-form strength exponent — SPECULATIVE",
            "sinter thermal duty / strength factor / residual O",
            "modulus-density exponent (Phani–Niyogi family)",
            "Jenike H(θ), unconfined yield, bridging multiple",
            "magnetic cohesion pressure — SPECULATIVE decade band",
        ],
        "out_of_scope": [
            "die-fill segregation & tooling wear CAPEX detail",
            "full finite-element compaction mechanics (single Heckel law)",
            "drum/briquette geometry spectra (single pillow size)",
            "binder chemistry (binder-less press assumed)",
            "feedstock_logistics/dark_mill wiring (export block ready; "
            "consumers not wired yet)",
        ],
    }


# ────────────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────────────

def _spec_row(sp: ShippableSpec) -> str:
    return (f"{sp.line:<34} D_rel {sp.relative_density:>5.3f}"
            f"  σ_g {sp.green_or_sintered_strength_MPa:>6.1f} MPa"
            f"  CCS {sp.cold_crush_strength_N:>7.0f} N"
            f"  fines {sp.fines_wt_pct:>5.2f} wt%"
            f"  press {sp.press_kWh_per_t:>5.2f} kWh/t"
            f"  O {sp.residual_o_wt_pct:>5.3f} wt%  {sp.verdict}")


def main() -> None:  # pragma: no cover - CLI wrapper
    print(f"briquetting — densification, shippable-product spec  [{SCREENING_FLAG}]")
    print()
    feed = as_deposited_yield_MPa()
    print(f"as-deposited σ_y = {feed['sigma_y_MPa']:.0f} MPa "
          f"({feed['source']})")
    print(f"design press {_aval('PRESS_DESIGN_MPA'):.0f} MPa; hot press at "
          f"{_aval('HOT_PRESS_T_C'):.0f} °C (soften ×"
          f"{_aval('HOT_PRESS_SIGMA_SOFTEN'):.2f})")
    print()
    lines = (
        BriquettingLine(form=column_powder()),
        BriquettingLine(form=column_powder(), hot=True),
        BriquettingLine(form=drum_flake()),
        BriquettingLine(form=drum_flake(), hot=True),
        BriquettingLine(form=column_powder(), sinter_first=True),
    )
    for ln in lines:
        sp = evaluate_line(ln)
        print(_spec_row(sp))
        for r in sp.reasons:
            print(f"     · {r}")
    print()
    spec = default_shipped_spec()
    print(f"recommended Option-A line → {spec.line}: {spec.verdict}; "
          f"shipped fines {spec.fines_wt_pct:.2f} wt% feed melt_balance "
          f"live (fallback anchor {100*_aval('FINES_FRACTION_PASSIVATED'):.0f} wt%)")
    for kind in ("powder", "flake"):
        rh = rathole_critical_outlet_m(FeedForm(kind))
        print(f"hopper screen [{kind}]: rathole-critical outlet "
              f"{rh['brathole_m']*1000:.0f} mm (σ_c {rh['sigma_c_Pa']:.0f} Pa "
              f"+ magnetic {rh['p_magnetic_Pa']:.0f} Pa)")
    br = bridging_min_outlet_m()
    print(f"briquette bin: bridging-rule outlet ≥ "
          f"{br['min_outlet_m']*1000:.0f} mm")


if __name__ == "__main__":  # pragma: no cover
    main()
