"""Rinse carryover — bath liquor becomes melt-shop sulfur (V6 §1.3).

Why this module exists
----------------------
``hot_shortness.py`` (Round 5, E1) caps S in the *finished steel* and
``bath_impurity_codeposition.py`` puts bath S into the *deposit*.  The third
sulfur channel was never modelled: **adherent bath liquor**.  A flake web
pulled from the sulfate bath carries a Landau–Levich liquid film; a powder
column carries interstitial liquor.  Dried, that film is FeSO₄ + Na₂SO₄ +
boronate; charged into a melt the sulfate decomposes and the sulfur enters
the metal — at **kilograms per tonne**, unrinsed.  This module makes "how
much rinsing?" a **steel-grade decision** (V6 §1.3): stage count and rinse
ratio that keep charge S under the grade split, plus the water/effluent
price and the tank-return loop.

Physics / chemistry
-------------------
.. code-block:: text

    carryover liquor per tonne (two product-form mechanisms):
      web (foil/flake strip):  Landau–Levich film on both faces
          h_f = 0.94·l_c·Ca^(2/3),  l_c = √(σ/ρ_l g),  Ca = μ v / σ
          μ live from bath_rheology (Herschel–Bulkley, fixed-point:
          film shear γ ≈ v/h_f — viscosity and film solve each other)
          m_liq = A_t·2h_f·ρ_l·retention,  A_t = 1 tonne / (ρ_Fe·t_foil)
      powder column: interstitial cake liquor
          m_liq = f_cake/(1 − f_cake) per tonne dry Fe

    counter-current rinse train (n stages, water:drag-out ratio r):
      c_k/c_0 = (Σ_{i=0}^{n−k} r^i) / (Σ_{i=0}^{n} r^i)
      residual factor on the product:  F_n = 1 / Σ_{i=0}^{n} r^i
        …the celebrated counterflow result: n stages at one fresh-water
        stream r dilute as if each stage saw fresh water r.  (Erratum vs
        the V6 §1.3 print, which wrote c_n = c_0·(r/(1+r))^n — with r the
        water:drag-out ratio the tankhouse form is 1/Σ r^k; V6's printed
        r corresponds to drag-out:water here.)

    budgets per tonne = retained liquor volume × c_final:
      S from total sulfate (FeSO₄ + Na₂SO₄ + acid), Na, B, Fe-salt
      water demand = r · V_drag-out; rinse-1 returns to tank (V6 §1.3)
      final-rinse conductivity ≈ σ_bath · F_n  →  metrology endpoint

Live derivations
----------------
* shear viscosity from ``bath_rheology.herschel_bulkley_viscosity`` and
  liquor density from ``BathRheologyParams`` (fixed-point with the film).
* web withdrawal speeds from ``cell_architecture`` default velocities
  (drum/belt family); anchor fallback ``WEB_SPEED_M_S``.
* the deep-drawing S split from ``bath_impurity_codeposition``
  ``LOW_SULFUR_S_MAX`` (0.020 wt%) — the grade gate.
* ``melt_balance.py`` consumes ``default_charge_s_wt_pct()`` live for its
  charge-S default (anchor ``CHARGE_S_WT_PCT`` is the fallback), closing
  the V6 §1.3 → §1.5 feed.

Screening flag
--------------
L1.  Film retention, cake hold-up, rinse ratio, stage count, surface
tension and conductivities are anchored screening proxies; the cascade
mass balances are exact.  Particulate carryover, splashing and
rinse-water chemistry beyond the counterflow train are out of scope.
The boron channel is *tracked, not gated* (ppm-B is boron-steel
territory — a product-form decision for the ladder, not a rinse defect).

References
----------
* docs/CHEM_PHYS_IMPROVEMENTS_V6.md §1.3 (gap statement).
* Landau & Levich (1942) — withdrawal film law.
* Tankhouse Cu/Ni rinse-ratio and counter-current cascade practice.
* docs/BATH_SPEC.md §1 — liquor composition defaults.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .anchors import get_anchor
from .electrochemistry import RHO_FE

SCREENING_FLAG = "unvalidated (L1)"

# exact constants (g/mol unless noted)
G_GRAV = 9.80665
M_S = 32.06
M_NA = 22.990
M_B = 10.81
M_FE_G = 55.845

PRODUCT_FORMS = ("web", "powder")
MELT_SCREENING_S_WT_PCT = 0.010      # melt_balance CHARGE_S_WT_PCT band


def _aval(key: str) -> float:
    return float(get_anchor(key).value)


# ────────────────────────────────────────────────────────────────────────
#  Inputs: liquor, product form, rinse train
# ────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BathLiquor:
    """The bath that clings to the product (BATH_SPEC §1 defaults)."""

    fe2_M: Optional[float] = None       # → RINSE_BATH_FE2_M
    na2so4_M: Optional[float] = None    # → RINSE_BATH_NA2SO4_M
    h3bo3_M: Optional[float] = None     # → RINSE_BATH_H3BO3_M
    pH: float = 2.0

    def resolved(self) -> "BathLiquor":
        return BathLiquor(
            fe2_M=self.fe2_M if self.fe2_M is not None
            else _aval("RINSE_BATH_FE2_M"),
            na2so4_M=self.na2so4_M if self.na2so4_M is not None
            else _aval("RINSE_BATH_NA2SO4_M"),
            h3bo3_M=self.h3bo3_M if self.h3bo3_M is not None
            else _aval("RINSE_BATH_H3BO3_M"),
            pH=self.pH,
        )

    def sulfate_M(self) -> float:
        """Total sulfate: FeSO₄ + Na₂SO₄ + acid-side (pH → [H⁺])."""
        r = self.resolved()
        return r.fe2_M + r.na2so4_M + 10.0 ** (-r.pH)

    def species_g_per_L(self) -> Dict[str, float]:
        """Carryover-relevant solute mass concentrations (g/L), un-rinsed."""
        r = self.resolved()
        return {
            "S": self.sulfate_M() * M_S,
            "Na": 2.0 * r.na2so4_M * M_NA,
            "B": r.h3bo3_M * M_B,
            "Fe_salt": r.fe2_M * M_FE_G,
        }


def divided_cell_liquor() -> BathLiquor:
    """chemical_osmosis.py catholyte variant: 1.5 M FeSO₄ + 0.5 M Na₂SO₄."""
    return BathLiquor(fe2_M=1.5, na2so4_M=0.5)


@dataclass(frozen=True)
class ProductForm:
    """How the product leaves the bath (two carryover mechanisms).

    ``kind``: ``"web"`` (foil/flake strip — Landau–Levich film) or
    ``"powder"`` (powder column — interstitial cake liquor).  ``None``
    fields resolve live/anchored at call time.
    """

    kind: str
    architecture_id: Optional[str] = None     # live web speed hint
    web_speed_m_s: Optional[float] = None
    foil_thickness_um: Optional[float] = None
    drain_retention: Optional[float] = None
    cake_liquor_frac: Optional[float] = None

    def __post_init__(self) -> None:
        if self.kind not in PRODUCT_FORMS:
            raise ValueError(
                f"product kind must be one of {PRODUCT_FORMS}, got {self.kind!r}")


def foil_web() -> ProductForm:
    """Drum-and-strip foil (the ladder's foil rungs)."""
    return ProductForm("web", architecture_id="drum_and_strip")


def powder_column() -> ProductForm:
    """Rotating-cylinder powder (the ladder's flake/powder feed rungs)."""
    return ProductForm("powder", architecture_id="rotating_cylinder")


@dataclass(frozen=True)
class RinseTrain:
    """Counter-current rinse train."""

    n_stages: Optional[int] = None      # → RINSE_STAGES
    r_ratio: Optional[float] = None     # → RINSE_RATIO (water : drag-out)
    return_first_stage_to_tank: bool = True

    def resolved(self) -> "RinseTrain":
        return RinseTrain(
            n_stages=(int(round(_aval("RINSE_STAGES")))
                      if self.n_stages is None else self.n_stages),
            r_ratio=self.r_ratio if self.r_ratio is not None
            else _aval("RINSE_RATIO"),
            return_first_stage_to_tank=self.return_first_stage_to_tank,
        )


# ────────────────────────────────────────────────────────────────────────
#  Carryover liquor per tonne
# ────────────────────────────────────────────────────────────────────────

def _live_web_speed_m_s(architecture_id: Optional[str]) -> float:
    """Withdrawal speed from the architecture screen (peripheral/belt)."""
    if architecture_id:
        try:
            from .cell_architecture import ARCHITECTURES

            for arch in ARCHITECTURES:
                if arch.id == architecture_id:
                    return float(arch.default_velocity_m_s)
        except Exception:  # pragma: no cover - defensive
            pass
    return _aval("WEB_SPEED_M_S")


def _live_liquor_density_kg_m3() -> float:
    try:
        from .bath_rheology import BathRheologyParams

        return float(BathRheologyParams().density_kg_m3)
    except Exception:  # pragma: no cover - defensive
        return 1200.0


def withdrawal_film_um(
    web_speed_m_s: float,
    surface_tension_N_m: Optional[float] = None,
) -> Dict[str, float]:
    """Landau–Levich withdrawal film (µm), viscosity fixed-point.

    h_f = 0.94·l_c·Ca^(2/3) with Ca = μ(γ)·v/σ and the film shear
    γ ≈ v/h_f — the film sets its own shear rate, and the (shear-thinning,
    live) Herschel–Bulkley viscosity sets the film.  Solved by damped
    fixed-point iteration; both numbers are reported so the fixed point
    is auditable.
    """
    sigma = (surface_tension_N_m if surface_tension_N_m is not None
             else _aval("BATH_SURFACE_TENSION_N_M"))
    rho_l = _live_liquor_density_kg_m3()
    l_c = math.sqrt(sigma / (rho_l * G_GRAV))
    try:
        from .bath_rheology import BathRheologyParams, herschel_bulkley_viscosity

        rheo = BathRheologyParams()

        def mu(gamma: float) -> float:
            return herschel_bulkley_viscosity(gamma, rheo)
    except Exception:  # pragma: no cover - defensive
        def mu(gamma: float) -> float:
            return 1.0e-3

    # start from the Newtonian water guess, iterate to the HB fixed point
    mu_eff = 1.0e-3
    h = 0.94 * l_c * (mu_eff * web_speed_m_s / sigma) ** (2.0 / 3.0)
    for _ in range(100):
        gamma = web_speed_m_s / max(h, 1.0e-9)
        mu_new = mu(gamma)
        h_new = 0.94 * l_c * (mu_new * web_speed_m_s / sigma) ** (2.0 / 3.0)
        if abs(h_new - h) < 1.0e-12 and abs(mu_new - mu_eff) < 1.0e-9:
            h, mu_eff = h_new, mu_new
            break
        h = 0.5 * (h + h_new)          # damped iteration
        mu_eff = mu_new
    return {
        "film_um": h * 1.0e6,
        "capillary_length_mm": l_c * 1.0e3,
        "viscosity_Pa_s": mu_eff,
        "film_shear_s": web_speed_m_s / max(h, 1.0e-9),
        "capillary_number": mu_eff * web_speed_m_s / sigma,
        "surface_tension_N_m": sigma,
    }


def carryover_liquor_kg_per_t(product: ProductForm) -> Dict[str, float]:
    """Bath liquor mass (kg) carried per tonne of dry product."""
    rho_l = _live_liquor_density_kg_m3()
    if product.kind == "powder":
        f = (product.cake_liquor_frac if product.cake_liquor_frac is not None
             else _aval("POWDER_CAKE_LIQUOR_FRAC"))
        return {
            "m_liquor_kg_t": 1000.0 * f / (1.0 - f),
            "mechanism": "powder",
            "cake_liquor_frac": f,
            "liquor_density_kg_m3": rho_l,
        }
    speed = (product.web_speed_m_s if product.web_speed_m_s is not None
             else _live_web_speed_m_s(product.architecture_id))
    t_um = (product.foil_thickness_um if product.foil_thickness_um is not None
            else _aval("PRODUCT_FOIL_THICKNESS_UM"))
    retention = (product.drain_retention if product.drain_retention is not None
                 else _aval("DRAIN_RETENTION_FRAC"))
    film = withdrawal_film_um(speed)
    area_m2_per_t = (1000.0 / RHO_FE) / (t_um * 1.0e-6)
    m = (area_m2_per_t * 2.0 * film["film_um"] * 1.0e-6
         * rho_l * retention)
    return {
        "m_liquor_kg_t": m,
        "mechanism": "web",
        "web_speed_m_s": speed,
        "foil_thickness_um": t_um,
        "drain_retention": retention,
        "landau_levich_film_um": film["film_um"],
        "web_area_m2_per_t": area_m2_per_t,
        "viscosity_Pa_s": film["viscosity_Pa_s"],
        "capillary_number": film["capillary_number"],
        "liquor_density_kg_m3": rho_l,
    }


# ────────────────────────────────────────────────────────────────────────
#  Counter-current cascade
# ────────────────────────────────────────────────────────────────────────

def cascade_dilution(n_stages: int, r_ratio: float) -> float:
    """Residual concentration factor on the product, counter-current train.

    F_n = 1 / Σ_{i=0}^{n} r^i  (exact counterflow result; r = 1 handled —
    F_n = 1/(n+1)).  ``n_stages=0`` is the unrinsed baseline F_0 = 1.
    """
    if n_stages < 0:
        raise ValueError("n_stages must be >= 0")
    if r_ratio <= 1.0:
        return 1.0 / (n_stages + 1.0)
    return 1.0 / ((r_ratio ** (n_stages + 1) - 1.0) / (r_ratio - 1.0))


def stage1_concentration_ratio(n_stages: int, r_ratio: float) -> float:
    """c_1/c_0 of the first (dirtiest) rinse stage — the tank-return strength."""
    if n_stages < 1:
        return 1.0
    if r_ratio <= 1.0:
        num = float(n_stages)
        den = float(n_stages + 1)
        return num / den
    num = (r_ratio ** n_stages - 1.0) / (r_ratio - 1.0)
    den = (r_ratio ** (n_stages + 1) - 1.0) / (r_ratio - 1.0)
    return num / den


# ────────────────────────────────────────────────────────────────────────
#  Evaluation
# ────────────────────────────────────────────────────────────────────────

def _low_sulfur_split_wt_pct() -> float:
    try:
        from .bath_impurity_codeposition import LOW_SULFUR_S_MAX

        return float(LOW_SULFUR_S_MAX)
    except Exception:  # pragma: no cover - defensive
        return 0.020


@dataclass
class RinseReport:
    """Charge-borne budgets for one product form through one rinse train."""

    product_kind: str
    n_stages: int
    r_ratio: float
    m_liquor_kg_t: float
    dilution_factor: float
    budgets: Dict[str, Dict[str, float]]     # species → g/t, wt_pct
    charge_s_wt_pct: float
    residual_salt_g_t: float
    water_m3_t: float
    tank_return_salt_kg_t: float
    effluent_salt_kg_t: float
    final_rinse_conductivity_uS_cm: float
    endpoint_ok: bool
    deep_draw_split_wt_pct: float
    stages_for_melt_band: Optional[int]
    carryover: Dict[str, Any]
    verdict: str = ""
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["reasons"] = list(self.reasons)
        d["flag"] = SCREENING_FLAG
        return d


def evaluate_rinse(
    product: Optional[ProductForm] = None,
    rinse: Optional[RinseTrain] = None,
    bath: Optional[BathLiquor] = None,
) -> RinseReport:
    """Run one product form through one counter-current rinse train."""
    p = product or powder_column()
    t = (rinse or RinseTrain()).resolved()
    b = (bath or BathLiquor()).resolved()
    rho_l = _live_liquor_density_kg_m3()

    carry = carryover_liquor_kg_per_t(p)
    v_liquor_L = carry["m_liquor_kg_t"] / (rho_l / 1000.0)   # L per tonne
    dilution = cascade_dilution(t.n_stages, t.r_ratio)
    c1_ratio = stage1_concentration_ratio(t.n_stages, t.r_ratio)

    species_g_L = b.species_g_per_L()
    budgets: Dict[str, Dict[str, float]] = {}
    for name, g_L in species_g_L.items():
        g_t = v_liquor_L * g_L * dilution
        budgets[name] = {"g_per_t": g_t, "wt_pct": g_t / 1.0e4}
    charge_s = budgets["S"]["wt_pct"]
    residual_salt = sum(v["g_per_t"] for v in budgets.values())

    # water & salt loops (V6 §1.3: rinse #1 returns to the tank)
    water_L_t = t.r_ratio * v_liquor_L
    return_salt_kg = 0.0
    effluent_salt_kg = 0.0
    if t.return_first_stage_to_tank:
        return_salt_kg = (water_L_t * c1_ratio
                          * sum(species_g_L.values()) / 1000.0)
    else:
        effluent_salt_kg = (water_L_t * c1_ratio
                            * sum(species_g_L.values()) / 1000.0)

    sigma_bath_uS = _aval("BATH_CONDUCTIVITY_MS_CM") * 1000.0
    endpoint_uS = _aval("RINSE_ENDPOINT_US_CM")
    final_uS = sigma_bath_uS * dilution
    endpoint_ok = final_uS <= endpoint_uS

    split = _low_sulfur_split_wt_pct()
    stages_needed: Optional[int] = None
    if charge_s > MELT_SCREENING_S_WT_PCT:
        for n in range(t.n_stages + 1, 9):
            d = cascade_dilution(n, t.r_ratio)
            if v_liquor_L * species_g_L["S"] * d / 1.0e4 <= MELT_SCREENING_S_WT_PCT:
                stages_needed = n
                break

    reasons: List[str] = []
    if charge_s <= MELT_SCREENING_S_WT_PCT and endpoint_ok:
        verdict = "rinse-qualified"
    elif charge_s <= split:
        verdict = "conditional"
    else:
        verdict = "fails"

    g_s = budgets["S"]["g_per_t"]
    if t.n_stages == 0:
        reasons.append(
            f"unrinsed: {g_s:.0f} g S/t ({charge_s:.3f} wt%) — bath liquor "
            "dried straight onto the product; unmelt-feed-able")
    reasons.append(
        f"charge S {g_s:.1f} g/t = {charge_s:.4f} wt% "
        f"(melt band {MELT_SCREENING_S_WT_PCT} wt%, deep-draw split "
        f"{split} wt%)")
    if stages_needed is not None:
        reasons.append(f"needs stage {stages_needed} at r={t.r_ratio:g} to "
                       "reach the melt screening band")
    if not endpoint_ok:
        reasons.append(
            f"final-rinse conductivity {final_uS:.0f} µS/cm above the "
            f"{endpoint_uS:.0f} endpoint — metrology rejects the train "
            "(one more stage or higher r)")
    if p.kind == "web":
        reasons.append(
            f"web carryover: Landau–Levich film "
            f"{carry['landau_levich_film_um']:.1f} µm × drain retention "
            f"{carry['drain_retention']:.2f} → "
            f"{carry['m_liquor_kg_t']:.0f} kg liquor/t on "
            f"{carry['web_area_m2_per_t']:.0f} m² web/t")
    b_g = budgets["B"]["g_per_t"]
    if b_g >= 1.0:
        reasons.append(
            f"boron channel: {b_g:.1f} g B/t ({budgets['B']['wt_pct']*1e4:.0f} ppm) "
            "— ppm-B is boron-steel territory; tracked, not gated")

    return RinseReport(
        product_kind=p.kind,
        n_stages=t.n_stages,
        r_ratio=t.r_ratio,
        m_liquor_kg_t=carry["m_liquor_kg_t"],
        dilution_factor=dilution,
        budgets=budgets,
        charge_s_wt_pct=charge_s,
        residual_salt_g_t=residual_salt,
        water_m3_t=water_L_t / 1000.0,
        tank_return_salt_kg_t=return_salt_kg,
        effluent_salt_kg_t=effluent_salt_kg,
        final_rinse_conductivity_uS_cm=final_uS,
        endpoint_ok=endpoint_ok,
        deep_draw_split_wt_pct=split,
        stages_for_melt_band=stages_needed,
        carryover=carry,
        verdict=verdict,
        reasons=reasons,
    )


def stages_to_meet(
    target_s_wt_pct: float,
    product: Optional[ProductForm] = None,
    bath: Optional[BathLiquor] = None,
    r_ratio: Optional[float] = None,
    max_stages: int = 8,
) -> Optional[int]:
    """First stage count meeting a charge-S target (or None within max)."""
    for n in range(0, max_stages + 1):
        rep = evaluate_rinse(product=product, bath=bath,
                             rinse=RinseTrain(n_stages=n, r_ratio=r_ratio))
        if rep.charge_s_wt_pct <= target_s_wt_pct:
            return n
    return None


def default_charge_s_wt_pct(product_form: str = "briquette") -> float:
    """Live charge-S default for ``melt_balance`` (V6 §1.3 → §1.5 feed).

    ``briquette`` maps to the powder-column carryover family (the Option A
    briquettes densify the rotating-cylinder powder); ``foil``/``flake``
    map to the web mechanism; anything else falls back to powder.
    """
    if product_form in ("foil", "flake", "plate_or_foil"):
        rep = evaluate_rinse(product=foil_web())
    else:
        rep = evaluate_rinse(product=powder_column())
    return rep.charge_s_wt_pct


# ────────────────────────────────────────────────────────────────────────
#  Scope + CLI
# ────────────────────────────────────────────────────────────────────────

def model_scope() -> Dict[str, Any]:
    return {
        "screening_flag": SCREENING_FLAG,
        "live_derivations": [
            "bath_rheology.herschel_bulkley_viscosity + density (film "
            "viscosity fixed-point)",
            "cell_architecture default velocities (web withdrawal)",
            "bath_impurity_codeposition.LOW_SULFUR_S_MAX (grade split)",
        ],
        "live_consumers": [
            "melt_balance charge-S default (default_charge_s_wt_pct)",
        ],
        "exact": [
            "counter-current cascade mass balances (F_n = 1/Σ r^k)",
            "species budgets (stoichiometric sulfate → S)",
            "Landau–Levich film (0.94·l_c·Ca^(2/3) with fixed-point μ)",
        ],
        "screening_proxies_anchored": [
            "drain retention / cake hold-up",
            "rinse ratio & stage count (tankhouse practice)",
            "surface tension, conductivities, endpoint",
            "liquor composition (BATH_SPEC §1 family)",
        ],
        "out_of_scope": [
            "particulate carryover & splashing",
            "rinse-water chemistry beyond the counterflow train (reuse, "
            "softening)",
            "phosphorus chemistry (default 0 — purification.py owns feed P)",
            "sulfur fate inside the melt (melt_balance.py lime practice)",
            "deposit-metrology integration wiring (the endpoint contract "
            "is exported; deposit_metrology is thickness QC today)",
        ],
    }


def _row(rep: RinseReport) -> str:
    return (f"{rep.product_kind:<8} n={rep.n_stages} r={rep.r_ratio:>4.1f}"
            f"  S {rep.budgets['S']['g_per_t']:>8.1f} g/t"
            f"  ({rep.charge_s_wt_pct:.4f} wt%)"
            f"  salt {rep.residual_salt_g_t:>8.0f} g/t"
            f"  water {rep.water_m3_t:>5.2f} m³/t"
            f"  σ_end {rep.final_rinse_conductivity_uS_cm:>7.0f} µS/cm"
            f"  {rep.verdict}")


def main() -> None:  # pragma: no cover - CLI wrapper
    print(f"rinse_carryover — bath liquor → melt-shop sulfur  [{SCREENING_FLAG}]")
    print()
    for product, label in ((powder_column(), "powder column "
                            "(rotating_cylinder family)"),
                           (foil_web(), "foil web (drum_and_strip family)")):
        rep0 = evaluate_rinse(product=product, rinse=RinseTrain(n_stages=0))
        print(f"{label}: {rep0.m_liquor_kg_t:.0f} kg liquor/t unrinsed → "
              f"{rep0.budgets['S']['g_per_t']:.0f} g S/t "
              f"({rep0.charge_s_wt_pct:.3f} wt%)")
        for n in (1, 2, 3, 4):
            print("   ", _row(evaluate_rinse(
                product=product, rinse=RinseTrain(n_stages=n))))
        n_melt = stages_to_meet(MELT_SCREENING_S_WT_PCT, product=product)
        n_end = next(
            (n for n in range(1, 9)
             if evaluate_rinse(product=product,
                               rinse=RinseTrain(n_stages=n)).endpoint_ok),
            None)
        print(f"    least stages: melt band n={n_melt}; "
              f"conductivity endpoint n={n_end}")
        print()
    rb = evaluate_rinse(product=powder_column(), bath=divided_cell_liquor())
    print(f"divided-cell liquor (1.5 M FeSO₄ + 0.5 M Na₂SO₄) on powder: "
          f"S {rb.budgets['S']['g_per_t']:.1f} g/t, "
          f"Na {rb.budgets['Na']['g_per_t']:.1f} g/t, "
          f"verdict {rb.verdict}")
    rep = evaluate_rinse()
    print()
    print(f"defaults: n={rep.n_stages}, r={rep.r_ratio:g} → charge S "
          f"{rep.charge_s_wt_pct:.4f} wt% feeds melt_balance live "
          f"(anchor band {MELT_SCREENING_S_WT_PCT} wt%)")
    print(f"tank-return salt {rep.tank_return_salt_kg_t:.2f} kg/t; "
          f"water demand {rep.water_m3_t:.2f} m³/t "
          "(rinse-1 returns to tank — chemical_osmosis water budget)")


if __name__ == "__main__":  # pragma: no cover
    main()
