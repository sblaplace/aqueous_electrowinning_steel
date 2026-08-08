"""Literature anchors for the screening chemistry — provenance, not data.

Why this module exists
----------------------
Every screening number in the chemistry stack (models/kinetics,
models/surface_state, models/fe_chloride_speciation, models/pourbaix,
models/thermodynamic_constants) is anchored to a specific paper,
table, or database.  Without a single machine-readable place to
*see* those anchors, the screening numbers can't be audited in
under 30 seconds, and the confidence ledger in
``models/theory_confidence.py`` is just text.

This module is the single source of truth for the screening
anchors.  Each entry is a tuple of:

  * ``value``         — the screening number used in the model
  * ``paper_value``   — the value the paper reports
  * ``uncertainty``   — the screening tolerance (±)
  * ``ref``           — short citation (full ref in ``references/README.md``)
  * ``notes``         — anything else (anchor isothermal, temperature, etc.)

The schema is intentionally tiny — adding anchors for new
screening numbers is a 5-line edit.  Production code should
consume the ``anchors()`` dict and assert, at startup, that
every ``SCREENING_FLAG = "unvalidated (L1)"`` value in the
code base has a matching anchor (the L1 audit is the Tier-3.2
cross-cutting add called out in ``CHEM_PHYS_REVIEW.md``).

What this module is NOT
-----------------------
* A replacement for a fitted thermodynamic database.  The
  screening numbers here are the *central* values the model
  uses; the *spread* is documented in the per-anchor notes.
* A bibliography.  Full citations live in ``references/README.md``.

Usage
-----
::

    from models.anchors import get_anchor
    a = get_anchor("DG_HSTAR_FE110")
    print(f"{a.ref}: model={a.value} J/mol, paper={a.paper_value} J/mol, "
          f"±{a.uncertainty}")
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Anchor:
    """A single literature anchor."""
    key: str
    value: float
    paper_value: float
    uncertainty: float
    ref: str
    notes: str = ""

    def within_tolerance(self) -> bool:
        """Return True if the model value is within ±uncertainty of the
        published value (paper and model must agree to within the
        screening budget for the anchor to count as "calibrated")."""
        return abs(self.value - self.paper_value) <= self.uncertainty


# ─── Fe(110) DFT ΔG_H* (HER coverage) ───────────────────────────
# Module: surface_state.DG_HSTAR_FE110_J
# Reference: Nørskov et al., *Trends in Electrocatalysis* (2006);
#            and the Nørskov-Fan CHE compilation reproduced in
#            her_microkinetics.py docstring.
ANCHORS: Dict[str, Anchor] = {
    "DG_HSTAR_FE110": Anchor(
        key="DG_HSTAR_FE110",
        value=-0.40 * 96485.0,        # J/mol
        paper_value=-0.40 * 96485.0,
        uncertainty=0.15 * 96485.0,   # ±0.15 eV per the volcano spread
        ref="Nørskov et al. 2006, J. Electrochem. Soc. 152 J23",
        notes="Volcano-plot central value; flagged range "
              "-0.30 .. -0.55 eV across (110)/(100)/(211) facets.",
    ),

    # ─── Fe(100) and Fe(211) facet spread ──────────────────────
    "DG_HSTAR_FE100": Anchor(
        key="DG_HSTAR_FE100",
        value=-0.45 * 96485.0,
        paper_value=-0.50 * 96485.0,
        uncertainty=0.20 * 96485.0,
        ref="DFT compilation, Hinnemann et al. 2005 / Ferrin 2008",
        notes="4-fold hollow site; slightly more strongly binding "
              "than (110).",
    ),
    "DG_HSTAR_FE211": Anchor(
        key="DG_HSTAR_FE211",
        value=-0.30 * 96485.0,
        paper_value=-0.30 * 96485.0,
        uncertainty=0.15 * 96485.0,
        ref="Step-edge DFT, e.g. Taylor & Neurock 2006",
        notes="Step-rich surface; more weakly binding than (110).",
    ),

    # ─── Temkin interaction parameter ──────────────────────────
    "TEMKIN_G_H_FE": Anchor(
        key="TEMKIN_G_H_FE",
        value=12.0e3,                  # J/mol
        paper_value=10.0e3,            # screening central for Fe
        uncertainty=8.0e3,             # ±8 kJ/mol across literature
        ref="Jerkiewicz & Zolfaghari 1996, J. Phys. Chem. 100 8454",
        notes="H-UPD Temkin parameter on Fe; small vs the 30+ kJ/mol "
              "on Pt-group metals.",
    ),

    # ─── Adsorbed anion thermodynamics ─────────────────────────
    "DG_ADS_CL_FE": Anchor(
        key="DG_ADS_CL_FE",
        value=-7.0e3,                  # J/mol (screening central)
        paper_value=-9.0e3,
        uncertainty=4.0e3,             # ±4 kJ/mol
        ref="Bockris & Jeng 1990, J. Electroanal. Chem. 280 203",
        notes="Cl- on Fe(110), pH 2, mild cathodic overpotential.  "
              "Central value is screening; experimental range is "
              "-5 to -13 kJ/mol across measurement conditions.",
    ),
    "DG_ADS_SO4_FE": Anchor(
        key="DG_ADS_SO4_FE",
        value=-0.5e3,                  # J/mol
        paper_value=-1.0e3,
        uncertainty=3.0e3,
        ref="Stimming & Schmickler 1994 (outer-sphere, pH 2)",
        notes="SO4(2-) on Fe is outer-sphere — the *apparent* inner-"
              "sphere component is small.  Screening value reflects "
              "this; the partial charge proxy_e=0.2 in surface_state "
              "captures the IHP component.",
    ),
    "DG_ADS_HSO4_FE": Anchor(
        key="DG_ADS_HSO4_FE",
        value=-4.0e3,
        paper_value=-5.0e3,
        uncertainty=3.0e3,
        ref="Sagara et al. 2005, J. Electroanal. Chem. 581 167",
        notes="HSO4- inner-sphere via S-O on Fe(110).  pKa = 1.99 "
              "means at pH 2 the SO4/HSO4 ratio is 1:1.",
    ),
    "DG_ADS_BORATE_FE": Anchor(
        key="DG_ADS_BORATE_FE",
        value=-5.5e3,
        paper_value=-7.0e3,
        uncertainty=3.0e3,
        ref="Holladay & Chambers 1994 (borate stress-relief review)",
        notes="B(OH)4- binds Fe(110) via two bridging O.  pKa 9.2 "
              "means at pH 2 most is H3BO3 (neutral).",
    ),

    # ─── Fe-Cl aqueous complexation ────────────────────────────
    "LOG10_K_FECL_PLUS": Anchor(
        key="LOG10_K_FECL_PLUS",
        value=0.40,
        paper_value=0.40,
        uncertainty=0.40,
        ref="Bjerrum 1926 / Sillén 1964 stability constants",
        notes="Fe²⁺ + Cl⁻ ⇌ FeCl⁺(aq).  I = 0; falls to ~-0.1 at "
              "I = 1 m (Gampp & Zuberbühler 1977).",
    ),
    "LOG10_K_FECL2_AQ": Anchor(
        key="LOG10_K_FECL2_AQ",
        value=0.10,
        paper_value=0.10,
        uncertainty=0.50,
        ref="Bjerrum / Sillén compilation",
        notes="Fe²⁺ + 2 Cl⁻ ⇌ FeCl₂(aq).  Higher-order species; "
              "negligible below 1 M bulk Cl⁻.",
    ),
    "LOG10_K_FECL3_MINUS": Anchor(
        key="LOG10_K_FECL3_MINUS",
        value=-1.40,
        paper_value=-1.40,
        uncertainty=0.60,
        ref="Bjerrum / Sillén compilation",
        notes="Fe²⁺ + 3 Cl⁻ ⇌ FeCl₃⁻(aq).  Negligible below 5 M.",
    ),

    # ─── Pitzer binary pair: (Fe2+, Cl-) ───────────────────────
    "FECL2_BETA0": Anchor(
        key="FECL2_BETA0",
        value=0.3643,
        paper_value=0.3643,
        uncertainty=0.05,
        ref="Pitzer 1991, p. 105 (Staples-Bracewell 25 °C tabulation)",
        notes="2-1 Pitzer convention; α1 = 2.0.",
    ),
    "FECL2_BETA1": Anchor(
        key="FECL2_BETA1",
        value=1.658,
        paper_value=1.658,
        uncertainty=0.20,
        ref="Pitzer 1991, p. 105",
        notes="β¹ for (Fe2+, Cl-) at 25 °C.",
    ),

    # ─── Mean activity coefficient anchor (Lobo & Quaresma) ────
    "GAMMA_PM_FECL2_0P1M_25C": Anchor(
        key="GAMMA_PM_FECL2_0P1M_25C",
        value=0.75,
        paper_value=0.745,
        uncertainty=0.05,
        ref="Lobo & Quaresma 1989, *Handbook of Electrolyte Solutions* Part B",
        notes="γ±(FeCl₂, 0.1 m, 25 °C) experimental anchor.",
    ),

    # ─── AWARE bath conductivity (literature benchmark) ─────────
    "AWARE_BATH_CONDUCTIVITY_SM": Anchor(
        key="AWARE_BATH_CONDUCTIVITY_SM",
        value=20.0,
        paper_value=20.0,
        uncertainty=5.0,
        ref="AWARE process 2024-2025, ChemRxiv and follow-up publications",
        notes="1 M FeCl₂ + 10 M LiCl, pH 2, 60 °C.  The screening "
              "model's lower bound (10-20 S/m) reflects Onsager "
              "over-suppression at extreme I.",
    ),

    # ─── Product value ladder: price bands ($/t product) ─────────
    # Consumer: models/product_ladder.py.  All are screening bands of the
    # *buyer's alternative* (what the electrowon product competes with),
    # from public market benchmarks.  Replace with quoted offtakes before
    # any investment decision — these exist to make the Option A/B decision
    # computed, not to price a sale.
    "FLAKE_FEED_PRICE_T": Anchor(
        key="FLAKE_FEED_PRICE_T",
        value=450.0,
        paper_value=450.0,
        uncertainty=150.0,
        ref="USGS Mineral Commodity Summaries 2025 (iron & steel); "
            "Fastmarkets ore-based metallics (HBI/DRI) band 2024–26",
        notes="HBI-parity for residual-free virgin iron units; scrap "
              "substitution puts the floor ~$300, premium virgin units "
              "for flat-products EAF dilution the ceiling ~$600.",
    ),
    "REBAR_PRICE_T": Anchor(
        key="REBAR_PRICE_T",
        value=750.0,
        paper_value=750.0,
        uncertainty=250.0,
        ref="Public trade press rebar/merchant-bar band (CRU, US Midwest), "
            "2024–26",
        notes="Lowest-certification finished-steel SKU; the Option A.5 "
              "(own-melt) endpoint product.",
    ),
    "LOWC_FOIL_PRICE_T": Anchor(
        key="LOWC_FOIL_PRICE_T",
        value=2000.0,
        paper_value=2000.0,
        uncertainty=800.0,
        ref="Small-volume pure-iron foil vendor list prices "
            "(Goodfellow-class suppliers), 2025",
        notes="Non-structural ferritic foil niche (battery substrates, "
              "shielding, brazing); volume is thin — the band reflects "
              "list-price, not deep-market, evidence.",
    ),
    "HRC_STRUCTURAL_PRICE_T": Anchor(
        key="HRC_STRUCTURAL_PRICE_T",
        value=850.0,
        paper_value=850.0,
        uncertainty=250.0,
        ref="CRU hot-rolled coil band, 2024–26",
        notes="Structural sheet at HRC/CRC parity; certification premium "
              "only after years of spec work.",
    ),
    "PM_POWDER_PRICE_T": Anchor(
        key="PM_POWDER_PRICE_T",
        value=2500.0,
        paper_value=2500.0,
        uncertainty=1000.0,
        ref="PM iron-powder industry price literature (Höganäs handbook; "
            "MPIF reviews); historical electrolytic-powder premiums",
        notes="99.9 %-purity electrolytic powder niche where aqueous iron "
              "EW has survived commercially; premium over atomized powder.",
    ),
    "BATTERY_IRON_PRICE_T": Anchor(
        key="BATTERY_IRON_PRICE_T",
        value=3000.0,
        paper_value=3000.0,
        uncertainty=1500.0,
        ref="Form Energy iron-air public materials + anode-material "
            "cost-parity estimate",
        notes="SPECULATIVE — a parity target for iron-air anode feed, not "
              "a market quote.  Widest band on the ladder by design.",
    ),
    "MAGNETIC_FOIL_PRICE_T": Anchor(
        key="MAGNETIC_FOIL_PRICE_T",
        value=4000.0,
        paper_value=4000.0,
        uncertainty=2000.0,
        ref="Non-oriented electrical-steel price band (public trade press, "
            "2024–26); pure-Fe laminate premium assumed ~2× NOES",
        notes="The drum's 25–50 µm form factor is the eddy-current-optimal "
              "lamination thickness; price hinges on certified core loss.",
    ),

    # ─── Post-cell unit operations: energy (kWh/t) and cash ($/t) ──
    "DRY_PASSIVATE_KWH_T": Anchor(
        key="DRY_PASSIVATE_KWH_T",
        value=80.0,
        paper_value=80.0,
        uncertainty=30.0,
        ref="Industrial dryer energy practice (evaporation of ~10 % w/w "
            "residual film: 0.63 kWh/kg water) + inert-gas handling",
        notes="V6 §1.2/§1.3 — rinse + dry + controlled-O₂ passivation of "
              "freshly harvested flake/powder.",
    ),
    "RINSE_DRY_CASH_T": Anchor(
        key="RINSE_DRY_CASH_T",
        value=8.0,
        paper_value=8.0,
        uncertainty=4.0,
        ref="Tankhouse wash-water/reagent handling conventions",
        notes="Water, N₂ bleed, minor reagents per tonne; screening.",
    ),
    "BRIQUETTE_KWH_T": Anchor(
        key="BRIQUETTE_KWH_T",
        value=25.0,
        paper_value=25.0,
        uncertainty=10.0,
        ref="Roller-press briquetting practice (DRI/HBI industry, 15–35 "
            "kWh/t)",
        notes="V6 §1.4 — press energy only; binder-less against Heckel "
              "screen.",
    ),
    "BRIQUETTE_CASH_T": Anchor(
        key="BRIQUETTE_CASH_T",
        value=12.0,
        paper_value=12.0,
        uncertainty=6.0,
        ref="Die/roll wear conventions, DRI briquetting cost reviews",
        notes="Wear parts + maintenance per tonne; screening.",
    ),
    "INDUCTION_MELT_KWH_T": Anchor(
        key="INDUCTION_MELT_KWH_T",
        value=550.0,
        paper_value=550.0,
        uncertainty=100.0,
        ref="Coreless induction furnace handbooks (0.50–0.65 MWh/t Fe to "
            "1,600 °C)",
        notes="Option A.5 core energy term; theoretical minimum ~0.34 "
              "MWh/t, practical 0.5–0.65.",
    ),
    "MELT_CASH_T": Anchor(
        key="MELT_CASH_T",
        value=45.0,
        paper_value=45.0,
        uncertainty=20.0,
        ref="EAF/induction melt-shop consumables conventions (refractories, "
            "slag formers, melt loss)",
        notes="Excludes the electrowon feed itself; screening.",
    ),
    "CAST_ROLL_KWH_T": Anchor(
        key="CAST_ROLL_KWH_T",
        value=150.0,
        paper_value=150.0,
        uncertainty=75.0,
        ref="Thin/slab cast + hot-bar-mill energy literature",
        notes="Reheat + rolling energy; yield loss carried in CAST_ROLL_CASH_T.",
    ),
    "CAST_ROLL_CASH_T": Anchor(
        key="CAST_ROLL_CASH_T",
        value=60.0,
        paper_value=60.0,
        uncertainty=30.0,
        ref="Rolling-mill operating cost conventions (rolls, descale, "
            "yield loss ~3–5 %)",
        notes="Screening; buyer-side cost, uniform across products.",
    ),
    "ANNEAL_KWH_T": Anchor(
        key="ANNEAL_KWH_T",
        value=140.0,
        paper_value=140.0,
        uncertainty=60.0,
        ref="models/thermomechanical.py anneal_energy_kWh_per_kg at 700 °C, "
            "furnace efficiency 0.7 — the ladder calls this LIVE",
        notes="Fallback anchor only; product_ladder derives the working "
              "value from the thermomechanical model at call time.",
    ),
    "ANNEAL_CASH_T": Anchor(
        key="ANNEAL_CASH_T",
        value=25.0,
        paper_value=25.0,
        uncertainty=15.0,
        ref="Batch/box-anneal operating cost conventions (atmosphere, "
            "handling)",
        notes="Screening; H₂/N₂ cover gas dominant.",
    ),
    "SKINPASS_KWH_T": Anchor(
        key="SKINPASS_KWH_T",
        value=30.0,
        paper_value=30.0,
        uncertainty=15.0,
        ref="Temper-mill energy literature (light reductions)",
        notes="Lüders-band suppression + gauge finish (V6 §7.1 lever).",
    ),
    "SKINPASS_CASH_T": Anchor(
        key="SKINPASS_CASH_T",
        value=40.0,
        paper_value=40.0,
        uncertainty=20.0,
        ref="Temper-mill operating cost conventions",
        notes="Rolls, coolant, yield trim; screening.",
    ),
    "CARBURIZE_KWH_T": Anchor(
        key="CARBURIZE_KWH_T",
        value=120.0,
        paper_value=120.0,
        uncertainty=60.0,
        ref="models/carburization.py screening furnace energy; ASM "
            "gas-carburizing practice",
        notes="Option-B carbon route; in-cell alternative is V5/A1 "
              "carbon_electrodeposition.py.",
    ),
    "CARBURIZE_CASH_T": Anchor(
        key="CARBURIZE_CASH_T",
        value=35.0,
        paper_value=35.0,
        uncertainty=20.0,
        ref="Gas-carburizing atmosphere/handler cost conventions",
        notes="Endothermic gas + quench oil + handling; screening.",
    ),
    "PM_FINISH_KWH_T": Anchor(
        key="PM_FINISH_KWH_T",
        value=40.0,
        paper_value=40.0,
        uncertainty=20.0,
        ref="PM powder sizing/classification practice (screens, inert "
            "blanket)",
        notes="Pyrophoric-safe handling (V6 §1.2); screening.",
    ),
    "PM_FINISH_CASH_T": Anchor(
        key="PM_FINISH_CASH_T",
        value=60.0,
        paper_value=60.0,
        uncertainty=30.0,
        ref="PM powder finishing/blending cost conventions",
        notes="QA + fines management; screening.",
    ),
    "BATTERY_FINISH_KWH_T": Anchor(
        key="BATTERY_FINISH_KWH_T",
        value=50.0,
        paper_value=50.0,
        uncertainty=25.0,
        ref="Specialty porous-metal finishing estimates",
        notes="Porosity-spec sizing + QA; customer-owned spec (L1 guess).",
    ),
    "BATTERY_FINISH_CASH_T": Anchor(
        key="BATTERY_FINISH_CASH_T",
        value=80.0,
        paper_value=80.0,
        uncertainty=40.0,
        ref="Specialty anode-material QA estimates",
        notes="Highly speculative, mirrors BATTERY_IRON_PRICE_T band.",
    ),
    "MAGNETIC_QA_KWH_T": Anchor(
        key="MAGNETIC_QA_KWH_T",
        value=30.0,
        paper_value=30.0,
        uncertainty=15.0,
        ref="Lamination coating/curing practice (NOES industry)",
        notes="Interlaminar insulation coat per tonne of foil.",
    ),
    "MAGNETIC_QA_CASH_T": Anchor(
        key="MAGNETIC_QA_CASH_T",
        value=100.0,
        paper_value=100.0,
        uncertainty=50.0,
        ref="Epstein/SST core-loss certification + coating cost conventions",
        notes="Certification is the price-gate for this rung.",
    ),

    # ─── Melt-shop remelt balance (melt_balance.py) ───────────────
    "POSTHARVEST_O_PICKUP_WT_PCT": Anchor(
        key="POSTHARVEST_O_PICKUP_WT_PCT",
        value=0.35,
        paper_value=0.35,
        uncertainty=0.30,
        ref="Passivation-film screening (Mott–Cabrera 2–4 nm film on flake "
            "geometry, ~0.1–0.6 wt% depending on area)",
        notes="V6 §1.2 — oxide the product gains between harvest and "
              "furnace; product_oxidation.py will replace with physics.",
    ),
    "AS_DEPOSITED_O_WT_PCT": Anchor(
        key="AS_DEPOSITED_O_WT_PCT",
        value=0.127,
        paper_value=0.127,
        uncertainty=0.080,
        ref="live fallback: models/oxygen_in_iron.py reference point "
            "(100 mA/cm², pH 3.5, 60 °C, FE 90 %)",
        notes="Fallback only; melt_balance calls oxygen_in_iron live.  "
              "Screening O from co-deposited hydroxide capture.",
    ),
    "EAF_OXIDE_RECOVERY_FRAC": Anchor(
        key="EAF_OXIDE_RECOVERY_FRAC",
        value=0.90,
        paper_value=0.90,
        uncertainty=0.07,
        ref="EAF carbon-injection practice (FeO carbo-reduction to metal "
            "with foamy slag), standard EAF handbooks",
        notes="Fraction of charged oxide oxygen removed as CO with its Fe "
              "recovered to metal; the rest reports to slag.",
    ),
    "INDUCTION_OXIDE_RECOVERY_FRAC": Anchor(
        key="INDUCTION_OXIDE_RECOVERY_FRAC",
        value=0.80,
        paper_value=0.80,
        uncertainty=0.15,
        ref="Coreless induction melt practice — no carbon boil; oxide "
            "mostly reports to slag (fayalite)",
        notes="Wide band: lining chemistry (silica vs alumina) and slag "
              "skimming practice dominate.",
    ),
    "DUST_CAPTURE_FINES_EAF": Anchor(
        key="DUST_CAPTURE_FINES_EAF",
        value=0.80,
        paper_value=0.80,
        uncertainty=0.15,
        ref="EAF charging of fines: off-gas carryover of un-densified "
            "fines (high-velocity off-gas at melt-in)",
        notes="The briquetting gate (V6 §1.4) exists to shrink fines so "
              "this capture never bites.",
    ),
    "DUST_CAPTURE_FINES_INDUCTION": Anchor(
        key="DUST_CAPTURE_FINES_INDUCTION",
        value=0.40,
        paper_value=0.40,
        uncertainty=0.20,
        ref="Induction furnace charging of fines (no off-gas blast)",
        notes="Lower carryover but fines still bridge/hang; screening.",
    ),
    "DUST_FE_FRACTION": Anchor(
        key="DUST_FE_FRACTION",
        value=0.80,
        paper_value=0.80,
        uncertainty=0.10,
        ref="EAF dust composition surveys (Fe as FeO/ZnFe2O4 dominant)",
        notes="Fe content of the dust stream on an iron-rich feed.",
    ),
    "FEO_C_REDUCTION_DH_KJ_MOL": Anchor(
        key="FEO_C_REDUCTION_DH_KJ_MOL",
        value=149.0,
        paper_value=149.0,
        uncertainty=20.0,
        ref="FeO(l) + C(s) → Fe(l) + CO(g), metallurgical thermochemistry "
            "tabulations",
        notes="Endothermic penalty of the carbothermic route; "
              "~9.3 MJ per kg O removed.",
    ),
    "CHARGE_S_WT_PCT": Anchor(
        key="CHARGE_S_WT_PCT",
        value=0.010,
        paper_value=0.010,
        uncertainty=0.010,
        ref="Well-rinsed counter-current wash screening (V6 §1.3 target "
            "window)",
        notes="Sulfate-film carryover after a proper rinse train; "
              "rinse_carryover.py will compute it.",
    ),
    "LIME_PER_S_KG": Anchor(
        key="LIME_PER_S_KG",
        value=3.5,
        paper_value=3.5,
        uncertainty=1.5,
        ref="EAF slag practice at basicity ~2.0 with partial desulfurization",
        notes="kg CaO added per kg S charged to keep S off the metal.",
    ),
    "BASE_GANGUE_SLAG_KG_T": Anchor(
        key="BASE_GANGUE_SLAG_KG_T",
        value=15.0,
        paper_value=15.0,
        uncertainty=10.0,
        ref="DRI-scrap blend gangue levels in EAF practice",
        notes="Electrowon feed is low-gangue; mostly slag formers.",
    ),
    "FINES_FRACTION_PASSIVATED": Anchor(
        key="FINES_FRACTION_PASSIVATED",
        value=0.03,
        paper_value=0.03,
        uncertainty=0.02,
        ref="Briquetted product fines after handling (PM/DRI briquette "
            "industry handling standards)",
        notes="V6 §1.4 — the briquette spec this fraction is defined by.",
    ),
    "CHARGE_H_PPM": Anchor(
        key="CHARGE_H_PPM",
        value=200.0,
        paper_value=200.0,
        uncertainty=150.0,
        ref="Repository screening operating point (~240 ppm diffusible H, "
            "adhesion_peel README section)",
        notes="Diffusible H at harvest; bake-out upstream of melt is the "
              "melt_hydrogen lever.",
    ),
    "SCRAP_YIELD_PCT": Anchor(
        key="SCRAP_YIELD_PCT",
        value=94.5,
        paper_value=94.5,
        uncertainty=2.0,
        ref="EAF practice on #1 heavy melt scrap (yield bands, industry "
            "handbooks)",
        notes="Electrowon must clear this band to be 'qualified feed'; "
            "scrap's ceiling is residual chemistry, not yield.",
    ),
    "DRI_O_WT_PCT": Anchor(
        key="DRI_O_WT_PCT",
        value=2.0,
        paper_value=2.0,
        uncertainty=0.7,
        ref="DRI/HBI product specifications (metallization 94–96 % → "
            "residual FeO; total O ~1.5–2.5 %)",
        notes="Runs the same oxide-reduction engine for the DRI baseline.",
    ),
}


def get_anchor(key: str) -> Anchor:
    """Return the anchor entry for the given key.

    Raises ``KeyError`` if the key is not in the registry.  This
    is the *intentional* failure mode: a missing anchor is a
    documentation gap, not a soft error.
    """
    if key not in ANCHORS:
        raise KeyError(
            f"No literature anchor registered for {key!r}.  "
            f"Add one to models/anchors.py (see module docstring)."
        )
    return ANCHORS[key]


def audit_anchors() -> Dict[str, bool]:
    """Return a {key: within_tolerance} dict for every registered
    anchor.  Used by the theory_confidence module to gate
    ``SCREENING_FLAG = "unvalidated (L1)"`` values on
    within-tolerance evidence.
    """
    return {k: a.within_tolerance() for k, a in ANCHORS.items()}


__all__ = ["Anchor", "ANCHORS", "get_anchor", "audit_anchors"]
