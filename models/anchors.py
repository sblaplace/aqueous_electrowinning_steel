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
        ref="Well-rinsed counter-current wash screening band; "
            "rinse_carryover.py defaults land at or below it",
        notes="Fallback only; melt_balance calls rinse_carryover live "
              "(V6 §1.3).",
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
    # ─── Deposit corrosion at open circuit & ferric etch ────────
    #     (deposit_corrosion.py; V6 §1.1)
    "FE_ACID_JCORR_REF_UA_CM2": Anchor(
        key="FE_ACID_JCORR_REF_UA_CM2",
        value=10.0,
        paper_value=10.0,
        uncertainty=9.0,
        ref="Stern (1955); Kelly (1965) — mixed-potential corrosion of "
            "iron in deaerated dilute acid",
        notes="Reference state: pH 2.0, 25 °C, deaerated, clean "
              "(additive-free) Fe.  V6 §1.1 screening band 1–50 µA/cm²; "
              "aerated warm acid sits at the top of the band because O₂ "
              "mass transfer adds cathodic current — that part is derived "
              "live in deposit_corrosion, not anchored.",
    ),
    "FE_ACID_ANODIC_TAFEL_MV_DEC": Anchor(
        key="FE_ACID_ANODIC_TAFEL_MV_DEC",
        value=40.0,
        paper_value=40.0,
        uncertainty=10.0,
        ref="Stern (1955) — Fe dissolution Tafel slope in acid, "
            "b_a ≈ 30–44 mV/dec",
        notes="Bockris/Heusler mechanism band; screening central value 40.",
    ),
    "FE_ACID_HER_TAFEL_MV_DEC": Anchor(
        key="FE_ACID_HER_TAFEL_MV_DEC",
        value=120.0,
        paper_value=120.0,
        uncertainty=20.0,
        ref="Stern (1955) — H₂ evolution on iron in acid, "
            "b_c ≈ 0.10–0.12 V/dec",
        notes="2.3×(2RT/F) family; screening central value 120.",
    ),
    "FE_CORR_EA_KJ_MOL": Anchor(
        key="FE_CORR_EA_KJ_MOL",
        value=40.0,
        paper_value=40.0,
        uncertainty=20.0,
        ref="Apparent Arrhenius activation energy for iron corrosion in "
            "dilute acid (Stern-family data)",
        notes="Applied to both half-reaction exchange currents (screening); "
              "literature apparent Ea spans ~30–60 kJ/mol.",
    ),
    "O2_DIFFUSIVITY_25C_M2_S": Anchor(
        key="O2_DIFFUSIVITY_25C_M2_S",
        value=2.10e-9,
        paper_value=2.10e-9,
        uncertainty=0.3e-9,
        ref="Standard aqueous diffusivity tables, O₂ in water at 25 °C",
        notes="Companion to the Weiss (1970) solubility anchor already "
              "used in bath_startup.fe2_oxidation_rate.",
    ),
    "DIFFUSION_EA_KJ_MOL": Anchor(
        key="DIFFUSION_EA_KJ_MOL",
        value=15.0,
        paper_value=15.0,
        uncertainty=5.0,
        ref="Stokes–Einstein screening for aqueous diffusivity T-scaling",
        notes="Applied to D_O2 and D_Fe3 in deposit_corrosion; same "
              "family as the temperature scaling used across the "
              "transport modules.",
    ),
    "DIFFUSION_LAYER_IDLE_M": Anchor(
        key="DIFFUSION_LAYER_IDLE_M",
        value=50e-6,
        paper_value=50e-6,
        uncertainty=25e-6,
        ref="Screening boundary layer for a stagnant bath, same family as "
            "fe3_shuttle.py boundary_layer_m (50 µm)",
        notes="Quasi-steady layer thickness for the stirred/O₂-limited "
              "channels at idle; for the stagnant Fe³⁺ etch the layer "
              "grows as sqrt(2·D·t) instead — see module docstring.",
    ),
    "FE3_ETCH_K_REF_MOL_M2_S": Anchor(
        key="FE3_ETCH_K_REF_MOL_M2_S",
        value=5.0e-7,
        paper_value=5.0e-7,
        uncertainty=4.5e-7,
        ref="USBM RI-series chloride iron-EW flowsheets — ferric etch as "
            "the classical current-efficiency killer; FeCl₃-class iron "
            "etching practice, half-order in a_Fe3",
        notes="SPECULATIVE screening central value at the reference state "
              "(a_Fe3 = 0.05 M, pH 2.0, 25 °C).  order-of-magnitude band "
              "5e-8–5e-6; the V6 §1.1 text lists the etch as 'half-order "
              "in a_Fe3, strong T dependence, acid-catalysed'.  Replace "
              "with a measured coupon rate before relying on the number.",
    ),
    "FE3_ETCH_REF_M": Anchor(
        key="FE3_ETCH_REF_M",
        value=0.05,
        paper_value=0.05,
        uncertainty=0.02,
        ref="Definition — reference Fe³⁺ activity for the half-order "
            "etch law",
        notes="Chosen inside the membrane-crossover/PRE screening window "
              "for bath Fe³⁺; not a measurement.",
    ),
    "FE3_ETCH_H_ORDER": Anchor(
        key="FE3_ETCH_H_ORDER",
        value=0.5,
        paper_value=0.5,
        uncertainty=0.5,
        ref="V6 §1.1 'acid-catalysed' — screening reaction order in a_H",
        notes="Screening only; order 0–1 reported across etching systems.",
    ),
    "FE3_ETCH_EA_KJ_MOL": Anchor(
        key="FE3_ETCH_EA_KJ_MOL",
        value=50.0,
        paper_value=50.0,
        uncertainty=25.0,
        ref="Chemical etching Arrhenius screening ('strong T dependence', "
            "V6 §1.1)",
        notes="SPECULATIVE; wet-etch practice values span ~30–60 kJ/mol.",
    ),
    "ADDITIVE_BLOCKING_COVERAGE": Anchor(
        key="ADDITIVE_BLOCKING_COVERAGE",
        value=0.5,
        paper_value=0.5,
        uncertainty=0.4,
        ref="Organic-additive adsorption blocking of iron corrosion — "
            "pickling-inhibitor / brightener practice, screening",
        notes="Fraction of the surface blocked for the kinetic "
              "(H⁺ and Fe) terms; the O₂-limiting transport term is "
              "coverage-insensitive at this level of modelling.",
    ),
    "IDLE_BATH_FE3_M": Anchor(
        key="IDLE_BATH_FE3_M",
        value=1.0e-4,
        paper_value=1.0e-4,
        uncertainty=9.0e-5,
        ref="Fallback only — live default derives from fe3_shuttle "
            "steady-state [Fe³⁺] (open-headspace scenario)",
        notes="Screening band 1e-5–1e-3 M residual ferric in a "
              "hydrolysis-capped sulfate bath at pH ~2.",
    ),
    "O2_FRACTION_SAT_IDLE": Anchor(
        key="O2_FRACTION_SAT_IDLE",
        value=0.05,
        paper_value=0.05,
        uncertainty=0.05,
        ref="Screening stand-in for headspace contact during idle — same "
            "convention as fe3_shuttle.ShuttleScenario.o2_fraction_of_sat",
        notes="Sealed cell ≈ 0.005 (fe3_shuttle.sealed_divided_cell); "
              "idle-with-access ≈ 0.05 central; open headspace ≈ 1.0.  "
              "O₂ mass transfer is the dominant idle-corrosion driver, so "
              "this knob moves the answer most.",
    ),
    # ─── Rinse carryover — bath liquor becomes melt-shop sulfur ────
    #     (rinse_carryover.py; V6 §1.3)
    "BATH_SURFACE_TENSION_N_M": Anchor(
        key="BATH_SURFACE_TENSION_N_M",
        value=0.060,
        paper_value=0.060,
        uncertainty=0.015,
        ref="Concentrated sulfate electrolyte ± plating additives "
            "(water 0.072; electrolyte/surfactant shifts down), screening",
        notes="Landau–Levich film: l_c = √(σ/ρg), Ca = μv/σ.",
    ),
    "DRAIN_RETENTION_FRAC": Anchor(
        key="DRAIN_RETENTION_FRAC",
        value=0.30,
        paper_value=0.30,
        uncertainty=0.20,
        ref="Gravity-drain + wiper/air-knife practice on withdrawn webs "
            "(tankhouse/film-coating practice), screening",
        notes="Fraction of the Landau–Levich film that survives drainage "
              "to the rinse train; without wipers the full film carries.",
    ),
    "POWDER_CAKE_LIQUOR_FRAC": Anchor(
        key="POWDER_CAKE_LIQUOR_FRAC",
        value=0.08,
        paper_value=0.08,
        uncertainty=0.04,
        ref="Drained filter-cake liquor hold-up (industrial filtration / "
            "centrifuge practice for fine powders), screening",
        notes="Interstitial liquor per kg of powder cake at the rinse-"
              "train inlet; the powder column's carryover mechanism "
              "(vs the web's Landau–Levich film).",
    ),
    "RINSE_RATIO": Anchor(
        key="RINSE_RATIO",
        value=5.0,
        paper_value=5.0,
        uncertainty=3.0,
        ref="Tankhouse rinse practice — fresh-water : drag-out volume "
            "ratio for a counter-current rinse train, screening",
        notes="Cu/Ni tankhouses run r ≈ 2–20 per train depending on "
              "water cost and drag-out value; counter-flow staging is "
              "what makes low r effective (c_n = c_0/(Σ r^k)).",
    ),
    "RINSE_STAGES": Anchor(
        key="RINSE_STAGES",
        value=3.0,
        paper_value=3.0,
        uncertainty=1.0,
        ref="Tankhouse cascade practice — counter-current rinse stages "
            "before drying, screening",
        notes="The V6 §1.3 steel-grade knob: stage count × rinse ratio "
              "sets charge S on the product.",
    ),
    "PRODUCT_FOIL_THICKNESS_UM": Anchor(
        key="PRODUCT_FOIL_THICKNESS_UM",
        value=15.0,
        paper_value=15.0,
        uncertainty=10.0,
        ref="Drum/harvested foil-flake thickness band (electrofoil "
            "practice, cell_architecture drum_and_strip family), screening",
        notes="Sets web area per tonne — and therefore the liquor film "
              "mass per tonne — for the web carryover mechanism.",
    ),
    "WEB_SPEED_M_S": Anchor(
        key="WEB_SPEED_M_S",
        value=0.25,
        paper_value=0.25,
        uncertainty=0.15,
        ref="Fallback only — live default from cell_architecture default "
            "velocities (drum/belt 0.25–1.0 m/s, Cu-foil practice 5–30 "
            "m/min band)",
        notes="Fallback for the withdrawal speed when cell_architecture "
              "is unavailable.",
    ),
    "BATH_CONDUCTIVITY_MS_CM": Anchor(
        key="BATH_CONDUCTIVITY_MS_CM",
        value=100.0,
        paper_value=100.0,
        uncertainty=50.0,
        ref="1–1.5 M sulfate electrolyte conductivity, screening "
            "(order 50–150 mS/cm)",
        notes="Final-rinse endpoint baseline: rinse-water conductivity ≈ "
              "bath conductivity × the cascade dilution factor.",
    ),
    "RINSE_ENDPOINT_US_CM": Anchor(
        key="RINSE_ENDPOINT_US_CM",
        value=500.0,
        paper_value=500.0,
        uncertainty=400.0,
        ref="Tankhouse final-rinse conductivity acceptance practice, "
            "screening",
        notes="The metrology consumable: a cheap conductivity pen on the "
              "last stage accepts/rejects the rinse train endpoint.",
    ),
    "RINSE_BATH_FE2_M": Anchor(
        key="RINSE_BATH_FE2_M",
        value=1.0,
        paper_value=1.0,
        uncertainty=0.5,
        ref="docs/BATH_SPEC.md §1.1 (1.0 M Fe²⁺ operating target; "
            "chemical_osmosis catholyte variant 1.5 M)",
        notes="Sulfate speciation screening: total SO₄²⁻ ≈ [Fe²⁺] + "
              "[Na₂SO₄] + [H⁺] (pH 2 → 0.01 M).",
    ),
    "RINSE_BATH_NA2SO4_M": Anchor(
        key="RINSE_BATH_NA2SO4_M",
        value=0.0,
        paper_value=0.0,
        uncertainty=0.5,
        ref="docs/BATH_SPEC.md §1 (Na₂SO₄ optional, 0 for first runs); "
            "chemical_osmosis.py divided-cell catholyte runs 0.5 M",
        notes="Supporting-electrolyte sodium: the Na budget channel.",
    ),
    "RINSE_BATH_H3BO3_M": Anchor(
        key="RINSE_BATH_H3BO3_M",
        value=0.40,
        paper_value=0.40,
        uncertainty=0.10,
        ref="docs/BATH_SPEC.md §1.2 (0.40 M boric-acid buffer)",
        notes="Borate carryover → charge boron: ppm-B is boron-steel "
              "territory; tracked, not gated here.",
    ),
    # ─── Product oxidation, drying & pyrophoricity ───────────────
    #     (product_oxidation.py; V6 §1.2)
    "PASSIV_FILM_LIM_NM": Anchor(
        key="PASSIV_FILM_LIM_NM",
        value=3.0,
        paper_value=3.0,
        uncertainty=1.0,
        ref="Mott–Cabrera native passive film on iron at room temperature "
            "(2–4 nm air-formed/fixed film)",
        notes="Post-harvest passivation target: grow this film under "
              "controlled p_O2/T and oxidation effectively stops.",
    ),
    "PASSIV_TAU_S": Anchor(
        key="PASSIV_TAU_S",
        value=3600.0,
        paper_value=3600.0,
        uncertainty=3000.0,
        ref="Room-temperature approach to the limiting passive film "
            "(minutes–hours), screening",
        notes="Log-law passivation timescale; after ~5τ the RT film is "
              "the Mott–Cabrera limiting film.",
    ),
    "OXIDE_O_MASS_FRAC": Anchor(
        key="OXIDE_O_MASS_FRAC",
        value=0.276,
        paper_value=0.276,
        uncertainty=0.020,
        ref="Passive film composition proxy Fe₃O₄ (27.6 wt% O; band "
            "FeO 0.223 – Fe₂O₃ 0.300 – FeOOH ~0.286)",
        notes="Mass of charge oxygen per mass of passive film (the "
              "film → charge-O converter).",
    ),
    "OXIDE_DENSITY_KG_M3": Anchor(
        key="OXIDE_DENSITY_KG_M3",
        value=5170.0,
        paper_value=5170.0,
        uncertainty=300.0,
        ref="Magnetite density (passive-film proxy), standard tables",
        notes="Pairs with OXIDE_O_MASS_FRAC to convert film nm → O mass "
              "per unit area.",
    ),
    "POWDER_D50_UM": Anchor(
        key="POWDER_D50_UM",
        value=60.0,
        paper_value=60.0,
        uncertainty=40.0,
        ref="Electrolytic / PM iron powder mesh band (−100 mesh family, "
            "PM industry catalogs), screening",
        notes="d50 of the rotating-column powder before any PM finishing "
              "(pm_powder_finish will own the sizing spec).",
    ),
    "POWDER_ROUGHNESS_FACTOR": Anchor(
        key="POWDER_ROUGHNESS_FACTOR",
        value=3.0,
        paper_value=3.0,
        uncertainty=2.0,
        ref="Dendritic electrowon powder vs smooth spheres (electrolytic "
            "powder morphology), screening",
        notes="Area multiplier on the equivalent-sphere specific surface.",
    ),
    "OX_RATE_REF_MOL_M2_S": Anchor(
        key="OX_RATE_REF_MOL_M2_S",
        value=1.0e-8,
        paper_value=1.0e-8,
        uncertainty=9.0e-9,
        ref="SPECULATIVE — fine iron powder oxidation rate at 60 °C, 3 nm "
            "film, air (parabolic inverse-film law), calibrated so "
            "passivated −100 mesh powder hot-dries with a slim margin "
            "while unpassivated <45 µm powder runs away (PM-industry "
            "phenomenology)",
        notes="mol O consumed per m² per s at the reference state; the "
              "dominant speculative constant of the module (decade band).",
    ),
    "OX_EA_KJ_MOL": Anchor(
        key="OX_EA_KJ_MOL",
        value=90.0,
        paper_value=90.0,
        uncertainty=30.0,
        ref="Parabolic oxidation activation energy screening for thin-"
            "film iron oxidation",
        notes="Sets the self-heating temperature leverage in the Semenov "
              "balance.",
    ),
    "OX_HEAT_KJ_MOL_O": Anchor(
        key="OX_HEAT_KJ_MOL_O",
        value=280.0,
        paper_value=280.0,
        uncertainty=40.0,
        ref="Fe₃O₄ formation enthalpy −1118 kJ/mol oxide → ~280 kJ per "
            "mol O atoms bound (metallurgical thermochemistry)",
        notes="Heat release per mol O in the Semenov generation term.",
    ),
    "DRYER_H_W_M2K": Anchor(
        key="DRYER_H_W_M2K",
        value=15.0,
        paper_value=15.0,
        uncertainty=10.0,
        ref="Convective tray/rotary dryer bed heat-transfer screening",
        notes="Heat-removal coefficient for the Semenov loss term.",
    ),
    "TRAY_BED_DEPTH_M": Anchor(
        key="TRAY_BED_DEPTH_M",
        value=0.02,
        paper_value=0.02,
        uncertainty=0.01,
        ref="Tray bed depth screening (top-cooled lump, A/V = 1/depth)",
        notes="Geometry of the drying lump the Semenov balance runs on.",
    ),
    "POWDER_BULK_DENSITY_KG_M3": Anchor(
        key="POWDER_BULK_DENSITY_KG_M3",
        value=2500.0,
        paper_value=2500.0,
        uncertainty=500.0,
        ref="Poured/tapped electrolytic iron powder bulk density "
            "(~30 % theoretical), screening",
        notes="Converts bed geometry to per-kg heat loss.",
    ),
    "PASSIV_PO2_FRAC": Anchor(
        key="PASSIV_PO2_FRAC",
        value=0.010,
        paper_value=0.010,
        uncertainty=0.009,
        ref="Controlled-passivation practice: ~1 % O₂ in N₂ blankets "
            "(PM powder passivation/industry handling)",
        notes="The passivation window's oxygen partial-pressure setting.",
    ),
    "PASSIV_PROTOCOL_T_C": Anchor(
        key="PASSIV_PROTOCOL_T_C",
        value=50.0,
        paper_value=50.0,
        uncertainty=20.0,
        ref="Passivation/drying protocol temperature screening "
            "(warm, sub-critical)",
        notes="Warm enough to dry, cool enough that the parabolic branch "
              "cannot outrun dissipation.",
    ),
    "DRYER_AIR_T_C": Anchor(
        key="DRYER_AIR_T_C",
        value=80.0,
        paper_value=80.0,
        uncertainty=20.0,
        ref="Standard powder drying-air temperature screening",
        notes="The hot-and-fast-dried fault case V6 §1.2 warns about; "
              "checked against the Semenov criterion.",
    ),
    "PRODUCT_PASSIV_O_MAX_WT_PCT": Anchor(
        key="PRODUCT_PASSIV_O_MAX_WT_PCT",
        value=1.5,
        paper_value=1.5,
        uncertainty=0.5,
        ref="V6 §1.2 product-spec text: 'flake, passivated, ≤1.5 wt% O, "
            "non-pyrophoric' is a buyable article",
        notes="The passivated-product oxygen ceiling for the spec "
            "verdict; every 1 wt% O is ~4.5 wt% FeO the melt must boil.",
    ),
    "STORAGE_HOURS": Anchor(
        key="STORAGE_HOURS",
        value=72.0,
        paper_value=72.0,
        uncertainty=48.0,
        ref="Bin/bag residence between passivation and charge, screening",
        notes="Air-exposure budget at RT log-law film growth after the "
              "controlled passivation step.",
    ),
    "COMBUSTIBLE_DUST_D50_UM": Anchor(
        key="COMBUSTIBLE_DUST_D50_UM",
        value=45.0,
        paper_value=45.0,
        uncertainty=20.0,
        ref="NFPA/PM-industry combustible-dust classification boundary "
            "for iron powder handling",
        notes="Below this d50, unpassivated + H-bearing powder is "
              "handled as a pyrophoricity candidate.",
    ),
    # ── V6 §1.4 briquetting / densification ────────────────────────────
    "TAP_DENSITY_POWDER_REL": Anchor(
        key="TAP_DENSITY_POWDER_REL",
        value=0.42,
        paper_value=0.42,
        uncertainty=0.08,
        ref="Electrolytic iron powder apparent-density band ~2.9–3.4 g/cm³ "
            "of 7.874 (Höganäs/MPIF powder grades)",
        notes="As-scraped fill state: the Heckel A = ln(1/(1−D_tap)) "
              "rearrangement branch is pinned to this exactly.",
    ),
    "TAP_DENSITY_FLAKE_REL": Anchor(
        key="TAP_DENSITY_FLAKE_REL",
        value=0.30,
        paper_value=0.30,
        uncertainty=0.08,
        ref="Flake packs poorly (high aspect ratio); PM flake fill "
            "practice",
        notes="Drum-and-strip harvest state; lower fill than powder is "
              "the flake's densification handicap.",
    ),
    "AS_DEPOSITED_SIGMA_Y_MPA": Anchor(
        key="AS_DEPOSITED_SIGMA_Y_MPA",
        value=450.0,
        paper_value=450.0,
        uncertainty=150.0,
        ref="Electrodeposited α-Fe yield screening band (nanocrystalline "
            "600–1000 MPa; coarse coupon 300–600 MPa)",
        notes="Fallback only; briquetting.py takes σ_y live from "
              "mechanical_properties (V6 §1.4 Heckel-constant feed).",
    ),
    "HECKEL_FRIABLE_FACTOR": Anchor(
        key="HECKEL_FRIABLE_FACTOR",
        value=1.3,
        paper_value=1.3,
        uncertainty=0.3,
        ref="Screening factor for crush-assisted densification of "
            "porous/friable electrodeposit particles",
        notes="Multiplies the Heckel K — RESEARCH_PROGRAM Option A's "
              "inverted design goal (a friable deposit is a virtue) made "
              "quantitative.",
    ),
    "PRESS_DESIGN_MPA": Anchor(
        key="PRESS_DESIGN_MPA",
        value=550.0,
        paper_value=550.0,
        uncertainty=150.0,
        ref="Commercial iron-powder die-press band 400–830 MPa "
            "(Höganäs/MPIF practice)",
        notes="Design compaction pressure for the screening lines.",
    ),
    "PRESS_HYDRAULIC_ETA": Anchor(
        key="PRESS_HYDRAULIC_ETA",
        value=0.50,
        paper_value=0.50,
        uncertainty=0.20,
        ref="Hydraulic-press mechanical efficiency screening band",
        notes="Delivered press kWh/t = ideal compaction work / η.",
    ),
    "GREEN_STRENGTH_PRE_MPA": Anchor(
        key="GREEN_STRENGTH_PRE_MPA",
        value=0.45,
        paper_value=0.45,
        uncertainty=0.20,
        ref="PM iron-powder green-strength master curves at 550–690 MPa "
            "(8–20 MPa) read back through the exp(b·D) law",
        notes="σ_g0 of σ_g = σ_g0·exp(b·D_rel); pinned with B so the "
              "design-density state lands mid-band.",
    ),
    "GREEN_STRENGTH_B": Anchor(
        key="GREEN_STRENGTH_B",
        value=4.0,
        paper_value=4.0,
        uncertainty=1.0,
        ref="PM iron-powder green-strength master curves",
        notes="Density exponent of the green-strength law; screening.",
    ),
    "GREEN_STRENGTH_REF_MPA": Anchor(
        key="GREEN_STRENGTH_REF_MPA",
        value=15.0,
        paper_value=15.0,
        uncertainty=5.0,
        ref="Reference green/crush strength at which the fines reference "
            "was measured (tumble/drop practice)",
        notes="fines(σ_ref) = FINES_REF_PCT by construction (ratio form).",
    ),
    "FINES_REF_PCT": Anchor(
        key="FINES_REF_PCT",
        value=1.5,
        paper_value=1.5,
        uncertainty=0.5,
        ref="Tumble/drop-attrition fines of a 15 MPa-class briquette "
            "(PM/DRI handling standards)",
        notes="Shipped fines at the reference strength; the ratio-form "
              "law scales against green strength.",
    ),
    "FINES_STRENGTH_EXP": Anchor(
        key="FINES_STRENGTH_EXP",
        value=1.0,
        paper_value=1.0,
        uncertainty=0.5,
        ref="SPECULATIVE — no published fines-vs-strength exponent for "
            "electrolytic iron briquettes; ratio-form screening",
        notes="The module's weakest constant; orders the handling-fines "
              "trade against density, nothing more.",
    ),
    "SPRINGBACK_MODULUS_EXP": Anchor(
        key="SPRINGBACK_MODULUS_EXP",
        value=3.0,
        paper_value=3.0,
        uncertainty=0.5,
        ref="Phani–Niyogi family modulus-density power laws for porous "
            "sintered compacts (m ≈ 3)",
        notes="E(D) = E_Fe·D^m drives the springback ε = P/E screen.",
    ),
    "DIE_WALL_MU": Anchor(
        key="DIE_WALL_MU",
        value=0.10,
        paper_value=0.10,
        uncertainty=0.05,
        ref="Zinc-stearate-lubricated die-wall friction, PM industry "
            "practice",
        notes="Ejection force fraction ≈ μ·k_radial.",
    ),
    "RADIAL_STRESS_FRACTION": Anchor(
        key="RADIAL_STRESS_FRACTION",
        value=0.30,
        paper_value=0.30,
        uncertainty=0.10,
        ref="Residual radial/axial stress ratio for uniaxial iron-powder "
            "die compaction (Poisson-based screening)",
        notes="Second anchor of the ejection-fraction estimate.",
    ),
    "HOT_PRESS_T_C": Anchor(
        key="HOT_PRESS_T_C",
        value=600.0,
        paper_value=600.0,
        uncertainty=100.0,
        ref="HBI hot-briquetting practice on DRI (~600–700 °C), Midrex "
            "family disclosure",
        notes="Hot die pressing is to the powder briquette what HBI is "
              "to DRI — the recommended Option-A line.",
    ),
    "HOT_PRESS_SIGMA_SOFTEN": Anchor(
        key="HOT_PRESS_SIGMA_SOFTEN",
        value=0.25,
        paper_value=0.25,
        uncertainty=0.10,
        ref="α-Fe flow-stress ratio at 600 °C vs RT, screening band from "
            "hot-compression data",
        notes="σ_y(T) = σ_y(ref)·soften(T); the whole hot-press leverage "
              "rides on this one ratio.",
    ),
    "SINTER_T_C": Anchor(
        key="SINTER_T_C",
        value=700.0,
        paper_value=700.0,
        uncertainty=100.0,
        ref="Optional hot dense-state sinter 600–800 °C (V6 §1.4 text)",
        notes="Reduction-sinter branch temperature.",
    ),
    "SINTER_KWH_PER_T": Anchor(
        key="SINTER_KWH_PER_T",
        value=150.0,
        paper_value=150.0,
        uncertainty=75.0,
        ref="Thermal duty (Cp·ΔT ≈ 84 kWh/t) + furnace/atmosphere "
            "overhead, screening",
        notes="Delivered energy of the sinter-first branch; the O-vs-energy "
              "trade against passivate-first.",
    ),
    "SINTER_STRENGTH_FACTOR": Anchor(
        key="SINTER_STRENGTH_FACTOR",
        value=10.0,
        paper_value=10.0,
        uncertainty=5.0,
        ref="Green → sintered bond strength multiplier, PM sintered-iron "
            "strength ratios",
        notes="Sinter fixes strength and fines, NOT density — the "
              "sink-and-size spec still gates cold-pressed lines.",
    ),
    "SINTER_RESIDUAL_O_WT_PCT": Anchor(
        key="SINTER_RESIDUAL_O_WT_PCT",
        value=0.20,
        paper_value=0.20,
        uncertainty=0.10,
        ref="H₂-reduction of nm-scale iron passive film at 600–800 °C, "
            "screening remnant",
        notes="Sinter-first residual O; the passivate-first branch takes "
              "the live product_oxidation pickup instead.",
    ),
    "DENSITY_REL_SPEC": Anchor(
        key="DENSITY_REL_SPEC",
        value=0.80,
        paper_value=0.80,
        uncertainty=0.05,
        ref="Sink-and-size screening target for EAF-feed briquettes "
            "(HBI ships ~0.65–0.70 relative density; margin above)",
        notes="Light briquettes float on slag and blow out with the fume; "
              "this is the physical floor, screening.",
    ),
    "CCS_SPEC_N": Anchor(
        key="CCS_SPEC_N",
        value=2500.0,
        paper_value=2500.0,
        uncertainty=500.0,
        ref="ISO 4700-family cold-crushing screening floor for shipped "
            "agglomerates",
        notes="Per-briquette crush force the logistics chain is assumed "
              "to accept; buyer to replace with their spec.",
    ),
    "FINES_SPEC_PCT": Anchor(
        key="FINES_SPEC_PCT",
        value=2.0,
        paper_value=2.0,
        uncertainty=1.0,
        ref="Buyer screening ceiling on shipped fines",
        notes="Fines are the pyrophoric fraction (§1.2) and the fume "
              "loss (§1.5); the ceiling prices them together.",
    ),
    "BRIQUETTE_SIZE_MM": Anchor(
        key="BRIQUETTE_SIZE_MM",
        value=40.0,
        paper_value=40.0,
        uncertainty=10.0,
        ref="Pillow-briquette characteristic dimension (DRI/HBI briquette "
            "sizing)",
        notes="Crush face = size²; bridging rule scales with the same "
              "dimension.",
    ),
    "UNCONFINED_YIELD_PA": Anchor(
        key="UNCONFINED_YIELD_PA",
        value=500.0,
        paper_value=500.0,
        uncertainty=300.0,
        ref="Jenike shear-tester band for slightly cohesive fine iron "
            "powder",
        notes="Cohesive strength of the loose feed in the rathole screen.",
    ),
    "JENIKE_H_THETA": Anchor(
        key="JENIKE_H_THETA",
        value=2.2,
        paper_value=2.2,
        uncertainty=0.3,
        ref="Jenike H(θ) ≈ 2.0–2.3 for conical hoppers in mass flow "
            "(Bull. 123 charts)",
        notes="Multiplier of the critical rathole diameter.",
    ),
    "MAGNETIC_COHESION_PA": Anchor(
        key="MAGNETIC_COHESION_PA",
        value=350.0,
        paper_value=350.0,
        uncertainty=300.0,
        ref="SPECULATIVE — order-of-B_r²/(2μ₀) pressure at ~30 mT contact "
            "induction between magnetised iron particles",
        notes="The V6 §1.4 ferromagnetic-agglomeration screening term; "
              "decade band, orders hopper design only.",
    ),
    "BRIDGING_RULE_MULTIPLE": Anchor(
        key="BRIDGING_RULE_MULTIPLE",
        value=7.0,
        paper_value=7.0,
        uncertainty=1.0,
        ref="Bunker-design rule: outlet ≥ 6–8 × particle dimension for "
            "coarse free-flowing bodies",
        notes="The briquetted product's flow floor (bridging, not "
              "ratholing).",
    ),
    # ── V6 §4.1 — Ti drum hydriding (ti_hydriding.py) ──
    "TI_H_D_60C_M2_S": Anchor(
        key="TI_H_D_60C_M2_S",
        value=5.0e-12,
        paper_value=5.0e-12,
        uncertainty=4.0e-12,
        ref="H diffusivity in α-Ti at 60 °C ~1e-12–1e-11 m²/s "
            "(McQuillan & classic Ti–H compilations)",
        notes="Fast interstitial diffusion; decade band.",
    ),
    "TI_H_D_EA_J_MOL": Anchor(
        key="TI_H_D_EA_J_MOL",
        value=27_000.0,
        paper_value=27_000.0,
        uncertainty=5_000.0,
        ref="Activation energy for H diffusion in α-Ti, ~26–30 kJ/mol "
            "(Ti–H compilations)",
        notes="Arrhenius slope for D_H about the 60 °C reference.",
    ),
    "TI_H_TSS_WT_PPM_60C": Anchor(
        key="TI_H_TSS_WT_PPM_60C",
        value=60.0,
        paper_value=60.0,
        uncertainty=40.0,
        ref="Terminal solid solubility of H in α-Ti at 60–90 °C, "
            "≲30–100 wt-ppm (Ti–H phase-diagram reviews)",
        notes="Crossing TSS precipitates δ-TiH₍₂₋ₓ₎ — the damage onset.",
    ),
    "TI_H_ENTRY_FRAC": Anchor(
        key="TI_H_ENTRY_FRAC",
        value=0.05,
        paper_value=0.05,
        uncertainty=0.045,
        ref="Cathodic-charging H entry fraction into Ti in acid service, "
            "order 1–10 % (screening; Ti corrosion/hydriding literature)",
        notes="Weakest link of the chain together with TI_H_SHIELD_FRAC; "
              "the module reports the verdict across the band.",
    ),
    "TI_H_SHIELD_FRAC": Anchor(
        key="TI_H_SHIELD_FRAC",
        value=0.01,
        paper_value=0.01,
        uncertainty=0.009,
        ref="SPECULATIVE — fraction of drum area exposed to bath H "
            "(pinholes, strip edges, peel front); the deposit shields "
            "the rest and is itself an H sink",
        notes="Auditable on a real drum (pinhole/edge inspection); "
              "paired with TI_H_ENTRY_FRAC as the design target product.",
    ),
    "TI_HYD_H_PER_TI": Anchor(
        key="TI_HYD_H_PER_TI",
        value=1.7,
        paper_value=1.7,
        uncertainty=0.2,
        ref="δ-TiH₍₂₋ₓ₎ stoichiometry boundary composition, H/Ti ≈ 1.5–1.9 "
            "(Ti–H phase diagram)",
        notes="Case inventory per unit hydride volume.",
    ),
    "TI_HYD_CRIT_CASE_UM": Anchor(
        key="TI_HYD_CRIT_CASE_UM",
        value=20.0,
        paper_value=20.0,
        uncertainty=15.0,
        ref="SPECULATIVE — hydride case depth at which the TiO₂ scale "
            "spalls / peel morphology flips (foil-machine drum-service "
            "practice, re-skin intervals)",
        notes="Stands in for the shelved stress/K_IC/buckling mechanics; "
              "calibrate against first drum campaign.",
    ),
    "TI_HYD_GC_FLOOR_FRAC": Anchor(
        key="TI_HYD_GC_FLOOR_FRAC",
        value=0.4,
        paper_value=0.4,
        uncertainty=0.2,
        ref="SPECULATIVE — residual G_c fraction of a fully hydrated/"
            "scale-spalled interface (adhesion_peel practice analogy)",
        notes="Floor of the hydride G_c drift multiplier.",
    ),

    # ─── Operational-pH metrology (V6 §5.1) ────────────────────────
    "PH_METROLOGY_OVERLAP_M": Anchor(
        key="PH_METROLOGY_OVERLAP_M", value=1.0e-3, paper_value=1.0e-3,
        uncertainty=8.0e-4,
        ref="SPECULATIVE — diffuse bridge/bath overlap concentration floor "
            "for Planck–Henderson screening",
        notes="Numerical regularizer standing in for bridge mixing and ion pairing; "
              "replace with the HCl/LiCl concentration-cell check.",
    ),
    "PH_METROLOGY_DRIFT_MV_DAY": Anchor(
        key="PH_METROLOGY_DRIFT_MV_DAY", value=3.0, paper_value=3.0,
        uncertainty=2.0,
        ref="SPECULATIVE — Fe(OH)3-colloid bridge-clogging drift scale "
            "(glass-electrode service practice)",
        notes="Coefficient of log(1 + bridge age / day), sign must be measured.",
    ),
    "PH_METROLOGY_SINGLE_ION_OFFSET_PH": Anchor(
        key="PH_METROLOGY_SINGLE_ION_OFFSET_PH", value=0.30, paper_value=0.30,
        uncertainty=0.20,
        ref="IUPAC pH convention / Bates pH metrology; 0.2–0.5 pH-unit "
            "high-ionic-strength single-ion convention band",
        notes="Central operational-to-Pitzer convention offset, not a universal constant.",
    ),
    "PH_METROLOGY_JUNCTION_SIGMA_MV": Anchor(
        key="PH_METROLOGY_JUNCTION_SIGMA_MV", value=20.0, paper_value=20.0,
        uncertainty=10.0,
        ref="Henderson/Planck liquid-junction-potential practice: 10–40 mV "
            "concentrated-brine band",
        notes="One-sigma screening uncertainty carried into pH correction.",
    ),
    "PH_METROLOGY_SINGLE_ION_SIGMA_PH": Anchor(
        key="PH_METROLOGY_SINGLE_ION_SIGMA_PH", value=0.20, paper_value=0.20,
        uncertainty=0.10,
        ref="IUPAC pH convention / Bates pH metrology",
        notes="Independent convention-component uncertainty, in pH units.",
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
