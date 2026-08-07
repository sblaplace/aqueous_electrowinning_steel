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
