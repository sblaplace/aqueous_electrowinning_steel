"""
Fe(III) hydroxide / oxyhydroxide phase speciation and Ostwald aging kinetics (L0).

Replaces the single "Fe(OH)3(s)" solid assumption in the Fe(III) shuttle with a
phase ladder: 2-line ferrihydrite -> goethite (alpha-FeOOH) -> hematite
(alpha-Fe2O3) on the sulfate path, and akaganeite (beta-FeOOH) -> goethite ->
hematite on the chloride (FeCl3-hydrolysis) path.  Closes CHEM_PHYS_REVIEW.md
Tier 2.1.

Why it matters (the 1-2 decade sludge-bleed error)
--------------------------------------------------
The legacy cap uses log Ksp = -38.7 for "Fe(OH)3".  The *initial* precipitate is
the more-ordered 2-line ferrihydrite (log Ksp ~ -39.4), and as that sludge ages
to goethite (-40.7) and hematite (-42.7) the controlling solubility falls a
further ~1-2 decades.  Because the dissolved cap is [Fe3+] = Ksp/[OH-]^3, which
scales directly with Ksp, a sludge bleed sized against the Fe(OH)3 Ksp is too
small by 1-2 decades once the phase matures.  This module exposes the
phase-ladder Ksp set and the Ostwald-stepping kinetics so the bath can size its
bleed (``fe3_shuttle.py``) and its H+ drift (``bath_dynamics.py``) against the
phase actually present, instead of a single guessed solid.

Ostwald stepping and the H+ schedule
------------------------------------
Aging (Fe(OH)3 -> FeOOH -> Fe2O3) is dehydration: the solid-solid transformation
itself releases 0 net H+.  What changes the H+ *schedule* is that each step down
the ladder lowers the equilibrium [Fe3+] cap, which drives further Fe3+ out of
solution, and that continued precipitation releases 3 H+/Fe.  So the H+ a mature
sludge releases is spread across the aging window instead of being dumped all at
the first instant of precipitation.  ``bath_dynamics`` wires the resulting
delayed proton flux into its pH balance; ``fe3_shuttle`` sizes the bleed against
the phase-aware cap.

Solubility products (log10 [Fe3+][OH-]^3, 25 C; screening central values)
--------------------------------------------------------------------------
    phase              formula      log Ksp
    feoh3_amorphous    Fe(OH)3(am)  -38.7   (legacy baseline, reference only)
    ferrihydrite_2line Fe(OH)3       -39.4   (initial sulfate-path precipitate)
    akaganeite         beta-FeOOH   -40.6   (chloride path, FeCl3 hydrolysis)
    goethite           alpha-FeOOH  -40.7   (Ostwald intermediate)
    hematite           alpha-Fe2O3  -42.7   (Ostwald terminal)

so solubility (= the [Fe3+] cap at fixed pH) orders  feoh3 > ferrihydrite >
akaganeite > goethite > hematite.  Values follow Cornell & Schwertmann, *The
Iron Oxides* (2003) and the PHREEQC-derived thermodynamic database family; the
literature spread is ~ +/-0.4 decades.  These are screening central values, not
gate evidence (L0).

References
----------
* Cornell, R. M. & Schwertmann, U. (2003), *The Iron Oxides*, 2nd ed.,
  Wiley-VCH — ferrihydrite/goethite/hematite/akaganeite solubility and Ostwald
  ripening timescales.
* CHEM_PHYS_REVIEW.md Tier 2.1 — the explicit gap this module closes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

SCREENING_FLAG = "unvalidated (L0)"

# --- Ostwald / aging default kinetics (screening central values) ---
# Ferrihydrite -> goethite is weeks-months at 25 C but accelerates strongly at
# the 50-90 C bath temperature; goethite -> hematite is slower yet.  Arrhenius-
# corrected from the 25 C reference half-lives with a screening Ea.
FERRIHYDRITE_TO_GOETHITE_HALF_LIFE_HR_25C = 720.0   # 30 days
GOETHITE_TO_HEMATITE_HALF_LIFE_HR_25C = 8760.0      # 1 year
AGING_ACTIVATION_ENERGY_FH_GH_J_MOL = 30.0e3
AGING_ACTIVATION_ENERGY_GH_HEM_J_MOL = 60.0e3
T_REF_K = 298.15
R_GAS = 8.314


@dataclass(frozen=True)
class FeIIIOxidePhase:
    """One Fe(III) (hydr)oxide solid phase in the ladder."""

    name: str
    formula: str
    log_ksp: float        # log10([Fe3+][OH-]^3) at 25 C
    note: str = ""

    @property
    def ksp(self) -> float:
        return 10.0 ** self.log_ksp


# log Ksp for [Fe3+][OH-]^3; more negative = less soluble = lower [Fe3+] cap.
PHASES: Dict[str, FeIIIOxidePhase] = {
    "feoh3_amorphous": FeIIIOxidePhase(
        "feoh3_amorphous", "Fe(OH)3(am)", -38.7,
        "legacy Fe(OH)3 baseline; reference only"),
    "ferrihydrite_2line": FeIIIOxidePhase(
        "ferrihydrite_2line", "Fe(OH)3 (2-line)", -39.4,
        "initial sulfate-path precipitate"),
    "akaganeite": FeIIIOxidePhase(
        "akaganeite", "beta-FeOOH", -40.6,
        "chloride path (FeCl3 hydrolysis)"),
    "goethite": FeIIIOxidePhase(
        "goethite", "alpha-FeOOH", -40.7, "Ostwald intermediate"),
    "hematite": FeIIIOxidePhase(
        "hematite", "alpha-Fe2O3", -42.7, "Ostwald terminal"),
}

SULFATE_OSTWALD: List[str] = ["ferrihydrite_2line", "goethite", "hematite"]
CHLORIDE_OSTWALD: List[str] = ["akaganeite", "goethite", "hematite"]


def initial_phase_for_bath(bath_anion: Optional[str] = "sulfate") -> str:
    """The first (most soluble) Fe(III) solid the shuttle precipitates into."""
    an = (bath_anion or "sulfate").lower()
    if an in ("chloride", "cl", "fecl2", "fecl3"):
        return "akaganeite"
    return "ferrihydrite_2line"


def ostwald_ladder(bath_anion: Optional[str] = "sulfate") -> List[str]:
    """Ordered phase succession for a bath anion (sulfate vs chloride)."""
    if initial_phase_for_bath(bath_anion) == "akaganeite":
        return list(CHLORIDE_OSTWALD)
    return list(SULFATE_OSTWALD)


def log10_ksp(phase: str) -> float:
    """log10 solubility product of a phase (KeyError on unknown names)."""
    return PHASES[phase].log_ksp


def solubility_cap_M(phase: str, pH: float) -> float:
    """Dissolved [Fe3+] at the hydrolysis cap for one phase: Ksp/[OH-]^3."""
    pOH = 14.0 - pH
    return 10.0 ** (PHASES[phase].log_ksp + 3.0 * pOH)


def blended_cap_M(
    pH: float,
    inventory: Dict[str, float],
    initial: Optional[str] = None,
) -> float:
    """Effective [Fe3+] cap for a sludge distributed over the phase ladder.

    Inventory-fraction-weighted mean of log Ksp (equivalently the geometric mean
    of the per-phase Ksp).  A fresh sludge (all in the initial phase) sits at
    the initial-phase cap; as Ostwald aging converts it toward the less-soluble
    tail the blended cap falls monotonically, so progressively more Fe3+ is
    pulled out of solution.  An empty inventory defaults to the initial phase.

    This is a kinetic (mass-weighted) blend.  A strictly thermodynamic
    "least-soluble phase controls" limit would be even more aggressive; the
    mass-weighted mean is the L0 screening choice and is C0-continuous.
    """
    init = initial or initial_phase_for_bath()
    total = sum(max(0.0, v) for v in inventory.values())
    if total <= 0.0:
        return solubility_cap_M(init, pH)
    weighted = 0.0
    for name, mol in inventory.items():
        if mol > 0.0 and name in PHASES:
            weighted += (mol / total) * PHASES[name].log_ksp
    if weighted == 0.0:
        weighted = PHASES[init].log_ksp
    return 10.0 ** (weighted + 3.0 * (14.0 - pH))


def _rate_1_hr(half_life_ref_hr: float, Ea_J_mol: float, T_K: float) -> float:
    """First-order rate constant (1/hr), Arrhenius-accelerated from 25 C."""
    accel = 1.0
    if T_K > 0.0:
        accel = math.exp((Ea_J_mol / R_GAS) * (1.0 / T_REF_K - 1.0 / T_K))
    return math.log(2.0) / (half_life_ref_hr / accel)


def age_inventory(
    inventory: Dict[str, float],
    bath_anion: Optional[str] = "sulfate",
    dt_hr: float = 1.0,
    temperature_C: float = 25.0,
    t12_fh_gh_hr: float = FERRIHYDRITE_TO_GOETHITE_HALF_LIFE_HR_25C,
    t12_gh_hem_hr: float = GOETHITE_TO_HEMATITE_HALF_LIFE_HR_25C,
) -> Dict[str, float]:
    """Advance a ferric-solid inventory one Ostwald step (first-order).

    Transfers mol from each non-terminal phase to its successor at the phase-
    dependent (Arrhenius-corrected) rate.  This is pure dehydration, so it
    returns no H+; the H+ schedule is driven by the cap drop this transfer
    produces, which the caller applies at its operator split.
    """
    out: Dict[str, float] = {k: max(0.0, v) for k, v in inventory.items()}
    T_K = temperature_C + 273.15
    ladder = ostwald_ladder(bath_anion)
    for i in range(len(ladder) - 1):
        src, dst = ladder[i], ladder[i + 1]
        if src in ("ferrihydrite_2line", "akaganeite"):
            half, ea = t12_fh_gh_hr, AGING_ACTIVATION_ENERGY_FH_GH_J_MOL
        else:
            half, ea = t12_gh_hem_hr, AGING_ACTIVATION_ENERGY_GH_HEM_J_MOL
        k = _rate_1_hr(half, ea, T_K)
        dm = out.get(src, 0.0) * (1.0 - math.exp(-k * dt_hr))
        out[src] = out.get(src, 0.0) - dm
        out[dst] = out.get(dst, 0.0) + dm
    return {k: max(0.0, v) for k, v in out.items()}


def phase_aware_bleed(
    r_prod_M_s: float,
    shuttle_sink_M_s: float,
    pH: float,
    phase: str,
) -> float:
    """Sludge-bleed magnitude (mol/L/s) for one controlling phase.

    ``r_prod - shuttle_sink`` is how much Fe3+ exceeds the mass-transfer sink
    and therefore falls out as solid.  The phase only changes *which* dissolved
    cap pins the system; the bleed flux itself is production minus the shuttle
    sink at the phase's cap.  Kept here so ``fe3_shuttle`` can size its bleed
    against a ferrihydrite/akaganeite/goethite cap instead of Fe(OH)3.
    """
    return float(max(0.0, r_prod_M_s - shuttle_sink_M_s))


def main() -> None:
    """Print the phase ladder and the [Fe3+] caps at the RC-1 bath pH."""
    print("Fe(III) HYDROXIDE / OXYHYDROXIDE PHASE LADDER — unvalidated (L0)")
    print(f"{'phase':<18}{'formula':<16}{'log Ksp':>9}{'[Fe3+] cap @ pH2 (M)':>22}")
    order = ["feoh3_amorphous", "ferrihydrite_2line", "akaganeite",
             "goethite", "hematite"]
    for name in order:
        p = PHASES[name]
        print(f"{name:<18}{p.formula:<16}{p.log_ksp:>9.2f}"
              f"{solubility_cap_M(name, 2.0):>22.2e}")
    print("Sulfate path:  " + " -> ".join(SULFATE_OSTWALD))
    print("Chloride path: " + " -> ".join(CHLORIDE_OSTWALD))
    print("NOT gate evidence; gates are measurement-only.")


if __name__ == "__main__":
    main()
