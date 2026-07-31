"""
Cell architecture screen — areal productivity and $/m² for candidate reactor types.

Why this module exists
----------------------
The program's own framing (``docs/RESEARCH_PROGRAM.md``) states the binding
constraint plainly: electricity is *not* the problem, **CAPEX per m² of
installed cell is**.  At 100 mA/cm² and 85% FE an electrowinning cell makes
roughly 7 t of iron per m² of electrode per year.  A zinc tankhouse costs
~$1,000–1,500 per annual tonne of capacity to build — for a product worth
$2,500–3,000/t.  Iron is worth $400–600/t.  So the program needs roughly
**5× a zinc tankhouse's areal productivity, or 5× cheaper cells, or some
product of the two.**

That statement was previously untested by any code.  ``technoeconomic.py``
hard-codes a flat ``$150 + $80 + $100`` per m² with no dependence on how the
cell is actually built or how the solid product gets removed from it.  Kill
criterion #3 — *"cost per m² of a cell that can be stripped continuously >
threshold → pivot to cell architecture work"* — had no computable threshold.

This module supplies the missing link:

1. **Mass transport per architecture.**  Each reactor type gets a literature
   Sherwood correlation, ``Sh = a·Re^b·Sc^c``, evaluated at its own
   characteristic length and velocity.  That fixes the transport-limited
   current density, and therefore the ceiling on areal productivity.
2. **A practical current ceiling.**  Transport is not always the binding
   limit.  Fluidized/particulate beds are limited by *potential distribution*
   through the bed depth, not by film transport; drums are limited by foil
   handling.  Each spec carries an explicit ``max_practical_j_A_m2`` so the
   model never reports a transport limit the hardware cannot use.
3. **Harvesting continuity.**  A batch cell must stop to be stripped.  The
   faster it plates, the more often it stops.  That feedback — high j causing
   *more* downtime — is what makes continuous harvesting valuable, and it is
   modelled explicitly rather than assumed away.
4. **Cost per annual tonne, and the inverse kill criterion.**
   ``max_affordable_cost_per_m2`` answers the question the program actually
   needs answered: given a capital-charge budget in $/t Fe, how expensive is
   the cell allowed to be?

Scope and honesty
-----------------
This is a **screening** model, at the same evidence tier as ``hull_cell.py``
or ``scale_up.py``.  The Sherwood correlations are literature values measured
in *other* chemistries (ferricyanide, copper sulfate) and transferred here.
The cost figures are engineering estimates, not quotes.  Every architecture
carries an ``evidence_level`` recording whether the configuration is
commercially operated, piloted, demonstrated at lab scale, or a concept.
Nothing in this module is wet-lab data for iron.

Mass-transfer correlations
--------------------------
================  ===============================  =========================
Architecture      Correlation                      Source
================  ===============================  =========================
Parallel plate    Sh = 0.023 Re^0.80 Sc^0.33       Turbulent duct (Chilton–
                                                   Colburn); FM01-LC empirical
                                                   values 0.18–0.24 Re^0.7
                                                   bracket this at lower Re.
Rotating cylinder Sh = 0.0791 Re^0.70 Sc^0.356     Eisenberg, Tobias & Wilke
                                                   (1954) J. Electrochem. Soc.
                                                   101, 306.
Drum / foil       Sh = 0.0791 Re^0.70 Sc^0.356     ETW applied to the rotating
                                                   drum, derated by immersion.
Moving belt       Sh = 0.023 Re^0.80 Sc^0.33       Duct flow with a moving
                                                   wall; conservative.
Fluidized bed     Sh = 2 + 0.6 Re^0.50 Sc^0.33     Ranz–Marshall particle
                                                   correlation.
================  ===============================  =========================

References
----------
- Eisenberg, M., Tobias, C.W., Wilke, C.R. (1954) *Ionic mass transfer and
  concentration polarization at rotating electrodes*, J. Electrochem. Soc.
  101(6), 306.
- Walsh, F.C., Ponce de León, C. (2018) *Progress in electrochemical flow
  reactors*, Electrochim. Acta — RCE and filter-press reactor review.
- Brown, C.J., Pletcher, D., Walsh, F.C. et al. — FM01-LC filter-press
  reactor mass transport, Sh = 0.22 Re^0.71 Sc^0.33 (200 < Re < 1000).
- Ranz, W.E., Marshall, W.R. (1952) *Evaporation from drops*, Chem. Eng. Prog.
- Electrodeposited copper foil practice: 30–120 A/dm² on 1–3 m titanium drum
  cathodes, 5–30 m/min surface speed (industrial practice, patent literature).

See ``models/README.md`` for the model-scope contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from .electrochemistry import FARADAY, M_FE, Z_FE

# ─── Electrolyte defaults ────────────────────────────────────────────
# Consistent with models/scale_up.py and models/kinetics.py.
DEFAULT_DIFFUSIVITY_M2_S = 7.2e-10   # Fe²⁺ in aqueous sulfate
DEFAULT_KINEMATIC_VISCOSITY = 1.0e-6  # m²/s, dilute aqueous at ~50 °C
DEFAULT_FE_CONC_M = 1.0              # mol/L
DEFAULT_TEMPERATURE_C = 50.0
DEFAULT_FARADAIC_EFFICIENCY = 0.85

# ─── Benchmark: zinc tankhouse ───────────────────────────────────────
# The comparator named in docs/RESEARCH_PROGRAM.md.  A zinc tankhouse runs
# ~500 A/m² and costs ~$1,000–1,500 per annual tonne of capacity, making a
# product worth $2,500–3,000/t.  These are industry figures, not our model.
ZINC_TANKHOUSE = {
    "current_density_A_m2": 500.0,
    "capex_per_annual_tonne_low": 1000.0,
    "capex_per_annual_tonne_high": 1500.0,
    "product_value_per_t_low": 2500.0,
    "product_value_per_t_high": 3000.0,
    "note": (
        "Industry benchmark quoted in docs/RESEARCH_PROGRAM.md. Zinc "
        "electrowinning is the closest mature analogue: aqueous sulfate, "
        "insoluble anodes, periodic manual/robotic cathode stripping."
    ),
}

# Iron product value bracket for the feedstock (Option A) path.
IRON_PRODUCT_VALUE_PER_T = {"low": 400.0, "high": 600.0}

VelocityKind = Literal["channel_flow", "peripheral", "belt_speed", "superficial"]
HarvestMode = Literal["batch", "semi_continuous", "continuous"]
EvidenceLevel = Literal["commercial", "pilot", "lab", "concept"]


# ═════════════════════════════════════════════════════════════════════
#  Dimensionless groups
# ═════════════════════════════════════════════════════════════════════

def schmidt_number(
    kinematic_viscosity_m2_s: float = DEFAULT_KINEMATIC_VISCOSITY,
    diffusivity_m2_s: float = DEFAULT_DIFFUSIVITY_M2_S,
) -> float:
    """Sc = ν / D — momentum vs. mass diffusivity.

    For Fe²⁺ in aqueous sulfate this is ~1,400: momentum diffuses ~1,400×
    faster than iron, so the concentration boundary layer is much thinner
    than the hydrodynamic one.
    """
    if diffusivity_m2_s <= 0:
        raise ValueError("diffusivity must be positive")
    return kinematic_viscosity_m2_s / diffusivity_m2_s


def reynolds_number(
    velocity_m_s: float,
    characteristic_length_m: float,
    kinematic_viscosity_m2_s: float = DEFAULT_KINEMATIC_VISCOSITY,
) -> float:
    """Re = u·L / ν."""
    if kinematic_viscosity_m2_s <= 0:
        raise ValueError("kinematic viscosity must be positive")
    return abs(velocity_m_s) * characteristic_length_m / kinematic_viscosity_m2_s


def sherwood_number(
    Re: float,
    Sc: float,
    a: float,
    b: float,
    c: float,
    additive: float = 0.0,
) -> float:
    """Sh = additive + a·Re^b·Sc^c.

    ``additive`` carries the stagnant-limit term (2.0) of the Ranz–Marshall
    particle correlation; it is zero for the duct and cylinder correlations.
    """
    if Re < 0 or Sc <= 0:
        raise ValueError("Re must be non-negative and Sc positive")
    return additive + a * (Re ** b) * (Sc ** c)


def mass_transfer_coefficient(
    Sh: float,
    characteristic_length_m: float,
    diffusivity_m2_s: float = DEFAULT_DIFFUSIVITY_M2_S,
) -> float:
    """k_m = Sh·D / L  (m/s)."""
    if characteristic_length_m <= 0:
        raise ValueError("characteristic length must be positive")
    return Sh * diffusivity_m2_s / characteristic_length_m


def limiting_current_from_km(
    k_m_m_s: float,
    concentration_mol_L: float = DEFAULT_FE_CONC_M,
    z: int = Z_FE,
) -> float:
    """i_lim = z·F·k_m·C  (A/m²), with C supplied in mol/L."""
    return z * FARADAY * k_m_m_s * concentration_mol_L * 1000.0


# ═════════════════════════════════════════════════════════════════════
#  Productivity primitives
# ═════════════════════════════════════════════════════════════════════

def areal_productivity_t_m2_yr(
    j_A_m2: float,
    faradaic_efficiency: float = DEFAULT_FARADAIC_EFFICIENCY,
    capacity_factor: float = 1.0,
    hours_per_year: float = 8760.0,
) -> float:
    """Annual iron output per m² of active electrode area (t/(m²·yr)).

    ṁ = j·FE·M / (z·F)  in kg/(m²·s); integrate over the operated hours.

    At 1,000 A/m² (100 mA/cm²), FE = 0.85 and full-year operation this
    returns ~7.8 t/(m²·yr), matching the figure quoted in the program doc.
    """
    if j_A_m2 < 0:
        raise ValueError("current density must be non-negative")
    kg_per_m2_s = j_A_m2 * faradaic_efficiency * M_FE / (Z_FE * FARADAY)
    return kg_per_m2_s * 3600.0 * hours_per_year * capacity_factor / 1000.0


def deposition_rate_um_hr(
    j_A_m2: float,
    faradaic_efficiency: float = DEFAULT_FARADAIC_EFFICIENCY,
    density_kg_m3: float = 7874.0,
) -> float:
    """Linear growth rate of the deposit (µm/hr)."""
    kg_per_m2_s = j_A_m2 * faradaic_efficiency * M_FE / (Z_FE * FARADAY)
    return kg_per_m2_s / density_kg_m3 * 3600.0 * 1e6


def capital_recovery_factor(discount_rate: float = 0.08, lifetime_yr: int = 25) -> float:
    """CRF = r(1+r)^n / ((1+r)^n − 1).

    Matches the convention in ``models/technoeconomic.py``.
    """
    if lifetime_yr <= 0:
        raise ValueError("lifetime must be positive")
    if discount_rate <= 0:
        return 1.0 / lifetime_yr
    f = (1.0 + discount_rate) ** lifetime_yr
    return discount_rate * f / (f - 1.0)


# ═════════════════════════════════════════════════════════════════════
#  Architecture specification
# ═════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ArchitectureSpec:
    """One candidate cell architecture.

    Cost fields are **direct equipment** costs per m² of *installed* (costed)
    electrode area.  Balance-of-plant, infrastructure, engineering and
    contingency are applied later via ``installed_cost_factor`` so the buildup
    stays comparable to ``technoeconomic.CAPEXModel``.

    Attributes
    ----------
    active_area_ratio
        Active electrochemical area per unit of costed electrode area.  1.0
        for a flat plate; ~0.5 for a drum (only the immersed arc plates);
        >>1 for a packed/fluidized bed, where the particle surface area far
        exceeds the cell footprint being paid for.
    max_practical_j_A_m2
        Engineering ceiling on the *active-area* current density, independent
        of film transport — foil handling speed, bubble management, or
        product quality.  The operating point is ``min(transport limit, this)``.
    max_footprint_current_A_m2
        Ceiling on current per unit of *costed* area.  This is what actually
        binds three-dimensional electrodes: a fluidized bed has ~600 m² of
        particle area per m² of footprint, but ohmic potential drop through
        the bed depth means only a fraction is polarized usefully.  Expressed
        per footprint it is equivalent to a volumetric limit in A/m³ of bed.
        ``None`` means no separate footprint ceiling applies.
    flow_enhancement_factor
        Multiplier on ``k_m`` for transport mechanisms the base correlation
        does not capture — forced/impinging electrolyte injection, turbulence
        promoters, or gas sparging.  1.0 means the correlation stands alone.
    transport_utilization
        Fraction of the transport limit that can be run without the deposit
        degrading.  Architectures whose *product* is powder can exceed 1.0,
        because for them dendritic/powdery growth is the goal, not a defect.
    """

    id: str
    name: str
    product_form: str
    harvest_mode: HarvestMode
    evidence_level: EvidenceLevel

    # Mass transfer correlation: Sh = additive + a·Re^b·Sc^c
    sh_coefficient: float
    sh_re_exponent: float
    sh_sc_exponent: float
    sh_additive: float = 0.0

    # Hydrodynamics
    characteristic_length_m: float = 0.02
    velocity_kind: VelocityKind = "channel_flow"
    default_velocity_m_s: float = 0.1
    flow_enhancement_factor: float = 1.0

    # Area accounting
    active_area_ratio: float = 1.0

    # Operating ceilings
    max_practical_j_A_m2: float = 5000.0
    max_footprint_current_A_m2: Optional[float] = None
    transport_utilization: float = 0.7

    # Direct equipment cost buildup ($/m² of costed electrode area)
    electrode_cost_per_m2: float = 150.0
    separator_cost_per_m2: float = 80.0
    hardware_cost_per_m2: float = 100.0
    harvesting_cost_per_m2: float = 0.0

    # Harvest cycle
    harvest_downtime_hr: float = 0.0        # per harvest event (batch only)
    target_deposit_thickness_um: float = 500.0
    base_availability: float = 0.92          # excludes harvest downtime

    # Qualitative
    notes: str = ""
    limitations: List[str] = field(default_factory=list)

    @property
    def direct_cost_per_m2(self) -> float:
        """Sum of the direct equipment cost lines ($/m²)."""
        return (
            self.electrode_cost_per_m2
            + self.separator_cost_per_m2
            + self.hardware_cost_per_m2
            + self.harvesting_cost_per_m2
        )

    @property
    def is_continuous(self) -> bool:
        return self.harvest_mode == "continuous"

    def cost_breakdown(self) -> Dict[str, float]:
        return {
            "electrodes_per_m2": self.electrode_cost_per_m2,
            "separator_per_m2": self.separator_cost_per_m2,
            "hardware_per_m2": self.hardware_cost_per_m2,
            "harvesting_per_m2": self.harvesting_cost_per_m2,
            "direct_total_per_m2": self.direct_cost_per_m2,
        }


# ═════════════════════════════════════════════════════════════════════
#  The candidate set
# ═════════════════════════════════════════════════════════════════════
#
# Costs are engineering estimates on a 2024–2026 USD basis.  The plate-and-
# frame line matches technoeconomic.CAPEXModel defaults exactly (150/80/100)
# so this module does not silently move the baseline.

ARCHITECTURES: Dict[str, ArchitectureSpec] = {
    "plate_and_frame": ArchitectureSpec(
        id="plate_and_frame",
        name="Plate-and-frame (filter press), batch strip",
        product_form="plate_or_foil",
        harvest_mode="batch",
        evidence_level="commercial",
        sh_coefficient=0.023, sh_re_exponent=0.80, sh_sc_exponent=0.33,
        characteristic_length_m=0.02,   # hydraulic diameter ≈ 2 × 10 mm gap
        velocity_kind="channel_flow",
        default_velocity_m_s=0.1,
        active_area_ratio=1.0,
        max_practical_j_A_m2=6000.0,    # chlor-alkali/PEM stacks reach this
        transport_utilization=0.7,
        electrode_cost_per_m2=150.0,
        separator_cost_per_m2=80.0,
        hardware_cost_per_m2=100.0,
        harvesting_cost_per_m2=0.0,     # manual/robotic strip, not in $/m²
        harvest_downtime_hr=4.0,
        target_deposit_thickness_um=500.0,
        notes=(
            "The zinc/copper tankhouse geometry and the chlor-alkali stack. "
            "Mature, cheap per m², and the reference case in "
            "technoeconomic.CAPEXModel. Its weakness is harvesting: the cell "
            "must be opened and the cathode stripped."
        ),
        limitations=[
            "Batch stripping forces downtime that scales with plating rate",
            "Edge effects and current non-uniformity grow with panel size",
            "Gas management at high j is a real constraint in a narrow gap",
        ],
    ),
    "rotating_cylinder": ArchitectureSpec(
        id="rotating_cylinder",
        name="Rotating cylinder electrode with scraper (Eco-Cell type)",
        product_form="powder_or_particle",
        harvest_mode="continuous",
        evidence_level="commercial",
        sh_coefficient=0.0791, sh_re_exponent=0.70, sh_sc_exponent=0.356,
        characteristic_length_m=0.10,   # cylinder diameter
        velocity_kind="peripheral",
        default_velocity_m_s=1.0,       # ~190 rpm on a 100 mm cylinder
        active_area_ratio=1.0,
        max_practical_j_A_m2=8000.0,
        transport_utilization=1.2,      # powder is the product, not a defect
        electrode_cost_per_m2=200.0,
        separator_cost_per_m2=80.0,
        hardware_cost_per_m2=450.0,     # rotating seals, drive, bearings
        harvesting_cost_per_m2=150.0,   # scraper, slurry handling
        harvest_downtime_hr=0.0,
        target_deposit_thickness_um=50.0,
        notes=(
            "Turbulent RCE mass transfer (Eisenberg-Tobias-Wilke) is roughly "
            "6x a parallel plate at comparable pumping, and the scraper makes "
            "harvesting continuous. Commercially proven for metal recovery. "
            "Produces powder/flake, which suits the feedstock path (Option A) "
            "and rules it out for near-net-shape product."
        ),
        limitations=[
            "Rotating machinery in acid: seals and bearings are the wear item",
            "Product is powder/flake only",
            "$/m² is dominated by mechanical hardware, not electrodes",
        ],
    ),
    "drum_and_strip": ArchitectureSpec(
        id="drum_and_strip",
        name="Rotating drum with continuous foil strip (Cu-foil type)",
        product_form="plate_or_foil",
        harvest_mode="continuous",
        evidence_level="commercial",
        sh_coefficient=0.0791, sh_re_exponent=0.70, sh_sc_exponent=0.356,
        characteristic_length_m=2.0,    # drum diameter, industrial scale
        velocity_kind="peripheral",
        default_velocity_m_s=0.25,      # 15 m/min surface speed
        # Industrial foil drums do not rely on rotation for transport: they
        # pump 500-2000 L/min of electrolyte into the drum/anode gap. Rotation
        # alone cannot support the 300-1200 mA/cm² these machines actually run.
        flow_enhancement_factor=6.0,
        active_area_ratio=0.5,          # roughly half the drum is immersed
        max_practical_j_A_m2=12000.0,   # Cu foil drums run 30-120 A/dm²
        transport_utilization=0.8,
        electrode_cost_per_m2=600.0,    # precision Ti drum, ground and polished
        separator_cost_per_m2=80.0,
        hardware_cost_per_m2=500.0,     # drive, tank, current collection
        harvesting_cost_per_m2=250.0,   # peel, wind, tension control
        harvest_downtime_hr=0.0,
        target_deposit_thickness_um=25.0,
        notes=(
            "The electrodeposited copper foil machine, adapted. Industrially "
            "proven at 300-1200 mA/cm² with continuous peel, which is exactly "
            "the combination the program asks for. The open question is "
            "whether iron peels: Cu foil relies on a passive TiO2 release "
            "layer, and Fe adhesion on titanium is not characterised here."
        ),
        limitations=[
            "Peelability of iron from the drum is UNVERIFIED - the gating risk",
            "High $/m²: precision drum plus winding line",
            "Thin foil only; thickness is set by drum speed",
            "Hydrogen in the deposit may embrittle the foil during winding",
        ],
    ),
    "moving_belt": ArchitectureSpec(
        id="moving_belt",
        name="Moving-belt cathode with doctor blade",
        product_form="flake",
        harvest_mode="continuous",
        evidence_level="concept",
        sh_coefficient=0.023, sh_re_exponent=0.80, sh_sc_exponent=0.33,
        characteristic_length_m=0.02,
        velocity_kind="belt_speed",
        default_velocity_m_s=0.15,
        active_area_ratio=0.7,          # return path is not plating
        max_practical_j_A_m2=5000.0,
        transport_utilization=0.7,
        electrode_cost_per_m2=250.0,
        separator_cost_per_m2=80.0,
        hardware_cost_per_m2=350.0,
        harvesting_cost_per_m2=200.0,
        harvest_downtime_hr=0.0,
        target_deposit_thickness_um=100.0,
        notes=(
            "Conceptually the cheapest route to continuous harvesting in a "
            "planar geometry: plate on a belt, scrape it off at the return "
            "roller. No iron-specific demonstration is known to this program. "
            "Belt tracking, edge sealing and current collection through a "
            "moving contact are the unsolved engineering."
        ),
        limitations=[
            "No known iron demonstration - concept tier",
            "Sliding current collection at high current is unproven here",
            "Belt sealing against the electrolyte compartment is difficult",
        ],
    ),
    "fluidized_bed": ArchitectureSpec(
        id="fluidized_bed",
        name="Fluidized / particulate bed cathode",
        product_form="powder_or_particle",
        harvest_mode="semi_continuous",
        evidence_level="pilot",
        sh_coefficient=0.6, sh_re_exponent=0.50, sh_sc_exponent=0.33,
        sh_additive=2.0,                # Ranz-Marshall stagnant limit
        characteristic_length_m=5.0e-4,  # 0.5 mm seed particles
        velocity_kind="superficial",
        default_velocity_m_s=0.05,
        # a = 6(1-ε)/d_p with ε=0.5, d_p=0.5 mm → 6,000 m²/m³; over a 0.1 m
        # bed that is 600 m² of particle area per m² of cell footprint.
        active_area_ratio=600.0,
        # Beds are limited by potential distribution through the bed depth,
        # NOT by film transport. Per particle area the usable current density
        # is very low; this is the dominant constraint on the architecture.
        max_practical_j_A_m2=30.0,
        # And the footprint ceiling is the one that actually binds. Reported
        # fluidized-bed reactors operate at ~1-10 kA/m³ of bed; over a 0.1 m
        # bed that is ~100-1000 A per m² of footprint. Without this ceiling
        # the 600x area ratio would imply a physically absurd 18 kA/m².
        max_footprint_current_A_m2=1000.0,
        transport_utilization=1.5,
        electrode_cost_per_m2=300.0,
        separator_cost_per_m2=80.0,
        hardware_cost_per_m2=400.0,
        harvesting_cost_per_m2=250.0,
        harvest_downtime_hr=1.0,
        target_deposit_thickness_um=50.0,
        notes=(
            "Enormous specific area (~600 m² of particle surface per m² of "
            "footprint) at a very low current density per unit of that area. "
            "The trade is real but the binding constraint is ohmic potential "
            "distribution through the bed, not mass transfer - the Ranz-"
            "Marshall transport limit is never reached in practice. Proven "
            "for dilute-solution metal recovery, not for bulk production."
        ),
        limitations=[
            "Potential distribution through bed depth is the real limit",
            "Particle agglomeration and bed defluidization are failure modes",
            "Current feeder corrosion/contact resistance in a moving bed",
            "Product needs washing and dewatering",
        ],
    ),
}


# ═════════════════════════════════════════════════════════════════════
#  Operating conditions and results
# ═════════════════════════════════════════════════════════════════════

@dataclass
class OperatingConditions:
    """Electrolyte and economic context shared across architectures."""

    fe_conc_M: float = DEFAULT_FE_CONC_M
    temperature_C: float = DEFAULT_TEMPERATURE_C
    diffusivity_m2_s: float = DEFAULT_DIFFUSIVITY_M2_S
    kinematic_viscosity_m2_s: float = DEFAULT_KINEMATIC_VISCOSITY
    faradaic_efficiency: float = DEFAULT_FARADAIC_EFFICIENCY
    cell_voltage_V: float = 2.5

    # Cost context
    installed_cost_factor: float = 2.6
    """Multiplier from direct equipment $/m² to fully installed $/m².

    Covers rectifiers, electrolyte handling, infrastructure, engineering and
    contingency.  ``technoeconomic.CAPEXModel`` applies 1.15 assembly ×
    (1 + 0.25 infra + 0.15 eng) × 1.15 contingency ≈ 1.85 on stack costs, plus
    separate BOP lines; 2.6 is the round-trip equivalent including BOP.
    """

    discount_rate: float = 0.08
    plant_lifetime_yr: int = 25
    hours_per_year: float = 8760.0

    @property
    def schmidt(self) -> float:
        return schmidt_number(self.kinematic_viscosity_m2_s, self.diffusivity_m2_s)

    @property
    def crf(self) -> float:
        return capital_recovery_factor(self.discount_rate, self.plant_lifetime_yr)


@dataclass
class ArchitectureResult:
    """Screening result for one architecture at one operating point."""

    architecture_id: str
    name: str
    product_form: str
    harvest_mode: str
    evidence_level: str

    # Transport
    reynolds: float
    schmidt: float
    sherwood: float
    mass_transfer_coefficient_m_s: float
    transport_limit_A_m2: float

    # Operating point (per unit ACTIVE area)
    j_operating_A_m2: float
    limited_by: str                     # "transport" | "practical_ceiling"

    # Per unit INSTALLED (costed) area
    j_installed_A_m2: float
    deposition_rate_um_hr: float

    # Harvest cycle
    plating_cycle_hr: Optional[float]
    capacity_factor: float

    # Productivity and cost
    areal_productivity_t_m2_yr: float
    direct_cost_per_m2: float
    installed_cost_per_m2: float
    capex_per_annual_tonne: float
    capital_charge_per_t_fe: float

    # Context
    flow_enhancement_factor: float = 1.0
    notes: str = ""
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "architecture_id": self.architecture_id,
            "name": self.name,
            "product_form": self.product_form,
            "harvest_mode": self.harvest_mode,
            "evidence_level": self.evidence_level,
            "flow_enhancement_applied": self.flow_enhancement_factor,
            "Re": round(self.reynolds, 1),
            "Sc": round(self.schmidt, 1),
            "Sh": round(self.sherwood, 2),
            "k_m (m/s)": float(f"{self.mass_transfer_coefficient_m_s:.4g}"),
            "transport_limit (A/m2)": round(self.transport_limit_A_m2, 1),
            "j_operating (A/m2)": round(self.j_operating_A_m2, 1),
            "j_operating (mA/cm2)": round(self.j_operating_A_m2 / 10.0, 1),
            "limited_by": self.limited_by,
            "j_installed (A/m2)": round(self.j_installed_A_m2, 1),
            "deposition_rate (um/hr)": round(self.deposition_rate_um_hr, 1),
            "plating_cycle (hr)": (
                round(self.plating_cycle_hr, 2) if self.plating_cycle_hr else None
            ),
            "capacity_factor": round(self.capacity_factor, 4),
            "areal_productivity (t/m2/yr)": round(self.areal_productivity_t_m2_yr, 2),
            "direct_cost ($/m2)": round(self.direct_cost_per_m2, 0),
            "installed_cost ($/m2)": round(self.installed_cost_per_m2, 0),
            "capex_per_annual_tonne ($/t/yr)": round(self.capex_per_annual_tonne, 0),
            "capital_charge ($/t Fe)": round(self.capital_charge_per_t_fe, 2),
            "notes": self.notes,
            "limitations": self.limitations,
        }

    def summary(self) -> str:
        lines = [
            f"{self.name}",
            f"  evidence          {self.evidence_level}   "
            f"product: {self.product_form}   harvest: {self.harvest_mode}",
            f"  Re={self.reynolds:,.0f}  Sc={self.schmidt:,.0f}  "
            f"Sh={self.sherwood:,.1f}  k_m={self.mass_transfer_coefficient_m_s:.3g} m/s",
            f"  transport limit   {self.transport_limit_A_m2:,.0f} A/m² "
            f"({self.transport_limit_A_m2/10:,.0f} mA/cm²)",
            f"  operating j       {self.j_operating_A_m2:,.0f} A/m² "
            f"(limited by {self.limited_by})",
            f"  capacity factor   {self.capacity_factor:.3f}",
            f"  productivity      {self.areal_productivity_t_m2_yr:.2f} t/(m²·yr)",
            f"  installed cost    ${self.installed_cost_per_m2:,.0f}/m²",
            f"  capex intensity   ${self.capex_per_annual_tonne:,.0f} per annual tonne",
            f"  capital charge    ${self.capital_charge_per_t_fe:,.2f}/t Fe",
        ]
        return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════
#  Evaluation
# ═════════════════════════════════════════════════════════════════════

def _harvest_capacity_factor(
    spec: ArchitectureSpec,
    deposition_rate_um_hr_value: float,
) -> tuple:
    """Return (capacity_factor, plating_cycle_hr).

    Continuous architectures never stop, so they keep the base availability.
    Batch architectures must stop to be stripped, and *the faster they plate
    the more often they stop* — which is precisely why continuous harvesting
    earns its capital cost.
    """
    if spec.is_continuous or spec.harvest_downtime_hr <= 0:
        return spec.base_availability, None

    if deposition_rate_um_hr_value <= 0:
        return 0.0, math.inf

    cycle_hr = spec.target_deposit_thickness_um / deposition_rate_um_hr_value
    duty = cycle_hr / (cycle_hr + spec.harvest_downtime_hr)
    return spec.base_availability * duty, cycle_hr


def evaluate_architecture(
    spec: ArchitectureSpec,
    conditions: Optional[OperatingConditions] = None,
    velocity_m_s: Optional[float] = None,
) -> ArchitectureResult:
    """Screen one architecture: transport → operating j → productivity → $/t.

    Parameters
    ----------
    spec
        The architecture to evaluate.
    conditions
        Electrolyte and economic context.  Defaults to ``OperatingConditions()``.
    velocity_m_s
        Override the architecture's characteristic velocity (channel velocity,
        peripheral speed, belt speed or superficial velocity depending on
        ``spec.velocity_kind``).
    """
    cond = conditions or OperatingConditions()
    u = spec.default_velocity_m_s if velocity_m_s is None else velocity_m_s

    Sc = cond.schmidt
    Re = reynolds_number(u, spec.characteristic_length_m, cond.kinematic_viscosity_m2_s)
    Sh = sherwood_number(
        Re, Sc,
        spec.sh_coefficient, spec.sh_re_exponent, spec.sh_sc_exponent,
        additive=spec.sh_additive,
    )
    k_m = mass_transfer_coefficient(Sh, spec.characteristic_length_m, cond.diffusivity_m2_s)
    k_m *= spec.flow_enhancement_factor
    i_lim = limiting_current_from_km(k_m, cond.fe_conc_M)

    # Operating point: the lower of what transport allows and what the
    # hardware can actually do, on the ACTIVE area.
    j_transport = i_lim * spec.transport_utilization
    if j_transport <= spec.max_practical_j_A_m2:
        j_op = j_transport
        limited_by = "transport"
    else:
        j_op = spec.max_practical_j_A_m2
        limited_by = "practical_ceiling"

    # Productivity is quoted per m² of the area we PAY for, so scale the
    # active-area current density by the active/costed area ratio.
    j_installed = j_op * spec.active_area_ratio

    # Three-dimensional electrodes are then capped per unit footprint, which
    # is what ohmic potential distribution through the bed actually limits.
    if (
        spec.max_footprint_current_A_m2 is not None
        and j_installed > spec.max_footprint_current_A_m2
    ):
        j_installed = spec.max_footprint_current_A_m2
        j_op = j_installed / max(spec.active_area_ratio, 1e-12)
        limited_by = "footprint_ceiling"

    rate_um_hr = deposition_rate_um_hr(j_op, cond.faradaic_efficiency)
    capacity_factor, cycle_hr = _harvest_capacity_factor(spec, rate_um_hr)
    productivity = areal_productivity_t_m2_yr(
        j_installed,
        faradaic_efficiency=cond.faradaic_efficiency,
        capacity_factor=capacity_factor,
        hours_per_year=cond.hours_per_year,
    )

    installed_cost = spec.direct_cost_per_m2 * cond.installed_cost_factor
    if productivity > 0:
        capex_per_annual_t = installed_cost / productivity
        capital_charge = capex_per_annual_t * cond.crf
    else:
        capex_per_annual_t = math.inf
        capital_charge = math.inf

    return ArchitectureResult(
        architecture_id=spec.id,
        name=spec.name,
        product_form=spec.product_form,
        harvest_mode=spec.harvest_mode,
        evidence_level=spec.evidence_level,
        reynolds=float(Re),
        schmidt=float(Sc),
        sherwood=float(Sh),
        mass_transfer_coefficient_m_s=float(k_m),
        transport_limit_A_m2=float(i_lim),
        j_operating_A_m2=float(j_op),
        limited_by=limited_by,
        j_installed_A_m2=float(j_installed),
        deposition_rate_um_hr=float(rate_um_hr),
        plating_cycle_hr=cycle_hr,
        capacity_factor=float(capacity_factor),
        areal_productivity_t_m2_yr=float(productivity),
        direct_cost_per_m2=float(spec.direct_cost_per_m2),
        installed_cost_per_m2=float(installed_cost),
        capex_per_annual_tonne=float(capex_per_annual_t),
        capital_charge_per_t_fe=float(capital_charge),
        flow_enhancement_factor=float(spec.flow_enhancement_factor),
        notes=spec.notes,
        limitations=list(spec.limitations),
    )


def compare_architectures(
    conditions: Optional[OperatingConditions] = None,
    architecture_ids: Optional[List[str]] = None,
) -> List[ArchitectureResult]:
    """Evaluate every architecture, cheapest capital charge first."""
    cond = conditions or OperatingConditions()
    ids = architecture_ids or list(ARCHITECTURES.keys())
    results = [evaluate_architecture(ARCHITECTURES[i], cond) for i in ids]
    return sorted(results, key=lambda r: r.capital_charge_per_t_fe)


# ═════════════════════════════════════════════════════════════════════
#  Kill criterion #3 — the inverse question
# ═════════════════════════════════════════════════════════════════════

def max_affordable_cost_per_m2(
    areal_productivity_t_m2_yr_value: float,
    capital_charge_budget_per_t_fe: float,
    discount_rate: float = 0.08,
    plant_lifetime_yr: int = 25,
) -> float:
    """The kill-criterion threshold, in $/m² of installed cell.

    Kill criterion #3 in ``docs/RESEARCH_PROGRAM.md`` reads: *"Cost per m² of
    a cell that can be stripped continuously > threshold → pivot to cell
    architecture work."*  It never defined the threshold.  This does:

    .. math::
        \\$/m^2_{max} = \\frac{\\text{budget } [\\$/t] \\times
        \\text{productivity } [t/(m^2 \\cdot yr)]}{CRF}

    A cell costing more than this cannot meet the capital-charge budget at
    that productivity, no matter how good the electrochemistry is.
    """
    if areal_productivity_t_m2_yr_value <= 0:
        return 0.0
    crf = capital_recovery_factor(discount_rate, plant_lifetime_yr)
    return capital_charge_budget_per_t_fe * areal_productivity_t_m2_yr_value / crf


def capital_charge_per_t_fe(
    installed_cost_per_m2: float,
    areal_productivity_t_m2_yr_value: float,
    discount_rate: float = 0.08,
    plant_lifetime_yr: int = 25,
) -> float:
    """Annualized cell capital cost attributable to each tonne of iron ($/t)."""
    if areal_productivity_t_m2_yr_value <= 0:
        return math.inf
    crf = capital_recovery_factor(discount_rate, plant_lifetime_yr)
    return installed_cost_per_m2 * crf / areal_productivity_t_m2_yr_value


def zinc_tankhouse_productivity(
    faradaic_efficiency: float = 0.90,
    capacity_factor: float = 0.95,
    hours_per_year: float = 8760.0,
) -> float:
    """Areal productivity a zinc tankhouse would show *if it made iron*.

    Runs the benchmark's 500 A/m² through the iron Faraday arithmetic, so the
    "5× a zinc tankhouse" target is computed rather than asserted.
    """
    return areal_productivity_t_m2_yr(
        ZINC_TANKHOUSE["current_density_A_m2"],
        faradaic_efficiency=faradaic_efficiency,
        capacity_factor=capacity_factor,
        hours_per_year=hours_per_year,
    )


def kill_criterion_assessment(
    conditions: Optional[OperatingConditions] = None,
    capital_charge_budget_per_t_fe: float = 60.0,
    architecture_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Adjudicate kill criterion #3 across the architecture set.

    Parameters
    ----------
    capital_charge_budget_per_t_fe
        How much annualized cell capital each tonne of iron may carry.
        Default $60/t is ~10-15% of the $400-600/t product value, leaving
        room for electricity (~$85-145/t), feedstock, separations and labor.

    Returns
    -------
    dict
        Per-architecture verdicts plus the zinc benchmark comparison.
    """
    cond = conditions or OperatingConditions()
    results = compare_architectures(cond, architecture_ids)
    zinc_prod = zinc_tankhouse_productivity()

    verdicts = []
    for r in results:
        threshold = max_affordable_cost_per_m2(
            r.areal_productivity_t_m2_yr,
            capital_charge_budget_per_t_fe,
            cond.discount_rate,
            cond.plant_lifetime_yr,
        )
        passes = r.installed_cost_per_m2 <= threshold
        headroom = threshold - r.installed_cost_per_m2
        verdicts.append({
            "architecture_id": r.architecture_id,
            "name": r.name,
            "evidence_level": r.evidence_level,
            "harvest_mode": r.harvest_mode,
            "areal_productivity_t_m2_yr": round(r.areal_productivity_t_m2_yr, 2),
            "productivity_vs_zinc": round(
                r.areal_productivity_t_m2_yr / zinc_prod, 2
            ) if zinc_prod > 0 else None,
            "installed_cost_per_m2": round(r.installed_cost_per_m2, 0),
            "max_affordable_cost_per_m2": round(threshold, 0),
            "headroom_per_m2": round(headroom, 0),
            "capital_charge_per_t_fe": round(r.capital_charge_per_t_fe, 2),
            "passes": bool(passes),
            "verdict": "within budget" if passes else "exceeds budget",
        })

    passing = [v for v in verdicts if v["passes"]]
    return {
        "capital_charge_budget_per_t_fe": capital_charge_budget_per_t_fe,
        "crf": round(cond.crf, 5),
        "zinc_benchmark": {
            "current_density_A_m2": ZINC_TANKHOUSE["current_density_A_m2"],
            "iron_equivalent_productivity_t_m2_yr": round(zinc_prod, 2),
            "capex_per_annual_tonne_range": [
                ZINC_TANKHOUSE["capex_per_annual_tonne_low"],
                ZINC_TANKHOUSE["capex_per_annual_tonne_high"],
            ],
            "note": ZINC_TANKHOUSE["note"],
        },
        "architectures": verdicts,
        "n_passing": len(passing),
        "best": verdicts[0] if verdicts else None,
        "criterion": (
            "Kill criterion #3 (docs/RESEARCH_PROGRAM.md): cost per m² of a "
            "continuously strippable cell above the affordability threshold "
            "→ pivot to cell architecture work."
        ),
    }


# ═════════════════════════════════════════════════════════════════════
#  Sweeps
# ═════════════════════════════════════════════════════════════════════

def velocity_sweep(
    spec: ArchitectureSpec,
    velocities_m_s: Optional[np.ndarray] = None,
    conditions: Optional[OperatingConditions] = None,
) -> Dict[str, Any]:
    """Sweep the characteristic velocity for one architecture.

    Shows where pumping/rotation stops buying productivity because the
    practical ceiling, not transport, has become the binding constraint.
    """
    cond = conditions or OperatingConditions()
    if velocities_m_s is None:
        velocities_m_s = np.logspace(-2, 0.7, 40)
    velocities_m_s = np.asarray(velocities_m_s, dtype=float)

    rows = [evaluate_architecture(spec, cond, float(u)) for u in velocities_m_s]
    return {
        "architecture_id": spec.id,
        "velocity_m_s": velocities_m_s.tolist(),
        "transport_limit_A_m2": [r.transport_limit_A_m2 for r in rows],
        "j_operating_A_m2": [r.j_operating_A_m2 for r in rows],
        "areal_productivity_t_m2_yr": [r.areal_productivity_t_m2_yr for r in rows],
        "capital_charge_per_t_fe": [r.capital_charge_per_t_fe for r in rows],
        "limited_by": [r.limited_by for r in rows],
    }


def concentration_sweep(
    spec: ArchitectureSpec,
    concentrations_M: Optional[np.ndarray] = None,
    conditions: Optional[OperatingConditions] = None,
) -> Dict[str, Any]:
    """Sweep bulk Fe²⁺ concentration for one architecture.

    The transport limit is linear in bulk concentration, so this is the other
    lever (with temperature and δ) named in the program's missing-physics list.
    """
    cond = conditions or OperatingConditions()
    if concentrations_M is None:
        concentrations_M = np.linspace(0.1, 3.0, 30)
    concentrations_M = np.asarray(concentrations_M, dtype=float)

    rows = []
    for c in concentrations_M:
        cc = OperatingConditions(**{**cond.__dict__, "fe_conc_M": float(c)})
        rows.append(evaluate_architecture(spec, cc))

    return {
        "architecture_id": spec.id,
        "fe_conc_M": concentrations_M.tolist(),
        "transport_limit_A_m2": [r.transport_limit_A_m2 for r in rows],
        "j_operating_A_m2": [r.j_operating_A_m2 for r in rows],
        "areal_productivity_t_m2_yr": [r.areal_productivity_t_m2_yr for r in rows],
        "capital_charge_per_t_fe": [r.capital_charge_per_t_fe for r in rows],
    }


def comparison_table(results: List[ArchitectureResult]) -> str:
    """Fixed-width comparison table for console/report output."""
    header = (
        f"{'Architecture':<34} {'Harvest':<15} {'Evid.':<11} "
        f"{'j act.':>10} {'j ftpt.':>9} {'t/(m²·yr)':>10} {'$/m²':>9} {'$/t Fe':>9}"
    )
    sep = "─" * len(header)
    lines = [header, sep]
    for r in results:
        j_act = r.j_operating_A_m2 / 10.0      # mA/cm² on active area
        j_ftp = r.j_installed_A_m2 / 10.0      # mA/cm² on costed footprint
        act_s = f"{j_act:,.2f}" if j_act < 10 else f"{j_act:,.0f}"
        ftp_s = f"{j_ftp:,.2f}" if j_ftp < 10 else f"{j_ftp:,.0f}"
        lines.append(
            f"{r.name[:33]:<34} {r.harvest_mode:<15} {r.evidence_level:<11} "
            f"{act_s:>10} {ftp_s:>9} "
            f"{r.areal_productivity_t_m2_yr:>10,.2f} "
            f"{r.installed_cost_per_m2:>9,.0f} "
            f"{r.capital_charge_per_t_fe:>9,.2f}"
        )
    lines.append("")
    lines.append("j act. = mA/cm² on active electrode area; "
                 "j ftpt. = mA/cm² on costed footprint area")
    return "\n".join(lines)


def model_scope() -> Dict[str, Any]:
    """Machine-readable statement of what this model is and is not."""
    return {
        "provenance": (
            "Screening model. No wet-lab iron data. Sherwood correlations are "
            "literature values measured in other chemistries (ferricyanide, "
            "copper sulfate) and transferred to Fe²⁺ sulfate."
        ),
        "computes": [
            "Architecture-specific mass transfer and transport-limited current",
            "Operating current density under transport AND practical ceilings",
            "Harvest-cycle capacity factor (batch downtime scales with rate)",
            "Areal productivity per m² of costed electrode area",
            "Installed $/m², $/annual tonne, and $/t Fe capital charge",
            "Kill criterion #3 affordability threshold",
        ],
        "does_not_compute": [
            "Faradaic efficiency (supplied by kinetics/diffusion_layer_1d)",
            "Cell voltage (supplied by electrochemistry.CellVoltageModel)",
            "Deposit adhesion or peelability - the gating unknown for drums",
            "Current distribution within the electrode (see scale_up.py)",
            "Bubble effects on transport or ohmic drop",
            "Mechanical reliability of rotating/moving hardware in acid",
        ],
        "calibration_required": [
            "Sherwood coefficients for the actual cell and electrolyte",
            "Equipment costs from vendor quotes, not engineering estimates",
            "Harvest downtime from operated hardware",
            "Practical current ceilings from deposit-quality experiments",
        ],
    }
