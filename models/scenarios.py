"""
Scenario definitions for aqueous electrowinning techno-economic analysis.

Each scenario represents a different electrolyte/operating regime with
parameters drawn from published literature and projected targets.

Scenarios:
  1. Conservative (Base)       — Near-term lab-scale, alkaline
  2. Optimized Alkaline        — Yuan et al. / Kempler et al. regime
  3. AWARE Acidic              — 2024–2025 anion-rich electrolyte breakthrough
  4. Future Target             — Aspirational with advanced catalysts/membranes
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import math

from .electrochemistry import R_GAS
from .kinetics import DepositionKinetics
from .surface_state import CL_NA_AWARE
from .fe_chloride_speciation import aware_default_bath, solve_chloride_speciation

if TYPE_CHECKING:
    from .anode import AnodeKinetics


@dataclass
class Scenario:
    """
    A complete scenario definition for techno-economic evaluation.

    The ``anode`` field holds a pre-built :class:`.anode.AnodeKinetics` object
    that drives ``eta_anode`` and ``E_anode_eq`` from first principles.  When
    ``anode`` is ``None`` the legacy fixed ``eta_anode`` / ``E_anode_eq`` values
    are used (for backwards compatibility).
    """

    name: str
    description: str

    # Electrolyte / chemistry
    electrolyte_type: str           # "alkaline", "acidic_anion_rich"
    electrolyte_composition: str    # human-readable description

    # Operating conditions
    current_density_mA_cm2: float
    current_efficiency: float       # fraction (0–1)
    temperature_C: float

    # Cell voltage decomposition
    E_cathode_eq: float             # V vs. SHE
    E_anode_eq: float               # V vs. SHE (legacy / fallback)
    eta_cathode: float              # V
    eta_anode: float                # V (legacy / fallback)
    ir_drop: float                  # V

    # Anode type
    anode_type: str                 # "DSA_IrO2_Ta2O5", "Ni_co_spinel", "Pt_Ti"

    # Optional first-principles anode model (overrides eta_anode / E_anode_eq)
    anode: Optional["AnodeKinetics"] = field(default=None, repr=False)

    # Physics provenance for scenario numbers that are *derived* rather than
    # assumed (Tier 1.4 chloride-bath wiring).  When populated it records the
    # mechanism that produced ``current_efficiency``, ``ir_drop`` /
    # ``conductivity_S_m``, so a reader can audit that a headline number is a
    # consequence of physics, not a preset knob.  Empty for scenarios whose
    # values are still inputs.
    physical_derivation: dict = field(default_factory=dict, repr=False)

    # Economic modifiers (relative to base CAPEX/OPEX models)
    capex_modifier: float = 1.0
    electricity_price_kWh: float = 0.04
    electrolyte_makeup_per_t_Fe: float = 15.0
    ore_cost_per_t_Fe: float = 40.0

    # References
    references: str = ""

    @property
    def _effective_anode_eq(self) -> float:
        """Anode equilibrium potential (V vs. SHE): model or legacy value.

        For a soluble Fe anode this is the Fe²⁺/Fe potential; for an inert
        DSA it is the OER equilibrium.
        """
        if self.anode is not None:
            if getattr(self.anode, "is_soluble", False):
                return self.anode.fe_dissolution_equilibrium()
            return self.anode.oer_equilibrium()
        return self.E_anode_eq

    @property
    def _effective_eta_anode(self) -> float:
        """Anode overpotential (V): model or legacy value."""
        if self.anode is not None:
            return self.anode.eta_anode(self.current_density_mA_cm2)
        return self.eta_anode

    @property
    def V_cell(self) -> float:
        """Total cell voltage (V) using model where available."""
        return (
            abs(self._effective_anode_eq - self.E_cathode_eq)
            + self.eta_cathode
            + self._effective_eta_anode
            + self.ir_drop
        )

    @property
    def anode_summary(self) -> dict:
        """Full anode decomposition (model) or empty dict (legacy)."""
        if self.anode is not None:
            return self.anode.overpotential_at_current(self.current_density_mA_cm2)
        return {}


# ─── Chloride-bath physics derivation (CHEM_PHYS_REVIEW §1.4) ──────
#
# These helpers turn the AWARE / concentrated-chloride route's headline
# numbers — near-unity FE and the bath conductivity — into *derivations*
# from the chloride physics in models/surface_state (Cl-site-blocking HER
# suppression) and models/fe_chloride_speciation (Pitzer Fe-Cl-water +
# FeCl⁺/FeCl₂(aq) pairing + Onsager conductivity).  Before this wiring the
# scenario carried ``current_efficiency=0.99`` and an input electrolyte
# resistivity as preset knobs; now they are computed and auditable via
# ``Scenario.physical_derivation``.  The sulfate (default) path is
# untouched — this is opt-in chloride chemistry.

# Screening base HER exchange current density on Fe in uninhibited acid
# (A/m², anchored at the kinetics reference temperature), before any
# chloride site blocking.  Cl⁻ suppresses this by blocking H-adsorption
# sites, which is the *robust* mechanism: the Frumkin IHP-shift
# amplification is a wide sensitivity band (see models/surface_state,
# frumkin_sensitivity_sweep) and is deliberately excluded here so the
# FE derivation stays a single defensible number.
AWARE_BASE_HER_I0_ACID = 1e-4      # A/m²
# Screening inter-electrode gap for turning conductivity into ir-drop.
AWARE_ELECTRODE_GAP_M = 1.0e-3     # 1 mm


def chloride_theta_block(c_cl_M: float, T_C: float = 60.0) -> float:
    """Fraction of Fe surface sites blocked by specifically adsorbed Cl⁻.

    Competitive-Langmuir coverage θ = K·c / (1 + K·c) at bulk chloride
    molality, with K = exp(−ΔG_ads/RT) using the screening ΔG_ads of
    ``CL_NA_AWARE``.  This is the chloride-induced HER-suppression term:
    a Cl-covered site cannot form H*, so the effective HER exchange
    current scales as (1 − θ_block).  Monotone in c_Cl.
    """
    T_K = T_C + 273.15
    K = math.exp(-CL_NA_AWARE.DG_ads_J_mol / (R_GAS * T_K))
    return float(K * c_cl_M / (1.0 + K * c_cl_M))


def derive_aware_her_suppression(
    c_FeCl2: float = 1.0,
    c_LiCl: float = 10.0,
    T_C: float = 60.0,
    base_her_i0: float = AWARE_BASE_HER_I0_ACID,
) -> dict:
    """Chloride site-blocking HER suppression for the AWARE bath.

    Returns ``{"theta_block", "c_Cl_M", "base_her_i0_A_m2",
    "her_i0_A_m2"}`` where ``her_i0_A_m2`` is the base HER exchange
    current reduced by the chloride blocking coverage.
    """
    c_cl = 2.0 * c_FeCl2 + c_LiCl
    tb = chloride_theta_block(c_cl, T_C)
    return {
        "theta_block": tb,
        "c_Cl_M": c_cl,
        "base_her_i0_A_m2": base_her_i0,
        "her_i0_A_m2": base_her_i0 * (1.0 - tb),
    }


def derive_aware_current_efficiency(
    current_density_mA_cm2: float = 500.0,
    T_C: float = 60.0,
    c_FeCl2: float = 1.0,
    c_LiCl: float = 10.0,
    pH: float = 2.0,
    fe_i0: float = 1e-2,
    base_her_i0: float = AWARE_BASE_HER_I0_ACID,
    boundary_layer_m: float = 5e-5,
) -> float:
    """FE for the chloride bath, *derived* from chloride-induced HER suppression.

    The HER exchange current is reduced by the chloride site-blocking
    coverage, then the competing Fe/HER Butler–Volmer kinetics are solved
    at the scenario current density.  Near-unity FE is therefore a
    consequence of Cl⁻ displacing H-adsorption sites, not a preset
    ``current_efficiency`` parameter.  Rises monotonically with ``c_LiCl``.
    """
    supp = derive_aware_her_suppression(c_FeCl2, c_LiCl, T_C, base_her_i0)
    dk = DepositionKinetics(
        pH=pH, temperature_C=T_C, fe_i0=fe_i0,
        her_i0=supp["her_i0_A_m2"], fe_conc_M=c_FeCl2,
        boundary_layer_m=boundary_layer_m,
    )
    return float(dk.efficiency_at_current(current_density_mA_cm2))


def aware_bath_conductivity_S_m(
    c_FeCl2: float = 1.0,
    c_LiCl: float = 10.0,
    T_C: float = 60.0,
) -> float:
    """Computed ionic conductivity of the concentrated-LiCl AWARE bath (S/m).

    Delegates to ``fe_chloride_speciation.solve_chloride_speciation``
    (Pitzer Fe-Cl-water activities, FeCl⁺/FeCl₂(aq) pairing, and the
    Onsager √I ionic-mobility sum) — the conductivity is asserted by the
    model, not supplied as an input.
    """
    sol = solve_chloride_speciation(
        aware_default_bath(c_FeCl2=c_FeCl2, c_LiCl=c_LiCl, c_HCl=0.01, T_C=T_C)
    )
    return float(sol["conductivity_S_m"])


def derive_aware_ir_drop_V(
    current_density_mA_cm2: float = 500.0,
    T_C: float = 60.0,
    c_FeCl2: float = 1.0,
    c_LiCl: float = 10.0,
    gap_m: float = AWARE_ELECTRODE_GAP_M,
) -> float:
    """Ohmic ir-drop (V) from the *computed* conductivity.

    ir = j · d_gap / κ, with κ from :func:`aware_bath_conductivity_S_m`.
    The electrode gap is a documented screening input; the conductivity
    that dominates it is computed.
    """
    kappa = aware_bath_conductivity_S_m(c_FeCl2, c_LiCl, T_C)
    return current_density_mA_cm2 * 10.0 * gap_m / kappa


# ─── Factory helpers ────────────────────────────────────────────────────

def _build_dsa_acidic(
    temperature_C: float,
    pH: float,
    j_mA_cm2: float,
) -> "AnodeKinetics":
    """IrO₂–Ta₂O₅ DSA in acidic / neutral bath."""
    from .anode import AnodeKinetics, DSA_IRO2_TA2O5
    mat = DSA_IRO2_TA2O5
    mat = mat.__class__(
        name=mat.name,
        oer_i0=mat.oer_i0,
        oer_tafel_V=mat.oer_tafel_V,
        cer_i0=mat.cer_i0,
        cer_tafel_V=mat.cer_tafel_V,
        cer_n=mat.cer_n,
        oer_n=mat.oer_n,
        max_bubble_fraction=mat.max_bubble_fraction,
        temperature_C=temperature_C,
        oer_ea_kj_mol=mat.oer_ea_kj_mol,
        references=mat.references,
    )
    return AnodeKinetics(
        material=mat,
        electrolyte_type="acidic",
        pH=pH,
        electrolyte_resistivity_ohm_m2=0.0005,  # 0.005 Ω·cm² (acidic electrolyte)
    )


def _build_nico_spinel_alkaline(
    temperature_C: float,
    pH: float,
    j_mA_cm2: float,
) -> "AnodeKinetics":
    """NiCo₂O₄ / Ni foam in alkaline bath."""
    from .anode import AnodeKinetics, NICO_SPINEL
    mat = NICO_SPINEL
    mat = mat.__class__(
        name=mat.name,
        oer_i0=mat.oer_i0,
        oer_tafel_V=mat.oer_tafel_V,
        cer_i0=mat.cer_i0,
        cer_tafel_V=mat.cer_tafel_V,
        cer_n=mat.cer_n,
        oer_n=mat.oer_n,
        max_bubble_fraction=mat.max_bubble_fraction,
        temperature_C=temperature_C,
        oer_ea_kj_mol=mat.oer_ea_kj_mol,
        references=mat.references,
    )
    return AnodeKinetics(
        material=mat,
        electrolyte_type="alkaline",
        pH=pH,
        electrolyte_resistivity_ohm_m2=0.0003,  # 0.003 Ω·cm² (concentrated NaOH)
    )


def _build_aware_acidic(
    temperature_C: float,
    j_mA_cm2: float,
) -> "AnodeKinetics":
    """AWARE process: concentrated LiCl acidic bath (very high conductivity).

    The electrolyte conductivity and therefore the cell's ohmic behaviour
    are *computed* (Tier 1.4) from the chloride-bath physics
    (:func:`aware_bath_conductivity_S_m`) rather than supplied as an
    input resistivity — the high-κ concentrated-LiCl bath is why the
    ohmic term is small.
    """
    from .anode import AnodeKinetics, DSA_IRO2_TA2O5
    mat = DSA_IRO2_TA2O5
    mat = mat.__class__(
        name=mat.name,
        oer_i0=mat.oer_i0,
        oer_tafel_V=mat.oer_tafel_V,
        cer_i0=mat.cer_i0,
        cer_tafel_V=mat.cer_tafel_V,
        cer_n=mat.cer_n,
        oer_n=mat.oer_n,
        max_bubble_fraction=mat.max_bubble_fraction,
        temperature_C=temperature_C,
        oer_ea_kj_mol=mat.oer_ea_kj_mol,
        references=mat.references,
    )
    kappa = aware_bath_conductivity_S_m(c_FeCl2=1.0, c_LiCl=10.0, T_C=temperature_C)
    return AnodeKinetics(
        material=mat,
        electrolyte_type="acidic_chloride",
        pH=0.0,
        a_Cl_molar=12.0,   # concentrated LiCl (≥10 M Cl⁻)
        electrolyte_conductivity_S_m=kappa,
        electrolyte_resistivity_ohm_m2=AWARE_ELECTRODE_GAP_M / kappa,  # computed, not input
    )


def _build_soluble_fe(
    temperature_C: float,
    pH: float,
    j_mA_cm2: float,
    fe2_conc_M: float = 1.0,
) -> "AnodeKinetics":
    """Soluble Fe anode (Fe → Fe²⁺ + 2e⁻): classical iron electrorefining.

    No gas evolution, near Fe²⁺/Fe equilibrium, so the cell runs at a
    small inter-electrode voltage (~0.2–0.4 V) rather than the ~2 V of
    an oxygen-evolving DSA.
    """
    from .anode import AnodeKinetics, DSA_IRO2_TA2O5
    # Material only supplies the temperature carrier; the chemistry flag
    # switches all kinetics to Fe dissolution.
    mat = DSA_IRO2_TA2O5.__class__(
        name="Soluble Fe anode",
        oer_i0=DSA_IRO2_TA2O5.oer_i0,
        oer_tafel_V=DSA_IRO2_TA2O5.oer_tafel_V,
        cer_i0=DSA_IRO2_TA2O5.cer_i0,
        cer_tafel_V=DSA_IRO2_TA2O5.cer_tafel_V,
        cer_n=DSA_IRO2_TA2O5.cer_n,
        oer_n=DSA_IRO2_TA2O5.oer_n,
        max_bubble_fraction=0.0,
        temperature_C=temperature_C,
        oer_ea_kj_mol=DSA_IRO2_TA2O5.oer_ea_kj_mol,
        references="Classical Fe electrorefining; soluble iron anode",
    )
    return AnodeKinetics(
        material=mat,
        electrolyte_type="acidic",
        pH=pH,
        electrolyte_resistivity_ohm_m2=0.0005,
        anode_chemistry="soluble",
        fe2_conc_M=fe2_conc_M,
    )


# ─── Scenario Definitions ────────────────────────────────────────────────

CONSERVATIVE_ALKALINE = Scenario(
    name="Conservative (Base)",
    description=(
        "Near-term alkaline electrowinning with conventional electrode materials "
        "and moderate current density. Represents a cautious estimate based on "
        "established electroplating technology scaled to bulk production."
    ),
    electrolyte_type="alkaline",
    electrolyte_composition="1–3 M NaOH, citrate complexant",
    current_density_mA_cm2=100.0,
    current_efficiency=0.90,
    temperature_C=70.0,
    E_cathode_eq=-0.440,
    E_anode_eq=0.401,          # alkaline OER at pH 14 (legacy reference)
    eta_cathode=0.30,
    eta_anode=0.40,            # legacy fallback
    ir_drop=0.20,
    anode_type="DSA_IrO2_Ta2O5",
    anode=_build_dsa_acidic(temperature_C=70.0, pH=14.0, j_mA_cm2=100.0),
    capex_modifier=1.0,
    electricity_price_kWh=0.04,
    electrolyte_makeup_per_t_Fe=15.0,
    references="Standard alkaline electroplating literature; Yuan & Haarberg (2009)",
)

OPTIMIZED_ALKALINE = Scenario(
    name="Optimized Alkaline",
    description=(
        "High-efficiency alkaline route using concentrated NaOH with nanoporous "
        "Fe₂O₃ intermediates and rotating cathode for enhanced mass transport. "
        "Based on Yuan et al. (2009) and Kempler et al. (2025) demonstrating "
        ">90% current efficiency with unique twin-crystal morphology."
    ),
    electrolyte_type="alkaline",
    electrolyte_composition="10–14 M NaOH, nanoporous Fe₂O₃ suspension",
    current_density_mA_cm2=200.0,
    current_efficiency=0.93,
    temperature_C=90.0,
    E_cathode_eq=-0.440,
    E_anode_eq=0.401,          # alkaline OER (legacy reference)
    eta_cathode=0.22,
    eta_anode=0.35,            # legacy fallback
    ir_drop=0.15,
    anode_type="Ni_co_spinel",
    anode=_build_nico_spinel_alkaline(temperature_C=90.0, pH=14.0, j_mA_cm2=200.0),
    capex_modifier=1.10,
    electricity_price_kWh=0.04,
    electrolyte_makeup_per_t_Fe=10.0,
    ore_cost_per_t_Fe=35.0,
    references="Yuan & Haarberg (2009); Kempler et al. (2025) ACS Nano",
)

# Physics-derived AWARE headline values (Tier 1.4): computed once at
# import so the scenario dataclass and its provenance dict share them.
# The near-unity FE is *derived* from chloride-induced HER suppression
# (site blocking), and the conductivity / ir-drop come from the
# chloride-bath Pitzer + Onsager model — neither is a preset knob.
_AWARE_FE_PHYSICS = derive_aware_current_efficiency(
    current_density_mA_cm2=500.0, T_C=60.0, c_FeCl2=1.0, c_LiCl=10.0, pH=2.0)
_AWARE_KAPPA_S_M = aware_bath_conductivity_S_m(c_FeCl2=1.0, c_LiCl=10.0, T_C=60.0)
_AWARE_IR_DROP_V = derive_aware_ir_drop_V(500.0, 60.0, 1.0, 10.0)

AWARE_ACIDIC = Scenario(
    name="AWARE Acidic",
    description=(
        "Acidic electro-Winning in Anion-Rich Electrolytes using concentrated "
        "LiCl-based systems. Achieves near-unity Coulombic efficiency by "
        "thermodynamically suppressing HER through high chloride activity and "
        "stabilizing Fe²⁺/Fe³⁺ in solution. Zero-waste operation with high "
        "impurity tolerance."
    ),
    electrolyte_type="acidic_anion_rich",
    electrolyte_composition="Concentrated LiCl (≥10 M), acidic pH < 2",
    current_density_mA_cm2=500.0,
    current_efficiency=_AWARE_FE_PHYSICS,   # derived from Cl⁻ HER suppression, not assumed
    temperature_C=60.0,
    E_cathode_eq=-0.440,
    E_anode_eq=1.360,          # CER dominates in conc. Cl⁻ (legacy reference)
    eta_cathode=0.12,
    eta_anode=0.25,            # legacy fallback (overridden by the anode model)
    ir_drop=_AWARE_IR_DROP_V,  # from computed conductivity κ = d_gap/κ, not input
    anode_type="DSA_IrO2_Ta2O5",
    anode=_build_aware_acidic(temperature_C=60.0, j_mA_cm2=500.0),
    capex_modifier=1.20,
    electricity_price_kWh=0.04,
    electrolyte_makeup_per_t_Fe=20.0,
    ore_cost_per_t_Fe=40.0,
    physical_derivation={
        "activity_model": "pitzer_fecl2 + surface_state",
        "fe_source": "derived",
        "current_efficiency_derived": _AWARE_FE_PHYSICS,
        **derive_aware_her_suppression(c_FeCl2=1.0, c_LiCl=10.0, T_C=60.0),
        "conductivity_S_m_computed": _AWARE_KAPPA_S_M,
        "ir_drop_derived_V": _AWARE_IR_DROP_V,
        "electrode_gap_m": AWARE_ELECTRODE_GAP_M,
        "screening_flag": "unvalidated (L1)",
    },
    references="AWARE process (2024–2025), ChemRxiv; follow-up publications",
)

FUTURE_TARGET = Scenario(
    name="Future Target",
    description=(
        "Aspirational scenario assuming R&D breakthroughs in catalyst design, "
        "membrane technology, and process integration. Combines high current "
        "density with low cell voltage through advanced electrode architectures "
        "(nanostructured cathodes, ultra-thin separators) and AI-optimized "
        "pulse-reverse deposition protocols."
    ),
    electrolyte_type="acidic_anion_rich",
    electrolyte_composition="Optimized mixed-anion electrolyte (TBD)",
    current_density_mA_cm2=400.0,
    current_efficiency=0.97,
    temperature_C=50.0,
    E_cathode_eq=-0.440,
    E_anode_eq=1.229,          # OER baseline (legacy reference)
    eta_cathode=0.10,
    eta_anode=0.20,            # legacy fallback
    ir_drop=0.08,
    anode_type="DSA_IrO2_Ta2O5",
    anode=_build_dsa_acidic(temperature_C=50.0, pH=2.0, j_mA_cm2=400.0),
    capex_modifier=0.85,
    electricity_price_kWh=0.03,
    electrolyte_makeup_per_t_Fe=8.0,
    ore_cost_per_t_Fe=35.0,
    references="Projected; based on extrapolation of current trends",
)

SOLUBLE_FE_ACIDIC = Scenario(
    name="Soluble Fe anode (acidic sulfate)",
    description=(
        "Classical iron electrorefining configuration: a soluble Fe anode "
        "dissolves (Fe → Fe²⁺ + 2e⁻) at near the Fe²⁺/Fe potential while Fe "
        "plates at the cathode. No oxygen/chlorine gas, no bubble penalty, "
        "and a cell voltage an order of magnitude below an oxygen-evolving "
        "DSA — the lowest-energy route, contingent on a supply of scrap or "
        "high-purity iron anodes (it refines rather than wins iron from ore)."
    ),
    electrolyte_type="acidic",
    electrolyte_composition="1 M FeSO₄ / 0.5 M Na₂SO₄, pH 2–3",
    current_density_mA_cm2=200.0,
    current_efficiency=0.95,
    temperature_C=60.0,
    E_cathode_eq=-0.440,
    E_anode_eq=-0.440,          # symmetric Fe/Fe²⁺ cell
    eta_cathode=0.15,
    eta_anode=0.10,            # legacy fallback (overridden by model)
    ir_drop=0.10,
    anode_type="soluble_Fe",
    anode=_build_soluble_fe(temperature_C=60.0, pH=2.5, j_mA_cm2=200.0),
    capex_modifier=0.90,
    electricity_price_kWh=0.04,
    electrolyte_makeup_per_t_Fe=5.0,
    ore_cost_per_t_Fe=120.0,   # scrap/Fe anode feedstock, not ore
    references="Classical Fe electrorefining; soluble anode electrochemistry",
)

ALL_SCENARIOS = [
    CONSERVATIVE_ALKALINE,
    OPTIMIZED_ALKALINE,
    AWARE_ACIDIC,
    FUTURE_TARGET,
    SOLUBLE_FE_ACIDIC,
]
