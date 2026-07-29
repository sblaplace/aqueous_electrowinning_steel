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

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Scenario:
    """
    A complete scenario definition for techno-economic evaluation.
    """
    name: str
    description: str

    # Electrolyte / chemistry
    electrolyte_type: str           # "alkaline", "acidic_anion_rich", "chloride"
    electrolyte_composition: str    # human-readable description

    # Operating conditions
    current_density_mA_cm2: float
    current_efficiency: float       # fraction (0–1)
    temperature_C: float

    # Cell voltage decomposition
    E_cathode_eq: float             # V vs. SHE
    E_anode_eq: float               # V vs. SHE
    eta_cathode: float              # V
    eta_anode: float                # V
    ir_drop: float                  # V

    # Anode type
    anode_type: str                 # "DSA_IrO2_Ta2O5", "Ni_co_spinel", "Pt_Ti"

    # Economic modifiers (relative to base CAPEX/OPEX models)
    capex_modifier: float = 1.0     # multiplier on total CAPEX
    electricity_price_kWh: float = 0.04
    electrolyte_makeup_per_t_Fe: float = 15.0
    ore_cost_per_t_Fe: float = 40.0

    # References
    references: str = ""

    @property
    def V_cell(self) -> float:
        """Total cell voltage."""
        return abs(self.E_anode_eq - self.E_cathode_eq) + self.eta_cathode + self.eta_anode + self.ir_drop

    def summary_dict(self) -> dict:
        return {
            "name": self.name,
            "electrolyte": self.electrolyte_composition,
            "j (mA/cm²)": self.current_density_mA_cm2,
            "CE (%)": f"{self.current_efficiency * 100:.1f}",
            "T (°C)": self.temperature_C,
            "V_cell (V)": round(self.V_cell, 2),
            "anode": self.anode_type,
        }


# ─── Scenario Definitions ──────────────────────────────────────────────

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
    E_anode_eq=1.229,
    eta_cathode=0.30,
    eta_anode=0.40,
    ir_drop=0.20,
    anode_type="DSA_IrO2_Ta2O5",
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
    E_anode_eq=1.229,
    eta_cathode=0.22,     # Better cathode kinetics at high T + rotating cathode
    eta_anode=0.35,      # Ni-Co spinel anode with lower OER overpotential
    ir_drop=0.15,        # Narrower gap, higher conductivity at high NaOH conc.
    anode_type="Ni_co_spinel",
    capex_modifier=1.10,   # Slightly higher CAPEX for rotating cathode system
    electricity_price_kWh=0.04,
    electrolyte_makeup_per_t_Fe=10.0,  # Less makeup with stable NaOH system
    ore_cost_per_t_Fe=35.0,
    references="Yuan & Haarberg (2009); Kempler et al. (2025) ACS Nano",
)

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
    current_efficiency=0.99,
    temperature_C=60.0,
    E_cathode_eq=-0.440,
    E_anode_eq=1.360,    # Cl₂/Cl⁻ or modified OER in high-Cl⁻ media
    eta_cathode=0.12,     # Excellent Fe deposition kinetics in Cl⁻-rich media
    eta_anode=0.25,       # Low OER/CER overpotential on DSA
    ir_drop=0.10,         # Very high electrolyte conductivity
    anode_type="DSA_IrO2_Ta2O5",
    capex_modifier=1.20,   # Higher CAPEX for corrosion-resistant materials
    electricity_price_kWh=0.04,
    electrolyte_makeup_per_t_Fe=20.0,   # LiCl is more expensive
    ore_cost_per_t_Fe=40.0,
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
    E_anode_eq=1.229,
    eta_cathode=0.10,     # Advanced nanostructured cathode
    eta_anode=0.20,       # Next-gen OER catalyst
    ir_drop=0.08,         # Ultra-thin membrane, zero-gap cell
    anode_type="DSA_IrO2_Ta2O5",
    capex_modifier=0.85,   # Learning curve and scale effects reduce CAPEX
    electricity_price_kWh=0.03,  # Assumes cheaper renewable PPA
    electrolyte_makeup_per_t_Fe=8.0,
    ore_cost_per_t_Fe=35.0,
    references="Projected; based on extrapolation of current trends",
)

ALL_SCENARIOS = [
    CONSERVATIVE_ALKALINE,
    OPTIMIZED_ALKALINE,
    AWARE_ACIDIC,
    FUTURE_TARGET,
]
