"""
Pilot-plant CAPEX/OPEX model for aqueous electrowinning + carburizing + tempering.

Equipment list derived from the pilot P&ID (models/pid.py). Uses six-tenths
rule for scaling across three production scales: lab (1 kg/day), pilot
(10 kg/day), and small production (100 kg/day).

All costs in USD (2024 basis).

References
----------
- Peters, Timmerhaus & West, Plant Design and Economics (4th ed.)
- Six-tenths rule: C₂ = C₁ × (Q₂/Q₁)^0.6
- Humbert et al. (2024), J. Sustainable Metallurgy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json


# ─── Production scales (kg Fe / day) ─────────────────────────────────
SCALE_LAB = 1.0        # kg/day
SCALE_PILOT = 10.0     # kg/day
SCALE_PRODUCTION = 100.0  # kg/day

REFERENCE_SCALE = SCALE_PILOT  # base costs are for pilot scale


@dataclass
class EquipmentItem:
    """Single P&ID equipment with reference cost."""
    tag: str
    name: str
    category: str       # "tanks", "cell", "furnace", "gas", "instruments", "piping"
    reference_cost_usd: float  # cost at pilot (10 kg/day) scale
    scale_exponent: float = 0.6  # default six-tenths rule
    notes: str = ""


# ─── P&ID Equipment List (reference costs at 10 kg/day pilot scale) ──

PID_EQUIPMENT: List[EquipmentItem] = [
    # Leaching / electrolyte prep
    EquipmentItem("TK-101", "Leaching tank, FRP 2 m³, agitator", "tanks", 5_000,
                  notes="FRP construction, variable-speed agitator"),
    EquipmentItem("TK-102", "Electrolyte prep tank", "tanks", 3_000,
                  notes="Prep + filtration"),
    EquipmentItem("TK-103", "Electrolyte storage tank", "tanks", 3_000,
                  notes="Atmospheric storage"),

    # Electrowinning
    EquipmentItem("C-201", "Electrowinning cell stack (DSA anodes, cathode blanks)",
                  "cell", 150_000,
                  notes="10-cell bipolar stack, DSA Ti/IrO₂-Ta₂O₅ anodes"),
    EquipmentItem("P-201", "Recirculation pump + HE-201 heat exchanger",
                  "tanks", 2_000,
                  notes="Mag-drive pump + plate HE"),
    EquipmentItem("TK-202", "CSTR electrolyte recycle", "tanks", 8_000,
                  notes="Stirred tank + purge line"),
    EquipmentItem("FL-201", "Electrolyte filter (crossflow)", "tanks", 2_500,
                  notes="Ceramic membrane filter"),

    # Gas handling
    EquipmentItem("TK-301A", "O₂ vent header", "gas", 3_000,
                  notes="PVC vent + check valve"),
    EquipmentItem("TK-301B", "Cl₂ scrubber (NaOH)", "gas", 8_000,
                  notes="Packed tower, NaOH recirc"),
    EquipmentItem("TK-302", "Purge treatment tank", "gas", 3_000,
                  notes="Chemical neutralization"),

    # Post-deposition
    EquipmentItem("TK-401", "Wash / drying station", "tanks", 2_000,
                  notes="DI water wash + hot air dryer"),

    # Carburizing
    EquipmentItem("F-501", "Carburizing retort (SiC retort, endo gas)",
                  "furnace", 80_000,
                  notes="SiC retort, 950 °C max, endo gas atmosphere"),
    EquipmentItem("GS-501", "Gas manifold (MFC for CO/CO₂/CH₄/H₂)",
                  "gas", 15_000,
                  notes="4× MFC + mixing header + safety interlock"),

    # Quench & temper
    EquipmentItem("TK-502", "Quench system (oil/polymer tank + agitation)",
                  "tanks", 10_000,
                  notes="Oil or polymer quench with temperature control"),
    EquipmentItem("F-503", "Tempering furnace", "furnace", 30_000,
                  notes="Air circulation, 400–700 °C"),

    # Instruments (aggregate)
    EquipmentItem("INST", "Instruments (all LT/FT/TT/PT/AT/pHAT/AIT/MFC)",
                  "instruments", 25_000,
                  notes="ISA instruments + SCADA I/O modules"),
]

# Piping, valves, structural is calculated as fraction of equipment cost
PIPING_STRUCTURAL_FRACTION = 0.30
# Engineering + contingency on top of equipment + piping
ENGINEERING_FRACTION = 0.15
CONTINGENCY_FRACTION = 0.15


def six_tenths_scale(
    reference_cost: float,
    reference_scale: float,
    target_scale: float,
    exponent: float = 0.6,
) -> float:
    """Scale equipment cost using the six-tenths rule.

    C_target = C_ref × (Q_target / Q_ref) ^ exponent
    """
    if target_scale <= 0 or reference_scale <= 0:
        raise ValueError("Scales must be positive")
    return reference_cost * (target_scale / reference_scale) ** exponent


@dataclass
class PilotCAPEXResult:
    """Result of a pilot CAPEX estimate at a given scale."""
    scale_kg_day: float
    equipment: Dict[str, float]        # tag -> scaled cost
    subtotal_equipment: float
    piping_structural: float
    engineering: float
    contingency: float
    total_capex: float

    def to_dict(self) -> dict:
        return {
            "scale_kg_day": self.scale_kg_day,
            "equipment": {k: round(v, 0) for k, v in self.equipment.items()},
            "subtotal_equipment": round(self.subtotal_equipment, 0),
            "piping_structural": round(self.piping_structural, 0),
            "engineering": round(self.engineering, 0),
            "contingency": round(self.contingency, 0),
            "total_capex": round(self.total_capex, 0),
            "total_capex_k": round(self.total_capex / 1e3, 1),
        }


def estimate_capex(
    scale_kg_day: float,
    equipment_list: Optional[List[EquipmentItem]] = None,
) -> PilotCAPEXResult:
    """Estimate total CAPEX for a given production scale.

    Parameters
    ----------
    scale_kg_day : float
        Production target in kg Fe / day.
    equipment_list : list of EquipmentItem, optional
        Override the default P&ID equipment list.

    Returns
    -------
    PilotCAPEXResult
    """
    if equipment_list is None:
        equipment_list = PID_EQUIPMENT

    equipment_costs: Dict[str, float] = {}
    for item in equipment_list:
        scaled = six_tenths_scale(
            item.reference_cost_usd,
            REFERENCE_SCALE,
            scale_kg_day,
            item.scale_exponent,
        )
        equipment_costs[item.tag] = scaled

    subtotal = sum(equipment_costs.values())
    piping = PIPING_STRUCTURAL_FRACTION * subtotal
    eng = ENGINEERING_FRACTION * (subtotal + piping)
    cont = CONTINGENCY_FRACTION * (subtotal + piping + eng)
    total = subtotal + piping + eng + cont

    return PilotCAPEXResult(
        scale_kg_day=scale_kg_day,
        equipment=equipment_costs,
        subtotal_equipment=subtotal,
        piping_structural=piping,
        engineering=eng,
        contingency=cont,
        total_capex=total,
    )


def capex_at_all_scales() -> Dict[str, PilotCAPEXResult]:
    """Compute CAPEX at lab, pilot, and production scales."""
    return {
        "lab": estimate_capex(SCALE_LAB),
        "pilot": estimate_capex(SCALE_PILOT),
        "production": estimate_capex(SCALE_PRODUCTION),
    }


# ─── OPEX Model ──────────────────────────────────────────────────────

@dataclass
class PilotOPEXModel:
    """Operating costs for the pilot plant including post-deposition processing.

    All per-kg or per-year costs in USD.
    """
    # Electricity for electrowinning (from existing techno-economic)
    electricity_price_kWh: float = 0.06          # $/kWh (grid / PPA)
    electrowinning_kWh_per_kg: float = 5.5       # kWh/kg Fe (from cell model)

    # Gas costs (endothermic gas, CH₄, H₂)
    endo_gas_cost_per_m3: float = 0.15           # $/m³ (generated on-site)
    endo_gas_flow_m3_per_hr: float = 0.5         # m³/hr at pilot scale
    endo_gas_hours_per_batch: float = 6.0        # hr per carburizing batch
    ch4_cost_per_m3: float = 0.40                # $/m³ (bottled)
    ch4_flow_m3_per_hr: float = 0.02
    h2_cost_per_m3: float = 1.50                 # $/m³ (bottled)
    h2_flow_m3_per_hr: float = 0.03

    # Furnace electricity
    carburizing_power_kW: float = 15.0           # kW (retort furnace)
    carburizing_hours_per_batch: float = 6.0
    tempering_power_kW: float = 8.0              # kW
    tempering_hours_per_batch: float = 2.0

    # Quench media
    quench_oil_cost_per_kg_product: float = 0.50  # $/kg (consumption + makeup)
    quench_polymer_cost_per_kg_product: float = 0.30

    # Consumables
    electrolyte_makeup_per_kg: float = 0.015     # $/kg Fe
    anode_replacement_per_m2_yr: float = 30.0    # $/m²/yr (DSA recoating)
    membrane_filter_per_m2_yr: float = 20.0      # $/m²/yr
    ore_cost_per_kg: float = 0.04                # $/kg Fe (iron ore)

    # Maintenance
    instrument_maintenance_pct: float = 0.05     # 5% of instrument CAPEX/year
    general_maintenance_pct_capex: float = 0.03  # 3% of total CAPEX/year

    # Labor (scaled)
    labor_per_yr: float = 150_000.0              # $/yr (pilot: 2 operators)

    def gas_cost_per_batch(self) -> float:
        """Total gas cost per carburizing batch."""
        endo = self.endo_gas_cost_per_m3 * self.endo_gas_flow_m3_per_hr * self.endo_gas_hours_per_batch
        ch4 = self.ch4_cost_per_m3 * self.ch4_flow_m3_per_hr * self.carburizing_hours_per_batch
        h2 = self.h2_cost_per_m3 * self.h2_flow_m3_per_hr * self.carburizing_hours_per_batch
        return endo + ch4 + h2

    def furnace_electricity_per_batch(self) -> float:
        """Electricity cost per carburize + temper batch ($)."""
        carb = self.carburizing_power_kW * self.carburizing_hours_per_batch * self.electricity_price_kWh
        temp = self.tempering_power_kW * self.tempering_hours_per_batch * self.electricity_price_kWh
        return carb + temp

    def estimate(
        self,
        scale_kg_day: float,
        capex_result: PilotCAPEXResult,
        quench_type: str = "oil",
        operating_days_per_yr: float = 300.0,
    ) -> Dict[str, float]:
        """Estimate annual OPEX.

        Parameters
        ----------
        scale_kg_day : float
        capex_result : PilotCAPEXResult
            CAPEX result (used for maintenance and instrument costs).
        quench_type : str
            "oil" or "polymer"
        operating_days_per_yr : float

        Returns
        -------
        dict with itemized OPEX
        """
        annual_kg = scale_kg_day * operating_days_per_yr
        annual_t = annual_kg / 1000.0

        # Electrowinning electricity
        ew_elec = self.electrowinning_kWh_per_kg * annual_kg * self.electricity_price_kWh

        # Gas costs (carburizing is per batch; assume 1 batch = ~20 kg at pilot)
        batch_size_kg = max(scale_kg_day * 0.5, 1.0)  # ~half-day production per batch
        batches_per_yr = annual_kg / batch_size_kg
        gas_total = self.gas_cost_per_batch() * batches_per_yr

        # Furnace electricity
        furnace_elec = self.furnace_electricity_per_batch() * batches_per_yr

        # Quench media
        quench_rate = (self.quench_oil_cost_per_kg_product if quench_type == "oil"
                       else self.quench_polymer_cost_per_kg_product)
        quench_cost = quench_rate * annual_kg

        # Consumables
        electrolyte = self.electrolyte_makeup_per_kg * annual_kg
        ore = self.ore_cost_per_kg * annual_kg

        # Electrode area estimate: ~0.1 m² per kg/day capacity
        electrode_area = 0.1 * scale_kg_day
        anode_repl = self.anode_replacement_per_m2_yr * electrode_area
        membrane_repl = self.membrane_filter_per_m2_yr * electrode_area

        # Maintenance
        instrument_cost_tag = capex_result.equipment.get("INST", 0)
        instrument_maint = self.instrument_maintenance_pct * instrument_cost_tag
        general_maint = self.general_maintenance_pct_capex * capex_result.total_capex

        # Labor (scales sub-linearly)
        labor = self.labor_per_yr * (scale_kg_day / SCALE_PILOT) ** 0.4

        # Sum
        variable = ew_elec + gas_total + furnace_elec + quench_cost + electrolyte + ore
        consumable = anode_repl + membrane_repl
        fixed = instrument_maint + general_maint + labor
        total = variable + consumable + fixed

        return {
            "Electrowinning electricity ($/yr)": round(ew_elec, 0),
            "Gas costs ($/yr)": round(gas_total, 0),
            "Furnace electricity ($/yr)": round(furnace_elec, 0),
            "Quench media ($/yr)": round(quench_cost, 0),
            "Electrolyte makeup ($/yr)": round(electrolyte, 0),
            "Iron ore ($/yr)": round(ore, 0),
            "Anode replacement ($/yr)": round(anode_repl, 0),
            "Membrane/filter replacement ($/yr)": round(membrane_repl, 0),
            "Instrument maintenance ($/yr)": round(instrument_maint, 0),
            "General maintenance ($/yr)": round(general_maint, 0),
            "Labor ($/yr)": round(labor, 0),
            "Variable OPEX ($/yr)": round(variable, 0),
            "Consumables ($/yr)": round(consumable, 0),
            "Fixed OPEX ($/yr)": round(fixed, 0),
            "Total OPEX ($/yr)": round(total, 0),
            "Total OPEX (k$/yr)": round(total / 1e3, 1),
            "Specific OPEX ($/kg Fe)": round(total / annual_kg, 3) if annual_kg > 0 else 0,
            "Annual production (t/yr)": round(annual_t, 2),
            "Batches per year": round(batches_per_yr, 0),
        }


# ─── Sensitivity ─────────────────────────────────────────────────────

def capex_sensitivity_tornado(
    base_scale: float = SCALE_PILOT,
    param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Dict[str, float]]:
    """Tornado sensitivity: CAPEX vs key parameters.

    Parameters
    ----------
    base_scale : float
        Base production scale (kg/day).
    param_ranges : dict, optional
        Maps parameter name to (low, high) values. Defaults to:
        - cell_current (cost multiplier on C-201): (0.5, 2.0)
        - carburizing_temp (cost multiplier on F-501): (0.8, 1.5)
        - production_scale: (1.0, 200.0) kg/day

    Returns
    -------
    dict with "base" and "<param>_<low|high>" entries.
    """
    if param_ranges is None:
        param_ranges = {
            "cell_current": (0.5, 2.0),
            "carburizing_temp": (0.8, 1.5),
            "production_scale": (1.0, 200.0),
        }

    base = estimate_capex(base_scale)
    results: Dict[str, Dict[str, float]] = {"base": {"CAPEX": base.total_capex, "scale": base_scale}}

    for param, (low, high) in param_ranges.items():
        for val, tag in [(low, "low"), (high, "high")]:
            if param == "production_scale":
                r = estimate_capex(val)
                results[f"{param}_{tag}"] = {"CAPEX": r.total_capex, "scale": val}
            elif param == "cell_current":
                modified = [
                    EquipmentItem(
                        e.tag, e.name, e.category,
                        e.reference_cost_usd * val if e.tag == "C-201" else e.reference_cost_usd,
                        e.scale_exponent, e.notes,
                    )
                    for e in PID_EQUIPMENT
                ]
                r = estimate_capex(base_scale, modified)
                results[f"{param}_{tag}"] = {"CAPEX": r.total_capex, "value": val}
            elif param == "carburizing_temp":
                modified = [
                    EquipmentItem(
                        e.tag, e.name, e.category,
                        e.reference_cost_usd * val if e.tag == "F-501" else e.reference_cost_usd,
                        e.scale_exponent, e.notes,
                    )
                    for e in PID_EQUIPMENT
                ]
                r = estimate_capex(base_scale, modified)
                results[f"{param}_{tag}"] = {"CAPEX": r.total_capex, "value": val}

    return results


# ─── Category aggregation ────────────────────────────────────────────

def capex_by_category(result: PilotCAPEXResult) -> Dict[str, float]:
    """Aggregate equipment costs by category."""
    cats: Dict[str, float] = {}
    for item in PID_EQUIPMENT:
        cats[item.category] = cats.get(item.category, 0) + result.equipment.get(item.tag, 0)
    return cats


def equipment_table() -> List[Dict[str, object]]:
    """Return the full equipment list as a list of dicts."""
    return [
        {
            "tag": e.tag,
            "name": e.name,
            "category": e.category,
            "ref_cost_usd": e.reference_cost_usd,
            "scale_exponent": e.scale_exponent,
            "notes": e.notes,
        }
        for e in PID_EQUIPMENT
    ]
