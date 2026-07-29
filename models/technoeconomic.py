"""
Techno-economic model for aqueous electrowinning of iron/steel.

Calculates levelized cost of iron production and benchmarks against
conventional (BF-BOF) and emerging (H₂-DRI, MOE) routes.

References
----------
- Humbert et al. (2024), J. Sustainable Metallurgy, 10, 1679–1701.
- AWARE process (2024), ChemRxiv.
- DOE H₂-DRI cost estimates (2023–2024).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import json

from .electrochemistry import (
    CellVoltageModel, specific_energy_kWh_per_t,
    specific_energy_kWh_per_kg, FARADAY, M_FE, Z_FE,
    current_density_to_production,
)


# ─── Benchmark Cost Ranges ($/t Fe) ──────────────────────────────────
# These are approximate 2024–2026 ranges from literature
BENCHMARK_COSTS = {
    "BF-BOF": {
        "low": 350, "mid": 450, "high": 600,
        "CO2_t_per_t_Fe": 1.8,
        "notes": "Mature technology; cost varies with coal/ore prices",
    },
    "H2-DRI + EAF": {
        "low": 450, "mid": 600, "high": 800,
        "CO2_t_per_t_Fe": 0.05,
        "notes": "Green H₂ at $2–4/kg; EAF melting adds ~200 kWh/t",
    },
    "Molten Oxide Electrolysis": {
        "low": 600, "mid": 900, "high": 1200,
        "CO2_t_per_t_Fe": 0.02,
        "notes": "Boston Metal approach; high-temp refractory costs",
    },
}


@dataclass
class ElectrolyzerParams:
    """
    Parameters for a single electrolyzer cell / stack.
    """
    # Operating conditions
    current_density_mA_cm2: float = 100.0      # mA/cm²
    current_efficiency: float = 0.90            # fraction (0–1)
    cell_voltage: float = 2.50                  # V (total, incl. overpotentials)
    temperature_C: float = 70.0

    # Cell geometry
    electrode_area_m2: float = 1.0             # per electrode face
    n_cells: int = 100                          # cells in series (bipolar stack)

    # Electrolyte
    electrolyte_type: str = "alkaline"          # "alkaline", "acidic", "chloride"

    def total_current_A(self) -> float:
        return self.current_density_mA_cm2 * 10.0 * self.electrode_area_m2

    def stack_voltage_V(self) -> float:
        return self.cell_voltage * self.n_cells

    def stack_power_kW(self) -> float:
        return self.total_current_A() * self.stack_voltage_V() / 1000.0

    def production_rate_kg_hr(self) -> float:
        """Total stack production rate (kg/hr)."""
        per_cell = current_density_to_production(
            self.current_density_mA_cm2,
            self.electrode_area_m2,
            self.current_efficiency,
        )
        return per_cell * self.n_cells

    def production_rate_t_yr(self, operating_hours: float = 8000.0) -> float:
        """Annual production per stack (tonnes/year)."""
        return self.production_rate_kg_hr() * operating_hours / 1000.0


@dataclass
class CAPEXModel:
    """
    Capital expenditure model for an aqueous electrowinning plant.

    Cost breakdown follows standard electrolyzer plant conventions.
    All costs in USD (2024 basis).
    """
    # Electrolyzer stack
    electrode_cost_per_m2: float = 150.0        # $/m² (DSA anode + cathode substrate)
    membrane_separator_cost_per_m2: float = 80.0 # $/m²
    cell_hardware_cost_per_m2: float = 100.0     # $/m² (frames, gaskets, flow channels)
    stack_assembly_factor: float = 1.15           # multiplier for assembly labor

    # Balance of plant
    rectifier_cost_per_kW: float = 120.0         # $/kW (power electronics)
    electrolyte_system_cost_per_m3: float = 5000.0  # $/m³ (tanks, pumps, heat exchangers)
    electrolyte_volume_per_m2: float = 0.02       # m³ per m² electrode area

    # Ore processing / leaching
    leaching_cost_per_tpy: float = 50.0          # $/t annual capacity (ore dissolution)

    # Infrastructure
    infrastructure_factor: float = 0.25          # fraction of direct costs (buildings, piping, controls)
    engineering_factor: float = 0.15             # fraction of direct costs
    contingency_factor: float = 0.15             # fraction of total direct costs

    def estimate(self, params: ElectrolyzerParams, n_stacks: int = 10) -> dict:
        """
        Estimate total CAPEX for a plant with n_stacks electrolyzer stacks.

        Returns dict with itemized costs.
        """
        area_per_stack = params.electrode_area_m2 * params.n_cells * 2  # both faces

        # Stack costs
        electrode_cost = self.electrode_cost_per_m2 * area_per_stack * n_stacks
        membrane_cost = self.membrane_separator_cost_per_m2 * area_per_stack * n_stacks
        hardware_cost = self.cell_hardware_cost_per_m2 * area_per_stack * n_stacks
        stack_subtotal = (electrode_cost + membrane_cost + hardware_cost) * self.stack_assembly_factor

        # Balance of plant
        total_power_kW = params.stack_power_kW() * n_stacks
        rectifier_cost = self.rectifier_cost_per_kW * total_power_kW
        total_electrolyte_m3 = self.electrolyte_volume_per_m2 * area_per_stack * n_stacks
        electrolyte_system = self.electrolyte_system_cost_per_m3 * total_electrolyte_m3

        # Annual capacity for leaching estimate
        annual_capacity_t = params.production_rate_t_yr() * n_stacks
        leaching = self.leaching_cost_per_tpy * annual_capacity_t

        bop_subtotal = rectifier_cost + electrolyte_system + leaching

        # Direct costs
        direct_costs = stack_subtotal + bop_subtotal
        infrastructure = self.infrastructure_factor * direct_costs
        engineering = self.engineering_factor * direct_costs
        subtotal = direct_costs + infrastructure + engineering
        contingency = self.contingency_factor * subtotal
        total_capex = subtotal + contingency

        return {
            "Electrodes ($)": round(electrode_cost, 0),
            "Membranes/separators ($)": round(membrane_cost, 0),
            "Cell hardware ($)": round(hardware_cost, 0),
            "Stack subtotal ($)": round(stack_subtotal, 0),
            "Rectifiers ($)": round(rectifier_cost, 0),
            "Electrolyte system ($)": round(electrolyte_system, 0),
            "Ore leaching ($)": round(leaching, 0),
            "BOP subtotal ($)": round(bop_subtotal, 0),
            "Infrastructure ($)": round(infrastructure, 0),
            "Engineering ($)": round(engineering, 0),
            "Contingency ($)": round(contingency, 0),
            "Total CAPEX ($)": round(total_capex, 0),
            "Total CAPEX (M$)": round(total_capex / 1e6, 2),
            "Annual capacity (t/yr)": round(annual_capacity_t, 0),
            "n_stacks": n_stacks,
        }


@dataclass
class OPEXModel:
    """
    Operating expenditure model.

    All costs in USD (2024 basis).
    """
    # Electricity
    electricity_price_kWh: float = 0.04           # $/kWh (renewable PPA)

    # Consumables
    electrolyte_makeup_per_t_Fe: float = 15.0     # $/t Fe (losses, purification)
    anode_replacement_cost_per_m2_yr: float = 30.0  # $/m²/yr (DSA recoating)
    ore_cost_per_t_Fe: float = 40.0               # $/t Fe (iron ore feedstock)
    water_cost_per_t_Fe: float = 2.0              # $/t Fe

    # Fixed costs
    maintenance_pct_capex: float = 0.03           # fraction of CAPEX per year
    labor_cost_per_yr: float = 2_000_000          # $/yr (plant operations staff)
    insurance_pct_capex: float = 0.01             # fraction of CAPEX per year
    overhead_pct: float = 0.10                    # fraction of total variable OPEX

    def estimate(
        self,
        params: ElectrolyzerParams,
        capex_total: float,
        n_stacks: int = 10,
        operating_hours: float = 8000.0,
    ) -> dict:
        """Estimate annual OPEX."""
        annual_production_t = params.production_rate_t_yr(operating_hours) * n_stacks
        specific_energy = specific_energy_kWh_per_t(
            params.cell_voltage, params.current_efficiency
        )

        # Variable costs
        electricity_cost = specific_energy * self.electricity_price_kWh * annual_production_t
        electrolyte_cost = self.electrolyte_makeup_per_t_Fe * annual_production_t
        ore_cost = self.ore_cost_per_t_Fe * annual_production_t
        water_cost = self.water_cost_per_t_Fe * annual_production_t

        area_total = params.electrode_area_m2 * params.n_cells * 2 * n_stacks
        anode_cost = self.anode_replacement_cost_per_m2_yr * area_total

        variable_opex = electricity_cost + electrolyte_cost + ore_cost + water_cost + anode_cost

        # Fixed costs
        maintenance = self.maintenance_pct_capex * capex_total
        insurance = self.insurance_pct_capex * capex_total
        labor = self.labor_cost_per_yr
        overhead = self.overhead_pct * variable_opex
        fixed_opex = maintenance + insurance + labor + overhead

        total_opex = variable_opex + fixed_opex

        return {
            "Electricity ($/yr)": round(electricity_cost, 0),
            "Electrolyte makeup ($/yr)": round(electrolyte_cost, 0),
            "Iron ore feedstock ($/yr)": round(ore_cost, 0),
            "Water ($/yr)": round(water_cost, 0),
            "Anode replacement ($/yr)": round(anode_cost, 0),
            "Variable OPEX ($/yr)": round(variable_opex, 0),
            "Maintenance ($/yr)": round(maintenance, 0),
            "Insurance ($/yr)": round(insurance, 0),
            "Labor ($/yr)": round(labor, 0),
            "Overhead ($/yr)": round(overhead, 0),
            "Fixed OPEX ($/yr)": round(fixed_opex, 0),
            "Total OPEX ($/yr)": round(total_opex, 0),
            "Total OPEX (M$/yr)": round(total_opex / 1e6, 2),
            "Annual production (t/yr)": round(annual_production_t, 0),
            "Specific energy (kWh/t Fe)": round(specific_energy, 0),
            "Electricity cost ($/t Fe)": round(
                specific_energy * self.electricity_price_kWh, 2
            ),
        }


@dataclass
class LevelizedCost:
    """
    Levelized cost of iron (LCOFe) calculation.

    Uses a simplified discounted cash flow approach.
    """
    plant_lifetime_yr: int = 25
    discount_rate: float = 0.08          # WACC
    construction_period_yr: int = 2
    capacity_factor: float = 0.913       # ~8000 hrs / 8760 hrs

    def calculate(self, total_capex: float, annual_opex: float,
                  annual_production_t: float) -> dict:
        """
        Calculate levelized cost of iron production ($/t Fe).

        LCOFe = (CRF × CAPEX + OPEX) / Annual Production

        where CRF = r(1+r)^n / ((1+r)^n - 1) is the capital recovery factor.
        """
        r = self.discount_rate
        n = self.plant_lifetime_yr

        # Capital recovery factor
        crf = (r * (1 + r)**n) / ((1 + r)**n - 1)

        # Annualized CAPEX
        annual_capex = crf * total_capex

        # Total annual cost
        total_annual = annual_capex + annual_opex

        # Levelized cost
        lcofe = total_annual / annual_production_t

        return {
            "Capital recovery factor": round(crf, 4),
            "Annualized CAPEX ($/yr)": round(annual_capex, 0),
            "Annual OPEX ($/yr)": round(annual_opex, 0),
            "Total annual cost ($/yr)": round(total_annual, 0),
            "Annual production (t/yr)": round(annual_production_t, 0),
            "LCOFe ($/t Fe)": round(lcofe, 0),
            "CAPEX share (%)": round(annual_capex / total_annual * 100, 1),
            "OPEX share (%)": round(annual_opex / total_annual * 100, 1),
        }


def sensitivity_analysis(
    base_params: ElectrolyzerParams,
    n_stacks: int = 10,
    capex_model: Optional[CAPEXModel] = None,
    opex_model: Optional[OPEXModel] = None,
) -> dict:
    """
    One-at-a-time sensitivity analysis on key parameters.

    Returns LCOFe for each parameter at low/high values.
    """
    if capex_model is None:
        capex_model = CAPEXModel()
    if opex_model is None:
        opex_model = OPEXModel()

    lc_model = LevelizedCost()
    results = {}

    # Base case
    capex = capex_model.estimate(base_params, n_stacks)
    opex = opex_model.estimate(base_params, capex["Total CAPEX ($)"], n_stacks)
    annual_prod = capex["Annual capacity (t/yr)"]
    base_lcofe = lc_model.calculate(capex["Total CAPEX ($)"], opex["Total OPEX ($/yr)"], annual_prod)
    results["base"] = {"LCOFe": base_lcofe["LCOFe ($/t Fe)"]}

    # Sensitivity variables: (attribute_name, low_value, high_value, label)
    sensitivities = [
        ("current_efficiency", 0.70, 0.98, "Current efficiency"),
        ("current_density_mA_cm2", 50.0, 300.0, "Current density (mA/cm²)"),
        ("cell_voltage", 2.0, 3.5, "Cell voltage (V)"),
    ]

    for attr, low, high, label in sensitivities:
        for val, tag in [(low, "low"), (high, "high")]:
            p = ElectrolyzerParams(**{
                **base_params.__dict__,
                attr: val,
            })
            c = capex_model.estimate(p, n_stacks)
            o = opex_model.estimate(p, c["Total CAPEX ($)"], n_stacks)
            prod = c["Annual capacity (t/yr)"]
            l = lc_model.calculate(c["Total CAPEX ($)"], o["Total OPEX ($/yr)"], prod)
            results[f"{label}_{tag}"] = {
                "LCOFe": l["LCOFe ($/t Fe)"],
                "value": val,
                "param": label,
            }

    # Electricity price sensitivity (in OPEX model)
    for price, tag in [(0.02, "low"), (0.08, "high")]:
        om = OPEXModel(electricity_price_kWh=price)
        o = om.estimate(base_params, capex["Total CAPEX ($)"], n_stacks)
        l = lc_model.calculate(capex["Total CAPEX ($)"], o["Total OPEX ($/yr)"], annual_prod)
        results[f"Electricity price_{tag}"] = {
            "LCOFe": l["LCOFe ($/t Fe)"],
            "value": price,
            "param": "Electricity price ($/kWh)",
        }

    return results


def compare_routes(
    aq_lcofe: float,
    carbon_price_tCO2: float = 50.0,
) -> dict:
    """
    Compare aqueous electrowinning against benchmark routes,
    optionally including a carbon price penalty.
    """
    comparison = {}
    for route, data in BENCHMARK_COSTS.items():
        co2_cost = data["CO2_t_per_t_Fe"] * carbon_price_tCO2
        adjusted_mid = data["mid"] + co2_cost
        comparison[route] = {
            "Base cost ($/t Fe)": data["mid"],
            "CO₂ emissions (t/t Fe)": data["CO2_t_per_t_Fe"],
            "Carbon cost @${}/tCO2 ($/t Fe)".format(int(carbon_price_tCO2)): round(co2_cost, 0),
            "Adjusted cost ($/t Fe)": round(adjusted_mid, 0),
            "Cost range ($/t Fe)": f"{data['low']}–{data['high']}",
        }

    comparison["Aqueous Electrowinning"] = {
        "Base cost ($/t Fe)": aq_lcofe,
        "CO₂ emissions (t/t Fe)": 0.02,  # minimal, from grid mix
        "Carbon cost @${}/tCO2 ($/t Fe)".format(int(carbon_price_tCO2)): round(0.02 * carbon_price_tCO2, 0),
        "Adjusted cost ($/t Fe)": round(aq_lcofe + 0.02 * carbon_price_tCO2, 0),
        "Cost range ($/t Fe)": "model-dependent",
    }

    return comparison
