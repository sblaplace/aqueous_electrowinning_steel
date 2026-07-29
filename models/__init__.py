"""
models — Electrochemical and techno-economic modeling for aqueous electrowinning.
"""

from .electrochemistry import (
    CellVoltageModel,
    specific_energy_kWh_per_kg,
    specific_energy_kWh_per_t,
    production_rate_kg_per_hr,
    current_density_to_production,
)
from .technoeconomic import (
    ElectrolyzerParams,
    CAPEXModel,
    OPEXModel,
    LevelizedCost,
    sensitivity_analysis,
    compare_routes,
    BENCHMARK_COSTS,
)

__all__ = [
    "CellVoltageModel",
    "specific_energy_kWh_per_kg",
    "specific_energy_kWh_per_t",
    "production_rate_kg_per_hr",
    "current_density_to_production",
    "ElectrolyzerParams",
    "CAPEXModel",
    "OPEXModel",
    "LevelizedCost",
    "sensitivity_analysis",
    "compare_routes",
    "BENCHMARK_COSTS",
]
