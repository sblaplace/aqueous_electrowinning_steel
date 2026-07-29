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
from .pourbaix import (
    FePourbaix,
    her_line,
    oer_line,
    nernst_pH_line,
)
from .kinetics import (
    DepositionKinetics,
    TafelBranch,
    limiting_current_density,
)
from .boundary_layer import (
    CathodeBoundaryLayer,
    BoundaryLayerState,
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
    "FePourbaix",
    "her_line",
    "oer_line",
    "nernst_pH_line",
    "DepositionKinetics",
    "TafelBranch",
    "limiting_current_density",
    "CathodeBoundaryLayer",
    "BoundaryLayerState",
    "ElectrolyzerParams",
    "CAPEXModel",
    "OPEXModel",
    "LevelizedCost",
    "sensitivity_analysis",
    "compare_routes",
    "BENCHMARK_COSTS",
]
