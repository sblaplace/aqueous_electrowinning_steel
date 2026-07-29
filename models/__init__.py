"""
models — Electrochemical, transport, transient pulse, and techno-economic modeling for aqueous electrowinning.
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
from .transport import (
    NernstPlanckFilm,
    NernstPlanckState,
    FilmProfile,
    compare_support_levels,
)
from .pulse import (
    PulseDepositionModel,
    PulseWaveform,
    PulseResult,
    compare_dc_vs_pulse,
)
from .eis import (
    RandlesFit,
    load_spectrum,
    summarize_spectrum,
    cpe_impedance,
    warburg_impedance,
    randles_impedance,
    randles_cpe_impedance,
    fit_randles_spectrum,
    exchange_current_from_rct,
    synthetic_randles_spectrum,
)
from .hull_cell import (
    HullCellGeometry,
    GravimetricFEResult,
    hull_current_distribution,
    summarize_hull_distribution,
    current_density_window,
    load_galvanostatic_trace,
    load_gravimetry,
    cathodic_charge_C,
    gravimetric_faradaic_efficiency,
    analyze_gravimetric_efficiency,
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
    "NernstPlanckFilm",
    "NernstPlanckState",
    "FilmProfile",
    "compare_support_levels",
    "PulseDepositionModel",
    "PulseWaveform",
    "PulseResult",
    "compare_dc_vs_pulse",
    "RandlesFit",
    "load_spectrum",
    "summarize_spectrum",
    "cpe_impedance",
    "warburg_impedance",
    "randles_impedance",
    "randles_cpe_impedance",
    "fit_randles_spectrum",
    "exchange_current_from_rct",
    "synthetic_randles_spectrum",
    "HullCellGeometry",
    "GravimetricFEResult",
    "hull_current_distribution",
    "summarize_hull_distribution",
    "current_density_window",
    "load_galvanostatic_trace",
    "load_gravimetry",
    "cathodic_charge_C",
    "gravimetric_faradaic_efficiency",
    "analyze_gravimetric_efficiency",
    "ElectrolyzerParams",
    "CAPEXModel",
    "OPEXModel",
    "LevelizedCost",
    "sensitivity_analysis",
    "compare_routes",
    "BENCHMARK_COSTS",
]
