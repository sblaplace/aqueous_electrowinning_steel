"""
Uncertainty and specification checking for electrowinning models.

Public API:
    REGISTRY            — central parameter registry (dict of Parameter)
    Parameter           — dataclass for a single parameter entry
    sample_parameters   — draw N parameter vectors
    sobol_sequence      — quasi-random sequence for DOE
    parameter_matrix_to_kwargs — map samples to model constructor kwargs
    MonteCarloEngine    — propagate parameter uncertainty through full model chain
    MonteCarloResult    — structured result container
"""

from .parameter_registry import Parameter, REGISTRY, registry_summary
from .sample import sample_parameters, parameter_matrix_to_kwargs, sobol_sequence
from .specification import (
    Specification,
    SpecResult,
    SpecReport,
    load_specs_from_yaml,
    check_specifications,
    SPECS_A36,
    SPECS_1010,
    SPECS_1020,
    SPECS_CARBURIZED,
    SPECS_ELECTROWINNING,
    ALL_STANDARD_SPECS,
)
from .monte_carlo import MonteCarloEngine, MonteCarloResult

__all__ = [
    "Parameter",
    "REGISTRY",
    "registry_summary",
    "sample_parameters",
    "parameter_matrix_to_kwargs",
    "sobol_sequence",
    "Specification",
    "SpecResult",
    "SpecReport",
    "load_specs_from_yaml",
    "check_specifications",
    "SPECS_A36",
    "SPECS_1010",
    "SPECS_1020",
    "SPECS_CARBURIZED",
    "SPECS_ELECTROWINNING",
    "ALL_STANDARD_SPECS",
    "MonteCarloEngine",
    "MonteCarloResult",
]
