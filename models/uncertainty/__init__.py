"""
Uncertainty quantification package for the aqueous electrowinning model chain.

Public API:
    REGISTRY            — central parameter registry (dict of Parameter)
    Parameter           — dataclass for a single parameter entry
    sample_parameters   — draw N parameter vectors
    sobol_sequence      — quasi-random sequence for DOE
    parameter_matrix_to_kwargs — map samples to model constructor kwargs
"""

from .parameter_registry import Parameter, REGISTRY, registry_summary
from .sample import sample_parameters, parameter_matrix_to_kwargs, sobol_sequence

__all__ = [
    "Parameter",
    "REGISTRY",
    "registry_summary",
    "sample_parameters",
    "parameter_matrix_to_kwargs",
    "sobol_sequence",
]
