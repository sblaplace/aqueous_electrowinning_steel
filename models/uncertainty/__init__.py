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
from .validation_planner import (
    Experiment,
    ValidationPlan,
    UncertaintyTrajectory,
    experiment_catalog,
    plan_validation_experiments,
    sequential_planner,
    uncertainty_reduction_trajectory,
)
from .fmea import (
    FailureMode,
    FMEAReport,
    generate_fmea,
    critical_failure_paths,
    mitigation_roadmap,
)
from .design_space import (
    DesignSpaceResult,
    RobustOptimum,
    ParetoFront,
    explore_design_space,
    robust_optimum,
    pareto_front_robust,
)
from .confidence_report import (
    ConfidenceReport,
    generate_confidence_report,
    qualification_verdict,
)
from .sensitivity import (
    SobolResult,
    TornadoResult,
    MorrisResult,
    sobol_analysis,
    tornado_chart,
    morris_screening,
)

__all__ = [
    "Parameter",
    "REGISTRY",
    "registry_summary",
    "sample_parameters",
    "parameter_matrix_to_kwargs",
    "sobol_sequence",
    "Experiment",
    "ValidationPlan",
    "UncertaintyTrajectory",
    "experiment_catalog",
    "plan_validation_experiments",
    "sequential_planner",
    "uncertainty_reduction_trajectory",
    "FailureMode",
    "FMEAReport",
    "generate_fmea",
    "critical_failure_paths",
    "mitigation_roadmap",
    "DesignSpaceResult",
    "RobustOptimum",
    "ParetoFront",
    "explore_design_space",
    "robust_optimum",
    "pareto_front_robust",
    "ConfidenceReport",
    "generate_confidence_report",
    "qualification_verdict",
    "SobolResult",
    "TornadoResult",
    "MorrisResult",
    "sobol_analysis",
    "tornado_chart",
    "morris_screening",
]
