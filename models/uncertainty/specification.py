"""
Specification framework — define and check design requirements.

Formalizes what "the design works" means by defining pass/fail specifications
for each model output and checking them against predictions.  Supports:

* Pre-defined spec sets for common steel standards (ASTM A36, AISI 1010/1020,
  carburized case, electrowinning process)
* Custom specs via YAML files
* Integration with dict-based results (co-deposition, carburization) and
  dataclass results (MechanicalPropertiesResult)
* Monte Carlo result checking with pass-rate statistics and failure diagnosis

Example
-------
>>> from models.uncertainty import SPECS_A36, check_specifications
>>> from models.mechanical_properties import MechanicalPropertiesModel
>>> model = MechanicalPropertiesModel()
>>> result = model.predict(
...     j_avg_mA_cm2=150, waveform="pe", duty_cycle=0.5,
...     ni_wt_percent=2.0, carbon_wt_percent=0.5,
...     current_efficiency_percent=93.0,
... )
>>> report = check_specifications(result, SPECS_A36)
>>> print(report.pass_rate, report.summary_text)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np


# ---------------------------------------------------------------------------
# Specification definition
# ---------------------------------------------------------------------------

class Operator(str, Enum):
    """Comparison operators for specification checks."""
    GE = ">="        # value >= threshold
    LE = "<="        # value <= threshold
    RANGE = "range"  # lower <= value <= upper (threshold = lower, threshold_upper = upper)
    EQ = "=="        # value == threshold (rarely used)


class Criticality(str, Enum):
    """How critical is this specification?"""
    SAFETY = "safety"
    PERFORMANCE = "performance"
    COST = "cost"


@dataclass(frozen=True)
class Specification:
    """
    A single design requirement.

    Parameters
    ----------
    name : str
        Human-readable name (e.g. "Yield strength >= 250 MPa").
    output_key : str
        Dot-separated path into the result dict/dataclass (e.g.
        ``"sigma_y_MPa"`` or ``"alloy_kinetics.ni_wt_percent"``).
        For numpy array values the *last* element is used (final state).
    operator : str
        One of ``>=``, ``<=``, ``range``, ``==``.
    threshold : float
        Primary threshold (lower bound for ``>=``, upper for ``<=``,
        lower for ``range``).
    unit : str
        Human-readable unit (e.g. "MPa", "µm", "wt%", "kWh/kg").
    source : str
        Normative source (e.g. "ASTM A36", "AISI 1010").
    criticality : str
        One of ``safety``, ``performance``, ``cost``.
    threshold_upper : float | None
        Upper bound for ``range`` operator.
    tolerance_pct : float
        Percentage tolerance band around the threshold.
        0 = strict, 5 = ±5% relaxation.  The margin is applied
        in the *passing* direction only (e.g. for ``>=`` the
        effective lower bound is ``threshold * (1 - tol/100)``).
    description : str
        Extended description / rationale.
    """

    name: str
    output_key: str
    operator: str
    threshold: float
    unit: str
    source: str = ""
    criticality: str = "performance"
    threshold_upper: Optional[float] = None
    tolerance_pct: float = 0.0
    description: str = ""

    def __post_init__(self):
        if self.operator not in (">=", "<=", "range", "=="):
            raise ValueError(f"Unknown operator: {self.operator!r}")
        if self.operator == "range" and self.threshold_upper is None:
            raise ValueError("range operator requires threshold_upper")
        if self.tolerance_pct < 0:
            raise ValueError("tolerance_pct must be non-negative")


# ---------------------------------------------------------------------------
# Value extraction from results
# ---------------------------------------------------------------------------

def _extract_value(result: Any, key: str) -> Optional[float]:
    """
    Extract a numeric value from *result* using a dot-separated *key*.

    Works with:
    * Plain dicts (including nested)
    * Dataclass instances (via getattr)
    * Mixed dict/dataclass nesting
    * numpy arrays → returns the last element
    """
    parts = key.split(".")
    current: Any = result

    for part in parts:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
        else:
            try:
                current = getattr(current, part, None)
            except Exception:
                return None

    if current is None:
        return None

    # Unwrap numpy arrays to scalar
    if isinstance(current, np.ndarray):
        if current.size == 0:
            return None
        current = float(current.flat[-1])

    try:
        return float(current)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Specification checking
# ---------------------------------------------------------------------------

@dataclass
class SpecResult:
    """Result of checking a single specification."""

    spec: Specification
    value: Optional[float]
    passed: bool
    margin: Optional[float] = None   # how far from threshold (positive = passing)
    diagnosis: str = ""

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        val = f"{self.value:.4g}" if self.value is not None else "N/A"
        return f"<SpecResult {status} {self.spec.name!r} value={val} {self.spec.unit}>"


@dataclass
class SpecReport:
    """Aggregated specification check report."""

    results: List[SpecResult] = field(default_factory=list)
    spec_set_name: str = ""

    # ── Aggregate metrics ──────────────────────────────────────────────

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    @property
    def all_passed(self) -> bool:
        return self.total > 0 and self.passed == self.total

    # ── Failure diagnosis ──────────────────────────────────────────────

    @property
    def failures(self) -> List[SpecResult]:
        return [r for r in self.results if not r.passed]

    @property
    def safety_failures(self) -> List[SpecResult]:
        return [r for r in self.failures if r.spec.criticality == "safety"]

    @property
    def performance_failures(self) -> List[SpecResult]:
        return [r for r in self.failures if r.spec.criticality == "performance"]

    def diagnose(self) -> List[str]:
        """Return human-readable failure diagnoses."""
        lines: List[str] = []
        for r in self.failures:
            if r.value is None:
                lines.append(
                    f"[{r.spec.criticality.upper()}] {r.spec.name}: "
                    f"output '{r.spec.output_key}' not found in result"
                )
            else:
                op = r.spec.operator
                if op == ">=":
                    deficit = r.spec.threshold - r.value
                    lines.append(
                        f"[{r.spec.criticality.upper()}] {r.spec.name}: "
                        f"{r.value:.4g} {r.spec.unit} < threshold {r.spec.threshold} "
                        f"(deficit {deficit:+.4g} {r.spec.unit})"
                    )
                elif op == "<=":
                    excess = r.value - r.spec.threshold
                    lines.append(
                        f"[{r.spec.criticality.upper()}] {r.spec.name}: "
                        f"{r.value:.4g} {r.spec.unit} > threshold {r.spec.threshold} "
                        f"(excess {excess:+.4g} {r.spec.unit})"
                    )
                elif op == "range":
                    lo, hi = r.spec.threshold, r.spec.threshold_upper
                    if r.value < lo:
                        lines.append(
                            f"[{r.spec.criticality.upper()}] {r.spec.name}: "
                            f"{r.value:.4g} {r.spec.unit} below range [{lo}, {hi}]"
                        )
                    else:
                        lines.append(
                            f"[{r.spec.criticality.upper()}] {r.spec.name}: "
                            f"{r.value:.4g} {r.spec.unit} above range [{lo}, {hi}]"
                        )
                else:
                    lines.append(
                        f"[{r.spec.criticality.upper()}] {r.spec.name}: "
                        f"{r.value:.4g} {r.spec.unit} != {r.spec.threshold}"
                    )
        return lines

    @property
    def summary_text(self) -> str:
        """One-line summary."""
        if self.all_passed:
            return (
                f"ALL {self.total} specs PASSED"
                + (f" ({self.spec_set_name})" if self.spec_set_name else "")
            )
        return (
            f"{self.passed}/{self.total} specs passed"
            + (f" ({self.spec_set_name})" if self.spec_set_name else "")
            + f" — {self.failed} failure(s)"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Machine-readable report."""
        return {
            "spec_set_name": self.spec_set_name,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": round(self.pass_rate, 4),
            "all_passed": self.all_passed,
            "failures": [
                {
                    "name": r.spec.name,
                    "output_key": r.spec.output_key,
                    "value": r.value,
                    "threshold": r.spec.threshold,
                    "threshold_upper": r.spec.threshold_upper,
                    "operator": r.spec.operator,
                    "unit": r.spec.unit,
                    "criticality": r.spec.criticality,
                    "margin": r.margin,
                    "diagnosis": r.diagnosis,
                }
                for r in self.failures
            ],
            "diagnoses": self.diagnose(),
        }


def check_specifications(
    result: Any,
    specs: Sequence[Specification],
    spec_set_name: str = "",
    tolerance_override_pct: Optional[float] = None,
) -> SpecReport:
    """
    Check *result* against a list of *specs*.

    Parameters
    ----------
    result : dict | dataclass | Any
        Model output.  Values are extracted via dot-separated keys
        (see :func:`_extract_value`).
    specs : sequence of Specification
        Specifications to check.
    spec_set_name : str
        Optional label for the report.
    tolerance_override_pct : float | None
        If set, overrides per-spec tolerance for all specs.

    Returns
    -------
    SpecReport
    """
    results: List[SpecResult] = []

    for spec in specs:
        value = _extract_value(result, spec.output_key)
        tol = (
            tolerance_override_pct
            if tolerance_override_pct is not None
            else spec.tolerance_pct
        )

        passed, margin = _check_one(value, spec.operator, spec.threshold,
                                     spec.threshold_upper, tol)

        diagnosis = ""
        if not passed and value is not None:
            diagnosis = _diagnose_one(value, spec)

        results.append(SpecResult(
            spec=spec,
            value=value,
            passed=passed,
            margin=margin,
            diagnosis=diagnosis,
        ))

    return SpecReport(results=results, spec_set_name=spec_set_name)


def _check_one(
    value: Optional[float],
    operator: str,
    threshold: float,
    threshold_upper: Optional[float],
    tolerance_pct: float,
) -> tuple[bool, Optional[float]]:
    """Check a single value against a threshold with tolerance.

    Returns (passed, margin).  Margin is positive when passing.
    """
    if value is None:
        return False, None

    tol = tolerance_pct / 100.0

    if operator == ">=":
        effective = threshold * (1.0 - tol)
        margin = value - effective
        return value >= effective, margin

    if operator == "<=":
        effective = threshold * (1.0 + tol)
        margin = effective - value
        return value <= effective, margin

    if operator == "range":
        lo = threshold * (1.0 - tol)
        hi = threshold_upper * (1.0 + tol)  # type: ignore
        in_range = lo <= value <= hi
        margin = min(value - lo, hi - value)
        return in_range, margin

    if operator == "==":
        # Equality with tolerance: absolute % of threshold
        tol_abs = abs(threshold) * tol if threshold != 0 else tol
        margin = tol_abs - abs(value - threshold)
        return abs(value - threshold) <= max(tol_abs, 1e-12), margin

    return False, None


def _diagnose_one(value: float, spec: Specification) -> str:
    """One-line diagnosis for a failed check."""
    op = spec.operator
    if op == ">=":
        return f"{value:.4g} {spec.unit} < {spec.threshold} (deficit {spec.threshold - value:+.4g})"
    if op == "<=":
        return f"{value:.4g} {spec.unit} > {spec.threshold} (excess {value - spec.threshold:+.4g})"
    if op == "range":
        return f"{value:.4g} {spec.unit} outside [{spec.threshold}, {spec.threshold_upper}]"
    return f"{value:.4g} {spec.unit} != {spec.threshold}"


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def load_specs_from_yaml(path: Union[str, Path]) -> List[Specification]:
    """
    Load specifications from a YAML file.

    Expected format::

        specifications:
          - name: "Yield strength >= 250 MPa"
            output_key: sigma_y_MPa
            operator: ">="
            threshold: 250
            unit: MPa
            source: "ASTM A36"
            criticality: safety
            tolerance_pct: 5
            description: "Minimum yield strength for structural steel"

    Parameters
    ----------
    path : str or Path
        Path to the YAML file.

    Returns
    -------
    list of Specification
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for YAML spec loading. "
            "Install with: pip install pyyaml"
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Spec YAML not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "specifications" not in data:
        raise ValueError(
            "YAML spec file must have a top-level 'specifications' key "
            "containing a list of spec definitions"
        )

    specs: List[Specification] = []
    for item in data["specifications"]:
        if not isinstance(item, dict):
            raise ValueError(f"Each spec must be a mapping, got {type(item)}")
        specs.append(Specification(
            name=str(item["name"]),
            output_key=str(item["output_key"]),
            operator=str(item["operator"]),
            threshold=float(item["threshold"]),
            unit=str(item.get("unit", "")),
            source=str(item.get("source", "")),
            criticality=str(item.get("criticality", "performance")),
            threshold_upper=(
                float(item["threshold_upper"])
                if "threshold_upper" in item
                else None
            ),
            tolerance_pct=float(item.get("tolerance_pct", 0.0)),
            description=str(item.get("description", "")),
        ))

    return specs


def load_specs_from_json(path: Union[str, Path]) -> List[Specification]:
    """Load specifications from a JSON file (same schema as YAML)."""
    import json

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Spec JSON not found: {path}")

    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, dict) or "specifications" not in data:
        raise ValueError("JSON spec file must have 'specifications' key")

    specs: List[Specification] = []
    for item in data["specifications"]:
        specs.append(Specification(
            name=str(item["name"]),
            output_key=str(item["output_key"]),
            operator=str(item["operator"]),
            threshold=float(item["threshold"]),
            unit=str(item.get("unit", "")),
            source=str(item.get("source", "")),
            criticality=str(item.get("criticality", "performance")),
            threshold_upper=(
                float(item["threshold_upper"])
                if "threshold_upper" in item
                else None
            ),
            tolerance_pct=float(item.get("tolerance_pct", 0.0)),
            description=str(item.get("description", "")),
        ))

    return specs


# ---------------------------------------------------------------------------
# Pre-defined specification sets
# ---------------------------------------------------------------------------

SPECS_A36: List[Specification] = [
    Specification(
        name="ASTM A36 yield strength",
        output_key="sigma_y_MPa",
        operator=">=",
        threshold=250.0,
        unit="MPa",
        source="ASTM A36",
        criticality="safety",
        tolerance_pct=5.0,
        description="Minimum yield strength for ASTM A36 structural steel",
    ),
    Specification(
        name="ASTM A36 ultimate tensile strength",
        output_key="uts_MPa",
        operator=">=",
        threshold=400.0,
        unit="MPa",
        source="ASTM A36",
        criticality="safety",
        tolerance_pct=5.0,
        description="Minimum UTS for ASTM A36",
    ),
    Specification(
        name="ASTM A36 elongation",
        output_key="elongation_pct",
        operator=">=",
        threshold=20.0,
        unit="%",
        source="ASTM A36",
        criticality="performance",
        tolerance_pct=10.0,
        description="Minimum elongation for structural ductility",
    ),
    Specification(
        name="ASTM A36 hardness range",
        output_key="vickers_hv",
        operator="range",
        threshold=120.0,
        threshold_upper=180.0,
        unit="HV",
        source="ASTM A36 (typical)",
        criticality="performance",
        tolerance_pct=10.0,
        description="Typical Vickers hardness range for normalized A36",
    ),
]


SPECS_1010: List[Specification] = [
    Specification(
        name="AISI 1010 yield strength",
        output_key="sigma_y_MPa",
        operator=">=",
        threshold=305.0,
        unit="MPa",
        source="AISI 1010 (cold-rolled)",
        criticality="safety",
        tolerance_pct=5.0,
        description="Minimum yield strength for cold-rolled AISI 1010",
    ),
    Specification(
        name="AISI 1010 ultimate tensile strength",
        output_key="uts_MPa",
        operator=">=",
        threshold=365.0,
        unit="MPa",
        source="AISI 1010 (cold-rolled)",
        criticality="safety",
        tolerance_pct=5.0,
        description="Minimum UTS for cold-rolled AISI 1010",
    ),
    Specification(
        name="AISI 1010 elongation",
        output_key="elongation_pct",
        operator=">=",
        threshold=20.0,
        unit="%",
        source="AISI 1010",
        criticality="performance",
        tolerance_pct=10.0,
        description="Minimum elongation for AISI 1010",
    ),
]


SPECS_1020: List[Specification] = [
    Specification(
        name="AISI 1020 yield strength",
        output_key="sigma_y_MPa",
        operator=">=",
        threshold=350.0,
        unit="MPa",
        source="AISI 1020 (cold-drawn)",
        criticality="safety",
        tolerance_pct=5.0,
        description="Minimum yield strength for cold-drawn AISI 1020",
    ),
    Specification(
        name="AISI 1020 ultimate tensile strength",
        output_key="uts_MPa",
        operator=">=",
        threshold=420.0,
        unit="MPa",
        source="AISI 1020 (cold-drawn)",
        criticality="safety",
        tolerance_pct=5.0,
        description="Minimum UTS for cold-drawn AISI 1020",
    ),
    Specification(
        name="AISI 1020 elongation",
        output_key="elongation_pct",
        operator=">=",
        threshold=15.0,
        unit="%",
        source="AISI 1020",
        criticality="performance",
        tolerance_pct=10.0,
        description="Minimum elongation for AISI 1020",
    ),
]


SPECS_CARBURIZED: List[Specification] = [
    Specification(
        name="Carburized case depth",
        output_key="final_case_depth_035_um",
        operator=">=",
        threshold=500.0,
        unit="µm",
        source="Typical automotive case-hardening",
        criticality="performance",
        tolerance_pct=10.0,
        description="Minimum effective case depth (0.35 wt% C threshold)",
    ),
    Specification(
        name="Carburized surface hardness",
        output_key="final_surface_hv",
        operator=">=",
        threshold=700.0,
        unit="HV",
        source="Typical carburized surface requirement",
        criticality="safety",
        tolerance_pct=5.0,
        description="Minimum surface Vickers hardness after carburizing",
    ),
    Specification(
        name="Carburized core carbon",
        output_key="final_core_c_wt",
        operator="range",
        threshold=0.10,
        threshold_upper=0.30,
        unit="wt%",
        source="Typical case-hardening steel",
        criticality="performance",
        tolerance_pct=20.0,
        description="Core carbon content should remain low for toughness",
    ),
    Specification(
        name="Surface carbon content",
        output_key="surface_C_wt_percent",
        operator="range",
        threshold=0.6,
        threshold_upper=1.2,
        unit="wt%",
        source="Carburizing practice",
        criticality="performance",
        tolerance_pct=10.0,
        description="Surface carbon for martensitic hardening without excessive cementite",
    ),
]


SPECS_ELECTROWINNING: List[Specification] = [
    Specification(
        name="Faradaic efficiency",
        output_key="integrated_metrics.adjusted_overall_current_efficiency_percent",
        operator=">=",
        threshold=70.0,
        unit="%",
        source="Process viability target",
        criticality="performance",
        tolerance_pct=5.0,
        description="Minimum current efficiency for economic viability",
    ),
    Specification(
        name="Specific energy consumption",
        output_key="specific_energy_kWh_per_kg",
        operator="<=",
        threshold=8.0,
        unit="kWh/kg",
        source="Techno-economic target",
        criticality="cost",
        tolerance_pct=10.0,
        description="Maximum specific energy for cost-competitive iron production",
    ),
    Specification(
        name="Nickel impurity limit",
        output_key="alloy_kinetics.ni_wt_percent",
        operator="<=",
        threshold=2.0,
        unit="wt%",
        source="Low-alloy steel purity",
        criticality="performance",
        tolerance_pct=20.0,
        description="Max Ni content for plain-carbon steel classification",
    ),
    Specification(
        name="Carbon incorporation",
        output_key="carbon_incorporation.predicted_carbon_wt_percent",
        operator="range",
        threshold=0.05,
        threshold_upper=2.0,
        unit="wt%",
        source="Composite deposit design",
        criticality="performance",
        tolerance_pct=10.0,
        description="Carbon particle incorporation in useful range",
    ),
]


ALL_STANDARD_SPECS: Dict[str, List[Specification]] = {
    "ASTM_A36": SPECS_A36,
    "AISI_1010": SPECS_1010,
    "AISI_1020": SPECS_1020,
    "CARBURIZED": SPECS_CARBURIZED,
    "ELECTROWINNING": SPECS_ELECTROWINNING,
}


# ---------------------------------------------------------------------------
# Monte Carlo integration helpers
# ---------------------------------------------------------------------------

def check_mc_specifications(
    mc_results: Sequence[Any],
    specs: Sequence[Specification],
    spec_set_name: str = "",
) -> Dict[str, Any]:
    """
    Check specifications across a set of Monte Carlo samples.

    Parameters
    ----------
    mc_results : sequence of result objects (dicts or dataclasses)
        Each element is one MC sample's output.
    specs : sequence of Specification
        Specifications to check.
    spec_set_name : str
        Optional label.

    Returns
    -------
    dict with keys:
        individual_reports : list of SpecReport
        pass_rates : dict mapping spec name -> fraction passed
        overall_pass_rate : fraction of samples where ALL specs pass
        failure_histogram : dict mapping spec name -> count of failures
        worst_spec : name of the spec that fails most often
    """
    reports = [check_specifications(r, specs, spec_set_name) for r in mc_results]
    n_samples = len(reports)
    if n_samples == 0:
        return {
            "individual_reports": [],
            "pass_rates": {},
            "overall_pass_rate": 0.0,
            "failure_histogram": {},
            "worst_spec": None,
        }

    # Per-spec pass rates
    pass_rates: Dict[str, float] = {}
    failure_counts: Dict[str, int] = {}
    for spec in specs:
        n_pass = sum(
            1 for rpt in reports
            for sr in rpt.results
            if sr.spec.name == spec.name and sr.passed
        )
        pass_rates[spec.name] = n_pass / n_samples
        failure_counts[spec.name] = n_samples - n_pass

    # Overall: all specs pass in a sample
    overall = sum(1 for rpt in reports if rpt.all_passed) / n_samples

    # Worst spec
    worst = max(failure_counts, key=failure_counts.get) if failure_counts else None  # type: ignore

    return {
        "individual_reports": reports,
        "pass_rates": pass_rates,
        "overall_pass_rate": overall,
        "failure_histogram": failure_counts,
        "worst_spec": worst,
    }
