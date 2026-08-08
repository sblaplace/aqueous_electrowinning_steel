"""
Confidence report — end-to-end design qualification.

For a given design point, produces a single-page confidence report:
'This design meets ASTM A36 with 97% confidence. Top 3 uncertainty
drivers: X, Y, Z. Recommended validation: A, B, C. Failure modes
ranked by RPN.'

Pipeline
--------
1. Sample parameters from registry
2. Run Monte Carlo
3. Check specifications
4. Compute sensitivity
5. Run FMEA
6. Generate validation plan
7. Produce confidence report

References:
  - IEC 61025 — Fault tree analysis
  - IEC 60812:2018 — FMEA
  - ASTM E2020 — Standard guide for FMEA
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .parameter_registry import Parameter, REGISTRY
from .specification import (
    Specification,
    SPECS_A36,
    SPECS_1010,
    SPECS_1020,
    SPECS_CARBURIZED,
    ALL_STANDARD_SPECS,
)
from .monte_carlo import (
    MonteCarloEngine,
    MonteCarloResult,
    DEFAULT_DESIGN_POINT,
)
from .fmea import (
    FMEAReport,
    FailureMode,
    generate_fmea,
    critical_failure_paths,
)
from .validation_planner import (
    ValidationPlan,
    Experiment,
    plan_validation_experiments,
)


# ---------------------------------------------------------------------------
# Spec set lookup
# ---------------------------------------------------------------------------

SPEC_SET_MAP: Dict[str, List[Specification]] = {
    "A36": SPECS_A36,
    "ASTM_A36": SPECS_A36,
    "1010": SPECS_1010,
    "AISI_1010": SPECS_1010,
    "1020": SPECS_1020,
    "AISI_1020": SPECS_1020,
    "CARBURIZED": SPECS_CARBURIZED,
}

# Canonical names for each spec set
_CANONICAL_NAMES: Dict[str, str] = {
    "A36": "ASTM_A36",
    "ASTM_A36": "ASTM_A36",
    "1010": "AISI_1010",
    "AISI_1010": "AISI_1010",
    "1020": "AISI_1020",
    "AISI_1020": "AISI_1020",
    "CARBURIZED": "CARBURIZED",
}


def _resolve_spec_set(spec_set: str) -> Tuple[List[Specification], str]:
    """Resolve a spec set name to (specs, canonical_name).

    Raises KeyError if the spec set is unknown.
    """
    key = spec_set.upper().replace("-", "_").replace(" ", "_")
    if key in SPEC_SET_MAP:
        return SPEC_SET_MAP[key], _CANONICAL_NAMES[key]
    if key in ALL_STANDARD_SPECS:
        return ALL_STANDARD_SPECS[key], key
    raise KeyError(
        f"Unknown spec set '{spec_set}'. "
        f"Available: {list(SPEC_SET_MAP.keys())}"
    )


# ---------------------------------------------------------------------------
# Top uncertainty drivers
# ---------------------------------------------------------------------------

def _top_uncertainty_drivers(
    mc_result: MonteCarloResult,
    top_n: int = 5,
) -> List[Tuple[str, float]]:
    """Identify top uncertainty drivers from MC sensitivity.

    Aggregates correlation-based sensitivity across all outputs.
    Returns list of (param_name, aggregate_importance) sorted descending.
    """
    param_importance: Dict[str, float] = {}
    for output_key, param_scores in mc_result.sensitivity.items():
        for param_name, score in param_scores.items():
            if param_name not in param_importance:
                param_importance[param_name] = 0.0
            param_importance[param_name] += abs(score)

    ranked = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


# ---------------------------------------------------------------------------
# Design margins
# ---------------------------------------------------------------------------

def _compute_design_margins(
    mc_result: MonteCarloResult,
    specs: List[Specification],
) -> Dict[str, float]:
    """Compute design margins for each specification.

    Margin = (median_output - threshold) / threshold for >= specs.
    Positive = passing with margin.  Negative = failing on median.
    """
    margins: Dict[str, float] = {}
    for spec in specs:
        arr = mc_result.output_distributions.get(spec.output_key)
        if arr is None:
            continue
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            continue
        median = float(np.median(valid))
        if spec.operator == ">=":
            if spec.threshold != 0:
                margins[spec.name] = (median - spec.threshold) / abs(spec.threshold)
            else:
                margins[spec.name] = median
        elif spec.operator == "<=":
            if spec.threshold != 0:
                margins[spec.name] = (spec.threshold - median) / abs(spec.threshold)
            else:
                margins[spec.name] = -median
        elif spec.operator == "range":
            lo = spec.threshold
            hi = spec.threshold_upper
            if hi is not None and hi != lo:
                center = (lo + hi) / 2.0
                half_width = (hi - lo) / 2.0
                margins[spec.name] = (half_width - abs(median - center)) / half_width
            else:
                margins[spec.name] = 0.0
        else:
            margins[spec.name] = 0.0

    return margins


# ---------------------------------------------------------------------------
# ConfidenceReport dataclass
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceReport:
    """End-to-end design qualification report.

    Integrates Monte Carlo propagation, specification checking,
    sensitivity analysis, FMEA, and validation planning into a
    single coherent confidence assessment.
    """

    design_point: Dict[str, Any]
    spec_set_name: str
    specs: List[Specification]
    mc_result: MonteCarloResult
    sensitivity: Dict[str, Dict[str, float]]
    fmea: FMEAReport
    validation_plan: ValidationPlan
    overall_confidence: float
    spec_pass_rates: Dict[str, float]
    top_uncertainty_drivers: List[Tuple[str, float]]
    critical_failures: List[FailureMode]
    recommended_experiments: List[Experiment]
    design_margins: Dict[str, float]
    verdict: str = ""

    def summary_text(self) -> str:
        """One-line human-readable summary."""
        pct = self.overall_confidence * 100.0
        top3 = ", ".join(name for name, _ in self.top_uncertainty_drivers[:3])
        return (
            f"This design meets {self.spec_set_name} with {pct:.0f}% confidence. "
            f"Top 3 uncertainty drivers: {top3}. "
            f"Verdict: {self.verdict}."
        )

    def to_dict(self) -> Dict[str, Any]:
        """Machine-readable report."""
        return {
            "design_point": self.design_point,
            "spec_set_name": self.spec_set_name,
            "overall_confidence": round(self.overall_confidence, 4),
            "verdict": self.verdict,
            "spec_pass_rates": {k: round(v, 4) for k, v in self.spec_pass_rates.items()},
            "design_margins": {k: round(v, 4) for k, v in self.design_margins.items()},
            "top_uncertainty_drivers": [
                {"parameter": name, "importance": round(imp, 4)}
                for name, imp in self.top_uncertainty_drivers
            ],
            "critical_failures": [
                {"id": fm.id, "mode": fm.mode, "component": fm.component, "rpn": fm.rpn}
                for fm in self.critical_failures
            ],
            "recommended_experiments": [
                {"name": e.name, "cost_usd": e.cost_usd, "description": e.description}
                for e in self.recommended_experiments
            ],
            "mc_summary": self.mc_result.summary_dict(),
            "fmea_summary": self.fmea.to_dict(),
            "validation_plan_summary": {
                "total_cost_usd": self.validation_plan.total_cost_usd,
                "total_duration_hours": self.validation_plan.total_duration_hours,
                "n_experiments": len(self.validation_plan.experiments),
            },
        }


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def qualification_verdict(report: ConfidenceReport, target: float = 0.95) -> str:
    """Determine the qualification verdict.

    Rules:
    - PASS: overall_confidence >= target AND no safety-critical failures
    - CONDITIONAL PASS: overall_confidence >= target * 0.85 OR no safety
      failures but marginal confidence
    - FAIL: otherwise

    Parameters
    ----------
    report : ConfidenceReport
    target : float
        Target confidence level (default 0.95).

    Returns
    -------
    str
        'PASS', 'CONDITIONAL PASS', or 'FAIL'
    """
    conf = report.overall_confidence
    safety_fails = [fm for fm in report.critical_failures
                    if fm.severity >= 9 and fm.rpn > 150]

    # Hard fail: safety-critical failures with high RPN
    if safety_fails and conf < target:
        return "FAIL"

    # PASS: meets target with no safety issues
    if conf >= target and not safety_fails:
        return "PASS"

    # CONDITIONAL PASS: close to target or has safety mitigations
    if conf >= target * 0.85:
        return "CONDITIONAL PASS"

    return "FAIL"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_confidence_report(
    design_point: Optional[Dict[str, Any]] = None,
    spec_set: str = "A36",
    mc_samples: int = 10_000,
    target: float = 0.95,
    seed: int = 42,
    n_jobs: int = -1,
    registry: Optional[Dict[str, Parameter]] = None,
) -> ConfidenceReport:
    """Generate a confidence report for a design point.

    Runs the full pipeline:
    1. Monte Carlo uncertainty propagation
    2. Specification checking
    3. Sensitivity analysis (correlation-based from MC + tornado)
    4. FMEA with MC-adjusted occurrence
    5. Validation experiment planning
    6. Confidence verdict

    Parameters
    ----------
    design_point : dict, optional
        Operating conditions.  Defaults to DEFAULT_DESIGN_POINT.
    spec_set : str
        Specification set name ('A36', '1010', '1020', 'CARBURIZED').
    mc_samples : int
        Number of Monte Carlo samples (default 10 000).
    target : float
        Target confidence level for verdict (default 0.95).
    seed : int
        Random seed for reproducibility.
    n_jobs : int
        Parallel workers for MC (-1 = all cores).
    registry : dict, optional
        Parameter registry.  Defaults to REGISTRY.

    Returns
    -------
    ConfidenceReport
    """
    dp = design_point or dict(DEFAULT_DESIGN_POINT)
    reg = registry or REGISTRY

    # ── 1. Resolve specs ────────────────────────────────────────────
    specs, canonical_name = _resolve_spec_set(spec_set)

    # ── 2. Run Monte Carlo ──────────────────────────────────────────
    engine = MonteCarloEngine(
        n_samples=mc_samples, seed=seed, n_jobs=n_jobs, registry=reg,
    )
    mc_result = engine.run(
        design_point=dp, specs=specs, spec_set_name=canonical_name,
    )

    # ── 3. Sensitivity (already computed in MC) ─────────────────────
    sensitivity = mc_result.sensitivity

    # ── 4. FMEA ─────────────────────────────────────────────────────
    fmea = generate_fmea(design_point=dp, mc_result=mc_result)

    # ── 5. Validation plan ──────────────────────────────────────────
    validation_plan = plan_validation_experiments(
        registry=reg, sensitivity=sensitivity,
    )

    # ── 6. Aggregate metrics ────────────────────────────────────────
    overall_confidence = mc_result.overall_confidence
    spec_pass_rates = mc_result.pass_rates

    top_drivers = _top_uncertainty_drivers(mc_result, top_n=5)

    critical = critical_failure_paths(fmea, rpn_threshold=100)

    recommended = validation_plan.experiments[:5]  # top-5 by gain/$

    margins = _compute_design_margins(mc_result, specs)

    # ── 7. Build report ─────────────────────────────────────────────
    report = ConfidenceReport(
        design_point=dp,
        spec_set_name=canonical_name,
        specs=specs,
        mc_result=mc_result,
        sensitivity=sensitivity,
        fmea=fmea,
        validation_plan=validation_plan,
        overall_confidence=overall_confidence,
        spec_pass_rates=spec_pass_rates,
        top_uncertainty_drivers=top_drivers,
        critical_failures=critical,
        recommended_experiments=recommended,
        design_margins=margins,
    )

    # ── 8. Verdict ──────────────────────────────────────────────────
    report.verdict = qualification_verdict(report, target=target)

    return report


# ---------------------------------------------------------------------------
# Summary figure (4 quadrants)
# ---------------------------------------------------------------------------

def plot_confidence_report(
    report: ConfidenceReport,
    out_path: str | None = None,
    figsize: Tuple[float, float] = (16, 12),
) -> None:
    """Generate the 4-quadrant summary figure.

    Top-left:     spec pass rates (green/red bars)
    Top-right:    sensitivity tornado (top 5)
    Bottom-left:  output distributions with spec boundaries
    Bottom-right: risk matrix (severity vs occurrence)

    Parameters
    ----------
    report : ConfidenceReport
    out_path : str, optional
        File path to save the figure.  If None, uses default.
    figsize : tuple
        Figure size (width, height) in inches.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        f"Confidence Report — {report.spec_set_name}  "
        f"[{report.verdict}]  "
        f"Confidence: {report.overall_confidence * 100:.1f}%",
        fontsize=14, fontweight="bold",
    )

    # ── Top-left: Spec pass rates ───────────────────────────────────
    ax = axes[0, 0]
    spec_names = list(report.spec_pass_rates.keys())
    pass_vals = [report.spec_pass_rates[k] * 100.0 for k in spec_names]
    colors = ["#2ecc71" if v >= 95 else "#e74c3c" for v in pass_vals]
    short_names = [n.replace("ASTM ", "").replace("AISI ", "")[:25] for n in spec_names]
    y_pos = range(len(spec_names))
    ax.barh(y_pos, pass_vals, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel("Pass rate (%)")
    ax.set_title("Specification Pass Rates")
    ax.axvline(x=95, color="gray", linestyle="--", alpha=0.5, label="95% target")
    ax.set_xlim(0, 105)
    ax.legend(fontsize=7)

    # ── Top-right: Sensitivity tornado ──────────────────────────────
    ax = axes[0, 1]
    if report.top_uncertainty_drivers:
        names = [d[0] for d in report.top_uncertainty_drivers[:5]]
        vals = [d[1] for d in report.top_uncertainty_drivers[:5]]
        # Normalise to max=1
        max_val = max(vals) if vals else 1.0
        norm_vals = [v / max_val for v in vals]
        colors_t = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(names)))
        y_pos = range(len(names))
        ax.barh(y_pos, norm_vals, color=colors_t, edgecolor="white", height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Normalised importance")
        ax.set_title("Top Uncertainty Drivers")
    else:
        ax.text(0.5, 0.5, "No sensitivity data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Top Uncertainty Drivers")

    # ── Bottom-left: Output distributions with spec boundaries ──────
    ax = axes[1, 0]
    key_outputs = ["sigma_y_MPa", "uts_MPa", "vickers_hv", "elongation_pct"]
    plotted = 0
    for out_key in key_outputs:
        arr = report.mc_result.output_distributions.get(out_key)
        if arr is None:
            continue
        valid = arr[~np.isnan(arr)]
        if len(valid) < 10:
            continue
        # Normalise to [0, 1] for overlay
        lo, hi = np.percentile(valid, [1, 99])
        if hi - lo < 1e-10:
            continue
        normed = (valid - lo) / (hi - lo)
        ax.hist(normed, bins=40, alpha=0.5, label=out_key.replace("_", " "), density=True)
        plotted += 1

    # Mark spec boundaries
    for spec in report.specs:
        if spec.output_key in key_outputs and spec.operator == ">=":
            arr = report.mc_result.output_distributions.get(spec.output_key)
            if arr is not None:
                valid = arr[~np.isnan(arr)]
                lo, hi = np.percentile(valid, [1, 99])
                if hi - lo > 1e-10:
                    norm_thresh = (spec.threshold - lo) / (hi - lo)
                    ax.axvline(x=norm_thresh, color="red", linestyle="--", alpha=0.7)

    ax.set_xlabel("Normalised output value")
    ax.set_title("Output Distributions (normalised)")
    if plotted > 0:
        ax.legend(fontsize=7, loc="upper right")

    # ── Bottom-right: Risk matrix (severity vs occurrence) ──────────
    ax = axes[1, 1]
    fmea_modes = report.fmea.ranked()
    if fmea_modes:
        severities = [fm.severity for fm in fmea_modes]
        occurrences = [fm.occurrence for fm in fmea_modes]
        rpns = [fm.rpn for fm in fmea_modes]
        detections = [fm.detection for fm in fmea_modes]

        # Bubble size proportional to detection (harder to detect = bigger)
        sizes = [d * 20 + 20 for d in detections]
        scatter = ax.scatter(
            occurrences, severities, s=sizes, c=rpns,
            cmap="RdYlGn_r", alpha=0.7, edgecolors="black", linewidths=0.5,
        )
        plt.colorbar(scatter, ax=ax, label="RPN")

        # Label critical modes (RPN > 100)
        for fm in fmea_modes:
            if fm.rpn > 100:
                ax.annotate(
                    fm.id, (fm.occurrence, fm.severity),
                    fontsize=6, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points",
                )

    ax.set_xlabel("Occurrence")
    ax.set_ylabel("Severity")
    ax.set_title("FMEA Risk Matrix (bubble = detection)")
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.5, 10.5)
    ax.axhspan(0.5, 5.5, xmin=0, xmax=0.5, alpha=0.05, color="green")
    ax.axhspan(5.5, 10.5, xmin=0.5, xmax=1, alpha=0.05, color="red")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if out_path is None:
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        out_path = str(root / "docs" / "figures" / f"confidence_report_{report.spec_set_name}.png")

    from pathlib import Path
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
