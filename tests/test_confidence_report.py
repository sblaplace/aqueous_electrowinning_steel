"""
Tests for the confidence report module — >=8 test cases covering:
1. generate_confidence_report full pipeline end-to-end
2. ConfidenceReport structure and fields populated
3. Verdict PASS / CONDITIONAL PASS / FAIL logic
4. Works for multiple spec sets (A36, 1010, 1020, CARBURIZED)
5. Top uncertainty drivers computed
6. FMEA integration (critical failures populated)
7. Validation plan integration (experiments recommended)
8. Design margins computed
9. 4-quadrant figure generation
10. summary_text format
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pytest

from models.uncertainty.confidence_report import (
    ConfidenceReport,
    generate_confidence_report,
    qualification_verdict,
    plot_confidence_report,
    _resolve_spec_set,
    _top_uncertainty_drivers,
    _compute_design_margins,
    SPEC_SET_MAP,
)
from models.uncertainty.monte_carlo import MonteCarloResult
from models.uncertainty.specification import (
    SPECS_A36,
    SPECS_1010,
    SPECS_1020,
    SPECS_CARBURIZED,
    Specification,
)
from models.uncertainty.fmea import FMEAReport, FailureMode, generate_fmea
from models.uncertainty.validation_planner import (
    ValidationPlan,
    plan_validation_experiments,
)
from models.uncertainty.parameter_registry import REGISTRY


# ---------------------------------------------------------------------------
# Test 1: Full pipeline end-to-end (small N for speed)
# ---------------------------------------------------------------------------

def test_full_pipeline_e2e():
    """generate_confidence_report runs the full pipeline and returns a
    populated ConfidenceReport with all fields filled."""
    report = generate_confidence_report(
        spec_set="A36", mc_samples=100, target=0.95, seed=42, n_jobs=1,
    )
    assert isinstance(report, ConfidenceReport)
    assert report.overall_confidence >= 0.0
    assert report.overall_confidence <= 1.0
    assert report.spec_set_name == "ASTM_A36"
    assert len(report.specs) == len(SPECS_A36)
    assert report.verdict in ("PASS", "CONDITIONAL PASS", "FAIL")
    assert isinstance(report.mc_result, MonteCarloResult)
    assert report.mc_result.n_samples == 100


# ---------------------------------------------------------------------------
# Test 2: ConfidenceReport fields populated
# ---------------------------------------------------------------------------

def test_report_fields_populated():
    """All required fields of ConfidenceReport are populated after generation."""
    report = generate_confidence_report(
        spec_set="A36", mc_samples=100, seed=42, n_jobs=1,
    )
    assert report.design_point is not None
    assert isinstance(report.spec_pass_rates, dict)
    assert len(report.spec_pass_rates) > 0
    assert isinstance(report.top_uncertainty_drivers, list)
    assert isinstance(report.critical_failures, list)
    assert isinstance(report.recommended_experiments, list)
    assert isinstance(report.design_margins, dict)
    assert isinstance(report.sensitivity, dict)
    assert isinstance(report.fmea, FMEAReport)
    assert isinstance(report.validation_plan, ValidationPlan)


# ---------------------------------------------------------------------------
# Test 3: Verdict logic — PASS for high confidence
# ---------------------------------------------------------------------------

def test_verdict_pass():
    """Verdict is PASS when confidence >= target and no safety failures."""
    # Create a synthetic report with high confidence and no critical failures
    from models.uncertainty.monte_carlo import DEFAULT_DESIGN_POINT

    mc = MonteCarloResult(
        n_samples=100,
        design_point=DEFAULT_DESIGN_POINT,
        output_distributions={},
        pass_rates={"spec1": 0.99},
        overall_confidence=0.98,
        sensitivity={},
        failure_ranking={},
        parameter_correlations={},
    )
    fmea = FMEAReport(failure_modes=[])
    plan = ValidationPlan(
        experiments=[], expected_variance_reduction=0.0,
        total_cost_usd=0.0, total_duration_hours=0.0, gain_per_dollar=[],
    )
    report = ConfidenceReport(
        design_point=DEFAULT_DESIGN_POINT,
        spec_set_name="ASTM_A36",
        specs=SPECS_A36,
        mc_result=mc,
        sensitivity={},
        fmea=fmea,
        validation_plan=plan,
        overall_confidence=0.98,
        spec_pass_rates={"spec1": 0.99},
        top_uncertainty_drivers=[],
        critical_failures=[],
        recommended_experiments=[],
        design_margins={},
    )
    verdict = qualification_verdict(report, target=0.95)
    assert verdict == "PASS"


# ---------------------------------------------------------------------------
# Test 4: Verdict logic — FAIL for low confidence
# ---------------------------------------------------------------------------

def test_verdict_fail():
    """Verdict is FAIL when confidence is well below target."""
    from models.uncertainty.monte_carlo import DEFAULT_DESIGN_POINT

    mc = MonteCarloResult(
        n_samples=100,
        design_point=DEFAULT_DESIGN_POINT,
        output_distributions={},
        pass_rates={"spec1": 0.50},
        overall_confidence=0.40,
        sensitivity={},
        failure_ranking={},
        parameter_correlations={},
    )
    critical_fm = FailureMode(
        id="MS-003", component="Microstructure",
        mode="Hydrogen embrittlement",
        effect="Delayed cracking",
        severity=10, cause="HER",
        occurrence=7, detection=7,
    )
    fmea = FMEAReport(failure_modes=[critical_fm])
    plan = ValidationPlan(
        experiments=[], expected_variance_reduction=0.0,
        total_cost_usd=0.0, total_duration_hours=0.0, gain_per_dollar=[],
    )
    report = ConfidenceReport(
        design_point=DEFAULT_DESIGN_POINT,
        spec_set_name="ASTM_A36",
        specs=SPECS_A36,
        mc_result=mc,
        sensitivity={},
        fmea=fmea,
        validation_plan=plan,
        overall_confidence=0.40,
        spec_pass_rates={"spec1": 0.50},
        top_uncertainty_drivers=[],
        critical_failures=[critical_fm],
        recommended_experiments=[],
        design_margins={},
    )
    verdict = qualification_verdict(report, target=0.95)
    assert verdict == "FAIL"


# ---------------------------------------------------------------------------
# Test 5: Multiple spec sets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec_set", ["A36", "1010", "1020", "CARBURIZED"])
def test_spec_sets(spec_set: str):
    """generate_confidence_report works for A36, 1010, 1020, CARBURIZED."""
    report = generate_confidence_report(
        spec_set=spec_set, mc_samples=80, seed=42, n_jobs=1,
    )
    assert isinstance(report, ConfidenceReport)
    assert report.spec_set_name.upper() in (
        "ASTM_A36", "AISI_1010", "AISI_1020", "CARBURIZED",
    )
    assert len(report.spec_pass_rates) > 0


# ---------------------------------------------------------------------------
# Test 6: Top uncertainty drivers
# ---------------------------------------------------------------------------

def test_top_uncertainty_drivers():
    """Top uncertainty drivers are identified from MC sensitivity."""
    report = generate_confidence_report(
        spec_set="A36", mc_samples=100, seed=42, n_jobs=1,
    )
    assert len(report.top_uncertainty_drivers) > 0
    # Each driver is (name, importance)
    for name, imp in report.top_uncertainty_drivers:
        assert isinstance(name, str)
        assert imp >= 0.0
    # Should be sorted descending
    imps = [imp for _, imp in report.top_uncertainty_drivers]
    assert imps == sorted(imps, reverse=True)


# ---------------------------------------------------------------------------
# Test 7: FMEA and validation plan integration
# ---------------------------------------------------------------------------

def test_fmea_and_validation_integration():
    """Confidence report integrates FMEA critical failures and validation plan."""
    report = generate_confidence_report(
        spec_set="A36", mc_samples=100, seed=42, n_jobs=1,
    )
    # FMEA should have failure modes
    assert report.fmea.total > 0
    # Critical failures should be identified (RPN > 100)
    assert len(report.critical_failures) > 0
    for fm in report.critical_failures:
        assert fm.rpn > 100
    # Validation plan should recommend experiments
    assert len(report.recommended_experiments) > 0
    for exp in report.recommended_experiments:
        assert exp.cost_usd > 0


# ---------------------------------------------------------------------------
# Test 8: Design margins computed
# ---------------------------------------------------------------------------

def test_design_margins():
    """Design margins are computed for each spec."""
    report = generate_confidence_report(
        spec_set="A36", mc_samples=100, seed=42, n_jobs=1,
    )
    assert len(report.design_margins) > 0
    for name, margin in report.design_margins.items():
        assert isinstance(margin, float)
        # Margin should be finite
        assert not math.isnan(margin)
        assert not math.isinf(margin)


# ---------------------------------------------------------------------------
# Test 9: Summary figure generation
# ---------------------------------------------------------------------------

def test_plot_confidence_report(tmp_path: Path):
    """plot_confidence_report creates a PNG file with 4 quadrants."""
    report = generate_confidence_report(
        spec_set="A36", mc_samples=80, seed=42, n_jobs=1,
    )
    out_path = str(tmp_path / "test_confidence.png")
    plot_confidence_report(report, out_path=out_path)
    assert Path(out_path).exists()
    assert Path(out_path).stat().st_size > 1000  # non-trivial PNG


# ---------------------------------------------------------------------------
# Test 10: summary_text format
# ---------------------------------------------------------------------------

def test_summary_text():
    """summary_text contains key elements: spec set, confidence, verdict."""
    report = generate_confidence_report(
        spec_set="A36", mc_samples=80, seed=42, n_jobs=1,
    )
    text = report.summary_text()
    assert "ASTM_A36" in text
    assert "confidence" in text.lower() or "%" in text
    assert report.verdict in text


# ---------------------------------------------------------------------------
# Test 11: to_dict structure
# ---------------------------------------------------------------------------

def test_to_dict():
    """to_dict returns a well-structured dictionary."""
    report = generate_confidence_report(
        spec_set="A36", mc_samples=80, seed=42, n_jobs=1,
    )
    d = report.to_dict()
    assert "design_point" in d
    assert "overall_confidence" in d
    assert "verdict" in d
    assert "spec_pass_rates" in d
    assert "top_uncertainty_drivers" in d
    assert "critical_failures" in d
    assert "recommended_experiments" in d
    assert "mc_summary" in d
    assert "fmea_summary" in d
    assert "validation_plan_summary" in d


# ---------------------------------------------------------------------------
# Test 12: Spec set resolution
# ---------------------------------------------------------------------------

def test_resolve_spec_set():
    """_resolve_spec_set handles various name formats."""
    specs, name = _resolve_spec_set("A36")
    assert name == "ASTM_A36"
    assert len(specs) == len(SPECS_A36)

    specs, name = _resolve_spec_set("a36")
    assert name == "ASTM_A36"

    specs, name = _resolve_spec_set("CARBURIZED")
    assert name == "CARBURIZED"
    assert len(specs) == len(SPECS_CARBURIZED)

    with pytest.raises(KeyError):
        _resolve_spec_set("NONEXISTENT")
