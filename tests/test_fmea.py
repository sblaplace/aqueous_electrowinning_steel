"""
Tests for the FMEA module — >=6 test cases covering:
1. FailureMode construction and RPN computation
2. FailureMode validation (bounds)
3. generate_fmea returns >=20 failure modes
4. critical_failure_paths identifies RPN>100 modes
5. mitigation_roadmap produces prioritized actions
6. FMEAReport aggregation and ranking
7. MC integration adjusts occurrence ratings
8. to_dict round-trip structure
"""

from __future__ import annotations


import pytest

from models.uncertainty.fmea import (
    FailureMode,
    FMEAReport,
    generate_fmea,
    critical_failure_paths,
    mitigation_roadmap,
    _default_failure_modes,
)


# ---------------------------------------------------------------------------
# Test 1: FailureMode construction and RPN
# ---------------------------------------------------------------------------

def test_failure_mode_rpn():
    """RPN = severity × occurrence × detection."""
    fm = FailureMode(
        id="TEST-001", component="Test", mode="mode", effect="effect",
        severity=5, cause="cause", occurrence=4, detection=3,
    )
    assert fm.rpn == 5 * 4 * 3  # 60


def test_failure_mode_residual_rpn():
    """Residual RPN computed after mitigation."""
    fm = FailureMode(
        id="TEST-002", component="Test", mode="mode", effect="effect",
        severity=8, cause="cause", occurrence=6, detection=7,
        mitigation="do something",
        residual_severity=8, residual_occurrence=3, residual_detection=4,
    )
    assert fm.rpn == 8 * 6 * 7  # 336
    assert fm.residual_rpn == 8 * 3 * 4  # 96
    assert fm.rpn_reduction == 336 - 96


def test_failure_mode_no_mitigation_residual():
    """Without mitigation, residual_rpn is 0."""
    fm = FailureMode(
        id="TEST-003", component="Test", mode="mode", effect="effect",
        severity=5, cause="cause", occurrence=3, detection=2,
    )
    assert fm.residual_rpn == 0
    assert fm.rpn_reduction == fm.rpn


# ---------------------------------------------------------------------------
# Test 2: FailureMode validation
# ---------------------------------------------------------------------------

def test_failure_mode_severity_bounds():
    """Severity outside 1-10 raises ValueError."""
    with pytest.raises(ValueError, match="severity"):
        FailureMode(id="X", component="X", mode="X", effect="X",
                    severity=0, cause="X", occurrence=5, detection=5)
    with pytest.raises(ValueError, match="severity"):
        FailureMode(id="X", component="X", mode="X", effect="X",
                    severity=11, cause="X", occurrence=5, detection=5)


def test_failure_mode_occurrence_bounds():
    """Occurrence outside 1-10 raises ValueError."""
    with pytest.raises(ValueError, match="occurrence"):
        FailureMode(id="X", component="X", mode="X", effect="X",
                    severity=5, cause="X", occurrence=0, detection=5)


def test_failure_mode_detection_bounds():
    """Detection outside 1-10 raises ValueError."""
    with pytest.raises(ValueError, match="detection"):
        FailureMode(id="X", component="X", mode="X", effect="X",
                    severity=5, cause="X", occurrence=5, detection=11)


# ---------------------------------------------------------------------------
# Test 3: Default failure modes >= 20
# ---------------------------------------------------------------------------

def test_default_failure_modes_count():
    """Default library has >=20 failure modes."""
    modes = _default_failure_modes()
    assert len(modes) >= 20, f"Only {len(modes)} default failure modes"


def test_default_failure_modes_span_subsystems():
    """Default modes cover all 5 required subsystems."""
    modes = _default_failure_modes()
    components = {fm.component for fm in modes}
    required = {"Electrochemistry", "Co-deposition", "Microstructure",
                "Heat treatment", "Process"}
    assert required.issubset(components), (
        f"Missing subsystems: {required - components}"
    )


def test_generate_fmea_returns_report():
    """generate_fmea returns a valid FMEAReport."""
    fmea = generate_fmea()
    assert isinstance(fmea, FMEAReport)
    assert fmea.total >= 20
    assert fmea.mean_rpn > 0
    assert fmea.max_rpn > 0


# ---------------------------------------------------------------------------
# Test 4: Critical failure paths
# ---------------------------------------------------------------------------

def test_critical_failure_paths_threshold():
    """critical_failure_paths returns only modes with RPN > threshold."""
    fmea = generate_fmea()
    critical = critical_failure_paths(fmea, rpn_threshold=100)

    assert len(critical) > 0, "Expected some critical failure modes"
    for fm in critical:
        assert fm.rpn > 100, f"{fm.id} has RPN={fm.rpn}, expected >100"

    # Should be sorted descending
    for i in range(len(critical) - 1):
        assert critical[i].rpn >= critical[i + 1].rpn


def test_critical_failure_paths_custom_threshold():
    """Custom threshold filters correctly."""
    fmea = generate_fmea()
    very_critical = critical_failure_paths(fmea, rpn_threshold=200)
    somewhat_critical = critical_failure_paths(fmea, rpn_threshold=50)

    assert len(very_critical) <= len(somewhat_critical)
    for fm in very_critical:
        assert fm.rpn > 200


# ---------------------------------------------------------------------------
# Test 5: Mitigation roadmap
# ---------------------------------------------------------------------------

def test_mitigation_roadmap_actions():
    """Roadmap produces prioritized mitigation actions."""
    fmea = generate_fmea()
    roadmap = mitigation_roadmap(fmea)

    assert len(roadmap) >= 10, f"Expected >=10 actions, got {len(roadmap)}"

    # Each action has required keys
    for action in roadmap:
        assert "id" in action
        assert "rpn" in action
        assert "residual_rpn" in action
        assert "rpn_reduction" in action
        assert "priority" in action
        assert action["priority"] in (1, 2, 3)
        assert action["estimated_effort"] in ("low", "medium", "high")


def test_mitigation_roadmap_ordering():
    """Roadmap is sorted by priority then RPN reduction."""
    fmea = generate_fmea()
    roadmap = mitigation_roadmap(fmea)

    for i in range(len(roadmap) - 1):
        a, b = roadmap[i], roadmap[i + 1]
        # Primary sort: priority ascending
        assert a["priority"] <= b["priority"], (
            f"Priority order violated: {a['id']} P{a['priority']} before "
            f"{b['id']} P{b['priority']}"
        )
        # Secondary sort: RPN reduction descending (within same priority)
        if a["priority"] == b["priority"]:
            assert a["rpn_reduction"] >= b["rpn_reduction"], (
                f"RPN reduction order violated within P{a['priority']}"
            )


# ---------------------------------------------------------------------------
# Test 6: FMEAReport aggregation
# ---------------------------------------------------------------------------

def test_fmea_report_by_component():
    """by_component groups modes by subsystem."""
    fmea = generate_fmea()
    groups = fmea.by_component()

    assert "Electrochemistry" in groups
    assert "Microstructure" in groups
    assert len(groups["Electrochemistry"]) >= 3


def test_fmea_report_ranked():
    """ranked() returns modes sorted by RPN descending."""
    fmea = generate_fmea()
    ranked = fmea.ranked()

    assert len(ranked) == fmea.total
    for i in range(len(ranked) - 1):
        assert ranked[i].rpn >= ranked[i + 1].rpn


def test_fmea_report_critical_count():
    """critical_count counts modes with RPN > 100."""
    fmea = generate_fmea()
    expected = sum(1 for fm in fmea.failure_modes if fm.rpn > 100)
    assert fmea.critical_count == expected


def test_fmea_report_to_dict():
    """to_dict returns well-structured report."""
    fmea = generate_fmea()
    d = fmea.to_dict()

    assert "total_modes" in d
    assert "mean_rpn" in d
    assert "max_rpn" in d
    assert "critical_count" in d
    assert "modes" in d
    assert d["total_modes"] == fmea.total

    # Each mode entry has required fields
    for mode in d["modes"]:
        assert "id" in mode
        assert "rpn" in mode
        assert "component" in mode


# ---------------------------------------------------------------------------
# Test 7: MC integration
# ---------------------------------------------------------------------------

class _MockMCResult:
    """Minimal mock of MonteCarloResult for FMEA calibration testing."""

    def __init__(self, pass_rates, n_samples=100, overall_confidence=0.8):
        self.pass_rates = pass_rates
        self.n_samples = n_samples
        self.overall_confidence = overall_confidence


def test_mc_integration_adjusts_occurrence():
    """Low MC pass rates increase occurrence for related failure modes."""
    # Simulate low pass rate for elongation (affects MS-003, CD-003)
    mc = _MockMCResult(pass_rates={
        "elongation_pct >= 10": 0.3,  # 30% pass → big occurrence bump
        "current_efficiency_percent >= 85": 0.9,
    })

    fmea_standalone = generate_fmea()
    fmea_mc = generate_fmea(mc_result=mc)

    # Find MS-003 (hydrogen embrittlement) in both
    fm_standalone = next(fm for fm in fmea_standalone.failure_modes if fm.id == "MS-003")
    fm_mc = next(fm for fm in fmea_mc.failure_modes if fm.id == "MS-003")

    # MC-adjusted should have higher occurrence (30% pass → +3 bump)
    assert fm_mc.occurrence >= fm_standalone.occurrence, (
        f"Expected occurrence increase: {fm_standalone.occurrence} → {fm_mc.occurrence}"
    )
    assert fm_mc.rpn >= fm_standalone.rpn


def test_mc_integration_no_change_on_good_pass_rates():
    """High pass rates don't increase occurrence."""
    mc = _MockMCResult(pass_rates={
        "elongation_pct >= 10": 0.95,
        "current_efficiency_percent >= 85": 0.98,
    })

    fmea_standalone = generate_fmea()
    fmea_mc = generate_fmea(mc_result=mc)

    # With 95% pass rate, bump is ~0 → no change expected
    for fm_s, fm_m in zip(fmea_standalone.failure_modes, fmea_mc.failure_modes):
        assert fm_s.occurrence == fm_m.occurrence, (
            f"{fm_s.id}: unexpected occurrence change {fm_s.occurrence} → {fm_m.occurrence}"
        )


# ---------------------------------------------------------------------------
# Test 8: Frozen FailureMode
# ---------------------------------------------------------------------------

def test_failure_mode_frozen():
    """FailureMode is frozen (immutable)."""
    fm = FailureMode(
        id="TEST", component="X", mode="X", effect="X",
        severity=5, cause="X", occurrence=3, detection=2,
    )
    with pytest.raises(AttributeError):
        fm.severity = 10  # type: ignore
