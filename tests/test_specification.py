"""Tests for the specification framework."""

import json
import tempfile
from pathlib import Path

import pytest
import numpy as np

from models.uncertainty.specification import (
    Specification,
    check_specifications,
    check_mc_specifications,
    load_specs_from_yaml,
    load_specs_from_json,
    _extract_value,
    _check_one,
    SPECS_A36,
    SPECS_CARBURIZED,
    SPECS_ELECTROWINNING,
    ALL_STANDARD_SPECS,
)


# ── Spec set definitions ──────────────────────────────────────────────────


def test_five_spec_sets_defined():
    """>=5 spec sets are defined."""
    assert len(ALL_STANDARD_SPECS) >= 5
    expected_names = {"ASTM_A36", "AISI_1010", "AISI_1020", "CARBURIZED", "ELECTROWINNING"}
    assert expected_names.issubset(set(ALL_STANDARD_SPECS.keys()))


def test_a36_specs_content():
    """SPECS_A36 has YS>=250, UTS>=400, elong>=20, HV 120-180."""
    ys = next(s for s in SPECS_A36 if "yield" in s.name.lower())
    uts = next(s for s in SPECS_A36 if "ultimate" in s.name.lower())
    elong = next(s for s in SPECS_A36 if "elongation" in s.name.lower())
    hv = next(s for s in SPECS_A36 if "hardness" in s.name.lower())

    assert ys.operator == ">=" and ys.threshold == 250.0
    assert uts.operator == ">=" and uts.threshold == 400.0
    assert elong.operator == ">=" and elong.threshold == 20.0
    assert hv.operator == "range"
    assert hv.threshold == 120.0
    assert hv.threshold_upper == 180.0


def test_electrowinning_specs_content():
    """SPECS_ELECTROWINNING has FE>=70%, energy<=8, Ni<=2."""
    fe = next(s for s in SPECS_ELECTROWINNING if "faradaic" in s.name.lower() or "efficiency" in s.name.lower())
    energy = next(s for s in SPECS_ELECTROWINNING if "energy" in s.name.lower())
    ni = next(s for s in SPECS_ELECTROWINNING if "nickel" in s.name.lower() or "ni" in s.name.lower())

    assert fe.operator == ">=" and fe.threshold == 70.0
    assert energy.operator == "<=" and energy.threshold == 8.0
    assert ni.operator == "<=" and ni.threshold == 2.0


# ── Value extraction ──────────────────────────────────────────────────────


def test_extract_value_flat_dict():
    result = {"sigma_y_MPa": 320.5, "uts_MPa": 450.0}
    assert _extract_value(result, "sigma_y_MPa") == 320.5


def test_extract_value_nested_dict():
    result = {
        "alloy_kinetics": {"ni_wt_percent": 1.5},
        "carbon_incorporation": {"predicted_carbon_wt_percent": 0.8},
    }
    assert _extract_value(result, "alloy_kinetics.ni_wt_percent") == 1.5
    assert _extract_value(result, "carbon_incorporation.predicted_carbon_wt_percent") == 0.8


def test_extract_value_numpy_array():
    result = {"surface_hv": np.array([500, 600, 700])}
    assert _extract_value(result, "surface_hv") == 700.0


def test_extract_value_missing_key():
    result = {"a": 1}
    assert _extract_value(result, "b") is None
    assert _extract_value(result, "a.b.c") is None


def test_extract_value_dataclass():
    """Extraction from a dataclass-like object with attributes."""
    class FakeResult:
        sigma_y_MPa = 350.0
        uts_MPa = 480.0
    r = FakeResult()
    assert _extract_value(r, "sigma_y_MPa") == 350.0
    assert _extract_value(r, "missing_key") is None


# ── Single-check logic ────────────────────────────────────────────────────


def test_check_one_ge():
    passed, margin = _check_one(300.0, ">=", 250.0, None, 0.0)
    assert passed is True
    assert margin == pytest.approx(50.0)

    passed, margin = _check_one(200.0, ">=", 250.0, None, 0.0)
    assert passed is False
    assert margin < 0


def test_check_one_le():
    passed, _ = _check_one(5.0, "<=", 8.0, None, 0.0)
    assert passed is True

    passed, _ = _check_one(10.0, "<=", 8.0, None, 0.0)
    assert passed is False


def test_check_one_range():
    passed, _ = _check_one(150.0, "range", 120.0, 180.0, 0.0)
    assert passed is True

    passed, _ = _check_one(100.0, "range", 120.0, 180.0, 0.0)
    assert passed is False

    passed, _ = _check_one(200.0, "range", 120.0, 180.0, 0.0)
    assert passed is False


def test_check_one_tolerance():
    """5% tolerance on >=250 should pass 240 (250*0.95=237.5)."""
    passed, _ = _check_one(240.0, ">=", 250.0, None, 5.0)
    assert passed is True

    # Strict: 240 < 250
    passed, _ = _check_one(240.0, ">=", 250.0, None, 0.0)
    assert passed is False


def test_check_one_none_value():
    passed, margin = _check_one(None, ">=", 250.0, None, 0.0)
    assert passed is False
    assert margin is None


# ── Full specification checking ───────────────────────────────────────────


def test_check_specifications_pass():
    """A result that meets A36 specs should pass all."""
    result = {
        "sigma_y_MPa": 280.0,
        "uts_MPa": 420.0,
        "elongation_pct": 25.0,
        "vickers_hv": 150.0,
    }
    report = check_specifications(result, SPECS_A36, spec_set_name="ASTM_A36")
    assert report.all_passed
    assert report.pass_rate == 1.0
    assert report.failed == 0
    assert report.passed == len(SPECS_A36)
    assert len(report.diagnose()) == 0


def test_check_specifications_fail():
    """A weak result should fail some A36 specs."""
    result = {
        "sigma_y_MPa": 200.0,     # FAIL (< 250)
        "uts_MPa": 420.0,         # PASS
        "elongation_pct": 15.0,   # FAIL (< 20)
        "vickers_hv": 150.0,      # PASS (in range)
    }
    report = check_specifications(result, SPECS_A36, spec_set_name="ASTM_A36")
    assert not report.all_passed
    assert report.failed >= 2
    assert report.pass_rate < 1.0
    diagnoses = report.diagnose()
    assert len(diagnoses) >= 2
    # Check that safety failures are reported
    assert any("safety" in d.lower() for d in diagnoses)


def test_check_specifications_with_nested_dict():
    """Check electrowinning specs against nested co-deposition result."""
    result = {
        "alloy_kinetics": {
            "ni_wt_percent": 1.5,
            "current_efficiency_percent": 85.0,
        },
        "carbon_incorporation": {
            "predicted_carbon_wt_percent": 0.8,
        },
        "integrated_metrics": {
            "adjusted_overall_current_efficiency_percent": 75.0,
        },
        "specific_energy_kWh_per_kg": 6.5,
    }
    report = check_specifications(result, SPECS_ELECTROWINNING)
    # FE>=70 (75 pass), energy<=8 (6.5 pass), Ni<=2 (1.5 pass), C range (0.8 pass)
    assert report.all_passed


def test_spec_report_to_dict():
    """SpecReport.to_dict() produces valid JSON-serializable output."""
    result = {"sigma_y_MPa": 200.0, "uts_MPa": 300.0, "elongation_pct": 10.0, "vickers_hv": 90.0}
    report = check_specifications(result, SPECS_A36)
    d = report.to_dict()
    assert isinstance(d, dict)
    assert "pass_rate" in d
    assert "failures" in d
    assert "diagnoses" in d
    # Should be JSON-serializable
    json.dumps(d)


# ── YAML / JSON loading ──────────────────────────────────────────────────


def test_load_specs_from_json():
    """Custom spec loading from JSON works."""
    specs_data = {
        "specifications": [
            {
                "name": "Custom yield strength",
                "output_key": "sigma_y_MPa",
                "operator": ">=",
                "threshold": 300.0,
                "unit": "MPa",
                "source": "Custom",
                "criticality": "safety",
                "tolerance_pct": 5.0,
                "description": "Custom minimum yield",
            },
            {
                "name": "Custom hardness range",
                "output_key": "vickers_hv",
                "operator": "range",
                "threshold": 100.0,
                "threshold_upper": 200.0,
                "unit": "HV",
                "tolerance_pct": 0.0,
            },
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(specs_data, f)
        f.flush()
        path = Path(f.name)

    try:
        loaded = load_specs_from_json(path)
        assert len(loaded) == 2
        assert loaded[0].name == "Custom yield strength"
        assert loaded[0].threshold == 300.0
        assert loaded[1].operator == "range"
        assert loaded[1].threshold_upper == 200.0

        # Verify they work with check_specifications
        result = {"sigma_y_MPa": 350.0, "vickers_hv": 150.0}
        report = check_specifications(result, loaded)
        assert report.all_passed
    finally:
        path.unlink()


def test_load_specs_from_yaml():
    """Custom spec loading from YAML works."""
    yaml_content = """
specifications:
  - name: "Min hardness"
    output_key: vickers_hv
    operator: ">="
    threshold: 200
    unit: HV
    source: "Test"
    criticality: performance
    tolerance_pct: 10
  - name: "Max energy"
    output_key: energy_kWh_kg
    operator: "<="
    threshold: 10.0
    unit: kWh/kg
    tolerance_pct: 0
"""
    try:
        pass
    except pytest.skip.Exception:
        pytest.skip("PyYAML not installed")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        path = Path(f.name)

    try:
        loaded = load_specs_from_yaml(path)
        assert len(loaded) == 2
        assert loaded[0].name == "Min hardness"
        assert loaded[1].operator == "<="

        result = {"vickers_hv": 250.0, "energy_kWh_kg": 7.0}
        report = check_specifications(result, loaded)
        assert report.all_passed
    finally:
        path.unlink()


# ── Monte Carlo integration ──────────────────────────────────────────────


def test_check_mc_specifications():
    """Monte Carlo pass-rate statistics work correctly."""
    # 3 samples: 2 pass all A36, 1 fails YS
    mc_results = [
        {"sigma_y_MPa": 280.0, "uts_MPa": 420.0, "elongation_pct": 25.0, "vickers_hv": 150.0},
        {"sigma_y_MPa": 260.0, "uts_MPa": 410.0, "elongation_pct": 22.0, "vickers_hv": 140.0},
        {"sigma_y_MPa": 230.0, "uts_MPa": 380.0, "elongation_pct": 18.0, "vickers_hv": 110.0},
    ]
    result = check_mc_specifications(mc_results, SPECS_A36, spec_set_name="A36_MC_test")

    assert len(result["individual_reports"]) == 3
    assert result["overall_pass_rate"] == pytest.approx(2.0 / 3.0)
    assert "ASTM A36 yield strength" in result["pass_rates"]
    assert result["pass_rates"]["ASTM A36 yield strength"] == pytest.approx(2.0 / 3.0)
    assert result["failure_histogram"]["ASTM A36 yield strength"] == 1
    # Worst spec should be the one that fails most
    assert result["worst_spec"] is not None


def test_check_mc_empty():
    """Empty MC results returns sensible defaults."""
    result = check_mc_specifications([], SPECS_A36)
    assert result["overall_pass_rate"] == 0.0
    assert result["worst_spec"] is None


# ── Tolerance override ───────────────────────────────────────────────────


def test_tolerance_override():
    """Global tolerance override works."""
    # 230 MPa fails strict A36 (>=250 with 5% tol -> 237.5) but passes with 10% tolerance
    result = {"sigma_y_MPa": 230.0, "uts_MPa": 420.0, "elongation_pct": 25.0, "vickers_hv": 150.0}
    strict = check_specifications(result, SPECS_A36)
    assert not strict.all_passed  # fails YS at 5% tolerance

    relaxed = check_specifications(result, SPECS_A36, tolerance_override_pct=10.0)
    # 250 * 0.9 = 225, so 230 passes
    ys_result = next(r for r in relaxed.results if "yield" in r.spec.name.lower())
    assert ys_result.passed


# ── Carburized spec against result summary ────────────────────────────────


def test_carburized_specs_against_summary():
    """SPECS_CARBURIZED works with carburization summary dict."""
    # Simulate CarburizationResult.summary() output
    result = {
        "final_case_depth_035_um": 600.0,
        "final_surface_hv": 750.0,
        "final_core_c_wt": 0.15,
        "surface_C_wt_percent": 0.95,
    }
    report = check_specifications(result, SPECS_CARBURIZED, spec_set_name="CARBURIZED")
    assert report.all_passed
    assert report.pass_rate == 1.0


# ── Integration with MechanicalPropertiesModel ───────────────────────────


def test_integration_with_mechanical_model():
    """Full integration: predict -> check -> report."""
    from models.mechanical_properties import MechanicalPropertiesModel

    model = MechanicalPropertiesModel()
    result = model.predict(
        j_avg_mA_cm2=150,
        j_peak_mA_cm2=300,
        duty_cycle=0.5,
        waveform="pre",
        ni_wt_percent=2.0,
        carbon_wt_percent=0.8,
        current_efficiency_percent=93.0,
    )
    # Check against A36 (should pass at these conditions)
    report = check_specifications(result, SPECS_A36, spec_set_name="A36_via_model")
    assert report.total == len(SPECS_A36)
    # The result should be dict-serializable via summary()
    summary = result.summary()
    report2 = check_specifications(summary, SPECS_A36)
    assert report2.total == len(SPECS_A36)


# ── Specification constructor validation ──────────────────────────────────


def test_specification_validation():
    """Specification constructor rejects bad inputs."""
    with pytest.raises(ValueError, match="Unknown operator"):
        Specification(name="bad", output_key="x", operator="!=", threshold=0, unit="")

    with pytest.raises(ValueError, match="range.*threshold_upper"):
        Specification(name="bad", output_key="x", operator="range", threshold=0, unit="")

    with pytest.raises(ValueError, match="tolerance"):
        Specification(name="bad", output_key="x", operator=">=", threshold=0, unit="",
                       tolerance_pct=-5.0)
