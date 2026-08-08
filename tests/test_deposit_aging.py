"""Fast contracts for V6 §5.2 deposit self-annealing."""

import pytest

from models.deposit_aging import (
    aging_factors,
    correct_to_as_deposited,
    correct_to_standard,
    correction_for,
    diffusible_h_remaining_frac,
    effective_tau_hours,
    metrology_standard,
    model_scope,
    sweep_aging,
    validate_aging_record,
)


def test_factors_at_zero_is_one_and_monotone_to_floor():
    f0 = aging_factors(0.0)
    assert f0.f_stress == pytest.approx(1.0)
    assert f0.f_hv == pytest.approx(1.0)
    prev = 1.0
    for t in (1, 4, 24, 120, 1000):
        f = aging_factors(float(t))
        assert 0.4 < f.f_stress <= prev + 1e-9
        prev = f.f_stress
    longtime = aging_factors(1e5)
    assert longtime.f_stress == pytest.approx(0.50, abs=0.01)
    assert longtime.f_hv > longtime.f_stress  # hv relaxes less
    assert longtime.f_resistivity > longtime.f_hv


def test_tau_shrinks_with_temperature_and_hydrogen():
    cold = effective_tau_hours(temperature_C=10, diffusible_h_ppm=1.0)
    hot = effective_tau_hours(temperature_C=40, diffusible_h_ppm=1.0)
    assert hot < cold
    low_h = effective_tau_hours(temperature_C=20, diffusible_h_ppm=0.2)
    high_h = effective_tau_hours(temperature_C=20, diffusible_h_ppm=5.0)
    assert high_h < low_h


def test_h_egress_fraction_thickness_and_time():
    assert diffusible_h_remaining_frac(0.0) == pytest.approx(1.0)
    thin = diffusible_h_remaining_frac(24.0, foil_thickness_um=10.0, temperature_C=20.0)
    thick = diffusible_h_remaining_frac(24.0, foil_thickness_um=100.0, temperature_C=20.0)
    assert thin < thick
    early = diffusible_h_remaining_frac(1.0, foil_thickness_um=25.0)
    late = diffusible_h_remaining_frac(48.0, foil_thickness_um=25.0)
    assert late < early
    assert 0.0 <= late <= 1.0


def test_correction_round_trips_through_standard():
    measured = 400.0  # MPa stress at 4 h
    fac4 = aging_factors(4.0).f_stress
    as_dep = correct_to_as_deposited(measured, 4.0, fac4)
    assert as_dep == pytest.approx(measured / fac4)
    # 4 h reading mapped to 24 h standard must be lower (more aged)
    std_val = correct_to_standard(measured, 4.0, 24.0, property="stress")
    assert std_val < measured
    # round-trip: map corrected std back to 4 h recovers original within float
    back = correct_to_standard(std_val, 24.0, 4.0, property="stress")
    assert back == pytest.approx(measured, rel=1e-9)


def test_correction_invalid_inputs():
    with pytest.raises(ValueError):
        aging_factors(-1.0)
    with pytest.raises(ValueError):
        correct_to_as_deposited(100.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        correct_to_standard(100.0, 4.0, property="bad")


def test_validate_aging_record_warns_when_timestamp_missing():
    empty = validate_aging_record({})
    assert any(i["path"] == "metadata.aging_hours" for i in empty["issues"])
    assert any("storage_temperature" in i["path"] for i in empty["issues"])
    ok = validate_aging_record({"aging_hours": 24.0, "storage_temperature_C": 20.0})
    assert not any(i["path"] == "metadata.aging_hours" for i in ok["issues"])
    # negative aging
    neg = validate_aging_record({"aging_hours": -2.0, "storage_temperature_C": 20.0})
    assert any("non-negative" in i["message"] for i in neg["issues"])


def test_standard_and_sweep_and_scope():
    std = metrology_standard()
    assert std["standard_aging_hours"] == 24.0
    assert std["allowed_window_hours"] == [18.0, 30.0]
    rows = sweep_aging()
    assert len(rows) == 6
    assert rows[0]["aging_hours"] == 1.0
    assert rows[0]["f_stress"] > rows[-1]["f_stress"]
    scope = model_scope()
    assert scope["screening_flag"] == "unvalidated (L1)"
    assert any("hydrogen_trapping" in s for s in scope["live_derivations"])
    # convenience wrapper respects property name aliases
    c = correction_for(200.0, 4.0, property="hardness")
    assert c.as_deposited_value > c.measured_value > c.standard_value
