import numpy as np
import pytest
from models.pulse import (
    PulseDepositionModel,
    PulseResult,
    PulseWaveform,
    compare_dc_vs_pulse,
)


def test_waveform_properties_and_evaluation():
    wf = PulseWaveform(
        j_cathodic_mA_cm2=100.0,
        t_cathodic_s=0.05,
        j_anodic_mA_cm2=-20.0,
        t_anodic_s=0.01,
        t_off_s=0.04,
    )
    assert wf.t_cycle_s == pytest.approx(0.10)
    assert wf.frequency_Hz == pytest.approx(10.0)
    assert wf.duty_cycle == pytest.approx(0.50)
    assert wf.j_avg_mA_cm2 == pytest.approx(48.0)  # (100*0.05 - 20*0.01) / 0.10

    # Current evaluation at different points in the cycle
    assert wf.evaluate_current_A_m2(0.02) == pytest.approx(1000.0)  # 100 mA/cm2 = 1000 A/m2
    assert wf.evaluate_current_A_m2(0.055) == pytest.approx(-200.0)  # -20 mA/cm2 = -200 A/m2
    assert wf.evaluate_current_A_m2(0.08) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"j_cathodic_mA_cm2": -10.0, "t_cathodic_s": 0.05},
        {"j_cathodic_mA_cm2": 100.0, "t_cathodic_s": 0.0},
        {"j_cathodic_mA_cm2": 100.0, "t_cathodic_s": 0.05, "j_anodic_mA_cm2": 10.0},
        {"j_cathodic_mA_cm2": 100.0, "t_cathodic_s": 0.05, "t_anodic_s": -0.01},
        {"j_cathodic_mA_cm2": 100.0, "t_cathodic_s": 0.05, "t_off_s": -0.01},
    ],
)
def test_waveform_invalid_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        PulseWaveform(**kwargs)


def test_simulation_runs_and_returns_result():
    wf = PulseWaveform(
        j_cathodic_mA_cm2=50.0,
        t_cathodic_s=0.01,
        t_off_s=0.01,
    )
    model = PulseDepositionModel(fe_bulk_M=1.0, bulk_pH=2.0)
    res = model.simulate(wf, n_cycles=5, steps_per_cycle=20)

    assert isinstance(res, PulseResult)
    assert len(res.time_s) == 101  # 5 * 20 + 1
    assert len(res.surface_fe_M) == 101
    assert len(res.surface_pH) == 101
    assert res.cycle_avg_efficiency > 0.0
    assert res.net_fe_deposited_g_m2 > 0.0
    assert res.plating_rate_um_hr > 0.0
    assert 0.0 <= res.peak_surface_depletion_ratio <= 1.0


def test_pulse_off_time_allows_surface_fe_recovery():
    """Pulse off time must allow surface Fe2+ concentration to recover toward bulk."""
    wf = PulseWaveform(
        j_cathodic_mA_cm2=150.0,
        t_cathodic_s=0.05,
        t_off_s=0.15,
    )
    model = PulseDepositionModel(fe_bulk_M=1.0, bulk_pH=2.0)
    res = model.simulate(wf, n_cycles=3, steps_per_cycle=100)

    # Find surface Fe2+ at end of pulse ON vs end of pulse OFF in first cycle
    t_cycle = wf.t_cycle_s
    time_arr = res.time_s
    idx_end_on = np.argmin(np.abs(time_arr - wf.t_cathodic_s))
    idx_end_off = np.argmin(np.abs(time_arr - t_cycle))

    fe_end_on = res.surface_fe_M[idx_end_on]
    fe_end_off = res.surface_fe_M[idx_end_off]

    # Surface Fe2+ should recover during off period
    assert fe_end_off > fe_end_on
    assert fe_end_off == pytest.approx(1.0, rel=0.1)


def test_pulse_reverse_reduces_peak_surface_pH_rise():
    """Pulse-reverse electrodeposition should suppress local pH spike vs continuous DC."""
    wf_pre = PulseWaveform(
        j_cathodic_mA_cm2=100.0,
        t_cathodic_s=0.02,
        j_anodic_mA_cm2=-20.0,
        t_anodic_s=0.005,
        t_off_s=0.015,
    )
    wf_dc = PulseWaveform(
        j_cathodic_mA_cm2=100.0,
        t_cathodic_s=0.04 * 10,
    )
    model = PulseDepositionModel(fe_bulk_M=1.0, bulk_pH=2.0)

    res_pre = model.simulate(wf_pre, n_cycles=10, steps_per_cycle=40)
    res_dc = model.simulate(wf_dc, n_cycles=1, steps_per_cycle=400)

    # PRE max surface pH rise should be lower or equal to continuous DC at high peak current
    assert res_pre.max_surface_pH <= res_dc.max_surface_pH + 0.1
    # Surface Fe2+ depletion should be less severe in PRE
    assert res_pre.peak_surface_depletion_ratio > res_dc.peak_surface_depletion_ratio


def test_compare_dc_vs_pulse_dictionary_keys():
    comparison = compare_dc_vs_pulse(j_peak_mA_cm2=80.0, n_cycles=5)
    for key in ("dc_peak", "dc_avg", "pulsed", "pulse_reverse"):
        assert key in comparison
        assert isinstance(comparison[key], PulseResult)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"boundary_layer_m": 0.0},
        {"fe_bulk_M": -1.0},
        {"bulk_pH": 15.0},
        {"grid_points": 2},
    ],
)
def test_model_invalid_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        PulseDepositionModel(**kwargs)
