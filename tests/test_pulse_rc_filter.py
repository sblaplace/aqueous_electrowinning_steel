"""
Unit tests for pulse RC double-layer filtering and frequency cutoff model.
"""

from models.pulse_rc_filter import (
    PulseCircuitParams,
    simulate_pulse_rc_response,
    max_practical_frequency_Hz,
)


def test_pulse_circuit_properties():
    """Verify equivalent circuit time constants and cutoff frequency."""
    circuit = PulseCircuitParams(c_dl_uF_cm2=30.0, r_ohm_ohm_cm2=1.0)
    # tau = R_ohm * C_dl = 1.0e-4 * 0.3 = 3.0e-5 s (0.03 ms)
    assert 1e-6 <= circuit.tau_charging_s <= 1e-3
    assert circuit.cutoff_frequency_Hz > 100.0


def test_low_frequency_high_fidelity():
    """At low pulse frequency (10 Hz), Faradaic current reaches near 100% of peak."""
    res = simulate_pulse_rc_response(
        frequency_Hz=10.0,
        duty_cycle=0.20,
        peak_current_mA_cm2=200.0,
    )
    assert res.peak_attenuation_ratio >= 0.95
    assert res.actual_trough_faradaic_mA_cm2 <= 0.05 * 200.0
    assert "high fidelity" in res.waveform_fidelity
    assert res.is_frequency_feasible


def test_high_frequency_rc_attenuation():
    """At high pulse frequency (10 kHz), double-layer filtering collapses the peak."""
    res = simulate_pulse_rc_response(
        frequency_Hz=10000.0,
        duty_cycle=0.20,
        peak_current_mA_cm2=200.0,
    )
    assert res.peak_attenuation_ratio < 0.80
    assert res.actual_trough_faradaic_mA_cm2 > 5.0  # Fails to reach zero during off-time
    assert not res.is_frequency_feasible


def test_max_practical_frequency_calculation():
    """Verify maximum allowable frequency decreases with higher double-layer capacitance."""
    c_low = PulseCircuitParams(c_dl_uF_cm2=20.0, r_ohm_ohm_cm2=1.0)
    c_high = PulseCircuitParams(c_dl_uF_cm2=80.0, r_ohm_ohm_cm2=1.0)

    f_max_low = max_practical_frequency_Hz(duty_cycle=0.20, circuit=c_low)
    f_max_high = max_practical_frequency_Hz(duty_cycle=0.20, circuit=c_high)

    assert f_max_low > f_max_high
    assert f_max_low > 100.0
