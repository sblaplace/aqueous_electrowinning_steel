import numpy as np
import pandas as pd
import pytest

from models.eis import (
    load_spectrum,
    summarize_spectrum,
    cpe_impedance,
    warburg_impedance,
    randles_impedance,
    randles_cpe_impedance,
    fit_randles_spectrum,
    exchange_current_from_rct,
    synthetic_randles_spectrum,
    GAS_CONSTANT,
    FARADAY_CONSTANT,
)


def test_randles_limits():
    # Classic Randles: high-frequency intercept is Rs, low-frequency is Rs+Rct
    omega = np.array([1e8, 1e-6])
    z = randles_impedance(omega, rs_ohm=8.0, rct_ohm=12.0, cdl_F=50e-6)
    assert z[0].real == pytest.approx(8.0, rel=1e-3)
    assert z[0].imag == pytest.approx(0.0, abs=1e-3)
    assert z[1].real == pytest.approx(20.0, rel=1e-3)
    assert z[1].imag == pytest.approx(0.0, abs=1e-3)


def test_capacitive_loop_has_negative_imaginary_part():
    z = randles_impedance(np.array([2 * np.pi * 100.0]), 8.0, 12.0, 50e-6)
    assert z[0].imag < 0
    assert z[0].real > 8.0 and z[0].real < 20.0


def test_warburg_raises_no_infinite_plateau():
    # With Warburg diffusion, Re(Z) keeps growing at low frequency (no plateau)
    omega = 2 * np.pi * np.logspace(0, -4, 9)
    z = randles_impedance(omega, 8.0, 12.0, 50e-6, sigma_ohm_s_neg_half=3.0)
    assert z.real[-1] > z.real[0] + 3.0


def test_cpe_alpha_one_is_ideal_capacitor():
    omega = np.array([10.0, 1000.0])
    assert cpe_impedance(omega, q=50e-6, alpha=1.0) == pytest.approx(
        1.0 / (1j * omega * 50e-6))


def test_cpe_rejects_invalid_alpha():
    with pytest.raises(ValueError, match="alpha"):
        cpe_impedance(np.array([1.0]), q=1.0, alpha=1.5)


def test_randles_cpe_alpha_one_matches_randles():
    omega = 2 * np.pi * np.logspace(5, -2, 30)
    z_cpe = randles_cpe_impedance(omega, 8.0, 12.0, 50e-6, 1.0, sigma_ohm_s_neg_half=3.0)
    z_ideal = randles_impedance(omega, 8.0, 12.0, 50e-6, sigma_ohm_s_neg_half=3.0)
    assert z_cpe == pytest.approx(z_ideal, rel=1e-10)


def test_fit_recovers_noiseless_randles_parameters():
    freq = np.logspace(5, -2, 40)
    z_true = randles_impedance(2 * np.pi * freq, 8.0, 12.0, 50e-6)
    fit = fit_randles_spectrum(freq, z_true)
    assert fit.converged
    assert fit.rs_ohm == pytest.approx(8.0, rel=1e-3)
    assert fit.rct_ohm == pytest.approx(12.0, rel=1e-3)
    assert fit.cdl_F == pytest.approx(50e-6, rel=1e-3)
    assert fit.sigma_warburg_ohm_s_neg_half is None
    assert fit.r_squared_magnitude == pytest.approx(1.0, rel=1e-9)


def test_fit_recovers_warburg_parameters():
    freq = np.logspace(5, -2, 50)
    z_true = randles_impedance(2 * np.pi * freq, 8.0, 12.0, 50e-6, 3.0)
    fit = fit_randles_spectrum(freq, z_true, include_warburg=True)
    assert fit.converged
    assert fit.rs_ohm == pytest.approx(8.0, rel=1e-3)
    assert fit.rct_ohm == pytest.approx(12.0, rel=1e-3)
    assert fit.cdl_F == pytest.approx(50e-6, rel=1e-3)
    assert fit.sigma_warburg_ohm_s_neg_half == pytest.approx(3.0, rel=1e-3)
    assert fit.r_squared_magnitude == pytest.approx(1.0, rel=1e-6)


def test_fit_recovers_noisy_synthetic_spectrum():
    spectrum = synthetic_randles_spectrum(8.0, 12.0, 50e-6, sigma_ohm_s_neg_half=3.0,
                                          freq_min_hz=0.01, freq_max_hz=1e5,
                                          noise_rel=0.01)
    z = spectrum["z_real_ohm"].to_numpy() + 1j * spectrum["z_imag_ohm"].to_numpy()
    fit = fit_randles_spectrum(spectrum["frequency_hz"].to_numpy(), z,
                               include_warburg=True)
    assert fit.converged
    assert fit.rs_ohm == pytest.approx(8.0, rel=0.1)
    assert fit.rct_ohm == pytest.approx(12.0, rel=0.1)
    assert fit.cdl_F == pytest.approx(50e-6, rel=0.25)


def test_fit_rejects_insufficient_points():
    freq = np.array([1e5, 1e3, 1e1])
    z = randles_impedance(2 * np.pi * freq, 8.0, 12.0, 50e-6)
    with pytest.raises(ValueError, match="valid points"):
        fit_randles_spectrum(freq, z)


def test_exchange_current_from_rct_matches_butler_volmer_linearization():
    # i0 = RT / (n F Rct) for the whole measured electrode
    rct = 12.0
    expected = GAS_CONSTANT * 333.15 / (2 * FARADAY_CONSTANT * rct)
    assert exchange_current_from_rct(rct, n_electrons=2,
                                     temperature_K=333.15) == pytest.approx(expected)
    with pytest.raises(ValueError, match="positive"):
        exchange_current_from_rct(-1.0)


def test_template_loads(tmp_path):
    data = load_spectrum("experiments/data/eis_template.csv")
    assert summarize_spectrum(data)["n_points"] == 1
    assert "z_magnitude_ohm" in data.columns
    assert "z_real_ohm_cm2" in data.columns


def test_loader_rejects_missing_required_column(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"frequency_hz": [100.0], "z_real_ohm": [9.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="z_imag_ohm"):
        load_spectrum(path)


def test_loader_rejects_non_positive_frequency(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"frequency_hz": [0.0], "z_real_ohm": [9.0],
                  "z_imag_ohm": [-1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="positive"):
        load_spectrum(path)


def test_summarize_detects_semicircle_top_and_intercepts():
    freq = np.logspace(5, -2, 40)
    z = randles_impedance(2 * np.pi * freq, 8.0, 12.0, 50e-6)
    data = pd.DataFrame({"frequency_hz": freq, "z_real_ohm": z.real,
                         "z_imag_ohm": z.imag})
    summary = summarize_spectrum(data)
    assert summary["high_freq_real_ohm"] == pytest.approx(8.0, abs=0.1)
    assert summary["low_freq_real_ohm"] == pytest.approx(20.0, rel=1e-2)
    # The semicircle top is near 1/(2π Rct Cdl) ≈ 265 Hz
    assert summary["semicircle_top_freq_hz"] == pytest.approx(265.0, rel=0.3)
