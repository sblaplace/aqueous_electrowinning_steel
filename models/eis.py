"""Electrochemical impedance spectroscopy (EIS) models and fitting for Phase I.

Implements equivalent-circuit impedance models (Randles cell, constant phase
element, semi-infinite Warburg diffusion), complex non-linear least-squares
spectrum fitting, and conversion of charge-transfer resistance into an
exchange current for comparison with the Tafel analysis in `models/tafel.py`.

Sign convention: a capacitive semicircle has negative imaginary impedance
(the standard EIS convention). Nyquist plots therefore show -Im(Z).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .electrochemistry import FARADAY, R_GAS

REQUIRED_COLUMNS = {"frequency_hz", "z_real_ohm", "z_imag_ohm"}
OPTIONAL_COLUMNS = {
    "z_magnitude_ohm", "phase_deg", "working_electrode_area_cm2",
    "dc_bias_V_vs_ref", "temperature_C", "pH", "fe2_concentration_M",
    "electrolyte_id", "reference_electrode", "notes",
}


def load_spectrum(path: str | Path) -> pd.DataFrame:
    """Load an EIS spectrum CSV and validate/derive its numeric columns.

    The canonical schema (see ``experiments/data/eis_template.csv``) holds one
    frequency point per row. Imaginary impedance is negative for the
    capacitive semicircle. If the electrode area is present, area-normalized
    columns (ohm cm²) are derived.
    """
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    numeric = ["frequency_hz", "z_real_ohm", "z_imag_ohm", "z_magnitude_ohm",
               "phase_deg", "working_electrode_area_cm2", "dc_bias_V_vs_ref",
               "temperature_C", "pH", "fe2_concentration_M"]
    for column in set(numeric) & set(frame.columns):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if (frame["frequency_hz"] <= 0).any():
        raise ValueError("frequency_hz must be positive")
    if "z_magnitude_ohm" not in frame.columns:
        frame["z_magnitude_ohm"] = np.hypot(frame["z_real_ohm"], frame["z_imag_ohm"])
    if "phase_deg" not in frame.columns:
        frame["phase_deg"] = -np.degrees(
            np.arctan2(frame["z_imag_ohm"], frame["z_real_ohm"]))
    if "working_electrode_area_cm2" in frame.columns:
        if (frame["working_electrode_area_cm2"] <= 0).any():
            raise ValueError("working_electrode_area_cm2 must be positive")
        frame["z_real_ohm_cm2"] = \
            frame["z_real_ohm"] * frame["working_electrode_area_cm2"]
        frame["z_imag_ohm_cm2"] = \
            frame["z_imag_ohm"] * frame["working_electrode_area_cm2"]
    return frame


def _semicircle_top_frequency(freq_hz: np.ndarray, z_complex: np.ndarray) -> float:
    """Frequency of the faradaic arc top: the first local maximum of -Im(Z)
    scanning from the high-frequency end.

    With a Warburg tail, -Im(Z) grows again at the lowest frequencies, so a
    plain argmin(Im) would wrongly return the lowest measured point. Falls
    back to the global -Im maximum when no local peak exists.
    """
    order = np.argsort(freq_hz)[::-1]  # high to low frequency
    neg_im = -z_complex.imag[order]
    threshold = 0.05 * float(neg_im.max())
    for k in range(1, len(neg_im)):
        if neg_im[k] < neg_im[k - 1] and neg_im[k - 1] >= threshold:
            return float(freq_hz[order[k - 1]])
    return float(freq_hz[order[int(np.argmax(neg_im))]])


def summarize_spectrum(data: pd.DataFrame) -> dict:
    """Return basic, unit-labelled metrics for a loaded spectrum.

    ``semicircle_top_freq_hz`` is the frequency of the faradaic arc top, i.e.
    the first -Im(Z) peak from the high-frequency end (not the low-frequency
    Warburg tail).
    """
    if data.empty:
        raise ValueError("Cannot summarize an empty spectrum")
    freq = data["frequency_hz"].to_numpy(float)
    z = data["z_real_ohm"].to_numpy(float) + 1j * data["z_imag_ohm"].to_numpy(float)
    hi, lo = int(np.argmax(freq)), int(np.argmin(freq))
    return {
        "n_points": int(len(data)),
        "frequency_min_hz": float(freq.min()),
        "frequency_max_hz": float(freq.max()),
        "decades": float(np.log10(freq.max() / freq.min())),
        "z_magnitude_max_ohm": float(np.abs(z).max()),
        "z_magnitude_min_ohm": float(np.abs(z).min()),
        "high_freq_real_ohm": float(z.real[hi]),
        "low_freq_real_ohm": float(z.real[lo]),
        "semicircle_top_freq_hz": _semicircle_top_frequency(freq, z),
    }


def cpe_impedance(omega_rad_s: np.ndarray, q: float, alpha: float) -> np.ndarray:
    """Constant phase element impedance  Z = 1 / (Q (jω)^α).

    α = 1 recovers an ideal capacitor with C = Q; α = 0.5 is the distributed
    (Warburg) limit. Units of Q are S·s^α (F for α = 1).
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must lie in (0, 1]")
    omega = np.asarray(omega_rad_s, dtype=float)
    return 1.0 / (q * (1j * omega) ** alpha)


def warburg_impedance(omega_rad_s: np.ndarray, sigma_ohm_s_neg_half: float) -> np.ndarray:
    """Semi-infinite Warburg diffusion impedance  Z = σ(1 − j) / √ω.

    σ is the Warburg coefficient in Ω·s^(−1/2).
    """
    omega = np.asarray(omega_rad_s, dtype=float)
    return sigma_ohm_s_neg_half * (1.0 - 1j) / np.sqrt(omega)


def randles_impedance(omega_rad_s: np.ndarray, rs_ohm: float, rct_ohm: float,
                      cdl_F: float,
                      sigma_ohm_s_neg_half: float | None = None) -> np.ndarray:
    """Randles cell: Rs in series with (Cdl ∥ (Rct + Z_W)).

    With ``sigma_ohm_s_neg_half`` unset the classic semi-circular Randles
    response is recovered:  Z = Rs + Rct / (1 + jωRct·Cdl).
    """
    omega = np.asarray(omega_rad_s, dtype=float)
    z_faradaic = rct_ohm
    if sigma_ohm_s_neg_half is not None:
        z_faradaic = rct_ohm + warburg_impedance(omega, sigma_ohm_s_neg_half)
    z_parallel = 1.0 / (1j * omega * cdl_F + 1.0 / z_faradaic)
    return rs_ohm + z_parallel


def randles_cpe_impedance(omega_rad_s: np.ndarray, rs_ohm: float, rct_ohm: float,
                          q: float, alpha: float,
                          sigma_ohm_s_neg_half: float | None = None) -> np.ndarray:
    """Randles cell with the double layer represented by a CPE."""
    omega = np.asarray(omega_rad_s, dtype=float)
    z_faradaic = rct_ohm
    if sigma_ohm_s_neg_half is not None:
        z_faradaic = rct_ohm + warburg_impedance(omega, sigma_ohm_s_neg_half)
    z_cpe = cpe_impedance(omega, q, alpha)
    return rs_ohm + 1.0 / (1.0 / z_cpe + 1.0 / z_faradaic)


@dataclass(frozen=True)
class RandlesFit:
    rs_ohm: float
    rct_ohm: float
    cdl_F: float
    sigma_warburg_ohm_s_neg_half: float | None
    chi_squared: float
    r_squared_magnitude: float
    n_points: int
    frequency_min_hz: float
    frequency_max_hz: float
    converged: bool


def fit_randles_spectrum(freq_hz: np.ndarray, z_complex: np.ndarray,
                         include_warburg: bool = False) -> RandlesFit:
    """Fit a Randles equivalent circuit to measured impedance by complex NLLS.

    Residuals are weighted by 1/|Z| (relative weighting) so that the small
    high-frequency part of the spectrum is not dominated by the large
    low-frequency response, and several starting points are tried to avoid
    local minima. ``chi_squared`` is the sum of weighted squared residuals.
    """
    freq = np.asarray(freq_hz, dtype=float)
    z = np.asarray(z_complex, dtype=complex)
    valid = np.isfinite(freq) & np.isfinite(z.real) & np.isfinite(z.imag) & (freq > 0)
    if valid.sum() < (6 if include_warburg else 5):
        raise ValueError(
            f"Randles fit requires at least {6 if include_warburg else 5} valid points")
    freq, z = freq[valid], z[valid]
    weight = np.abs(z)
    weight = np.where(weight <= 0, weight[weight > 0].min(), weight)

    from scipy.optimize import least_squares

    def unpack(p: np.ndarray):
        return p[0], p[1], p[2], (p[3] if include_warburg else None)

    def residuals(p: np.ndarray) -> np.ndarray:
        model = randles_impedance(2.0 * np.pi * freq, *unpack(p))
        diff = model - z
        return np.concatenate([diff.real / weight, diff.imag / weight])

    omega = 2.0 * np.pi * freq
    hi, lo = int(np.argmax(freq)), int(np.argmin(freq))
    rs0 = max(float(z.real[hi]), 1e-9)
    rct0 = max(float(z.real[lo]) - rs0, 1e-9)
    cdl0 = 1.0 / (2.0 * np.pi * _semicircle_top_frequency(freq, z) * rct0)
    starts = [(rs0, rct0, cdl0), (rs0, rct0, 1e-4), (1.0, rct0, cdl0)]
    if include_warburg:
        sigma0 = max(float(-z.imag[lo]) * np.sqrt(omega[lo]), 1e-9)
        starts = [s + (sigma0,) for s in starts]

    lower = [1e-9, 1e-9, 1e-12] + ([1e-9] if include_warburg else [])
    upper = [1e6, 1e9, 1.0] + ([1e9] if include_warburg else [])
    best = None
    for p0 in starts:
        try:
            result = least_squares(residuals, np.asarray(p0, float),
                                   bounds=(lower, upper))
        except Exception:
            continue
        if best is None or result.cost < best.cost:
            best = result
    if best is None:
        raise RuntimeError("Randles fit failed to converge from any starting point")

    rs, rct, cdl, sigma = unpack(best.x)
    model = randles_impedance(2.0 * np.pi * freq, rs, rct, cdl, sigma)
    mag_data, mag_model = np.abs(z), np.abs(model)
    ss_res = float(np.sum((mag_data - mag_model) ** 2))
    ss_tot = float(np.sum((mag_data - mag_data.mean()) ** 2))
    r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return RandlesFit(
        float(rs), float(rct), float(cdl),
        None if sigma is None else float(sigma),
        float(2.0 * best.cost), float(r2), int(len(z)),
        float(freq.min()), float(freq.max()), bool(best.success),
    )


def exchange_current_from_rct(rct_ohm: float, n_electrons: int = 2,
                              temperature_K: float = 298.15) -> float:
    """Exchange current (A) implied by a charge-transfer resistance (ohm).

    Linearizing Butler–Volmer about equilibrium gives i₀ = RT / (nF·Rct)
    for the whole measured electrode; divide by the electrode area for an
    exchange current density. Valid for small perturbations about an
    equilibrium potential; at a large cathodic DC bias the measured Rct
    instead reflects the combined faradaic conductance of Fe deposition and
    competing HER.
    """
    if rct_ohm <= 0:
        raise ValueError("rct_ohm must be positive")
    return (R_GAS * temperature_K
            / (n_electrons * FARADAY * rct_ohm))


def synthetic_randles_spectrum(rs_ohm: float, rct_ohm: float, cdl_F: float,
                               sigma_ohm_s_neg_half: float | None,
                               freq_min_hz: float, freq_max_hz: float,
                               points_per_decade: int = 10,
                               noise_rel: float = 0.0, seed: int = 42,
                               area_cm2: float = 1.0,
                               dc_bias_V_vs_ref: float = -0.70,
                               ) -> pd.DataFrame:
    """Generate a synthetic Randles(+Warburg) spectrum in the canonical schema.

    ``noise_rel`` adds Gaussian relative noise (fraction of |Z|) to each
    component. The returned frame carries Phase I metadata columns so it can
    be written directly and re-loaded with :func:`load_spectrum`.
    """
    if not 0 < freq_min_hz < freq_max_hz:
        raise ValueError("Require 0 < freq_min_hz < freq_max_hz")
    n_dec = np.log10(freq_max_hz / freq_min_hz)
    n = max(int(round(points_per_decade * n_dec)), 2)
    freq = np.logspace(np.log10(freq_max_hz), np.log10(freq_min_hz), n)
    omega = 2.0 * np.pi * freq
    z = randles_impedance(omega, rs_ohm, rct_ohm, cdl_F, sigma_ohm_s_neg_half)
    rng = np.random.default_rng(seed)
    mag = np.abs(z)
    z_real = z.real + rng.normal(0.0, noise_rel, n) * mag
    z_imag = z.imag + rng.normal(0.0, noise_rel, n) * mag
    return pd.DataFrame({
        "frequency_hz": freq,
        "z_real_ohm": z_real,
        "z_imag_ohm": z_imag,
        "z_magnitude_ohm": np.hypot(z_real, z_imag),
        "phase_deg": -np.degrees(np.arctan2(z_imag, z_real)),
        "working_electrode_area_cm2": area_cm2,
        "dc_bias_V_vs_ref": dc_bias_V_vs_ref,
        "temperature_C": 60.0,
        "pH": 3.0,
        "fe2_concentration_M": 1.0,
        "electrolyte_id": "FE-SO4-SYNTHETIC",
        "reference_electrode": "Ag/AgCl",
        "notes": "Synthetic Randles spectrum; imaginary part negative for the "
                 "capacitive semicircle",
    })


def nyquist_plot(z_data: np.ndarray, z_model: np.ndarray | None = None, ax=None):
    """Nyquist plot (-Im(Z) vs Re(Z)); returns the matplotlib Axes."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    z_data = np.asarray(z_data, dtype=complex)
    ax.plot(z_data.real, -z_data.imag, "o", markersize=5, label="data")
    if z_model is not None:
        z_model = np.asarray(z_model, dtype=complex)
        ax.plot(z_model.real, -z_model.imag, "-", linewidth=1.5, label="Randles fit")
    ax.set(xlabel="Re(Z) (Ω)", ylabel="-Im(Z) (Ω)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    return ax


def bode_plot(freq_hz: np.ndarray, z_data: np.ndarray,
              z_model: np.ndarray | None = None, axes=None):
    """Bode plot (|Z| and phase vs frequency); returns the matplotlib axes."""
    import matplotlib.pyplot as plt
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(6.5, 7), sharex=True)
    freq, z_data = np.asarray(freq_hz, float), np.asarray(z_data, complex)
    phase_deg = -np.degrees(np.angle(z_data))
    axes[0].plot(freq, np.abs(z_data), "o", markersize=5, label="data")
    axes[1].plot(freq, phase_deg, "o", markersize=5, label="data")
    if z_model is not None:
        z_model = np.asarray(z_model, complex)
        axes[0].plot(freq, np.abs(z_model), "-", linewidth=1.5, label="Randles fit")
        axes[1].plot(freq, -np.degrees(np.angle(z_model)), "-", linewidth=1.5,
                     label="Randles fit")
    axes[0].set(xscale="log", yscale="log", ylabel=r"$|Z|$ (Ω)")
    axes[1].set(xscale="log", ylabel="Phase (deg, $-\\varphi$)",
                xlabel="Frequency (Hz)")
    for ax in axes:
        ax.grid(alpha=0.25, which="both")
    return axes
