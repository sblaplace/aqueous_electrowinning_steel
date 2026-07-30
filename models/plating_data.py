"""Plating data loader — load, validate, compute, and flag.

Loads an experiment directory containing ``manifest.json``,
``timeseries.csv``, ``mass_log.csv``, and optionally ``video_index.csv``.

Derived quantities
------------------
- **Faradaic efficiency**: from raw coupon mass gain and integrated
  cathodic charge — never from a user-entered FE value.
- **Charge passed (C)**: trapezoidal integration of cathodic current.
- **Current density**: charge / (area × time).
- **Energy (Wh)**: ∫ V × I dt.

Anomaly detection
-----------------
- Current drift > 5% of setpoint triggers a flag.
- Temperature drift > 2 °C from the first reading triggers a flag.
- Missing timestamps (gaps > 2× median step) trigger a flag.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd

# ── Physical constants (from electrochemistry.py) ─────────────────
from .electrochemistry import FARADAY as FARADAY_C_MOL, M_FE_G as M_FE_G_MOL, Z_FE


# ── Data classes ────────────────────────────────────────────────────

@dataclass
class AnomalyFlag:
    """One anomaly detected during validation."""
    kind: str
    message: str
    severity: str = "warning"   # "warning" | "error"
    timestamp_s: float | None = None


@dataclass
class PlatingDerived:
    """Computed quantities from raw instrument data."""
    charge_C: float                           # total cathodic charge
    duration_s: float                         # experiment duration
    mean_cathodic_current_A: float            # Q / t
    mean_voltage_V: float                     # time-averaged cell voltage
    energy_Wh: float                          # ∫ V*I dt / 3600
    current_density_mA_cm2: float | None      # if area known
    faradaic_efficiency: float | None         # if mass_log present
    faradaic_efficiency_percent: float | None
    theoretical_fe_mass_g: float | None
    net_deposit_mass_g: float | None


@dataclass
class PlatingRun:
    """Complete loaded and validated experiment."""
    manifest: dict[str, Any]
    timeseries: pd.DataFrame
    mass_log: pd.DataFrame | None
    video_index: pd.DataFrame | None
    derived: PlatingDerived
    anomalies: list[AnomalyFlag]

    def summary(self) -> dict[str, Any]:
        """JSON-serializable summary."""
        return {
            "run_id": self.manifest.get("run_id"),
            "experiment_type": self.manifest.get("experiment_type"),
            "bath_batch": self.manifest.get("bath_batch"),
            "n_timeseries_rows": len(self.timeseries),
            "duration_s": self.derived.duration_s,
            "charge_C": self.derived.charge_C,
            "mean_cathodic_current_A": self.derived.mean_cathodic_current_A,
            "mean_voltage_V": self.derived.mean_voltage_V,
            "energy_Wh": self.derived.energy_Wh,
            "current_density_mA_cm2": self.derived.current_density_mA_cm2,
            "faradaic_efficiency_percent": self.derived.faradaic_efficiency_percent,
            "net_deposit_mass_g": self.derived.net_deposit_mass_g,
            "n_anomalies": len(self.anomalies),
            "anomalies": [
                {"kind": a.kind, "message": a.message, "severity": a.severity,
                 "timestamp_s": a.timestamp_s}
                for a in self.anomalies
            ],
        }


# ── Loaders ─────────────────────────────────────────────────────────

def _load_csv(path: Path, required_columns: set[str], label: str) -> pd.DataFrame:
    """Load a CSV and validate required columns exist."""
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    frame = pd.read_csv(path, keep_default_na=True)
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {', '.join(sorted(missing))}")
    for col in frame.columns:
        if col != "notes":
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if frame.empty:
        raise ValueError(f"{label} must contain at least one row")
    return frame


def load_timeseries(path: Path) -> pd.DataFrame:
    """Load a timeseries CSV with at least timestamp_s and current_actual_A."""
    return _load_csv(path, {"timestamp_s", "current_actual_A", "voltage_V"}, "Timeseries")


def load_mass_log(path: Path) -> pd.DataFrame:
    """Load a mass_log CSV with mass_before_g and mass_after_g."""
    return _load_csv(path, {"mass_before_g", "mass_after_g"}, "Mass log")


def load_video_index(path: Path) -> pd.DataFrame | None:
    """Load video_index.csv if it exists; return None otherwise."""
    if not path.is_file():
        return None
    return _load_csv(path, {"timestamp_s", "camera", "filename"}, "Video index")


def load_manifest(path: Path) -> dict[str, Any]:
    """Load the experiment manifest JSON."""
    if not path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Manifest is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Manifest must be a JSON object")
    required = {"run_id", "experiment_type", "bath_batch"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Manifest missing required keys: {', '.join(sorted(missing))}")
    return data


# ── Derived quantities ──────────────────────────────────────────────

def _integrate_cathodic_charge(
    time: np.ndarray,
    current: np.ndarray,
    cathodic_sign: str = "negative",
) -> tuple[float, float]:
    """Integrate cathodic charge.  Returns (charge_C, duration_s)."""
    if len(time) < 2:
        raise ValueError("At least two time points required to integrate charge")
    if cathodic_sign == "negative":
        cathodic = np.clip(-current, 0.0, None)
    else:
        cathodic = np.clip(current, 0.0, None)
    charge = float(np.trapezoid(cathodic, time))
    duration = float(time[-1] - time[0])
    return charge, duration


def compute_derived(
    timeseries: pd.DataFrame,
    mass_log: pd.DataFrame | None = None,
    cathode_area_cm2: float | None = None,
    cathodic_sign: str = "negative",
) -> PlatingDerived:
    """Compute all derived quantities from raw data.

    Faradaic efficiency is computed from raw mass gain and integrated
    charge — never from a user-entered FE value.
    """
    time = timeseries["timestamp_s"].to_numpy(float)
    current = timeseries["current_actual_A"].to_numpy(float)
    voltage = timeseries["voltage_V"].to_numpy(float)

    charge_C, duration_s = _integrate_cathodic_charge(time, current, cathodic_sign)

    mean_current = charge_C / duration_s if duration_s > 0 else 0.0
    mean_voltage = float(np.nanmean(voltage))
    # Energy = ∫ V * |I| dt  (total energy into cell, regardless of sign)
    abs_current = np.abs(current)
    energy_J = float(np.trapezoid(voltage * abs_current, time))
    energy_Wh = energy_J / 3600.0

    # Current density — requires cathode area
    current_density = None
    if cathode_area_cm2 is not None and cathode_area_cm2 > 0:
        current_density = mean_current / cathode_area_cm2 * 1000.0  # mA/cm²

    # Faradaic efficiency — requires mass_log
    fe = None
    fe_pct = None
    theoretical_mass = None
    net_mass = None
    if mass_log is not None and not mass_log.empty:
        mass_before = float(mass_log["mass_before_g"].iloc[0])
        mass_after = float(mass_log["mass_after_g"].iloc[0])
        blank = 0.0
        if "blank_mass_change_g" in mass_log.columns:
            blank = float(mass_log["blank_mass_change_g"].iloc[0])
        mass_gain = mass_after - mass_before
        net_mass = mass_gain - blank
        theoretical_mass = charge_C * M_FE_G_MOL / (Z_FE * FARADAY_C_MOL)
        if theoretical_mass > 0:
            fe = net_mass / theoretical_mass
            fe_pct = fe * 100.0

    return PlatingDerived(
        charge_C=charge_C,
        duration_s=duration_s,
        mean_cathodic_current_A=mean_current,
        mean_voltage_V=mean_voltage,
        energy_Wh=energy_Wh,
        current_density_mA_cm2=current_density,
        faradaic_efficiency=fe,
        faradaic_efficiency_percent=fe_pct,
        theoretical_fe_mass_g=theoretical_mass,
        net_deposit_mass_g=net_mass,
    )


# ── Anomaly detection ───────────────────────────────────────────────

def detect_anomalies(
    timeseries: pd.DataFrame,
    current_drift_threshold: float = 0.05,
    temperature_drift_threshold_C: float = 2.0,
) -> list[AnomalyFlag]:
    """Flag anomalies in a timeseries trace.

    Current drift: max deviation from setpoint > threshold × |setpoint|.
    Temperature drift: max deviation from first reading > threshold.
    Missing timestamps: gaps > 2× the median step.
    """
    flags: list[AnomalyFlag] = []

    # ── Current drift ─────────────────────────────────────────────────
    if "current_setpoint_A" in timeseries.columns:
        setpoint = timeseries["current_setpoint_A"].to_numpy(float)
        actual = timeseries["current_actual_A"].to_numpy(float)
        time = timeseries["timestamp_s"].to_numpy(float)
        sp_abs = np.abs(setpoint)
        # Avoid division by zero when setpoint is 0
        nonzero = sp_abs > 1e-12
        if nonzero.any():
            deviation = np.abs(actual[nonzero] - setpoint[nonzero]) / sp_abs[nonzero]
            max_dev = float(np.nanmax(deviation))
            if max_dev > current_drift_threshold:
                worst_idx = int(np.nanargmax(deviation))
                flags.append(AnomalyFlag(
                    kind="current_drift",
                    message=(
                        f"Current drift {max_dev*100:.1f}% exceeds "
                        f"{current_drift_threshold*100:.0f}% threshold"),
                    timestamp_s=float(time[nonzero][worst_idx]),
                ))

    # ── Temperature drift ─────────────────────────────────────────────
    if "temperature_C" in timeseries.columns:
        temp = timeseries["temperature_C"].to_numpy(float)
        time = timeseries["timestamp_s"].to_numpy(float)
        valid = np.isfinite(temp)
        if valid.any():
            first_temp = float(temp[valid][0])
            max_drift = float(np.nanmax(np.abs(temp - first_temp)))
            if max_drift > temperature_drift_threshold_C:
                worst_idx = int(np.nanargmax(np.abs(temp - first_temp)))
                flags.append(AnomalyFlag(
                    kind="temperature_drift",
                    message=(
                        f"Temperature drift {max_drift:.1f}°C exceeds "
                        f"{temperature_drift_threshold_C:.1f}°C threshold"),
                    timestamp_s=float(time[worst_idx]),
                ))

    # ── Missing timestamps ────────────────────────────────────────────
    time = timeseries["timestamp_s"].to_numpy(float)
    if len(time) >= 2:
        dt = np.diff(time)
        median_dt = float(np.nanmedian(dt))
        if median_dt > 0:
            gaps = dt > 2.0 * median_dt
            if gaps.any():
                gap_indices = np.where(gaps)[0]
                for idx in gap_indices:
                    flags.append(AnomalyFlag(
                        kind="missing_timestamps",
                        message=(
                            f"Timestamp gap {dt[idx]:.1f}s > 2× median step "
                            f"{median_dt:.1f}s at t={float(time[idx+1]):.1f}s"),
                        timestamp_s=float(time[idx + 1]),
                    ))

    # ── NaN values ────────────────────────────────────────────────────
    for col in ("current_actual_A", "voltage_V"):
        if col in timeseries.columns:
            n_nan = int(timeseries[col].isna().sum())
            if n_nan > 0:
                flags.append(AnomalyFlag(
                    kind="missing_data",
                    message=f"{n_nan} NaN values in {col}",
                ))

    return flags


# ── Main loader ─────────────────────────────────────────────────────

def load_plating_run(
    experiment_dir: str | Path,
    cathode_area_cm2: float | None = None,
    cathodic_sign: str = "negative",
) -> PlatingRun:
    """Load a complete plating experiment directory.

    Expected files:
    - manifest.json   (required)
    - timeseries.csv  (required)
    - mass_log.csv    (optional — present for gravimetric FE)
    - video_index.csv (optional)

    ``cathode_area_cm2`` enables current-density calculation.  If omitted,
    current density will be None.
    """
    experiment_dir = Path(experiment_dir)

    manifest = load_manifest(experiment_dir / "manifest.json")
    timeseries = load_timeseries(experiment_dir / "timeseries.csv")

    mass_log_path = experiment_dir / "mass_log.csv"
    mass_log = load_mass_log(mass_log_path) if mass_log_path.is_file() else None

    video_index = load_video_index(experiment_dir / "video_index.csv")

    # Cathode area: try the function arg first, then manifest setup
    area = cathode_area_cm2
    if area is None:
        setup = manifest.get("setup", {})
        cathode = setup.get("cathode", {})
        area = cathode.get("area_cm2")

    derived = compute_derived(
        timeseries, mass_log,
        cathode_area_cm2=area,
        cathodic_sign=cathodic_sign,
    )
    anomalies = detect_anomalies(timeseries)

    return PlatingRun(
        manifest=manifest,
        timeseries=timeseries,
        mass_log=mass_log,
        video_index=video_index,
        derived=derived,
        anomalies=anomalies,
    )
