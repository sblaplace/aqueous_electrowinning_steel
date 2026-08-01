"""
Digital twin — real-time model updating from sensor data.

Continuously re-estimates design confidence by feeding P&ID sensor
readings back into the model chain via an Extended Kalman filter (EKF).

Architecture
------------
* **Sensor interface** – defines expected instrument streams and
  physical bounds for the electrowinning pilot plant.
* **State estimation** – EKF that maps sensor observations onto a
  reduced model state vector and propagates uncertainty.
* **Anomaly detection** – flags residual, drift, and rate-of-change
  violations against model predictions.
* **Prediction** – forward-projects state + confidence envelope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import the process model and bath dynamics
from .twin_physics import CellProcessModel
from . import bath_dynamics as _bath_dynamics
from .env_coupling import disturbance_from_environment


# ---------------------------------------------------------------------------
# Sensor definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SensorSpec:
    """Metadata for a single P&ID instrument."""
    tag: str
    quantity: str
    unit: str
    physical_min: float
    physical_max: float
    noise_std: float  # 1-sigma measurement noise


SENSOR_SPECS: Dict[str, SensorSpec] = {
    "TT-101": SensorSpec("TT-101", "catholyte_temperature", "C", 20.0, 95.0, 0.5),
    "TT-201": SensorSpec("TT-201", "anolyte_temperature", "C", 20.0, 95.0, 0.5),
    "pHAT-101": SensorSpec("pHAT-101", "catholyte_pH", "pH", 0.0, 14.0, 0.05),
    "AT-202": SensorSpec("AT-202", "anolyte_pH", "pH", 0.0, 14.0, 0.05),
    "FT-201": SensorSpec("FT-201", "catholyte_flow", "L_min", 0.0, 50.0, 0.2),
    "FT-202": SensorSpec("FT-202", "anolyte_flow", "L_min", 0.0, 50.0, 0.2),
    "FT-103": SensorSpec("FT-103", "electrolyte_flow", "L_min", 0.0, 100.0, 0.3),
    "AT-201": SensorSpec("AT-201", "dissolved_O2", "mg_L", 0.0, 20.0, 0.1),
    "AT-301A": SensorSpec("AT-301A", "gas_O2", "pct", 0.0, 100.0, 0.2),
    "AIT-501": SensorSpec("AIT-501", "O2_probe_mV", "mV", -500.0, 500.0, 1.0),
    "AIT-502": SensorSpec("AIT-502", "dew_point", "C", -40.0, 60.0, 0.3),
    "VT-201": SensorSpec("VT-201", "cell_voltage", "V", 0.0, 10.0, 0.01),
    "CT-201": SensorSpec("CT-201", "rectifier_current", "A", 0.0, 5000.0, 5.0),
    # Optional L1 sensor suite recommended by docs/TWIN_OBSERVABILITY.md (sm §3/§4).
    # These are OFF by default: nothing observes them unless a caller supplies the
    # tag in a readings dict.  Noise floors match the observability analysis
    # (`models/observability.py`).  THK-101 restores full observability (deposit
    # thickness is divergent-unobservable with the base 5-sensor suite); CVT-201 and
    # FE2P-101 fix residual conditioning (cell_voltage, bulk_fe2).
    "THK-101":  SensorSpec("THK-101",  "deposit_thickness", "um", 0.0, 500.0, 0.5),
    "CVT-201":  SensorSpec("CVT-201",  "cell_voltage", "V", 0.0, 10.0, 0.01),
    "FE2P-101": SensorSpec("FE2P-101", "bulk_fe2", "M", 0.0, 2.0, 0.02),
}

# Physical state vector bounded to what the physics predicts and the cell tracks:
# catholyte/anolyte temperature, bulk Fe²⁺, bulk pH, current density, deposit thickness, cell voltage.
STATE_KEYS: List[str] = [
    "catholyte_temperature",  # 0
    "anolyte_temperature",    # 1
    "bulk_fe2",               # 2  (Fe²⁺ concentration in M)
    "bulk_pH",                # 3
    "current_density",        # 4  (mA/cm²)
    "deposit_thickness",      # 5  (µm)
    "cell_voltage",           # 6  (V)
]

STATE_INDEX: Dict[str, int] = {k: i for i, k in enumerate(STATE_KEYS)}
N_STATES = len(STATE_KEYS)

# Observation matrix mapping: which sensor observes which state directly
# TT-101 (catholyte temp) -> 0
# TT-201 (anolyte temp) -> 1
# pHAT-101 (catholyte bulk pH) -> 3
# CT-201 (current density) -> 4
# VT-201 (cell voltage) -> 6
_OBS_MAP: Dict[str, int] = {
    "TT-101": 0,
    "TT-201": 1,
    "pHAT-101": 3,
    "CT-201": 4,
    "VT-201": 6,
}

# Sensor tags that can be used as observations (the current 5-sensor suite)
OBSERVABLE_TAGS: List[str] = list(_OBS_MAP.keys())

# Optional L1 sensor suite (recommended by docs/TWIN_OBSERVABILITY.md §3/§4).
# Each directly observes a state the base suite observes weakly or not at all:
#   THK-101  -> 5 deposit_thickness  (divergent-unobservable with the base suite)
#   CVT-201  -> 6 cell_voltage       (base VT-201 observes physics v_cell, not x[6])
#   FE2P-101 -> 2 bulk_fe2           (base suite only sees it via the v_cell coupling)
# OFF by default: a tag is only observed when it appears in a readings dict passed
# to `DigitalTwin.update`.  Supplying none of them keeps the twin byte-identical to
# the base 5-sensor twin, so this is a pure, opt-in capability.
L1_SENSOR_OBS_MAP: Dict[str, int] = {
    "THK-101": 5,
    "CVT-201": 6,
    "FE2P-101": 2,
}

# All observation tags the EKF can consume when present in a readings dict.
_ALL_OBS_MAP: Dict[str, int] = {**_OBS_MAP, **L1_SENSOR_OBS_MAP}


# ---------------------------------------------------------------------------
# Module level default process model caching for fast initialization in tests
# ---------------------------------------------------------------------------

_DEFAULT_MODEL: Optional[CellProcessModel] = None

def get_default_process_model() -> CellProcessModel:
    """Retrieve or build the default CellProcessModel."""
    global _DEFAULT_MODEL
    if _DEFAULT_MODEL is None:
        from .twin_physics import default_process_model
        _DEFAULT_MODEL = default_process_model()
    return _DEFAULT_MODEL


_DEFAULT_DP = {
    "temperature_C": 60.0,
    "pH": 3.5,
    "cell_voltage_V": 2.5,
    "j_avg_mA_cm2": 150.0,
    "electrode_area_m2": 1.0,
    "electrolyte_volume_L": 1000.0,
    "fe2_M": 1.0,
    # Bath dynamics defaults (conservation-law dynamics; see bath_dynamics.py)
    "recirculation_flow_L_hr": 6000.0,
    "reservoir_volume_L": 50000.0,
    "catholyte_volume_L": 800.0,
    "anolyte_volume_L": 2000.0,
    "fe2_makeup_rate_M_hr": 0.0,
    "fe2_reservoir_M": 1.0,
    "buffer_capacity_beta": 0.05,
    "acid_dose_rate_M_hr": 0.0,
    "pH_reservoir": 3.5,
    # cooling_power_W: not set — auto-balances Joule heating
    "T_reservoir_C": 55.0,
}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TwinState:
    """Snapshot of the digital twin after an update step."""
    timestamp_hr: float
    state_mean: np.ndarray          # (N_STATES,)
    state_covariance: np.ndarray    # (N_STATES, N_STATES)
    innovation: np.ndarray          # observation residual
    innovation_cov: np.ndarray      # S = HPH' + R
    anomalies: List["Anomaly"] = field(default_factory=list)

    @property
    def state_dict(self) -> Dict[str, float]:
        return {k: float(self.state_mean[i]) for k, i in STATE_INDEX.items()}

    @property
    def uncertainty_2sigma(self) -> Dict[str, float]:
        return {
            k: float(2.0 * math.sqrt(max(self.state_covariance[i, i], 0.0)))
            for k, i in STATE_INDEX.items()
        }


@dataclass
class PredictionEnvelope:
    """Forward prediction with growing uncertainty."""
    horizon_hr: float
    timestamps_hr: np.ndarray       # (T,)
    mean_trajectories: np.ndarray   # (T, N_STATES)
    sigma_trajectories: np.ndarray  # (T, N_STATES) — 1-sigma bounds
    confidence: np.ndarray          # (T,) — P(all specs met)


@dataclass
class Anomaly:
    """Flagged sensor inconsistency."""
    sensor_tag: str
    timestamp_hr: float
    kind: str                       # "residual", "drift", "rate_of_change"
    severity: float                 # dimensionless, higher = worse
    detail: str


# ---------------------------------------------------------------------------
# Synthetic sensor generator (for testing and driver)
# ---------------------------------------------------------------------------

def generate_synthetic_readings(
    design_point: Dict[str, float],
    t_hr: float,
    rng: np.random.Generator,
    fault: Optional[Dict[str, Any]] = None,
    include_l1_sensors: bool = False,
) -> Dict[str, float]:
    """Generate plausible sensor readings for a given time.

    Parameters
    ----------
    design_point : dict
        Nominal operating point.
    t_hr : float
        Time in hours (for slow drift simulation).
    rng : numpy Generator
        Random number generator.
    fault : dict, optional
        If set, inject a fault: {"tag": sensor, "kind": "bias"|"stuck"|"spike",
        "magnitude": float}.
    include_l1_sensors : bool, default False
        If True, also emit the optional L1 sensor suite (THK-101, CVT-201,
        FE2P-101) that directly observe deposit thickness, cell voltage and bulk
        Fe2+.  OFF by default so the base stream is unchanged.
    """
    dp = design_point
    readings: Dict[str, float] = {}

    # Slow sinusoidal drift in temperature
    temp_drift = 2.0 * math.sin(2.0 * math.pi * t_hr / 24.0)

    readings["TT-101"] = dp.get("temperature_C", 60.0) + temp_drift + rng.normal(0, 0.5)
    readings["TT-201"] = dp.get("temperature_C", 60.0) + temp_drift + 1.5 + rng.normal(0, 0.5)
    readings["pHAT-101"] = dp.get("pH", 3.5) + 0.1 * math.sin(2 * math.pi * t_hr / 12.0) + rng.normal(0, 0.05)
    readings["AT-202"] = dp.get("pH", 3.5) + 0.15 + rng.normal(0, 0.05)
    readings["FT-201"] = 10.0 + 0.5 * math.sin(2 * math.pi * t_hr / 6.0) + rng.normal(0, 0.2)
    readings["FT-202"] = 10.2 + rng.normal(0, 0.2)
    readings["FT-103"] = 20.0 + rng.normal(0, 0.3)
    readings["AT-201"] = 7.5 + 0.3 * math.sin(2 * math.pi * t_hr / 8.0) + rng.normal(0, 0.1)
    readings["AT-301A"] = 20.9 + rng.normal(0, 0.2)
    readings["AIT-501"] = 420.0 + 5.0 * math.sin(2 * math.pi * t_hr / 24.0) + rng.normal(0, 1.0)
    readings["AIT-502"] = 15.0 + rng.normal(0, 0.3)
    # VT-201: the measurement model h_obs reports the physics-predicted v_cell,
    # so a self-consistent synthetic stream must too.  (Emitting an arbitrary
    # design ``cell_voltage_V`` here makes the EKF see a persistent large
    # innovation, which destabilises the filter once the states track well.)
    pmodel = get_default_process_model()
    _p = pmodel.predict(
        dp.get("j_avg_mA_cm2", 150.0), dp.get("temperature_C", 60.0),
        dp.get("fe2_M", 1.0))
    readings["VT-201"] = _p.v_cell_V + 0.05 * math.sin(2 * math.pi * t_hr / 4.0) + rng.normal(0, 0.01)
    readings["CT-201"] = dp.get("j_avg_mA_cm2", 150.0) * dp.get("electrode_area_m2", 1.0) * 10.0 + rng.normal(0, 5.0)

    # Optional L1 sensor suite (opt-in, disabled by default).  Emitted values are
    # self-consistent with `h_obs` so the EKF sees a coherent stream: THK-101 reads
    # the deposit-thickness design point (state 5), CVT-201 the cell_voltage design
    # point (state 6), FE2P-101 the bulk Fe2+ design point (state 2).
    if include_l1_sensors:
        readings["THK-101"] = dp.get("deposit_thickness_um", 0.0) + rng.normal(0, 0.5)
        readings["CVT-201"] = dp.get("cell_voltage_V", _p.v_cell_V) + rng.normal(0, 0.01)
        readings["FE2P-101"] = dp.get("fe2_M", 1.0) + rng.normal(0, 0.02)

    # Inject fault
    if fault is not None:
        tag = fault["tag"]
        kind = fault["kind"]
        mag = fault.get("magnitude", 5.0)
        if tag in readings:
            if kind == "bias":
                readings[tag] += mag
            elif kind == "stuck":
                readings[tag] = fault.get("stuck_value", readings[tag])
            elif kind == "spike":
                if fault.get("spike_active", True):
                    readings[tag] += mag

    return readings


# ---------------------------------------------------------------------------
# EKF implementation helper functions
# ---------------------------------------------------------------------------

def _f_state_transition(
    x: np.ndarray,
    dt_hr: float,
    model: Optional[CellProcessModel] = None,
    design_point: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Deterministic state transition using coupled bath/recirculation dynamics.

    Each state's dynamics come from a conservation law (Fe2+ mass, acid/base +
    buffer for pH, thermal balance for T) plus recirculation exchange between
    the cell and a finite reservoir.  Cell voltage relaxes toward the physics
    model's predicted voltage through a state-dependent electrical time constant.

    The auxiliary reservoir state (T_reservoir, fe2_reservoir, pH_reservoir)
    is stored in ``design_point["_bath_aux"]`` and advanced alongside the EKF
    state.  It is *not* part of the 7-state EKF vector.
    """
    if model is None:
        model = get_default_process_model()
    if design_point is None:
        dp = _DEFAULT_DP
    else:
        dp = design_point

    # Retrieve (or create) the auxiliary reservoir state
    aux = _bath_dynamics.get_aux(dp)

    # Advance both the EKF state and the auxiliary reservoir
    x_next, aux_next = _bath_dynamics.step(x, aux, dt_hr, dp, model)

    # Write the advanced aux back into design_point so the next predict step
    # picks up where this one left off.
    _bath_dynamics.set_aux(dp, aux_next)

    return x_next


def _F_jacobian(
    x: np.ndarray,
    dt_hr: float,
    model: Optional[CellProcessModel] = None,
    design_point: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Numerical Jacobian of the state transition function.

    Snapshots the auxiliary reservoir state before each perturbation so the
    Jacobian is computed consistently (the aux state advances only once, via
    the final ``_f_state_transition`` call in ``predict``).
    """
    if model is None:
        model = get_default_process_model()
    if design_point is None:
        dp = _DEFAULT_DP
    else:
        dp = design_point

    n = len(x)
    F = np.zeros((n, n))
    eps = 1e-5

    # Snapshot the auxiliary state so perturbations don't accumulate aux drift
    aux_snapshot = _bath_dynamics.get_aux(dp)

    fx = _f_state_transition(x, dt_hr, model, dp)
    # Restore aux after the base evaluation
    _bath_dynamics.set_aux(dp, aux_snapshot)

    for i in range(n):
        x_perturbed = x.copy()
        x_perturbed[i] += eps
        # Restore aux before each perturbation
        _bath_dynamics.set_aux(dp, aux_snapshot)
        fx_perturbed = _f_state_transition(x_perturbed, dt_hr, model, dp)
        F[:, i] = (fx_perturbed - fx) / eps

    # Leave aux at the snapshot (the predict step will advance it properly
    # via the subsequent _f_state_transition call)
    _bath_dynamics.set_aux(dp, aux_snapshot)

    return F


def h_obs(
    x: np.ndarray,
    obs_tags: List[str],
    model: CellProcessModel,
    design_point: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Non-linear measurement model mapping state x to predicted observations."""
    if design_point is None:
        dp = _DEFAULT_DP
    else:
        dp = design_point

    j_val = max(1e-3, x[4])
    T_val = max(0.0, x[0])
    fe2_val = max(1e-6, x[2])
    pred = model.predict(j_mA_cm2=j_val, temperature_C=T_val, fe2_M=fe2_val)

    area = dp.get("electrode_area_m2", 1.0)

    preds = []
    for tag in obs_tags:
        if tag == "VT-201":
            # Physics-predicted cell voltage (basis for the bulk-Fe2+ coupling)
            preds.append(pred.v_cell_V)
        elif tag == "CT-201":
            # CT-201 measures total current in Amperes: j_mA_cm2 * area * 10
            preds.append(x[4] * area * 10.0)
        else:
            i = _ALL_OBS_MAP.get(tag)
            if i is None:
                preds.append(0.0)
            elif tag in ("THK-101", "FE2P-101"):
                # physically non-negative direct readings (thickness, Fe2+)
                preds.append(max(0.0, x[i]))
            else:
                # direct observation of the tagged state
                preds.append(x[i])
    return np.array(preds)


def H_jacobian(
    x: np.ndarray,
    obs_tags: List[str],
    model: CellProcessModel,
    design_point: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Numerical Jacobian of the measurement model h_obs."""
    n_obs = len(obs_tags)
    H = np.zeros((n_obs, N_STATES))
    eps = 1e-5
    h0 = h_obs(x, obs_tags, model, design_point)
    for i in range(N_STATES):
        x_perturbed = x.copy()
        x_perturbed[i] += eps
        h_perturbed = h_obs(x_perturbed, obs_tags, model, design_point)
        H[:, i] = (h_perturbed - h0) / eps
    return H


def _R_observation(tags: List[str]) -> np.ndarray:
    """Construct the measurement noise covariance matrix R."""
    n_obs = len(tags)
    R = np.zeros((n_obs, n_obs))
    for i, tag in enumerate(tags):
        spec = SENSOR_SPECS.get(tag)
        if spec is not None:
            R[i, i] = spec.noise_std ** 2
        else:
            R[i, i] = 1e-4
    return R


def _H_observation(
    tags: List[str],
    x: Optional[np.ndarray] = None,
    model: Optional[CellProcessModel] = None,
    design_point: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct observation matrix H and measurement noise matrix R.

    Provided for backwards compatibility and test interface conformity.
    """
    if x is None:
        x = np.array([60.0, 61.5, 1.0, 3.5, 150.0, 0.0, 2.5])
    if model is None:
        model = get_default_process_model()
    H = H_jacobian(x, tags, model, design_point)
    R = _R_observation(tags)
    return H, R


# ---------------------------------------------------------------------------
# EKF implementation
# ---------------------------------------------------------------------------

class ExtendedKalmanFilter:
    """EKF for the digital twin state estimation."""

    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        process_noise_std: Optional[np.ndarray] = None,
    ):
        self.x = x0.copy()
        self.P = P0.copy()
        # Process noise standard deviations for each state
        # Increased temperature noise (0.5°C) to account for thermal model uncertainties
        self.Q_diag = (
            process_noise_std ** 2
            if process_noise_std is not None
            else np.array([0.5, 0.5, 0.01, 0.05, 1.0, 0.1, 0.05])
        )

    def predict(
        self,
        dt_hr: float,
        model: CellProcessModel,
        design_point: Dict[str, float],
    ) -> None:
        F = _F_jacobian(self.x, dt_hr, model, design_point)
        self.x = _f_state_transition(self.x, dt_hr, model, design_point)
        self.P = F @ self.P @ F.T + np.diag(self.Q_diag)

    def update(
        self,
        z: np.ndarray,
        obs_tags: List[str],
        R: np.ndarray,
        model: CellProcessModel,
        design_point: Dict[str, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Kalman update. Returns (innovation, S)."""
        h_x = h_obs(self.x, obs_tags, model, design_point)
        y = z - h_x
        H = H_jacobian(self.x, obs_tags, model, design_point)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        identity = np.eye(N_STATES)
        self.P = (identity - K @ H) @ self.P
        return y, S


# ---------------------------------------------------------------------------
# Anomaly detector
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """Flags sensor readings inconsistent with model predictions."""

    def __init__(
        self,
        residual_threshold_sigma: float = 3.0,
        drift_window: int = 10,
        drift_threshold_sigma: float = 2.0,
        rate_limit_per_hr: Optional[Dict[str, float]] = None,
    ):
        self.residual_threshold = residual_threshold_sigma
        self.drift_window = drift_window
        self.drift_threshold = drift_threshold_sigma
        self.rate_limit = rate_limit_per_hr or {
            tag: (s.physical_max - s.physical_min) / 6.0  # full range in 6 min = fault
            for tag, s in SENSOR_SPECS.items()
        }
        # History per sensor: list of (time, value)
        self._history: Dict[str, List[Tuple[float, float]]] = {tag: [] for tag in SENSOR_SPECS}

    def check(  # noqa: C901
        self,
        readings: Dict[str, float],
        predicted: Dict[str, float],
        predicted_sigma: Dict[str, float],
        t_hr: float,
        innovation: np.ndarray,
        innovation_cov: np.ndarray,
        obs_tags: List[str],
    ) -> List[Anomaly]:
        anomalies: List[Anomaly] = []

        # 1. Residual check (using innovation)
        for i, tag in enumerate(obs_tags):
            if i >= len(innovation):
                break
            s = SENSOR_SPECS.get(tag)
            if s is None:
                continue
            sigma = math.sqrt(max(innovation_cov[i, i], 1e-12))
            residual_sigma = abs(float(innovation[i])) / sigma
            if residual_sigma > self.residual_threshold:
                anomalies.append(Anomaly(
                    sensor_tag=tag,
                    timestamp_hr=t_hr,
                    kind="residual",
                    severity=residual_sigma,
                    detail=f"Residual {residual_sigma:.1f}sigma exceeds {self.residual_threshold}sigma",
                ))

        # 2. Drift detection (running mean shift)
        for tag, val in readings.items():
            if tag not in self._history:
                continue
            hist = self._history[tag]
            hist.append((t_hr, val))
            # Keep last N
            if len(hist) > self.drift_window * 3:
                self._history[tag] = hist[-self.drift_window * 3:]
                hist = self._history[tag]

            if len(hist) >= self.drift_window:
                recent = [h[1] for h in hist[-self.drift_window:]]
                older = [h[1] for h in hist[-2 * self.drift_window:-self.drift_window]] if len(hist) >= 2 * self.drift_window else None
                if older is not None:
                    s = SENSOR_SPECS.get(tag)
                    if s:
                        sigma = s.noise_std
                        mean_shift = abs(np.mean(recent) - np.mean(older))
                        drift_sigma = mean_shift / max(sigma * math.sqrt(2.0 / self.drift_window), 1e-12)
                        if drift_sigma > self.drift_threshold:
                            anomalies.append(Anomaly(
                                sensor_tag=tag,
                                timestamp_hr=t_hr,
                                kind="drift",
                                severity=drift_sigma,
                                detail=f"Persistent drift {drift_sigma:.1f}sigma over {self.drift_window} samples",
                            ))

        # 3. Rate-of-change check
        for tag, val in readings.items():
            if tag not in self._history:
                continue
            hist = self._history[tag]
            if len(hist) >= 2:
                t_prev, v_prev = hist[-2]
                dt = t_hr - t_prev
                if dt > 0:
                    rate = abs(val - v_prev) / dt
                    limit = self.rate_limit.get(tag, float("inf"))
                    if rate > limit:
                        anomalies.append(Anomaly(
                            sensor_tag=tag,
                            timestamp_hr=t_hr,
                            kind="rate_of_change",
                            severity=rate / limit,
                            detail=f"Rate {rate:.2f}/hr exceeds limit {limit:.2f}/hr",
                        ))

        return anomalies


# ---------------------------------------------------------------------------
# Digital Twin
# ---------------------------------------------------------------------------

class DigitalTwin:
    """Main digital twin class: EKF state estimation + anomaly detection +
    prediction + confidence tracking."""

    def __init__(
        self,
        design_point: Optional[Dict[str, float]] = None,
        model: Optional[Any] = None,
        seed: int = 42,
        env_state: Optional[Dict[str, Any]] = None,
        crate_state: Optional[Dict[str, Any]] = None,
    ):
        self.design_point = design_point or {
            "temperature_C": 60.0,
            "pH": 3.5,
            "cell_voltage_V": 2.5,
            "j_avg_mA_cm2": 150.0,
            "electrode_area_m2": 1.0,
            "electrolyte_volume_L": 1000.0,
            "fe2_M": 1.0,
        }

        # Seed operating point from physics if model is supplied but design_point is not
        if model is not None and not design_point:
            nom = getattr(model, "nominal", None)
            n = None
            if callable(nom):
                n = nom()
            elif isinstance(nom, dict):
                n = nom
            if n:
                self.design_point = {
                    "temperature_C": n.get("temperature_C", 60.0),
                    "pH": 3.5,
                    "cell_voltage_V": n.get("cell_voltage_V", 2.5),
                    "j_avg_mA_cm2": n.get("j_avg_mA_cm2", 150.0),
                    "electrode_area_m2": 1.0,
                    "electrolyte_volume_L": 1000.0,
                    "fe2_M": n.get("fe2_M", 1.0),
                }

        self.model = model if model is not None else get_default_process_model()
        self.rng = np.random.default_rng(seed)

        # Initialize the auxiliary reservoir state if not already present
        if "_bath_aux" not in self.design_point:
            T_nom = self.design_point.get("temperature_C", 60.0)
            _bath_dynamics.set_aux(self.design_point, _bath_dynamics.BathAux(
                T_reservoir_C=self.design_point.get("T_reservoir_C", T_nom),
                fe2_reservoir_M=self.design_point.get(
                    "fe2_reservoir_M",
                    self.design_point.get("fe2_M", 1.0)),
                pH_reservoir=self.design_point.get("pH_reservoir",
                    self.design_point.get("pH", 3.5)),
            ))

        # Initial state
        # With coupled bath dynamics, the anolyte temperature emerges from the
        # energy balance rather than being a fixed offset.  Initialize it near
        # the catholyte temperature so the system starts close to steady state.
        T_nom = self.design_point.get("temperature_C", 60.0)
        j_nom = self.design_point.get("j_avg_mA_cm2", 150.0)
        fe2_nom = self.design_point.get("fe2_M", 1.0)
        V_init = self.design_point.get("cell_voltage_V", 2.5)

        x0 = np.array([
            T_nom,                      # catholyte_temp
            T_nom,                      # anolyte_temp (emerges from energy balance)
            fe2_nom,                    # bulk_fe2
            self.design_point.get("pH", 3.5),                    # bulk_pH
            j_nom,                      # current_density
            0.0,                                                 # deposit_thickness
            V_init,                     # cell_voltage (from design_point)
        ], dtype=float)

        P0 = np.diag([1.0, 1.0, 0.01, 0.1, 10.0, 1.0, 0.05])

        self.ekf = ExtendedKalmanFilter(x0, P0)
        self.anomaly_detector = AnomalyDetector()

        self._t_hr: float = 0.0
        self._history: List[TwinState] = []
        self._anomaly_log: List[Anomaly] = []

    def set_environment(
        self,
        env_state: Optional[Dict[str, Any]] = None,
        crate_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set the environmental-disturbance coupling from env/crate observations.

        ``disturbance_from_environment`` is a pure, deterministic mapping.  With
        no observations it returns a disabled / zero :class:`DisturbanceInputs`,
        so switching coupling on only changes the disturbance terms the bath
        dynamics apply (the uncoupled twin is unchanged).  Call again to update
        the disturbance as the environment / site snapshot changes.
        """
        self.design_point["_env_dist"] = disturbance_from_environment(
            env_state, crate_state
        )

    def update(self, sensor_readings: Dict[str, float], dt_hr: float = 0.1) -> TwinState:
        """Incorporate new sensor readings into the twin state.

        Parameters
        ----------
        sensor_readings : dict
            Mapping of sensor tag -> reading value.
        dt_hr : float
            Time step in hours since last update.

        Returns
        -------
        TwinState
        """
        self._t_hr += dt_hr

        # Predict state forward
        self.ekf.predict(dt_hr, self.model, self.design_point)

        # Build observation from available tags (keeping the order defined in
        # _ALL_OBS_MAP, i.e. base suite then optional L1 sensors).  Only tags
        # actually present in the readings are observed — the L1 sensors engage
        # solely when supplied, leaving the base 5-sensor twin unchanged.
        obs_tags = [t for t in _ALL_OBS_MAP if t in sensor_readings]
        if obs_tags:
            R = _R_observation(obs_tags)
            z = np.array([sensor_readings[t] for t in obs_tags])
            innovation, S = self.ekf.update(z, obs_tags, R, self.model, self.design_point)
        else:
            innovation = np.zeros(len(OBSERVABLE_TAGS))
            S = np.eye(len(OBSERVABLE_TAGS))

        # Predicted values for anomaly detection
        h_x = h_obs(self.ekf.x, obs_tags, self.model, self.design_point)
        predicted = {
            tag: float(h_x[i]) for i, tag in enumerate(obs_tags)
        }

        # Calculate predicted standard deviations of the measurements (diag of H P H.T)
        H = H_jacobian(self.ekf.x, obs_tags, self.model, self.design_point)
        meas_cov = H @ self.ekf.P @ H.T
        predicted_sigma = {
            tag: float(math.sqrt(max(meas_cov[i, i], 0.0)))
            for i, tag in enumerate(obs_tags)
        }

        anomalies = self.anomaly_detector.check(
            sensor_readings, predicted, predicted_sigma,
            self._t_hr, innovation, S, obs_tags,
        )
        self._anomaly_log.extend(anomalies)

        state = TwinState(
            timestamp_hr=self._t_hr,
            state_mean=self.ekf.x.copy(),
            state_covariance=self.ekf.P.copy(),
            innovation=innovation,
            innovation_cov=S,
            anomalies=anomalies,
        )
        self._history.append(state)
        return state

    def physics_predict(self, j_mA_cm2: float, temperature_C: float, fe2_M: float):
        """Physics-predicted observables at an operating point via the process model."""
        if self.model is None:
            raise ValueError(
                "DigitalTwin has no physics process model. "
                "Construct it with DigitalTwin(model=CellProcessModel(...))."
            )
        return self.model.predict(j_mA_cm2, temperature_C, fe2_M)

    def predict_ahead(self, horizon_hr: float, n_steps: int = 50) -> PredictionEnvelope:
        """Forward-predict state and uncertainty.

        Parameters
        ----------
        horizon_hr : float
            Prediction horizon in hours.
        n_steps : int
            Number of prediction steps.

        Returns
        -------
        PredictionEnvelope
        """
        dt = horizon_hr / n_steps
        ts = np.linspace(self._t_hr, self._t_hr + horizon_hr, n_steps)
        means = np.zeros((n_steps, N_STATES))
        sigmas = np.zeros((n_steps, N_STATES))

        x = self.ekf.x.copy()
        P = self.ekf.P.copy()
        Q_diag = self.ekf.Q_diag

        for i in range(n_steps):
            F = _F_jacobian(x, dt, self.model, self.design_point)
            x = _f_state_transition(x, dt, self.model, self.design_point)
            P = F @ P @ F.T + np.diag(Q_diag)
            means[i] = x
            sigmas[i] = np.sqrt(np.maximum(np.diag(P), 0.0))

        # Confidence: probability that states stay within physical tolerances
        # around the CURRENT state
        tolerances = np.array([5.0, 5.0, 0.2, 0.5, 20.0, 50.0, 0.2])
        reference = self.ekf.x.copy()
        confidence = np.ones(n_steps)
        for i in range(n_steps):
            for j in range(N_STATES):
                p_in = _normal_cdf(reference[j] + tolerances[j], means[i, j], sigmas[i, j])
                p_below = _normal_cdf(reference[j] - tolerances[j], means[i, j], sigmas[i, j])
                p_in_band = p_in - p_below
                confidence[i] *= p_in_band

        return PredictionEnvelope(
            horizon_hr=horizon_hr,
            timestamps_hr=ts,
            mean_trajectories=means,
            sigma_trajectories=sigmas,
            confidence=confidence,
        )

    def anomaly_report(self) -> List[Anomaly]:
        """Return all logged anomalies."""
        return list(self._anomaly_log)

    def confidence_trajectory(self) -> List[Tuple[float, float]]:
        """Return (timestamp, P_all_specs_met) for each update step."""
        result = []
        for state in self._history:
            # Confidence based on covariance trace
            trace = np.trace(state.state_covariance)
            conf = float(np.exp(-trace / 100.0))
            result.append((state.timestamp_hr, conf))
        return result

    @property
    def history(self) -> List[TwinState]:
        return list(self._history)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    """Standard normal CDF without scipy."""
    z = (x - mu) / max(sigma, 1e-12)
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
