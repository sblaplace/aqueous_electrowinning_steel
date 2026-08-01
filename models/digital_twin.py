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
}

# Map sensor tags to the state vector index
STATE_KEYS: List[str] = [
    "catholyte_temperature",  # 0
    "anolyte_temperature",    # 1
    "catholyte_pH",           # 2
    "dissolved_O2",           # 3
    "cell_voltage",           # 4
    "current_density",        # 5  (mA/cm² derived from current / area)
    "carbon_activity",        # 6  (a_C, dimensionless 0-1)
    "ni_concentration",       # 7  (M)
]

STATE_INDEX: Dict[str, int] = {k: i for i, k in enumerate(STATE_KEYS)}
N_STATES = len(STATE_KEYS)

# Observation matrix mapping: which sensor observes which state
_OBS_MAP: Dict[str, int] = {
    "TT-101": 0,
    "TT-201": 1,
    "pHAT-101": 2,
    "AT-201": 3,
    "VT-201": 4,
    "CT-201": 5,
}

# Sensor tags that can be used as observations
OBSERVABLE_TAGS: List[str] = list(_OBS_MAP.keys())


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
    readings["VT-201"] = dp.get("cell_voltage_V", 2.5) + 0.05 * math.sin(2 * math.pi * t_hr / 4.0) + rng.normal(0, 0.01)
    readings["CT-201"] = dp.get("j_avg_mA_cm2", 150.0) * dp.get("electrode_area_m2", 1.0) * 10.0 + rng.normal(0, 5.0)

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
# EKF implementation
# ---------------------------------------------------------------------------

def _f_state_transition(x: np.ndarray, dt_hr: float) -> np.ndarray:
    """Deterministic state transition (constant + slow mean-reversion)."""
    # Mean-revert toward nominal values
    nominal = np.array([60.0, 61.5, 3.5, 7.5, 2.5, 150.0, 0.8, 0.5])
    tau_hr = np.array([2.0, 2.0, 4.0, 3.0, 1.0, 0.5, 6.0, 8.0])  # time constants
    alpha = 1.0 - np.exp(-dt_hr / tau_hr)
    return x + alpha * (nominal - x)


def _F_jacobian(x: np.ndarray, dt_hr: float) -> np.ndarray:
    """Jacobian of state transition."""
    tau_hr = np.array([2.0, 2.0, 4.0, 3.0, 1.0, 0.5, 6.0, 8.0])
    alpha = 1.0 - np.exp(-dt_hr / tau_hr)
    F = np.eye(N_STATES)
    for i in range(N_STATES):
        F[i, i] = 1.0 - alpha[i]
    return F


def _H_observation(tags: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Build observation matrix H, noise R, and observation vector z from tags.

    Returns (H, R).
    """
    n_obs = len(tags)
    H = np.zeros((n_obs, N_STATES))
    R = np.zeros((n_obs, n_obs))
    for i, tag in enumerate(tags):
        if tag in _OBS_MAP:
            H[i, _OBS_MAP[tag]] = 1.0
            R[i, i] = SENSOR_SPECS[tag].noise_std ** 2
    return H, R


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
        self.Q_diag = (
            process_noise_std ** 2
            if process_noise_std is not None
            else np.array([0.1, 0.1, 0.01, 0.05, 0.005, 1.0, 0.005, 0.005])
        )

    def predict(self, dt_hr: float) -> None:
        F = _F_jacobian(self.x, dt_hr)
        self.x = _f_state_transition(self.x, dt_hr)
        self.P = F @ self.P @ F.T + np.diag(self.Q_diag)

    def update(
        self,
        z: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Kalman update. Returns (innovation, S)."""
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(N_STATES)
        I_KH = I - K @ H
        self.P = I_KH @ self.P  # Joseph form omitted for speed; acceptable for well-conditioned S
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

    def check(
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
    ):
        self.design_point = design_point or {
            "temperature_C": 60.0,
            "pH": 3.5,
            "cell_voltage_V": 2.5,
            "j_avg_mA_cm2": 150.0,
            "electrode_area_m2": 1.0,
        }
        self.model = model
        # If a physics process model is supplied, seed the operating point from
        # physics so the EKF starts at a self-consistent reference condition.
        if model is not None and not design_point:
            n = None
            nom = getattr(model, "nominal", None)
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
                }
        self.rng = np.random.default_rng(seed)

        # Initial state
        x0 = np.array([
            self.design_point.get("temperature_C", 60.0),   # catholyte_temp
            self.design_point.get("temperature_C", 60.0) + 1.5,  # anolyte_temp
            self.design_point.get("pH", 3.5),                # pH
            7.5,                                               # dissolved_O2
            self.design_point.get("cell_voltage_V", 2.5),    # cell_voltage
            self.design_point.get("j_avg_mA_cm2", 150.0),   # current_density
            0.8,                                               # carbon_activity
            0.5,                                               # ni_concentration
        ], dtype=float)

        P0 = np.diag([1.0, 1.0, 0.1, 0.5, 0.05, 10.0, 0.05, 0.05])

        self.ekf = ExtendedKalmanFilter(x0, P0)
        self.anomaly_detector = AnomalyDetector()

        self._t_hr: float = 0.0
        self._history: List[TwinState] = []
        self._anomaly_log: List[Anomaly] = []

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

        # Predict
        self.ekf.predict(dt_hr)

        # Build observation from available tags
        obs_tags = [t for t in sensor_readings if t in _OBS_MAP]
        if obs_tags:
            H, R = _H_observation(obs_tags)
            z = np.array([sensor_readings[t] for t in obs_tags])
            innovation, S = self.ekf.update(z, H, R)
        else:
            innovation = np.zeros(len(OBSERVABLE_TAGS))
            S = np.eye(len(OBSERVABLE_TAGS))

        # Predicted values for anomaly detection
        predicted = {
            tag: float(self.ekf.x[_OBS_MAP[tag]]) for tag in obs_tags if tag in _OBS_MAP
        }
        predicted_sigma = {
            tag: float(math.sqrt(max(self.ekf.P[_OBS_MAP[tag], _OBS_MAP[tag]], 0)))
            for tag in obs_tags if tag in _OBS_MAP
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
        """Physics-predicted observables at an operating point via the process model.

        Requires the twin to have been constructed with a ``CellProcessModel``
        (the ``model`` argument).  Raises if no physics model is attached.
        """
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
            F = _F_jacobian(x, dt)
            x = _f_state_transition(x, dt)
            P = F @ P @ F.T + np.diag(Q_diag)
            means[i] = x
            sigmas[i] = np.sqrt(np.maximum(np.diag(P), 0.0))

        # Confidence: probability that states stay within physical tolerances
        # around the CURRENT state (not a fixed nominal)
        tolerances = np.array([5.0, 5.0, 0.5, 2.0, 0.2, 20.0, 0.2, 0.2])
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
            # Simplified: confidence based on covariance trace
            trace = np.trace(state.state_covariance)
            # Normalize: small trace = high confidence
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
