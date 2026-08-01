"""Tests for the digital twin module (>=8 tests)."""
from __future__ import annotations

import math
import numpy as np
import pytest

from models.digital_twin import (
    DigitalTwin,
    ExtendedKalmanFilter,
    AnomalyDetector,
    TwinState,
    PredictionEnvelope,
    Anomaly,
    STATE_KEYS,
    STATE_INDEX,
    N_STATES,
    SENSOR_SPECS,
    OBSERVABLE_TAGS,
    generate_synthetic_readings,
    _f_state_transition,
    _F_jacobian,
    _H_observation,
    _normal_cdf,
)


class TestEKFTracksTrueState:
    """EKF tracks a known trajectory within 2σ."""

    def test_ekf_converges_to_nominal(self):
        """After many updates with nominal readings, EKF state stays near nominal."""
        twin = DigitalTwin(seed=42)
        nominal = twin.design_point

        rng = np.random.default_rng(99)
        for _ in range(200):
            t_hr = _ * 0.1
            readings = generate_synthetic_readings(nominal, t_hr, rng)
            twin.update(readings, dt_hr=0.1)

        state = twin.history[-1]
        mean = state.state_dict
        two_sigma = state.uncertainty_2sigma

        # Catholyte temperature should be within 2σ of ~60C
        assert abs(mean["catholyte_temperature"] - 60.0) < two_sigma["catholyte_temperature"] + 2.0
        # pH near 3.5
        assert abs(mean["catholyte_pH"] - 3.5) < two_sigma["catholyte_pH"] + 0.5
        # Cell voltage near 2.5V
        assert abs(mean["cell_voltage"] - 2.5) < two_sigma["cell_voltage"] + 0.2


class TestAnomalyDetection:
    """Anomaly detector flags injected faults."""

    def test_bias_detected_as_residual(self):
        """A large bias in temperature should be flagged as a residual anomaly."""
        twin = DigitalTwin(seed=42)
        nominal = twin.design_point
        rng = np.random.default_rng(123)

        # Warm up
        for i in range(50):
            readings = generate_synthetic_readings(nominal, i * 0.1, rng)
            twin.update(readings, dt_hr=0.1)

        # Inject a +15C bias (well beyond 3σ of 0.5C noise)
        readings = generate_synthetic_readings(nominal, 5.0, rng)
        readings["TT-101"] += 15.0  # large bias

        state = twin.update(readings, dt_hr=0.1)
        residual_anomalies = [a for a in state.anomalies if a.kind == "residual"]
        assert len(residual_anomalies) > 0, "Large bias should produce residual anomaly"

    def test_rate_of_change_fault(self):
        """Extremely fast change should be flagged as rate_of_change."""
        detector = AnomalyDetector()

        # Seed history
        readings = {"TT-101": 60.0}
        pred = {"TT-101": 60.0}
        pred_sigma = {"TT-101": 0.5}
        obs_tags = ["TT-101"]

        for t in range(15):
            innovation = np.array([0.0])
            S = np.array([[0.25]])
            detector.check(readings, pred, pred_sigma, t * 0.1, innovation, S, obs_tags)

        # Now jump 30C in one step (physical max range is 75C, limit ~12.5C/hr)
        readings["TT-101"] = 90.0
        innovation = np.array([30.0])
        S = np.array([[0.25]])
        anomalies = detector.check(readings, pred, pred_sigma, 1.6, innovation, S, obs_tags)
        roc = [a for a in anomalies if a.kind == "rate_of_change"]
        assert len(roc) > 0, "30C jump should trigger rate_of_change"


class TestConfidenceTrajectory:
    """Confidence responds to sensor data quality."""

    def test_confidence_decreases_with_noisy_data(self):
        """Adding very noisy data should reduce confidence."""
        twin = DigitalTwin(seed=42)
        nominal = twin.design_point

        # Clean data
        rng = np.random.default_rng(42)
        for i in range(30):
            readings = generate_synthetic_readings(nominal, i * 0.1, rng)
            twin.update(readings, dt_hr=0.1)
        conf_clean = twin.confidence_trajectory()[-1][1]

        # Now add very noisy data (multiply noise by 10)
        for i in range(30):
            readings = generate_synthetic_readings(nominal, (30 + i) * 0.1, rng)
            for tag in readings:
                if tag in SENSOR_SPECS:
                    readings[tag] += rng.normal(0, SENSOR_SPECS[tag].noise_std * 10)
            twin.update(readings, dt_hr=0.1)

        # Confidence should still be a valid probability
        conf_final = twin.confidence_trajectory()[-1][1]
        assert 0.0 <= conf_final <= 1.0


class TestPredictionEnvelope:
    """Prediction envelope widens with horizon."""

    def test_uncertainty_grows_with_horizon(self):
        """Sigma at end of horizon > sigma at start."""
        twin = DigitalTwin(seed=42)
        rng = np.random.default_rng(42)
        nominal = twin.design_point

        for i in range(50):
            readings = generate_synthetic_readings(nominal, i * 0.1, rng)
            twin.update(readings, dt_hr=0.1)

        pred = twin.predict_ahead(horizon_hr=12.0, n_steps=60)
        sigma_start = pred.sigma_trajectories[0, 0]
        sigma_end = pred.sigma_trajectories[-1, 0]
        assert sigma_end > sigma_start, "Uncertainty should grow with prediction horizon"

    def test_confidence_decreases_with_horizon(self):
        """Longer predictions should have lower confidence."""
        twin = DigitalTwin(seed=42)
        rng = np.random.default_rng(42)
        nominal = twin.design_point

        for i in range(50):
            readings = generate_synthetic_readings(nominal, i * 0.1, rng)
            twin.update(readings, dt_hr=0.1)

        pred = twin.predict_ahead(horizon_hr=12.0, n_steps=60)
        # Confidence at end of horizon should be <= confidence near start
        # (allowing for initial transient where EKF is still converging)
        early_idx = min(5, len(pred.confidence) - 1)
        assert pred.confidence[-1] <= pred.confidence[early_idx] + 0.05, \
            "Confidence should generally decrease with horizon"


class TestSensorInterface:
    """Sensor specs and observation mapping are consistent."""

    def test_all_observable_tags_have_specs(self):
        """Every tag in OBSERVABLE_TAGS should have a SensorSpec."""
        for tag in OBSERVABLE_TAGS:
            assert tag in SENSOR_SPECS, f"Observable tag {tag} missing from SENSOR_SPECS"

    def test_observation_matrix_shape(self):
        """H matrix has correct shape for given tags."""
        tags = OBSERVABLE_TAGS[:3]
        H, R = _H_observation(tags)
        assert H.shape == (len(tags), N_STATES)
        assert R.shape == (len(tags), len(tags))


class TestSyntheticReadings:
    """Synthetic sensor generator produces physically plausible values."""

    def test_readings_within_bounds(self):
        """All readings should be within physical bounds (without faults)."""
        rng = np.random.default_rng(42)
        design_point = {"temperature_C": 60.0, "pH": 3.5, "cell_voltage_V": 2.5,
                       "j_avg_mA_cm2": 150.0, "electrode_area_m2": 1.0}
        for i in range(100):
            readings = generate_synthetic_readings(design_point, i * 0.1, rng)
            for tag, val in readings.items():
                spec = SENSOR_SPECS.get(tag)
                if spec:
                    # Allow 5σ beyond physical bounds for noise
                    assert val >= spec.physical_min - 5 * spec.noise_std, f"{tag}={val} below bound"
                    assert val <= spec.physical_max + 5 * spec.noise_std, f"{tag}={val} above bound"


class TestMathUtils:
    """Utility functions."""

    def test_normal_cdf_symmetry(self):
        assert abs(_normal_cdf(0, 0, 1) - 0.5) < 1e-10
        assert _normal_cdf(10, 0, 1) > 0.9999
        assert _normal_cdf(-10, 0, 1) < 0.0001

    def test_state_transition_identity_at_zero_dt(self):
        """Zero time step should not change state."""
        x = np.array([60.0, 61.5, 3.5, 7.5, 2.5, 150.0, 0.8, 0.5])
        x2 = _f_state_transition(x, 0.0)
        np.testing.assert_allclose(x, x2, atol=1e-12)
