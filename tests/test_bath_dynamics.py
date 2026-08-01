"""Tests for the coupled bath/recirculation dynamics module.

These tests verify that the conservation-law dynamics correctly implement:
1. Fe2+ mass balance (consumption + makeup + recirculation)
2. Thermal energy balance (Joule heating + cooling + membrane + ambient + recirculation)
3. pH buffer dynamics (acid/base dose + HER hydroxide production + recirculation)
4. Physical bounds (pH in [0,14], fe2 >= 0, deposit >= 0)
"""

from __future__ import annotations

import numpy as np

from models import bath_dynamics
from models.bath_dynamics import BathAux, step
from models.twin_physics import CellProcessModel


def _make_model_and_dp():
    """Create a process model and design_point for testing."""
    model = CellProcessModel()
    dp = {
        "temperature_C": 60.0,
        "pH": 3.5,
        "cell_voltage_V": 5.0,
        "j_avg_mA_cm2": 150.0,
        "electrode_area_m2": 1.0,
        "electrolyte_volume_L": 1000.0,
        "fe2_M": 1.0,
        # Bath dynamics parameters
        "recirculation_flow_L_hr": 6000.0,
        "reservoir_volume_L": 50000.0,
        "catholyte_volume_L": 800.0,
        "anolyte_volume_L": 2000.0,
        "fe2_makeup_rate_M_hr": 0.0,
        "buffer_capacity_beta": 0.05,
        "acid_dose_rate_M_hr": 0.0,
    }
    return model, dp


class TestFe2MassBalance:
    """Fe2+ mass balance: consumption + makeup + recirculation."""

    def test_fe2_depletes_without_makeup(self):
        """Without makeup, fe2 should deplete due to Faraday consumption."""
        model, dp = _make_model_and_dp()
        dp["fe2_makeup_rate_M_hr"] = 0.0  # no makeup

        x = np.array([60.0, 60.0, 1.0, 3.5, 150.0, 0.0, 5.0])
        aux = BathAux(T_reservoir_C=60.0, fe2_reservoir_M=1.0, pH_reservoir=3.5)

        # Run for 1 hour
        for _ in range(10):
            x, aux = step(x, aux, 0.1, dp, model)

        # Fe2+ should have depleted
        assert x[2] < 1.0, f"fe2 should deplete without makeup, got {x[2]}"

    def test_fe2_steady_with_makeup(self):
        """With makeup balancing consumption, fe2 should reach steady state."""
        model, dp = _make_model_and_dp()

        # Compute steady-state makeup rate
        # Set makeup to balance consumption at nominal operating point
        pred = model.predict(j_mA_cm2=150.0, temperature_C=60.0, fe2_M=1.0)
        j_A_m2 = 150.0 * 10.0
        consumption = (j_A_m2 * pred.current_efficiency / (2.0 * 96485.3321)) * 1.0 * 3600.0 / 800.0
        dp["fe2_makeup_rate_M_hr"] = consumption

        x = np.array([60.0, 60.0, 1.0, 3.5, 150.0, 0.0, 5.0])
        aux = BathAux(T_reservoir_C=60.0, fe2_reservoir_M=1.0, pH_reservoir=3.5)

        # Run for 10 hours
        for _ in range(100):
            x, aux = step(x, aux, 0.1, dp, model)

        # Fe2+ should be near steady state (within 0.2 M of initial)
        assert abs(x[2] - 1.0) < 0.2, f"fe2 should be near steady state, got {x[2]}"

    def test_fe2_recirculation_exchange(self):
        """Recirculation should exchange fe2 between cell and reservoir."""
        model, dp = _make_model_and_dp()
        dp["fe2_makeup_rate_M_hr"] = 0.0

        # Start with different fe2 in cell and reservoir
        x = np.array([60.0, 60.0, 0.5, 3.5, 150.0, 0.0, 5.0])  # cell fe2 = 0.5
        aux = BathAux(T_reservoir_C=60.0, fe2_reservoir_M=1.5, pH_reservoir=3.5)  # res fe2 = 1.5

        # Run for 1 hour
        for _ in range(10):
            x, aux = step(x, aux, 0.1, dp, model)

        # Cell fe2 should increase (toward reservoir), reservoir fe2 should decrease
        assert x[2] > 0.5, f"cell fe2 should increase from recirculation, got {x[2]}"


class TestThermalBalance:
    """Thermal energy balance: Joule heating + cooling + membrane + ambient + recirculation."""

    def test_temperature_stable_with_auto_cooling(self):
        """With auto-balance cooling, temperature should stay near initial."""
        model, dp = _make_model_and_dp()
        # Don't set cooling_power_W — use auto-balance

        x = np.array([60.0, 60.0, 1.0, 3.5, 150.0, 0.0, 5.0])
        aux = BathAux(T_reservoir_C=60.0, fe2_reservoir_M=1.0, pH_reservoir=3.5)

        # Run for 5 hours
        for _ in range(50):
            x, aux = step(x, aux, 0.1, dp, model)

        # Temperature should stay within 5°C of initial
        assert abs(x[0] - 60.0) < 5.0, f"catholyte temp should be stable, got {x[0]}"

    def test_anolyte_tracks_catholyte(self):
        """Anolyte temperature should track catholyte via coupling."""
        model, dp = _make_model_and_dp()

        # Start with different temperatures
        x = np.array([60.0, 70.0, 1.0, 3.5, 150.0, 0.0, 5.0])  # anolyte 10°C hotter
        aux = BathAux(T_reservoir_C=60.0, fe2_reservoir_M=1.0, pH_reservoir=3.5)

        # Run for 5 hours
        for _ in range(50):
            x, aux = step(x, aux, 0.1, dp, model)

        # Anolyte should have cooled toward catholyte (within 5°C)
        assert abs(x[1] - x[0]) < 5.0, f"anolyte should track catholyte, got T_anol={x[1]}, T_cath={x[0]}"


class TestPHBufferDynamics:
    """pH buffer dynamics: acid/base dose + HER hydroxide + recirculation."""

    def test_ph_stays_in_bounds(self):
        """pH should stay in [0, 14] under all conditions."""
        model, dp = _make_model_and_dp()

        x = np.array([60.0, 60.0, 1.0, 3.5, 150.0, 0.0, 5.0])
        aux = BathAux(T_reservoir_C=60.0, fe2_reservoir_M=1.0, pH_reservoir=3.5)

        # Run for 10 hours
        for _ in range(100):
            x, aux = step(x, aux, 0.1, dp, model)

        # pH should be in valid range
        assert 0.0 <= x[3] <= 14.0, f"pH out of bounds: {x[3]}"

    def test_acid_dose_lowers_ph(self):
        """Acid dose should lower pH."""
        model, dp = _make_model_and_dp()
        dp["acid_dose_rate_M_hr"] = 0.1  # add acid

        x = np.array([60.0, 60.0, 1.0, 3.5, 150.0, 0.0, 5.0])
        aux = BathAux(T_reservoir_C=60.0, fe2_reservoir_M=1.0, pH_reservoir=3.5)

        # Run for 1 hour
        for _ in range(10):
            x, aux = step(x, aux, 0.1, dp, model)

        # pH should have decreased
        assert x[3] < 3.5, f"acid dose should lower pH, got {x[3]}"


class TestPhysicalBounds:
    """Physical bounds and invariants."""

    def test_fe2_nonnegative(self):
        """Fe2+ concentration should never go negative."""
        model, dp = _make_model_and_dp()
        dp["fe2_makeup_rate_M_hr"] = 0.0

        # Start with low fe2 and run for a long time
        x = np.array([60.0, 60.0, 0.01, 3.5, 150.0, 0.0, 5.0])
        aux = BathAux(T_reservoir_C=60.0, fe2_reservoir_M=0.01, pH_reservoir=3.5)

        for _ in range(100):
            x, aux = step(x, aux, 0.1, dp, model)

        assert x[2] >= 0.0, f"fe2 should be nonnegative, got {x[2]}"

    def test_deposit_nonnegative(self):
        """Deposit thickness should never go negative."""
        model, dp = _make_model_and_dp()

        x = np.array([60.0, 60.0, 1.0, 3.5, 150.0, 0.0, 5.0])
        aux = BathAux(T_reservoir_C=60.0, fe2_reservoir_M=1.0, pH_reservoir=3.5)

        for _ in range(100):
            x, aux = step(x, aux, 0.1, dp, model)

        assert x[5] >= 0.0, f"deposit should be nonnegative, got {x[5]}"


class TestSteadyStateHelpers:
    """Test the steady-state computation helpers."""

    def test_steady_state_fe2(self):
        """Steady-state fe2 helper should return a reasonable value."""
        model, dp = _make_model_and_dp()

        fe2_ss = bath_dynamics.steady_state_fe2_M(dp, model)

        # Should be positive and reasonable
        assert 0.0 < fe2_ss < 10.0, f"steady-state fe2 unreasonable: {fe2_ss}"

    def test_steady_state_acid_dose(self):
        """Steady-state acid dose helper should return a reasonable value."""
        model, dp = _make_model_and_dp()

        acid_dose = bath_dynamics.steady_state_acid_dose_M_hr(dp, model)

        # Should be finite
        assert np.isfinite(acid_dose), f"steady-state acid dose not finite: {acid_dose}"


class TestInterfaceCompatibility:
    """Test that the bath dynamics interface is compatible with the EKF."""

    def test_step_returns_correct_shapes(self):
        """step() should return (7,) array and BathAux."""
        model, dp = _make_model_and_dp()

        x = np.array([60.0, 60.0, 1.0, 3.5, 150.0, 0.0, 5.0])
        aux = BathAux(T_reservoir_C=60.0, fe2_reservoir_M=1.0, pH_reservoir=3.5)

        x_next, aux_next = step(x, aux, 0.1, dp, model)

        assert x_next.shape == (7,), f"x_next shape wrong: {x_next.shape}"
        assert isinstance(aux_next, BathAux), f"aux_next type wrong: {type(aux_next)}"

    def test_zero_dt_identity(self):
        """Zero time step should not change state."""
        model, dp = _make_model_and_dp()

        x = np.array([60.0, 60.0, 1.0, 3.5, 150.0, 0.0, 5.0])
        aux = BathAux(T_reservoir_C=60.0, fe2_reservoir_M=1.0, pH_reservoir=3.5)

        x_next, aux_next = step(x, aux, 0.0, dp, model)

        np.testing.assert_allclose(x, x_next, atol=1e-12)
        assert aux.T_reservoir_C == aux_next.T_reservoir_C
        assert aux.fe2_reservoir_M == aux_next.fe2_reservoir_M
        assert aux.pH_reservoir == aux_next.pH_reservoir
