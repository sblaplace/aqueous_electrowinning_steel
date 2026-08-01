"""Tests for the cell<->crate environmental-disturbance coupling (L0).

Load-bearing guarantees these lock in:

* Coupling is OFF by default: an absent or disabled disturbance changes
  nothing — the EKF step is byte-identical to the uncoupled case.
* Physical direction: higher wind => stronger convective cooling => colder
  cell; ingress dilution lowers bulk Fe2+ and drags pH toward neutral.
* The disturbance is a pure, deterministic mapping from env/crate
  observations, wired through DigitalTwin.set_environment().
"""

from __future__ import annotations

import numpy as np

from models.bath_dynamics import BathAux, step
from models.digital_twin import DigitalTwin
from models.env_coupling import DisturbanceInputs, disturbance_from_environment
from models.twin_physics import CellProcessModel

X0 = np.array([60.0, 60.0, 1.0, 3.5, 150.0, 0.0, 5.0])
AUX0 = BathAux(T_reservoir_C=60.0, fe2_reservoir_M=1.0, pH_reservoir=3.5)


def _dp(**kw) -> dict:
    base = {
        "temperature_C": 60.0, "pH": 3.5, "cell_voltage_V": 5.0,
        "j_avg_mA_cm2": 150.0, "electrode_area_m2": 1.0,
        "electrolyte_volume_L": 1000.0, "fe2_M": 1.0,
        "recirculation_flow_L_hr": 6000.0, "reservoir_volume_L": 50000.0,
        "catholyte_volume_L": 800.0, "anolyte_volume_L": 2000.0,
        "fe2_makeup_rate_M_hr": 0.0, "buffer_capacity_beta": 0.05,
        "acid_dose_rate_M_hr": 0.0, "heat_exchange_area_m2": 10.0,
        "T_ambient_C": 25.0,
    }
    base.update(kw)
    return base


def _run(n_steps: int, dp: dict, model: CellProcessModel) -> np.ndarray:
    x = X0.copy()
    aux = AUX0
    for _ in range(n_steps):
        x, aux = step(x, aux, 0.1, dp, model)
    return x


class TestDefaultZero:
    def test_empty_env_returns_disabled(self):
        d = disturbance_from_environment({}, {})
        assert d.enabled is False
        assert d.h_conv_W_m2_K == 0.0
        assert d.rain_cooling_W_m2 == 0.0
        assert d.ingress_dilution_rate_1_hr == 0.0

    def test_coupling_off_is_byte_identical(self):
        model = CellProcessModel()
        x_ref = _run(50, _dp(), model)
        dp = _dp()
        dp["_env_dist"] = DisturbanceInputs(enabled=False)  # disabled coupling
        x_test = _run(50, dp, model)
        np.testing.assert_allclose(x_ref, x_test, atol=1e-12)

    def test_no_env_key_is_identical_to_disabled(self):
        model = CellProcessModel()
        x_no_key = _run(20, _dp(), model)
        d = disturbance_from_environment({}, {})  # returns disabled DisturbanceInputs
        dp = _dp()
        dp["_env_dist"] = d
        x_with_key = _run(20, dp, model)
        np.testing.assert_allclose(x_no_key, x_with_key, atol=1e-12)


class TestPhysicalDirection:
    def test_higher_wind_stronger_convection_coefficient(self):
        calm = disturbance_from_environment({"wind_gust_m_s": 5.0}, {})
        storm = disturbance_from_environment({"wind_gust_m_s": 40.0}, {})
        assert calm.enabled is True
        assert storm.h_conv_W_m2_K > calm.h_conv_W_m2_K > 0.0
        assert storm.rain_cooling_W_m2 == 0.0  # no rain in this scenario

    def test_cold_wind_cools_catholyte(self):
        model = CellProcessModel()
        uncoupled = _run(50, _dp(), model)
        d = disturbance_from_environment(
            {"wind_gust_m_s": 40.0, "T_ambient_C": 0.0}, {}
        )
        dp = _dp()
        dp["_env_dist"] = d
        coupled = _run(50, dp, model)
        assert coupled[0] < uncoupled[0], f"wind should cool catholyte: {coupled[0]} vs {uncoupled[0]}"

    def test_rain_cooling_adds_convective_heat_loss(self):
        d = disturbance_from_environment({"rain_intensity_mm_hr": 100.0}, {})
        assert d.rain_cooling_W_m2 > 0.0
        assert d.rain_cooling_W_m2 == 100.0 * 0.5  # 0.5 W/m2 per mm/hr


class TestIngressGated:
    def test_ingress_dilutes_fe2_and_neutralizes_ph(self):
        model = CellProcessModel()
        dp = _dp()
        dp["_env_dist"] = DisturbanceInputs(
            T_ambient_C=25.0, ingress_dilution_rate_1_hr=0.5, enabled=True
        )
        x = _run(100, dp, model)
        assert x[2] < 1.0, f"ingress should dilute fe2, got {x[2]}"
        assert x[3] > 3.5, f"ingress should raise pH toward neutral, got {x[3]}"

    def test_no_ingress_does_not_dilute(self):
        d = disturbance_from_environment({"wind_gust_m_s": 10.0}, {})
        assert d.ingress_dilution_rate_1_hr == 0.0
        assert d.enabled is True  # wind alone is a real, non-diluting disturbance

    def test_flood_sets_dilution(self):
        d = disturbance_from_environment({"flood_depth_m": 0.5}, {})
        assert d.ingress_dilution_rate_1_hr > 0.0
        assert d.enabled is True


class TestDigitalTwinIntegration:
    def _twin_final_temperature(self, env_state):
        twin = DigitalTwin(seed=1)
        twin.set_environment(env_state, {})
        for _ in range(30):
            twin.update({}, dt_hr=0.1)  # no sensor obs; pure prediction
        return twin.ekf.x[0]

    def test_set_environment_empty_is_noop(self):
        t_empty = self._twin_final_temperature({})
        t_none = self._twin_final_temperature(None)
        assert abs(t_empty - t_none) < 1e-9

    def test_set_environment_wind_cools_twin(self):
        t_default = self._twin_final_temperature({})
        t_storm = self._twin_final_temperature(
            {"wind_gust_m_s": 40.0, "T_ambient_C": 0.0}
        )
        assert t_storm < t_default, f"storm env should cool twin: {t_storm} vs {t_default}"
