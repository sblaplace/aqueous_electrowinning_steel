import pytest
import numpy as np
from models.foil_calibration import (
    FoilMeasurement,
    o2_probe_mv_to_pO2,
    pO2_to_carbon_activity_via_co_co2,
    fit_diffusivity_from_foil_data,
    fit_carbon_potential_offset,
    fit_tempering_softening,
    fit_mechanical_hall_petch,
)


def test_o2_probe_to_pO2():
    pO2 = o2_probe_mv_to_pO2(1100, 930)
    assert 1e-20 < pO2 < 1e-3
    # higher mV → lower pO2
    pO2_low = o2_probe_mv_to_pO2(800, 930)
    assert pO2_low > pO2


def test_pO2_to_aC():
    pO2 = o2_probe_mv_to_pO2(1100, 930)
    aC = pO2_to_carbon_activity_via_co_co2(pO2, 0.2, 930)
    assert aC > 0


def test_foil_fit():
    # synthetic foils increasing C with time
    foils = [
        FoilMeasurement(time_hr=0.5, temperature_C=930, pCO_atm=0.2, pCO2_atm=0.001, foil_thickness_um=75, measured_avg_C_wt_percent=0.35, o2_probe_mV=1150),
        FoilMeasurement(time_hr=1.0, temperature_C=930, pCO_atm=0.2, pCO2_atm=0.001, foil_thickness_um=75, measured_avg_C_wt_percent=0.65, o2_probe_mV=1120),
        FoilMeasurement(time_hr=2.0, temperature_C=930, pCO_atm=0.2, pCO2_atm=0.001, foil_thickness_um=75, measured_avg_C_wt_percent=0.95, o2_probe_mV=1100),
    ]
    res = fit_diffusivity_from_foil_data(foils, initial_C_wt=0.02)
    assert "D_fit_m2_s" in res
    assert res["D_fit_m2_s"] > 0
    assert res["Cs_fit_wt_percent"] > 0

    offset = fit_carbon_potential_offset(foils)
    assert "offset_factor_aC_probe_over_theory_mean" in offset or offset["n_o2_measurements"] == 3


def test_tempering_fit():
    data = [
        {"T_C": 200, "t_hr": 1.0, "HV_q": 800, "HV_measured": 750},
        {"T_C": 400, "t_hr": 1.0, "HV_q": 800, "HV_measured": 550},
        {"T_C": 600, "t_hr": 1.0, "HV_q": 800, "HV_measured": 350},
    ]
    res = fit_tempering_softening(data)
    assert res["k_fit"] > 0
    assert res["n_points"] == 3


def test_hall_petch_fit():
    # synthetic: d=5 µm → 300 MPa, d=0.5 µm → 500 MPa
    d_um = np.array([5.0, 2.0, 1.0, 0.5, 0.2])
    y_MPa = np.array([300, 350, 400, 500, 650])
    res = fit_mechanical_hall_petch(d_um, y_MPa)
    assert res["sigma0_MPa"] > 0
    assert res["k_HP_MPa_sqrt_m"] > 0
    assert res["r_squared"] > 0.8
