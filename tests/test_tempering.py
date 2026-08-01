import numpy as np
from models.tempering import (
    AlloyComposition,
    martensite_start_C,
    retained_austenite_fraction_koistinen_marburger,
    hollomon_jaffe_parameter,
    tempered_hardness_hollomon_jaffe,
    tempering_curve,
    case_hardness_after_tempering,
    recommended_tempering_for_target_hv,
)


def test_Ms_decreases_with_C():
    ms_low = martensite_start_C(AlloyComposition(C=0.2))
    ms_high = martensite_start_C(AlloyComposition(C=0.8))
    assert ms_high < ms_low


def test_retained_austenite():
    chem = AlloyComposition(C=0.8)
    Ms = martensite_start_C(chem)
    ra_quench = retained_austenite_fraction_koistinen_marburger(Ms, 25)
    ra_above = retained_austenite_fraction_koistinen_marburger(Ms, Ms+50)
    assert 0 < ra_quench < 1
    assert ra_above == 1.0
    # lower Tq → less RA
    ra_cold = retained_austenite_fraction_koistinen_marburger(Ms, -50)
    assert ra_cold < ra_quench


def test_hollomon_jaffe_increases():
    P_low = hollomon_jaffe_parameter(200, 1.0)
    P_high = hollomon_jaffe_parameter(500, 1.0)
    assert P_high > P_low
    P_long = hollomon_jaffe_parameter(400, 10.0)
    P_short = hollomon_jaffe_parameter(400, 0.1)
    assert P_long > P_short


def test_tempered_hardness_decreases():
    HV_q = 800
    P_low = hollomon_jaffe_parameter(200, 1.0)
    P_high = hollomon_jaffe_parameter(600, 1.0)
    hv_low = tempered_hardness_hollomon_jaffe(HV_q, P_low)
    hv_high = tempered_hardness_hollomon_jaffe(HV_q, P_high)
    assert hv_high <= hv_low
    assert hv_high >= 150


def test_tempering_curve():
    chem = AlloyComposition(C=0.6)
    curve = tempering_curve(HV_q=800, chem=chem, t_hr=1.0, T_range_C=(150, 600), n_points=20)
    assert len(curve["T_C"]) == 20
    assert curve["HV_tempered"][0] >= curve["HV_tempered"][-1]
    assert np.all(curve["f_RA_remaining"] <= curve["f_RA_as_quenched"] + 1e-9)


def test_case_hardness():
    C_prof = np.array([1.1, 0.8, 0.5, 0.2])
    x = np.array([0, 100, 300, 600])
    HV_q, fRA_q, HV_t = case_hardness_after_tempering(C_prof, x, temper_T_C=180, temper_t_hr=1.0)
    assert len(HV_q) == 4
    assert HV_q[0] >= HV_q[-1]
    assert np.all(HV_t <= HV_q + 1e-6)


def test_recommended_tempering():
    HV_q = 800
    target = 500
    T_rec = recommended_tempering_for_target_hv(HV_q, target, t_hr=1.0)
    assert T_rec is not None
    assert 150 <= T_rec <= 700
    # unreachable high target
    T_none = recommended_tempering_for_target_hv(HV_q, 850, t_hr=1.0)
    assert T_none is None
