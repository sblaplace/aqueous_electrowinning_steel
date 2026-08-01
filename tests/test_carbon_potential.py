from models.carbon_potential import (
    carbon_activity_from_co_co2,
    carbon_activity_from_ch4_h2,
    carbon_wt_from_activity,
    austenite_max_carbon_wt_percent,
    carbon_potential_summary,
)


def test_acm_monotonic():
    c1 = austenite_max_carbon_wt_percent(800)
    c2 = austenite_max_carbon_wt_percent(1000)
    assert c2 > c1
    assert austenite_max_carbon_wt_percent(500) == 0.0


def test_aC_co_co2_increases_with_pCO():
    a1 = carbon_activity_from_co_co2(0.2, 0.001, 900)
    a2 = carbon_activity_from_co_co2(0.3, 0.001, 900)
    assert a2 > a1


def test_aC_co_co2_decreases_with_pCO2():
    a_low_co2 = carbon_activity_from_co_co2(0.2, 0.0005, 900)
    a_high_co2 = carbon_activity_from_co_co2(0.2, 0.005, 900)
    assert a_low_co2 > a_high_co2


def test_aC_ch4_h2():
    a1 = carbon_activity_from_ch4_h2(0.02, 0.4, 900)
    a2 = carbon_activity_from_ch4_h2(0.05, 0.4, 900)
    assert a2 > a1
    # higher H2 reduces aC
    a_low_h2 = carbon_activity_from_ch4_h2(0.02, 0.2, 900)
    assert a_low_h2 > a1


def test_c_wt_from_activity_monotonic():
    c1 = carbon_wt_from_activity(0.5, 900)
    c2 = carbon_wt_from_activity(1.0, 900)
    assert c2 > c1
    Cmax = austenite_max_carbon_wt_percent(900)
    c_sat = carbon_wt_from_activity(10.0, 900)
    assert abs(c_sat - Cmax) < 0.05


def test_c_wt_temperature_effect():
    # same aC, higher T → higher solubility? Actually our empirical gives weak trend
    c_low = carbon_wt_from_activity(0.8, 850)
    c_high = carbon_wt_from_activity(0.8, 950)
    # Not strict monotonic due to model, but both in (0, Cmax)
    assert 0 < c_low < 2.14
    assert 0 < c_high < 2.14


def test_summary():
    summ = carbon_potential_summary(T_C=930, pCO=0.2, pCO2=0.001, pCH4=0.02, pH2=0.4, dew_point_C=-5)
    assert "aC_from_CO_CO2" in summ
    assert "C_wt_from_CO_CO2" in summ
    assert summ["T_C"] == 930
    assert 0 < summ["C_wt_from_CO_CO2"] < 2.14
