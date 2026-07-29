"""Tests for the membrane transport model — Fe³⁺ crossover, acid balance,
anolyte drift, membrane IR drop, and purge scheduling."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.membrane_transport import (  # noqa: E402
    AnolyteState,
    CatholyteState,
    FUMASEP_FKE50,
    MembraneSpec,
    MembraneTransportModel,
    NAFION_N117,
    crossover_vs_current_density,
)


# ─── Fe³⁺ crossover flux vs j for 2 membrane types ────────────────────
def test_crossover_flux_increases_with_current_density():
    """Fe³⁺ crossover must grow with applied current density."""
    for mem in (NAFION_N117, FUMASEP_FKE50):
        model = MembraneTransportModel(membrane=mem, j_mA_cm2=10.0)
        model.anolyte.fe3_M = 0.1
        low = model.evaluate().fe3_crossover_flux

        model2 = MembraneTransportModel(membrane=mem, j_mA_cm2=400.0)
        model2.anolyte.fe3_M = 0.1
        high = model2.evaluate().fe3_crossover_flux

        assert high > low, f"{mem.name}: crossover must increase with j"


def test_crossover_flux_two_membranes_differ():
    """Nafion N117 and Fumasep FKE-50 must give different crossover at the same j."""
    nafion = MembraneTransportModel(membrane=NAFION_N117, j_mA_cm2=100.0)
    nafion.anolyte.fe3_M = 0.1
    fumasep = MembraneTransportModel(membrane=FUMASEP_FKE50, j_mA_cm2=100.0)
    fumasep.anolyte.fe3_M = 0.1

    f_n = nafion.evaluate().fe3_crossover_flux
    f_f = fumasep.evaluate().fe3_crossover_flux
    assert f_n != pytest.approx(f_f, rel=1e-3), "Membranes must give different fluxes"


def test_crossover_sweep_returns_expected_shape():
    """crossover_vs_current_density returns one row per (membrane, j) pair."""
    rows = crossover_vs_current_density()
    assert len(rows) == 2 * 5  # 2 membranes × 5 default j values
    for row in rows:
        assert "fe3_flux_mol_m2_s" in row
        assert "membrane_V_drop" in row
        assert row["fe3_flux_mol_m2_s"] > 0


# ─── Catholyte pH drift over 8h operation ──────────────────────────────
def test_catholyte_pH_drift_over_8_hours():
    """Catholyte H⁺ concentration must change over an 8h simulation."""
    model = MembraneTransportModel(
        j_mA_cm2=100.0,
        catholyte=CatholyteState(h_M=0.1),  # pH 1
    )
    result = model.simulate(duration_hr=8.0, dt_hr=0.1)

    # The catholyte H⁺ must evolve (from membrane input and HER consumption)
    assert not np.allclose(result.catholyte_h_M, result.catholyte_h_M[0], rtol=1e-6)

    # Final pH should be reported and finite
    final_pH = -np.log10(max(result.catholyte_h_M[-1], 1e-30))
    assert np.isfinite(final_pH)


def test_catholyte_pH_drift_rate_scales_with_current():
    """Higher current → faster H⁺ transport → faster pH drift."""
    low = MembraneTransportModel(j_mA_cm2=50.0)
    high = MembraneTransportModel(j_mA_cm2=300.0)
    assert abs(high.catholyte_pH_drift_rate()) > abs(low.catholyte_pH_drift_rate())


# ─── Anolyte Fe³⁺ accumulation vs charge passed ───────────────────────
def test_anolyte_fe3_accumulates_with_charge():
    """Anolyte Fe³⁺ must increase monotonically during operation."""
    model = MembraneTransportModel(j_mA_cm2=100.0)
    result = model.simulate(duration_hr=2.0, dt_hr=0.05)

    # Fe³⁺ starts at zero and must grow (anode produces it)
    assert result.anolyte_fe3_M[-1] > result.anolyte_fe3_M[0]
    assert result.anolyte_fe3_M[0] == pytest.approx(0.0)

    # Monotonic growth (ignoring any purge resets within the interval)
    diffs = np.diff(result.anolyte_fe3_M)
    # All positive except possibly around purges
    non_purge = diffs[result.anolyte_fe3_M[:-1] < 0.3]  # below purge threshold
    assert np.all(non_purge >= -1e-15)


def test_anolyte_fe3_higher_at_higher_current():
    """More current → faster Fe²⁺→Fe³⁺ oxidation → more Fe³⁺."""
    low = MembraneTransportModel(j_mA_cm2=50.0)
    high = MembraneTransportModel(j_mA_cm2=200.0)
    r_low = low.simulate(duration_hr=1.0, dt_hr=0.1)
    r_high = high.simulate(duration_hr=1.0, dt_hr=0.1)
    assert r_high.anolyte_fe3_M[-1] > r_low.anolyte_fe3_M[-1]


# ─── Membrane IR drop vs j ─────────────────────────────────────────────
def test_membrane_ir_drop_increases_linearly_with_j():
    """V_membrane = j · L / κ  must be linear in j."""
    drops = []
    for j in (10.0, 50.0, 100.0, 200.0, 400.0):
        m = MembraneTransportModel(j_mA_cm2=j)
        drops.append(m.membrane_ohmic_drop())

    # Linearity: V(2j) ≈ 2 V(j);  j values: 10, 50, 100, 200, 400
    assert drops[2] == pytest.approx(2.0 * drops[1], rel=1e-6)   # 100 = 2×50
    assert drops[3] == pytest.approx(4.0 * drops[1], rel=1e-6)   # 200 = 4×50
    assert drops[4] == pytest.approx(8.0 * drops[1], rel=1e-6)   # 400 = 8×50


def test_membrane_ir_drop_nafion_vs_fumasep():
    """Thinner Fumasep membrane must have lower IR drop than Nafion at the same j."""
    nafion = MembraneTransportModel(membrane=NAFION_N117, j_mA_cm2=100.0)
    fumasep = MembraneTransportModel(membrane=FUMASEP_FKE50, j_mA_cm2=100.0)
    assert fumasep.membrane_ohmic_drop() < nafion.membrane_ohmic_drop()


def test_membrane_ir_drop_agrees_with_summary():
    """The evaluate() V_drop must match membrane_ohmic_drop()."""
    m = MembraneTransportModel(j_mA_cm2=150.0)
    step = m.evaluate()
    assert step.membrane_V_drop == pytest.approx(m.membrane_ohmic_drop(), rel=1e-12)


# ─── Purge criterion: when anolyte Fe³⁺ > threshold ───────────────────
def test_purge_fires_when_fe3_exceeds_threshold():
    """Purge events must appear when anolyte Fe³⁺ crosses the threshold."""
    model = MembraneTransportModel(
        j_mA_cm2=200.0,
        purge_fe3_threshold_M=0.1,
        purge_fraction=0.3,
    )
    result = model.simulate(duration_hr=4.0, dt_hr=0.05)

    # At 200 mA/cm² the anolyte Fe³⁺ production is fast; threshold 0.1 M
    # should be reached within 4 h.
    assert len(result.purge_events) >= 1, "At least one purge event expected"

    # After purge, Fe³⁺ should drop below threshold
    for t_hr, fe3_before in result.purge_events:
        assert fe3_before >= model.purge_fe3_threshold_M - 1e-10


def test_time_to_purge_is_finite():
    """time_to_purge() must return a positive finite value."""
    m = MembraneTransportModel(j_mA_cm2=100.0)
    t = m.time_to_purge()
    assert 0 < t < 100, f"Expected reasonable purge time, got {t} h"


def test_no_purge_when_below_threshold():
    """No purge events if simulation is short and current is low."""
    model = MembraneTransportModel(
        j_mA_cm2=1.0,
        purge_fe3_threshold_M=5.0,
    )
    result = model.simulate(duration_hr=1.0, dt_hr=0.1)
    assert len(result.purge_events) == 0


# ─── H⁺ transport number ──────────────────────────────────────────────
def test_h_transport_number_close_to_0p9_for_nafion():
    """For Nafion with low Fe³⁺, t_H⁺ should be near 0.9 (literature value)."""
    model = MembraneTransportModel(membrane=NAFION_N117)
    model.anolyte.fe3_M = 0.001  # very low Fe³⁺
    t = model.h_transport_number()
    assert 0.7 < t < 1.0, f"Expected t_H⁺ near 0.9, got {t}"


def test_h_transport_number_drops_with_high_fe3():
    """As anolyte Fe³⁺ rises, H⁺ transport number should decrease."""
    model = MembraneTransportModel()
    model.anolyte.fe3_M = 0.001
    t_low_fe3 = model.h_transport_number()

    model.anolyte.fe3_M = 2.0
    t_high_fe3 = model.h_transport_number()
    assert t_low_fe3 > t_high_fe3


# ─── Invalid parameters ────────────────────────────────────────────────
@pytest.mark.parametrize(
    "kwargs",
    [
        {"electrode_area_m2": 0.0},
        {"electrode_area_m2": -1.0},
        {"j_mA_cm2": -10.0},
    ],
)
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        MembraneTransportModel(**kwargs)


def test_simulate_rejects_nonpositive_duration():
    m = MembraneTransportModel()
    with pytest.raises(ValueError):
        m.simulate(duration_hr=0.0)
    with pytest.raises(ValueError):
        m.simulate(dt_hr=-1.0)


# ─── Summary / snapshot ────────────────────────────────────────────────
def test_evaluate_returns_all_fields():
    """evaluate() result must have all expected attributes."""
    m = MembraneTransportModel()
    s = m.evaluate()
    assert s.fe3_crossover_flux >= 0
    assert s.membrane_V_drop > 0
    assert s.anolyte_fe3_production_mol_s > 0


def test_summary_contains_expected_keys():
    m = MembraneTransportModel()
    s = m.summary()
    for key in (
        "membrane",
        "membrane_V_drop (V)",
        "Fe³⁺ crossover flux (mol/m²/s)",
        "t_H⁺",
        "t_Fe³⁺",
        "time_to_purge (h)",
    ):
        assert key in s, f"Missing key: {key}"


def test_simulation_result_summary():
    m = MembraneTransportModel(j_mA_cm2=100.0)
    r = m.simulate(duration_hr=1.0)
    s = r.summary()
    assert "duration_hr" in s
    assert s["duration_hr"] == pytest.approx(1.0, abs=0.15)


# ─── Diffusion vs migration decomposition ──────────────────────────────
def test_diffusion_flux_driven_by_concentration_gradient():
    """Higher anolyte Fe³⁺ → larger diffusion flux (same sign as migration)."""
    model = MembraneTransportModel(j_mA_cm2=0.0, electrode_area_m2=0.01)
    model.anolyte.fe3_M = 0.5
    model.catholyte.fe3_M = 0.0
    total, mig, diff = model.fe3_crossover_flux()
    # At j=0 there is no migration, only diffusion
    assert mig == pytest.approx(0.0, abs=1e-30)
    assert diff > 0, "Diffusion drives Fe³⁺ from high to low concentration"
    assert total == pytest.approx(diff, rel=1e-12)


def test_migration_flux_zero_at_zero_current():
    """At j=0, the migration component must be exactly zero."""
    model = MembraneTransportModel(j_mA_cm2=0.0)
    model.anolyte.fe3_M = 0.5
    _, mig, _ = model.fe3_crossover_flux()
    assert mig == pytest.approx(0.0, abs=1e-30)
