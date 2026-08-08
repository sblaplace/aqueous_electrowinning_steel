"""
Tests for the digital-twin EKF observability & sensor-placement analysis.

Load-bearing findings these tests lock in:

* The current 5-sensor suite gives an observability Gramian of rank 6, not 7.
* ``deposit_thickness`` is structurally unobservable (its column of the
  observation matrix is identically zero) and, being a pure integrator, its
  estimation error diverges over a run.  A thickness sensor restores it.
* ``cell_voltage`` is not directly measured (VT-201 reports the physics-predicted
  ``v_cell``, not state 6), but it is now *weakly observable* through the coupled
  PR #29 dynamics, and its error stays bounded.  A direct observation strengthens it.
* ``bulk_fe2`` is observable through the ``v_cell`` coupling; an inline Fe2+ probe
  further tightens its conditioning.
* The recommended minimum sensor set makes all 7 states observable at every
  tested operating point.
"""

from __future__ import annotations

import numpy as np
import pytest

from models.digital_twin import (
    N_STATES,
    STATE_KEYS,
    _OBS_MAP,
    get_default_process_model,
)
from models.observability import (
    FE2_PROBE,
    THICKNESS_SENSOR,
    CELL_VOLTAGE_DIRECT,
    RECOMMENDED_MINIMUM_SENSORS,
    MINIMUM_SET_FOR_FULL_OBSERVABILITY,
    analyze_observability,
    analyze_sensor_set,
    characterize_current_suite,
    evaluate_sensor_set_over_grid,
    rank_candidate_sensors,
    operating_points_from_model,
    state_vector_from_operating_point,
)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def model():
    """Shared physics surrogate (built once; the EKF's measurement model)."""
    return get_default_process_model()


@pytest.fixture(scope="module")
def nominal_x(model):
    return state_vector_from_operating_point(operating_points_from_model(model)[0], model)


class TestTwinContractUnchanged:
    """Guard rails: the EKF state vector / measurement model were not touched."""

    def test_state_vector_unchanged(self):
        assert N_STATES == 7
        assert STATE_KEYS == [
            "catholyte_temperature",
            "anolyte_temperature",
            "bulk_fe2",
            "bulk_pH",
            "current_density",
            "deposit_thickness",
            "cell_voltage",
        ]

    def test_observation_map_unchanged(self):
        # The current suite is exactly these 5 direct-observation tags.
        assert set(_OBS_MAP) == {"TT-101", "TT-201", "pHAT-101", "CT-201", "VT-201"}

    def test_deposit_and_voltage_not_directly_measured(self, model, nominal_x):
        """h_obs has no direct row for deposit_thickness or cell_voltage.

        deposit_thickness (5) appears in no output equation; VT-201 reports the
        physics-predicted ``v_cell``, so cell_voltage (6) is not read directly.
        """
        res = analyze_observability(nominal_x, model=model)
        assert np.allclose(res.H[:, 5], 0.0)
        assert np.allclose(res.H[:, 6], 0.0)


class TestCurrentSuiteObservability:
    """Rank + conditioning of the current 5-sensor suite."""

    def test_rank_is_6_at_every_operating_point(self, model):
        results = characterize_current_suite(model)
        assert len(results) >= 4
        for name, r in results.items():
            assert r.rank == 6, f"{name}: expected rank 6, got {r.rank}"

    def test_observability_matrix_shape(self, model, nominal_x):
        r = analyze_observability(nominal_x, model=model)
        n_obs, n_states = r.H.shape
        assert r.observability_matrix.shape == (n_obs * r.horizon_steps, n_states)
        assert r.gramian.shape == (N_STATES, N_STATES)

    def test_gramian_is_symmetric_psd(self, model, nominal_x):
        r = analyze_observability(nominal_x, model=model)
        np.testing.assert_allclose(r.gramian, r.gramian.T, atol=1e-9)
        assert np.all(np.linalg.eigvalsh(r.gramian) >= -1e-9)

    def test_directly_observed_states_are_observable(self, model):
        results = characterize_current_suite(model)
        r = results["nominal"]
        for key in ("catholyte_temperature", "anolyte_temperature", "bulk_pH", "current_density"):
            assert r.per_state_flag[key] == "observable", key


class TestDepositThicknessUnobservable:
    """deposit_thickness is structurally unobservable and its error diverges."""

    def test_deposit_structurally_unobservable(self, model):
        for name, r in characterize_current_suite(model).items():
            assert r.per_state_flag["deposit_thickness"].startswith("unobservable")
            assert r.per_state_score[STATE_KEYS.index("deposit_thickness")] == pytest.approx(
                0.0, abs=1e-6
            )

    def test_deposit_error_grows_unbounded(self, model, nominal_x):
        """Over a run, the deposit covariance keeps growing (pure integrator)."""
        r = analyze_observability(nominal_x, model=model)
        assert r.per_state_flag["deposit_thickness"] == "unobservable_divergent"
        # Final sigma far exceeds the initial P0 (1.0 -> ~5.1 over 24 h).
        assert r.per_state_sigma[5] > 2.0

    def test_thickness_sensor_restores_observability(self, model, nominal_x):
        r0 = analyze_observability(nominal_x, model=model)
        r1 = analyze_sensor_set(nominal_x, list(r0.tags), [THICKNESS_SENSOR], model=model)
        assert r1.per_state_flag["deposit_thickness"] == "observable"
        assert r1.rank > r0.rank
        assert r1.rank == 7


class TestCellVoltageWeakBounded:
    """cell_voltage is not directly measured but is weakly observable via the
    coupled dynamics (detectable, bounded); a direct observation strengthens it."""

    def test_cell_voltage_flag(self, model, nominal_x):
        r = analyze_observability(nominal_x, model=model)
        assert r.per_state_flag["cell_voltage"] == "weak"

    def test_cell_voltage_error_stays_bounded(self, model, nominal_x):
        r = analyze_observability(nominal_x, model=model)
        # Contractive dynamics + fast v_cell tracking -> small bounded sigma,
        # unlike deposit (a pure integrator that diverges).
        assert r.per_state_sigma[6] < 1.0

    def test_cell_voltage_direct_observation_restores_it(self, model, nominal_x):
        r0 = analyze_observability(nominal_x, model=model)
        r1 = analyze_sensor_set(nominal_x, list(r0.tags), [CELL_VOLTAGE_DIRECT], model=model)
        assert r1.per_state_flag["cell_voltage"] == "observable"
        assert r1.rank == 6


class TestBulkFe2Conditioning:
    """bulk_fe2 is observable via the v_cell coupling; a probe tightens it."""

    def test_bulk_fe2_observable_via_coupling(self, model):
        r = analyze_observability(
            state_vector_from_operating_point(operating_points_from_model(model)[0], model),
            model=model,
        )
        assert r.per_state_flag["bulk_fe2"] == "observable"

    def test_fe2_probe_improves_conditioning(self, model, nominal_x):
        r0 = analyze_observability(nominal_x, model=model)
        r1 = analyze_sensor_set(nominal_x, list(r0.tags), [FE2_PROBE], model=model)
        # A direct probe further tightens the (already-observable) bulk_fe2 state.
        assert r1.per_state_sigma[2] < r0.per_state_sigma[2]


class TestRecommendedMinimumSet:
    """The recommended minimum set makes all 7 states observable everywhere."""

    def test_recommended_set_makes_all_7_observable(self, model):
        results = evaluate_sensor_set_over_grid(list(RECOMMENDED_MINIMUM_SENSORS), model)
        assert len(results) >= 4
        for name, r in results.items():
            assert r.rank == N_STATES, f"{name}: rank {r.rank}"
            for key, flag in r.per_state_flag.items():
                assert not flag.startswith("unobservable"), (name, key, flag)

    def test_minimum_set_for_full_observability_is_subset(self):
        # The strict minimum (deposit + cell-voltage) is a subset of the
        # recommended set, which also adds the Fe2+ probe for conditioning.
        recommended = set(RECOMMENDED_MINIMUM_SENSORS)
        for s in MINIMUM_SET_FOR_FULL_OBSERVABILITY:
            assert s in recommended

    def test_recommended_set_full_rank_at_every_point(self, model):
        results = evaluate_sensor_set_over_grid(list(RECOMMENDED_MINIMUM_SENSORS), model)
        for name, r in results.items():
            assert r.smallest_singular_value > 0.0, name
            assert np.isfinite(r.condition_number), name


class TestSensorRanking:
    """Marginal-information ranking prioritises observability-restoring sensors."""

    def test_ranking_resolves_unobservability_first(self, model, nominal_x):
        ranking = rank_candidate_sensors(nominal_x, model=model)
        assert len(ranking) >= 4
        top = ranking[0]
        assert top.resolves_unobservability
        # Deposit thickness is the single most information-critical state.
        assert top.target_key == "deposit_thickness"
        # deposit_thickness is the only structurally-unobservable state; it is
        # the single top priority.
        resolving = [r for r in ranking if r.resolves_unobservability]
        assert {r.target_key for r in resolving} == {"deposit_thickness"}
        assert resolving  # non-empty; deposit is the resolvable priority
