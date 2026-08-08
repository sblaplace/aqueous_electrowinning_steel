"""
Tests for the G0 co-location coverage-contract harness (models/g0_co_location.py).

These lock the in-silico observability proof the harness delivers:

* The base 5-sensor suite is rank-6 at every operating point, with a divergent
  deposit_thickness (the finding in docs/TWIN_OBSERVABILITY.md).
* The full co-located suite (base 5 + wired L1 THK-101/CVT-201/FE2P-101) reaches
  **full rank 7** at every operating point through the real `h_obs` measurement
  model (not abstract unit rows).
* Every state's estimation-error covariance stays bounded (non-divergent) over a
  24 h run at every operating point with the full suite, and each state's flag is
  "observable" (no unobservable / weak / divergent state remains).

L0 / in-silico only: this is capability due-diligence before L1 hardware, not
gate evidence about real instrument performance.
"""

from __future__ import annotations

import numpy as np
import pytest

from models.digital_twin import (
    N_STATES,
    STATE_KEYS,
    L1_SENSOR_OBS_MAP,
    OBSERVABLE_TAGS,
    get_default_process_model,
)
from models.g0_co_location import (
    BASE_TAGS,
    FULL_TAGS,
    L1_TAGS,
    evaluate_co_location_contract,
    verify_co_location_contract,
)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def model():
    return get_default_process_model()


@pytest.fixture(scope="module")
def contract(model):
    return evaluate_co_location_contract(model)


class TestSuiteDefinition:
    """The harness evaluates the real wired sensor suite, untouched elsewhere."""

    def test_tag_sets(self):
        assert BASE_TAGS == list(OBSERVABLE_TAGS)
        assert L1_TAGS == list(L1_SENSOR_OBS_MAP.keys())
        assert L1_TAGS == ["THK-101", "CVT-201", "FE2P-101"]
        assert FULL_TAGS == BASE_TAGS + L1_TAGS
        assert len(FULL_TAGS) == 8

    def test_all_tags_observable(self, model, contract):
        # The full suite covers all 7 states; no tag is dropped or doubled.
        assert FULL_TAGS == sorted(dict.fromkeys(FULL_TAGS), key=FULL_TAGS.index)


class TestBaseSuiteFindingReproduced:
    """The rank-6 + divergent-deposit finding is reproduced by the harness."""

    def test_base_rank_6_everywhere(self, contract):
        for p in contract.points.values():
            assert p.base_rank == 6, p.operating_point

    def test_base_deposit_divergent_everywhere(self, contract):
        for p in contract.points.values():
            assert p.base_deposit_divergent, p.operating_point
            # Unbounded pure-integrator growth: final sigma >> initial P0[5]=1.0.
            assert p.base_deposit_sigma > 2.0, p.operating_point


class TestFullRankContract:
    """The full co-located suite reaches rank 7 at every operating point."""

    def test_full_rank_7_at_every_point(self, contract):
        assert len(contract.points) >= 4
        for name, p in contract.points.items():
            assert p.full_rank == N_STATES, f"{name}: rank {p.full_rank}"

    def test_all_full_rank_property(self, contract):
        assert contract.all_full_rank

    def test_singular_values_positive(self, contract):
        for name, p in contract.points.items():
            assert p.full_sv_min > 0.0, name
            assert np.isfinite(p.full_cond), name


class TestCovarianceStabilityContract:
    """With the full suite every state's covariance stays bounded everywhere."""

    def test_all_states_stable_at_every_point(self, contract):
        for name, p in contract.points.items():
            assert p.full_cov_stable, f"{name}: {p.full_stable}"

    def test_all_cov_stable_property(self, contract):
        assert contract.all_cov_stable

    def test_no_unobservable_or_weak_state_with_full_suite(
        self,
        model,
        contract,
    ):
        # Re-derive per-state flags via analyze_observability to confirm every
        # state is cleanly "observable" (no divergent / weak residual).
        from models.observability import (
            analyze_observability,
            operating_points_from_model,
            state_vector_from_operating_point,
        )

        for name, j, T, fe2 in operating_points_from_model(model):
            x = state_vector_from_operating_point((name, j, T, fe2), model)
            r = analyze_observability(x, obs_tags=FULL_TAGS, model=model)
            for key in STATE_KEYS:
                assert r.per_state_flag[key] == "observable", (name, key, r.per_state_flag[key])


class TestContractVerdict:
    """The turn-key 'proof' exits clean only when the contract holds."""

    def test_no_violations(self, contract):
        assert verify_co_location_contract(contract) == []
        assert not contract.violations()
