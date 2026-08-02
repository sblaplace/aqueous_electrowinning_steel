"""Tests for RC-1 deployable reference-cell design synthesis."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from models.reference_cell_design import (
    build_reference_cell_operating_twin,
    CandidateDesign,
    channel_hydraulics,
    enumerate_candidates,
    evaluate_design,
    load_reference_cell_config,
    synthesize_reference_cell_design,
)


@pytest.fixture
def config():
    return load_reference_cell_config()


class TestReferenceCellConfig:
    def test_rc1_maps_300_milliamp_per_cm2_to_three_amp(self, config):
        assert config.active_area_cm2 == pytest.approx(10.0)
        assert config.operating_current_A(300.0) == pytest.approx(3.0)
        assert config.max_current_A == pytest.approx(3.0)

    def test_rc1_has_explicit_chemistry_not_generic_twin_defaults(self, config):
        assert config.bath.c_FeSO4_M == pytest.approx(1.0)
        assert config.bath.pH == pytest.approx(2.0)
        assert config.target_temperature_C == pytest.approx(60.0)
        assert config.total_volume_L == pytest.approx(2.0)

    def test_candidate_space_contains_the_controlled_nominal_design(self, config):
        candidates = enumerate_candidates(config)
        assert CandidateDesign(10.0, 3e-3, 0.5) in candidates
        assert len(candidates) == 36

    def test_operating_twin_uses_rc1_hardware_limits(self, config):
        twin = build_reference_cell_operating_twin(config)
        assert twin.config.cell_id == "RC-1"
        assert twin.config.max_current_A == pytest.approx(3.0)
        assert twin.config.max_current_density_mA_cm2 == pytest.approx(300.0)


class TestHydraulics:
    def test_higher_flow_increases_velocity_and_pressure_drop(self, config):
        low = channel_hydraulics(config, CandidateDesign(10.0, 3e-3, 0.1))
        high = channel_hydraulics(config, CandidateDesign(10.0, 3e-3, 1.0))
        assert high["superficial_velocity_m_s"] > low["superficial_velocity_m_s"]
        assert high["pressure_drop_Pa"] > low["pressure_drop_Pa"]
        assert high["channel_residence_time_s"] < low["channel_residence_time_s"]

    def test_shallower_channel_has_higher_pressure_drop(self, config):
        shallow = channel_hydraulics(config, CandidateDesign(10.0, 2e-3, 0.5))
        deep = channel_hydraulics(config, CandidateDesign(10.0, 5e-3, 0.5))
        assert shallow["pressure_drop_Pa"] > deep["pressure_drop_Pa"]

    def test_nominal_flow_stays_inside_laminar_sizing_envelope(self, config):
        nominal = channel_hydraulics(config, CandidateDesign(10.0, 3e-3, 0.5))
        assert nominal["reynolds_number"] < 2_100.0


class TestDesignSynthesis:
    def test_design_rejects_candidate_above_current_limit(self, config, monkeypatch):
        _stub_physics(monkeypatch)
        result = evaluate_design(config, CandidateDesign(25.0, 3e-3, 0.5))
        assert not result.feasible
        assert "current_limit" in result.failures
        assert "hydrogen_design_limit" in result.failures

    def test_synthesis_selects_nominal_rc1_geometry(self, config, monkeypatch):
        _stub_physics(monkeypatch)
        report = synthesize_reference_cell_design(config)
        selected = report["selected_design"]["candidate"]
        assert selected["active_area_cm2"] == pytest.approx(10.0)
        assert selected["channel_depth_mm"] == pytest.approx(3.0)
        assert selected["flow_L_min"] == pytest.approx(0.5)
        assert report["feasible_candidate_count"] > 0


def _stub_physics(monkeypatch):
    """Keep design-selection tests fast; electrochemistry has its own test suite."""
    class StubCellPhysics:
        def __init__(self, bath, geometry, conditions):
            self.geometry = geometry

        def solve_at_j(self, j):
            return SimpleNamespace(
                V_cell=2.5,
                current_efficiency=0.8,
                specific_energy_kWh_t=3_000.0,
                deposition_rate_um_hr=100.0,
            )

    monkeypatch.setattr("models.reference_cell_design.CellPhysics", StubCellPhysics)
