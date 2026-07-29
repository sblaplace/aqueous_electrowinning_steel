"""Tests for local cathode pH and Fe(OH)2 boundary-layer feedback."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.boundary_layer import CathodeBoundaryLayer


def test_surface_pH_rises_above_bulk():
    model = CathodeBoundaryLayer(bulk_pH=2.0, her_i0=1e-4)
    state = model.solve(100.0)
    assert state.surface_pH > state.bulk_pH
    assert state.local_pH_rise > 0.0


def test_more_her_gives_larger_local_pH_rise():
    suppressed = CathodeBoundaryLayer(her_i0=1e-7).solve(100.0)
    active = CathodeBoundaryLayer(her_i0=1e-3).solve(100.0)
    assert active.local_pH_rise > suppressed.local_pH_rise


def test_agitation_reduces_local_pH_rise():
    stagnant = CathodeBoundaryLayer(boundary_layer_m=2e-4).solve(100.0)
    agitated = CathodeBoundaryLayer(boundary_layer_m=2e-5).solve(100.0)
    assert agitated.local_pH_rise < stagnant.local_pH_rise


def test_fe_surface_depletes_and_transport_limit_scales():
    model = CathodeBoundaryLayer(fe_conc_M=0.1)
    state = model.solve(10.0)
    assert 0.0 < state.surface_fe_M <= model.fe_conc_M
    thin = CathodeBoundaryLayer(fe_conc_M=0.1, boundary_layer_m=2e-5)
    thick = CathodeBoundaryLayer(fe_conc_M=0.1, boundary_layer_m=2e-4)
    assert thin.fe_transport_limit_A_m2 == pytest.approx(
        10.0 * thick.fe_transport_limit_A_m2
    )


def test_precipitation_activates_at_high_bulk_pH():
    state = CathodeBoundaryLayer(bulk_pH=9.0, her_i0=1e-4).solve(5.0)
    assert state.precipitation_active
    assert state.feoh2_supersaturation >= 1.0


def test_profiles_are_monotonic_and_end_at_bulk_values():
    model = CathodeBoundaryLayer(her_i0=1e-4)
    state = model.solve(100.0)
    x, fe, oh = model.concentration_profiles(state, points=11)
    assert len(x) == len(fe) == len(oh) == 11
    assert np.all(np.diff(x) > 0.0)
    assert fe[0] == pytest.approx(state.surface_fe_M)
    assert fe[-1] == pytest.approx(model.fe_conc_M)
    assert oh[0] == pytest.approx(state.surface_oh_M)
    assert oh[-1] == pytest.approx(model.bulk_oh_M)


def test_nonpositive_current_is_rejected():
    with pytest.raises(ValueError):
        CathodeBoundaryLayer().solve(0.0)
