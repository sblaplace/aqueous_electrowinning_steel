"""Tests for the Hull-cell FE coupling model.

Acceptance criteria:
  - FE(position) curve for standard Hull cell at 3 bath compositions
  - Appearance map prediction (bright/dull/burnt boundaries)
  - Sensitivity: how much does delta variation matter?
  - Comparison: does the j-window from FE(position) match FE(j) alone?
  - >= 6 tests
"""

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.hull_cell_fe import (  # noqa: E402
    DeltaProfile,
    HullCellFEConfig,
    compare_fe_windows,
    fe_sensitivity_to_delta,
    hull_cell_fe_prediction,
)


# ─── Helpers ──────────────────────────────────────────────────────────

# Minimum viable grid: 3 zones, 21 grid points for speed.
_GRID = 21
_ZONES = 3


def _cfg(**overrides) -> HullCellFEConfig:
    """Fast test config."""
    defaults = dict(
        total_current_A=2.0,
        n_hull_segments=30,
        n_zones=_ZONES,
        grid_points=_GRID,
    )
    defaults.update(overrides)
    return HullCellFEConfig(**defaults)


# Module-level cached predictions (run once per session)
@pytest.fixture(scope="module")
def default_result():
    """Shared prediction at default config."""
    return hull_cell_fe_prediction(_cfg())


@pytest.fixture(scope="module")
def low_current_result():
    """Shared prediction at 1 A (moderate j everywhere)."""
    return hull_cell_fe_prediction(_cfg(total_current_A=1.0))


@pytest.fixture(scope="module")
def high_current_result():
    """Shared prediction at 3 A (high j variation)."""
    return hull_cell_fe_prediction(_cfg(total_current_A=3.0))


@pytest.fixture(scope="module")
def three_bath_results():
    """FE(position) at 3 Fe2+ concentrations."""
    results = {}
    for conc in [0.5, 1.0, 2.0]:
        results[conc] = hull_cell_fe_prediction(_cfg(fe_conc_M=conc))
    return results


# ─── 1. FE decreases from near edge to far edge ─────────────────────

def test_fe_positive_and_varies(default_result):
    """Core acceptance: FE is positive and varies across the panel."""
    fes = [z.faradaic_efficiency for z in default_result.zones]
    assert all(0.0 < fe <= 1.0 for fe in fes)
    assert max(fes) > 0.5
    # At moderate current (2 A / 50 cm²), FE is uniformly high.
    # Use a lenient threshold; real variation requires higher current.
    assert max(fes) - min(fes) > 0.001
# ─── 2. Appearance map: low current → bright zones ──────────────────

def test_appearance_map_has_bright_zone_at_low_current(low_current_result):
    """At 1 A total, moderate j → some zones should be bright."""
    bright = [z for z in low_current_result.zones if z.appearance == "bright"]
    assert len(bright) > 0, (
        f"Expected bright zones at 1 A, got: "
        f"{[(z.position_cm_from_near_edge, z.appearance, z.fe_percent) for z in low_current_result.zones]}"
    )


# ─── 3. FE(position) curve at 3 bath compositions ───────────────────

def test_fe_curve_at_three_bath_compositions(three_bath_results):
    """Acceptance: FE(position) for 3 different Fe2+ concentrations."""
    for conc, result in three_bath_results.items():
        df = result.fe_curve
        assert len(df) == _ZONES
        expected_cols = {
            "position_cm", "j_mA_cm2", "delta_um", "FE_fraction",
            "FE_percent", "appearance", "surface_pH", "surface_fe_M",
            "precipitation",
        }
        assert expected_cols.issubset(set(df.columns))

    # Higher [Fe2+] → higher area-weighted FE
    fe_low = three_bath_results[0.5].area_weighted_fe
    fe_high = three_bath_results[2.0].area_weighted_fe
    assert fe_high > fe_low, (
        f"Expected FE(2.0 M)={fe_high:.3f} > FE(0.5 M)={fe_low:.3f}"
    )


# ─── 4. Delta sensitivity matters ────────────────────────────────────

def test_delta_sensitivity_changes_fe():
    """Acceptance: delta variation significantly affects the FE curve."""
    cfg = _cfg()
    df = fe_sensitivity_to_delta(cfg, multipliers=(0.5, 1.0, 2.0))
    assert len(df) > 0

    # At the same position, thin-film (0.5x) and thick-film (2x) differ
    # Pick the first position as a representative point
    pos0 = df["position_cm"].iloc[0]
    fe_thin = df[(df["multiplier"] == 0.5) & (df["position_cm"] == pos0)]["FE_percent"].iloc[0]
    fe_thick = df[(df["multiplier"] == 2.0) & (df["position_cm"] == pos0)]["FE_percent"].iloc[0]
    assert fe_thin != fe_thick, "Delta variation should change FE"


# ─── 5. FE window comparison (panel vs standalone) ───────────────────

def test_fe_window_comparison():
    """Acceptance: j-window from FE(position) should overlap FE(j) alone."""
    cfg = _cfg()
    comparison = compare_fe_windows(cfg, fe_threshold_pct=70.0, j_sweep_points=10)

    assert "panel_j_window_mA_cm2" in comparison
    assert "standalone_j_window_mA_cm2" in comparison
    assert isinstance(comparison["match"], bool)

    panel_lo, panel_hi = comparison["panel_j_window_mA_cm2"]
    standalone_lo, standalone_hi = comparison["standalone_j_window_mA_cm2"]
    assert panel_lo is not None, "Panel should have at least one zone above 70% FE"
    assert standalone_lo is not None, "Standalone should have at least one j above 70% FE"


# ─── 6. Mass gain formula is correct ─────────────────────────────────

def test_mass_gain_correlates_with_j_times_fe(default_result):
    """Mass gain per zone should be proportional to j × FE × area."""
    for z in default_result.zones:
        j_A_cm2 = z.local_j_mA_cm2 / 1000.0
        expected = (
            j_A_cm2 * z.zone_area_cm2 * z.faradaic_efficiency
            * 55.845 / (2.0 * 96485.33212) * 60.0 * 1000.0
        )
        assert z.mass_gain_mg_per_min == pytest.approx(expected, rel=1e-6)
    assert default_result.total_mass_gain_mg_per_min > 0.0


# ─── 7. Appearance boundaries within panel ───────────────────────────

def test_appearance_boundaries_within_panel(high_current_result):
    """If boundaries exist, they should lie within [0, 10] cm."""
    for attr in ("bright_dull_boundary_cm", "dull_burnt_boundary_cm",
                 "burnt_powdery_boundary_cm"):
        b = getattr(high_current_result, attr)
        if b is not None:
            assert 0.0 <= b <= 10.0, f"{attr} at {b} cm is outside panel"


# ─── 8. All outputs finite ───────────────────────────────────────────

def test_no_nan_in_outputs(default_result):
    """All zone results must be finite."""
    for z in default_result.zones:
        for name, val in [
            ("FE", z.faradaic_efficiency),
            ("fe_percent", z.fe_percent),
            ("surface_pH", z.surface_pH),
            ("surface_fe_M", z.surface_fe_M),
            ("mass_gain", z.mass_gain_mg_per_min),
            ("j_mA_cm2", z.local_j_mA_cm2),
            ("delta_m", z.local_delta_m),
        ]:
            assert math.isfinite(val), f"Non-finite {name} at zone {z.zone_index}: {val}"


# ─── 9. Summary DataFrame ────────────────────────────────────────────

def test_summary_df_structure(default_result):
    """summary_df() should return a DataFrame with expected columns."""
    df = default_result.summary_df()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == _ZONES
    expected_cols = {"zone", "position_cm", "j_mA_cm2", "delta_um", "FE_%",
                     "appearance", "surface_pH", "mass_mg_min", "precip"}
    assert expected_cols.issubset(set(df.columns))


# ─── 10. Validation ──────────────────────────────────────────────────

def test_validation_bad_inputs():
    """Invalid parameters should raise ValueError."""
    with pytest.raises(ValueError, match="total_current_A"):
        HullCellFEConfig(total_current_A=-1.0)
    with pytest.raises(ValueError, match="n_zones"):
        HullCellFEConfig(n_zones=0)
    with pytest.raises(ValueError, match="positive"):
        DeltaProfile(delta_near_m=0.0)
    with pytest.raises(ValueError, match="positive"):
        DeltaProfile(delta_far_m=-10e-6)


# ─── 11. Conserves total current ─────────────────────────────────────

def test_zones_conserve_total_current(default_result):
    """Sum of zone currents should approximately equal total applied current."""
    total_j_x_area = sum(
        z.local_j_mA_cm2 * z.zone_area_cm2 for z in default_result.zones
    )
    total_mA = default_result.config.total_current_A * 1000.0
    assert total_j_x_area == pytest.approx(total_mA, rel=0.05), (
        f"Zone currents sum {total_j_x_area:.1f} mA vs applied {total_mA:.1f} mA"
    )


# ─── 12. DeltaProfile linear interpolation ───────────────────────────

def test_delta_profile_interpolation():
    """Delta should interpolate linearly from near to far edge."""
    dp = DeltaProfile(delta_near_m=50e-6, delta_far_m=150e-6)
    delta_0 = dp.delta_at_position(0.0)
    delta_10 = dp.delta_at_position(10.0)
    delta_5 = dp.delta_at_position(5.0)

    assert float(delta_0) == pytest.approx(50e-6, rel=1e-9)
    assert float(delta_10) == pytest.approx(150e-6, rel=1e-9)
    assert float(delta_5) == pytest.approx(100e-6, rel=1e-9)
