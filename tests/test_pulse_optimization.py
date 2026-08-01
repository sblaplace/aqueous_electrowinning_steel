"""Tests for pulsed electrodeposition Pareto optimization.

Acceptance criteria from task t_d631a54b:
- Pareto front is non-empty
- All Pareto solutions are truly non-dominated
- DC (duty=1.0) is never on the Pareto front for grain size
- Higher j_peak gives finer grain (monotonic)
- Recommended operating points are within Pareto set
- Grid covers 6×6×6×2×3 = 1296 combinations
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.slow

from models.pulse_optimization import (
    PulseOptimizationSweep,
    is_non_dominated,
    _frequency_grain_factor,
    _frequency_ce_factor,
    J_PEAK_VALUES,
)
from models.mechanical_properties import estimate_grain_size_um


# ─── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def small_sweep() -> pd.DataFrame:
    """Run a reduced sweep (single mechanism, fewer frequencies) for fast tests."""
    sweep = PulseOptimizationSweep(
        mechanism_values=["hydroxide_suppression"],
        frequency_values=[10.0, 100.0],
    )
    return sweep.run_full_sweep()


@pytest.fixture(scope="module")
def small_fronts(small_sweep: pd.DataFrame) -> dict:
    """Pareto fronts from the small sweep."""
    sweep = PulseOptimizationSweep()
    return sweep.compute_pareto_fronts(small_sweep)


@pytest.fixture(scope="module")
def full_grid_sweep() -> pd.DataFrame:
    """Full 1296-point sweep (module-scoped, takes ~2-5 min)."""
    sweep = PulseOptimizationSweep()
    assert sweep.grid_size() == 6 * 6 * 6 * 2 * 3  # 1296
    return sweep.run_full_sweep()


# ─── Test: grid size ─────────────────────────────────────────────────────


def test_grid_size_1296():
    """Parameter grid covers 6×6×6×2×3 = 1296 combinations."""
    sweep = PulseOptimizationSweep()
    assert sweep.grid_size() == 1296


# ─── Test: Pareto front non-empty ───────────────────────────────────────


def test_pareto_fronts_non_empty(small_fronts: dict):
    """All three Pareto fronts must be non-empty."""
    for name, fdf in small_fronts.items():
        assert len(fdf) > 0, f"Pareto front '{name}' is empty"


# ─── Test: non-dominance ────────────────────────────────────────────────


def test_pareto_solutions_are_non_dominated(small_fronts: dict):
    """Every point on a Pareto front must be truly non-dominated."""
    for name, fdf in small_fronts.items():
        if name == "grain_vs_efficiency":
            obj = fdf[["grain_size_um", "current_efficiency_pct"]].values
            maximize = np.array([False, True])
        elif name == "strength_vs_energy":
            obj = fdf[["energy_cost_USD_per_kg", "yield_strength_MPa"]].values
            maximize = np.array([False, True])
        elif name == "carbon_vs_grain":
            obj = fdf[["grain_size_um", "carbon_wt_pct"]].values
            maximize = np.array([False, True])
        else:
            continue

        mask = is_non_dominated(obj, maximize=maximize)
        assert mask.all(), (
            f"Front '{name}' contains dominated points: "
            f"{fdf.index[~mask].tolist()}"
        )


# ─── Test: DC not on grain Pareto front ─────────────────────────────────


def test_dc_not_on_grain_pareto_front(full_grid_sweep: pd.DataFrame):
    """DC (duty=1.0) should not appear on the grain-vs-efficiency Pareto front.

    Pulsed operation (PE/PRE) always produces finer grains than DC at the
    same average current density, so DC should be dominated.
    """
    sweep = PulseOptimizationSweep()
    fronts = sweep.compute_pareto_fronts(full_grid_sweep)
    grain_front = fronts["grain_vs_efficiency"]

    dc_on_front = grain_front[grain_front["duty_cycle"] == 1.0]
    assert len(dc_on_front) == 0, (
        f"DC points found on grain-vs-efficiency Pareto front: "
        f"{dc_on_front[['j_peak_mA_cm2', 'duty_cycle', 'waveform']].to_dict('records')}"
    )


# ─── Test: monotonic grain refinement with j_peak ───────────────────────


def test_higher_j_peak_finer_grain():
    """Higher j_peak gives finer grain at fixed duty and waveform (monotonic)."""
    for waveform in ["pe", "pre"]:
        grains = []
        for jp in J_PEAK_VALUES:
            g = estimate_grain_size_um(
                j_avg_mA_cm2=jp * 0.5,
                j_peak_mA_cm2=jp,
                duty_cycle=0.5,
                waveform=waveform,
            )
            grains.append(g)

        for i in range(len(grains) - 1):
            assert grains[i] >= grains[i + 1], (
                f"Grain size not monotonically decreasing with j_peak for {waveform}: "
                f"j_peak={J_PEAK_VALUES[i]} → {grains[i]:.4f} µm, "
                f"j_peak={J_PEAK_VALUES[i+1]} → {grains[i+1]:.4f} µm"
            )


# ─── Test: recommendations within Pareto set ────────────────────────────


def test_recommendations_within_pareto_set(small_sweep: pd.DataFrame, small_fronts: dict):
    """Recommended operating points must correspond to points in the Pareto set."""
    sweep = PulseOptimizationSweep()
    recs = sweep.recommend_operating_points(small_fronts)

    assert len(recs) > 0, "No operating points recommended"

    # Build index of all Pareto front members across all fronts
    all_front_indices = set()
    for fdf in small_fronts.values():
        all_front_indices.update(fdf.index.tolist())

    for rec in recs:
        # Find matching row in sweep
        mask = (
            (small_sweep["j_peak_mA_cm2"] == rec["j_peak_mA_cm2"])
            & (small_sweep["duty_cycle"] == rec["duty_cycle"])
            & (small_sweep["frequency_Hz"] == rec["frequency_Hz"])
            & (small_sweep["waveform"] == rec["waveform"])
            & (small_sweep["mechanism"] == rec["mechanism"])
        )
        matching = small_sweep.index[mask].tolist()
        assert len(matching) > 0, (
            f"Recommended point {rec['j_peak_mA_cm2']}/{rec['duty_cycle']}/"
            f"{rec['frequency_Hz']}/{rec['waveform']}/{rec['mechanism']} "
            f"not found in sweep results"
        )
        on_front = any(idx in all_front_indices for idx in matching)
        assert on_front, (
            f"Recommended point not on any Pareto front: "
            f"j_peak={rec['j_peak_mA_cm2']}, duty={rec['duty_cycle']}"
        )


# ─── Test: is_non_dominated basic correctness ──────────────────────────


def test_is_non_dominated_simple():
    """Verify is_non_dominated with a known 2D example."""
    # Points: (1,5), (2,3), (3,2), (4,1), (2,4) — all minimizing
    obj = np.array([[1, 5], [2, 3], [3, 2], [4, 1], [2, 4]])
    mask = is_non_dominated(obj)

    # (2,4) is dominated by (2,3) — same first obj, worse second
    # (1,5), (2,3), (3,2), (4,1) are all non-dominated (trade-off curve)
    expected = np.array([True, True, True, True, False])
    np.testing.assert_array_equal(mask, expected)


def test_is_non_dominated_with_maximize():
    """Verify is_non_dominated with mixed min/max objectives."""
    # Minimize x, maximize y
    obj = np.array([[1, 5], [2, 3], [1, 3], [3, 1]])
    maximize = np.array([False, True])
    mask = is_non_dominated(obj, maximize=maximize)

    # (1,5) non-dominated; (2,3) dominated by (1,3); (1,3) non-dominated; (3,1) dominated by (1,1)?
    # Actually: (1,5) > (1,3) on y → (1,5) dominates (1,3) if we maximize y
    # (2,3): dominated by (1,3) on x? No, (1,3) has x=1 < 2 (better for min), y=3 = 3 (same)
    # So (2,3) is dominated by (1,3)
    # (3,1): dominated by (1,1)? No such point. (1,5) has x=1<3, y=5>1 → dominates (3,1)
    # (1,3): x=1 (best), y=3 — not dominated by (1,5) since y=3<5 → (1,5) dominates (1,3) on y
    # So: (1,5) non-dominated; (1,3) dominated by (1,5); (2,3) dominated by (1,3) which is dominated;
    #     (3,1) dominated by (1,5)
    expected = np.array([True, False, False, False])
    np.testing.assert_array_equal(mask, expected)


# ─── Test: frequency correction factors ─────────────────────────────────


def test_frequency_grain_factor_decreases_with_freq():
    """Higher frequency should give factor < 1 (finer grain)."""
    f1 = _frequency_grain_factor(1.0)
    f10 = _frequency_grain_factor(10.0)
    f100 = _frequency_grain_factor(100.0)
    f1000 = _frequency_grain_factor(1000.0)

    assert f1 > f10 > f100 > f1000, (
        f"Frequency grain factor not monotonically decreasing: "
        f"f(1)={f1:.4f}, f(10)={f10:.4f}, f(100)={f100:.4f}, f(1000)={f1000:.4f}"
    )
    # Reference at 10 Hz should be 1.0
    assert abs(f10 - 1.0) < 1e-6


def test_frequency_ce_factor_near_one():
    """CE frequency correction should be weak (within ±5%)."""
    for f in [1.0, 10.0, 100.0, 1000.0]:
        factor = _frequency_ce_factor(f)
        assert 0.95 <= factor <= 1.05, (
            f"CE frequency factor at {f} Hz = {factor:.4f} — expected near 1.0"
        )
