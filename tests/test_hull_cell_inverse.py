"""Tests for the inverse Hull-cell thickness-profile → FE(j) analysis."""

import numpy as np
import pandas as pd
import pytest

from models.hull_cell import HullCellGeometry, hull_current_distribution
from models.hull_cell_inverse import (
    IRON_DENSITY_G_CM3,
    LogitFeFit,
    aggregate_fe_curve,
    analyze_hull_panel,
    faradaic_efficiency_from_thickness,
    fit_fe_vs_j,
    implied_panel_faradaic_efficiency,
    logit_fe,
    mass_closure,
    synthesize_thickness_profile,
    thickness_from_faraday,
    thickness_to_local_faradaic_efficiency,
)

# Day-1 panel geometry (FIRST_LAB_DAY.md §3): 10 × 5 cm, 1.5 → 9 cm gap, 2 A.
GEOMETRY = HullCellGeometry(
    panel_length_cm=10.0,
    panel_width_cm=5.0,
    near_edge_gap_cm=1.5,
    far_edge_gap_cm=9.0,
)
N_SEGMENTS = 10


def _distribution(total_current_A: float = 2.0) -> pd.DataFrame:
    return hull_current_distribution(GEOMETRY, total_current_A, n_segments=N_SEGMENTS)


def _known_profile(total_current_A: float = 2.0, duration_s: float = 3600.0):
    """A noiseless profile from a known logit FE model (85 % @ 100 mA/cm²)."""
    a = np.log(0.85 / 0.15) - (-0.35) * np.log(100.0)
    distribution = _distribution(total_current_A)
    profile = synthesize_thickness_profile(
        distribution, duration_s, a, -0.35, noise_sigma_um=0.0
    )
    return distribution, profile, a, -0.35


def test_faraday_round_trip_units():
    j = np.array([50.0, 100.0, 200.0, 400.0])
    fe = np.array([0.9, 0.85, 0.75, 0.6])
    t = 3600.0
    h = thickness_from_faraday(j, fe, t)
    fe_back = faradaic_efficiency_from_thickness(j, h, t)
    assert np.allclose(fe_back, fe, rtol=1e-9)
    # Sanity anchor: 100 mA/cm² at 100 % FE for 1 h ≈ 132 µm of Fe.
    assert thickness_from_faraday(100.0, 1.0, 3600.0)[0] == pytest.approx(132.3, rel=1e-3)
    # Sanity anchor: 100 mA/cm² at 85 % FE for 15 min ≈ 28 µm (README example).
    assert thickness_from_faraday(100.0, 0.85, 900.0)[0] == pytest.approx(28.1, rel=1e-2)


def test_noiseless_inverse_recovers_known_fe_profile():
    distribution, profile, a_true, b_true = _known_profile()
    local = thickness_to_local_faradaic_efficiency(
        distribution, profile["measured_thickness_um"], 3600.0
    )
    # Recovered strip FE must match the truth model (noiseless profile).
    assert np.allclose(
        local["apparent_faradaic_efficiency"],
        logit_fe(distribution["current_density_mA_cm2"], a_true, b_true),
        rtol=1e-9,
    )
    assert not local["fe_qa_flag"].isin({"above_100", "zero_deposit"}).any()
    # The deposition current integrates to FE_panel × I_applied: only the
    # iron-making fraction of the applied current appears in the deposit.
    dep_total = float(
        (local["deposition_current_density_mA_cm2"] / 1000.0
         * local["segment_area_cm2"]).sum()
    )
    assert dep_total == pytest.approx(
        2.0 * implied_panel_faradaic_efficiency(local), rel=1e-9
    )


def test_logit_fit_recovers_truth_within_tolerance():
    distribution, profile, a_true, b_true = _known_profile()
    local = thickness_to_local_faradaic_efficiency(
        distribution, profile["measured_thickness_um"], 3600.0
    )
    fit = fit_fe_vs_j(local)
    assert isinstance(fit, LogitFeFit)
    assert fit.a == pytest.approx(a_true, rel=1e-3)
    assert fit.b == pytest.approx(b_true, rel=1e-3)
    assert fit.r_squared is not None and fit.r_squared > 0.999
    assert fit.fe_at_reference == pytest.approx(0.85, rel=1e-3)


def test_noisy_single_panel_fit_is_sane():
    distribution = _distribution()
    a = np.log(0.85 / 0.15) - (-0.35) * np.log(100.0)
    rng = np.random.default_rng(7)
    profile = synthesize_thickness_profile(
        distribution, 3600.0, a, -0.35, noise_sigma_um=1.0, rng=rng
    )
    local = thickness_to_local_faradaic_efficiency(
        distribution, profile["measured_thickness_um"], 3600.0,
        thickness_uncertainty_um=1.0,
    )
    fit = fit_fe_vs_j(local)
    # FE declines with j (HER takes a larger share) and the reference anchor
    # stays close to truth despite noise.
    assert fit.b < 0.0
    assert fit.b == pytest.approx(-0.35, abs=0.15)
    assert fit.fe_at_reference == pytest.approx(0.85, abs=0.02)
    # Uncertainty must be present and positive where there is deposit.
    assert np.all(local["fe_uncertainty"].dropna() > 0)


def test_monte_carlo_fit_is_unbiased_and_documents_slope_sensitivity():
    """Statistical round trip: FE-space NLS is unbiased in expectation.

    Also locks in the protocol finding: at point-micrometer noise (2 µm) a
    single 10-strip panel pins FE at the reference j but not the slope; at
    profilometry-grade noise (0.5 µm) the slope is recovered too.
    """
    distribution = _distribution()
    a = np.log(0.85 / 0.15) - (-0.35) * np.log(100.0)

    def recover(noise_um: float, n_seeds: int):
        slopes, fe_refs = [], []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            profile = synthesize_thickness_profile(
                distribution, 3600.0, a, -0.35,
                noise_sigma_um=noise_um, rng=rng,
            )
            local = thickness_to_local_faradaic_efficiency(
                distribution, profile["measured_thickness_um"], 3600.0,
                thickness_uncertainty_um=noise_um,
            )
            fit = fit_fe_vs_j(local)
            slopes.append(fit.b)
            fe_refs.append(fit.fe_at_reference)
        return np.array(slopes), np.array(fe_refs)

    slopes_fine, fe_fine = recover(0.5, 300)
    assert np.mean(slopes_fine) == pytest.approx(-0.35, abs=0.05)
    assert np.std(slopes_fine) < 0.10
    assert np.mean(fe_fine) == pytest.approx(0.85, abs=0.01)

    slopes_coarse, fe_coarse = recover(2.0, 300)
    # The reference-j FE stays robust at realistic noise …
    assert np.mean(fe_coarse) == pytest.approx(0.85, abs=0.02)
    # … while the slope is not resolvable from one coarse panel (documented
    # protocol limitation, not a bug: use profilometry/more strips/panels).
    assert np.std(slopes_coarse) > 0.10


def test_current_weighted_mean_equals_gravimetric_fe():
    distribution, profile, _, _ = _known_profile(duration_s=1800.0)
    measured = profile["measured_thickness_um"].to_numpy(float)
    # 'Weighing' the same panel: integrate the true profile mass.
    gravimetric_g = float(
        (measured * distribution["segment_area_cm2"].to_numpy(float)).sum()
    ) * IRON_DENSITY_G_CM3 * 1e-4
    local = thickness_to_local_faradaic_efficiency(distribution, measured, 1800.0)
    implied = implied_panel_faradaic_efficiency(local)

    charge_C = 2.0 * 1800.0
    theoretical_mass_g = charge_C * 55.845 / (2 * 96485.3321)
    gravimetric_fe = gravimetric_g / theoretical_mass_g
    assert implied == pytest.approx(gravimetric_fe, rel=1e-9)
    # And the closure check must pass on a consistent profile.
    closure = mass_closure(distribution, measured, gravimetric_g)
    assert closure["mass_balanced"] is True
    assert closure["closure_ratio"] == pytest.approx(1.0, rel=1e-9)


def test_mass_closure_flags_inconsistent_profile():
    distribution, profile, _, _ = _known_profile()
    measured = profile["measured_thickness_um"].to_numpy(float)
    gravimetric_g = float(
        (measured * distribution["segment_area_cm2"].to_numpy(float)).sum()
    ) * IRON_DENSITY_G_CM3 * 1e-4
    # Weighing 30 % light vs the profile (e.g. retained-salt error or a thin
    # deposit that does not match the profilometry) must fail the closure check.
    closure = mass_closure(distribution, measured, 0.7 * gravimetric_g)
    assert closure["mass_balanced"] is False
    assert closure["closure_ratio"] == pytest.approx(1.0 / 0.7, rel=1e-9)
    # Tight tolerance fails a 10 % mismatch.
    closure_tight = mass_closure(
        distribution, measured, 0.9 * gravimetric_g, tolerance=0.05
    )
    assert closure_tight["mass_balanced"] is False
    # Loose tolerance accepts it.
    closure_loose = mass_closure(
        distribution, measured, 0.9 * gravimetric_g, tolerance=0.15
    )
    assert closure_loose["mass_balanced"] is True


def test_aggregate_fe_curve_is_current_weighted_and_partitions():
    distribution, profile, _, _ = _known_profile()
    local = thickness_to_local_faradaic_efficiency(
        distribution, profile["measured_thickness_um"], 3600.0
    )
    binned = aggregate_fe_curve(local, n_bins=4)
    assert len(binned) <= 4
    # Bins are ordered by increasing j and the area/current fractions sum to 1.
    assert (binned["j_min_mA_cm2"].diff().dropna() > 0).all()
    assert binned["area_fraction"].sum() == pytest.approx(1.0, rel=1e-9)
    assert binned["current_fraction"].sum() == pytest.approx(1.0, rel=1e-9)
    # Current-weighted FE declines with j for the truth model.
    assert binned["fe_current_weighted_mean"].iloc[0] > binned[
        "fe_current_weighted_mean"
    ].iloc[-1]
    # Consistency: Σ bin FE_i·I_i / Σ I_i equals the implied panel FE.
    overall = float(
        (binned["fe_current_weighted_mean"] * binned["current_fraction"]).sum()
    )
    assert overall == pytest.approx(implied_panel_faradaic_efficiency(local), rel=1e-9)


def test_qa_flags_above_100_retained_not_clipped():
    distribution = _distribution()
    # A thickness profile 5 % thicker than 100 % FE predicts at every strip.
    h = thickness_from_faraday(
        distribution["current_density_mA_cm2"], 1.05, 3600.0
    )
    local = thickness_to_local_faradaic_efficiency(distribution, h, 3600.0)
    assert np.all(local["apparent_faradaic_efficiency"] > 1.0)
    assert set(local["fe_qa_flag"]) == {"above_100"}
    # The logit fit must exclude them and degrade gracefully, not crash.
    fit = fit_fe_vs_j(local)
    assert fit.n_points == 0 or np.isnan(fit.a)


def test_zero_deposit_strips_flagged_and_fit_excludes_them():
    distribution = _distribution()
    h = np.zeros(N_SEGMENTS)
    h[3:] = 20.0
    local = thickness_to_local_faradaic_efficiency(distribution, h, 3600.0)
    assert local["fe_qa_flag"].iloc[0] == "zero_deposit"
    assert local["apparent_faradaic_efficiency"].iloc[0] == 0.0
    assert local["deposit_mass_mg_cm2"].iloc[0] == 0.0
    fit = fit_fe_vs_j(local)
    assert fit.n_points == N_SEGMENTS - 3


def test_analyze_hull_panel_bundles_everything():
    distribution, profile, _, _ = _known_profile(duration_s=1800.0)
    measured = profile["measured_thickness_um"].to_numpy(float)
    gravimetric_g = float(
        (measured * distribution["segment_area_cm2"].to_numpy(float)).sum()
    ) * IRON_DENSITY_G_CM3 * 1e-4
    result = analyze_hull_panel(
        distribution, measured, 1800.0, gravimetric_mass_gain_g=gravimetric_g
    )
    assert set(result) == {
        "local_faradaic_efficiency",
        "binned_fe_curve",
        "logit_fit",
        "implied_panel_faradaic_efficiency",
        "mass_closure",
    }
    assert result["mass_closure"]["mass_balanced"] is True


def test_validation_errors():
    distribution = _distribution()
    h = np.full(N_SEGMENTS, 20.0)
    with pytest.raises(ValueError):
        thickness_to_local_faradaic_efficiency(distribution, h[:-1], 3600.0)
    with pytest.raises(ValueError):
        thickness_to_local_faradaic_efficiency(distribution, h, -1.0)
    with pytest.raises(ValueError):
        thickness_to_local_faradaic_efficiency(distribution, np.full(N_SEGMENTS, -1.0), 3600.0)
    with pytest.raises(ValueError):
        thickness_to_local_faradaic_efficiency(
            distribution, h, 3600.0, thickness_uncertainty_um=np.full(N_SEGMENTS - 1, 1.0)
        )
    with pytest.raises(ValueError):
        mass_closure(distribution, h, 0.0)
    with pytest.raises(ValueError):
        aggregate_fe_curve(pd.DataFrame({"current_density_mA_cm2": [1.0, 2.0]}))
    with pytest.raises(ValueError):
        logit_fe(np.array([0.0, 1.0]), 1.0, -0.3)
    with pytest.raises(ValueError):
        thickness_from_faraday(100.0, 0.85, 0.0)
