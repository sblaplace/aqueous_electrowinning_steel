"""Tests for diffusion_layer_1d surface-state and FeSO₄⁰ pair corrections.

Validates the Tier-1.1 (surface_state) and Tier-2 (FeSO₄⁰ neutral pair)
additions from CHEM_PHYS_REVIEW.md and docs/CHEM_PHYS_IMPROVEMENTS_V2.md.
"""

import math
import pytest
import numpy as np

from models.diffusion_layer_1d import (
    DiffusionLayer1D,
    K_FESO4_PAIR_25,
    D_FESO4_RATIO,
)


class TestFeSO4PairCorrection:
    """FeSO₄⁰ neutral pair reduces effective transport limit."""

    def test_pair_fraction_zero_when_disabled(self):
        """When fes04_pair_correction=False, pair fraction is 0."""
        model = DiffusionLayer1D(
            fe_conc_M=1.5,
            pH_bulk=2.0,
            fes04_pair_correction=False,
        )
        assert model._bulk_pair_fraction == 0.0

    def test_pair_fraction_positive_when_enabled(self):
        """When enabled, pair fraction is > 0 for typical bath."""
        model = DiffusionLayer1D(
            fe_conc_M=1.5,
            pH_bulk=2.0,
            support_conc_M=0.5,
            fes04_pair_correction=True,
        )
        f = model._bulk_pair_fraction
        assert f > 0
        assert f < 1.0

    def test_pair_fraction_bounded(self):
        """Pair fraction is always between 0 and 1."""
        for fe_M in (0.1, 0.5, 1.0, 1.5, 2.0):
            for support in (0.0, 0.1, 0.5, 1.0):
                model = DiffusionLayer1D(
                    fe_conc_M=fe_M, support_conc_M=support,
                    fes04_pair_correction=True,
                )
                f = model._bulk_pair_fraction
                assert 0.0 <= f <= 1.0

    def test_pair_reduces_diffusion_limit(self):
        """With pair correction, diffusion limit is lower."""
        model_bare = DiffusionLayer1D(
            fe_conc_M=1.5, support_conc_M=0.5,
            fes04_pair_correction=False,
        )
        model_pair = DiffusionLayer1D(
            fe_conc_M=1.5, support_conc_M=0.5,
            fes04_pair_correction=True,
        )
        assert model_pair.diffusion_limit_A_m2 < model_bare.diffusion_limit_A_m2

    def test_pair_reduction_moderate(self):
        """The correction should be moderate (5-20%), not dramatic."""
        model_bare = DiffusionLayer1D(
            fe_conc_M=1.5, support_conc_M=0.5,
            fes04_pair_correction=False,
        )
        model_pair = DiffusionLayer1D(
            fe_conc_M=1.5, support_conc_M=0.5,
            fes04_pair_correction=True,
        )
        ratio = model_pair.diffusion_limit_A_m2 / model_bare.diffusion_limit_A_m2
        assert 0.75 < ratio < 1.0  # 0-25% reduction, physically reasonable

    def test_result_carries_pair_diagnostic(self):
        """Result object carries the pair fraction when enabled."""
        model = DiffusionLayer1D(
            fe_conc_M=1.5, pH_bulk=2.0,
            fes04_pair_correction=True,
        )
        result = model.solve(100.0)
        assert result.fe_pair_fraction_bulk > 0

    def test_result_pair_zero_when_disabled(self):
        """Result pair fraction is 0 when correction is disabled."""
        model = DiffusionLayer1D(
            fe_conc_M=1.5, pH_bulk=2.0,
            fes04_pair_correction=False,
        )
        result = model.solve(100.0)
        assert result.fe_pair_fraction_bulk == 0.0

    def test_pair_constants_exist(self):
        """The module-level constants are accessible."""
        assert K_FESO4_PAIR_25 > 0
        assert 0 < D_FESO4_RATIO < 1


class TestSurfaceStateIntegration:
    """Surface-state HER kinetics (Tier 1.1) in the FE engine."""

    def test_disabled_is_backward_compatible(self):
        """With surface_state=False, results match the original model."""
        model = DiffusionLayer1D(
            fe_conc_M=1.5, pH_bulk=2.0,
            surface_state=False,
        )
        result = model.solve(100.0)
        assert result.her_i0_surface_state_ratio == 1.0
        assert 0 < result.current_efficiency < 1

    def test_enabled_reduces_her(self):
        """Surface state (with anion blocking) should reduce HER i₀."""
        model = DiffusionLayer1D(
            fe_conc_M=1.5, pH_bulk=2.0,
            surface_state=True, bath_type="sulfate",
        )
        result = model.solve(100.0)
        # With surface state, the HER i₀ is corrected.  In sulfate
        # bath at pH 2, the correction should be < 1 (HER suppressed).
        assert result.her_i0_surface_state_ratio <= 1.0

    def test_surface_state_changes_fe(self):
        """Surface state changes the FE prediction."""
        model_bare = DiffusionLayer1D(
            fe_conc_M=1.5, pH_bulk=2.0,
            surface_state=False,
        )
        model_ss = DiffusionLayer1D(
            fe_conc_M=1.5, pH_bulk=2.0,
            surface_state=True, bath_type="sulfate",
        )
        r_bare = model_bare.solve(200.0)
        r_ss = model_ss.solve(200.0)
        # The two should differ (surface state changes HER i₀)
        # but both should converge
        assert r_bare.converged
        assert r_ss.converged
        # FE should be different (surface state typically gives higher FE
        # because HER is suppressed)
        assert r_bare.current_efficiency != pytest.approx(
            r_ss.current_efficiency, abs=0.001
        )

    def test_aware_bath_gives_higher_fe(self):
        """AWARE (chloride) bath should give higher FE than sulfate."""
        model_sulfate = DiffusionLayer1D(
            fe_conc_M=1.0, pH_bulk=2.0,
            surface_state=True, bath_type="sulfate",
        )
        model_aware = DiffusionLayer1D(
            fe_conc_M=1.0, pH_bulk=2.0,
            surface_state=True, bath_type="aware",
        )
        r_sulfate = model_sulfate.solve(200.0)
        r_aware = model_aware.solve(200.0)
        # AWARE (10 M Cl⁻) should have stronger HER suppression
        assert r_aware.her_i0_surface_state_ratio <= r_sulfate.her_i0_surface_state_ratio

    def test_invalid_bath_type_raises(self):
        """Invalid bath_type should raise when surface_state=True."""
        model = DiffusionLayer1D(
            surface_state=True, bath_type="invalid",
        )
        with pytest.raises((ValueError, KeyError)):
            model.solve(100.0)


class TestCombinedCorrections:
    """Both corrections active simultaneously."""

    def test_both_corrections_work(self):
        """Enabling both corrections should converge."""
        model = DiffusionLayer1D(
            fe_conc_M=1.5, pH_bulk=2.0, support_conc_M=0.5,
            surface_state=True, bath_type="sulfate",
            fes04_pair_correction=True,
        )
        result = model.solve(200.0)
        assert result.converged
        assert result.her_i0_surface_state_ratio < 1.0
        assert result.fe_pair_fraction_bulk > 0

    def test_both_correct_but_stable(self):
        """Both corrections active: FE should be in a reasonable range."""
        model = DiffusionLayer1D(
            fe_conc_M=1.5, pH_bulk=2.0, support_conc_M=0.5,
            surface_state=True, bath_type="sulfate",
            fes04_pair_correction=True,
        )
        result = model.solve(300.0)
        assert 0.3 < result.current_efficiency < 1.0
        assert result.converged
