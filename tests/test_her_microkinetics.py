"""Assertions for the DFT-anchored HER microkinetics consistency check."""
from math import isfinite, log10
import pytest
from models.her_microkinetics import (
    DG_HSTAR_FE110_J,
    DG_HSTAR_RANGE_J,
    HeyrovskyBranch,
    anchor_heyrovsky,
    consistency_report,
    hydrogen_coverage,
    microkinetic_tafel_slope_V,
)
from models.kinetics import DepositionKinetics


def test_coverage_pinned_high_on_fe_in_the_operating_window():
    """ΔG_H* ≈ −0.40 eV → θ_H ≈ 1 for all cathodic U (50 °C)."""
    for U in (0.0, -0.1, -0.3, -0.5):
        assert hydrogen_coverage(U, 323.15) > 0.99


def test_coverage_moves_with_dg_hstar_sign():
    """Weakly-binding surface (ΔG>0): coverage must drop (probe off U=−ΔG)."""
    assert hydrogen_coverage(-0.1, 298.15, dg_hstar_J=0.3 * 96485.0) < 0.1
    assert hydrogen_coverage(0.3, 298.15, dg_hstar_J=0.3 * 96485.0) < 1e-3


def test_microkinetic_slope_is_the_alpha_half_high_coverage_value():
    """b = 2.303RT/(αF): ~118 mV/dec at 25 °C, ~131 at 60 °C."""
    assert microkinetic_tafel_slope_V(298.15) == pytest.approx(0.1183, abs=2e-3)
    assert microkinetic_tafel_slope_V(333.15) == pytest.approx(0.1322, abs=2e-3)


def test_anchored_branch_reproduces_anchor_point_exactly():
    """k_Hey is solved, so the anchor state must be exact by construction."""
    branch = anchor_heyrovsky(
        anchor_current_A_m2=3.0, pH_ref=2.0, eta_ref_V=0.2, T_ref_K=323.15
    )
    i = branch.current(U_vs_RHE_V=-0.2, a_h=1e-2, T_K=323.15)
    assert i == pytest.approx(3.0, rel=1e-12)


def test_branch_scales_linearly_with_proton_activity_at_fixed_U():
    branch = HeyrovskyBranch(k_hey=1.0)
    i2 = branch.current(-0.3, 1e-2, 323.15)
    i3 = branch.current(-0.3, 1e-3, 323.15)
    assert i2 / i3 == pytest.approx(10.0, rel=1e-12)


def test_branch_is_tafel_in_eta_when_coverage_saturated():
    """At θ≈1 a decade in current costs b=2.303RT/(αF)."""
    branch = HeyrovskyBranch(k_hey=1.0)
    i1 = branch.current(-0.20, 1e-2, 298.15)
    i2 = branch.current(-0.31827, 1e-2, 298.15)  # +0.11827 V ≈ one b
    assert log10(i2 / i1) == pytest.approx(
        0.11827 / microkinetic_tafel_slope_V(298.15), abs=0.02
    )


def test_consistency_report_supports_the_empirical_branch():
    """Slopes within ~20% at the anchor; off-anchor ratios stay sane."""
    k = DepositionKinetics(pH=2.0, temperature_C=50.0)
    rep = consistency_report(k)
    assert rep["theta_H_operating"] > 0.99
    assert 0.7 < rep["slope_ratio"] < 1.0  # 128 vs 140 mV/dec at 50 °C
    for key in ("i_ratio_25C", "i_ratio_70C"):
        assert isfinite(rep[key]) and rep[key] > 0.0
    assert rep["flag"] == "unvalidated (L0)"


def test_dg_hstar_screening_range_brackets_the_central_value():
    lo, hi = DG_HSTAR_RANGE_J
    assert lo < DG_HSTAR_FE110_J < hi
    # Range stays inside the strong-binding (Heyrovský-RDS) regime.
    assert hi < 0.0
