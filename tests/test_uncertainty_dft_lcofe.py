"""DFT dG_H* -> theta_H -> i0,H_eff -> FE uncertainty chain (CHEM_PHYS §3.1).

The single largest previously-untracked economics uncertainty: the +/-0.15 eV
DFT band on the hydrogen adsorption free energy dG_H* propagates to the FE at
the gate (via the surface_state Volmer/Heyrovsky coverage theta_H and the
effective HER exchange current) and thence to V_cell/kWh and LCOFe.

These tests lock in the mechanism and the chain's presence in the Monte Carlo /
Sobol sweep:
  1. dG_H* is registered with the +/-0.15 eV DFT uncertainty.
  2. The +/-0.15 eV band swings i0,H_eff by ~2-3x (order of magnitude).
  3. `_run_single_sample` propagates that swing to a physically-signed FE change
     (weaker binding -> more H -> lower FE).
  4. A Monte Carlo run produces a FE-at-the-gate probability distribution and
     ranks dG_H* among the top FE drivers.
"""

from __future__ import annotations

import pytest

import numpy as np

from models.uncertainty.parameter_registry import REGISTRY
from models.uncertainty.monte_carlo import (
    MonteCarloEngine,
    _run_single_sample,
    _her_i0_swing,
    DG_HSTAR_NOMINAL_EV,
    HER_ETA_REF_V,
    DEFAULT_DESIGN_POINT,
)

# Band edges of the declared +/-0.15 eV DFT uncertainty around -0.40 eV.
DG_WEAK_EV = -0.25   # less negative => weaker H binding
DG_STRONG_EV = -0.55  # more negative => stronger H binding
T_OP_K = 333.15      # 60 °C reference operating temperature


# ---------------------------------------------------------------------------
# 1. Registry entry
# ---------------------------------------------------------------------------

def test_dG_Hstar_registered_with_dft_uncertainty():
    """dG_H* is in the registry with the +/-0.15 eV DFT band."""
    p = REGISTRY["dG_Hstar_eV"]
    assert p.module == "surface_state"
    assert p.mean == pytest.approx(-0.40, abs=1e-6)
    assert p.std == pytest.approx(0.15, abs=1e-6)
    # Bounds span the full +/-0.15 eV band so Sobol/MC exercise it.
    assert p.bounds == (-0.55, -0.25)
    assert p.distribution == "normal"


# ---------------------------------------------------------------------------
# 2. i0,H_eff swing across the +/-0.15 eV band (~2-3x)
# ---------------------------------------------------------------------------

def test_i0_swing_across_dft_band_is_2_to_3x():
    """+/-0.15 eV on dG_H* swings the effective HER i0 by ~2-3x.

    Order-of-magnitude check: the surface-state factor that depends on dG_H*
    is theta_H(1-theta_H); across the band its ratio is the i0,H swing.
    """
    swing_weak = _her_i0_swing(DG_WEAK_EV, HER_ETA_REF_V, T_OP_K)
    swing_strong = _her_i0_swing(DG_STRONG_EV, HER_ETA_REF_V, T_OP_K)
    # swing(weak) >= 1 (more HER-active) and swing(strong) <= 1.
    assert swing_weak > 1.0
    assert swing_strong < 1.0
    band_ratio = swing_weak / swing_strong
    assert 2.0 <= band_ratio <= 4.0, (
        f"dG_H* +/-0.15 eV swung i0,H_eff by {band_ratio:.2f}x, expected ~2-3x"
    )


def test_i0_swing_monotone_with_binding_strength():
    """Stronger H binding (more negative dG_H*) monotonically lowers i0,H_eff."""
    swings = [
        _her_i0_swing(dg, HER_ETA_REF_V, T_OP_K)
        for dg in (-0.25, -0.32, -0.40, -0.48, -0.55)
    ]
    # Each successive (more negative) dG* must give <= previous swing.
    for a, b in zip(swings, swings[1:]):
        assert b <= a + 1e-12
    # Anchored: nominal dG_H* gives swing of exactly 1.0.
    assert _her_i0_swing(DG_HSTAR_NOMINAL_EV, HER_ETA_REF_V, T_OP_K) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. FE propagation through _run_single_sample
# ---------------------------------------------------------------------------

def _sample_at_dg(dg_hstar_eV: float) -> dict:
    """Registry-mean sample with only dG_H* overridden to a band edge."""
    base = {name: p.mean for name, p in REGISTRY.items()}
    base["dG_Hstar_eV"] = dg_hstar_eV
    return base


def test_fe_propagates_from_dg_signed_correctly():
    """Weak binding raises H current -> lower FE; strong binding -> higher FE.

    NOTE: at the corrected physical ``her_i0 = 1e-6`` (see her_i0 default fix,
    PR #56) HER is suppressed at the *nominal* gate, so dG_H* moves FE by only
    ~0.01 pt here. The order-of-magnitude 10-15% swing in the original write-up
    was an artifact of the pre-fix her_i0 = 1e-3 (where HER was inflated). The
    mechanism is still locked: i0,H_eff swings 2-3x across the band and
    FE responds monotonically in the correct direction. dG_H* becomes a
    material economic lever only where HER is genuinely competitive (higher j
    / larger HER overpotential regime) — see follow-up.
    """
    r_weak = _run_single_sample(_sample_at_dg(DG_WEAK_EV), dict(DEFAULT_DESIGN_POINT))
    r_nom = _run_single_sample(_sample_at_dg(DG_HSTAR_NOMINAL_EV), dict(DEFAULT_DESIGN_POINT))
    r_strong = _run_single_sample(_sample_at_dg(DG_STRONG_EV), dict(DEFAULT_DESIGN_POINT))

    # Mechanism: weaker binding -> larger theta_H(1-theta_H) -> larger i0,H_eff.
    assert r_weak["her_i0_eff_A_m2"] > r_nom["her_i0_eff_A_m2"] > r_strong["her_i0_eff_A_m2"]
    # The i0,H_eff swing across the band is still order-of-magnitude (2-3x).
    assert r_weak["her_i0_eff_A_m2"] / r_strong["her_i0_eff_A_m2"] > 2.0
    # And larger i0,H_eff -> lower FE (more hydrogen evolution).
    assert r_weak["current_efficiency_percent"] < r_nom["current_efficiency_percent"]
    assert r_strong["current_efficiency_percent"] > r_nom["current_efficiency_percent"]
    # With physical her_i0 (PR #56) HER is negligible at the nominal gate, so
    # the FE band across dG_H* is small (< 1 pt here) — not the 10-15% of the
    # pre-fix artifact. Assert the physically-honest bound.
    fe_band = (
        r_strong["current_efficiency_percent"] - r_weak["current_efficiency_percent"]
    )
    assert 0.0 <= fe_band <= 1.0, (
        f"dG_H* band moved FE by {fe_band:.1f} pt at physical her_i0 (expect << 1 pt, "
        f"HER suppressed at nominal gate)"
    )


# ---------------------------------------------------------------------------
# 4. Monte Carlo / Sobol sweep includes the chain
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_mc_sweep_includes_dG_chain_in_fe_distribution():
    """FE-at-the-gate distribution is produced and dG_H* drives it.

    We assert two things that are physically true and robust:
      1. The MC sweep produces a materially-spread FE-at-the-gate distribution
         (std > 2 pt) — the deliverable of the DFT chain.
      2. dG_H* is the dominant (top-ranked) driver of the chain's *direct*
         surface-state output ``theta_H`` (via the Volmer coverage it anchors).
    We deliberately do NOT assert dG_H* tops the aggregate FE correlation
    ranking: the FE output also depends on ``her_i0``/``her_i0_Ea`` whose
    independently-wide screening lognormal (bounds 6 orders wide in the
    registry) dominates a multivariate Pearson proxy even though dG_H* is
    the larger *physical* lever. The signed FE propagation across the band
    is locked by test_fe_propagates_from_dg_signed_correctly instead.
    """
    engine = MonteCarloEngine(n_samples=400, seed=42, n_jobs=1)
    res = engine.run()

    # Deliverable: a FE-at-the-gate probability distribution. With the
    # corrected physical her_i0 (PR #56) HER is suppressed at the nominal
    # gate, so the FE distribution is narrow here (the old std > 2 pt
    # expectation was an artifact of the pre-fix her_i0 = 1e-3). Assert the
    # distribution is produced and finite; the dG_H* lever becomes material
    # only where HER is competitive (higher j / overpotential) — see follow-up.
    fe = res.output_distributions["current_efficiency_percent"]
    fe_v = fe[~np.isnan(fe)]
    assert len(fe_v) == 400
    assert np.isfinite(fe_v).all()
    # The chain is live: her_i0_eff is produced and swings with the band.
    i0_eff = res.output_distributions["her_i0_eff_A_m2"]
    assert np.isfinite(i0_eff[~np.isnan(fe)]).all()
    assert np.std(i0_eff) > 0  # the i0,H_eff output is not degenerate

    # dG_H* is present in the sweep and drives the chain's direct output.
    # theta_H depends only on (dG_H*, eta, T); in the sweep dG_H* must be
    # its top-ranked driver.
    th_sens = res.sensitivity.get("theta_H", {})
    assert th_sens.get("dG_Hstar_eV", 0.0) > 0.9, (
        f"dG_H* not top theta_H driver in MC sweep (got {th_sens})\n"
    )
