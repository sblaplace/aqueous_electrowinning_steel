"""Tests for the TEA sensitivity sweep."""
import numpy as np

from models.run_tea_sensitivity import (
    compute_lcofe,
    run_sensitivity_sweep,
    find_pareto_front,
    evaluate_kill_criteria,
    DRI_H2_MID,
)


class TestComputeLCOFe:
    def test_baseline_returns_finite(self):
        lcofe = compute_lcofe(
            j_mA_cm2=100, V_cell=2.5, FE=0.90,
            elec_price=0.04, cell_cost_per_m2=330,
        )
        assert np.isfinite(lcofe)
        assert 200 < lcofe < 3000

    def test_low_energy_cheap_beats_dri(self):
        """Optimistic point should handily beat DRI-H2."""
        lcofe = compute_lcofe(
            j_mA_cm2=200, V_cell=2.0, FE=0.95,
            elec_price=0.02, cell_cost_per_m2=100,
        )
        assert lcofe < DRI_H2_MID

    def test_high_energy_expensive_loses(self):
        """Pessimistic point should be well above DRI-H2."""
        lcofe = compute_lcofe(
            j_mA_cm2=50, V_cell=4.0, FE=0.50,
            elec_price=0.10, cell_cost_per_m2=2000,
        )
        assert lcofe > DRI_H2_MID


class TestSweep:
    def test_shapes(self):
        data = run_sensitivity_sweep(n_samples=500, seed=0)
        assert len(data["lcofe"]) == 500
        for key in ("j", "V", "FE", "elec", "cell_cost"):
            assert len(data[key]) == 500

    def test_lcofe_finite_fraction(self):
        data = run_sensitivity_sweep(n_samples=2000, seed=1)
        finite_frac = np.isfinite(data["lcofe"]).mean()
        assert finite_frac > 0.95


class TestParetoFront:
    def test_majority_below_target(self):
        data = run_sensitivity_sweep(n_samples=5000, seed=42)
        pareto = find_pareto_front(data, target=DRI_H2_MID)
        assert pareto["pct_below_target"] > 30  # conservative bound

    def test_winning_stats_present(self):
        data = run_sensitivity_sweep(n_samples=5000, seed=42)
        pareto = find_pareto_front(data, target=DRI_H2_MID)
        if pareto["n_below_target"] > 0:
            ws = pareto["winning_stats"]
            assert 50 <= ws["j_median"] <= 500
            assert 0.50 <= ws["FE_median"] <= 0.95


class TestKillCriteria:
    def test_verdict_present(self):
        data = run_sensitivity_sweep(n_samples=5000, seed=42)
        pareto = find_pareto_front(data, target=DRI_H2_MID)
        kill = evaluate_kill_criteria(data, pareto)
        assert "verdict" in kill
        assert "STRONG" in kill["verdict"] or "VIABLE" in kill["verdict"]
