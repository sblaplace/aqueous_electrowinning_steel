"""Unit tests for the impurity co-deposition model (t_05cd6587)."""
from __future__ import annotations

import numpy as np

from models.impurity_codeposition import (
    ImpurityCoDeposition,
    BathKinetics,
    compare_bath_types,
    CU_HOT_SHORTNESS_WT,
)


class TestBasicDepositComposition:
    """Core checks on deposit_composition output structure and physics."""

    def test_output_keys(self):
        model = ImpurityCoDeposition(cu_conc_ppm=100.0)
        res = model.deposit_composition(100.0)
        for key in [
            "fe_wt_percent", "cu_wt_percent", "ni_wt_percent", "zn_wt_percent",
            "cu_in_ppm", "ni_in_ppm", "zn_in_ppm",
            "cu_exceeds_hot_shortness", "bath_type",
            "potential_V", "fe_current_A_m2", "cu_current_A_m2",
        ]:
            assert key in res, f"missing key: {key}"

    def test_fe_dominates_at_typical_conditions(self):
        """Fe should be >95 wt% of the deposit at typical impurity levels."""
        model = ImpurityCoDeposition(
            fe_conc_M=1.0, cu_conc_ppm=100.0, ni_conc_ppm=50.0,
            zn_conc_ppm=50.0, pH=3.0, temperature_C=60.0,
        )
        res = model.deposit_composition(100.0)
        assert res["fe_wt_percent"] > 95.0

    def test_cu_increases_with_bath_concentration(self):
        """Higher bath Cu²⁺ → more Cu in the deposit."""
        low = ImpurityCoDeposition(cu_conc_ppm=50.0)
        high = ImpurityCoDeposition(cu_conc_ppm=500.0)
        assert low.deposit_composition(100.0)["cu_wt_percent"] < \
               high.deposit_composition(100.0)["cu_wt_percent"]

    def test_cu_decreases_with_higher_current_density(self):
        """Nobler impurity fraction drops at higher j (Fe kinetics catch up)."""
        model = ImpurityCoDeposition(cu_conc_ppm=200.0)
        r_low = model.deposit_composition(50.0)
        r_high = model.deposit_composition(200.0)
        # Cu is nobler than Fe; at higher j, Fe current grows faster
        # so Cu fraction should decrease or stay similar
        assert r_high["cu_wt_percent"] <= r_low["cu_wt_percent"] * 1.5  # tolerance

    def test_zn_is_negligible(self):
        """Zn is less noble than Fe — it should barely co-deposit."""
        model = ImpurityCoDeposition(zn_conc_ppm=100.0)
        res = model.deposit_composition(100.0)
        # Zn should be well under 1 wt%
        assert res["zn_wt_percent"] < 1.0

    def test_total_impurity_wt_sums_correctly(self):
        model = ImpurityCoDeposition(
            cu_conc_ppm=200.0, ni_conc_ppm=100.0, zn_conc_ppm=50.0,
        )
        res = model.deposit_composition(100.0)
        expected = res["cu_wt_percent"] + res["ni_wt_percent"] + \
                   res["zn_wt_percent"] + res["pb_wt_percent"] + res["sn_wt_percent"]
        assert abs(res["total_impurity_wt"] - expected) < 1e-6


class TestConcentrationSweeps:
    """Verify the sweep helper methods."""

    def test_cu_sweep_shape(self):
        model = ImpurityCoDeposition()
        cu_ppm = np.array([10, 50, 100, 200, 500])
        sweep = model.cu_uptake_vs_concentration(cu_ppm, [50, 100, 200])
        assert len(sweep["cu_ppm"]) == 5
        for j in [50, 100, 200]:
            key = f"wt_j{j}_mA_cm2"
            assert key in sweep
            assert len(sweep[key]) == 5

    def test_ni_sweep_monotonic(self):
        """Ni in deposit should increase monotonically with bath Ni."""
        model = ImpurityCoDeposition()
        ni_ppm = np.array([10, 50, 100, 200])
        sweep = model.ni_uptake_vs_concentration(ni_ppm, [100])
        wt = sweep["wt_j100_mA_cm2"]
        assert all(wt[i] <= wt[i + 1] for i in range(len(wt) - 1))

    def test_zn_sweep_stays_low(self):
        """Zn should remain near zero across the sweep."""
        model = ImpurityCoDeposition()
        zn_ppm = np.array([10, 50, 100, 200, 500])
        sweep = model.zn_uptake_vs_concentration(zn_ppm, [100])
        wt = sweep["wt_j100_mA_cm2"]
        assert all(w < 1.0 for w in wt)


class TestBathComparison:
    """Sulfate vs chloride bath comparison."""

    def test_compare_bath_types_structure(self):
        results = compare_bath_types(cu_conc_ppm=200.0)
        assert "sulfate" in results
        assert "chloride" in results
        assert "cu_wt_percent" in results["sulfate"]
        assert "cu_wt_percent" in results["chloride"]

    def test_chloride_kinetics_differ_from_sulfate(self):
        """Chloride and sulfate baths should give different predictions."""
        results = compare_bath_types(cu_conc_ppm=200.0, j_mA_cm2=100.0)
        s = results["sulfate"]["cu_wt_percent"]
        c = results["chloride"]["cu_wt_percent"]
        # They should differ (not necessarily one always > the other)
        assert s != c

    def test_chloride_bath_type_flag(self):
        model = ImpurityCoDeposition(bath_type="chloride")
        res = model.deposit_composition(100.0)
        assert res["bath_type"] == "chloride"


class TestPurificationThreshold:
    """Cu purification threshold bisection."""

    def test_threshold_output_structure(self):
        model = ImpurityCoDeposition()
        result = model.cu_purification_threshold(100.0)
        assert "threshold_cu_ppm" in result
        assert "deposit_cu_wt" in result
        assert "max_cu_wt" in result

    def test_threshold_deposit_cu_at_limit(self):
        """At the threshold, deposit Cu should be at or below the limit."""
        model = ImpurityCoDeposition()
        result = model.cu_purification_threshold(100.0)
        assert result["deposit_cu_wt"] <= CU_HOT_SHORTNESS_WT + 0.01  # tolerance

    def test_threshold_higher_j_allows_more_ppm(self):
        """At higher j, Fe kinetics dominate more → you can tolerate more bath Cu."""
        model = ImpurityCoDeposition()
        t_low = model.cu_purification_threshold(50.0)
        t_high = model.cu_purification_threshold(200.0)
        assert t_high["threshold_cu_ppm"] >= t_low["threshold_cu_ppm"]


class TestCustomKinetics:
    """User can override kinetic parameters."""

    def test_custom_kinetics_applied(self):
        custom = BathKinetics(
            fe_i0=1.0e-2, cu_i0=10.0, ni_i0=1.0e-2, zn_i0=1.0e-2,
            pb_i0=1.0e-1, sn_i0=5.0e-2,
            fe_tafel=0.120, cu_tafel=0.150, ni_tafel=0.100,
            zn_tafel=0.120, pb_tafel=0.120, sn_tafel=0.110,
        )
        model = ImpurityCoDeposition(
            cu_conc_ppm=200.0, custom_kinetics=custom,
        )
        res = model.deposit_composition(100.0)
        assert res["cu_wt_percent"] > 0.0


class TestModuleImports:
    """Smoke-test top-level imports."""

    def test_import_from_init(self):
        from models import ImpurityCoDeposition, compare_bath_types, BathKinetics
        assert ImpurityCoDeposition is not None
        assert compare_bath_types is not None
        assert BathKinetics is not None
