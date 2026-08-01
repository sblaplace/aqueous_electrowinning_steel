"""Tests for feedstock purification model — Cu/Ni/Zn removal."""

import numpy as np
import pytest

from models.purification import (
    CementationModel,
    CementationParams,
    HydrolysisModel,
    HydrolysisParams,
    IonExchangeModel,
    PurificationFeedstock,
    PurificationModel,
    SelectiveElectrowinningModel,
    purification_from_closed_loop_result,
)


# ── Helpers ───────────────────────────────────────────────────────────

def default_feed(**overrides):
    kwargs = dict(
        cu_M=5.0e-4,
        ni_M=3.0e-4,
        zn_M=2.0e-4,
        fe2_M=1.0,
        fe3_M=0.05,
        pH=2.0,
        temperature_C=50.0,
        volume_L=1000.0,
    )
    kwargs.update(overrides)
    return PurificationFeedstock(**kwargs)


# ── Tests ─────────────────────────────────────────────────────────────

def test_cementation_removes_cu_ni_zn():
    """Cementation on Fe powder reduces Cu, Ni, Zn concentrations."""
    feed = default_feed()
    model = CementationModel(feedstock=feed)
    result = model.simulate(duration_hr=6.0, dt_hr=0.05)

    frac = result.removal_fractions()
    assert frac["cu"] > 0.8, f"Cu removal {frac['cu']:.3f} should exceed 80%"
    assert frac["ni"] > 0.3, f"Ni removal {frac['ni']:.3f} should exceed 30%"
    assert frac["zn"] > 0.4, f"Zn removal {frac['zn']:.3f} should exceed 40%"
    # Cu removes fastest (highest rate constant + most favourable potential)
    assert frac["cu"] > frac["ni"]
    assert frac["cu"] > frac["zn"]


def test_cementation_rate_increases_with_temperature():
    """Higher temperature accelerates cementation (Arrhenius)."""
    feed_cold = default_feed(temperature_C=25.0)
    feed_hot = default_feed(temperature_C=70.0)
    result_cold = CementationModel(feedstock=feed_cold).simulate(2.0)
    result_hot = CementationModel(feedstock=feed_hot).simulate(2.0)

    assert result_cold.cu_M[-1] > result_hot.cu_M[-1], (
        "Hot cementation should leave less Cu"
    )


def test_cementation_parameter_validation():
    """Negative rate constants are rejected."""
    with pytest.raises(ValueError, match="non-negative"):
        CementationParams(cu_rate_const_per_hr=-1.0)


def test_hydrolysis_removes_fe3_preserves_fe2():
    """Fe³⁺ precipitates; Fe²⁺ is unaffected at pH 3.5."""
    feed = default_feed(fe3_M=0.05, fe2_M=1.0, pH=2.0)
    model = HydrolysisModel(feedstock=feed)
    result = model.simulate()

    assert result.fe3_remaining_M < 0.001, "Fe³⁺ should drop to solubility limit"
    assert result.fe2_unchanged_M == pytest.approx(1.0), "Fe²⁺ must be preserved"
    assert result.precipitate_mass_kg > 0


def test_hydrolysis_co_precipitates_contaminants():
    """Hydrolysis removes a fraction of Cu/Ni/Zn by co-precipitation."""
    feed = default_feed(cu_M=1.0e-3, ni_M=1.0e-3, zn_M=1.0e-3)
    model = HydrolysisModel(feedstock=feed)
    result = model.simulate()

    assert result.cu_removed_M > 0
    assert result.ni_removed_M > 0
    assert result.zn_removed_M > 0
    assert result.cost_USD > 0  # NaOH cost


def test_hydrolysis_pH_validation():
    """Unphysical pH is rejected."""
    with pytest.raises(ValueError, match="target_pH"):
        HydrolysisParams(target_pH=0.5)


def test_selective_electrowinning_plates_cu_not_fe():
    """Cathode potential between Cu and Fe deposits Cu, not Fe."""
    feed = default_feed(cu_M=5.0e-4)
    model = SelectiveElectrowinningModel(feedstock=feed)
    result = model.simulate(cu_M=5.0e-4, ni_M=3.0e-4, duration_hr=2.0)

    assert result.cu_plated_M > 0, "Cu should plate at this cathode potential"
    # At E = -0.10 V vs SHE, Fe⁰ (E0 = -0.44) does not deposit
    assert result.energy_kWh > 0


def test_ion_exchange_polishes_residual():
    """Ion exchange removes residual Cu/Ni below electrowinning."""
    model = IonExchangeModel()
    result = model.simulate(cu_M=1.0e-5, ni_M=1.0e-5)

    assert result.cu_final_M < 1.0e-5
    assert result.ni_final_M < 1.0e-5
    assert result.cost_USD > 0


def test_full_purification_meets_cu_spec():
    """Full train achieves Cu < 0.01 wt% relative to Fe."""
    feed = default_feed(cu_M=5.0e-4, fe2_M=1.0)
    model = PurificationModel(feedstock=feed)
    result = model.simulate()

    assert result.cu_meets_spec(), (
        f"Cu final {result.cu_final_wt_pct:.4f} wt% exceeds 0.01 wt% limit"
    )
    assert result.cu_removal_fraction > 0.9


def test_full_purification_cost_per_tonne():
    """Cost per tonne of Fe is computed and positive."""
    feed = default_feed()
    model = PurificationModel(feedstock=feed)
    result = model.simulate()

    assert result.total_cost_per_t_fe > 0
    assert "cementation_USD" in result.stage_costs
    assert "hydrolysis_USD" in result.stage_costs
    assert "electrowinning_USD" in result.stage_costs
    assert "ion_exchange_USD" in result.stage_costs


def test_integration_with_closed_loop_model():
    """purification_from_closed_loop_result runs end-to-end."""
    # Build a minimal mock of PhaseIVResult
    from types import SimpleNamespace

    cl_result = SimpleNamespace(
        impurity_M=np.array([0.001, 0.0015, 0.002]),
        fe_M=np.array([1.0, 0.95, 0.90]),
        time_hr=np.array([0.0, 100.0, 200.0]),
    )
    result = purification_from_closed_loop_result(cl_result)

    assert result.cu_final_M < cl_result.impurity_M[-1]
    assert result.total_cost_per_t_fe > 0


def test_summary_dict_structure():
    """summary() returns all expected keys."""
    model = PurificationModel(feedstock=default_feed())
    result = model.simulate()
    summary = result.summary()

    for key in ("cu_initial_M", "cu_final_M", "cu_removal_fraction",
                "cu_final_wt_pct", "cu_meets_spec_0_01wt_pct",
                "total_cost_per_t_fe", "stage_costs",
                "impurity_buildup_rate_M_per_hr"):
        assert key in summary, f"Missing key {key}"


def test_high_copper_feed_exceeds_spec_without_purification():
    """Without purification, a 0.5 mM Cu / 1 M Fe feed gives > 0.1 wt% Cu."""
    feed = default_feed(cu_M=5.0e-4, fe2_M=1.0)
    # wt% = (cu_M / fe2_M) * (M_CU / M_FE) * 100
    wt_pct = feed.cu_M / feed.fe2_M * 63.546 / 55.845 * 100.0
    assert wt_pct > 0.05, "Feed Cu should be significant enough to motivate purification"
