"""Cross-modal bottom-up theory — one shared parameter set across modalities.

Tests that a single :class:`SharedScenario` parameterization is passed through
the electrochem/transport (``cell_physics``), thermal (``thermal_balance``) and
crate/environment (``crate`` + ``env_coupling``) seams simultaneously, and that
the consistency harness honestly reports where *no single parameter set can
satisfy every modality* (the cross-modal conflicts).
"""
from __future__ import annotations

import pytest

from models.cross_modal_theory import (
    SharedScenario,
    run_all_modalities,
    consistent_scenario,
    find_consistent_cooling,
)
from models.uncertainty import REGISTRY, registry_summary

MODALITIES = ["electrochem", "transport", "thermal", "crate", "environment"]


# ── 1. The one shared parameter registry sources every modality ────────────

def test_registry_now_covers_thermal_and_crate_modules():
    """The central registry must include thermal + crate/env props, not just
    the metallurgy/electrochem constants, so one registry feeds every seam."""
    present = set(registry_summary()["by_module"].keys())
    assert "thermal" in present, "registry missing thermal props"
    assert "crate" in present, "registry missing crate/env props"
    # The shared thermal/transport props the task calls out must exist.
    for name in ("thermoneutral_V", "volume_L", "UA_amb_W_K", "UA_jacket_W_K",
                 "electrode_area_m2", "T_ambient_C",
                 "crate_mass_kg", "crate_height_m", "drag_coefficient",
                 "design_gust_m_s", "soil_bearing_kPa"):
        assert name in REGISTRY, f"registry missing shared param {name}"


def test_shared_scenario_is_built_from_registry():
    """build from REGISTRY: electrochem, transport, thermal and crate props
    all read their nominal values from the single registry."""
    s = SharedScenario.from_registry()
    assert s.fe_i0 == pytest.approx(REGISTRY["fe_i0"].mean)
    assert s.her_i0 == pytest.approx(REGISTRY["her_i0"].mean)
    assert s.fe_tafel_V == pytest.approx(REGISTRY["fe_tafel_V"].mean)
    assert s.T_operating_C == pytest.approx(REGISTRY["T_operating_C"].mean)
    assert s.volume_L == pytest.approx(REGISTRY["volume_L"].mean)
    assert s.UA_amb_W_K == pytest.approx(REGISTRY["UA_amb_W_K"].mean)
    assert s.thermoneutral_V == pytest.approx(REGISTRY["thermoneutral_V"].mean)
    assert s.electrode_area_m2 == pytest.approx(REGISTRY["electrode_area_m2"].mean)
    assert s.crate_mass_kg == pytest.approx(REGISTRY["crate_mass_kg"].mean)
    assert s.crate_height_m == pytest.approx(REGISTRY["crate_height_m"].mean)
    assert s.soil_bearing_kPa == pytest.approx(REGISTRY["soil_bearing_kPa"].mean)
    assert s.drag_coefficient == pytest.approx(REGISTRY["drag_coefficient"].mean)


# ── 2. Surface where a single parameter set cannot satisfy every modality ──

def test_default_parameterization_surfaces_thermal_conflict():
    """Registry nominals with no cooling: chemistry, transport, crate and env
    all PASS, but the thermal balance equilibrates well above the assumed
    operating temperature -> a single parameter set cannot hold both the 60 C
    chemistry and the ~76 C thermal steady state."""
    rep = run_all_modalities(SharedScenario.from_registry())
    assert rep.consistent is False
    assert rep.electrochem.passed
    assert rep.transport.passed
    assert not rep.thermal.passed, "thermal should be the conflict modal"
    assert rep.crate.passed
    assert rep.environment.passed
    # The deviation from the assumed operating temperature exceeds tolerance.
    assert abs(rep.T_ss_C - SharedScenario.from_registry().T_operating_C) > 15.0
    # The controlling parameters name the heat-generation side of the conflict.
    for k in ("V_cell_V", "current_A", "UA_amb_W_K", "volume_L"):
        assert k in rep.thermal.controlling, f"thermal missing controlling param {k}"


def test_consistent_parameterization_closes_every_modality():
    """Sizing the cooling jacket to the electrochem V_cell·I heat load re-closes
    the thermal loop: the SAME parameter set now satisfies every modality."""
    cs = consistent_scenario(SharedScenario.from_registry())
    rep = run_all_modalities(cs)
    assert rep.consistent is True
    for name in MODALITIES:
        m = getattr(rep, name)
        assert m.passed, f"{name} modal should pass in consistent config: {m.detail}"
    # The converged steady state sits within tolerance of the operating T.
    assert abs(rep.T_ss_C - cs.T_operating_C) <= cs.T_consistent_tol_C


def test_find_consistent_cooling_returns_jacket_sizing():
    """find_consistent_cooling returns a finite jacket UA that closes the loop."""
    ua = find_consistent_cooling(SharedScenario.from_registry())
    assert ua is not None
    assert ua > 0.0


def test_transport_limit_conflict():
    """Pushing j above the migration-enhanced transport limit fails transport
    (and, at high j, chemistry/precipitation) — transport discipline conflicts."""
    rep = run_all_modalities(SharedScenario.from_registry().with_(j_mA_cm2=1500.0))
    assert not rep.transport.passed
    limit = rep.transport.controlling["transport_limit_mA_cm2"]
    assert limit < 1500.0


def test_crate_stability_conflict():
    """A light, tall crate at high wind fails the envelope stability check."""
    rep = run_all_modalities(SharedScenario.from_registry().with_(
        gust_m_s=60.0, crate_mass_kg=500.0, crate_height_m=3.0,
    ))
    assert not rep.crate.passed
    assert "gust_m_s" in rep.crate.controlling
    assert "crate_mass_kg" in rep.crate.controlling


# ── 3. Environment→thermal wiring: the site the envelope sees ──────────────

def test_wind_overcools_bath_via_forced_convection():
    """The same wind the crate is sized for also drives forced convection that
    augments the thermal ambient loss, dragging the bath below its assumed
    operating temperature — a genuine cross-modal coupling."""
    calm = SharedScenario.from_registry().with_(gust_m_s=0.0, rain_mm_hr=0.0)
    windy = SharedScenario.from_registry().with_(gust_m_s=40.0, rain_mm_hr=0.0)
    r_calm = run_all_modalities(calm)
    r_windy = run_all_modalities(windy)
    # Env-coupling adds h_conv under wind -> more heat loss -> cooler steady state.
    assert r_windy.environment.controlling["h_conv_W_m2_K"] > 0.0
    assert r_windy.T_ss_C < r_calm.T_ss_C


def test_env_wiring_is_noop_when_site_calm():
    """With no wind/rain/ingress the disturbance is disabled and the thermal
    balance uses the ambient design point unchanged."""
    rep = run_all_modalities(SharedScenario.from_registry())
    assert rep.environment.passed
    assert rep.environment.controlling["h_conv_W_m2_K"] == 0.0
    # T_ambient fed to thermal equals the scenario's shared ambient temperature.
    assert rep.environment.controlling["T_ambient_C"] == pytest.approx(
        SharedScenario.from_registry().T_ambient_C
    )


# ── 4. Reporting surface ───────────────────────────────────────────────────

def test_report_exposes_all_modalities_and_controlling_params():
    rep = run_all_modalities(SharedScenario.from_registry())
    names = [m.name for m in rep.modalities()]
    assert set(names) == set(MODALITIES)
    for m in rep.modalities():
        assert isinstance(m.controlling, dict) and len(m.controlling) > 0, (
            f"{m.name} should report controlling parameters"
        )


def test_report_to_dict_is_serialisable():
    import json

    rep = run_all_modalities(consistent_scenario(SharedScenario.from_registry()))
    d = rep.to_dict()
    assert d["consistent"] is True
    assert len(d["modalities"]) == len(MODALITIES)
    # Round-trips through JSON cleanly (no numpy/non-serialisable floats).
    json.dumps(d)
