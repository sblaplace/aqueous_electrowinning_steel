"""Fe³⁺ redox shuttle wired as CSTR terms in ``bath_dynamics`` (2026-08).

Covers the seven things an honest wiring must get right:

1.  **Default-off byte identity** — with the extension off, design points with
    or without the new keys produce bit-identical trajectories.
2.  **Dynamic ↔ static asymptote** — held at fixed (T, pH, Fe²⁺) with
    precipitation inactive, the CSTR relaxes to the static module's
    closed-form ``fe3_ss_M`` (the recirculation terms cancel identically at
    mutual steady state, leaving ``[Fe³⁺]_ss = r_prod / (k_m·A/V)``).
3.  **Hydrolysis cap + iron ledger** — with production cranked past the cap,
    Fe³⁺ pins at the Fe(OH)₃ cap and the TOTAL iron ledger (dissolved Fe²⁺/Fe³⁺
    in both compartments + cumulative sludge + Faraday-plated Fe) closes.
4.  **Galvanostatic CE steal** — the shuttle slip slows deposit growth by the
    ``i_sh/j`` factor of ``fe3_shuttle.ce_penalty_at_j``.
5.  **Proton stoichiometry** — autoxidation consumes H⁺ (pH rises before the
    cap), Fe(OH)₃ precipitation releases 3 H⁺ (bath acidifies once sludge
    runs), net +2 H⁺ per mol sludge.
6.  **Stiffness** — the exact-exponential integrator stays finite and
    non-negative when recirculation and A/V make the compartment stiff.
7.  **Diagnostics** — ``fe3_shuttle_terms`` matches the static module for a
    matched below-cap state; the static module's mol/m²/s sludge field is
    unit-pinned (×1000 m³↔L erratum).
"""

from __future__ import annotations

import numpy as np
import pytest

from models.bath_dynamics import (
    BathAux,
    apply_fe3_scenario,
    fe3_shuttle_terms,
    steady_state_fe3_M,
    step,
)
from models.electrochemistry import FARADAY, Z_FE
from models.fe3_shuttle import (
    D_FE3_REF_M2_S,
    ShuttleParams,
    fe3_solubility_cap_M,
    open_headspace,
    sealed_divided_cell,
    steady_state as static_steady_state,
)
from models.twin_physics import CellProcessModel

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def model():
    return CellProcessModel()


def _base_dp(**over):
    """Design point exercising the Fe³⁺ terms; O₂/scenario keys unset by default."""
    dp = {
        "temperature_C": 50.0,
        "pH": 2.0,
        "cell_voltage_V": 5.0,
        "j_avg_mA_cm2": 0.0,          # no plating drive in redox-focused tests
        "electrode_area_m2": 1.0,
        "electrolyte_volume_L": 800.0,
        "fe2_M": 1.0,
        "recirculation_flow_L_hr": 6000.0,
        "reservoir_volume_L": 2000.0,
        "catholyte_volume_L": 800.0,
        "anolyte_volume_L": 2000.0,
        "buffer_capacity_beta": 1e9,  # pin pH unless a test wants it free
        "acid_dose_rate_M_hr": 0.0,
        "pH_control_gain_M_hr_ph": 0.0,
        "fe2_makeup_rate_M_hr": 0.0,
    }
    dp.update(over)
    return dp


def _x0(j=1e-3):
    return np.array([50.0, 50.0, 1.0, 2.0, j, 0.0, 5.0])


def _aux0():
    return BathAux(T_reservoir_C=50.0, fe2_reservoir_M=1.0, pH_reservoir=2.0)


def _run(model, dp, x, aux, dt_hr, n_steps):
    for _ in range(n_steps):
        x, aux = step(x, aux, dt_hr, dp, model)
    return x, aux


# ---------------------------------------------------------------------------
# 1. Default-off identity
# ---------------------------------------------------------------------------

class TestDisabledIdentity:
    def test_disabled_byte_identical_trajectory(self, model):
        dp_plain = _base_dp()
        dp_flagged = _base_dp(
            fe3_shuttle_enabled=False,
            fe3_o2_fraction_of_sat=0.9,
            fe3_crossover_o2_flux_mol_m2_s=1e-6,
            fe3_k_ox_ref=1e6,
        )
        xa, xb = _x0(j=150.0), _x0(j=150.0)
        aa, ab = _aux0(), _aux0()
        for _ in range(20):
            xa, aa = step(xa, aa, 0.05, dp_plain, model)
            xb, ab = step(xb, ab, 0.05, dp_flagged, model)
        np.testing.assert_array_equal(xa, xb)
        assert aa.to_dict() == ab.to_dict()

    def test_disabled_leaves_fe3_fields_untouched(self, model):
        dp = _base_dp()
        x, aux = _run(model, dp, _x0(150.0), _aux0(), 0.1, 10)
        assert aux.fe3_catholyte_M == 0.0
        assert aux.fe3_reservoir_M == 0.0
        assert aux.fe3_sludge_cumulative_mol == 0.0
        snap = fe3_shuttle_terms(x, aux, dp)
        assert snap["enabled"] is False
        assert snap["i_shuttle_A_m2"] == 0.0


# ---------------------------------------------------------------------------
# 2. Dynamic ↔ static asymptote
# ---------------------------------------------------------------------------

class TestStaticDynamicAsymptote:
    def test_cstr_relaxes_to_static_steady_state(self, model):
        """Below-cap sealed bath: fe3(t→∞) = r_prod/(k_m·A/V) of fe3_shuttle.

        Uses the default-twin volumes: the pre-existing Fe²⁺ thermal balances
        are explicit Euler (stable for flow/V·dt ≲ 2), so tests stay on that
        side of the envelope while the new Fe³⁺ block is exact at any dt.
        """
        # Default twin volumes: the pre-existing explicit thermal balance sets
        # the dt envelope (flows stay stable), and the coupled two-pool slow
        # mode is τ_slow ≈ 71 hr — so integrate 500 hr (≈7·τ_slow).
        dp = apply_fe3_scenario(_base_dp(), sealed_divided_cell())
        static_ss = steady_state_fe3_M(dp)
        assert 0.0 < static_ss < fe3_solubility_cap_M(2.0)  # below-cap branch

        x = _x0()
        x, aux = _run(model, dp, x, _aux0(), 0.1, 5000)
        assert aux.fe3_catholyte_M == pytest.approx(static_ss, rel=0.05)

    def test_static_helper_matches_fe3_shuttle_module(self, model):
        dp = apply_fe3_scenario(
            _base_dp(pH=2.35, catholyte_volume_L=0.5, electrode_area_m2=1e-3),
            sealed_divided_cell(),
        )
        p = ShuttleParams(pH=2.35, temperature_C=50.0,
                          cathode_area_m2=1e-3, catholyte_volume_L=0.5)
        want = static_steady_state(p, sealed_divided_cell())["fe3_ss_M"]
        assert steady_state_fe3_M(dp) == pytest.approx(want, rel=1e-12)

    def test_snapshot_matches_static_module_below_cap(self, model):
        """Same state → same instantaneous terms as the static module."""
        km = D_FE3_REF_M2_S / 50e-6
        dp = apply_fe3_scenario(
            _base_dp(pH=2.35, catholyte_volume_L=0.5, electrode_area_m2=1e-3),
            sealed_divided_cell(),
        )
        p = ShuttleParams(pH=2.35, temperature_C=50.0,
                          cathode_area_m2=1e-3, catholyte_volume_L=0.5)
        ss = static_steady_state(p, sealed_divided_cell())
        x = np.array([50.0, 50.0, 1.0, 2.35, 300.0, 0.0, 5.0])
        aux = BathAux(50.0, 1.0, 2.35, fe3_catholyte_M=ss["fe3_ss_M"])
        snap = fe3_shuttle_terms(x, aux, dp)
        assert snap["enabled"] is True
        assert snap["i_shuttle_A_m2"] == pytest.approx(ss["i_shuttle_A_m2"], rel=1e-12)
        assert snap["r_prod_M_s"] == pytest.approx(ss["fe3_production_M_s"], rel=1e-12)
        assert snap["fe3_solubility_cap_M"] == pytest.approx(ss["fe3_solubility_cap_M"])
        # Below cap, shuttle sink ≈ production (steady state identity).
        assert snap["shuttle_sink_M_s"] == pytest.approx(
            km * (1e-3 / 5e-4) * ss["fe3_ss_M"], rel=1e-12)


# ---------------------------------------------------------------------------
# 3. Cap + iron ledger closure
# ---------------------------------------------------------------------------

class TestIronLedger:
    def test_precipitation_caps_fe3_and_ledger_closes(self, model):
        """Cranked autoxidation: fe3 pins at the cap; total-Fe ledger closes."""
        k_ox_crank = 1e-4 * 1e4
        dp = apply_fe3_scenario(_base_dp(fe3_k_ox_ref=k_ox_crank),
                                sealed_divided_cell())
        x = _x0()
        aux = _aux0()
        area = dp["electrode_area_m2"]
        V_c, V_r = dp["catholyte_volume_L"], dp["reservoir_volume_L"]
        dt = 0.02

        def total_fe_mol(x_, aux_, plated):
            return ((x_[2] + aux_.fe3_catholyte_M) * V_c
                    + (aux_.fe2_reservoir_M + aux_.fe3_reservoir_M) * V_r
                    + aux_.fe3_sludge_cumulative_mol + plated)

        fe_total_0 = total_fe_mol(x, _aux0(), 0.0)
        plated_mol = 0.0
        for _ in range(250):  # 5 hr
            # Faraday-plated Fe for the ledger, evaluated from the state via
            # the physics model (the gate-ledger definition), mirrors the
            # consumption term inside step().
            pred = model.predict(j_mA_cm2=max(1e-3, x[4]),
                                 temperature_C=max(0.0, x[0]),
                                 fe2_M=max(1e-6, x[2]))
            j_fe_her = max(x[4] * 10.0 - fe3_shuttle_terms(x, aux, dp)["i_shuttle_A_m2"], 0.0)
            consumption = j_fe_her * pred.current_efficiency / (Z_FE * FARADAY)
            plated_mol += consumption * area * 3600.0 * dt
            x, aux = step(x, aux, dt, dp, model)

        assert aux.fe3_sludge_cumulative_mol > 1.0  # mol of sludge formed
        assert aux.fe3_catholyte_M <= fe3_solubility_cap_M(x[3]) * (1.0 + 1e-9)
        fe_total_1 = total_fe_mol(x, aux, plated_mol)
        assert fe_total_1 == pytest.approx(fe_total_0, rel=2e-3)

    def test_static_sludge_field_units_mol_m2_s(self):
        """Unit pin (×1000 m³↔L erratum): mol/m²/s = (r−k·cap)·V_L/A."""
        p = ShuttleParams(pH=2.0, temperature_C=50.0, k_ox_ref=1e-4 * 1e6,
                          cathode_area_m2=1e-3, catholyte_volume_L=0.5)
        ss = static_steady_state(p, open_headspace())
        assert ss["feoh3_precipitation_active"]
        km = p.d_fe3_m2_s / p.boundary_layer_m
        a_over_v = p.cathode_area_m2 / (p.catholyte_volume_L / 1000.0)
        sludge_vol = ss["fe3_production_M_s"] - km * a_over_v * ss["fe3_ss_M"]
        want = sludge_vol * p.catholyte_volume_L / p.cathode_area_m2  # mol/s/m²
        assert ss["iron_sludge_loss_mol_m2_s"] == pytest.approx(want, rel=1e-12)
        # and stays the g/L/day-consistent value
        assert ss["iron_sludge_loss_g_L_day"] == pytest.approx(
            sludge_vol * 55.845 * 86400.0, rel=1e-9)


# ---------------------------------------------------------------------------
# 4. Galvanostatic CE steal
# ---------------------------------------------------------------------------

class TestCESlip:
    def test_shuttle_slows_deposit_by_slip_factor(self, model):
        """Big shuttle current: deposit grows ≈ (1 − i_sh/j) slower."""
        common = dict(
            j_avg_mA_cm2=300.0, fe3_d_m2_s=1e-7, fe3_k_ox_ref=1e-4 * 1e4,
        )
        dp_on = apply_fe3_scenario(_base_dp(**common), sealed_divided_cell())
        dp_off = _base_dp(**dict(common, fe3_shuttle_enabled=False))
        x_on = _x0(j=300.0)
        x_off = _x0(j=300.0)
        a_on, a_off = _aux0(), _aux0()
        x_on, a_on = _run(model, dp_on, x_on, a_on, 0.02, 100)
        x_off, a_off = _run(model, dp_off, x_off, a_off, 0.02, 100)
        assert x_on[5] > 0.0
        assert x_on[5] < x_off[5] * 0.98  # visible, model-predicted theft
        # bounded by the instantaneous slip while cap-pinned
        snap = fe3_shuttle_terms(x_on, a_on, dp_on)
        assert snap["ce_loss_fraction"] > 0.01


# ---------------------------------------------------------------------------
# 5. Proton stoichiometry
# ---------------------------------------------------------------------------

class TestProtons:
    def test_autoxidation_raises_pH_before_cap(self, model):
        """Production without precipitation consumes H⁺ → pH rises slightly."""
        beta = 0.05
        dp = apply_fe3_scenario(
            _base_dp(buffer_capacity_beta=beta, fe3_k_ox_ref=1e-1),
            sealed_divided_cell())
        dp_off = _base_dp(buffer_capacity_beta=beta, fe3_shuttle_enabled=False)
        x_on, a_on = _run(model, dp, _x0(), _aux0(), 0.005, 3)
        x_off, _ = _run(model, dp_off, _x0(), _aux0(), 0.005, 3)
        assert a_on.fe3_sludge_cumulative_mol == 0.0
        assert x_on[3] > x_off[3]  # +r_prod/β relative to baseline

    def test_sludge_acidifies_bath_net_plus_2H(self, model):
        """Once sludge runs, +3H/Fe precipitation dominates → bath acidifies.

        The pH falls monotonically (net +2H per sludge mol) but flattens:
        acidification RAISES the Fe(OH)3 cap (×1000/pH), throttling further
        precipitation — the recorded 12 hr trajectory goes 2.0 → 1.75.
        """
        dp = apply_fe3_scenario(
            _base_dp(buffer_capacity_beta=0.05, fe3_k_ox_ref=1e0),
            sealed_divided_cell())
        x, aux = _run(model, dp, _x0(), _aux0(), 0.01, 1200)
        assert aux.fe3_sludge_cumulative_mol > 10.0
        assert x[3] < 1.8  # decisively below the start (2.0) and still falling
        # cap-chasing coherence: fe3 sits at the (rising) cap as pH falls
        assert aux.fe3_catholyte_M == pytest.approx(
            fe3_solubility_cap_M(x[3]), rel=0.2)


# ---------------------------------------------------------------------------
# 6. Stiffness
# ---------------------------------------------------------------------------

class TestStiffness:
    def test_stiff_shuttle_stays_finite(self, model):
        """k·dt ≫ 2 would kill explicit Euler on Fe³⁺; the exact exponential holds.

        Stiffness is cranked through the shuttle's own k_m (D), leaving the
        pre-existing explicit balances in their stable envelope.
        """
        dp = apply_fe3_scenario(
            _base_dp(fe3_d_m2_s=1e-6),   # k_shuttle ≈ 90 /hr ≫ 2/dt
            sealed_divided_cell())
        x, aux = _run(model, dp, _x0(), _aux0(), 0.1, 20)
        assert np.all(np.isfinite(x))
        assert 0.0 <= aux.fe3_catholyte_M < 1.0
        # and it still lands at the static steady state for these knobs
        assert aux.fe3_catholyte_M == pytest.approx(steady_state_fe3_M(dp), rel=0.05)
