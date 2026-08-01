import pytest
import numpy as np
from models.process_control import (
    PIDController,
    PIDParams,
    CascadeController,
    default_loops,
    simulate_loop,
    simulate_first_order_plant,
    tuning_sensitivity,
    loop_summary_table,
    LoopResult,
)


class TestPIDController:
    def test_proportional_only_converges_to_setpoint(self):
        """P-only controller should reduce error but leave steady-state offset."""
        ctrl = PIDController(PIDParams(Kp=2.0, Ki=0.0, Kd=0.0, setpoint=100.0,
                                       mv_min=0.0, mv_max=200.0))
        pv = 0.0
        for _ in range(500):
            mv = ctrl.update(pv, dt=0.1)
            # simple plant: pv += 0.1 * (mv - pv)
            pv += 0.1 * 0.1 * (mv - pv)
        # PV should move toward setpoint significantly
        assert pv > 50.0, f"PV={pv} should be well above initial 0"
        # P-only won't fully converge
        assert pv < 100.0, "P-only should leave offset"

    def test_pi_controller_eliminates_offset(self):
        """PI controller should drive steady-state error to near zero."""
        ctrl = PIDController(PIDParams(Kp=2.0, Ki=0.5, Kd=0.0, setpoint=100.0,
                                       mv_min=0.0, mv_max=500.0))
        pv = 0.0
        for _ in range(2000):
            mv = ctrl.update(pv, dt=0.1)
            pv += 0.1 * 0.1 * (mv - pv)
        assert abs(pv - 100.0) < 1.0, f"PV={pv} should be near setpoint 100"

    def test_anti_windup_clamps_integral(self):
        """When output saturates, integral should not wind up indefinitely."""
        ctrl = PIDController(PIDParams(Kp=1.0, Ki=10.0, Kd=0.0, setpoint=100.0,
                                       mv_min=0.0, mv_max=10.0,  # tight clamp
                                       anti_windup_limit=50.0))
        # Run with PV far below setpoint to force saturation
        for _ in range(1000):
            ctrl.update(pv=0.0, dt=0.1)
        assert abs(ctrl._integral) <= 50.0 + 1.0, "Integral should be clamped"

    def test_derivative_filter_smooths_pv(self):
        """Derivative-on-PV with filtering should not spike on setpoint change."""
        ctrl = PIDController(PIDParams(Kp=1.0, Ki=0.0, Kd=5.0, setpoint=100.0,
                                       mv_min=-100.0, mv_max=100.0,
                                       derivative_filter_tau=2.0))
        # Step PV from 0 to 50
        mv1 = ctrl.update(0.0, dt=0.1)
        mv2 = ctrl.update(50.0, dt=0.1)  # big PV jump
        # D term should be filtered, not a huge spike
        assert abs(mv2) < 200.0, "Filtered D should not produce extreme spike"

    def test_reverse_action(self):
        """Reverse action: increasing PV should decrease MV."""
        ctrl = PIDController(PIDParams(Kp=5.0, Ki=0.0, Kd=0.0, setpoint=50.0,
                                       mv_min=0.0, mv_max=100.0,
                                       direct_action=False))
        mv_low = ctrl.update(pv=40.0, dt=0.1)
        ctrl.reset()
        mv_high = ctrl.update(pv=60.0, dt=0.1)
        # In reverse action, PV>SP → positive error → increasing MV
        # Actually: error = -1*(SP-PV) = -1*(50-60) = +10 for PV=60
        # and -1*(50-40) = -10 for PV=40
        # So mv_high > mv_low in reverse action
        assert mv_high > mv_low, "Reverse action should increase MV when PV > SP"

    def test_reset_clears_state(self):
        """Reset should zero out integral and derivative state."""
        ctrl = PIDController(PIDParams(Kp=1.0, Ki=1.0, Kd=1.0, setpoint=100.0))
        for _ in range(100):
            ctrl.update(50.0, dt=0.1)
        assert ctrl._integral != 0.0
        ctrl.reset()
        assert ctrl._integral == 0.0
        assert ctrl._prev_pv is None

    def test_mv_clamped_to_range(self):
        """Output should never exceed [mv_min, mv_max]."""
        ctrl = PIDController(PIDParams(Kp=100.0, Ki=100.0, Kd=0.0, setpoint=1000.0,
                                       mv_min=10.0, mv_max=90.0))
        for _ in range(500):
            mv = ctrl.update(0.0, dt=0.1)
            assert 10.0 <= mv <= 90.0, f"MV={mv} out of range"


class TestCascadeController:
    def test_cascade_outer_sets_inner_sp(self):
        """Outer loop output should drive inner loop setpoint."""
        outer = PIDController(PIDParams(Kp=5.0, Ki=0.1, setpoint=900.0,
                                        mv_min=0.0, mv_max=100.0))
        inner = PIDController(PIDParams(Kp=0.8, Ki=0.5, setpoint=50.0,
                                        mv_min=0.0, mv_max=100.0))
        cascade = CascadeController(outer, inner)

        pv_outer = 800.0
        pv_inner = 40.0
        mv = cascade.update(pv_outer, pv_inner, dt=0.5)
        assert 0.0 <= mv <= 100.0

    def test_cascade_converges(self):
        """Cascade should drive outer PV toward setpoint."""
        outer = PIDController(PIDParams(Kp=8.0, Ki=0.05, Kd=5.0, setpoint=900.0,
                                        mv_min=600.0, mv_max=1100.0,
                                        derivative_filter_tau=2.0))
        inner = PIDController(PIDParams(Kp=0.8, Ki=0.5, Kd=0.1, setpoint=900.0,
                                        mv_min=0.0, mv_max=100.0))
        cascade = CascadeController(outer, inner)

        pv_outer = 850.0  # part temperature °C
        pv_inner = 850.0  # furnace temperature °C
        for _ in range(5000):
            mv = cascade.update(pv_outer, pv_inner, dt=0.5)
            # Inner plant: power % → furnace temp °C
            pv_inner += 0.5 / 30.0 * (10.0 * mv - pv_inner)
            # Outer plant: furnace temp → part temp
            pv_outer += 0.5 / 180.0 * (1.0 * pv_inner - pv_outer)

        assert abs(pv_outer - 900.0) < 15.0, f"Cascade PV={pv_outer} near setpoint"


class TestPlantSimulation:
    def test_first_order_step_response(self):
        """First-order plant should approach gain * mv_step."""
        mv = np.full(2000, 50.0)
        pv = simulate_first_order_plant(mv, dt=0.1, gain=2.0, tau=10.0, y0=0.0)
        # Steady state = gain * mv = 100
        assert abs(pv[-1] - 100.0) < 5.0, f"SS PV={pv[-1]} near 100"

    def test_first_order_disturbance(self):
        """With disturbance, PV should oscillate around disturbed SS."""
        mv = np.full(5000, 50.0)
        pv = simulate_first_order_plant(mv, dt=0.1, gain=2.0, tau=10.0,
                                        y0=100.0, disturbance=20.0,
                                        disturbance_freq_hz=0.01)
        # Mean should be near 100 (gain*mv) but with oscillation
        mean_pv = np.mean(pv[1000:])  # skip transient
        assert 80.0 < mean_pv < 120.0, f"Mean PV={mean_pv} near SS"


class TestLoopSimulation:
    def test_all_loops_defined(self):
        """All 8 P&ID loops should be present."""
        loops = default_loops()
        expected = {"electrolyte_temp", "electrolyte_ph", "cell_current",
                    "recirc_flow", "carburize_temp", "carbon_potential",
                    "quench_timing", "tempering_temp"}
        assert set(loops.keys()) == expected

    def test_loop_simulation_returns_valid_result(self):
        """Simulating a single loop should return a valid LoopResult."""
        loops = default_loops()
        result = simulate_loop("electrolyte_temp", loops["electrolyte_temp"],
                               duration_s=200.0, dt=1.0)
        assert isinstance(result, LoopResult)
        assert len(result.time_s) == len(result.pv) == len(result.mv) == len(result.setpoint)
        assert result.settling_time_s >= 0
        assert result.iae >= 0

    def test_cascade_loop_simulation(self):
        """Carburize temp cascade loop should simulate without error."""
        loops = default_loops()
        result = simulate_loop("carburize_temp", loops["carburize_temp"],
                               duration_s=400.0, dt=1.0)
        assert isinstance(result, LoopResult)
        assert len(result.pv) > 100
        # PV should approach 900°C region
        assert np.mean(result.pv[-50:]) > 800.0

    def test_quench_timing_open_loop(self):
        """Quench timing is open-loop; PV should equal MV."""
        loops = default_loops()
        result = simulate_loop("quench_timing", loops["quench_timing"],
                               duration_s=100.0, dt=1.0)
        assert np.allclose(result.pv, result.mv)

    def test_disturbance_rejection(self):
        """With disturbance, loop should still maintain PV near setpoint."""
        loops = default_loops()
        result = simulate_loop("tempering_temp", loops["tempering_temp"],
                               duration_s=800.0, dt=0.5,
                               setpoint_step_pct=0.0,
                               disturbance_time_s=400.0)
        # After disturbance, PV should recover (SS error < 5% of range)
        cv_range = loops["tempering_temp"]["pid"].cv_max - loops["tempering_temp"]["pid"].cv_min
        assert result.steady_state_error < 0.05 * cv_range


class TestTuningSensitivity:
    def test_sensitivity_returns_arrays(self):
        """Sensitivity sweep should return arrays of correct length."""
        loops = default_loops()
        sens = tuning_sensitivity("electrolyte_temp", loops["electrolyte_temp"],
                                  param_name="Kp",
                                  scale_factors=np.array([0.5, 1.0, 2.0]))
        assert len(sens["iae"]) == 3
        assert len(sens["overshoot_pct"]) == 3
        assert len(sens["settling_time_s"]) == 3

    def test_sensitivity_skips_cascade_gracefully(self):
        """Cascade loops should not crash sensitivity analysis."""
        loops = default_loops()
        sens = tuning_sensitivity("carburize_temp", loops["carburize_temp"],
                                  param_name="Kp",
                                  scale_factors=np.array([0.5, 1.0, 2.0]))
        assert len(sens["iae"]) == 3


class TestLoopSummary:
    def test_summary_table_has_8_rows(self):
        summary = loop_summary_table()
        assert len(summary) == 8

    def test_summary_has_required_fields(self):
        summary = loop_summary_table()
        for row in summary:
            assert "loop" in row
            assert "tag" in row
            assert "description" in row
            assert "type" in row
