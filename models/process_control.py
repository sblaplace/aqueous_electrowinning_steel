"""
Process control model for the pilot P&ID — PID loops, cascade control,
anti-windup, derivative filtering, and closed-loop simulation.

Defines 8 control loops from the P&ID:

1. Electrolyte temperature  TT-201  -> HE-201 cooling water valve
2. Electrolyte pH           pHAT-101 -> acid/base dosing pump
3. Cell current             CT-201  -> rectifier output
4. Recirculation flow       FT-201  -> VFD on P-201
5. Carburizing temperature  TT-501  -> F-501 furnace power (cascade inner)
6. Carbon potential          AIT-501 -> gas manifold FCs (CO/CO2/CH4/H2)
7. Quench timing            TT-502  -> quench immersion delay (open-loop)
8. Tempering temperature    TT-503  -> F-503 power

Cascade: loop 5 (carburizing temp) has an inner furnace-power loop and
an outer temperature loop.

All screening tuning — must be refined with plant step-test data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import math
import numpy as np


# ── PID Controller ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class PIDParams:
    """Tuning parameters for a single PID loop."""

    Kp: float = 1.0           # proportional gain
    Ki: float = 0.0           # integral gain (1/s)
    Kd: float = 0.0           # derivative gain (s)
    setpoint: float = 0.0     # target PV
    mv_min: float = 0.0       # manipulated variable lower limit
    mv_max: float = 100.0     # manipulated variable upper limit
    cv_min: float = -1e6      # controlled variable (PV) lower bound (info only)
    cv_max: float = 1e6       # controlled variable (PV) upper bound (info only)
    derivative_filter_tau: float = 0.1   # low-pass time constant on D term (s)
    anti_windup_limit: float = 1e6       # integral accumulator clamp magnitude
    direct_action: bool = True           # True = increasing PV increases MV


class PIDController:
    """
    Standard PID controller with:
    - Anti-windup integral clamping
    - Derivative-on-PV (low-pass filtered) to avoid setpoint kick
    - Direct/reverse action selection
    """

    def __init__(self, params: Optional[PIDParams] = None):
        self.params = params or PIDParams()
        self._integral = 0.0
        self._prev_pv = None
        self._prev_d_filtered = 0.0

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_pv = None
        self._prev_d_filtered = 0.0

    def update(self, pv: float, dt: float) -> float:
        """
        Compute controller output (MV) given current process variable and timestep.

        Parameters
        ----------
        pv : float
            Current process variable measurement.
        dt : float
            Timestep in seconds. Must be > 0.

        Returns
        -------
        float
            Manipulated variable (clamped to [mv_min, mv_max]).
        """
        if dt <= 0:
            raise ValueError("dt must be positive")

        p = self.params
        sign = 1.0 if p.direct_action else -1.0
        error = sign * (p.setpoint - pv)

        # Proportional
        P_term = p.Kp * error

        # Integral with anti-windup clamping
        self._integral += error * dt
        self._integral = np.clip(self._integral,
                                 -p.anti_windup_limit,
                                  p.anti_windup_limit)
        I_term = p.Ki * self._integral

        # Derivative on filtered PV (not error) to avoid setpoint kick
        if self._prev_pv is None:
            D_term = 0.0
        else:
            dpv = (pv - self._prev_pv) / dt
            alpha = dt / (p.derivative_filter_tau + dt)
            d_raw = -sign * dpv  # negative because derivative on PV
            d_filtered = alpha * d_raw + (1.0 - alpha) * self._prev_d_filtered
            D_term = p.Kd * d_filtered
            self._prev_d_filtered = d_filtered
        self._prev_pv = pv

        mv_unclamped = P_term + I_term + D_term
        mv = np.clip(mv_unclamped, p.mv_min, p.mv_max)

        # Back-calculation anti-windup: if output is saturated, freeze integral
        if mv != mv_unclamped:
            self._integral -= error * dt  # undo this step's integration

        return float(mv)


# ── Cascade Controller ─────────────────────────────────────────────────

class CascadeController:
    """
    Two-loop cascade: outer loop output becomes inner loop setpoint.

    Typical: outer = temperature, inner = furnace power.
    """

    def __init__(self, outer: PIDController, inner: PIDController):
        self.outer = outer
        self.inner = inner

    def reset(self) -> None:
        self.outer.reset()
        self.inner.reset()

    def update(self, pv_outer: float, pv_inner: float, dt: float) -> float:
        """
        Run one cascade step: outer sets inner setpoint, inner produces MV.
        Returns the final manipulated variable (inner loop output).
        """
        inner_sp = self.outer.update(pv_outer, dt)
        self.inner.params = PIDParams(
            Kp=self.inner.params.Kp,
            Ki=self.inner.params.Ki,
            Kd=self.inner.params.Kd,
            setpoint=inner_sp,
            mv_min=self.inner.params.mv_min,
            mv_max=self.inner.params.mv_max,
            cv_min=self.inner.params.cv_min,
            cv_max=self.inner.params.cv_max,
            derivative_filter_tau=self.inner.params.derivative_filter_tau,
            anti_windup_limit=self.inner.params.anti_windup_limit,
            direct_action=self.inner.params.direct_action,
        )
        return self.inner.update(pv_inner, dt)


# ── Control Loop Definitions (from P&ID) ───────────────────────────────

def default_loops() -> Dict[str, Dict[str, Any]]:
    """
    Return the 8 P&ID control loops with screening tuning.

    Each entry has: description, tag, PIDParams (or cascade params),
    plant model parameters for simulation.
    """
    return {
        "electrolyte_temp": {
            "description": "Electrolyte temperature TT-201 -> HE-201 cooling water valve",
            "tag": "TIC-201",
            "pid": PIDParams(
                Kp=3.0,       # % valve per °C error
                Ki=0.08,      # reset ~12 min
                Kd=2.0,
                setpoint=65.0,
                mv_min=0.0,   # valve % closed
                mv_max=100.0,
                cv_min=50.0,
                cv_max=80.0,
                derivative_filter_tau=0.5,
                direct_action=False,  # cooling: more valve = less temp
            ),
            "plant": {
                "type": "first_order",
                "gain": -0.3,       # °C per % valve (cooling)
                "tau": 120.0,       # s thermal time constant
                "load": 80.0,       # °C heat load (uncooled equilibrium)
                "disturbance": 2.0, # °C disturbance amplitude
            },
        },
        "electrolyte_ph": {
            "description": "Electrolyte pH pHAT-101 -> acid/base dosing pump",
            "tag": "pHIC-101",
            "pid": PIDParams(
                Kp=10.0,      # mL/min per pH unit error
                Ki=0.3,
                Kd=2.0,
                setpoint=2.0, # acidic mode
                mv_min=-50.0, # negative = acid, positive = base
                mv_max=50.0,
                cv_min=0.5,
                cv_max=14.0,
                derivative_filter_tau=1.0,
            ),
            "plant": {
                "type": "first_order",
                "gain": 0.04,       # pH per mL/min (positive: base raises pH)
                "tau": 60.0,
                "load": 4.0,        # neutral feed pH bias
                "disturbance": 0.1,
            },
        },
        "cell_current": {
            "description": "Cell current CT-201 -> rectifier output",
            "tag": "CIC-201",
            "pid": PIDParams(
                Kp=0.5,       # A per A error
                Ki=0.1,
                Kd=0.05,
                setpoint=400.0, # mA/cm² × area → A
                mv_min=0.0,
                mv_max=600.0,
                cv_min=0.0,
                cv_max=600.0,
                derivative_filter_tau=0.2,
            ),
            "plant": {
                "type": "first_order",
                "gain": 0.95,        # A output per A command (slightly <1)
                "tau": 2.0,
                "load": 20.0,        # A baseline offset
                "disturbance": 10.0,
            },
        },
        "recirc_flow": {
            "description": "Recirculation flow FT-201 -> VFD on P-201",
            "tag": "FIC-201",
            "pid": PIDParams(
                Kp=2.0,       # Hz per L/min error
                Ki=0.3,
                Kd=0.2,
                setpoint=50.0, # L/min (residence time based)
                mv_min=0.0,   # VFD Hz
                mv_max=60.0,
                cv_min=0.0,
                cv_max=80.0,
                derivative_filter_tau=0.3,
            ),
            "plant": {
                "type": "first_order",
                "gain": 1.2,         # L/min per Hz
                "tau": 5.0,
                "load": 0.0,
                "disturbance": 3.0,
            },
        },
        "carburize_temp": {
            "description": "Carburizing temperature TT-501 -> furnace power (cascade)",
            "tag": "TIC-501",
            "cascade": True,
            "outer_pid": PIDParams(
                Kp=8.0,       # % power per °C error
                Ki=0.05,
                Kd=5.0,
                setpoint=900.0,
                mv_min=600.0, # inner SP range = furnace temp °C
                mv_max=1100.0,
                cv_min=800.0,
                cv_max=1000.0,
                derivative_filter_tau=2.0,
            ),
            "inner_pid": PIDParams(
                Kp=0.8,       # power response per °C error
                Ki=0.5,
                Kd=0.1,
                setpoint=900.0,  # updated by outer
                mv_min=0.0,      # furnace power %
                mv_max=100.0,
                cv_min=600.0,    # furnace temp °C
                cv_max=1200.0,
                derivative_filter_tau=0.5,
            ),
            "plant": {
                "type": "first_order",
                "gain_inner": 10.0,  # °C per % power
                "tau_inner": 30.0,
                "load_inner": 0.0,   # no external load on furnace
                "gain_outer": 1.0,   # furnace temp → part temp (1:1)
                "tau_outer": 180.0,
                "load_outer": 0.0,   # no external load on part temp
                "disturbance": 5.0,
            },
        },
        "carbon_potential": {
            "description": "Carbon potential AIT-501 O2 probe -> gas manifold FCs",
            "tag": "AIC-501",
            "pid": PIDParams(
                Kp=80.0,      # % CO per unit aC error (aggressive: low-gain process)
                Ki=0.3,
                Kd=2.0,
                setpoint=1.0, # a_C = 1.0 (graphite ref)
                mv_min=0.0,
                mv_max=100.0, # CO flow %
                cv_min=0.0,
                cv_max=2.0,
                derivative_filter_tau=1.5,
            ),
            "plant": {
                "type": "first_order",
                "gain": 0.02,        # a_C per % CO flow
                "tau": 60.0,
                "load": -0.5,        # baseline low aC without enrichment
                "disturbance": 0.05,
            },
        },
        "quench_timing": {
            "description": "Quench timing TT-502 -> immersion delay (open-loop lookup)",
            "tag": "TS-502",
            "pid": PIDParams(
                Kp=0.0, Ki=0.0, Kd=0.0,  # open-loop
                setpoint=0.0,
                mv_min=0.0,
                mv_max=300.0,  # seconds delay
            ),
            "plant": {
                "type": "lookup",
                "thickness_to_delay_um_s": {500: 5.0, 1000: 15.0, 1500: 30.0, 2000: 60.0},
            },
        },
        "tempering_temp": {
            "description": "Tempering temperature TT-503 -> F-503 power",
            "tag": "TIC-503",
            "pid": PIDParams(
                Kp=4.0,
                Ki=0.06,
                Kd=3.0,
                setpoint=200.0,  # per hardness target
                mv_min=0.0,
                mv_max=100.0,
                cv_min=150.0,
                cv_max=650.0,
                derivative_filter_tau=1.5,
            ),
            "plant": {
                "type": "first_order",
                "gain": 4.0,         # °C per % power
                "tau": 120.0,
                "load": 0.0,         # no external heat load
                "disturbance": 3.0,
            },
        },
    }


# ── First-order plant model ────────────────────────────────────────────

def simulate_first_order_plant(
    mv_history: np.ndarray,
    dt: float,
    gain: float,
    tau: float,
    y0: float = 0.0,
    disturbance: float = 0.0,
    disturbance_freq_hz: float = 0.001,
    load: float = 0.0,
) -> np.ndarray:
    """
    Simulate a first-order (FOPTD) plant: tau * dy/dt + y = gain * mv + load + d(t).

    Parameters
    ----------
    mv_history : array of manipulated variable values
    dt : timestep (s)
    gain : process gain
    tau : time constant (s)
    y0 : initial PV
    disturbance : amplitude of sinusoidal disturbance
    disturbance_freq_hz : frequency of disturbance
    load : constant load/bias (heat input, feed composition, etc.)

    Returns
    -------
    np.ndarray of PV values matching mv_history length
    """
    n = len(mv_history)
    pv = np.zeros(n)
    pv[0] = y0
    for i in range(n - 1):
        t = i * dt
        d = disturbance * math.sin(2.0 * math.pi * disturbance_freq_hz * t)
        dy = (dt / tau) * (gain * mv_history[i] + load + d - pv[i])
        pv[i + 1] = pv[i] + dy
    return pv


# ── Closed-loop simulation ─────────────────────────────────────────────

@dataclass
class LoopResult:
    """Result of simulating a single control loop."""
    time_s: np.ndarray
    setpoint: np.ndarray
    pv: np.ndarray
    mv: np.ndarray
    loop_name: str
    tag: str
    settling_time_s: float = 0.0
    overshoot_pct: float = 0.0
    steady_state_error: float = 0.0
    iae: float = 0.0  # integral absolute error


def simulate_loop(
    loop_name: str,
    loop_cfg: Dict[str, Any],
    duration_s: float = 600.0,
    dt: float = 0.5,
    setpoint_step_pct: float = 10.0,
    disturbance_time_s: float = 300.0,
) -> LoopResult:
    """
    Simulate one control loop closed-loop with optional setpoint step
    and disturbance injection.

    Parameters
    ----------
    loop_name : str
    loop_cfg : dict from default_loops()
    duration_s : total simulation time
    dt : timestep
    setpoint_step_pct : % of CV range for setpoint step at t=0
    disturbance_time_s : time to inject disturbance step
    """
    plant = loop_cfg["plant"]

    # Handle open-loop lookup (quench timing)
    if plant.get("type") == "lookup":
        n = int(duration_s / dt) + 1
        t = np.arange(n) * dt
        sp = np.zeros(n)
        pv = np.zeros(n)
        mv = np.zeros(n)
        # open loop: MV = lookup(thickness), PV = MV (no feedback)
        for i in range(n):
            # 1000 µm default
            mv[i] = 15.0  # 15 s delay at 1000 µm
            pv[i] = mv[i]
            sp[i] = mv[i]
        return LoopResult(
            time_s=t, setpoint=sp, pv=pv, mv=mv,
            loop_name=loop_name, tag=loop_cfg.get("tag", ""),
        )

    # Cascade loop
    if loop_cfg.get("cascade"):
        return _simulate_cascade_loop(loop_name, loop_cfg, duration_s, dt,
                                      setpoint_step_pct, disturbance_time_s)

    # Single-loop PID
    pid_params: PIDParams = loop_cfg["pid"]
    gain = plant["gain"]
    tau = plant["tau"]
    dist_amp = plant.get("disturbance", 0.0)
    load = plant.get("load", 0.0)

    # Setpoint step
    cv_range = pid_params.cv_max - pid_params.cv_min
    sp0 = pid_params.setpoint
    sp_target = sp0 + setpoint_step_pct / 100.0 * cv_range

    n = int(duration_s / dt) + 1
    t = np.arange(n) * dt

    ctrl = PIDController(pid_params)
    sp = np.full(n, sp_target)
    # first quarter at original setpoint for baseline
    n_baseline = n // 4
    sp[:n_baseline] = sp0

    pv_arr = np.zeros(n)
    mv_arr = np.zeros(n)
    pv_arr[0] = sp0

    for i in range(n - 1):
        # Update setpoint for controller
        ctrl.params = PIDParams(
            Kp=ctrl.params.Kp, Ki=ctrl.params.Ki, Kd=ctrl.params.Kd,
            setpoint=sp[i],
            mv_min=ctrl.params.mv_min, mv_max=ctrl.params.mv_max,
            cv_min=ctrl.params.cv_min, cv_max=ctrl.params.cv_max,
            derivative_filter_tau=ctrl.params.derivative_filter_tau,
            anti_windup_limit=ctrl.params.anti_windup_limit,
            direct_action=ctrl.params.direct_action,
        )

        # Disturbance step at disturbance_time_s
        d = 0.0
        if t[i] >= disturbance_time_s:
            d = dist_amp

        mv = ctrl.update(pv_arr[i], dt)
        mv_arr[i] = mv

        # Plant: tau * dy/dt + y = gain * mv + load + d
        dy = (dt / tau) * (gain * mv + load + d - pv_arr[i])
        pv_arr[i + 1] = pv_arr[i] + dy

    mv_arr[-1] = mv_arr[-2] if n > 1 else 0.0

    # Metrics
    error = sp - pv_arr
    iae = float(np.trapezoid(np.abs(error), t)) if hasattr(np, 'trapezoid') else float(np.trapz(np.abs(error), t))

    # Overshoot (relative to final setpoint)
    final_sp = sp[-1]
    if abs(final_sp - sp0) > 1e-6:
        if final_sp > sp0:
            overshoot_pct = float((np.max(pv_arr[n_baseline:]) - final_sp) / abs(final_sp - sp0) * 100.0)
        else:
            overshoot_pct = float((final_sp - np.min(pv_arr[n_baseline:])) / abs(final_sp - sp0) * 100.0)
        overshoot_pct = max(0.0, overshoot_pct)
    else:
        overshoot_pct = 0.0

    # Settling time: last time PV leaves ±2% of setpoint range
    tol = 0.02 * abs(final_sp - sp0) if abs(final_sp - sp0) > 1e-6 else 0.01 * abs(final_sp) + 1e-6
    settled_mask = np.abs(error[n_baseline:]) <= tol
    if np.all(settled_mask):
        settling_time_s = float(t[n_baseline])
    elif np.any(settled_mask):
        # last violation after baseline
        violations = np.where(~settled_mask)[0]
        settling_time_s = float(t[n_baseline + violations[-1]]) if len(violations) > 0 else float(t[n_baseline])
    else:
        settling_time_s = float(duration_s)

    # Steady-state error (last 10% of simulation)
    ss_window = n // 10
    ss_error = float(np.mean(np.abs(error[-ss_window:])))

    return LoopResult(
        time_s=t, setpoint=sp, pv=pv_arr, mv=mv_arr,
        loop_name=loop_name, tag=loop_cfg.get("tag", ""),
        settling_time_s=settling_time_s,
        overshoot_pct=overshoot_pct,
        steady_state_error=ss_error,
        iae=iae,
    )


def _simulate_cascade_loop(
    loop_name: str,
    loop_cfg: Dict[str, Any],
    duration_s: float,
    dt: float,
    setpoint_step_pct: float,
    disturbance_time_s: float,
) -> LoopResult:
    """Simulate cascade control loop (carburizing temperature)."""
    outer_params: PIDParams = loop_cfg["outer_pid"]
    inner_params: PIDParams = loop_cfg["inner_pid"]
    plant = loop_cfg["plant"]

    gain_inner = plant["gain_inner"]
    tau_inner = plant["tau_inner"]
    load_inner = plant.get("load_inner", 0.0)
    gain_outer = plant.get("gain_outer", 1.0)
    tau_outer = plant.get("tau_outer", tau_inner * 3)
    load_outer = plant.get("load_outer", 0.0)
    dist_amp = plant.get("disturbance", 0.0)

    cv_range = outer_params.cv_max - outer_params.cv_min
    sp0 = outer_params.setpoint
    sp_target = sp0 + setpoint_step_pct / 100.0 * cv_range

    n = int(duration_s / dt) + 1
    t = np.arange(n) * dt

    cascade = CascadeController(
        PIDController(outer_params),
        PIDController(inner_params),
    )

    sp = np.full(n, sp_target)
    n_baseline = n // 4
    sp[:n_baseline] = sp0

    pv_outer = np.zeros(n)
    pv_inner = np.zeros(n)
    mv_arr = np.zeros(n)
    pv_outer[0] = sp0
    pv_inner[0] = inner_params.setpoint

    for i in range(n - 1):
        # Update outer setpoint dynamically
        cascade.outer.params = PIDParams(
            Kp=cascade.outer.params.Kp, Ki=cascade.outer.params.Ki,
            Kd=cascade.outer.params.Kd, setpoint=sp[i],
            mv_min=cascade.outer.params.mv_min, mv_max=cascade.outer.params.mv_max,
            cv_min=cascade.outer.params.cv_min, cv_max=cascade.outer.params.cv_max,
            derivative_filter_tau=cascade.outer.params.derivative_filter_tau,
            anti_windup_limit=cascade.outer.params.anti_windup_limit,
            direct_action=cascade.outer.params.direct_action,
        )

        d = 0.0
        if t[i] >= disturbance_time_s:
            d = dist_amp

        mv = cascade.update(pv_outer[i], pv_inner[i], dt)
        mv_arr[i] = mv

        # Inner plant: furnace power → temperature contribution
        dy_inner = (dt / tau_inner) * (gain_inner * mv + load_inner - pv_inner[i])
        pv_inner[i + 1] = pv_inner[i] + dy_inner

        # Outer plant: inner PV propagates to outer PV + disturbance
        dy_outer = (dt / tau_outer) * (gain_outer * pv_inner[i] + load_outer + d - pv_outer[i])
        pv_outer[i + 1] = pv_outer[i] + dy_outer

    mv_arr[-1] = mv_arr[-2] if n > 1 else 0.0

    error = sp - pv_outer
    iae = float(np.trapezoid(np.abs(error), t)) if hasattr(np, 'trapezoid') else float(np.trapz(np.abs(error), t))

    final_sp = sp[-1]
    if abs(final_sp - sp0) > 1e-6:
        overshoot_pct = max(0.0, float((np.max(pv_outer[n_baseline:]) - final_sp) / abs(final_sp - sp0) * 100.0))
    else:
        overshoot_pct = 0.0

    tol = 0.02 * abs(final_sp - sp0) if abs(final_sp - sp0) > 1e-6 else 0.01 * abs(final_sp) + 1e-6
    settled_mask = np.abs(error[n_baseline:]) <= tol
    if np.all(settled_mask):
        settling_time_s = float(t[n_baseline])
    elif np.any(settled_mask):
        violations = np.where(~settled_mask)[0]
        settling_time_s = float(t[n_baseline + violations[-1]]) if len(violations) > 0 else float(t[n_baseline])
    else:
        settling_time_s = float(duration_s)

    ss_window = n // 10
    ss_error = float(np.mean(np.abs(error[-ss_window:])))

    return LoopResult(
        time_s=t, setpoint=sp, pv=pv_outer, mv=mv_arr,
        loop_name=loop_name, tag=loop_cfg.get("tag", ""),
        settling_time_s=settling_time_s,
        overshoot_pct=overshoot_pct,
        steady_state_error=ss_error,
        iae=iae,
    )


# ── Tuning sensitivity analysis ────────────────────────────────────────

def tuning_sensitivity(
    loop_name: str,
    loop_cfg: Dict[str, Any],
    param_name: str = "Kp",
    scale_factors: Optional[np.ndarray] = None,
    duration_s: float = 600.0,
    dt: float = 0.5,
) -> Dict[str, np.ndarray]:
    """
    Sweep one PID parameter across scale factors and record IAE, overshoot,
    settling time for each.

    Returns dict with scale_factors, iae, overshoot_pct, settling_time_s.
    """
    if scale_factors is None:
        scale_factors = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0])

    if loop_cfg.get("cascade") or loop_cfg["plant"].get("type") == "lookup":
        # Skip cascade/lookup for sensitivity (uses outer PID)
        pid_params = loop_cfg.get("outer_pid", loop_cfg.get("pid", PIDParams()))
    else:
        pid_params = loop_cfg["pid"]

    iae_arr = np.zeros(len(scale_factors))
    os_arr = np.zeros(len(scale_factors))
    st_arr = np.zeros(len(scale_factors))

    for i, sf in enumerate(scale_factors):
        # Build modified config
        cfg_copy = dict(loop_cfg)
        orig_val = getattr(pid_params, param_name)
        new_params = PIDParams(
            **{**{k: getattr(pid_params, k) for k in PIDParams.__dataclass_fields__},
               param_name: orig_val * sf}
        )
        if loop_cfg.get("cascade"):
            cfg_copy["outer_pid"] = new_params
        else:
            cfg_copy["pid"] = new_params

        result = simulate_loop(loop_name, cfg_copy, duration_s, dt)
        iae_arr[i] = result.iae
        os_arr[i] = result.overshoot_pct
        st_arr[i] = result.settling_time_s

    return {
        "scale_factors": scale_factors,
        "iae": iae_arr,
        "overshoot_pct": os_arr,
        "settling_time_s": st_arr,
        "param_name": param_name,
        "loop_name": loop_name,
    }


# ── Loop summary table ─────────────────────────────────────────────────

def loop_summary_table() -> List[Dict[str, Any]]:
    """Return a list of dicts summarizing all 8 control loops."""
    loops = default_loops()
    rows = []
    for name, cfg in loops.items():
        if cfg.get("cascade"):
            op = cfg["outer_pid"]
            ip = cfg["inner_pid"]
            rows.append({
                "loop": name,
                "tag": cfg.get("tag", ""),
                "description": cfg["description"],
                "type": "cascade",
                "outer_Kp": op.Kp, "outer_Ki": op.Ki, "outer_Kd": op.Kd,
                "inner_Kp": ip.Kp, "inner_Ki": ip.Ki, "inner_Kd": ip.Kd,
                "setpoint": op.setpoint,
                "MV_range": f"[{op.mv_min}, {op.mv_max}]",
                "PV_range": f"[{op.cv_min}, {op.cv_max}]",
            })
        elif cfg["plant"].get("type") == "lookup":
            rows.append({
                "loop": name,
                "tag": cfg.get("tag", ""),
                "description": cfg["description"],
                "type": "open-loop",
                "Kp": 0, "Ki": 0, "Kd": 0,
                "setpoint": "N/A (lookup)",
                "MV_range": f"[0, {cfg['pid'].mv_max}]",
                "PV_range": "N/A",
            })
        else:
            p = cfg["pid"]
            rows.append({
                "loop": name,
                "tag": cfg.get("tag", ""),
                "description": cfg["description"],
                "type": "PID",
                "Kp": p.Kp, "Ki": p.Ki, "Kd": p.Kd,
                "setpoint": p.setpoint,
                "MV_range": f"[{p.mv_min}, {p.mv_max}]",
                "PV_range": f"[{p.cv_min}, {p.cv_max}]",
            })
    return rows
