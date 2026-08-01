"""
Transient process model for aqueous electrowinning steel plant.

Models non-steady-state operation: plant startup, shutdown, load changes,
and upset conditions. Uses first-order thermal/process lags and explicit
quality-degradation tracking.

This is a screening model — each subsystem (electrolyte, cell, furnace,
quench) is represented by a simple ODE with time constants from equipment
specs. Upset scenarios use threshold-based quality flags. No CFD, no
electrochemical kinetics detail, no real-time PLC emulation.

References (screening)
----------------------
* Thermal lag: τ = m·cp / (U·A) for stirred tanks, furnaces
* Electrolyte heating: typical 1-2 kW/m³ heater, 1000 L tank → τ ~30-60 min
* Furnace ramp: 5-10°C/min controlled ramp to avoid thermal shock
* Cell energization: 5-30 min ramp to avoid inrush and deposit stress
* O2 probe stabilization: 10-30 min after gas composition change
* pH drift without dosing: depends on electrolyte chemistry, ~0.1-0.5 pH/hr
  for buffered citrate systems, ~1-2 pH/hr unbuffered
* Fe(OH)3 precipitation threshold: pH ~3.5-4.0 for typical Fe²⁺ concentrations
* Deposit quality: cathodic efficiency drops when pH > 4, T > 80°C, or
  current interrupted mid-deposit; dissolution begins within minutes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

import numpy as np


# ─── Constants ──────────────────────────────────────────────────────────
AMBIENT_TEMP_C = 25.0
WATER_BOILING_C = 100.0
FE_OH3_PRECIPITATION_PH = 3.8  # approximate for 1 M Fe²⁺
SAFE_FURNACE_COOL_RATE_C_PER_MIN = 5.0  # max to avoid thermal shock
N2_PURGE_VOLUMES = 5.0  # retort volumes for safe purge


class UpsetType(str, Enum):
    """Enumeration of modeled upset scenarios."""
    POWER_INTERRUPTION = "power_interruption"
    PH_EXCURSION = "ph_excursion"
    TEMPERATURE_EXCURSION = "temperature_excursion"
    GAS_SUPPLY_INTERRUPTION = "gas_supply_interruption"
    CURRENT_INTERRUPTION = "current_interruption"
    RECTIFIER_FAULT = "rectifier_fault"


@dataclass(frozen=True)
class TransientConfig:
    """Operating parameters for the transient simulation.

    All temperatures in °C, times in minutes unless noted.
    """

    # ── Electrolyte ──
    electrolyte_volume_L: float = 1000.0
    electrolyte_operating_temp_C: float = 60.0
    electrolyte_target_pH: float = 2.5
    electrolyte_heater_kW: float = 1.5
    electrolyte_heat_capacity_kJ_per_kgK: float = 4.18  # ~water
    electrolyte_density_kg_per_L: float = 1.05
    electrolyte_temp_time_constant_min: float = 40.0  # τ for first-order lag
    electrolyte_ph_pump_rate_pH_per_min: float = 0.05  # dosing rate
    electrolyte_ph_drift_rate_pH_per_min: float = 0.02  # drift when pump fails
    fe_concentration_M: float = 1.0

    # ── Cell / rectifier ──
    target_current_density_mA_cm2: float = 300.0
    current_ramp_time_min: float = 15.0
    cell_voltage_V: float = 2.5
    current_efficiency_normal: float = 0.95
    electrode_area_m2: float = 1.0
    deposit_dissolution_rate_mg_per_min: float = 5.0  # when unpowered + immersed

    # ── Furnace ──
    furnace_operating_temp_C: float = 900.0
    furnace_ambient_temp_C: float = AMBIENT_TEMP_C
    furnace_ramp_rate_C_per_min: float = 8.0
    furnace_time_constant_min: float = 30.0
    furnace_thermal_mass_kJ_per_K: float = 5000.0
    o2_probe_time_constant_min: float = 15.0
    o2_probe_target_ppm: float = 50.0  # low O2 in carburizing atmosphere

    # ── Quench ──
    quench_operating_temp_C: float = 60.0
    quench_time_constant_min: float = 20.0
    quench_agitation_required: bool = True

    # ── Upset thresholds ──
    max_electrolyte_temp_C: float = 85.0  # boiling risk margin
    max_ph_for_fe_deposition: float = 4.5  # Fe(OH)3 onset
    min_gas_quality_hours: float = 0.5  # before decarburization risk
    deposit_quality_threshold: float = 0.7  # below = scrap

    # ── Simulation ──
    startup_duration_min: float = 120.0
    shutdown_duration_min: float = 90.0
    upset_duration_min: float = 30.0

    def __post_init__(self) -> None:
        if self.electrolyte_volume_L <= 0:
            raise ValueError("electrolyte_volume_L must be positive")
        if self.current_ramp_time_min <= 0:
            raise ValueError("current_ramp_time_min must be positive")
        if self.electrolyte_temp_time_constant_min <= 0:
            raise ValueError("electrolyte_temp_time_constant_min must be positive")
        if self.furnace_time_constant_min <= 0:
            raise ValueError("furnace_time_constant_min must be positive")
        if not 0 < self.current_efficiency_normal <= 1:
            raise ValueError("current_efficiency_normal must be in (0, 1]")
        if self.furnace_operating_temp_C <= self.furnace_ambient_temp_C:
            raise ValueError("furnace_operating_temp_C must exceed ambient")


@dataclass
class TransientResult:
    """Time-series output from a transient simulation."""

    time_min: np.ndarray
    electrolyte_temp_C: np.ndarray
    electrolyte_pH: np.ndarray
    fe_concentration_M: np.ndarray
    current_density_mA_cm2: np.ndarray
    furnace_temp_C: np.ndarray
    o2_probe_ppm: np.ndarray
    quench_temp_C: np.ndarray
    deposit_quality: np.ndarray  # 1.0 = perfect, 0.0 = scrap
    ce_fraction: np.ndarray  # current efficiency
    flags: List[List[str]] = field(default_factory=list)
    scenario: str = ""

    def summary(self) -> Dict[str, Any]:
        """Compact summary of the transient result."""
        n = len(self.time_min)
        flagged = sum(bool(f) for f in self.flags)
        return {
            "scenario": self.scenario,
            "duration_min": float(self.time_min[-1]) if n else 0.0,
            "n_steps": n,
            "final_electrolyte_temp_C": float(self.electrolyte_temp_C[-1]) if n else 0.0,
            "final_pH": float(self.electrolyte_pH[-1]) if n else 0.0,
            "final_fe_M": float(self.fe_concentration_M[-1]) if n else 0.0,
            "final_current_mA_cm2": float(self.current_density_mA_cm2[-1]) if n else 0.0,
            "final_furnace_temp_C": float(self.furnace_temp_C[-1]) if n else 0.0,
            "final_quench_temp_C": float(self.quench_temp_C[-1]) if n else 0.0,
            "final_deposit_quality": float(self.deposit_quality[-1]) if n else 0.0,
            "final_ce": float(self.ce_fraction[-1]) if n else 0.0,
            "min_deposit_quality": float(np.min(self.deposit_quality)) if n else 0.0,
            "flagged_time_steps": flagged,
        }

    def as_columns(self) -> Dict[str, np.ndarray]:
        """Return numeric columns for DataFrame / CSV."""
        return {
            "time_min": self.time_min,
            "electrolyte_temp_C": self.electrolyte_temp_C,
            "electrolyte_pH": self.electrolyte_pH,
            "fe_concentration_M": self.fe_concentration_M,
            "current_density_mA_cm2": self.current_density_mA_cm2,
            "furnace_temp_C": self.furnace_temp_C,
            "o2_probe_ppm": self.o2_probe_ppm,
            "quench_temp_C": self.quench_temp_C,
            "deposit_quality": self.deposit_quality,
            "ce_fraction": self.ce_fraction,
        }


# ─── Helper: first-order lag ────────────────────────────────────────────

def _first_order_lag(
    current: float,
    target: float,
    tau: float,
    dt: float,
) -> float:
    """Exponential approach: x += (target - x) * (1 - exp(-dt/tau))."""
    if tau <= 0:
        return target
    alpha = 1.0 - np.exp(-dt / tau)
    return current + (target - current) * alpha


def _ramp_target(
    t: float,
    ramp_duration: float,
    target_value: float,
    initial_value: float = 0.0,
) -> float:
    """Linear ramp from initial to target over ramp_duration."""
    if ramp_duration <= 0:
        return target_value
    frac = np.clip(t / ramp_duration, 0.0, 1.0)
    return initial_value + (target_value - initial_value) * frac


# ─── Core simulation engine ────────────────────────────────────────────

def _integrate(
    config: TransientConfig,
    duration_min: float,
    dt_min: float,
    current_profile_fn,
    ph_pump_on: bool,
    furnace_target_fn,
    gas_on: bool,
    quench_target_fn,
    scenario: str,
    initial_state: Optional[Dict[str, float]] = None,
) -> TransientResult:
    """Generic forward-Euler integrator for all transient scenarios.

    Each subsystem is a first-order lag toward a time-varying setpoint.
    The caller provides profile functions that return setpoints as a
    function of time, plus on/off flags for pumps and gas supply.

    Parameters
    ----------
    initial_state : dict, optional
        Starting values for state variables. Keys:
        electrolyte_temp_C, electrolyte_pH, fe_concentration_M,
        furnace_temp_C, o2_probe_ppm, quench_temp_C.
        Defaults to ambient/cold conditions when None.
    """
    t = np.arange(0.0, duration_min + 0.5 * dt_min, dt_min)
    n = len(t)

    # Allocate arrays
    elec_temp = np.zeros(n)
    elec_pH = np.zeros(n)
    fe_M = np.zeros(n)
    current = np.zeros(n)
    furn_temp = np.zeros(n)
    o2_ppm = np.zeros(n)
    quench_t = np.zeros(n)
    quality = np.ones(n)
    ce = np.zeros(n)
    flags: List[List[str]] = [[] for _ in range(n)]

    # Initial conditions: ambient (cold start) or provided state
    init = initial_state or {}
    elec_temp[0] = init.get("electrolyte_temp_C", AMBIENT_TEMP_C)
    elec_pH[0] = init.get("electrolyte_pH", config.electrolyte_target_pH + 1.0)
    fe_M[0] = init.get("fe_concentration_M", config.fe_concentration_M)
    furn_temp[0] = init.get("furnace_temp_C", config.furnace_ambient_temp_C)
    o2_ppm[0] = init.get("o2_probe_ppm", 210_000.0)
    quench_t[0] = init.get("quench_temp_C", AMBIENT_TEMP_C)

    for i in range(n):
        ti = t[i]
        fl: List[str] = []

        # ── Current profile ──
        current_setpoint = current_profile_fn(ti)
        current[i] = current_setpoint

        # ── Electrolyte temperature ──
        elec_temp_target = config.electrolyte_operating_temp_C
        if i > 0:
            elec_temp[i] = _first_order_lag(
                elec_temp[i - 1],
                elec_temp_target,
                config.electrolyte_temp_time_constant_min,
                dt_min,
            )

        # ── Electrolyte pH ──
        if i > 0:
            if ph_pump_on:
                # pH dosing toward target
                ph_target = config.electrolyte_target_pH
                rate = config.electrolyte_ph_pump_rate_pH_per_min
                delta = (ph_target - elec_pH[i - 1])
                step = np.clip(delta, -rate * dt_min, rate * dt_min)
                elec_pH[i] = elec_pH[i - 1] + step
            else:
                # Drift (acidic consumption from Fe deposition, natural drift)
                elec_pH[i] = elec_pH[i - 1] + config.electrolyte_ph_drift_rate_pH_per_min * dt_min

        # Fe(OH)3 precipitation risk
        if elec_pH[i] > config.max_ph_for_fe_deposition:
            fl.append(f"ph_excursion_risk_pH_{elec_pH[i]:.1f}")
            # Fe(OH)3 precipitation reduces effective Fe²⁺
            excess_ph = elec_pH[i] - config.max_ph_for_fe_deposition
            fe_loss_rate = 0.01 * excess_ph * dt_min  # screening
            fe_M[i] = max(0.0, (fe_M[i - 1] if i > 0 else fe_M[0]) - fe_loss_rate)
        elif i > 0:
            fe_M[i] = fe_M[i - 1]

        # ── Furnace ──
        furn_setpoint = furnace_target_fn(ti)
        if i > 0:
            furn_temp[i] = _first_order_lag(
                furn_temp[i - 1],
                furn_setpoint,
                config.furnace_time_constant_min,
                dt_min,
            )

        # ── O2 probe ──
        if gas_on:
            o2_target = config.o2_probe_target_ppm
        else:
            o2_target = 210_000.0  # air
        if i > 0:
            o2_ppm[i] = _first_order_lag(
                o2_ppm[i - 1],
                o2_target,
                config.o2_probe_time_constant_min,
                dt_min,
            )

        # ── Quench ──
        quench_setpoint = quench_target_fn(ti)
        if i > 0:
            quench_t[i] = _first_order_lag(
                quench_t[i - 1],
                quench_setpoint,
                config.quench_time_constant_min,
                dt_min,
            )

        # ── Deposit quality ──
        q_prev = quality[i - 1] if i > 0 else quality[0]
        q = q_prev

        # Current-dependent: no current → no deposition or dissolution
        if current[i] <= 0:
            if current[i] == 0 and i > 0 and current[i - 1] > 0:
                fl.append("current_lost")
            # Dissolution when unpowered + immersed
            q -= 0.005 * dt_min  # slow quality loss
        else:
            # Overcurrent fault
            if current[i] > config.target_current_density_mA_cm2 * 1.5:
                fl.append("overcurrent")
                q -= 0.01 * dt_min

        # pH effect on CE
        if elec_pH[i] > config.max_ph_for_fe_deposition:
            q -= 0.002 * (elec_pH[i] - config.max_ph_for_fe_deposition) * dt_min

        # Temperature effect
        if elec_temp[i] > config.max_electrolyte_temp_C:
            fl.append(f"high_electrolyte_temp_{elec_temp[i]:.0f}C")
            q -= 0.003 * (elec_temp[i] - config.max_electrolyte_temp_C) * dt_min

        # Gas interruption effect (decarburization during hold)
        if not gas_on and furn_temp[i] > 500:
            fl.append("gas_interruption_decarb_risk")

        # Reverse polarity fault
        if current[i] < 0:
            fl.append("reverse_polarity")
            q -= 0.05 * dt_min

        quality[i] = np.clip(q, 0.0, 1.0)

        # ── Current efficiency ──
        # Baseline CE, degraded by pH and temperature
        base_ce = config.current_efficiency_normal
        ph_penalty = max(0.0, (elec_pH[i] - config.max_ph_for_fe_deposition)) * 0.10
        temp_penalty = max(0.0, (elec_temp[i] - config.max_electrolyte_temp_C) / 20.0) * 0.05
        ce[i] = np.clip(base_ce - ph_penalty - temp_penalty, 0.0, 1.0)
        if current[i] <= 0:
            ce[i] = 0.0

        flags[i] = fl

    return TransientResult(
        time_min=t,
        electrolyte_temp_C=elec_temp,
        electrolyte_pH=elec_pH,
        fe_concentration_M=fe_M,
        current_density_mA_cm2=current,
        furnace_temp_C=furn_temp,
        o2_probe_ppm=o2_ppm,
        quench_temp_C=quench_t,
        deposit_quality=quality,
        ce_fraction=ce,
        flags=flags,
        scenario=scenario,
    )


# ─── Public simulation functions ───────────────────────────────────────

def simulate_startup(
    config: Optional[TransientConfig] = None,
    dt: float = 1.0,
) -> TransientResult:
    """Simulate plant startup from cold ambient to operating conditions.

    Startup sequence:
    1. Electrolyte preparation: heat to operating temp, adjust pH
    2. Cell energization: ramp current from 0 over current_ramp_time_min
    3. Furnace ramp to operating temperature
    4. Quench system pre-heat

    Parameters
    ----------
    config : TransientConfig, optional
        Operating parameters. Uses defaults if None.
    dt : float
        Time step in minutes.

    Returns
    -------
    TransientResult
        Time series of all state variables.
    """
    cfg = config or TransientConfig()

    def current_profile(t: float) -> float:
        return _ramp_target(
            t - cfg.electrolyte_temp_time_constant_min * 0.5,  # start after initial heating
            cfg.current_ramp_time_min,
            cfg.target_current_density_mA_cm2,
        )

    def furnace_target(t: float) -> float:
        # Start ramping early, reach operating temp
        return min(
            cfg.furnace_ambient_temp_C + cfg.furnace_ramp_rate_C_per_min * t,
            cfg.furnace_operating_temp_C,
        )

    def quench_target(t: float) -> float:
        return cfg.quench_operating_temp_C  # heat immediately

    return _integrate(
        config=cfg,
        duration_min=cfg.startup_duration_min,
        dt_min=dt,
        current_profile_fn=current_profile,
        ph_pump_on=True,
        furnace_target_fn=furnace_target,
        gas_on=True,
        quench_target_fn=quench_target,
        scenario="startup",
    )


def _operating_state(cfg: TransientConfig) -> Dict[str, float]:
    """Return steady-state operating conditions for warm-starting."""
    return {
        "electrolyte_temp_C": cfg.electrolyte_operating_temp_C,
        "electrolyte_pH": cfg.electrolyte_target_pH,
        "fe_concentration_M": cfg.fe_concentration_M,
        "furnace_temp_C": cfg.furnace_operating_temp_C,
        "o2_probe_ppm": cfg.o2_probe_target_ppm,
        "quench_temp_C": cfg.quench_operating_temp_C,
    }


def simulate_shutdown(
    config: Optional[TransientConfig] = None,
    dt: float = 1.0,
) -> TransientResult:
    """Simulate controlled plant shutdown.

    Shutdown sequence:
    1. Ramp current to 0
    2. Hold electrolyte temperature briefly, then let it cool
    3. Furnace controlled cool-down
    4. N2 gas purge

    Parameters
    ----------
    config : TransientConfig, optional
        Operating parameters.
    dt : float
        Time step in minutes.

    Returns
    -------
    TransientResult
    """
    cfg = config or TransientConfig()
    ramp_down_min = 10.0  # time to ramp current to 0

    def current_profile(t: float) -> float:
        # Start from operating and ramp to 0
        if t < ramp_down_min:
            return cfg.target_current_density_mA_cm2 * (1.0 - t / ramp_down_min)
        return 0.0

    furnace_cool_duration = (
        (cfg.furnace_operating_temp_C - AMBIENT_TEMP_C)
        / SAFE_FURNACE_COOL_RATE_C_PER_MIN
    )

    def furnace_target(t: float) -> float:
        temp = cfg.furnace_operating_temp_C - SAFE_FURNACE_COOL_RATE_C_PER_MIN * t
        return max(temp, AMBIENT_TEMP_C)

    def quench_target(t: float) -> float:
        return AMBIENT_TEMP_C  # let quench cool naturally

    return _integrate(
        config=cfg,
        duration_min=cfg.shutdown_duration_min,
        dt_min=dt,
        current_profile_fn=current_profile,
        ph_pump_on=False,  # pumps off during shutdown
        furnace_target_fn=furnace_target,
        gas_on=True,  # purge still running
        quench_target_fn=quench_target,
        scenario="shutdown",
        initial_state=_operating_state(cfg),
    )


def simulate_upset(
    config: Optional[TransientConfig] = None,
    upset_type: str | UpsetType = UpsetType.POWER_INTERRUPTION,
    duration: Optional[float] = None,
    dt: float = 1.0,
) -> TransientResult:
    """Simulate an upset scenario during steady-state operation.

    Starts from operating conditions and introduces the specified fault.

    Parameters
    ----------
    config : TransientConfig, optional
        Operating parameters.
    upset_type : str or UpsetType
        Which upset to simulate.
    duration : float, optional
        Duration in minutes. Uses config.upset_duration_min if None.
    dt : float
        Time step in minutes.

    Returns
    -------
    TransientResult
    """
    cfg = config or TransientConfig()
    if isinstance(upset_type, str):
        upset_type = UpsetType(upset_type)
    dur = duration if duration is not None else cfg.upset_duration_min

    if upset_type == UpsetType.POWER_INTERRUPTION:
        return _upset_power_interruption(cfg, dur, dt)
    elif upset_type == UpsetType.PH_EXCURSION:
        return _upset_ph_excursion(cfg, dur, dt)
    elif upset_type == UpsetType.TEMPERATURE_EXCURSION:
        return _upset_temperature_excursion(cfg, dur, dt)
    elif upset_type == UpsetType.GAS_SUPPLY_INTERRUPTION:
        return _upset_gas_interruption(cfg, dur, dt)
    elif upset_type == UpsetType.CURRENT_INTERRUPTION:
        return _upset_current_interruption(cfg, dur, dt)
    elif upset_type == UpsetType.RECTIFIER_FAULT:
        return _upset_rectifier_fault(cfg, dur, dt)
    else:
        raise ValueError(f"Unknown upset type: {upset_type}")


def _upset_power_interruption(
    cfg: TransientConfig, dur: float, dt: float,
) -> TransientResult:
    """Complete power loss: all electrically-driven systems stop."""
    def current_profile(t: float) -> float:
        return 0.0  # no power

    def furnace_target(t: float) -> float:
        return AMBIENT_TEMP_C  # furnace cools without power

    def quench_target(t: float) -> float:
        return AMBIENT_TEMP_C

    return _integrate(
        config=cfg, duration_min=dur, dt_min=dt,
        current_profile_fn=current_profile,
        ph_pump_on=False,  # pumps dead
        furnace_target_fn=furnace_target,
        gas_on=False,  # no blowers
        quench_target_fn=quench_target,
        scenario="upset_power_interruption",
        initial_state=_operating_state(cfg),
    )


def _upset_ph_excursion(
    cfg: TransientConfig, dur: float, dt: float,
) -> TransientResult:
    """Acid/base pump failure → pH rises (less acidic) → Fe(OH)3 risk."""
    def current_profile(t: float) -> float:
        return cfg.target_current_density_mA_cm2

    def furnace_target(t: float) -> float:
        return cfg.furnace_operating_temp_C

    def quench_target(t: float) -> float:
        return cfg.quench_operating_temp_C

    return _integrate(
        config=cfg, duration_min=dur, dt_min=dt,
        current_profile_fn=current_profile,
        ph_pump_on=False,  # pump failure
        furnace_target_fn=furnace_target,
        gas_on=True,
        quench_target_fn=quench_target,
        scenario="upset_ph_excursion",
        initial_state=_operating_state(cfg),
    )


def _upset_temperature_excursion(
    cfg: TransientConfig, dur: float, dt: float,
) -> TransientResult:
    """Cooling failure → electrolyte overheats."""
    def current_profile(t: float) -> float:
        return cfg.target_current_density_mA_cm2

    # Override: electrolyte keeps heating (simulate stuck heater + no cooling)
    class HotConfig(TransientConfig):
        pass

    # Create a config where the heater overshoots by forcing higher target
    hot_temp = cfg.max_electrolyte_temp_C + 40.0  # stuck heater overshoots well past safe

    def furnace_target(t: float) -> float:
        return cfg.furnace_operating_temp_C

    def quench_target(t: float) -> float:
        return cfg.quench_operating_temp_C

    # We integrate manually for this scenario to override electrolyte target
    t_arr = np.arange(0.0, dur + 0.5 * dt, dt)
    n = len(t_arr)
    elec_temp = np.zeros(n)
    elec_pH = np.zeros(n)
    fe_M = np.zeros(n)
    current = np.zeros(n)
    furn_temp = np.zeros(n)
    o2_ppm = np.zeros(n)
    quench_t = np.zeros(n)
    quality = np.ones(n)
    ce_arr = np.zeros(n)
    flags: List[List[str]] = [[] for _ in range(n)]

    elec_temp[0] = cfg.electrolyte_operating_temp_C  # starts at operating
    elec_pH[0] = cfg.electrolyte_target_pH
    fe_M[0] = cfg.fe_concentration_M
    furn_temp[0] = cfg.furnace_operating_temp_C
    o2_ppm[0] = cfg.o2_probe_target_ppm
    quench_t[0] = cfg.quench_operating_temp_C

    for i in range(n):
        ti = t_arr[i]
        fl: List[str] = []
        current[i] = cfg.target_current_density_mA_cm2

        # Electrolyte overheats (cooling failure, heater stuck on)
        elec_temp[i] = _first_order_lag(
            elec_temp[i - 1] if i > 0 else elec_temp[0],
            hot_temp, cfg.electrolyte_temp_time_constant_min, dt,
        )

        elec_pH[i] = cfg.electrolyte_target_pH  # pH pumps still working
        fe_M[i] = fe_M[i - 1] if i > 0 else fe_M[0]
        furn_temp[i] = _first_order_lag(
            furn_temp[i - 1] if i > 0 else furn_temp[0],
            cfg.furnace_operating_temp_C, cfg.furnace_time_constant_min, dt,
        )
        o2_ppm[i] = _first_order_lag(
            o2_ppm[i - 1] if i > 0 else o2_ppm[0],
            cfg.o2_probe_target_ppm, cfg.o2_probe_time_constant_min, dt,
        )
        quench_t[i] = _first_order_lag(
            quench_t[i - 1] if i > 0 else quench_t[0],
            cfg.quench_operating_temp_C, cfg.quench_time_constant_min, dt,
        )

        # Quality degradation from temperature
        q_prev = 1.0 if i == 0 else quality[i - 1]
        q = q_prev
        if elec_temp[i] > cfg.max_electrolyte_temp_C:
            fl.append(f"high_electrolyte_temp_{elec_temp[i]:.0f}C")
            q -= 0.005 * (elec_temp[i] - cfg.max_electrolyte_temp_C) * dt
        if elec_temp[i] > WATER_BOILING_C - 5:
            fl.append("boiling_risk")
            q -= 0.02 * dt
        quality[i] = np.clip(q, 0.0, 1.0)

        base_ce = cfg.current_efficiency_normal
        temp_penalty = max(0.0, (elec_temp[i] - cfg.max_electrolyte_temp_C) / 20.0) * 0.05
        ce_arr[i] = np.clip(base_ce - temp_penalty, 0.0, 1.0)
        flags[i] = fl

    return TransientResult(
        time_min=t_arr, electrolyte_temp_C=elec_temp, electrolyte_pH=elec_pH,
        fe_concentration_M=fe_M, current_density_mA_cm2=current,
        furnace_temp_C=furn_temp, o2_probe_ppm=o2_ppm, quench_temp_C=quench_t,
        deposit_quality=quality, ce_fraction=ce_arr, flags=flags,
        scenario="upset_temperature_excursion",
    )


def _upset_gas_interruption(
    cfg: TransientConfig, dur: float, dt: float,
) -> TransientResult:
    """CO/CH4 gas supply lost → O2 returns to air → decarburization."""
    def current_profile(t: float) -> float:
        return cfg.target_current_density_mA_cm2

    def furnace_target(t: float) -> float:
        return cfg.furnace_operating_temp_C

    def quench_target(t: float) -> float:
        return cfg.quench_operating_temp_C

    return _integrate(
        config=cfg, duration_min=dur, dt_min=dt,
        current_profile_fn=current_profile,
        ph_pump_on=True,
        furnace_target_fn=furnace_target,
        gas_on=False,  # gas supply lost
        quench_target_fn=quench_target,
        scenario="upset_gas_supply_interruption",
        initial_state=_operating_state(cfg),
    )


def _upset_current_interruption(
    cfg: TransientConfig, dur: float, dt: float,
) -> TransientResult:
    """Cell current lost but cathode stays immersed → dissolution."""
    def current_profile(t: float) -> float:
        return 0.0

    def furnace_target(t: float) -> float:
        return cfg.furnace_operating_temp_C

    def quench_target(t: float) -> float:
        return cfg.quench_operating_temp_C

    return _integrate(
        config=cfg, duration_min=dur, dt_min=dt,
        current_profile_fn=current_profile,
        ph_pump_on=True,  # pumps still running
        furnace_target_fn=furnace_target,
        gas_on=True,  # furnace gas still on
        quench_target_fn=quench_target,
        scenario="upset_current_interruption",
        initial_state=_operating_state(cfg),
    )


def _upset_rectifier_fault(
    cfg: TransientConfig, dur: float, dt: float,
) -> TransientResult:
    """Rectifier fault: overcurrent then reverse polarity pulse."""
    overcurrent_mA = cfg.target_current_density_mA_cm2 * 2.0
    fault_start = 5.0  # minutes into the scenario
    reverse_duration = 2.0  # minutes of reverse polarity
    overcurrent_duration = 3.0  # minutes of overcurrent

    def current_profile(t: float) -> float:
        if t < fault_start:
            return cfg.target_current_density_mA_cm2
        t_fault = t - fault_start
        if t_fault < overcurrent_duration:
            return overcurrent_mA  # overcurrent
        elif t_fault < overcurrent_duration + reverse_duration:
            return -cfg.target_current_density_mA_cm2  # reverse polarity
        else:
            return 0.0  # tripped offline

    def furnace_target(t: float) -> float:
        return cfg.furnace_operating_temp_C

    def quench_target(t: float) -> float:
        return cfg.quench_operating_temp_C

    return _integrate(
        config=cfg, duration_min=dur, dt_min=dt,
        current_profile_fn=current_profile,
        ph_pump_on=True,
        furnace_target_fn=furnace_target,
        gas_on=True,
        quench_target_fn=quench_target,
        scenario="upset_rectifier_fault",
        initial_state=_operating_state(cfg),
    )


# ─── Analysis functions ────────────────────────────────────────────────

def recovery_time(
    result: TransientResult,
    threshold: Optional[float] = None,
    parameter: str = "deposit_quality",
) -> float:
    """Estimate hours to return to steady-state spec after an upset.

    Searches for the first time after the minimum quality point where
    all monitored parameters return within acceptable bounds.

    Parameters
    ----------
    result : TransientResult
        Output from a simulation.
    threshold : float, optional
        Quality threshold (default 0.9). If quality never recovers, returns inf.
    parameter : str
        Which parameter to check ('deposit_quality', 'electrolyte_temp_C',
        'electrolyte_pH', 'current_density_mA_cm2').

    Returns
    -------
    float
        Recovery time in hours. inf if never recovers.
    """
    thresh = threshold if threshold is not None else 0.9

    data = getattr(result, parameter, result.deposit_quality)
    # Find the minimum index (worst point)
    min_idx = np.argmin(data)
    if min_idx >= len(data) - 1:
        return float("inf")

    # Search after the minimum for recovery above threshold
    for i in range(min_idx, len(data)):
        if data[i] >= thresh:
            return float(result.time_min[i] - result.time_min[min_idx]) / 60.0

    return float("inf")


def damage_assessment(result: TransientResult) -> Dict[str, Any]:
    """Assess the damage from a transient event.

    Returns a dictionary describing what degraded, by how much, and
    whether recovery is feasible.

    Parameters
    ----------
    result : TransientResult
        Output from a simulation.

    Returns
    -------
    dict
        Damage summary with fields:
        - min_quality: lowest deposit quality reached
        - quality_loss: total quality decrement (1.0 - min)
        - max_ph: highest pH recorded
        - ph_excursion: whether Fe(OH)3 precipitation was likely
        - max_temp: highest electrolyte temperature
        - boiling_risk: whether temperature approached boiling
        - gas_outage_min: minutes of gas interruption (O2 > 1000 ppm)
        - decarb_risk: whether decarburization is likely
        - current_outage_min: minutes without cathodic current
        - flagged_steps: count of flagged time steps
        - recovery_hours: estimated recovery time
        - scrap: whether deposit is likely scrapped
    """
    n = len(result.time_min)
    if n == 0:
        return {"error": "empty result"}

    min_q = float(np.min(result.deposit_quality))
    max_ph = float(np.max(result.electrolyte_pH))
    max_temp = float(np.max(result.electrolyte_temp_C))
    min_fe = float(np.min(result.fe_concentration_M))

    # Gas outage: O2 > 1000 ppm indicates air infiltration
    gas_out_mask = result.o2_probe_ppm > 1000.0
    gas_out_min = float(np.sum(gas_out_mask)) * (result.time_min[1] - result.time_min[0]) if n > 1 else 0.0

    # Current outage
    current_out_mask = result.current_density_mA_cm2 <= 0
    current_out_min = float(np.sum(current_out_mask)) * (result.time_min[1] - result.time_min[0]) if n > 1 else 0.0

    flagged = int(sum(bool(f) for f in result.flags))

    rec = recovery_time(result, threshold=0.9, parameter="deposit_quality")

    return {
        "scenario": result.scenario,
        "min_deposit_quality": round(float(min_q), 4),
        "quality_loss": round(float(1.0 - min_q), 4),
        "max_electrolyte_pH": round(float(max_ph), 2),
        "ph_excursion": bool(max_ph > 3.8),
        "max_electrolyte_temp_C": round(float(max_temp), 1),
        "boiling_risk": bool(max_temp > 85.0),
        "min_fe_M": round(float(min_fe), 3),
        "gas_outage_min": round(float(gas_out_min), 1),
        "decarb_risk": bool(gas_out_min > 30.0),
        "current_outage_min": round(float(current_out_min), 1),
        "flagged_time_steps": flagged,
        "recovery_hours": float(rec) if rec < 1e6 else "never_recovers",
        "scrap": bool(min_q < 0.5),
    }
