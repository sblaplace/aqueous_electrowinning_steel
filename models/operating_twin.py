"""Safety-first supervisory operating twin for a real electrowinning cell.

This module is deliberately narrower than a plant simulator.  It provides the
Level-5 *software boundary*: versioned state, sensor freshness/quality checks,
charge and iron accounting, hard trips, and bounded advisory/actuation
commands.  It does not claim that an uncalibrated chemistry model can control
hardware.  ``actuation_enabled`` must be explicitly armed after the named
cell, sensor set, calibration and operating envelope have been qualified.

The intended loop is::

    raw sensors -> SensorSnapshot -> OperatingTwin.update()
                 -> safety evaluation -> ControlCommand

A real driver should persist every snapshot, state, decision and command with
a run id before sending a command to a rectifier or pump controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from math import isfinite
from typing import Iterable, Mapping

from .electrochemistry import FARADAY, M_FE, Z_FE


class TwinMode(str, Enum):
    ADVISORY = "advisory"
    ACTUATION = "actuation"
    TRIPPED = "tripped"


@dataclass(frozen=True)
class TwinConfig:
    """Hard limits for one named, qualified cell configuration.

    These are deliberately required at construction time rather than silently
    borrowed from a screening model.  Values must come from the hardware
    qualification record before actuation is enabled.

    Extended with whole-system-twin environmental limits: high-wind / flood /
    ingress bound permissible operation (storm mode) as a safe-state limit.
    If any environmental max is set, exceeding it triggers a latched safe-state
    (TRIPPED or storm-hold) in the same way as a temperature trip.
    """

    cell_id: str
    max_current_A: float
    max_current_density_mA_cm2: float
    max_voltage_V: float
    min_temperature_C: float
    max_temperature_C: float
    min_fe2_M: float
    max_fe2_M: float
    max_pH: float
    min_pH: float
    max_stale_s: float = 2.0
    max_sensor_age_s: float = 5.0
    target_current_A: float = 0.0
    target_temperature_C: float = 50.0
    current_ramp_A_per_s: float = 0.1
    control_interval_s: float = 1.0

    # ── Whole-system twin: environmental safe-state limits (storm mode) ──
    max_wind_gust_m_s: float | None = None        # e.g. 40 m/s; None = no wind trip
    max_flood_depth_m: float | None = None        # e.g. 0.1 m
    max_rain_intensity_mm_hr: float | None = None # e.g. 100 mm/hr
    max_snow_load_kPa: float | None = None
    freeze_protection_required: bool = False      # if True, freeze triggers advisory

    def __post_init__(self) -> None:
        if not self.cell_id.strip():
            raise ValueError("cell_id is required")
        positive = (self.max_current_A, self.max_current_density_mA_cm2,
                    self.max_voltage_V, self.max_sensor_age_s,
                    self.current_ramp_A_per_s, self.control_interval_s)
        if any(x <= 0 or not isfinite(x) for x in positive):
            raise ValueError("hard limits and timing/ramp values must be finite and positive")
        if self.min_temperature_C >= self.max_temperature_C:
            raise ValueError("temperature limits must be ordered")
        if self.min_fe2_M < 0 or self.min_fe2_M >= self.max_fe2_M:
            raise ValueError("Fe2+ limits must be non-negative and ordered")
        if self.min_pH >= self.max_pH:
            raise ValueError("pH limits must be ordered")
        if not 0 <= self.target_current_A <= self.max_current_A:
            raise ValueError("target current must lie inside the current limit")
        if not self.min_temperature_C <= self.target_temperature_C <= self.max_temperature_C:
            raise ValueError("target temperature must lie inside temperature limits")
        # Environmental limits if set must be finite and non-negative
        env_vals = (self.max_wind_gust_m_s, self.max_flood_depth_m,
                    self.max_rain_intensity_mm_hr, self.max_snow_load_kPa)
        for v in env_vals:
            if v is not None and (not isfinite(v) or v < 0):
                raise ValueError("environmental limits must be finite and non-negative if set")


@dataclass(frozen=True)
class SensorSnapshot:
    """One synchronized, engineering-unit sensor snapshot.

    Extended with optional whole-system environmental fields for storm-mode
    safe-state evaluation. These are optional so existing unit tests and
    purely-cellular snapshots remain valid.
    """

    timestamp_s: float
    current_A: float
    voltage_V: float
    temperature_C: float
    pH: float
    fe2_M: float
    cathode_area_cm2: float
    sensor_quality: Mapping[str, str] = field(default_factory=dict)
    source_run_id: str = ""

    # ── Whole-system twin: environmental observation (optional) ──
    wind_gust_m_s: float | None = None
    flood_depth_m: float | None = None
    rain_intensity_mm_hr: float | None = None
    snow_load_kPa: float | None = None
    ingress_detected: bool = False
    freeze_detected: bool = False

    def __post_init__(self) -> None:
        values = (self.timestamp_s, self.current_A, self.voltage_V,
                  self.temperature_C, self.pH, self.fe2_M, self.cathode_area_cm2)
        if any(not isfinite(x) for x in values):
            raise ValueError("sensor values must be finite")
        if self.cathode_area_cm2 <= 0:
            raise ValueError("cathode_area_cm2 must be positive")
        if self.timestamp_s < 0:
            raise ValueError("timestamp_s cannot be negative")
        # Environmental values if provided must be finite and non-negative
        for name in ("wind_gust_m_s", "flood_depth_m", "rain_intensity_mm_hr", "snow_load_kPa"):
            v = getattr(self, name)
            if v is not None and (not isfinite(v) or v < 0):
                raise ValueError(f"{name} must be finite and non-negative if set")

    @property
    def current_density_mA_cm2(self) -> float:
        return self.current_A / self.cathode_area_cm2 * 1000.0


@dataclass(frozen=True)
class ControlCommand:
    """A bounded command and its safety decision."""

    timestamp_s: float
    mode: TwinMode
    current_A: float
    temperature_setpoint_C: float
    reasons: tuple[str, ...] = ()
    source_run_id: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp_s": self.timestamp_s,
            "mode": self.mode.value,
            "current_A": self.current_A,
            "temperature_setpoint_C": self.temperature_setpoint_C,
            "reasons": list(self.reasons),
            "source_run_id": self.source_run_id,
        }


@dataclass(frozen=True)
class TwinState:
    """Auditable state after one update."""

    mode: TwinMode
    last_timestamp_s: float | None = None
    last_snapshot: SensorSnapshot | None = None
    charge_cathodic_C: float = 0.0
    theoretical_fe_mass_g: float = 0.0
    trip_reasons: tuple[str, ...] = ()
    update_count: int = 0

    def to_dict(self) -> dict:
        snap = self.last_snapshot
        return {
            "mode": self.mode.value,
            "last_timestamp_s": self.last_timestamp_s,
            "charge_cathodic_C": self.charge_cathodic_C,
            "theoretical_fe_mass_g": self.theoretical_fe_mass_g,
            "trip_reasons": list(self.trip_reasons),
            "update_count": self.update_count,
            "last_current_A": None if snap is None else snap.current_A,
            "last_voltage_V": None if snap is None else snap.voltage_V,
            "last_temperature_C": None if snap is None else snap.temperature_C,
            "last_pH": None if snap is None else snap.pH,
            "last_fe2_M": None if snap is None else snap.fe2_M,
        }


class OperatingTwin:
    """Stateful supervisory twin with fail-safe control boundaries.

    The default mode is advisory.  ``arm_actuation`` requires an explicit
    qualification token equal to the configured cell id, making accidental
    actuation from a generic simulation difficult.  Any hard-limit violation,
    stale snapshot, bad sensor quality, timestamp regression or excessive
    voltage causes a latched trip; clearing it requires a separate operator
    action and a fresh safe snapshot.
    """

    def __init__(self, config: TwinConfig) -> None:
        self.config = config
        self.state = TwinState(mode=TwinMode.ADVISORY)
        self._armed = False

    @property
    def mode(self) -> TwinMode:
        return self.state.mode

    def arm_actuation(self, qualification_token: str) -> None:
        """Enable commands only for the exact qualified cell identity."""
        if qualification_token != self.config.cell_id:
            raise PermissionError("qualification token does not match configured cell")
        if self.state.mode == TwinMode.TRIPPED:
            raise RuntimeError("clear the latched trip before arming actuation")
        self._armed = True
        self.state = replace(self.state, mode=TwinMode.ACTUATION)

    def disarm(self) -> None:
        self._armed = False
        if self.state.mode != TwinMode.TRIPPED:
            self.state = replace(self.state, mode=TwinMode.ADVISORY)

    def clear_trip(self, operator: str, snapshot: SensorSnapshot) -> None:
        """Clear a trip only with an identified operator and safe snapshot."""
        if not operator.strip():
            raise ValueError("operator identity is required to clear a trip")
        reasons = self._safety_reasons(snapshot, now_s=snapshot.timestamp_s)
        if reasons:
            raise RuntimeError(f"cannot clear trip while unsafe: {', '.join(reasons)}")
        self._armed = False
        self.state = TwinState(mode=TwinMode.ADVISORY,
                               last_timestamp_s=snapshot.timestamp_s,
                               last_snapshot=snapshot,
                               update_count=self.state.update_count)

    def _safety_reasons(self, snapshot: SensorSnapshot, now_s: float) -> list[str]:
        c = self.config
        reasons: list[str] = []
        if snapshot.current_A < -1e-9 or snapshot.current_A > c.max_current_A:
            reasons.append("current_limit")
        if snapshot.current_density_mA_cm2 > c.max_current_density_mA_cm2:
            reasons.append("current_density_limit")
        if snapshot.voltage_V < 0 or snapshot.voltage_V > c.max_voltage_V:
            reasons.append("voltage_limit")
        if not c.min_temperature_C <= snapshot.temperature_C <= c.max_temperature_C:
            reasons.append("temperature_limit")
        if not c.min_fe2_M <= snapshot.fe2_M <= c.max_fe2_M:
            reasons.append("fe2_limit")
        if not c.min_pH <= snapshot.pH <= c.max_pH:
            reasons.append("ph_limit")
        if now_s - snapshot.timestamp_s > c.max_sensor_age_s:
            reasons.append("stale_snapshot")
        bad = [name for name, quality in snapshot.sensor_quality.items()
               if quality.lower() not in {"ok", "good", "valid"}]
        if bad:
            reasons.append("bad_sensor_quality:" + ",".join(sorted(bad)))

        # ── Whole-system twin: environmental safe-state limits (storm mode) ──
        if c.max_wind_gust_m_s is not None and snapshot.wind_gust_m_s is not None:
            if snapshot.wind_gust_m_s > c.max_wind_gust_m_s:
                reasons.append("high_wind")
        if c.max_flood_depth_m is not None and snapshot.flood_depth_m is not None:
            if snapshot.flood_depth_m > c.max_flood_depth_m:
                reasons.append("flood")
        if c.max_rain_intensity_mm_hr is not None and snapshot.rain_intensity_mm_hr is not None:
            if snapshot.rain_intensity_mm_hr > c.max_rain_intensity_mm_hr:
                reasons.append("heavy_rain")
        if c.max_snow_load_kPa is not None and snapshot.snow_load_kPa is not None:
            if snapshot.snow_load_kPa > c.max_snow_load_kPa:
                reasons.append("snow_overload")
        if snapshot.ingress_detected:
            reasons.append("ingress")
        if snapshot.freeze_detected and c.freeze_protection_required:
            reasons.append("freeze")

        return reasons

    def environmental_safe_state(self, snapshot: SensorSnapshot) -> str:
        """Return a human-readable safe-state action for the environment.

        This maps the safety reasons into the system-twin's environmental
        action vocabulary: normal_operation vs storm_mode_hold*.

        The method does NOT latch a trip; it is a pure assessment helper for
        the system twin driver and for operator displays.
        """
        reasons = self._safety_reasons(snapshot, now_s=snapshot.timestamp_s)
        # Filter to environmental reasons only
        env_reasons = [r for r in reasons if r in {
            "high_wind", "flood", "heavy_rain", "snow_overload", "ingress", "freeze"
        }]
        if not env_reasons:
            return "normal_operation"
        # Priority: flood > wind > ingress > rain/snow/freeze
        if "flood" in env_reasons:
            return "flood_hold_elevate_and_shutdown"
        if "high_wind" in env_reasons:
            return "storm_mode_hold_high_wind"
        if "ingress" in env_reasons:
            return "storm_mode_hold_ingress"
        if "heavy_rain" in env_reasons:
            return "storm_mode_hold_heavy_rain"
        if "snow_overload" in env_reasons:
            return "storm_mode_hold_snow"
        if "freeze" in env_reasons:
            return "storm_mode_hold_freeze"
        return "storm_mode_hold_" + "_".join(env_reasons)

    def update(self, snapshot: SensorSnapshot, now_s: float | None = None) -> TwinState:
        """Ingest a snapshot, update charge/iron ledgers and evaluate safety."""
        now = snapshot.timestamp_s if now_s is None else now_s
        previous = self.state.last_snapshot
        if previous is not None and snapshot.timestamp_s < previous.timestamp_s:
            reasons = ("timestamp_regression",)
            self.state = replace(self.state, mode=TwinMode.TRIPPED,
                                 trip_reasons=reasons, update_count=self.state.update_count + 1)
            self._armed = False
            return self.state

        dt = 0.0 if previous is None else snapshot.timestamp_s - previous.timestamp_s
        # Positive current is the operating-twin convention for cathodic load.
        # Negative current is rejected as an invalid actuation/sensor state.
        charge = self.state.charge_cathodic_C
        if previous is not None:
            charge += max(0.0, (previous.current_A + snapshot.current_A) * 0.5 * dt)
        mass_g = charge * M_FE * 1000.0 / (Z_FE * FARADAY)
        reasons = tuple(self._safety_reasons(snapshot, now))
        mode = TwinMode.TRIPPED if reasons else self.state.mode
        if reasons:
            self._armed = False
        self.state = TwinState(mode=mode, last_timestamp_s=snapshot.timestamp_s,
                               last_snapshot=snapshot, charge_cathodic_C=charge,
                               theoretical_fe_mass_g=mass_g,
                               trip_reasons=reasons if reasons else self.state.trip_reasons,
                               update_count=self.state.update_count + 1)
        return self.state

    def command(self, now_s: float | None = None) -> ControlCommand:
        """Return a bounded command; never returns a command after a trip."""
        if self.state.last_snapshot is None:
            return ControlCommand(0.0, TwinMode.ADVISORY, 0.0,
                                  self.config.target_temperature_C,
                                  ("no_snapshot",))
        snap = self.state.last_snapshot
        now = snap.timestamp_s if now_s is None else now_s
        reasons = list(self._safety_reasons(snap, now))
        if self.state.mode == TwinMode.TRIPPED or reasons:
            return ControlCommand(snap.timestamp_s, TwinMode.TRIPPED, 0.0,
                                  self.config.target_temperature_C,
                                  tuple(self.state.trip_reasons or reasons), snap.source_run_id)
        if not self._armed:
            return ControlCommand(snap.timestamp_s, TwinMode.ADVISORY, 0.0,
                                  self.config.target_temperature_C,
                                  ("actuation_not_armed",), snap.source_run_id)
        # Ramp-limit both increases and decreases.  A real implementation must
        # also apply the rectifier's independently configured hardware limits.
        delta = self.config.current_ramp_A_per_s * self.config.control_interval_s
        target = min(self.config.target_current_A, self.config.max_current_A)
        requested = max(0.0, min(self.config.max_current_A, snap.current_A +
                                 max(-delta, min(delta, target - snap.current_A))))
        return ControlCommand(snap.timestamp_s, TwinMode.ACTUATION, requested,
                              self.config.target_temperature_C, (), snap.source_run_id)


def summarize_snapshots(snapshots: Iterable[SensorSnapshot]) -> dict:
    """Small balance summary for a persisted run or replay test."""
    rows = list(snapshots)
    if not rows:
        raise ValueError("at least one snapshot is required")
    twin = OperatingTwin(TwinConfig(
        cell_id="replay", max_current_A=1e9, max_current_density_mA_cm2=1e9,
        max_voltage_V=1e9, min_temperature_C=-273.15, max_temperature_C=1000,
        min_fe2_M=0, max_fe2_M=1e9, min_pH=-1e9, max_pH=1e9,
        max_sensor_age_s=1e9))
    for row in rows:
        twin.update(row)
    return twin.state.to_dict()
