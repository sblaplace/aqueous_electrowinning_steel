"""Scripted fault-injection replay for the composed system twin.

This is a deterministic *software* campaign, not physical validation.  It
wires the physics surrogate, :class:`DigitalTwin`, :class:`Crate`, and
:class:`OperatingTwin`, then checks that faults request a safe state and never
produce a post-trip actuation command.  The independent shutdown channel is
represented by the returned ``ShutdownRequest``; this module deliberately has
no executor for it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .crate import Crate
from .dark_mill import EXAMPLE_SITES, site_to_crate_config
from .digital_twin import DigitalTwin
from .operating_twin import OperatingTwin, SensorSnapshot, TwinConfig
from .twin_physics import CellProcessModel


@dataclass(frozen=True)
class Scenario:
    name: str
    expected_action: str
    wind: float = 20.0
    flood: float = 0.0
    rain: float = 20.0
    ingress: bool = False
    freeze: bool = False
    current: float = 1.0
    voltage: float = 2.5
    pH: float = 2.0
    quality: dict[str, str] | None = None
    now_s: float = 0.0
    sample_timestamp_s: float = 0.0


SCENARIOS = (
    Scenario("storm", "storm_mode_hold_high_wind", wind=65.0),
    Scenario("flood", "flood_hold_elevate_and_shutdown", flood=0.30),
    Scenario("heavy_rain_ingress", "storm_mode_hold_ingress", rain=180.0, ingress=True),
    Scenario("sensor_bias", "sensor_fault_hold", pH=9.0, quality={"pH": "biased"}),
    Scenario("sensor_stuck", "sensor_fault_hold", now_s=10.0, sample_timestamp_s=0.0),
    Scenario("sensor_spike", "sensor_fault_hold", current=250.0),
    Scenario("power_loss", "sensor_fault_hold", current=0.0, voltage=0.0,
             quality={"power": "failed"}),
    Scenario("freeze", "storm_mode_hold_freeze", freeze=True),
    Scenario("storm_plus_ingress", "flood_hold_elevate_and_shutdown", wind=65.0,
             rain=180.0, ingress=True, flood=0.30),
)


def _config() -> TwinConfig:
    return TwinConfig(
        cell_id="replay-cell", max_current_A=10.0,
        max_current_density_mA_cm2=200.0, max_voltage_V=5.0,
        min_temperature_C=0.0, max_temperature_C=80.0,
        min_fe2_M=0.2, max_fe2_M=2.0, min_pH=0.5, max_pH=5.0,
        target_current_A=5.0, max_wind_gust_m_s=40.0,
        max_flood_depth_m=0.1, max_rain_intensity_mm_hr=100.0,
        freeze_protection_required=True,
    )


def replay_scenario(scenario: Scenario, physics: CellProcessModel | None = None) -> dict[str, Any]:
    """Run one scenario and raise only through the returned ``passed`` field."""
    site = EXAMPLE_SITES["pickle_liquor_us_midwest"]
    base = site_to_crate_config(site)
    cfg = replace(
        base,
        wind=replace(base.wind, gust_m_s=scenario.wind),
        ground=replace(base.ground, flood_depth_m=scenario.flood),
        env=replace(base.env, rain_intensity_mm_hr=scenario.rain,
                    sealing_class="industrial"),
    )
    crate = Crate().evaluate(cfg)
    snap = SensorSnapshot(
        timestamp_s=scenario.sample_timestamp_s, current_A=scenario.current,
        voltage_V=scenario.voltage, temperature_C=25.0, pH=scenario.pH,
        fe2_M=1.0, cathode_area_cm2=100.0,
        sensor_quality=scenario.quality or {}, source_run_id=scenario.name,
        wind_gust_m_s=scenario.wind, flood_depth_m=scenario.flood,
        rain_intensity_mm_hr=scenario.rain, ingress_detected=scenario.ingress,
        freeze_detected=scenario.freeze,
    )
    op = OperatingTwin(_config())
    op.arm_actuation("replay-cell")
    state = op.update(snap, now_s=scenario.now_s)
    request = op.shutdown_request(snap, now_s=scenario.now_s)
    command = op.command(now_s=scenario.now_s)

    # Exercise the two process layers as well. Their output is evidence of
    # composition only; it does not upgrade credibility above L0.
    physics = physics or CellProcessModel()
    prediction = physics.predict(j_mA_cm2=100.0, temperature_C=60.0, fe2_M=1.0)
    digital = DigitalTwin(model=physics)
    digital.update({"temperature_C": 60.0, "pH": 2.0, "cell_voltage_V": 2.5,
                    "j_avg_mA_cm2": 100.0}, dt_hr=0.01)

    action = request.action if request else "normal_operation"
    safe = state.mode.value == "tripped" and request is not None
    mounting = crate.mounting_spec.lower()
    crate_safe = crate.stable or any(token in mounting for token in
                                     ("ballast", "anchor", "elevate", "sealing", "drainage"))
    passed = (safe and crate_safe and command.current_A == 0.0
              and command.mode.value == "tripped")
    if scenario.name == "sensor_stuck":
        passed = passed and "stale_snapshot" in state.trip_reasons
    return {
        "scenario": scenario.name, "passed": passed,
        "crate": {"stable": crate.stable, "ingress_risk": crate.ingress_risk,
                   "min_ballast_kg": round(crate.min_ballast_kg, 1),
                   "mounting_spec": crate.mounting_spec},
        "operating_twin": {"mode": state.mode.value, "trip_reasons": list(state.trip_reasons),
                            "command": command.to_dict()},
        "environmental_safe_state": action,
        "expected_safe_state": scenario.expected_action,
        "shutdown_request": request.to_dict() if request else None,
        "physics_prediction": {"fe_percent": prediction.fe_percent,
                               "v_cell_V": prediction.v_cell_V},
        "credibility": "process L0 / crate L0 / site L0",
    }


def run_replay() -> dict[str, Any]:
    physics = CellProcessModel()
    rows = [replay_scenario(s, physics) for s in SCENARIOS]
    return {"campaign": "composed-system-twin-fault-injection", "all_passed": all(r["passed"] for r in rows),
            "credibility_note": "Synthetic scripted replay; no real fault/test campaign. All layers remain L0.",
            "independent_shutdown": "Twin emits ShutdownRequest only; a diverse hardwired channel executes it.",
            "scenarios": rows}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Replay the composed twin fault matrix")
    parser.add_argument("--output", default="experiments/data/twin_replay_report.json")
    parser.add_argument("--fig", default="docs/figures/twin_replay_scenario_matrix.png")
    args = parser.parse_args()
    report = run_replay()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    _figure(report, Path(args.fig))
    for row in report["scenarios"]:
        print(f"{row['scenario']:22s} safe={row['environmental_safe_state']:38s} "
              f"shutdown_request={row['shutdown_request'] is not None} pass={row['passed']}")
    print(f"Replay {'PASS' if report['all_passed'] else 'FAIL'}; report: {args.output}")


def _figure(report: dict[str, Any], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = report["scenarios"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(len(rows)), [int(r["passed"]) for r in rows], color="#167c80")
    ax.set_xticks(range(len(rows)), [r["scenario"].replace("_", "\n") for r in rows], rotation=0)
    ax.set_ylim(0, 1.25); ax.set_ylabel("safe-state invariant passed (1=yes)")
    ax.set_title("Composed system twin — scripted environmental + sensor-fault replay")
    ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
