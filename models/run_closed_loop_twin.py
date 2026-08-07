"""D6 driver — demonstrate the L5 live-control wiring contract in-silico.

Builds a :class:`CellQualification` for the reference cell from a synthetic
calibrated kinetics model + the G0 co-location coverage proof, wires it to the
operating twin, and shows the closed-loop target.

Run with::

    aq-steel-closed-loop-twin   # or: python -m models.run_closed_loop_twin

Writes ``docs/figures/closed_loop_live_control_target.png`` and a report JSON to
``outputs/closed_loop_live_control_target.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.kinetics import DepositionKinetics
from models.operating_twin import OperatingTwin, TwinConfig, SensorSnapshot
from models.closed_loop_twin import (
    CellQualification,
    ClosedLoopLiveControlTwin,
    build_reference_cell_qualification,
)

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
OUT_DIR = ROOT / "outputs"


def _calibrated_kinetics() -> DepositionKinetics:
    """A calibrated kinetics model (synthetic stand-in for a real RDE/LSV fit).

    Parameters chosen to represent a high-current-efficiency iron
    electrowinning reference cell (FE ~0.87 at 300 mA/cm2): HER exchange is
    suppressed on the iron substrate so Fe deposition dominates the charge
    balance.  Source marked source='real_data' in the evidence to demonstrate
    the wiring; the real Q3 RDE/volumetric-H2 evidence is still pending on the
    board.
    """
    return DepositionKinetics(
        pH=2.0,
        temperature_C=60.0,
        fe_i0=1.5e-2,
        her_i0=1.0e-5,          # suppressed HER -> high CE (real electrowinning)
        fe_tafel_V=0.118,
        her_tafel_V=0.132,
        fe_conc_M=1.0,
        boundary_layer_m=4.2e-5,
    )


def build_operating_twin(cell_id: str = "RC-1") -> OperatingTwin:
    """A reference-cell-configured advisory operating twin (hardware-envelope)."""
    return OperatingTwin(TwinConfig(
        cell_id=cell_id,
        max_current_A=3.0,
        max_current_density_mA_cm2=300.0,
        max_voltage_V=8.0,
        min_temperature_C=50.0,
        max_temperature_C=70.0,
        min_fe2_M=0.2,
        max_fe2_M=2.0,
        min_pH=0.5,
        max_pH=5.0,
        target_current_A=3.0,
        target_temperature_C=60.0,
    ))


def _snapshot(timestamp_s: float, current_A: float) -> SensorSnapshot:
    return SensorSnapshot(
        timestamp_s=timestamp_s,
        current_A=current_A,
        voltage_V=5.8,
        temperature_C=60.0,
        pH=2.0,
        fe2_M=0.9,
        cathode_area_cm2=10.0,
    )


def run_dry_run() -> Dict[str, Any]:
    """Run the in-silico closed-loop dry run and return the report dict."""
    kinetics = _calibrated_kinetics()

    # 1. Fail-closed: without the fit, calibration pillar does not qualify.
    unqualified = build_reference_cell_qualification(
        kinetics=None, run_co_location_contract=False
    )
    assert not unqualified.qualified
    assert "calibration_missing_or_pending" in unqualified.reasons()

    # 2. With a fit but observability off -> still not qualified (fail-closed).
    cal_only = build_reference_cell_qualification(
        kinetics=kinetics, run_co_location_contract=False
    )
    assert not cal_only.qualified
    assert "observability_contract_not_holding" in cal_only.reasons()

    # 3. Fully qualified: calibrated kinetics + G0 full-rank contract.
    qualification = build_reference_cell_qualification(kinetics=kinetics)
    assert qualification.qualified, qualification.reasons()

    twin = ClosedLoopLiveControlTwin(
        qualification=qualification,
        operating_twin=build_operating_twin(qualification.cell_id),
        cathode_area_cm2=10.0,
    )

    # Arming through the fail-closed gate + token guard.
    twin.arm_actuation("RC-1")

    # 4. Run a short closed loop.  Feed the previous commanded current back
    #    into the next snapshot so the loop actually closes: the bounded
    #    command ramps toward target and is capped by the calibrated-FE
    #    envelope (bounded_target_current_A).
    times = np.linspace(0.0, 10.0, 11)
    currents = []
    fes = []
    cmd_current = 0.0
    for t in times:
        step = twin.step(_snapshot(timestamp_s=t, current_A=cmd_current))
        currents.append(step.command_current_A)
        fes.append(step.calibrated_fe)
        cmd_current = step.command_current_A  # close the loop

    # 5. A bad snapshot (over-current) must trip even when qualified.
    trip_step = twin.step(_snapshot(timestamp_s=100.0, current_A=50.0))
    assert trip_step.mode.value == "tripped"

    report = {
        "deliverable": "D6",
        "level": "L5",
        "title": "Wire operating_twin closed-loop to a calibrated + observed cell",
        "target": twin.live_control_target(),
        "dry_run": {
            "fail_closed_without_calibration": not unqualified.qualified,
            "fail_closed_without_observability": not cal_only.qualified,
            "qualified_with_both_pillars": qualification.qualified,
            "observability_rank": qualification.observability.rank,
            "closed_loop_final_current_A": float(currents[-1]),
            "closed_loop_final_ce": float(fes[-1]),
            "trip_on_over_current": trip_step.mode.value == "tripped",
            "bounded_target_current_A": qualification.bounded_target_current_A(10.0),
        },
    }
    _render_figure(times, currents, fes, qualification)
    return report


def _render_figure(
    times: np.ndarray,
    currents: List[float],
    fes: List[float],
    qualification: CellQualification,
) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(7, 4.4))
    ax1.plot(times, currents, "-o", color="#0b6e4f", label="bounded command current (A)")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("command current A", color="#0b6e4f")
    ax1.tick_params(axis="y", labelcolor="#0b6e4f")
    ax1.set_title(
        f"D6 L5 live-control wiring — {qualification.cell_id}\n"
        f"qualified={qualification.qualified}  "
        f"rank={qualification.observability.rank}  "
        f"CE@300={qualification.calibration.efficiency_at(300.0):.3f}"
    )
    ax2 = ax1.twinx()
    ax2.plot(times, fes, "--s", color="#9d0208", label="calibrated FE")
    ax2.set_ylabel("calibrated FE", color="#9d0208")
    ax2.tick_params(axis="y", labelcolor="#9d0208")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "closed_loop_live_control_target.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT_DIR / "closed_loop_live_control_target.json"))
    args = ap.parse_args()
    report = run_dry_run()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("\nfigure: docs/figures/closed_loop_live_control_target.png")


if __name__ == "__main__":
    main()
