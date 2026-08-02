"""Co-live deposit-thickness metrology: optical + ultrasonic cross-validation pair.

This module turns the design note ``docs/DEPOSIT_METROLOGY.md`` into an
operable dry-run demonstration of a permanently co-live dual thickness sensor:

- **opt-101** — optical laser-line profilometer (surface height vs a reference
  plane; requires line-of-sight into the divided gassed cell).
- **thk-101** — pulse-echo ultrasonic thickness (through-thickness echo time of
  flight; blind to line-of-sight but weak on the iron-on-iron interface).

The two observe the *same* state (deposit thickness) through unrelated physics
with unrelated, disjoint error models (docs/DEPOSIT_METROLOGY.md sm3).  Their
value is *not* observability (one direct observation already restores rank 7) —
it is error characterization, validation and fault tolerance: agreement ⇒ real
confidence, divergence ⇒ a systematic error on one channel that is now
detectable and can feed the operating twin's safe-state/trip machinery.

Design contract (mirrors the repo's OFF-by-default convention):
- module is purely additive and self-contained; it does not modify
  ``digital_twin.py``/``operating_twin.py`` or the base/L1 sensor maps.
- The optic is a *validator* channel: it never enters the EKF fusion.  It is
  consumed only by the agreement logic, which marks per-sensor ``sensor_quality``
  on the :class:`~models.operating_twin.SensorSnapshot`.  A non-OK quality tag
  trips the existing operating twin (``bad_sensor_quality`` -> sensor_fault_hold),
  so the safe-state behaviour is provided by the already-wired machinery.
- The fused estimate (inverse-variance weighted mean of the two channels) is
  what the twin should consume as a thickness reading upstream of the EKF.

Everything is L0 screening scaffold: the error models are *assumptions* to be
validated on the reference cell, not calibrated hardware.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .digital_twin import DigitalTwin, generate_synthetic_readings
from .operating_twin import OperatingTwin, SensorSnapshot, TwinConfig

# ---------------------------------------------------------------------------
# Per-channel disjoint error models (L0 assumptions, to be characterized on the
# reference cell).  The whole point of the pair is that these envelopes are
# *independent*: an error that inflates one channel does not inflate the other.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelErrorModel:
    """One channel's total standard uncertainty budget (1-sigma, um)."""

    tag: str
    noise_std_um: float  # in-loop random measurement noise
    bias_budget_um: float  # slowly-varying systematic drift envelope

    @property
    def total_unc_um(self) -> float:
        return math.hypot(self.noise_std_um, self.bias_budget_um)


# Optical laser-line: small noise on a smooth surface, but surface-sensitive
# (roughness + refraction + bubble scatter) and loss-of-line-of-sight prone.
OPTICAL_ERROR = ChannelErrorModel("opt-101", noise_std_um=1.5, bias_budget_um=4.0)

# Ultrasonic pulse-echo: aggregates over the transducer footprint, and the
# iron-on-iron deposit/substrate echo is weak and muddied by gas bubbles.
ULTRASONIC_ERROR = ChannelErrorModel("thk-101", noise_std_um=4.0, bias_budget_um=3.0)

# Discrepancy gates, in units of the combined standard uncertainty.
#   |d| = |opt - us|,  u_c = hypot(OPTICAL.total_unc, ULTRASONIC.total_unc)
DEGRADE_Z = 3.0  # > 3 sigma: one channel is suspect (hold / advisory)
FAULT_Z = 6.0  # > 6 sigma: channels discordant (trip)

PASS_TAGS = {"ok", "good", "valid"}


@dataclass(frozen=True)
class PairVerdict:
    """Agreement assessment between the two co-live channels."""

    agreed: bool
    fused_thickness_um: float  # inverse-variance weighted mean of both channels
    sensor_quality: Dict[str, str]  # tag -> ok | degraded | faulted
    z_score: float  # |d| / u_c
    level: str  # "agree" | "degraded" | "faulted"
    suspect: str  # "none" | "opt-101" | "thk-101" | "both"


def _fused(optical_um: float, ultrasonic_um: float) -> float:
    w_o = 1.0 / OPTICAL_ERROR.noise_std_um ** 2
    w_u = 1.0 / ULTRASONIC_ERROR.noise_std_um ** 2
    return (w_o * optical_um + w_u * ultrasonic_um) / (w_o + w_u)


def assess_pair(optical_um: float, ultrasonic_um: float) -> PairVerdict:
    """Return the agreement/discrepancy verdict for one synchronized pair.

    The two channels are treated as two independent noise sources on the same
    state (docs/DEPOSIT_METROLOGY.md sm5).  The discrepancy ``|opt - us|`` is
    compared against their combined standard uncertainty ``u_c``.

    Attribution: with only two co-live channels, a discordant pair cannot say
    *which* channel is wrong (a biased channel pulls an inverse-variance fused
    mean toward itself, so the innocent one then looks off too).  The safe and
    honest behaviour is therefore to flag **both** as suspect on disagreement
    and let the operating twin hold.  Definitive attribution needs a third,
    independent reference (weighed-mass / periodic coupon, or a coulometric
    channel) — noted in docs/DEPOSIT_METROLOGY.md (sm Agreement logic).
    """
    delta = optical_um - ultrasonic_um
    u_c = math.hypot(OPTICAL_ERROR.total_unc_um, ULTRASONIC_ERROR.total_unc_um)
    z = abs(delta) / u_c
    fused = _fused(optical_um, ultrasonic_um)

    if z <= DEGRADE_Z:
        return PairVerdict(True, fused, {"opt-101": "ok", "thk-101": "ok"},
                           z, "agree", "none")

    level = "faulted" if z > FAULT_Z else "degraded"
    return PairVerdict(False, fused, {"opt-101": level, "thk-101": level},
                       z, level, "both")


# ---------------------------------------------------------------------------
# Simulated channels around a shared deposit trajectory
# ---------------------------------------------------------------------------


def optical_reading(truth_um: float, t_hr: float, rng: np.random.Generator,
                    fault: Optional[Dict[str, Any]] = None) -> float:
    """One optical-channel read: truth + ramp-on fault bias + channel noise."""
    bias = _fault_bias(t_hr, fault)
    return max(0.0, truth_um + bias
               + rng.normal(0.0, OPTICAL_ERROR.noise_std_um))


def ultrasonic_reading(truth_um: float, t_hr: float, rng: np.random.Generator,
                       fault: Optional[Dict[str, Any]] = None) -> float:
    """One ultrasonic-channel read: truth + ramp-on fault bias + channel noise."""
    bias = _fault_bias(t_hr, fault)
    return max(0.0, truth_um + bias
               + rng.normal(0.0, ULTRASONIC_ERROR.noise_std_um))


def _fault_bias(t_hr: float, fault: Optional[Dict[str, Any]]) -> float:
    if not fault:
        return 0.0
    mag = float(fault.get("magnitude_um", 0.0))
    ramp = float(fault.get("ramp_hr", 0.5))
    frac = min(1.0, max(0.0, t_hr / ramp)) if ramp > 0 else 1.0
    return mag * frac


def deposit_trajectory(t_hr: float, rate_um_hr: float = 120.0) -> float:
    """Reference deposit thickness (um) growing at ``rate_um_hr`` from zero."""
    return rate_um_hr * t_hr


# ---------------------------------------------------------------------------
# Dry-run demonstration driver (simulated THK + optical agree / disagree)
# ---------------------------------------------------------------------------

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "agree": {},
    "optical_foul": {"optical_fault": {"magnitude_um": 60.0, "ramp_hr": 0.5}},
    "ultrasound_interface": {"ultrasonic_fault": {"magnitude_um": -50.0,
                                                  "ramp_hr": 0.5}},
}

# Expected safe-state outcome per scenario (mapped in run_deposit_metrology_demo).
EXPECTED_TRIP = {
    "agree": False,
    "optical_foul": True,
    "ultrasound_interface": True,
}
EXPECTED_SUSPECT = {
    "agree": "none",
    "optical_foul": "both",
    "ultrasound_interface": "both",
}


def _operating_config() -> TwinConfig:
    return TwinConfig(
        cell_id="deposit-metro", max_current_A=10.0,
        max_current_density_mA_cm2=200.0, max_voltage_V=5.0,
        min_temperature_C=0.0, max_temperature_C=80.0,
        min_fe2_M=0.2, max_fe2_M=2.0, min_pH=0.5, max_pH=5.0,
        target_current_A=3.0,
    )


def run_deposit_metrology_demo(
    rate_um_hr: float = 120.0,
    n_steps: int = 20,
    dt_hr: float = 0.1,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run the agree/disagree dry run through the composed operating twin.

    For each scenario we:
      1. grow a reference deposit trajectory;
      2. simulate the two co-live channels around it (one possibly faulted);
      3. assess agreement and map the verdict onto per-sensor ``sensor_quality``;
      4. push a :class:`SensorSnapshot` with that quality into the operating
         twin and let its existing safe-state machinery decide a trip;
      5. also feed THK-101 to the EKF :class:`DigitalTwin` to confirm the
         deposit estimate stays bounded when the pair agrees.

    Returns a JSON-serializable report; single-source-of-truth for the dry run.
    """
    rng = np.random.default_rng(seed)
    steps_hr = [i * dt_hr for i in range(n_steps)]
    rows = []

    for name, faults in SCENARIOS.items():
        op = OperatingTwin(_operating_config())
        op.arm_actuation("deposit-metro")
        twin = DigitalTwin(seed=seed)
        timeline = []
        last_verdict = None
        last_snap: Optional[SensorSnapshot] = None

        for t_hr in steps_hr:
            truth = deposit_trajectory(t_hr, rate_um_hr)
            opt = optical_reading(truth, t_hr, rng, faults.get("optical_fault"))
            us = ultrasonic_reading(truth, t_hr, rng, faults.get("ultrasonic_fault"))
            verdict = assess_pair(opt, us)
            last_verdict = verdict

            snap = SensorSnapshot(
                timestamp_s=t_hr * 3600.0, current_A=3.0, voltage_V=2.5,
                temperature_C=60.0, pH=2.0, fe2_M=1.0, cathode_area_cm2=100.0,
                sensor_quality=verdict.sensor_quality, source_run_id=name,
            )
            op.update(snap, now_s=t_hr * 3600.0)

            # Feed the ultrasonic channel as the EKF thickness observation so the
            # Fused estimate stays bounded when the pair is healthy.
            readings = generate_synthetic_readings(
                twin.design_point, t_hr, rng, include_l1_sensors=True)
            readings["THK-101"] = us  # override with the metrology channel
            twin.update(readings, dt_hr=dt_hr)
            last_snap = snap

            timeline.append({
                "t_hr": round(t_hr, 3), "truth_um": round(truth, 2),
                "opt_um": round(opt, 2), "us_um": round(us, 2),
                "fused_um": round(verdict.fused_thickness_um, 2),
                "z": round(verdict.z_score, 2), "level": verdict.level,
            })

        state = op.state
        assert last_snap is not None
        request = op.shutdown_request(last_snap, now_s=steps_hr[-1] * 3600.0)
        command = op.command(now_s=steps_hr[-1] * 3600.0)
        assert last_verdict is not None
        trip = state.mode.value == "tripped"
        reasons = list(state.trip_reasons)

        passed = trip == EXPECTED_TRIP[name]
        if EXPECTED_TRIP[name]:
            # Disagree: the discordant pair must be flagged and the twin tripped
            # to a zero-current command (both channels suspect — see assess_pair).
            passed = passed and last_verdict.suspect == EXPECTED_SUSPECT[name]
            passed = passed and command.current_A == 0.0
        else:
            # Agree: no trip, actuation allowed, and the EKF deposit estimate
            # stays bounded (no divergent open-loop integration).
            dep2s = twin.history[-1].uncertainty_2sigma["deposit_thickness"]
            passed = passed and dep2s < 3.0

        rows.append({
            "scenario": name, "passed": bool(passed),
            "expected_trip": EXPECTED_TRIP[name],
            "final_verdict": {
                "level": last_verdict.level, "z": round(last_verdict.z_score, 2),
                "suspect": last_verdict.suspect,
                "sensor_quality": last_verdict.sensor_quality,
            },
            "operating_twin": {
                "mode": state.mode.value, "trip_reasons": reasons,
                "command_current_A": command.current_A,
            },
            "shutdown_action": request.action if request else "normal_operation",
            "timeline": timeline,
        })

    return {
        "campaign": "deposit-metrology-dual-sensor-dry-run",
        "all_passed": all(r["passed"] for r in rows),
        "error_models": {
            "opt-101": {"noise_std_um": OPTICAL_ERROR.noise_std_um,
                        "bias_budget_um": OPTICAL_ERROR.bias_budget_um,
                        "total_unc_um": round(OPTICAL_ERROR.total_unc_um, 2)},
            "thk-101": {"noise_std_um": ULTRASONIC_ERROR.noise_std_um,
                        "bias_budget_um": ULTRASONIC_ERROR.bias_budget_um,
                        "total_unc_um": round(ULTRASONIC_ERROR.total_unc_um, 2)},
            "gates": {"degrade_z": DEGRADE_Z, "fault_z": FAULT_Z},
        },
        "credibility_note": (
            "Synthetic dry run; error models are L0 assumptions to be "
            "characterized on the reference cell, not calibrated hardware."),
        "scenarios": rows,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Dry-run the co-live optical+ultrasonic deposit metrology pair")
    parser.add_argument("--output",
                        default="experiments/data/deposit_metrology_report.json")
    parser.add_argument("--fig", default="docs/figures/deposit_metrology_dryrun.png")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = run_deposit_metrology_demo(n_steps=args.steps, seed=args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    _figure(report, Path(args.fig))

    for row in report["scenarios"]:
        v = row["final_verdict"]
        print(f"{row['scenario']:24s} verdict={v['level']:8s} "
              f"suspect={v['suspect']:9s} mode={row['operating_twin']['mode']:9s} "
              f"trip={row['expected_trip']} pass={row['passed']}")
    print(f"Deposit metrology dry-run {'PASS' if report['all_passed'] else 'FAIL'}; "
          f"report: {args.output}")


def _figure(report: Dict[str, Any], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    agree = next(r for r in report["scenarios"] if r["scenario"] == "agree")
    foul = next(r for r in report["scenarios"] if r["scenario"] == "optical_foul")
    t_a = [p["t_hr"] for p in agree["timeline"]]
    t_f = [p["t_hr"] for p in foul["timeline"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    for key, label, color in (("truth_um", "truth", "#222"),
                              ("opt_um", "opt-101", "#167c80"),
                              ("us_um", "thk-101", "#c2542f")):
        ax.plot(t_a, [p[key] for p in agree["timeline"]], marker="o",
                ms=3, label=label, color=color)
    ax.set_title("AGREE: both channels track the reference deposit")
    ax.set_xlabel("t (hr)")
    ax.set_ylabel("thickness (um)")
    ax.legend()
    ax.grid(alpha=.25)

    ax = axes[1]
    ax.plot(t_f, [p["z"] for p in foul["timeline"]], marker="o", ms=3,
            color="#167c80", label="optical_foul |z|")
    ax.axhline(DEGRADE_Z, color="tab:orange", ls="--", label="degrade z=3")
    ax.axhline(FAULT_Z, color="tab:red", ls="--", label="fault z=6")
    ax.set_title("DISAGREE: optical fouling trips the pair; z rises past gates")
    ax.set_xlabel("t (hr)")
    ax.set_ylabel("|d| / u_c")
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
