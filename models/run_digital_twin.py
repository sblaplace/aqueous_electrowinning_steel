"""
Driver — digital twin simulation over a 24-hour run.

Simulates sensor readings with noise and injected faults, runs the
digital twin, and produces four PNG figures:

* digital_twin_state.png      – tracked state variables vs time
* digital_twin_anomalies.png  – anomaly markers on sensor timelines
* digital_twin_confidence.png – confidence trajectory over time
* digital_twin_prediction.png – forward prediction envelope

Usage
-----
python -m models.run_digital_twin
python -m models.run_digital_twin --hours 48 --dt 0.05
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.digital_twin import (  # noqa: E402
    DigitalTwin,
    STATE_KEYS,
    STATE_INDEX,
    generate_synthetic_readings,
)


def run_simulation(
    hours: float = 24.0,
    dt_hr: float = 0.1,
    seed: int = 42,
) -> dict:
    """Run the digital twin simulation.

    Returns dict with history, anomalies, confidence, and prediction.
    """
    rng = np.random.default_rng(seed)
    twin = DigitalTwin(seed=seed)

    design_point = twin.design_point
    n_steps = int(hours / dt_hr)

    # Inject faults at specific times
    fault_schedule: dict = {
        int(8.0 / dt_hr): {"tag": "TT-101", "kind": "bias", "magnitude": 8.0},      # temp bias
        int(15.0 / dt_hr): {"tag": "VT-201", "kind": "spike", "magnitude": 1.5},     # voltage spike
    }

    times = []
    state_log = {k: [] for k in STATE_KEYS}
    state_sigma = {k: [] for k in STATE_KEYS}

    for step in range(n_steps):
        t_hr = step * dt_hr

        # Fault injection
        fault = fault_schedule.get(step)
        # Spike is transient (lasts 3 steps)
        if fault and fault["kind"] == "spike":
            for s in range(step, min(step + 3, n_steps)):
                fault_schedule[s] = {**fault, "spike_active": True}
            fault_schedule.pop(step + 3, None) if (step + 3) in fault_schedule else None

        readings = generate_synthetic_readings(design_point, t_hr, rng, fault=fault)
        state = twin.update(readings, dt_hr=dt_hr)

        times.append(t_hr)
        for k in STATE_KEYS:
            state_log[k].append(float(state.state_mean[STATE_INDEX[k]]))
            state_sigma[k].append(float(np.sqrt(max(state.state_covariance[STATE_INDEX[k], STATE_INDEX[k]], 0))))

    # Confidence trajectory
    conf_traj = twin.confidence_trajectory()

    # Forward prediction
    prediction = twin.predict_ahead(horizon_hr=12.0, n_steps=60)

    # Anomaly report
    anomalies = twin.anomaly_report()

    return {
        "times": np.array(times),
        "state_log": {k: np.array(v) for k, v in state_log.items()},
        "state_sigma": {k: np.array(v) for k, v in state_sigma.items()},
        "confidence": conf_traj,
        "prediction": prediction,
        "anomalies": anomalies,
        "twin": twin,
    }


def plot_state(result: dict, save_path: Path) -> None:
    """Plot tracked state variables with 2-sigma bands."""
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
    axes = axes.flatten()

    times = result["times"]
    for idx, key in enumerate(STATE_KEYS):
        ax = axes[idx]
        mean = result["state_log"][key]
        sigma2 = 2.0 * result["state_sigma"][key]
        ax.plot(times, mean, linewidth=0.8, label="EKF estimate")
        ax.fill_between(times, mean - sigma2, mean + sigma2, alpha=0.2, label="2σ band")
        ax.set_ylabel(key.replace("_", " "))
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(len(STATE_KEYS), len(axes)):
        axes[idx].set_visible(False)

    # Set xlabel on bottom axes
    axes[4].set_xlabel("Time (hr)")
    axes[5].set_xlabel("Time (hr)")
    axes[6].set_xlabel("Time (hr)")

    fig.suptitle("Digital Twin — State Estimation", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_anomalies(result: dict, save_path: Path) -> None:
    """Plot anomalies overlaid on sensor timelines."""
    anomalies = result["anomalies"]
    times = result["times"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Group anomalies by sensor
    by_sensor: dict = {}
    for a in anomalies:
        by_sensor.setdefault(a.sensor_tag, []).append(a)

    # Plot a few key sensors
    for i, tag in enumerate(["TT-101", "VT-201"]):
        key_map = {"TT-101": "catholyte_temperature", "VT-201": "cell_voltage"}
        state_key = key_map.get(tag)
        if state_key and state_key in result["state_log"]:
            ax = axes[i]
            mean = result["state_log"][state_key]
            sigma2 = 2.0 * result["state_sigma"][state_key]
            ax.plot(times, mean, linewidth=0.8, label=f"{tag} estimate")
            ax.fill_between(times, mean - sigma2, mean + sigma2, alpha=0.15)

            # Mark anomalies
            for a in by_sensor.get(tag, []):
                color = {"residual": "red", "drift": "orange", "rate_of_change": "purple"}.get(a.kind, "gray")
                ax.axvline(a.timestamp_hr, color=color, alpha=0.5, linewidth=1.5, linestyle="--")
                idx = min(int(a.timestamp_hr / 0.1), len(mean)-1)
                ax.annotate(a.kind, (a.timestamp_hr, mean[idx]),
                           fontsize=6, color=color, rotation=45, xytext=(5, 5), textcoords="offset points")

            ax.set_ylabel(f"{tag} ({state_key})")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (hr)")
    fig.suptitle("Digital Twin — Anomaly Detection", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confidence(result: dict, save_path: Path) -> None:
    """Plot confidence trajectory."""
    conf = result["confidence"]
    ts = [c[0] for c in conf]
    vals = [c[1] for c in conf]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts, vals, linewidth=1.2, color="steelblue")
    ax.set_xlabel("Time (hr)")
    ax.set_ylabel("P(all specs met)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Digital Twin — Confidence Trajectory")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_prediction(result: dict, save_path: Path) -> None:
    """Plot forward prediction envelope."""
    pred = result["prediction"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_indices = [
        ("catholyte_temperature", STATE_INDEX["catholyte_temperature"]),
        ("cell_voltage", STATE_INDEX["cell_voltage"]),
        ("current_density", STATE_INDEX["current_density"]),
        ("confidence", None),
    ]

    for ax, (label, idx) in zip(axes.flatten(), plot_indices):
        if idx is not None:
            mean = pred.mean_trajectories[:, idx]
            sigma = pred.sigma_trajectories[:, idx]
            ax.plot(pred.timestamps_hr, mean, linewidth=1.0, label="predicted mean")
            ax.fill_between(pred.timestamps_hr, mean - 2*sigma, mean + 2*sigma,
                          alpha=0.2, color="steelblue", label="2σ envelope")
            ax.set_ylabel(label.replace("_", " "))
        else:
            ax.plot(pred.timestamps_hr, pred.confidence, linewidth=1.0, color="green", label="confidence")
            ax.set_ylabel("P(all specs met)")
            ax.set_ylim(0, 1.05)
        ax.set_xlabel("Time (hr)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Digital Twin — 12h Prediction Envelope", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Digital twin 24h simulation")
    parser.add_argument("--hours", type=float, default=24.0, help="Simulation duration (hr)")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step (hr)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--output-dir", type=str, default=str(FIG_DIR), help="Output directory for PNGs")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Running digital twin simulation: {args.hours}h, dt={args.dt}h, seed={args.seed}")
    result = run_simulation(hours=args.hours, dt_hr=args.dt, seed=args.seed)

    anomalies = result["anomalies"]
    conf = result["confidence"]
    print(f"  Steps: {len(result['times'])}")
    print(f"  Anomalies detected: {len(anomalies)}")
    print(f"  Final confidence: {conf[-1][1]:.4f}" if conf else "  No data")

    # Verify prediction envelope widens
    pred = result["prediction"]
    sigma_first = pred.sigma_trajectories[0, 0]
    sigma_last = pred.sigma_trajectories[-1, 0]
    print(f"  Prediction envelope: σ(t=0)={sigma_first:.4f}, σ(t={pred.horizon_hr}h)={sigma_last:.4f}")
    assert sigma_last > sigma_first, "Prediction envelope should widen with horizon"

    plot_state(result, out / "digital_twin_state.png")
    plot_anomalies(result, out / "digital_twin_anomalies.png")
    plot_confidence(result, out / "digital_twin_confidence.png")
    plot_prediction(result, out / "digital_twin_prediction.png")
    print(f"  Figures saved to {out}")


if __name__ == "__main__":
    main()
