"""
Driver — FMEA (Failure Mode and Effects Analysis) for the full chain.

Produces 3 output plots:
  fmea_rpn_ranking.png        — horizontal bar chart of RPN by failure mode
  fmea_risk_matrix.png        — severity × occurrence scatter (bubble = detection)
  fmea_mitigation_impact.png  — before/after RPN for mitigated modes

Usage:
  python -m models.run_fmea                     # standalone FMEA
  python -m models.run_fmea --with-mc           # adjust from Monte Carlo
  python -m models.run_fmea --mc-samples 500    # custom MC sample count
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "experiments" / "data"
FIG_DIR = ROOT / "docs" / "figures"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.uncertainty.fmea import (
    FMEAReport,
    generate_fmea,
    critical_failure_paths,
    mitigation_roadmap,
)


# ---------------------------------------------------------------------------
# Plot 1: RPN ranking
# ---------------------------------------------------------------------------

def plot_rpn_ranking(fmea: FMEAReport, out_path: Path) -> None:
    """Horizontal bar chart of RPN for all failure modes, color-coded by risk."""
    modes = fmea.ranked()
    if not modes:
        return

    labels = [f"{fm.id} {fm.mode}" for fm in modes]
    rpns = [fm.rpn for fm in modes]

    # Color: red for RPN>100, orange for 50-100, green for <50
    colors = []
    for rpn in rpns:
        if rpn > 100:
            colors.append("#e41a1c")
        elif rpn > 50:
            colors.append("#ff7f00")
        else:
            colors.append("#4daf4a")

    fig, ax = plt.subplots(figsize=(10, max(8, len(modes) * 0.4)))
    y_pos = range(len(modes))
    bars = ax.barh(y_pos, rpns, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Risk Priority Number (RPN)")
    ax.set_title(f"FMEA — RPN Ranking ({fmea.total} failure modes)", fontweight="bold")

    # RPN threshold line
    ax.axvline(100, color="red", ls="--", alpha=0.5, label="Critical threshold (100)")
    ax.axvline(50, color="orange", ls="--", alpha=0.3, label="Moderate threshold (50)")

    # Value labels on bars
    for bar, rpn in zip(bars, rpns):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                str(rpn), va="center", fontsize=7)

    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Risk matrix (severity × occurrence, bubble = detection)
# ---------------------------------------------------------------------------

def plot_risk_matrix(fmea: FMEAReport, out_path: Path) -> None:
    """Scatter plot: x=occurrence, y=severity, bubble size=detection."""
    modes = fmea.failure_modes
    if not modes:
        return

    fig, ax = plt.subplots(figsize=(9, 7))

    # Background risk zones
    # Low risk: S*O < 20 (green zone)
    # Medium risk: 20 <= S*O < 50 (yellow zone)
    # High risk: S*O >= 50 (red zone)
    for s in range(1, 11):
        for o in range(1, 11):
            so = s * o
            if so >= 50:
                color = "#fee0d2"
            elif so >= 20:
                color = "#fff7bc"
            else:
                color = "#e5f5e0"
            ax.add_patch(Rectangle((o - 0.5, s - 0.5), 1, 1,
                                       facecolor=color, edgecolor="white", linewidth=0.3))

    # Plot failure modes as bubbles
    for fm in modes:
        # Bubble size proportional to detection (1-10 → marker size 50-500)
        size = 50 + fm.detection * 45
        color = "#e41a1c" if fm.rpn > 100 else "#ff7f00" if fm.rpn > 50 else "#4daf4a"
        ax.scatter(fm.occurrence, fm.severity, s=size, c=color, alpha=0.7,
                   edgecolors="black", linewidth=0.5, zorder=5)
        ax.annotate(fm.id, (fm.occurrence, fm.severity),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=6, fontweight="bold")

    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.5, 10.5)
    ax.set_xlabel("Occurrence (O)", fontsize=10)
    ax.set_ylabel("Severity (S)", fontsize=10)
    ax.set_title("FMEA Risk Matrix (bubble size = Detection)", fontweight="bold")
    ax.set_xticks(range(1, 11))
    ax.set_yticks(range(1, 11))
    ax.grid(True, alpha=0.2)

    # Legend for bubble sizes
    for d_label, d_val in [("D=2", 2), ("D=5", 5), ("D=9", 9)]:
        ax.scatter([], [], s=50 + d_val * 45, c="gray", alpha=0.5,
                   edgecolors="black", linewidth=0.5, label=d_label)
    ax.legend(title="Detection", fontsize=8, title_fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Mitigation impact
# ---------------------------------------------------------------------------

def plot_mitigation_impact(fmea: FMEAReport, out_path: Path) -> None:
    """Grouped bar chart: current RPN vs residual RPN for mitigated modes."""
    mitigated = [fm for fm in fmea.ranked() if fm.mitigation and fm.residual_rpn > 0]
    if not mitigated:
        return

    labels = [f"{fm.id} {fm.mode}" for fm in mitigated]
    current_rpns = [fm.rpn for fm in mitigated]
    residual_rpns = [fm.residual_rpn for fm in mitigated]

    x = np.arange(len(mitigated))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, max(6, len(mitigated) * 0.4)))
    ax.barh(x - width / 2, current_rpns, width, label="Current RPN",
                     color="#e41a1c", alpha=0.8)
    ax.barh(x + width / 2, residual_rpns, width, label="Residual RPN",
                     color="#4daf4a", alpha=0.8)

    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("RPN")
    ax.set_title("FMEA — Mitigation Impact (current → residual RPN)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.axvline(100, color="red", ls="--", alpha=0.3)
    ax.grid(axis="x", alpha=0.3)

    # Reduction labels
    for i, fm in enumerate(mitigated):
        reduction = fm.rpn_reduction
        pct = (reduction / fm.rpn * 100) if fm.rpn > 0 else 0
        ax.text(max(fm.rpn, fm.residual_rpn) + 5, i,
                f"-{reduction} ({pct:.0f}%)", va="center", fontsize=7, color="#333")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    with_mc: bool = False,
    mc_samples: int = 500,
    mc_spec_set: str = "ASTM_A36",
) -> FMEAReport:
    print("=" * 72)
    print("FMEA — Failure Mode and Effects Analysis")
    print("=" * 72)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    mc_result = None
    if with_mc:
        print(f"\nRunning Monte Carlo (N={mc_samples}) to calibrate FMEA ratings...")
        from models.uncertainty.monte_carlo import MonteCarloEngine
        from models.uncertainty.specification import SPECS_A36

        engine = MonteCarloEngine(n_samples=mc_samples, seed=42, n_jobs=1)
        mc_result = engine.run(specs=SPECS_A36, spec_set_name=mc_spec_set)
        print(f"  MC confidence: {mc_result.overall_confidence*100:.1f}%")

    fmea = generate_fmea(mc_result=mc_result)

    # Summary
    print(f"\nTotal failure modes: {fmea.total}")
    print(f"Mean RPN: {fmea.mean_rpn:.1f}")
    print(f"Max RPN: {fmea.max_rpn}")
    print(f"Critical (RPN>100): {fmea.critical_count}")

    # Critical paths
    critical = critical_failure_paths(fmea)
    if critical:
        print("\nCritical failure paths (RPN > 100):")
        for fm in critical:
            print(f"  {fm.id}: {fm.mode} — RPN={fm.rpn} (S={fm.severity} O={fm.occurrence} D={fm.detection})")

    # Mitigation roadmap
    roadmap = mitigation_roadmap(fmea)
    print(f"\nMitigation roadmap ({len(roadmap)} actions):")
    for action in roadmap[:5]:
        print(f"  P{action['priority']} {action['id']}: {action['mode']} "
              f"RPN {action['rpn']}→{action['residual_rpn']} "
              f"(-{action['rpn_reduction']}, {action['estimated_effort']})")
    if len(roadmap) > 5:
        print(f"  ... and {len(roadmap) - 5} more")

    # Generate figures
    print("\nGenerating figures...")
    plot_rpn_ranking(fmea, FIG_DIR / "fmea_rpn_ranking.png")
    print(f"  ✅ {FIG_DIR / 'fmea_rpn_ranking.png'}")

    plot_risk_matrix(fmea, FIG_DIR / "fmea_risk_matrix.png")
    print(f"  ✅ {FIG_DIR / 'fmea_risk_matrix.png'}")

    plot_mitigation_impact(fmea, FIG_DIR / "fmea_mitigation_impact.png")
    print(f"  ✅ {FIG_DIR / 'fmea_mitigation_impact.png'}")

    # JSON report
    report = fmea.to_dict()
    report["critical_paths"] = [
        {"id": fm.id, "mode": fm.mode, "rpn": fm.rpn}
        for fm in critical
    ]
    report["mitigation_roadmap"] = roadmap

    report_path = DATA_DIR / "fmea_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  ✅ Report: {report_path}")

    return fmea


def cli():
    parser = argparse.ArgumentParser(description="FMEA analysis")
    parser.add_argument("--with-mc", action="store_true",
                        help="Calibrate FMEA from Monte Carlo results")
    parser.add_argument("--mc-samples", type=int, default=500,
                        help="MC sample count for calibration")
    parser.add_argument("--mc-spec-set", type=str, default="ASTM_A36",
                        help="Spec set for MC calibration")
    args = parser.parse_args()
    main(with_mc=args.with_mc, mc_samples=args.mc_samples,
         mc_spec_set=args.mc_spec_set)


if __name__ == "__main__":
    cli()
