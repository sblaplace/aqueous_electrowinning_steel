"""
Driver — confidence report generation for multiple design points.

Usage
-----
python -m models.run_confidence_report                         # all design points, A36
python -m models.run_confidence_report --spec-set AISI_1020    # different spec set
python -m models.run_confidence_report --n-samples 1000        # quick run
python -m models.run_confidence_report --design-point dc       # single design point

Outputs (per design point)
-------
* confidence_report_{dp}.png   – 4-quadrant summary figure
* confidence_report_{dp}.json – full machine-readable report

Design points
-------------
* dc  — direct current (j_avg=150, waveform=dc)
* pe  — pulsed electrodeposition (j_avg=150, j_peak=300, duty=0.5, waveform=pe)
* pre — pulse reversal (j_avg=150, j_peak=300, duty=0.5, waveform=pre)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.uncertainty.confidence_report import (
    ConfidenceReport,
    generate_confidence_report,
    plot_confidence_report,
)
from models.uncertainty.monte_carlo import DEFAULT_DESIGN_POINT


# ---------------------------------------------------------------------------
# Design point variants
# ---------------------------------------------------------------------------

DESIGN_POINTS = {
    "dc": {
        **DEFAULT_DESIGN_POINT,
        "j_avg_mA_cm2": 150.0,
        "j_peak_mA_cm2": 150.0,
        "duty_cycle": 1.0,
        "waveform": "dc",
    },
    "pe": {
        **DEFAULT_DESIGN_POINT,
        "j_avg_mA_cm2": 150.0,
        "j_peak_mA_cm2": 300.0,
        "duty_cycle": 0.5,
        "waveform": "pe",
    },
    "pre": {
        **DEFAULT_DESIGN_POINT,
        "j_avg_mA_cm2": 150.0,
        "j_peak_mA_cm2": 300.0,
        "duty_cycle": 0.5,
        "waveform": "pre",
    },
}


def run_and_save(
    name: str,
    design_point: dict,
    spec_set: str,
    n_samples: int,
    target: float,
    seed: int,
) -> ConfidenceReport:
    """Generate and save a confidence report for one design point."""
    print(f"\n{'='*60}")
    print(f"  Design point: {name}")
    print(f"  Spec set: {spec_set}")
    print(f"  MC samples: {n_samples}")
    print(f"{'='*60}\n")

    report = generate_confidence_report(
        design_point=design_point,
        spec_set=spec_set,
        mc_samples=n_samples,
        target=target,
        seed=seed,
    )

    # Save JSON
    json_path = FIG_DIR / f"confidence_report_{name}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
    print(f"  JSON saved: {json_path}")

    # Save figure
    png_path = FIG_DIR / f"confidence_report_{name}.png"
    plot_confidence_report(report, out_path=str(png_path))
    print(f"  Figure saved: {png_path}")

    # Print summary
    print(f"\n  {report.summary_text()}")
    print(f"  Overall confidence: {report.overall_confidence * 100:.1f}%")
    print(f"  Verdict: {report.verdict}")
    print(f"  Critical failures: {len(report.critical_failures)}")
    print(f"  Recommended experiments: {len(report.recommended_experiments)}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate confidence reports")
    parser.add_argument(
        "--spec-set", default="A36",
        help="Specification set (A36, AISI_1010, AISI_1020, CARBURIZED)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=1000,
        help="Monte Carlo sample count (default: 1000)",
    )
    parser.add_argument(
        "--target", type=float, default=0.95,
        help="Target confidence level (default: 0.95)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--design-point",
        choices=list(DESIGN_POINTS.keys()) + ["all"],
        default="all",
        help="Which design point(s) to run",
    )
    args = parser.parse_args()

    if args.design_point == "all":
        points = DESIGN_POINTS
    else:
        points = {args.design_point: DESIGN_POINTS[args.design_point]}

    reports = {}
    for name, dp in points.items():
        reports[name] = run_and_save(
            name=name,
            design_point=dp,
            spec_set=args.spec_set,
            n_samples=args.n_samples,
            target=args.target,
            seed=args.seed,
        )

    # Print summary table
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Design Point':<12} {'Confidence':>12} {'Verdict':<18} {'Critical':>8}")
    print(f"  {'-'*12} {'-'*12} {'-'*18} {'-'*8}")
    for name, rpt in reports.items():
        print(
            f"  {name:<12} {rpt.overall_confidence * 100:>11.1f}% "
            f"{rpt.verdict:<18} {len(rpt.critical_failures):>8}"
        )


if __name__ == "__main__":
    main()
