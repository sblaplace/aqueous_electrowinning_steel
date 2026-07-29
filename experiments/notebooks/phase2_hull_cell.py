"""Executable Phase II Hull-cell current-map and gravimetric-FE analysis.

Run from the repository root with the canonical templates or mapped instrument
exports:

    python experiments/notebooks/phase2_hull_cell.py \
        --trace experiments/data/hull_cell_galvanostatic_template.csv \
        --gravimetry experiments/data/hull_cell_gravimetry_template.csv

The panel map is an *un-calibrated primary-current screening model*.  It
normalizes a variable-gap (angled-panel) distribution to the applied current;
it does not infer real local current from coupon mass or replace a calibrated
Hull-cell map.  Gravimetric output is apparent Fe Faradaic efficiency until
composition and dryness have been verified.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.hull_cell import (
    HullCellGeometry,
    analyze_gravimetric_efficiency,
    current_density_window,
    hull_current_distribution,
    load_galvanostatic_trace,
    load_gravimetry,
    plot_hull_current_distribution,
    summarize_hull_distribution,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace", default="experiments/data/hull_cell_galvanostatic_template.csv",
        help="CSV time/current trace (timestamp_s, current_A)",
    )
    parser.add_argument(
        "--gravimetry", default="experiments/data/hull_cell_gravimetry_template.csv",
        help="CSV containing mass_before_g and mass_after_g",
    )
    parser.add_argument("--output", default="docs/figures/phase2_hull_cell_analysis.png")
    parser.add_argument("--panel-length-cm", type=float, default=10.0)
    parser.add_argument("--panel-width-cm", type=float, default=5.0)
    parser.add_argument("--near-gap-cm", type=float, default=1.5)
    parser.add_argument("--far-gap-cm", type=float, default=9.0)
    parser.add_argument("--segments", type=int, default=100)
    parser.add_argument(
        "--panel-current-A", type=float, default=None,
        help="Positive applied current magnitude for the map; default is charge/duration from the trace",
    )
    parser.add_argument("--minimum-j-mA-cm2", type=float, default=10.0)
    parser.add_argument("--maximum-j-mA-cm2", type=float, default=100.0)
    parser.add_argument("--cathodic-sign", choices=("negative", "positive"), default="negative")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    trace = load_galvanostatic_trace(args.trace)
    gravimetry = load_gravimetry(args.gravimetry)
    result = analyze_gravimetric_efficiency(
        trace, gravimetry, cathodic_sign=args.cathodic_sign,
    )

    geometry = HullCellGeometry(
        panel_length_cm=args.panel_length_cm,
        panel_width_cm=args.panel_width_cm,
        near_edge_gap_cm=args.near_gap_cm,
        far_edge_gap_cm=args.far_gap_cm,
    )
    panel_current_A = args.panel_current_A or result.equivalent_mean_cathodic_current_A
    if panel_current_A <= 0:
        raise ValueError("--panel-current-A must be positive")
    distribution = hull_current_distribution(geometry, panel_current_A, args.segments)
    summary = summarize_hull_distribution(distribution)
    window = current_density_window(
        distribution, args.minimum_j_mA_cm2, args.maximum_j_mA_cm2,
    )

    print("Hull-cell primary-current screening map")
    print({
        "panel_angle_deg": round(geometry.panel_angle_deg, 3),
        "panel_current_A": panel_current_A,
        "near_edge_j_mA_cm2": summary["near_edge_current_density_mA_cm2"],
        "far_edge_j_mA_cm2": summary["far_edge_current_density_mA_cm2"],
        "window_area_fraction": window["area_fraction"],
        "window_position_cm": (
            window["position_start_cm_from_near_edge"],
            window["position_end_cm_from_near_edge"],
        ),
    })
    print("Apparent gravimetric Faradaic efficiency")
    print(result.summary())

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    plot_hull_current_distribution(distribution, geometry, axes=axes)
    fig.suptitle(
        f"Phase II current screen; apparent gravimetric FE = "
        f"{result.apparent_faradaic_efficiency_percent:.1f}%",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
