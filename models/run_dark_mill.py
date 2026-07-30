"""
CLI runner for dark mill site assessments.

Usage:
    python -m models.run_dark_mill                  # all sites
    python -m models.run_dark_mill --site pickle_liquor_us_midwest
    python -m models.run_dark_mill --compare        # comparison table only
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from models.dark_mill import (
    EXAMPLE_SITES, run_site, run_all_sites, comparison_table, size_dark_mill,
)


def _json_default(obj):
    """Handle numpy types in JSON serialization."""
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main():
    parser = argparse.ArgumentParser(description="Dark Mill site assessment")
    parser.add_argument("--site", type=str, default=None,
                        help=f"Site key to assess. Options: {list(EXAMPLE_SITES.keys())}")
    parser.add_argument("--compare", action="store_true",
                        help="Print comparison table only")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--output-dir", type=str, default="experiments/data",
                        help="Output directory for reports")
    args = parser.parse_args()

    if args.site:
        # Single site
        report = run_site(args.site)
        if args.json:
            _print_json(report)
        else:
            print(report.summary())
        _save_report(report, args.output_dir)
    elif args.compare:
        # Comparison table
        reports = run_all_sites()
        print(comparison_table(reports))
    else:
        # All sites
        reports = run_all_sites()
        for key, report in reports.items():
            print(report.summary())
            print()
            _save_report(report, args.output_dir)
        print()
        print(comparison_table(reports))


def _save_report(report, output_dir: str):
    """Save a site report as JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    site_slug = report.site.name.lower().replace(" ", "_").replace("—", "-")[:40]
    path = out / f"dark_mill_{site_slug}.json"

    data = {
        "site_name": report.site.name,
        "feedstock": report.site.feedstock_key,
        "stack_design": {
            "n_stacks": report.stack_design.n_stacks,
            "cells_per_stack": report.stack_design.cells_per_stack,
            "electrode_area_m2": report.stack_design.electrode_area_m2,
            "current_density_mA_cm2": report.stack_design.current_density_mA_cm2,
            "cell_voltage_V": report.stack_design.cell_voltage_V,
            "total_power_kW": round(report.stack_design.total_power_kW, 1),
            "annual_production_t": round(report.stack_design.annual_production_t(), 0),
        },
        "mass_balance": {
            k: round(v, 2) if isinstance(v, float) else v
            for k, v in report.mass_balance.__dict__.items()
        },
        "capex": report.capex,
        "opex": report.opex,
        "lcofe": report.lcofe,
        "thermal": report.thermal,
        "go_no_go": report.go_no_go,
    }

    path.write_text(json.dumps(data, indent=2, default=_json_default))
    print(f"  Report saved: {path}")


def _print_json(report):
    """Print report as JSON to stdout."""
    data = {
        "site_name": report.site.name,
        "feedstock": report.site.feedstock_key,
        "stack_design": {
            "n_stacks": report.stack_design.n_stacks,
            "cells_per_stack": report.stack_design.cells_per_stack,
            "electrode_area_m2": report.stack_design.electrode_area_m2,
            "current_density_mA_cm2": report.stack_design.current_density_mA_cm2,
            "cell_voltage_V": report.stack_design.cell_voltage_V,
            "total_power_kW": round(report.stack_design.total_power_kW, 1),
            "annual_production_t": round(report.stack_design.annual_production_t(), 0),
        },
        "mass_balance": {
            k: round(v, 2) if isinstance(v, float) else v
            for k, v in report.mass_balance.__dict__.items()
        },
        "capex": report.capex,
        "opex": report.opex,
        "lcofe": report.lcofe,
        "thermal": report.thermal,
        "go_no_go": report.go_no_go,
    }
    print(json.dumps(data, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
