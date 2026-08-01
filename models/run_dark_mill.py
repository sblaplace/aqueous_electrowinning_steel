"""
CLI runner for dark mill site assessments.

Usage:
    python -m models.run_dark_mill                              # all sites, pure iron
    python -m models.run_dark_mill --site pickle_liquor_us_midwest
    python -m models.run_dark_mill --site pickle_liquor_us_midwest --grade AISI_1040
    python -m models.run_dark_mill --site pickle_liquor_us_midwest --grade AISI_8620 --route codeposit
    python -m models.run_dark_mill --compare                    # comparison table only
    python -m models.run_dark_mill --list-grades                # show available grades
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from models.dark_mill import (
    EXAMPLE_SITES, run_all_sites, comparison_table, size_dark_mill,
)
from models.steel_grade import (
    STEEL_GRADES, select_route,
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


def _print_grades():
    """List available steel grades."""
    print(f"{'Key':<20} {'Name':<45} {'C wt%':>6} {'Route':>10}")
    print("-" * 85)
    for key, grade in STEEL_GRADES.items():
        route = select_route(grade)
        print(f"{key:<20} {grade.name:<45} {grade.c_wt_percent_target:>5.2f}% {route:>10}")


def main():
    parser = argparse.ArgumentParser(description="Dark Mill site assessment")
    parser.add_argument("--site", type=str, default=None,
                        help=f"Site key to assess. Options: {list(EXAMPLE_SITES.keys())}")
    parser.add_argument("--grade", type=str, default=None,
                        help=f"Target steel grade. Options: {list(STEEL_GRADES.keys())}")
    parser.add_argument("--route", type=str, default=None,
                        choices=["none", "carburize", "codeposit"],
                        help="Post-processing route (auto-selected if omitted)")
    parser.add_argument("--compare", action="store_true",
                        help="Print comparison table only")
    parser.add_argument("--list-grades", action="store_true",
                        help="List available steel grades")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--output-dir", type=str, default="experiments/data",
                        help="Output directory for reports")
    args = parser.parse_args()

    if args.list_grades:
        _print_grades()
        return

    # Resolve grade
    grade = None
    if args.grade:
        if args.grade in STEEL_GRADES:
            grade = STEEL_GRADES[args.grade]
        else:
            print(f"Unknown grade: {args.grade}")
            print(f"Available: {list(STEEL_GRADES.keys())}")
            sys.exit(1)

    if args.site:
        # Single site
        site = EXAMPLE_SITES[args.site]
        if grade is not None:
            site = site.__class__(
                name=site.name,
                feedstock_key=site.feedstock_key,
                grid=site.grid,
                climate=site.climate,
                feedstock_distance_km=site.feedstock_distance_km,
                product_market_km=site.product_market_km,
                target_capacity_t_Fe_yr=site.target_capacity_t_Fe_yr,
                available_area_m2=site.available_area_m2,
                labor_cost_per_yr=site.labor_cost_per_yr,
                notes=site.notes,
                bath=site.bath,
                geometry=site.geometry,
                conditions=site.conditions,
                target_grade=grade,
                post_processing_route=args.route,
            )
        report = size_dark_mill(site)
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
