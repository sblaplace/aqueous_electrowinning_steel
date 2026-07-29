"""Driver script for the calibration pipeline.

Reads experimental CSVs, fits all 7 model domains, produces:
  - calibrated_parameters.json
  - 3 diagnostic figures (calibration_*.png + calibration_summary.png)

Usage:
    python -m models.run_calibration_pipeline [data_dir] [--output-dir OUTPUT] [--domains d1,d2,...]
    aq-steel-calibrate [data_dir]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .calibration_pipeline import (
    CalibrationReport,
    generate_calibration_figures,
    load_calibrated_params,
    run_calibration_pipeline,
    validate_calibration,
    load_csv_safe,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Auto-calibrate model coefficients from experimental CSVs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Domains: tafel, eis, hull_cell, diffusivity, carbon_potential, tempering, hall_petch
Expected CSVs in data_dir: <domain>.csv for each domain.

Output:
  calibrated_parameters.json  — all fitted coefficients
  calibration_figures/        — diagnostic plots
""",
    )
    parser.add_argument("data_dir", type=Path, help="Directory containing domain CSVs")
    parser.add_argument("--output-dir", type=Path, default=Path("calibration_output"),
                        help="Output directory (default: calibration_output)")
    parser.add_argument("--domains", type=str, default=None,
                        help="Comma-separated list of domains to calibrate (default: all)")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation")
    parser.add_argument("--validate", action="store_true",
                        help="Cross-validate fitted parameters against data")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    domains = args.domains.split(",") if args.domains else None

    print(f"Calibration Pipeline")
    print(f"  Data dir:    {args.data_dir}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"  Domains:     {', '.join(domains) if domains else 'all'}")
    print()

    # Run pipeline
    report = run_calibration_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        domains=domains,
        generate_figures=not args.no_figures,
    )

    # Summary
    print()
    print("=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)

    if not report.domain_results:
        print("No domains were successfully calibrated.")
        print(f"  Check that CSV files exist in {args.data_dir}")
        return 1

    summary = report.summary_table()
    print(summary.to_string(index=False))
    print()

    for domain, result in report.domain_results.items():
        print(f"[{domain}]")
        for name, value in result.parameters.items():
            print(f"  {name}: {value:.6e}")
        if result.r_squared is not None:
            print(f"  R²: {result.r_squared:.4f}")
        print(f"  Converged: {result.converged}")
        if result.notes:
            print(f"  Notes: {result.notes}")
        print()

    if report.output_path:
        print(f"Calibrated parameters written to: {report.output_path}")

    # Cross-validation
    if args.validate:
        print()
        print("=" * 60)
        print("CROSS-VALIDATION")
        print("=" * 60)
        for domain, result in report.domain_results.items():
            csv_path = args.data_dir / f"{domain}.csv"
            data = load_csv_safe(csv_path, domain)
            if data is not None:
                vr = validate_calibration(domain, data, result.parameters)
                print(f"  {domain}: R²={vr.r_squared:.4f}, RMSE={vr.rmse:.2e}, "
                      f"bounds_ok={vr.within_physical_bounds}, "
                      f"better_than_default={vr.improvement_over_default}")

    # Figure listing
    fig_dir = args.output_dir / "calibration_figures"
    if fig_dir.exists():
        figs = sorted(fig_dir.glob("*.png"))
        print(f"\nGenerated {len(figs)} figures in {fig_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
