"""CLI driver for traceable Phase-I LSV and optional EIS calibration.

Example:
python -m models.run_calibration experiments/data/campaign_manifest.csv P1-001 \
  --pH 3 --temperature-C 60 --fe-conc-M 1 --reference-to-she-V 0.210 \
  --output experiments/data/calibrations/P1-001_parameters.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .calibration import calibrate_lsv_run, fit_eis_exchange_current


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate Fe/HER screening kinetics from a QA-ready Phase-I run.")
    parser.add_argument("manifest", help="Campaign manifest CSV")
    parser.add_argument("run_id", help="QA-ready LSV/CV run_id")
    parser.add_argument("--pH", type=float, required=True)
    parser.add_argument("--temperature-C", type=float, required=True)
    parser.add_argument("--fe-conc-M", type=float, required=True)
    parser.add_argument("--reference-to-she-V", type=float, required=True,
                        help="Additive conversion from recorded reference potential to SHE")
    parser.add_argument("--eis", help="Optional mapped EIS CSV for a consistency fit")
    parser.add_argument("--no-warburg", action="store_true", help="Fit optional EIS without a Warburg element")
    parser.add_argument("--output", required=True, help="Versioned JSON parameter-report path")
    args = parser.parse_args()
    report = calibrate_lsv_run(
        args.manifest, args.run_id, pH=args.pH, temperature_C=args.temperature_C,
        fe_conc_M=args.fe_conc_M, reference_to_she_V=args.reference_to_she_V,
    )
    report["input_conditions"] = {
        "pH": args.pH, "temperature_C": args.temperature_C, "fe_conc_M": args.fe_conc_M,
        "reference_to_she_V": args.reference_to_she_V,
    }
    if args.eis:
        report["eis_consistency_fit"] = fit_eis_exchange_current(args.eis, include_warburg=not args.no_warburg)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote calibration report: {output}")


if __name__ == "__main__":
    main()
