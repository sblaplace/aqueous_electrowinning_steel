"""Driver — reference-cell theory-confidence (chain-of-claims) screening report.

Usage
-----
python -m models.run_theory_confidence

Prints the human-readable Level-0 screening report and writes a JSON copy of
the chain-of-claims table and the operating-point / thermal / ledger verdicts
to ``experiments/data/theory_confidence_report.json``.

All numbers are Level-0 screening and **not** gate evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cell_physics import CellPhysics
from .theory_confidence import (
    chain_of_claims,
    close_ledgers,
    main as render_report,
    reference_cell,
    solve_reference,
    thermal_balance,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reference-cell theory-confidence screening report (L0)."
    )
    parser.add_argument(
        "--out",
        default="experiments/data/theory_confidence_report.json",
        help="path to write the JSON report (default: experiments/data/theory_confidence_report.json)",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="only print the human-readable report; do not write JSON",
    )
    args = parser.parse_args()

    print(render_report())

    if args.no_json:
        return

    rc = reference_cell()
    solve = solve_reference(rc)
    cp = CellPhysics(rc.bath, rc.geometry, rc.conditions)
    op = cp.solve_at_j(solve["current_density_mA_cm2"])
    report = {
        "flag": "unvalidated (L0)",
        "not_gate_evidence": True,
        "reference_cell": {
            "name": rc.name,
            "cathode_area_cm2": rc.cathode_area_cm2,
            "volume_L": rc.volume_L,
            "bath": rc.bath.__dict__,
            "temperature_C": rc.conditions.temperature_C,
        },
        "operating_point": solve,
        "thermal": thermal_balance(rc, op),
        "ledgers": close_ledgers(rc, op),
        "chain_of_claims": chain_of_claims(),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
