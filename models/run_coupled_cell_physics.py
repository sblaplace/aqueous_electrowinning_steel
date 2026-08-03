"""Driver — coupled cell-physics × gas-hold-up operating-point report.

Usage
-----
python -m models.run_coupled_cell_physics
python -m models.run_coupled_cell_physics --json     # also write the JSON copy

Prints the uncoupled-vs-coupled voltage/energy contrast, the energy-gate
reachability verdict under coupling, and the isolated gas impact, then
optionally writes a machine-readable copy to
``experiments/data/coupled_cell_physics_report.json``.

Every number is a Level-0 → Level-1 boundary **prediction**, flagged
``unvalidated (L0)``, and is **not** gate evidence: the FE and energy gates are
measurement-only and live in ``models/process_gates.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .coupled_cell_physics import (
    BOUNDARY_NOTE,
    SCREENING_FLAG,
    coupled_energy_gate_reachable,
    coupled_operating_point,
    coupled_reference_cell,
    coupling_closure,
    gas_impact_summary,
    main as render_report,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments" / "data"


def build_report() -> Dict[str, Any]:
    """Assemble the JSON-serialisable coupled report."""
    reach = coupled_energy_gate_reachable()
    best = reach["best_combination_coupled"]
    cell = coupled_reference_cell(
        gap_m=best["interelectrode_gap_m"],
        contact_resistance_ohm_m2=best["contact_resistance_ohm_m2"],
    )
    j_best = float(best["j_mA_cm2"])
    return {
        "title": "Coupled cell-physics × gas-hold-up operating-point report",
        "purpose": (
            "Re-derive #40's energy-gate reachability verdict with the axial "
            "cathodic-gas void profile, the Bruggeman conductivity penalty and "
            "the current redistribution from models/gas_holdup.py coupled into "
            "the cell-physics voltage/energy solve."
        ),
        "model_tier": "Level-0 → Level-1 boundary (two coupled reduced-order 1-D models)",
        "flag": SCREENING_FLAG,
        "boundary_note": BOUNDARY_NOTE,
        "reachability": {k: v for k, v in reach.items() if k != "window"},
        "window": reach["window"],
        "best_point_detail": coupled_operating_point(cell, j_best),
        "gas_impact": gas_impact_summary(cell, j_best),
        "closure": coupling_closure(cell, j_best),
    }


def main() -> Dict[str, Any]:
    ap = argparse.ArgumentParser(
        description="Coupled cell-physics × gas-hold-up operating-point report"
    )
    ap.add_argument("--json", action="store_true",
                    help="also write experiments/data/coupled_cell_physics_report.json")
    args = ap.parse_args()

    render_report()

    if args.json:
        DATA.mkdir(parents=True, exist_ok=True)
        out = DATA / "coupled_cell_physics_report.json"
        out.write_text(json.dumps(build_report(), indent=2, default=float) + "\n",
                       encoding="utf-8")
        print(f"\nWrote {out.relative_to(ROOT)}")
    return {"flag": SCREENING_FLAG}


if __name__ == "__main__":
    main()
