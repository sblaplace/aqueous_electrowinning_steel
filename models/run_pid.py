"""
Driver for pilot P&ID diagrams.

Generates:
* docs/figures/pid_overview.png
* docs/figures/pid_detailed.png
* experiments/data/pid_report.json

Usage:
    python -m models.run_pid
"""

from __future__ import annotations

from pathlib import Path
import sys
import json
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.pid import generate_pid_overview, generate_pid_detailed

FIG_DIR = ROOT / "docs" / "figures"
DATA_DIR = ROOT / "experiments" / "data"


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("="*72)
    print("PILOT P&ID GENERATION — SCREENING")
    print("="*72)

    p1 = generate_pid_overview()
    print(f"  ✅ Saved: {p1}")

    p2 = generate_pid_detailed()
    print(f"  ✅ Saved: {p2}")

    report = {
        "title": "Pilot P&ID screening — aqueous electrowinning + carburizing + tempering",
        "date": datetime.now().isoformat(),
        "figures": [str(p1), str(p2)],
        "equipment_list": [
            "TK-101 Leaching tank with agitator M-101, LT-101, TT-101, pHAT-101, AT-101 FeTot",
            "FL-101 Filtration",
            "TK-102 Electrolyte prep tank M-102, LT-102, AT-102 Fe2+, TT-102",
            "TK-103 Storage tank LT-103, P-103 pump, FT-103",
            "C-201/E-201 Electrowinning cell stack, FT-201, TT-201, PT-201, VT-201, AT-201 Fe2+/Ni2+",
            "P-201 recirc pump, HE-201 heat exchanger TT-201B, FL-201 filter PDIT-201",
            "TK-202 Recycle CSTR M-202, LT-202, AT-202 impurity, FT-202 purge, TT-202",
            "TK-301A O2 vent AT-301A O2, TK-301B Cl2 scrub pHAT-301B",
            "TK-302 Purge treatment",
            "TK-401 Wash/dry LT-401",
            "F-501 Carburizing retort furnace H-501, TT-501, AIT-501 O2 probe, AIT-502 dew point, AT-501 C foil",
            "GS-501 Gas manifold FT-501A-D CO/CO2/CH4/H2",
            "TK-502 Quench tank TT-502",
            "F-503 Tempering furnace TT-503",
            "PK-601 Product QC AIT-601 HV/XRD/tensile",
        ],
        "instrument_tags": [
            "LT level transmitter, FT flow, TT temperature, PT pressure, VT voltage, AT analyzer, AIT analyzer indicator transmitter, PDIT differential pressure, pHAT",
            "Control loops: LT→LV, FT→FV, TT→TV/H, AT→FC (carbon potential -> gas flow), O2 probe -> CO/CO2 ratio",
        ],
        "notes": [
            "Screening P&ID — not for construction, for HAZOP and pilot costing",
            "Solid black: process flow, dashed orange: electrolyte recycle, dotted red: purge, dashed red: control signal",
            "Includes leaching, electrolyte prep, cell stack with rectifier, recirc with HE and filter, CSTR recycle, gas handling, wash, carburizing with gas manifold and O2/dewpoint, quench, tempering, QC",
            "Add safety interlocks (high T, high pH, low flow) and relief valves for detailed design",
        ],
    }

    report_path = DATA_DIR / "pid_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  ✅ Saved report: {report_path}")
    print("\n✅ P&ID driver complete!")
    return report


if __name__ == "__main__":
    main()
