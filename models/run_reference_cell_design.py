"""Run RC-1 reference-cell design synthesis and write a procurement design report.

Usage:
    python -m models.run_reference_cell_design
    python -m models.run_reference_cell_design --config processes/reference_cell_rc1.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .reference_cell_design import DEFAULT_CONFIG_PATH, load_reference_cell_config, synthesize_reference_cell_design

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "reference_cell_rc1_design_report.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize deployable RC-1 reference-cell design")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Controlled RC-1 YAML input")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="JSON design report path")
    args = parser.parse_args()

    config = load_reference_cell_config(args.config)
    report = synthesize_reference_cell_design(config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    selected = report["selected_design"]
    candidate = selected["candidate"]
    operating = selected["operating"]
    hydraulics = selected["hydraulics"]
    print(f"RC-1 design synthesis: {report['configuration_id']}")
    print(f"  Candidates: {report['feasible_candidate_count']}/{report['candidate_count']} feasible")
    print(
        "  Selected: "
        f"{candidate['active_area_cm2']:.1f} cm², "
        f"{candidate['channel_depth_mm']:.1f} mm channel, "
        f"{candidate['flow_L_min']:.2f} L/min"
    )
    print(
        f"  Duty: {operating['current_A']:.2f} A at {operating['current_density_mA_cm2']:.0f} mA/cm², "
        f"{operating['cell_voltage_V']:.2f} V, FE {operating['faradaic_efficiency']:.1%}"
    )
    print(
        f"  Hydraulics: Re {hydraulics['reynolds_number']:.0f}, "
        f"ΔP {hydraulics['pressure_drop_Pa'] / 1000:.2f} kPa"
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
