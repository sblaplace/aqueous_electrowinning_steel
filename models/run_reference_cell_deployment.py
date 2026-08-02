"""Generate RC-1 P&ID, wiring/sensor schedule, and controlled BOM artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .reference_cell_design import DEFAULT_CONFIG_PATH, load_reference_cell_config, synthesize_reference_cell_design
from .reference_cell_deployment import render_deployment_markdown, write_deployment_package

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESIGN_REPORT = ROOT / "outputs" / "reference_cell_rc1_design_report.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "RC1_DEPLOYMENT_PACKAGE.md"
DEFAULT_MANIFEST = ROOT / "outputs" / "reference_cell_rc1_deployment_manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the RC-1 deployment package")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--design-report", default=str(DEFAULT_DESIGN_REPORT))
    parser.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN))
    parser.add_argument("--manifest-output", default=str(DEFAULT_MANIFEST))
    args = parser.parse_args()

    config = load_reference_cell_config(args.config)
    design_path = Path(args.design_report)
    if design_path.exists():
        report = json.loads(design_path.read_text(encoding="utf-8"))
    else:
        report = synthesize_reference_cell_design(config)
    package = render_deployment_markdown(config, report)
    write_deployment_package(package, args.markdown_output, args.manifest_output)

    selected = package.selected_design
    print(f"RC-1 deployment package: {package.configuration_id}")
    print(f"  Instruments: {len(package.instruments)}")
    print(f"  Controlled BOM lines: {len(package.bom)}")
    print(f"  Selected duty: {selected['operating']['current_A']:.2f} A at "
          f"{selected['operating']['current_density_mA_cm2']:.0f} mA/cm²")
    print(f"Wrote {args.markdown_output}")
    print(f"Wrote {args.manifest_output}")


if __name__ == "__main__":
    main()
