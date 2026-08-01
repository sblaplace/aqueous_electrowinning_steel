"""
Export and CLI for the dark mill 3D model.

Usage (needs LD_LIBRARY_PATH for CadQuery/OCC on NixOS):
    python -m models.cad.dark_mill_cli
    python -m models.cad.dark_mill_cli --export-step docs/cad/dark_mill.step
    python -m models.cad.dark_mill_cli --export-stl docs/cad/dark_mill.stl
    python -m models.cad.dark_mill_cli --query
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dark_mill_config import (
    DarkMillConfig,
    check_transportability,
    check_rainwater,
    check_maintenance_access,
)
from .dark_mill_cad import build_dark_mill, build_dark_mill_assembly


def export_step(cfg: DarkMillConfig, path: str):
    """Export assembly to STEP format."""
    assy = build_dark_mill_assembly(cfg)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    assy.save(str(out))
    print(f"STEP exported: {out}")


def export_stl(cfg: DarkMillConfig, path: str):
    """Export combined geometry to STL."""
    import cadquery as cq
    model = build_dark_mill(cfg)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(model, str(out))
    print(f"STL exported: {out}")


def print_deployment_report(cfg: DarkMillConfig):
    """Print a full deployment assessment."""
    transport = check_transportability(cfg)
    rain = check_rainwater(cfg)
    access = check_maintenance_access(cfg)

    print("=" * 70)
    print("DARK MILL DEPLOYMENT ASSESSMENT")
    print("=" * 70)
    print()
    print(f"UNIT DIMENSIONS: {cfg.enclosure_length:.0f} × {cfg.enclosure_width:.0f} × {cfg.enclosure_height:.0f} mm")
    print(f"  ({cfg.enclosure_length/1000:.1f}m × {cfg.enclosure_width/1000:.1f}m × {cfg.enclosure_height/1000:.1f}m)")
    print()
    print("TRANSPORT:")
    print(f"  Fits on trailer:      {'YES' if transport['fits_trailer'] else 'NO'}")
    print(f"  Width OK:             {'YES' if transport['fits_width'] else 'NO'} ({cfg.enclosure_width:.0f} vs {transport['trailer_limit_mm'][1]:.0f} mm)")
    print(f"  Height OK:            {'YES' if transport['fits_height'] else 'NO'} ({cfg.enclosure_height:.0f} vs {transport['trailer_limit_mm'][2]:.0f} mm)")
    print(f"  20ft container:       {'YES' if transport['fits_length_20ft'] else 'NO — needs 40ft'}")
    print(f"  Weight (loaded):      ~{transport['weight_estimate_kg']:.0f} kg ({transport['weight_estimate_kg']/1000:.1f} t)")
    print(f"  Forklift pockets:     YES (2 × {cfg.forklift_pocket_length:.0f}mm)")
    print("  Crane lift points:    4 corners")
    print()
    print("RAINWATER:")
    print(f"  Roof slope:           {rain['roof_slope_deg']}° → {rain['roof_height_diff_mm']}mm drop")
    print(f"  Runoff direction:     {rain['runoff_direction']}")
    print(f"  Low side:             {rain['low_side']}")
    print(f"  Recommendation:       {rain['recommendation']}")
    print(f"  Door risk:            {rain['door_risk']}")
    print("  Puddle risk zones:")
    for zone in rain['puddle_risk_zones']:
        print(f"    - {zone}")
    print()
    print("MAINTENANCE ACCESS:")
    print(f"  Center aisle:         {access['center_aisle_width_mm']:.0f} mm ({'OK' if access['aisle_sufficient'] else 'TOO NARROW — min 800mm'})")
    print(f"  Stack access:         {access['stack_access']}")
    print(f"  Tank access:          {access['tank_access']}")
    print(f"  Electrical access:    {access['electrical_access']}")
    print(f"  Door:                 {access['door_type']}")
    print()
    print("INTERNAL LAYOUT:")
    print(f"  Cell stacks:          {cfg.n_stacks} stacks × {cfg.cells_per_stack} cells")
    print(f"  Stacks per row:       {cfg.n_stacks_per_row}")
    print(f"  Stack rows:           {cfg.n_stack_rows}")
    print(f"  Stack height:         {cfg.stack_height:.0f} mm ({cfg.stack_height/1000:.1f}m)")
    print(f"  Feed tank:            {cfg.feed_tank_length:.0f} × {cfg.feed_tank_width:.0f} × {cfg.feed_tank_height:.0f} mm")
    print(f"  Product tank:         {cfg.product_tank_length:.0f} × {cfg.product_tank_width:.0f} × {cfg.product_tank_height:.0f} mm")
    print(f"  Rectifier cabinet:    {cfg.rectifier_length:.0f} × {cfg.rectifier_width:.0f} × {cfg.rectifier_height:.0f} mm")
    print(f"  Heat exchanger:       {cfg.hx_length:.0f} × {cfg.hx_width:.0f} × {cfg.hx_height:.0f} mm")
    print(f"  Control cabinet:      {cfg.control_length:.0f} × {cfg.control_width:.0f} × {cfg.control_height:.0f} mm")
    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Dark Mill 3D model")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON config file (overrides defaults)")
    parser.add_argument("--export-step", type=str, default=None,
                        help="Export to STEP file")
    parser.add_argument("--export-stl", type=str, default=None,
                        help="Export to STL file")
    parser.add_argument("--query", action="store_true",
                        help="Print deployment assessment")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    args = parser.parse_args()

    # Load config
    if args.config:
        with open(args.config) as f:
            data = json.load(f)
        cfg = DarkMillConfig(**data)
    else:
        cfg = DarkMillConfig()

    # Export
    if args.export_step:
        export_step(cfg, args.export_step)
    if args.export_stl:
        export_stl(cfg, args.export_stl)

    # Query
    if args.query or args.json:
        if args.json:
            result = {
                "config": cfg.to_dict(),
                "transport": check_transportability(cfg),
                "rainwater": check_rainwater(cfg),
                "maintenance": check_maintenance_access(cfg),
            }
            print(json.dumps(result, indent=2))
        else:
            print_deployment_report(cfg)


if __name__ == "__main__":
    main()
