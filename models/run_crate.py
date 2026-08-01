"""Crate structural/environmental driver — how does the unit sit on a site?

Evaluates the autonomous-unit enclosure against site wind, rain, ground and
seismic conditions and prints a stability verdict + mounting spec.
Screening-grade; superseded by a real load/site check at the beachhead stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .crate import (
    Crate,
    CrateConfig,
    CrateSpec,
    EnvironmentalLoads,
    GroundSpec,
    WindLoad,
)

SCENARIOS = {
    "default": dict(),
    "gusty_end_on": dict(wind=dict(direction="end", gust_m_s=55.0)),
    "ballasted_storm": dict(
        wind=dict(direction="end", gust_m_s=55.0), ballast_kg=8000.0),
    "soft_ground": dict(ground=dict(p_allow_kPa=40.0)),
    "flooded_site": dict(
        env=dict(rain_intensity_mm_hr=120.0, sealing_class="industrial"),
        ground=dict(drainable=False, flood_depth_m=0.5)),
}


def _mk(**kw):
    crate = CrateSpec(**kw.pop("crate", {}))
    wind = WindLoad(**kw.pop("wind", {}))
    ground = GroundSpec(**kw.pop("ground", {}))
    env = EnvironmentalLoads(**kw.pop("env", {}))
    return CrateConfig(crate, wind, ground, env, ballast_kg=kw.pop("ballast_kg", 0.0))


def run(silent: bool = False) -> dict:
    model = Crate()
    out = {"scenarios": {}}
    for name, kw in SCENARIOS.items():
        v = model.evaluate(_mk(**kw))
        out["scenarios"][name] = v.to_dict()
        if not silent:
            print(f"\n── {name} ──")
            for k, val in v.to_dict().items():
                print(f"   {k:18s} {val}")
            print(f"   mounting:        {v.mounting_spec}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=str, default=str(
        Path(__file__).resolve().parent.parent / "experiments" / "data"
        / "crate_report.json"))
    args = ap.parse_args()
    out = run()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    main()
