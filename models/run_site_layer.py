"""
Site layer (L3) design runner — will the crate sink / flood / get wind-exposed?

Pairs with the D4 crate structural verdict (`aq-steel-crate`): that model checks
overturning / bearing / sliding / ingress for the unit; this runner turns those
into the **site-layer design** — foundation & ballast, flood & drainage, terrain
wind exposure, and site layout with feedstock / power / water / product access
(`models/site_layer.py`).

Usage:
    python -m models.run_site_layer                       # all example sites
    python -m models.run_site_layer --site copperas_tio2_plant
    python -m models.run_site_layer --json

Outputs:
  - console summary per site: sink / flood / wind-exposed verdict + design
  - experiments/data/site_layer_report.json  (all sites + design blocks)
  - docs/figures/site_layer_design.png       (sink/flood/wind matrix per site)

Screening-grade (L0) — every number superseded by a real soil / hydraulic /
wind survey at the named-beachhead stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .site_layer import SiteLayer
from .dark_mill import EXAMPLE_SITES, evaluate_crate_for_site


def _json_default(obj):
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _design_site(site, crate_verdict=None) -> dict:
    """Full L3 design block for one site, paired with its D4 crate verdict."""
    layer = SiteLayer()
    if crate_verdict is None:
        crate_verdict = evaluate_crate_for_site(site)
    verdict = layer.evaluate(site, crate_verdict=crate_verdict)
    return {
        "site": site.name,
        "site_key": _site_key(site),
        "crate_verdict": {
            "fs_overturn": round(crate_verdict.fs_overturn, 2),
            "net_bearing_kPa": round(crate_verdict.net_bearing_kPa, 2),
            "fs_slide": round(crate_verdict.fs_slide, 2),
            "ingress_risk": crate_verdict.ingress_risk,
            "mounting_spec": crate_verdict.mounting_spec,
        },
        "site_layer": verdict.flat(),
        "summary": verdict.summary(),
    }


def _site_key(site) -> str:
    for k, v in EXAMPLE_SITES.items():
        if v is site:
            return k
    return site.name.lower().replace(" ", "_")


def _build_payload(designs: Dict[str, dict]) -> dict:
    return {
        "note": ("Screening-grade L3 site design (L0) — reuses dark_mill "
                 "SiteDefinition + D4 crate verdict; no soil/hydraulic/wind "
                 "survey yet. Superseded at the named-beachhead stage."),
        "sites": designs,
    }


def _make_figure(designs: Dict[str, dict], fig_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"matplotlib not available, skipping figure: {e}")
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    keys = list(designs.keys())
    labels = [designs[k]["site"][:24] for k in keys]
    sink = [1 if designs[k]["site_layer"]["will_sink"] else 0 for k in keys]
    flood = [1 if designs[k]["site_layer"]["will_flood"] else 0 for k in keys]
    wind = [1 if designs[k]["site_layer"]["will_get_wind_exposed"] else 0 for k in keys]
    x = np.arange(len(keys))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w, sink, w, label="sink", color="#c0392b")
    ax.bar(x, flood, w, label="flood", color="#2980b9")
    ax.bar(x + w, wind, w, label="wind-exposed", color="#f39c12")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["no", "yes"])
    ax.set_ylim(-0.15, 1.15)
    ax.set_ylabel("Risk present")
    ax.set_title("Site layer (L3): sink / flood / wind-exposure screening per site")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "site_layer_design.png", dpi=150)
    plt.close(fig)
    print(f"Figure: {fig_dir / 'site_layer_design.png'}")


def run(site_keys: Optional[list] = None, silent: bool = False) -> dict:
    keys = site_keys or list(EXAMPLE_SITES.keys())
    designs: Dict[str, dict] = {}
    for k in keys:
        site = EXAMPLE_SITES[k]
        cv = evaluate_crate_for_site(site)
        d = _design_site(site, cv)
        designs[k] = d
        if not silent:
            print(f"\n----- {k} -----")
            print(d["summary"])
    return _build_payload(designs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", type=str, default=None,
                    help="one EXAMPLE_SITES key; default = all sites")
    ap.add_argument("--json", action="store_true", help="also write the JSON report")
    ap.add_argument("--output", type=str, default=str(
        Path(__file__).resolve().parent.parent / "experiments" / "data"
        / "site_layer_report.json"))
    args = ap.parse_args()

    payload = run([args.site] if args.site else None)
    for k, d in payload["sites"].items():
        print(f"\n{'_'*70}\nSITE: {d['site']}  ({k})")
        print(d["summary"])

    _make_figure(payload["sites"], Path(__file__).resolve().parent.parent
                 / "docs" / "figures")

    if args.json:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=_json_default))
        print(f"\nReport saved: {out}")


if __name__ == "__main__":
    main()
