"""
CLI runner for the whole-system twin — end-to-end site assessment.

Evaluates a named site from climatology + soil to cell FE/V and prints
one decision-grade verdict with per-layer credibilities.

Usage:
    python -m models.run_system_twin                              # all sites
    python -m models.run_system_twin --site pickle_liquor_us_midwest
    python -m models.run_system_twin --site wind_farm_ore --ballast 5000
    python -m models.run_system_twin --json
    python -m models.run_system_twin --compare
    aq-steel-system-twin --site pickle_liquor_us_midwest

Outputs:
  - console summary with credibility vector, stability verdict, ballast/mounting,
    environmental safe-state action, and GO/NO-GO
  - experiments/data/system_twin_report.json  (all sites + this run)
  - docs/figures/system_twin_*.png  (stability, ballast, credibility, GO matrix)

All numbers screening-grade (per-layer L0) until real data/load tests validate them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

from .system_twin import (
    EXAMPLE_SITES,
    LEGACY_THREE,
    evaluate_system_twin,
    evaluate_all_sites,
    CredibilityVector,
)
from .dark_mill import comparison_table


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


def _save_json_report(reports: Dict[str, Any], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Build minimal serialisable payload
    payload = {
        "credibility_note": "All numbers screening-grade per-layer L0 until validated",
        "legacy_three": LEGACY_THREE,
        "sites": {k: v.to_dict() for k, v in reports.items()},
    }
    out_path.write_text(json.dumps(payload, indent=2, default=_json_default))
    print(f"Report saved: {out_path}")


def _make_figures(reports: Dict, fig_dir: Path):
    """Generate screening figures for the system twin."""
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib not available, skipping figures: {e}")
        return

    fig_dir.mkdir(parents=True, exist_ok=True)

    keys = list(reports.keys())
    labels = [reports[k].site.name[:32] for k in keys]

    # 1. FS overturn / bearing / sliding
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fs_over = [reports[k].crate_verdict.fs_overturn for k in keys]
    fs_bear = [reports[k].crate_verdict.fs_bearing for k in keys]
    fs_slide = [reports[k].crate_verdict.fs_slide for k in keys]
    x = np.arange(len(keys))
    w = 0.22
    ax.bar(x - w, fs_over, w, label="FS overturn broadside")
    ax.bar(x, fs_bear, w, label="FS bearing")
    ax.bar(x + w, fs_slide, w, label="FS sliding")
    ax.axhline(1.5, color="r", linestyle="--", label="target 1.5")
    ax.axhline(1.0, color="k", linestyle=":", label="bearing min 1.0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Factor of safety")
    ax.set_title("System twin: crate stability FS per site (screening L0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "system_twin_stability.png", dpi=150)
    plt.close(fig)
    print(f"Figure: {fig_dir / 'system_twin_stability.png'}")

    # 2. Ballast required
    fig, ax = plt.subplots(figsize=(8, 4))
    ballast = [reports[k].required_ballast_kg for k in keys]
    ax.bar(labels, ballast, color="steelblue")
    ax.set_ylabel("min ballast kg to reach FS 1.5")
    ax.set_title("System twin: required ballast per site (end-on worst)")
    ax.tick_params(axis='x', rotation=20)
    fig.tight_layout()
    fig.savefig(fig_dir / "system_twin_ballast.png", dpi=150)
    plt.close(fig)
    print(f"Figure: {fig_dir / 'system_twin_ballast.png'}")

    # 3. Credibility vector (all L0 but shows structure)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    proc = [reports[k].credibility.process_level for k in keys]
    crate = [reports[k].credibility.crate_level for k in keys]
    site_c = [reports[k].credibility.site_level for k in keys]
    ax.scatter(proc, crate, s=120, label="crate vs process")
    for i, k in enumerate(keys):
        ax.text(proc[i] + 0.05, crate[i] + 0.05, labels[i][:12], fontsize=8)
    ax.set_xlabel("process credibility L")
    ax.set_ylabel("crate credibility L")
    ax.set_title(f"System twin credibility vector (all L0 screening) — {len(keys)} sites")
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.grid(True, alpha=0.3)
    # second subplot for site?
    fig.tight_layout()
    fig.savefig(fig_dir / "system_twin_credibility.png", dpi=150)
    plt.close(fig)
    print(f"Figure: {fig_dir / 'system_twin_credibility.png'}")

    # 4. GO/NO-GO matrix: overall GO vs crate stable
    fig, ax = plt.subplots(figsize=(9, 4))
    overall = [1 if reports[k].overall_go else 0 for k in keys]
    stable = [1 if reports[k].combined_stable else 0 for k in keys]
    x = np.arange(len(keys))
    ax.bar(x - 0.15, overall, 0.3, label="overall GO")
    ax.bar(x + 0.15, stable, 0.3, label="crate stable/stabilisable")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("GO (1) / NO-GO (0)")
    ax.set_title("System twin: GO/NO-GO per site")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "system_twin_go_matrix.png", dpi=150)
    plt.close(fig)
    print(f"Figure: {fig_dir / 'system_twin_go_matrix.png'}")

    # 5. Environmental safe-state actions per site
    fig, ax = plt.subplots(figsize=(9, 3.5))
    actions = [reports[k].environmental_safe_state for k in keys]
    # map to numeric for bar: 0 normal, 1 storm
    is_storm = [0 if a == "normal_operation" else 1 for a in actions]
    colors = ["green" if v == 0 else "orange" for v in is_storm]
    ax.bar(labels, is_storm, color=colors)
    ax.set_ylabel("storm mode? (1=yes)")
    ax.set_title("System twin: environmental safe-state per site")
    ax.tick_params(axis='x', rotation=20)
    # annotate action text
    for i, act in enumerate(actions):
        ax.text(i, is_storm[i] + 0.05, act[:20], rotation=45, ha="left", fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "system_twin_env_safestate.png", dpi=150)
    plt.close(fig)
    print(f"Figure: {fig_dir / 'system_twin_env_safestate.png'}")


def main():
    parser = argparse.ArgumentParser(description="Whole-system twin end-to-end site assessment")
    parser.add_argument("--site", type=str, default=None,
                        help=f"Site key. Options: {list(EXAMPLE_SITES.keys())}. "
                             f"Legacy three: {LEGACY_THREE}")
    parser.add_argument("--ballast", type=float, default=0.0,
                        help="Pre-applied ballast kg (default 0 — assess bare unit)")
    parser.add_argument("--json", action="store_true",
                        help="Print JSON to stdout instead of human summary")
    parser.add_argument("--compare", action="store_true",
                        help="Print comparison table (dark_mill style) + system twin summary")
    parser.add_argument("--output", type=str, default=str(
        Path(__file__).resolve().parent.parent / "experiments" / "data" / "system_twin_report.json"
    ))
    parser.add_argument("--fig-dir", type=str, default=str(
        Path(__file__).resolve().parent.parent / "docs" / "figures"
    ))
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip figure generation")
    args = parser.parse_args()

    if args.site:
        # Single site end-to-end
        report = evaluate_system_twin(args.site, ballast_kg=args.ballast, credibility=CredibilityVector.screening())
        if args.json:
            print(json.dumps(report.to_dict(), indent=2, default=_json_default))
        else:
            print(report.summary())
        # Save single-site + all report? Save all for persistence
        all_reports = evaluate_all_sites()
        _save_json_report(all_reports, Path(args.output))
        if not args.no_figures:
            _make_figures(all_reports, Path(args.fig_dir))
    elif args.compare:
        # Comparison table + system verdicts
        all_reports = evaluate_all_sites()
        # Dark mill comparison via site_report
        dark_reports = {k: v.site_report for k, v in all_reports.items()}
        print(comparison_table(dark_reports))
        print("\n" + "="*72)
        print("SYSTEM TWIN — per-layer credibility + stability + safe-state")
        print("="*72)
        for k, r in all_reports.items():
            print(f"{r.site.name:42s} cred={r.credibility.label():30s} "
                  f"FS_over={r.crate_verdict.fs_overturn:.2f} "
                  f"FS_end={r.crate_verdict_end_on.fs_overturn:.2f} "
                  f"ballast={r.required_ballast_kg:.0f}kg "
                  f"stable={r.combined_stable} GO={r.overall_go} "
                  f"env={r.environmental_safe_state}")
        _save_json_report(all_reports, Path(args.output))
        if not args.no_figures:
            _make_figures(all_reports, Path(args.fig_dir))
    else:
        # All sites
        all_reports = evaluate_all_sites()
        for k, r in all_reports.items():
            print(r.summary())
            print("\n")
        _save_json_report(all_reports, Path(args.output))
        if not args.no_figures:
            _make_figures(all_reports, Path(args.fig_dir))


if __name__ == "__main__":
    main()
