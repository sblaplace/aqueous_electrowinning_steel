"""
Driver: global (Sobol) sensitivity analysis of the 1D diffusion-layer FE engine.

Answers the program's open question -- *"which experiment / measurement to do
next?"* -- by decomposing FE, V_cell and surface-pH variance across the 10
experimental levers that are actually controllable or measurable in the lab
(see ``models/transport_sensitivity.py``).  This is the fix called for in
``docs/RESEARCH_PROGRAM.md`` (What to Freeze), which flagged the prior Sobol
work as "sensitivity analysis of a fiction" because it ran invented priors
through an unbuilt plant model instead of the transport model.

Writes
------
* ``experiments/data/transport_sensitivity_report.json``  -- S1/ST per output,
  parameter ranking, and ranked "experiment to do next" recommendations.
* ``docs/figures/transport_sensitivity_sobol.png``        -- S1 & ST tornado
  bars for FE and cell voltage.
* ``docs/figures/transport_sensitivity_experiments.png``  -- top experiments
  ranked by total-order index on FE.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from .transport_sensitivity import run_analysis

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "experiments" / "data"
FIG_DIR = ROOT / "docs" / "figures"


def _plot_sobol(analysis, out_path: Path) -> None:
    """Tornado bars of S1 and ST for FE_pct and V_cell_V."""
    param_names = analysis.outputs[0].param_names
    n = len(param_names)
    y = np.arange(n)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    for ax, key, title in [
        (axes[0], "FE_pct", "Faradaic efficiency (FE)\n— total & first-order"),
        (axes[1], "V_cell_V", "Cell voltage (V_cell)\n— total & first-order"),
    ]:
        out = next(o for o in analysis.outputs if o.output == key)
        ax.barh(y - 0.19, out.st, height=0.38, color="#c0392b",
                label="Total-order $S_T$")
        ax.barh(y + 0.19, out.s1, height=0.38, color="#2980b9",
                label="First-order $S_1$")
        ax.set_yticks(y)
        ax.set_yticklabels(param_names)
        ax.invert_yaxis()
        ax.axvline(0.0, color="k", lw=0.8)
        ax.set_xlabel("Sobol index")
        ax.set_xlim(0, 1.0)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Global sensitivity of the 1D diffusion-layer FE engine\n"
        "(Saltelli Sobol; 10 experimental levers)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_recommendations(analysis, out_path: Path) -> None:
    """Ranked 'experiment to do next' list from FE total-order indices."""
    fe = analysis.fe_output
    param_names = fe.param_names
    recs = analysis.recommendations
    top = [r["parameter"] for r in recs]
    st_vals = [fe.st[param_names.index(p)] for p in top]
    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.5 * len(top) + 1.0)))
    y = np.arange(len(top))
    ax.barh(y, st_vals, color="#16a085")
    ax.set_yticks(y)
    ax.set_yticklabels(top)
    ax.invert_yaxis()
    ax.set_xlabel("Total-order Sobol index on FE")
    ax.set_title("Experiments to do next — ranked by $S_T$ on Faradaic efficiency")
    for i, (p, s) in enumerate(zip(top, st_vals)):
        ax.text(s + 0.01, i, f"{s:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(samples: int = 128, seed: int = 0, workers: int = 1) -> dict:
    print("=== Global Sobol sensitivity: 1D diffusion-layer FE engine ===")
    analysis = run_analysis(n_samples=samples, seed=seed, n_workers=workers)
    print(f"  samples/stream = {samples}; "
          f"evaluated = {analysis.n_evaluated}; "
          f"dropped (non-converged) = {analysis.n_failed}")

    summary = analysis.summary_dict()
    for out in analysis.outputs:
        print(f"\n  [{out.output}] mean={out.mean:.3f} var={out.var:.4g}")
        print("    rank  parameter        S1      ST")
        for rank, name in enumerate(out.rank_by_st, start=1):
            i = out.param_names.index(name)
            print(f"    {rank:<5} {name:<14} {out.s1[i]:.3f}  {out.st[i]:.3f}")

    print("\n  Experiments to do next (ranked by ST on FE):")
    for r in analysis.recommendations:
        print(f"    {r['rank']}. {r['lever']}  (S1={r['S1']:.3f}, ST={r['ST']:.3f})")
        print(f"       {r['advice']}")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    json_path = DATA_DIR / "transport_sensitivity_report.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {json_path}")

    sob_path = FIG_DIR / "transport_sensitivity_sobol.png"
    _plot_sobol(analysis, sob_path)
    rec_path = FIG_DIR / "transport_sensitivity_experiments.png"
    _plot_recommendations(analysis, rec_path)
    print(f"Saved {sob_path}\nSaved {rec_path}")
    return summary


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Sobol sensitivity of the 1D diffusion-layer FE engine."
    )
    parser.add_argument("--samples", type=int, default=128,
                        help="Saltelli base samples per stream (default 128)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel processes for evaluation (default 1)")
    args = parser.parse_args()
    main(samples=args.samples, seed=args.seed, workers=args.workers)


if __name__ == "__main__":
    cli()
