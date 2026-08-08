"""Ladder × TEA: JSON report + regenerated markdown doc + figure.

Connects the product value ladder's contribution-margin screen to the full
plant cost stack in ``technoeconomic.py`` and answers, as a recomputed
number: **does the rung ranking survive complete costing?**  All physics
and cost lines re-derive live (cell_architecture / electrochemistry /
technoeconomic / thermomechanical / product_ladder); no constants are
introduced beyond the scenario knob (nameplate capacity).

Artifacts (publication tier, per docs/REPO_OUTPUT_POLICY.md):
* ``experiments/data/ladder_tea_report.json`` — full result + provenance.
* ``docs/LADDER_TEA.md`` — the regenerated decision document.
* ``docs/figures/ladder_tea_margin.png`` — ladder vs full-TEA margin per
  rung with the product-price band.

Run::

    python -m models.run_ladder_tea
    aq-steel-ladder-tea
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .ladder_tea import (
    DEFAULT_DOC_PATH,
    DEFAULT_FIGURE_PATH,
    DEFAULT_JSON_PATH,
    DEFAULT_PLANT_CAPACITY_T_YR,
    comparison_table,
    evaluate_ladder_tea,
    gap_table,
    write_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]


def make_figure(result, outpath: Path) -> Path:
    """Ladder contribution margin vs full-TEA margin per rung ($/t)."""
    rungs = result.rungs
    labels = [r.rung_id for r in rungs]
    x = np.arange(len(rungs))
    width = 0.38

    ladder = [r.ladder_margin_per_t for r in rungs]
    tea = [r.margin_per_t for r in rungs]
    err_low = [max(0.0, r.margin_per_t - r.margin_at_low) for r in rungs]
    err_high = [max(0.0, r.margin_at_high - r.margin_per_t) for r in rungs]

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.bar(x - width / 2, ladder, width, label="ladder contribution margin",
           color="tab:blue", alpha=0.55, edgecolor="black", linewidth=0.5)
    colors = [
        "tab:green" if r.verdict == "clears"
        else ("tab:orange" if r.verdict == "marginal" else "tab:gray")
        for r in rungs
    ]
    ax.bar(x + width / 2, tea, width, yerr=[err_low, err_high], capsize=4,
           label="full-TEA margin (band = price uncertainty)",
           color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", rotation=18)
    for lb in ax.get_xticklabels():
        lb.set_ha("right")
    ax.set_ylabel("margin  $/t product")
    ax.set_title(
        "Ladder × TEA — does the ranking survive full plant costing?\n"
        f"({result.capacity_t_yr / 1000:,.0f} kt/yr · V_cell "
        f"{result.cell_voltage_V} V · FE {result.faradaic_efficiency} · "
        f"${result.electricity_price_kWh}/kWh · "
        f"pairwise flips: {result.n_pairwise_flips})"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_yscale("symlog", linthresh=100.0)
    note = ("full-TEA excludes nothing: feedstock, anode, overhead, maintenance, "
            "insurance, labour, ore-side plant —\nthe ladder's blind spot, "
            "itemised in docs/LADDER_TEA.md §4 · grey = stalls at mid price")
    fig.text(0.01, 0.005, note, fontsize=7.5, va="bottom", color="0.25")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


def main() -> None:  # pragma: no cover - CLI wrapper
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--capacity", type=float, default=DEFAULT_PLANT_CAPACITY_T_YR)
    p.add_argument("--elec-price", type=float, default=None)
    args = p.parse_args()

    result = evaluate_ladder_tea(
        capacity_t_yr=args.capacity,
        electricity_price_kWh=args.elec_price,
    )
    print(comparison_table(result))
    print()
    print(gap_table(result))
    print()
    paths = write_artifacts(result)
    fig = make_figure(result, ROOT / DEFAULT_FIGURE_PATH.relative_to(ROOT))
    paths["figure"] = fig
    print(f"ranking preserved: {result.ranking_preserved} "
          f"(pairwise flips: {result.n_pairwise_flips})")
    for k, v in paths.items():
        print(f"[{k}] {v}")
    assert DEFAULT_JSON_PATH.exists() and DEFAULT_DOC_PATH.exists()


if __name__ == "__main__":  # pragma: no cover
    main()
