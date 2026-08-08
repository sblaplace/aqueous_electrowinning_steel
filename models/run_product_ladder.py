"""Product value ladder: JSON report + regenerated markdown doc + figure.

Answers the program's page-1 decision ("is the product melt-shop feedstock or
steel?") as a *recomputed number* instead of a text argument.  All physics and
cell cost re-derive live from ``cell_architecture`` / ``electrochemistry`` /
``technoeconomic`` / ``thermomechanical``; gate statuses probe the module
tree; only product prices and post-cell unit-ops are anchored constants.

Artifacts (publication tier, per docs/REPO_OUTPUT_POLICY.md):
* ``experiments/data/product_ladder_report.json`` — full result + provenance.
* ``docs/PRODUCT_VALUE_LADDER.md`` — the regenerated decision document.
* ``docs/figures/product_ladder_margin.png`` — margin per m²·yr across the
  price band, per rung.

Run::

    python -m models.run_product_ladder
    aq-steel-product-ladder
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .product_ladder import (
    DEFAULT_DOC_PATH,
    DEFAULT_JSON_PATH,
    RUNGS,
    comparison_table,
    evaluate_ladder,
    provenance,
    write_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "figures"


def make_figure(result, outpath: Path) -> Path:
    """Margin per m² of cell per year, banded by product-price uncertainty."""
    rungs = result.rungs
    labels = [r.rung_id for r in rungs]
    mids = [r.margin_per_m2_yr for r in rungs]
    lows = [r.margin_at_low * r.areal_productivity_t_m2_yr for r in rungs]
    highs = [r.margin_at_high * r.areal_productivity_t_m2_yr for r in rungs]
    err_low = [m - lo for m, lo in zip(mids, lows)]
    err_high = [hi - m for m, hi in zip(mids, highs)]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = [
        "tab:green" if r.verdict.startswith("clears")
        else ("tab:orange" if r.verdict.startswith("marginal")
              else "tab:gray")
        for r in rungs
    ]
    ax.bar(labels, mids, yerr=[err_low, err_high], capsize=5,
           color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("contribution margin  $/(m² cell · yr)")
    ax.set_title(
        "Product value ladder — margin per m²·yr at mid price\n"
        f"(error bars = product-price band only; V_cell "
        f"{result.cell_voltage_V} V, FE {result.faradaic_efficiency}, "
        f"${result.electricity_price_kWh}/kWh)"
    )
    ax.tick_params(axis="x", rotation=18)
    for lb in ax.get_xticklabels():
        lb.set_ha("right")
    ax.grid(axis="y", alpha=0.3)
    note = ("grey = verdict stalls at mid price · band = price uncertainty only "
            "\nall physics re-derived live from cell_architecture/"
            "electrochemistry/thermomechanical")
    ax.text(0.01, 0.98, note, transform=ax.transAxes, fontsize=7.5,
            va="top", color="dimgray")
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json-out", type=Path, default=None,
                        help=f"default: {DEFAULT_JSON_PATH}")
    parser.add_argument("--doc-out", type=Path, default=None,
                        help=f"default: {DEFAULT_DOC_PATH}")
    parser.add_argument("--no-figure", action="store_true",
                        help="skip the margin figure")
    args = parser.parse_args(argv)

    paths = write_artifacts(args.json_out, args.doc_out)
    result = evaluate_ladder()
    print(comparison_table(result))
    print()
    for name, path in paths.items():
        print(f"wrote {name}: {path}")
    if not args.no_figure:
        fig = make_figure(result, FIGURES / "product_ladder_margin.png")
        print(f"wrote figure: {fig}")
    prov = provenance()
    print(f"provenance mode={prov['mode']} sources="
          f"{len(prov['source_hashes'])} files hashed")
    # Headline statistic: the README's '~5×' recovered as a capital-share
    # ratio at commodity price — computed live, no budget constant needed.
    flake_price = RUNGS["flake_feed"].price.mid
    rc = next(r for r in result.rungs
              if r.architecture_id == "rotating_cylinder")
    drum = next(r for r in result.rungs
                if r.architecture_id == "drum_and_strip")
    ratio = ((drum.capital_charge_per_t / flake_price)
             / (rc.capital_charge_per_t / flake_price))
    top = max(result.rungs, key=lambda r: r.price_mid_per_t)
    print(
        f"\nPrice-artefact headline: at ${flake_price:,.0f}/t the drum's "
        f"capital share is {ratio:.1f}× the rotating cylinder's "
        f"({drum.capital_charge_per_t:.2f} vs "
        f"{rc.capital_charge_per_t:.2f} $/t) — the README's '5×'. "
        f"At ${top.price_mid_per_t:,.0f}/t ({top.rung_id}) the same drum's "
        f"share is {100 * top.capital_share_of_price:.1f}% of price. "
        f"See docs/PRODUCT_VALUE_LADDER.md §4."
    )


if __name__ == "__main__":
    main()
