"""Cell architecture screen: figures + JSON report.

Answers the question ``docs/RESEARCH_PROGRAM.md`` calls architectural and
gating: *is there a cell that combines filter-press current densities with
continuous solid harvesting?*  And its economic corollary, kill criterion #3:
*what may a continuously strippable cell cost per m²?*

Run::

    python -m models.run_cell_architecture
    aq-steel-architecture
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .cell_architecture import (
    ARCHITECTURES,
    IRON_PRODUCT_VALUE_PER_T,
    ZINC_TANKHOUSE,
    OperatingConditions,
    capital_recovery_factor,
    compare_architectures,
    comparison_table,
    concentration_sweep,
    kill_criterion_assessment,
    max_affordable_cost_per_m2,
    model_scope,
    velocity_sweep,
    zinc_tankhouse_productivity,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments" / "data"
FIGURES = ROOT / "docs" / "figures"

# Colour per architecture, stable across all figures.
COLORS = {
    "plate_and_frame": "tab:blue",
    "rotating_cylinder": "tab:orange",
    "drum_and_strip": "tab:green",
    "moving_belt": "tab:red",
    "fluidized_bed": "tab:purple",
}
SHORT = {
    "plate_and_frame": "Plate &\nframe",
    "rotating_cylinder": "Rotating\ncylinder",
    "drum_and_strip": "Drum &\nstrip",
    "moving_belt": "Moving\nbelt",
    "fluidized_bed": "Fluidized\nbed",
}
HATCH = {"commercial": "", "pilot": "//", "lab": "xx", "concept": ".."}


def _fig_comparison(results, conditions: OperatingConditions) -> Path:
    """Four-panel architecture comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), constrained_layout=True)
    ids = [r.architecture_id for r in results]
    labels = [SHORT[i] for i in ids]
    colors = [COLORS[i] for i in ids]
    hatches = [HATCH[r.evidence_level] for r in results]

    # (0,0) Areal productivity vs the zinc benchmark
    ax = axes[0, 0]
    prod = [r.areal_productivity_t_m2_yr for r in results]
    bars = ax.bar(labels, prod, color=colors, edgecolor="black")
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
    zinc = zinc_tankhouse_productivity()
    ax.axhline(zinc, color="black", linestyle="--", linewidth=1.4,
               label=f"Zn tankhouse equiv. ({zinc:.1f})")
    ax.axhline(5 * zinc, color="crimson", linestyle=":", linewidth=1.8,
               label=f"5x target ({5*zinc:.1f})")
    ax.set_ylabel("Areal productivity (t Fe/(m²·yr))")
    ax.set_title("Productivity per m² of costed electrode area")
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    for b, v in zip(bars, prod):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.08, f"{v:.1f}",
                ha="center", fontsize=8)

    # (0,1) Installed cost vs affordability threshold
    ax = axes[0, 1]
    x = np.arange(len(results))
    cost = [r.installed_cost_per_m2 for r in results]
    thresh = [
        max_affordable_cost_per_m2(
            r.areal_productivity_t_m2_yr, 60.0,
            conditions.discount_rate, conditions.plant_lifetime_yr,
        )
        for r in results
    ]
    ax.bar(x - 0.2, cost, 0.4, color=colors, edgecolor="black")
    ax.bar(x + 0.2, thresh, 0.4, color="lightgray", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"\$/m² installed")
    ax.set_title("Kill criterion #3: cost vs affordability")
    ax.set_yscale("log")
    # Build the legend by hand: the left bars are per-architecture colours,
    # so a colour-keyed legend entry would be misleading.
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="dimgray", edgecolor="black",
                  label="Installed cost (coloured by architecture)"),
            Patch(facecolor="lightgray", edgecolor="black",
                  label=r"Max affordable @ \$60/t Fe budget"),
        ],
        fontsize=8,
    )

    # (1,0) Capital charge per tonne
    ax = axes[1, 0]
    charge = [r.capital_charge_per_t_fe for r in results]
    bars = ax.bar(labels, charge, color=colors, edgecolor="black")
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
    ax.axhline(60.0, color="crimson", linestyle="--", linewidth=1.5,
               label="$60/t budget")
    ax.set_ylabel("Cell capital charge ($/t Fe)")
    ax.set_title("Annualized cell capital per tonne of iron")
    ax.legend(fontsize=8)
    for b, v in zip(bars, charge):
        ax.text(b.get_x() + b.get_width() / 2, v, f"${v:,.0f}",
                ha="center", va="bottom", fontsize=8)

    # (1,1) Operating current density, active vs footprint
    ax = axes[1, 1]
    j_act = [r.j_operating_A_m2 / 10.0 for r in results]
    j_ftp = [r.j_installed_A_m2 / 10.0 for r in results]
    ax.bar(x - 0.2, j_act, 0.4, label="Active area", color=colors,
           edgecolor="black")
    ax.bar(x + 0.2, j_ftp, 0.4, label="Costed footprint", color="lightgray",
           edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Current density (mA/cm²)")
    ax.set_title("Operating current density")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    for xi, r in zip(x, results):
        ax.text(xi, max(j_act[int(xi)], j_ftp[int(xi)]) * 1.3,
                r.limited_by.replace("_", "\n"), ha="center", fontsize=6.5)

    fig.suptitle(
        "Cell architecture screen — hatching marks evidence level "
        "(solid=commercial, //=pilot, ..=concept). Screening model, no iron wet-lab data.",
        fontweight="bold", fontsize=10,
    )
    path = FIGURES / "cell_architecture_comparison.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _fig_tradeoff(results) -> Path:
    """Productivity vs cost, with iso-capital-charge contours."""
    fig, ax = plt.subplots(figsize=(9.5, 7), constrained_layout=True)

    prod = np.array([r.areal_productivity_t_m2_yr for r in results])

    # Iso-lines of constant $/t Fe capital charge.
    crf = capital_recovery_factor()
    p_grid = np.logspace(np.log10(prod.min() * 0.4), np.log10(prod.max() * 2.2), 200)
    for budget, style in [(10, ":"), (30, "-."), (60, "--"), (150, "-")]:
        ax.plot(p_grid, budget * p_grid / crf, style, color="gray",
                linewidth=1.0, alpha=0.75)
        ax.annotate(f"${budget}/t", xy=(p_grid[-1], budget * p_grid[-1] / crf),
                    fontsize=7.5, color="gray", ha="right", va="bottom")

    for r in results:
        ax.scatter(r.areal_productivity_t_m2_yr, r.installed_cost_per_m2,
                   s=230, color=COLORS[r.architecture_id],
                   edgecolor="black", zorder=5,
                   marker="o" if r.harvest_mode == "continuous" else "s")
        ax.annotate(
            f"{SHORT[r.architecture_id]}".replace("\n", " "),
            xy=(r.areal_productivity_t_m2_yr, r.installed_cost_per_m2),
            xytext=(9, 7), textcoords="offset points", fontsize=9,
        )

    zinc = zinc_tankhouse_productivity()
    ax.axvline(zinc, color="black", linestyle="--", linewidth=1.2)
    ax.axvline(5 * zinc, color="crimson", linestyle=":", linewidth=1.6)
    ax.text(zinc, ax.get_ylim()[1] * 0.55, " Zn tankhouse", rotation=90,
            fontsize=8, va="top")
    ax.text(5 * zinc, ax.get_ylim()[1] * 0.55, " 5x target", rotation=90,
            fontsize=8, va="top", color="crimson")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Areal productivity (t Fe/(m²·yr))")
    ax.set_ylabel("Installed cell cost ($/m²)")
    ax.set_title(
        "Cost–productivity trade-off\n"
        "circles = continuous harvesting, squares = batch; "
        "below a gray line = within that capital budget",
        fontsize=10,
    )
    ax.grid(alpha=0.3, which="both")
    path = FIGURES / "cell_architecture_tradeoff.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _fig_sweeps(conditions: OperatingConditions) -> Path:
    """Velocity and concentration sensitivity."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    velocities = np.logspace(-2, 0.8, 45)
    concentrations = np.linspace(0.1, 3.0, 40)

    ax = axes[0]
    for aid, spec in ARCHITECTURES.items():
        s = velocity_sweep(spec, velocities, conditions)
        ax.plot(s["velocity_m_s"], np.array(s["transport_limit_A_m2"]) / 10.0,
                label=SHORT[aid].replace("\n", " "), color=COLORS[aid], linewidth=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Characteristic velocity (m/s)")
    ax.set_ylabel("Transport limit (mA/cm²)")
    ax.set_title("Mass-transport ceiling vs velocity")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    for aid, spec in ARCHITECTURES.items():
        s = velocity_sweep(spec, velocities, conditions)
        ax.plot(s["velocity_m_s"], s["areal_productivity_t_m2_yr"],
                label=SHORT[aid].replace("\n", " "), color=COLORS[aid], linewidth=2)
    ax.axhline(zinc_tankhouse_productivity(), color="black", linestyle="--",
               linewidth=1.2, label="Zn benchmark")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Characteristic velocity (m/s)")
    ax.set_ylabel("Areal productivity (t/(m²·yr))")
    ax.set_title("Productivity vs velocity\n(flat = practical ceiling reached)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    for aid, spec in ARCHITECTURES.items():
        s = concentration_sweep(spec, concentrations, conditions)
        ax.plot(s["fe_conc_M"], s["capital_charge_per_t_fe"],
                label=SHORT[aid].replace("\n", " "), color=COLORS[aid], linewidth=2)
    ax.axhline(60.0, color="crimson", linestyle="--", linewidth=1.4,
               label="$60/t budget")
    ax.set_yscale("log")
    ax.set_xlabel("Bulk [Fe²⁺] (mol/L)")
    ax.set_ylabel("Capital charge ($/t Fe)")
    ax.set_title("Capital charge vs iron concentration")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("Cell architecture sensitivity — screening model",
                 fontweight="bold")
    path = FIGURES / "cell_architecture_sweeps.png"
    fig.savefig(path, dpi=165)
    plt.close(fig)
    return path


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    conditions = OperatingConditions()
    results = compare_architectures(conditions)
    assessment = kill_criterion_assessment(conditions)

    print("=" * 78)
    print("CELL ARCHITECTURE SCREEN")
    print("=" * 78)
    print()
    print(comparison_table(results))
    print()

    zinc = zinc_tankhouse_productivity()
    print(f"Zinc tankhouse benchmark (running iron chemistry): {zinc:.2f} t/(m²·yr)")
    print(f"Program target (~5x zinc):                         {5*zinc:.2f} t/(m²·yr)")
    print()
    print("Kill criterion #3 — affordability at a $60/t Fe capital budget:")
    for v in assessment["architectures"]:
        flag = "PASS" if v["passes"] else "FAIL"
        print(
            f"  [{flag}] {v['architecture_id']:<20} "
            f"{v['productivity_vs_zinc']:>6.2f}x zinc   "
            f"${v['installed_cost_per_m2']:>7,.0f}/m² vs "
            f"${v['max_affordable_cost_per_m2']:>9,.0f}/m² affordable"
        )
    print()

    fig1 = _fig_comparison(results, conditions)
    fig2 = _fig_tradeoff(results)
    fig3 = _fig_sweeps(conditions)

    report = {
        "provenance": (
            "Screening output from models/cell_architecture.py. No wet-lab "
            "iron data. Mass-transfer correlations are literature values from "
            "other chemistries; costs are engineering estimates, not quotes."
        ),
        "model_scope": model_scope(),
        "conditions": {
            "fe_conc_M": conditions.fe_conc_M,
            "temperature_C": conditions.temperature_C,
            "faradaic_efficiency": conditions.faradaic_efficiency,
            "installed_cost_factor": conditions.installed_cost_factor,
            "discount_rate": conditions.discount_rate,
            "plant_lifetime_yr": conditions.plant_lifetime_yr,
            "schmidt_number": round(conditions.schmidt, 1),
            "crf": round(conditions.crf, 5),
        },
        "benchmarks": {
            "zinc_tankhouse": ZINC_TANKHOUSE,
            "zinc_iron_equivalent_productivity_t_m2_yr": round(zinc, 2),
            "program_target_t_m2_yr": round(5 * zinc, 2),
            "iron_product_value_per_t": IRON_PRODUCT_VALUE_PER_T,
        },
        "architectures": [r.to_dict() for r in results],
        "kill_criterion_3": assessment,
        "findings": _findings(results, assessment, zinc),
        "figures": [
            f"docs/figures/{fig1.name}",
            f"docs/figures/{fig2.name}",
            f"docs/figures/{fig3.name}",
        ],
    }

    out = DATA / "cell_architecture_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("Findings:")
    for f in report["findings"]:
        print(f"  - {f}")
    print()
    for p in (fig1, fig2, fig3):
        print(f"Wrote {p.relative_to(ROOT)}")
    print(f"Wrote {out.relative_to(ROOT)}")


def _findings(results, assessment, zinc: float) -> list:
    """Derive the narrative conclusions from the computed numbers."""
    best = results[0]
    continuous = [r for r in results if r.harvest_mode == "continuous"]
    batch = [r for r in results if r.harvest_mode == "batch"]
    findings = []

    findings.append(
        f"Lowest cell capital charge: {best.name} at "
        f"${best.capital_charge_per_t_fe:,.2f}/t Fe "
        f"({best.areal_productivity_t_m2_yr:.1f} t/(m²·yr), "
        f"{best.areal_productivity_t_m2_yr/zinc:.1f}x the zinc benchmark)."
    )

    if continuous and batch:
        c = min(r.capital_charge_per_t_fe for r in continuous)
        b = min(r.capital_charge_per_t_fe for r in batch)
        if c < b:
            findings.append(
                f"Continuous harvesting beats batch on capital charge "
                f"(${c:,.2f} vs ${b:,.2f}/t Fe) despite costing more per m², "
                f"because batch duty cycle falls as plating rate rises."
            )

    meets = [
        r for r in results
        if r.areal_productivity_t_m2_yr >= 5 * zinc
    ]
    if meets:
        findings.append(
            "Architectures meeting the ~5x zinc areal-productivity target: "
            + ", ".join(r.architecture_id for r in meets) + "."
        )
    else:
        findings.append(
            "No architecture meets the ~5x zinc target at these conditions — "
            "the program's own pivot trigger."
        )

    plate = next((r for r in results if r.architecture_id == "plate_and_frame"), None)
    if plate:
        findings.append(
            f"The plate-and-frame baseline currently assumed in "
            f"technoeconomic.py reaches only "
            f"{plate.areal_productivity_t_m2_yr/zinc:.2f}x the zinc benchmark "
            f"({plate.capacity_factor:.2f} capacity factor after harvest "
            f"downtime), which is the case for studying alternatives."
        )

    failing = [v for v in assessment["architectures"] if not v["passes"]]
    if failing:
        findings.append(
            "Exceeding the $60/t capital budget: "
            + ", ".join(v["architecture_id"] for v in failing) + "."
        )
    else:
        findings.append(
            "Every architecture screened falls within the $60/t Fe cell "
            "capital budget; cell cost is therefore not the binding "
            "constraint at these assumed productivities."
        )

    findings.append(
        "Highest-value unknown: whether iron peels cleanly from a drum "
        "cathode. Copper foil relies on a passive TiO₂ release layer; iron "
        "adhesion on titanium is uncharacterised experimentally and gates the "
        "drum-and-strip route entirely. models/adhesion_peel.py now screens "
        "that question and specifies the coupon test that settles it."
    )
    return findings


if __name__ == "__main__":
    main()
