"""
Process flow diagram generator for aqueous electrowinning.

Creates a schematic block-flow diagram (BFD) of the full process:
ore leaching → electrolyte preparation → electrowinning cell (cathode Fe,
anode O2/Cl2 with optional gas handling) → washing/drying → optional
carburization / annealing → product → electrolyte recycle with purge,
ligand/make-up, and anode durability.

This is a screening visualization to document material flows, not a
piping & instrumentation diagram (P&ID).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

DEFAULT_STYLE = {
    "leaching_color": "#a6cee3",
    "electrolyte_color": "#b2df8a",
    "cell_color": "#fdbf6f",
    "anode_color": "#fb9a99",
    "cathode_color": "#cab2d6",
    "post_color": "#ffff99",
    "recycle_color": "#ff7f00",
    "purge_color": "#e31a1c",
    "product_color": "#33a02c",
    "arrow_color": "#333333",
}


def _draw_block(ax, xy, width, height, text, facecolor, edgecolor="#333333", alpha=0.85, fontsize=9, fontweight="bold"):
    x, y = xy
    rect = patches.Rectangle((x, y), width, height, linewidth=1.5,
                             edgecolor=edgecolor, facecolor=facecolor, alpha=alpha, zorder=2)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, wrap=True, zorder=3)


def _draw_arrow(ax, start, end, label=None, color="#333333", style="-", lw=1.5, fontsize=8):
    x0, y0 = start
    x1, y1 = end
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle=style,
                                connectionstyle="arc3,rad=0.0"), zorder=1)
    if label:
        mx, my = (x0 + x1)/2, (y0 + y1)/2
        ax.text(mx, my + 0.03, label, ha="center", va="bottom", fontsize=fontsize,
                color=color, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7), zorder=4)


def generate_process_flow_diagram(
    output_path: str | Path = "docs/figures/process_flow_diagram.png",
    title: str = "Aqueous Electrowinning — Block Flow Diagram (screening)",
    style: Optional[Dict[str, Any]] = None,
    show_energy: bool = True,
) -> Path:
    """
    Generate a process flow block diagram and save to PNG.

    Returns the output path.
    """
    st = {**DEFAULT_STYLE, **(style or {})}
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Layout coordinates (normalized 0-1)
    # Top row: ore -> leaching -> electrolyte prep
    _draw_block(ax, (0.05, 0.65), 0.12, 0.18, "Iron Ore /\nWaste Feed\n(Fe₂O₃, pickle liquor)",
                st["leaching_color"])
    _draw_block(ax, (0.22, 0.65), 0.07, 0.18, "Crush /\nGrind\n(to ~10 μm)",
                st["leaching_color"], fontsize=7)
    _draw_block(ax, (0.33, 0.65), 0.12, 0.18, "Ore Leaching /\nDissolution\n(H₂SO₄ / HCl)",
                st["leaching_color"])
    _draw_block(ax, (0.49, 0.65), 0.14, 0.18, "Electrolyte\nPreparation\n(Fe²⁺ + complexant\npH/T control)",
                st["electrolyte_color"])

    # Center: electrowinning cell split anode/cathode
    _draw_block(ax, (0.42, 0.35), 0.16, 0.22, "Electrowinning\nCell\nCathode: Fe²⁺→Fe\nAnode: OER/CER\nj=100-500 mA/cm²",
                st["cell_color"], fontsize=10)

    # Cathode and anode sub-blocks visual hint (smaller inside)
    _draw_block(ax, (0.295, 0.38), 0.10, 0.14, "Cathode\nFe deposition\nPulse/PRE ready",
                st["cathode_color"], alpha=0.9, fontsize=7)
    _draw_block(ax, (0.60, 0.38), 0.10, 0.14, "Anode\nDSA IrO₂ / NiCo\nO₂/Cl₂ gas",
                st["anode_color"], alpha=0.9, fontsize=7)

    # Post-processing
    _draw_block(ax, (0.12, 0.35), 0.13, 0.16, "Washing /\nDrying /\nMass balance\n(FE gravimetric)",
                st["post_color"], fontsize=8)
    _draw_block(ax, (0.075, 0.08), 0.18, 0.18, "Post-processing\n• Carburization\n• Annealing\n• Rolling?\n(HV, UTS, XRD)",
                st["post_color"])

    # Product
    _draw_block(ax, (0.35, 0.05), 0.15, 0.12, "Steel Product\nSheet / Plate\nGrade: AISI 1008-4340\n(scr.)",
                "#b2df8a", fontsize=9)

    # Recycle loop
    _draw_block(ax, (0.65, 0.65), 0.18, 0.18, "Electrolyte Recycle\nCSTR + filtration\nFe/Ligand/Cl⁻/Impurity\nclosed_loop.py",
                st["recycle_color"], alpha=0.85)

    _draw_block(ax, (0.72, 0.35), 0.15, 0.08, "Gas Handling\nO₂ vent / Cl₂ scrub",
                "#f0f0f0", fontsize=7)

    _draw_block(ax, (0.87, 0.60), 0.08, 0.10, "Purge\n+ Treatment\n(impurity bleed)",
                st["purge_color"], alpha=0.6, fontsize=7)

    _draw_block(ax, (0.87, 0.75), 0.08, 0.08, "Make-up\nFe, ligand\nH₂O",
                st["electrolyte_color"], alpha=0.7, fontsize=7)

    _draw_block(ax, (0.65, 0.08), 0.20, 0.10, "Anode Durability\nCoating wear mg/kAh\nVoltage drift\nPhase IV model",
                "#d9d9d9", fontsize=7)

    # Arrows: main flow
    _draw_arrow(ax, (0.17, 0.74), (0.22, 0.74), label="ore / dust", color=st["arrow_color"])
    _draw_arrow(ax, (0.29, 0.74), (0.33, 0.74), label="", color=st["arrow_color"])
    _draw_arrow(ax, (0.45, 0.74), (0.49, 0.74), label="leachate", color=st["arrow_color"])
    _draw_arrow(ax, (0.56, 0.65), (0.56, 0.57), label="fresh electrolyte", color=st["arrow_color"])
    _draw_arrow(ax, (0.395, 0.46), (0.25, 0.46), label="Fe deposit", color=st["arrow_color"])
    _draw_arrow(ax, (0.185, 0.35), (0.16, 0.26), label="", color=st["arrow_color"])
    _draw_arrow(ax, (0.165, 0.08), (0.35, 0.11), label="treated Fe sheet", color=st["arrow_color"])

    # Gas arrow
    _draw_arrow(ax, (0.60, 0.52), (0.72, 0.45), label="O₂ / Cl₂", color=st["anode_color"])

    # Recycle arrows
    _draw_arrow(ax, (0.65, 0.74), (0.59, 0.74), label="barren / Fe-depleted", color=st["recycle_color"], style="--")
    _draw_arrow(ax, (0.65, 0.70), (0.49, 0.65), label="recycle / CSTR loop", color=st["recycle_color"], style="--")
    _draw_arrow(ax, (0.83, 0.74), (0.87, 0.70), label="", color=st["purge_color"], style=":")
    _draw_arrow(ax, (0.87, 0.75), (0.83, 0.80), label="ligand / Fe²⁺", color=st["electrolyte_color"], style=":")

    # Energy annotation if requested
    if show_energy:
        ax.text(0.50, 0.92, title + "\n25-90°C, 0.04 $/kWh, 2-3 kWh/kg, >90% CE target",
                ha="center", va="center", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#333333", alpha=0.9))

        ax.text(0.02, 0.02,
                "Models: pourbaix.py / kinetics.py / transport.py / pulse.py / hull_cell.py\n"
                "co_deposition.py / mechanical_properties.py / anode.py / closed_loop.py\n"
                "technoeconomic.py / scenarios.py  — screening, not P&ID",
                fontsize=6, va="bottom", ha="left", color="#666666")

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def generate_detailed_flow_with_composition(
    output_path: str | Path = "docs/figures/process_flow_detailed.png",
    include_carbon: bool = True,
) -> Path:
    """
    More detailed variant that shows Fe²⁺, Ni²⁺, carbon particle paths.
    """

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Left: block flow (simplified reuse of above but inline)
    _draw_block(ax, (0.05, 0.70), 0.18, 0.20, "Feed:\n• FeSO₄ / FeCl₂\n• NiSO₄ (0-0.5 M)\n• Citrate/Glycine", "#a6cee3")
    _draw_block(ax, (0.32, 0.70), 0.20, 0.20, "Electrolyte:\n+ Carbon particles\n1 g/L, zeta -25 mV\nSize 1.5 µm", "#b2df8a" if include_carbon else "#f0f0f0")
    _draw_block(ax, (0.30, 0.35), 0.25, 0.25, "Cell (anomalous co-dep)\nFe²⁺→Fe, Ni²⁺→Ni\nGuglielmi C incorp.\nCE 90-99%\nGrain 0.1-3.5 µm", "#fdbf6f")
    _draw_block(ax, (0.05, 0.35), 0.18, 0.20, "Deposit:\nFe(95%)-Ni(2-5%)-C(0-3%)\nMass balance\nSEM/EDS, comb.", "#cab2d6")
    _draw_block(ax, (0.05, 0.05), 0.48, 0.20, "Mechanical Props:\nYS 250-650 MPa (Hall-Petch + ss + dispersion)\nHV 100-250 kgf/mm²\nAISI 1008-4340 screening", "#ffffcc")

    _draw_arrow(ax, (0.23, 0.80), (0.32, 0.80), "dissolved", color="#333333")
    _draw_arrow(ax, (0.42, 0.70), (0.42, 0.60), "", color="#333333")
    _draw_arrow(ax, (0.30, 0.45), (0.23, 0.45), "panel", color="#333333")
    _draw_arrow(ax, (0.14, 0.35), (0.28, 0.18), "to testing", color="#333333", style="--")

    ax.set_title("Feed → Electrolyte → Co-deposition → Mechanical", fontsize=10, fontweight="bold")

    # Right: Sankey-like energy and mass balance illustrative
    ax2 = axes[1]
    ax2.axis("off")

    # Mass balance pie-style block diagram using rectangles
    _draw_block(ax2, (0.10, 0.70), 0.35, 0.15, "Current: 100%\n→ Fe 93%\n→ Ni 2%\n→ HER 5%", "#e0e0e0", fontsize=8)
    _draw_block(ax2, (0.10, 0.50), 0.35, 0.15, "Mass in deposit:\nFe 97wt%\nNi 2.3wt%\nC 0.7wt%", "#cab2d6", fontsize=8)
    _draw_block(ax2, (0.10, 0.30), 0.35, 0.15, "Strengthening:\nHP: 320 MPa\nss Ni: 45 MPa\nC disp: 85 MPa\n→ σy ~ 450 MPa", "#ffff99", fontsize=7)
    _draw_block(ax2, (0.10, 0.10), 0.35, 0.15, "Energy:\nVcell 2.4-2.6 V\n2.5-3.0 MWh/t\nO₂ bubbles 0.03V", "#fdbf6f", fontsize=8)

    _draw_block(ax2, (0.60, 0.10), 0.35, 0.60, "QA Flags:\n• low_fe\n• low_ligand_ratio\n• high_impurity\n• high_porosity\n• nanocrystalline\nAll explicit, not hidden", "#ffcccc", fontsize=7)

    _draw_arrow(ax2, (0.27, 0.70), (0.27, 0.65), "", color="#333333")
    _draw_arrow(ax2, (0.27, 0.50), (0.27, 0.45), "", color="#333333")
    _draw_arrow(ax2, (0.27, 0.30), (0.27, 0.25), "", color="#333333")

    fig.suptitle("Aqueous Electrowinning — Detailed Co-deposition + Mechanical Path (screening)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


if __name__ == "__main__":
    p1 = generate_process_flow_diagram()
    p2 = generate_detailed_flow_with_composition()
    print(f"Wrote {p1} and {p2}")
