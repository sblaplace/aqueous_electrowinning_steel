"""
Pilot-scale Piping & Instrumentation Diagram (P&ID) generator for aqueous electrowinning.

Extends the block-flow diagram (process_flow.py) with detailed equipment symbols,
instrument tags (ISA), and control loops for a pilot plant.

Equipment included (pilot 100 kg/day to 1 t/day scale):
* Leaching tank TK-101 (agitator, LT, TT, pHAT)
* Electrolyte prep / filter (TK-102, FT-102 filtration)
* Electrolyte storage TK-103
* Electrowinning cell stack (E-201 / C-201) with rectifier, TT, PT, FT, AT (Fe2+), LT
* Recirculation pump P-201, heat exchanger HE-201, filter FL-201
* CSTR electrolyte recycle TK-202 with LT, FT, AT impurity
* Gas handling: O2 vent (with O2 AT), Cl2 scrubber TK-301 with pH control
* Purge treatment TK-302
* Washing/drying station (TK-401)
* Carburizing retort furnace F-501 with gas manifold (FC CO, FC CO2, FC CH4, FC H2),
  zirconia O2 probe (AIT-501 / O2), dew point transmitter (AIT-502), TT-501
* Quench tank TK-502 with TT
* Tempering furnace F-503 with TT
* Product handling / QC station with HV/QC

Instruments: ISA bubbles: LT level, TT temperature, PT pressure, FT flow,
AT analyzer (Fe2+, O2, CO/CO2), pHAT pH, AH_T_ etc. Control valves FV, TV, LV.

Two figures:
* pid_overview.png — simplified P&ID (major equipment + recyc + gas)
* pid_detailed.png — full instrumentation with tag numbers and control loops

All screening — not a certified P&ID for construction, but suitable for
HAZOP and pilot costing.

Usage: python -m models.run_pid
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"

# Colors
COLOR_TANK = "#a6cee3"
COLOR_ELEC = "#b2df8a"
COLOR_CELL = "#fdbf6f"
COLOR_GAS = "#fb9a99"
COLOR_FURNACE = "#ff7f00"
COLOR_PUMP = "#cab2d6"
COLOR_FILTER = "#ffff99"
COLOR_HE = "#e5f5e0"
COLOR_QC = "#d9d9d9"


def _draw_tank(ax, xy, w, h, label, color=COLOR_TANK, tag=None):
    x, y = xy
    rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor="#333333", facecolor=color, alpha=0.9, zorder=2)
    ax.add_patch(rect)
    # top ellipse for tank
    ellipse = patches.Ellipse((x + w/2, y + h), w, h*0.15, linewidth=1.2, edgecolor="#333333", facecolor=color, alpha=1.0, zorder=3)
    ax.add_patch(ellipse)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=7, fontweight="bold", wrap=True, zorder=4)
    if tag:
        ax.text(x + w/2, y - 0.02, tag, ha="center", va="top", fontsize=6, color="#333333", fontweight="bold")


def _draw_pump(ax, xy, size=0.03, label="P-201", orient="h"):
    x, y = xy
    circle = patches.Circle((x, y), size, linewidth=1.2, edgecolor="#333333", facecolor=COLOR_PUMP, zorder=3)
    ax.add_patch(circle)
    # triangle inside for flow direction
    if orient == "h":
        tri = patches.Polygon([(x - size*0.5, y - size*0.4), (x - size*0.5, y + size*0.4), (x + size*0.5, y)], closed=True, facecolor="#333333", zorder=4)
    else:
        tri = patches.Polygon([(x - size*0.4, y + size*0.5), (x + size*0.4, y + size*0.5), (x, y - size*0.5)], closed=True, facecolor="#333333", zorder=4)
    ax.add_patch(tri)
    ax.text(x, y - size - 0.015, label, ha="center", va="top", fontsize=5, fontweight="bold")


def _draw_he(ax, xy, w, h, label="HE-201"):
    x, y = xy
    rect = patches.Rectangle((x, y), w, h, linewidth=1.2, edgecolor="#333333", facecolor=COLOR_HE, alpha=0.9, zorder=2)
    ax.add_patch(rect)
    # diagonal line for heat exchange
    ax.plot([x, x+w], [y, y+h], color="#333333", linewidth=1.0, zorder=3)
    ax.plot([x, x+w], [y+h, y], color="#333333", linewidth=1.0, zorder=3)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=5, fontweight="bold", zorder=4)


def _draw_filter(ax, xy, w, h, label="FL-201"):
    x, y = xy
    # diamond shape? Use trapezoid? We'll use rectangle with hatch
    rect = patches.Rectangle((x, y), w, h, linewidth=1.2, edgecolor="#333333", facecolor=COLOR_FILTER, hatch="///", alpha=0.8, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=5, fontweight="bold", zorder=4)


def _draw_instrument(ax, xy, tag, var="TT"):
    """ISA bubble: circle with tag inside, var prefix outside?"""
    x, y = xy
    circle = patches.Circle((x, y), 0.015, linewidth=1.0, edgecolor="#333333", facecolor="white", zorder=5)
    ax.add_patch(circle)
    ax.text(x, y, tag, ha="center", va="center", fontsize=4.5, fontweight="bold", zorder=6)
    # var label just above
    # ax.text(x, y+0.018, var, ha="center", va="bottom", fontsize=4, color="#666666")


def _draw_line(ax, start, end, label=None, style="-", color="#333333", lw=1.0, arrow=False):
    x0, y0 = start
    x1, y1 = end
    if arrow:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle=style), zorder=1)
    else:
        ax.plot([x0, x1], [y0, y1], color=color, linestyle=style, linewidth=lw, zorder=1)
    if label:
        mx, my = (x0 + x1)/2, (y0 + y1)/2
        ax.text(mx, my + 0.01, label, ha="center", va="bottom", fontsize=5, color=color,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7), zorder=4)


def _draw_ctrl_loop(ax, from_instrument, to_valve, style="--", color="#e41a1c"):
    x0, y0 = from_instrument
    x1, y1 = to_valve
    # Dashed red for control signal
    ax.plot([x0, x1], [y0, y1], color=color, linestyle=style, linewidth=0.8, zorder=2)


def generate_pid_overview(
    output_path: Path = FIG_DIR / "pid_overview.png",
) -> Path:
    """Simplified P&ID overview (pilot 1 t/day)."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # --- Leaching ---
    _draw_tank(ax, (0.05, 0.65), 0.10, 0.12, "Leaching\nTK-101\nAgitator", COLOR_TANK, "TK-101")
    _draw_instrument(ax, (0.055, 0.80), "LT-101")
    _draw_instrument(ax, (0.145, 0.80), "TT-101")
    _draw_instrument(ax, (0.10, 0.62), "pHAT-101")

    # Filter
    _draw_filter(ax, (0.18, 0.68), 0.06, 0.06, "FL-101")

    # Electrolyte prep TK-102
    _draw_tank(ax, (0.27, 0.65), 0.10, 0.12, "Electrolyte\nPrep\nTK-102", COLOR_ELEC, "TK-102")
    _draw_instrument(ax, (0.275, 0.80), "AT-102\nFe2+")

    # Storage TK-103
    _draw_tank(ax, (0.40, 0.65), 0.10, 0.12, "Storage\nTK-103", COLOR_ELEC, "TK-103")
    _draw_instrument(ax, (0.405, 0.80), "LT-103")

    # Pump P-103 to cell
    _draw_pump(ax, (0.52, 0.71), 0.02, "P-103")

    # Cell stack
    _draw_tank(ax, (0.56, 0.62), 0.12, 0.18, "Electrowinning\nCell Stack\nC-201 / E-201\nRectifier 257 kW", COLOR_CELL, "C-201")
    _draw_instrument(ax, (0.57, 0.84), "FT-201")
    _draw_instrument(ax, (0.62, 0.84), "TT-201")
    _draw_instrument(ax, (0.67, 0.84), "PT-201")
    _draw_instrument(ax, (0.60, 0.60), "AT-201\nFe2+")

    # Recirc pump + HE + filter
    _draw_pump(ax, (0.58, 0.55), 0.02, "P-201")
    _draw_he(ax, (0.62, 0.52), 0.06, 0.04, "HE-201")
    _draw_filter(ax, (0.70, 0.52), 0.05, 0.04, "FL-201")

    # CSTR recycle TK-202
    _draw_tank(ax, (0.78, 0.65), 0.10, 0.12, "Recycle CSTR\nTK-202\n+ Purge", "#ff7f00", "TK-202")
    _draw_instrument(ax, (0.785, 0.80), "LT-202")
    _draw_instrument(ax, (0.855, 0.80), "AT-202\nImp")

    # Gas handling
    _draw_tank(ax, (0.70, 0.68), 0.06, 0.06, "O2 Vent\nTK-301A", COLOR_GAS, "TK-301A")
    _draw_tank(ax, (0.62, 0.68), 0.06, 0.06, "Cl2 Scrub\nTK-301B\npH-301B", COLOR_GAS, "TK-301B")

    # Purge
    _draw_tank(ax, (0.90, 0.65), 0.06, 0.08, "Purge\nTK-302", "#e31a1c", "TK-302")

    # Washing
    _draw_tank(ax, (0.40, 0.35), 0.10, 0.10, "Wash/Dry\nTK-401", "#ffff99", "TK-401")

    # Carburizing furnace
    _draw_tank(ax, (0.55, 0.35), 0.12, 0.12, "Carburizing\nRetort F-501\nTT-501\nO2 AIT-501", COLOR_FURNACE, "F-501")
    _draw_instrument(ax, (0.56, 0.50), "AIT-501\nO2")
    _draw_instrument(ax, (0.65, 0.50), "AIT-502\nDP")

    # Gas manifold
    _draw_tank(ax, (0.70, 0.35), 0.08, 0.10, "Gas Manifold\nFC-501A-D\nCO/CO2/CH4/H2", COLOR_GAS, "GS-501")

    # Quench
    _draw_tank(ax, (0.40, 0.15), 0.10, 0.10, "Quench\nTK-502\nTT-502", "#a6cfe2", "TK-502")

    # Tempering furnace
    _draw_tank(ax, (0.55, 0.15), 0.12, 0.10, "Tempering\nF-503\nTT-503", COLOR_FURNACE, "F-503")

    # Product QC
    _draw_tank(ax, (0.70, 0.15), 0.12, 0.10, "Product QC\nHV, XRD, tensile\nAT-601", COLOR_QC, "PK-601")

    # Lines
    _draw_line(ax, (0.15, 0.71), (0.18, 0.71), "leachate", "-", "#333333", lw=1.2, arrow=True)
    _draw_line(ax, (0.24, 0.71), (0.27, 0.71), "", "-", "#333333", lw=1.2, arrow=True)
    _draw_line(ax, (0.37, 0.71), (0.40, 0.71), "", "-", "#333333", lw=1.2, arrow=True)
    _draw_line(ax, (0.50, 0.71), (0.52, 0.71), "", "-", "#333333", lw=1.2, arrow=True)
    _draw_line(ax, (0.54, 0.71), (0.56, 0.71), "fresh EL", "-", "#333333", lw=1.2, arrow=True)

    _draw_line(ax, (0.62, 0.62), (0.62, 0.56), "spent EL", "--", "#ff7f00", lw=1.2, arrow=True)
    _draw_line(ax, (0.60, 0.52), (0.70, 0.52), "", "--", "#ff7f00", lw=1.0, arrow=True)
    _draw_line(ax, (0.75, 0.54), (0.78, 0.65), "recycle", "--", "#ff7f00", lw=1.2, arrow=True)
    _draw_line(ax, (0.68, 0.68), (0.70, 0.68), "O2", "-", "#e31a1c", lw=1.0, arrow=True)
    _draw_line(ax, (0.68, 0.62), (0.78, 0.65), "barren", "--", "#ff7f00", lw=1.0, arrow=True)
    _draw_line(ax, (0.88, 0.69), (0.90, 0.69), "purge", ":", "#e31a1c", lw=1.0, arrow=True)

    # To wash
    _draw_line(ax, (0.56, 0.62), (0.45, 0.45), "Fe sheet", "-", "#333333", lw=1.2, arrow=True)
    _draw_line(ax, (0.45, 0.35), (0.55, 0.41), "", "-", "#333333", lw=1.0, arrow=True)
    _draw_line(ax, (0.67, 0.41), (0.70, 0.40), "CO/CH4/H2", "-", "#e31a1c", lw=1.0, arrow=True)
    _draw_line(ax, (0.61, 0.35), (0.50, 0.25), "carburized", "-", "#333333", lw=1.0, arrow=True)
    _draw_line(ax, (0.50, 0.15), (0.55, 0.15), "quenched", "-", "#333333", lw=1.0, arrow=True)
    _draw_line(ax, (0.67, 0.20), (0.70, 0.20), "tempered", "-", "#333333", lw=1.0, arrow=True)

    # Title
    ax.text(0.5, 0.92, "Pilot P&ID Overview — Aqueous Electrowinning + Carburizing + Tempering (1 t/day screening)\nTK Tank, P Pump, HE Heat Exchanger, FL Filter, F Furnace, LT/FT/TT/PT/AT/AIT Instrument",
            ha="center", va="center", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#333333", alpha=0.9))

    ax.text(0.01, 0.01, "Screening P&ID — ISA tags: LT level, FT flow, TT temperature, PT pressure, AT analyzer (Fe2+, O2), pHAT, AIT analyzer. Dashed = recycle/control signal.",
            fontsize=6, color="#666666")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def generate_pid_detailed(
    output_path: Path = FIG_DIR / "pid_detailed.png",
) -> Path:
    """Detailed P&ID with control loops and instrument bubbles."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(20, 11))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Reuse overview layout but add more instruments and control loops

    # Leaching TK-101 with agitator control
    _draw_tank(ax, (0.04, 0.65), 0.09, 0.12, "Leaching\nTK-101\nM-101 Agit", COLOR_TANK, "TK-101")
    _draw_instrument(ax, (0.045, 0.80), "LT-101")
    _draw_instrument(ax, (0.09, 0.80), "TT-101")
    _draw_instrument(ax, (0.13, 0.80), "pHAT-101")
    _draw_instrument(ax, (0.05, 0.62), "AT-101\nFeTot")
    _draw_pump(ax, (0.135, 0.71), 0.015, "P-101")
    _draw_line(ax, (0.125, 0.71), (0.135, 0.71), "", "-", "#333333", lw=1.0, arrow=True)
    _draw_ctrl_loop(ax, (0.045, 0.80), (0.135, 0.71), "--", "#e41a1c")  # LT to pump

    # FL-101
    _draw_filter(ax, (0.155, 0.68), 0.05, 0.05, "FL-101")

    # Electrolyte prep TK-102
    _draw_tank(ax, (0.22, 0.65), 0.09, 0.12, "Prep\nTK-102\nM-102", COLOR_ELEC, "TK-102")
    _draw_instrument(ax, (0.225, 0.80), "LT-102")
    _draw_instrument(ax, (0.265, 0.80), "AT-102\nFe2+")
    _draw_instrument(ax, (0.225, 0.62), "TT-102")

    # TK-103 storage
    _draw_tank(ax, (0.33, 0.65), 0.08, 0.12, "Storage\nTK-103", COLOR_ELEC, "TK-103")
    _draw_instrument(ax, (0.335, 0.80), "LT-103")
    _draw_pump(ax, (0.42, 0.71), 0.015, "P-103")
    _draw_instrument(ax, (0.405, 0.75), "FT-103")

    # Cell stack C-201
    _draw_tank(ax, (0.45, 0.60), 0.11, 0.18, "Cell Stack\nC-201\nE-201 Rect\n400 mA/cm2", COLOR_CELL, "C-201")
    _draw_instrument(ax, (0.455, 0.81), "FT-201")
    _draw_instrument(ax, (0.49, 0.81), "TT-201")
    _draw_instrument(ax, (0.525, 0.81), "PT-201")
    _draw_instrument(ax, (0.56, 0.81), "VT-201")
    _draw_instrument(ax, (0.455, 0.58), "AT-201\nFe2+")
    _draw_instrument(ax, (0.49, 0.58), "AT-201B\nNi2+")

    # Recirc loop
    _draw_pump(ax, (0.48, 0.55), 0.015, "P-201")
    _draw_he(ax, (0.51, 0.52), 0.05, 0.04, "HE-201")
    _draw_instrument(ax, (0.515, 0.51), "TT-201B")
    _draw_filter(ax, (0.57, 0.52), 0.04, 0.04, "FL-201")
    _draw_instrument(ax, (0.575, 0.50), "PDIT-201\nΔP")

    # CSTR TK-202
    _draw_tank(ax, (0.63, 0.65), 0.09, 0.12, "Recycle CSTR\nTK-202\nM-202", "#ff7f00", "TK-202")
    _draw_instrument(ax, (0.635, 0.80), "LT-202")
    _draw_instrument(ax, (0.675, 0.80), "AT-202\nImp")
    _draw_instrument(ax, (0.695, 0.80), "FT-202\nPurge")
    _draw_instrument(ax, (0.635, 0.62), "TT-202")

    # Gas handling
    _draw_tank(ax, (0.74, 0.68), 0.05, 0.06, "O2\nTK-301A", COLOR_GAS, "TK-301A")
    _draw_instrument(ax, (0.745, 0.66), "AT-301A\nO2")
    _draw_tank(ax, (0.74, 0.60), 0.05, 0.06, "Cl2 Scrub\nTK-301B", COLOR_GAS, "TK-301B")
    _draw_instrument(ax, (0.745, 0.58), "pHAT-301B")

    # Purge TK-302
    _draw_tank(ax, (0.81, 0.65), 0.05, 0.10, "Purge\nTK-302", "#e31a1c", "TK-302")

    # Wash TK-401
    _draw_tank(ax, (0.33, 0.35), 0.08, 0.10, "Wash/Dry\nTK-401", "#ffff99", "TK-401")
    _draw_instrument(ax, (0.335, 0.47), "LT-401")

    # Carburizing furnace F-501
    _draw_tank(ax, (0.45, 0.32), 0.11, 0.14, "Carburizing\nRetort F-501\nHeater H-501", COLOR_FURNACE, "F-501")
    _draw_instrument(ax, (0.455, 0.48), "TT-501")
    _draw_instrument(ax, (0.495, 0.48), "AIT-501\nO2")
    _draw_instrument(ax, (0.535, 0.48), "AIT-502\nDP")
    _draw_instrument(ax, (0.455, 0.30), "AT-501\nC foil")

    # Gas manifold GS-501
    _draw_tank(ax, (0.58, 0.32), 0.07, 0.14, "Gas Manifold\nGS-501\nFC-501A\nFC-501B\nFC-501C\nFC-501D", COLOR_GAS, "GS-501")
    _draw_instrument(ax, (0.585, 0.28), "FT-501A\nCO")
    _draw_instrument(ax, (0.62, 0.28), "FT-501B\nCO2")
    _draw_instrument(ax, (0.585, 0.25), "FT-501C\nCH4")
    _draw_instrument(ax, (0.62, 0.25), "FT-501D\nH2")

    # Quench TK-502
    _draw_tank(ax, (0.33, 0.15), 0.08, 0.10, "Quench\nTK-502", "#a6cfe2", "TK-502")
    _draw_instrument(ax, (0.335, 0.27), "TT-502")

    # Tempering furnace F-503
    _draw_tank(ax, (0.45, 0.12), 0.11, 0.12, "Tempering\nF-503\nH-503", COLOR_FURNACE, "F-503")
    _draw_instrument(ax, (0.455, 0.26), "TT-503")

    # Product QC PK-601
    _draw_tank(ax, (0.62, 0.12), 0.10, 0.12, "Product QC\nPK-601\nHV/XRD", COLOR_QC, "PK-601")
    _draw_instrument(ax, (0.625, 0.26), "AIT-601\nHV")

    # Lines
    _draw_line(ax, (0.13, 0.71), (0.155, 0.71), "leach", "-", "#333333", lw=1.0, arrow=True)
    _draw_line(ax, (0.205, 0.71), (0.22, 0.71), "", "-", "#333333", lw=1.0, arrow=True)
    _draw_line(ax, (0.31, 0.71), (0.33, 0.71), "", "-", "#333333", lw=1.0, arrow=True)
    _draw_line(ax, (0.41, 0.71), (0.42, 0.71), "", "-", "#333333", lw=1.0, arrow=True)
    _draw_line(ax, (0.435, 0.71), (0.45, 0.71), "fresh EL", "-", "#333333", lw=1.0, arrow=True)

    _draw_line(ax, (0.51, 0.60), (0.51, 0.56), "spent EL", "--", "#ff7f00", lw=1.0, arrow=True)
    _draw_line(ax, (0.495, 0.52), (0.57, 0.52), "", "--", "#ff7f00", lw=1.0, arrow=True)
    _draw_line(ax, (0.61, 0.54), (0.63, 0.65), "recycle", "--", "#ff7f00", lw=1.2, arrow=True)
    _draw_line(ax, (0.56, 0.68), (0.63, 0.68), "barren", "--", "#ff7f00", lw=1.0, arrow=True)
    _draw_line(ax, (0.56, 0.60), (0.74, 0.63), "O2/Cl2", "-", "#e31a1c", lw=1.0, arrow=True)
    _draw_line(ax, (0.72, 0.65), (0.81, 0.70), "purge", ":", "#e31a1c", lw=1.0, arrow=True)

    # Wash to carburizing
    _draw_line(ax, (0.45, 0.60), (0.37, 0.45), "Fe sheet", "-", "#333333", lw=1.2, arrow=True)
    _draw_line(ax, (0.38, 0.35), (0.45, 0.39), "", "-", "#333333", lw=1.0, arrow=True)
    _draw_line(ax, (0.58, 0.39), (0.58, 0.46), "gas mix", "-", "#e31a1c", lw=1.0, arrow=True)
    _draw_line(ax, (0.51, 0.32), (0.41, 0.25), "carburized", "-", "#333333", lw=1.0, arrow=True)
    _draw_line(ax, (0.41, 0.15), (0.45, 0.18), "quenched", "-", "#333333", lw=1.0, arrow=True)
    _draw_line(ax, (0.56, 0.18), (0.62, 0.18), "tempered", "-", "#333333", lw=1.0, arrow=True)

    # Control loops (dashed red)
    # TT-501 to heater
    _draw_ctrl_loop(ax, (0.455, 0.48), (0.45, 0.42), "--", "#e41a1c")
    # O2 probe to gas FCs
    _draw_ctrl_loop(ax, (0.495, 0.48), (0.60, 0.35), "--", "#e41a1c")
    # FT-201 to P-201
    _draw_ctrl_loop(ax, (0.455, 0.81), (0.48, 0.55), "--", "#e41a1c")

    ax.text(0.5, 0.92, "Pilot P&ID Detailed — Aqueous Electrowinning + Gas Carburizing + Tempering (ISA Tags)\n"
                      "Solid: process flow, Dashed orange: electrolyte recycle, Dashed red: control signal, Dotted red: purge",
            ha="center", va="center", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#333333", alpha=0.95))

    ax.text(0.01, 0.01,
            "Equipment: TK tank, P pump, FL filter, HE heat exchanger, F furnace, C cell stack, GS gas station, PK product QC\n"
            "Instruments: LT level, FT flow, TT temperature, PT pressure, VT voltage, AT analyzer, AIT analyzer indicator transmitter, PDIT differential pressure, pHAT\n"
            "Control: LV level valve, FV flow valve, TV temperature valve, H heater\n"
            "Screening P&ID — not for construction. For HAZOP and pilot costing.",
            fontsize=5.5, color="#555555")

    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    p1 = generate_pid_overview()
    p2 = generate_pid_detailed()
    print(f"Wrote {p1} and {p2}")
