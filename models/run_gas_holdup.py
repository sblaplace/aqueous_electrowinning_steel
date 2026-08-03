"""Two-phase gas hold-up in the cathode channel: figures + JSON report.

Closes the hydrodynamics/bubble gap named in ``docs/NEXT_STEPS.md`` §3.3,
``docs/REFERENCE_CELL_DESIGN_BASIS.md`` line 42 ("a design choice to be
observed, not a validated bubble model") and ``docs/SIM_THEORY_CONFIDENCE.md``
claim 2 ("gas/flow partially modeled").

Run::

    python -m models.run_gas_holdup
    python -m models.run_gas_holdup --quick   # skip the coupled FE-engine solve
    aq-steel-gas-holdup
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .gas_holdup import (
    CONTACT_ANGLE_H2_DEG,
    LFL_H2_VOL_FRAC,
    ChannelGeometry,
    current_density_sweep,
    departure_diameter_m,
    height_scaling_screen,
    holdup_profile,
    hydrogen_safety,
    measurement_protocol,
    model_scope,
    solve_coupled,
    solve_current_distribution,
    terminal_rise_velocity_m_s,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments" / "data"
FIGURES = ROOT / "docs" / "figures"

COLORS = {
    "void": "#2b5c8f",
    "kappa": "#d95f02",
    "current": "#1b9e77",
    "delta": "#7570b3",
    "warn": "#d73027",
    "ok": "#1a9850",
}

#: RC-1 kill-criterion current density (docs/PROGRAM_SUMMARY.md).
J_KILL_MA_CM2 = 300.0
#: Screening uniformity floor for an acceptable electrode.
UNIFORMITY_FLOOR = 0.90


def _fig_axial_profiles() -> Path:
    """Figure 1: axial void fraction, conductivity and current at three j."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=150)
    geom = ChannelGeometry()

    for j, style in ((100.0, "-"), (300.0, "--"), (500.0, ":")):
        prof = holdup_profile(j_mA_cm2=j, gas_current_fraction=0.15,
                              geometry=geom, n_segments=24)
        j_re = solve_current_distribution(
            j_mean_mA_cm2=j, kappa_eff_S_m=prof.kappa_eff_S_m,
            geometry=geom, surface_coverage=prof.surface_coverage,
        )
        y_mm = prof.y_m * 1000.0
        axes[0].plot(prof.void_fraction * 100.0, y_mm, style,
                     color=COLORS["void"], label=f"{j:.0f} mA/cm²")
        axes[1].plot(prof.kappa_eff_S_m, y_mm, style,
                     color=COLORS["kappa"], label=f"{j:.0f} mA/cm²")
        axes[2].plot(j_re / j, y_mm, style,
                     color=COLORS["current"], label=f"{j:.0f} mA/cm²")

    axes[0].set_xlabel("void fraction ε (%)")
    axes[0].set_ylabel("height above inlet (mm)")
    axes[0].set_title("Gas accumulates upward")
    axes[1].set_xlabel("κ_eff (S/m)")
    axes[1].set_title("Bruggeman conductivity")
    axes[2].axvline(1.0, color="0.6", lw=0.8)
    axes[2].set_xlabel("local j / mean j")
    axes[2].set_title("Current leaves the gassy top")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("RC-1 cathode channel: axial two-phase profiles (FE = 85 %, screening L0)",
                 fontsize=11)
    fig.tight_layout()
    out = FIGURES / "gas_holdup_axial_profiles.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _fig_scaling_and_safety(sweep: List[Dict[str, Any]],
                            heights: List[Dict[str, Any]]) -> Path:
    """Figure 2: j-sweep, electrode-height scaling limit, hydrogen dilution."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=150)

    # Panel 1 — void fraction and ohmic penalty vs current density
    js = [r["j_mA_cm2"] for r in sweep]
    ax = axes[0]
    ax.plot(js, [r["outlet_void_fraction"] * 100.0 for r in sweep],
            "o-", color=COLORS["void"], label="outlet ε (%)")
    ax.plot(js, [(r["conductivity_penalty"] - 1.0) * 100.0 for r in sweep],
            "s--", color=COLORS["kappa"], label="ohmic penalty (%)")
    ax.axvline(J_KILL_MA_CM2, color=COLORS["warn"], lw=1.0, ls=":")
    ax.annotate("kill criterion\n300 mA/cm²", xy=(J_KILL_MA_CM2, ax.get_ylim()[1] * 0.55),
                fontsize=7, color=COLORS["warn"], ha="right")
    ax.set_xlabel("current density (mA/cm²)")
    ax.set_ylabel("percent")
    ax.set_title("RC-1: hold-up is small")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2 — the scale-up limit
    ax = axes[1]
    hs = [r["height_mm"] for r in heights]
    uni = [r["current_uniformity"] for r in heights]
    colors = [COLORS["ok"] if r["passes_uniformity_floor"] else COLORS["warn"] for r in heights]
    ax.plot(hs, uni, "-", color="0.5", zorder=1)
    ax.scatter(hs, uni, c=colors, s=55, zorder=2)
    ax.axhline(UNIFORMITY_FLOOR, color=COLORS["warn"], ls="--", lw=1.0)
    ax.annotate("uniformity floor 0.90", xy=(hs[0], UNIFORMITY_FLOOR),
                xytext=(0, 5), textcoords="offset points", fontsize=7, color=COLORS["warn"])
    ax.set_xscale("log")
    ax.set_xlabel("electrode height (mm, log)")
    ax.set_ylabel("min j / max j")
    ax.set_title(f"Height limit @ {J_KILL_MA_CM2:.0f} mA/cm²")
    ax.grid(alpha=0.3, which="both")

    # Panel 3 — hydrogen dilution requirement
    ax = axes[2]
    currents = np.linspace(0.1, 3.0, 30)
    for fe, style in ((0.70, "--"), (0.85, "-"), (0.95, ":")):
        req = [hydrogen_safety(float(I), fe).required_dilution_flow_L_h for I in currents]
        ax.plot(currents, req, style, color=COLORS["void"], label=f"FE {fe:.0%}")
    ax.set_xlabel("cell current (A)")
    ax.set_ylabel("dilution air (L/h)")
    ax.set_title("Air to hold 25 % LFL (1 vol % H₂)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Gas hold-up: operating screen, scale-up limit and hydrogen safety (L0)",
                 fontsize=11)
    fig.tight_layout()
    out = FIGURES / "gas_holdup_scaling_safety.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _fig_coupling(coupled: List[Dict[str, Any]]) -> Path:
    """Figure 3: the two opposing bubble effects and their net FE result."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)

    js = [c["j_mean_mA_cm2"] for c in coupled]

    ax = axes[0]
    ax.plot(js, [c["FE_no_bubbles"] * 100.0 for c in coupled], "o--",
            color="0.5", label="FE, bubbles ignored")
    ax.plot(js, [c["area_average_FE"] * 100.0 for c in coupled], "s-",
            color=COLORS["current"], label="FE, coupled two-phase")
    ax.axhline(70.0, color=COLORS["warn"], ls=":", lw=1.0)
    ax.annotate("kill floor FE = 70 %", xy=(js[0], 70.5), fontsize=7, color=COLORS["warn"])
    ax.set_xlabel("mean current density (mA/cm²)")
    ax.set_ylabel("Faradaic efficiency (%)")
    ax.set_title("Net effect on the gated quantity")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    shift = [c["FE_shift_percentage_points"] for c in coupled]
    ax.bar([str(int(j)) for j in js], shift,
           color=[COLORS["ok"] if s >= 0 else COLORS["warn"] for s in shift])
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.set_xlabel("mean current density (mA/cm²)")
    ax.set_ylabel("ΔFE (percentage points)")
    ax.set_title("Microconvection gain − blanketing loss")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Coupled gas ↔ current ↔ FE fixed point (L0 screening)", fontsize=11)
    fig.tight_layout()
    out = FIGURES / "gas_holdup_fe_coupling.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _findings(sweep, heights, coupled, safety, geom) -> List[str]:
    out: List[str] = []

    d_um = departure_diameter_m(geom.superficial_liquid_velocity_m_s,
                               CONTACT_ANGLE_H2_DEG) * 1e6
    u_mm = terminal_rise_velocity_m_s(d_um * 1e-6) * 1000.0
    out.append(
        f"Departure diameter {d_um:.0f} µm rising at {u_mm:.0f} mm/s against a "
        f"{geom.superficial_liquid_velocity_m_s * 1000:.0f} mm/s liquid velocity: "
        "buoyancy, not forced flow, clears gas from the RC-1 channel."
    )

    kill = next((r for r in sweep if r["j_mA_cm2"] == J_KILL_MA_CM2), sweep[-1])
    out.append(
        f"At the {kill['j_mA_cm2']:.0f} mA/cm² kill-criterion point RC-1 reaches only "
        f"{kill['outlet_void_fraction'] * 100:.1f} % outlet void fraction, a "
        f"{(kill['conductivity_penalty'] - 1) * 100:.1f} % ohmic penalty and "
        f"{(1 - kill['current_uniformity']) * 100:.1f} % current spread — gas hold-up "
        "does not threaten the RC-1 decision run."
    )

    failing = [r for r in heights if not r["passes_uniformity_floor"]]
    passing = [r for r in heights if r["passes_uniformity_floor"]]
    if failing and passing:
        out.append(
            f"Hold-up becomes a scale-up constraint, not a bench one: uniformity holds to "
            f"{passing[-1]['height_mm']:.0f} mm of electrode height but fails the 0.90 floor "
            f"by {failing[0]['height_mm']:.0f} mm ({failing[0]['current_uniformity']:.2f}) and "
            f"reaches {failing[-1]['current_uniformity']:.2f} at "
            f"{failing[-1]['height_mm']:.0f} mm. RC-1 cannot observe this; a tall-cell "
            "geometry-transfer test must."
        )

    if coupled:
        worst = max(coupled, key=lambda c: abs(c["FE_shift_percentage_points"]))
        sign = "raises" if worst["FE_shift_percentage_points"] > 0 else "lowers"
        out.append(
            f"Coupling is net favourable at bench scale: bubbles {sign} area-average FE by "
            f"{abs(worst['FE_shift_percentage_points']):.2f} pp at "
            f"{worst['j_mean_mA_cm2']:.0f} mA/cm² "
            f"({worst['FE_no_bubbles'] * 100:.2f} % → {worst['area_average_FE'] * 100:.2f} %). "
            "Bubble microconvection thins the diffusion layer faster than blanketing and "
            "ohmic redistribution take FE away — so ignoring bubbles is conservative for "
            "FE, but it is not conservative for cell voltage."
        )

    out.append(
        f"Hydrogen: a 3 A all-HER cell makes {safety['hydrogen_flow_L_h']:.2f} L/h of wet gas "
        f"and needs {safety['required_dilution_flow_L_h']:.0f} L/h of dilution air to stay at "
        f"25 % of the {LFL_H2_VOL_FRAC * 100:.0f} vol % LFL. An unventilated 0.5 L headspace "
        f"reaches the LFL in {safety['time_to_LFL_min_unventilated']:.1f} min — which is "
        "the quantitative reason the design basis forbids a closed gas path."
    )

    out.append(
        "Dominant uncertainty is bubble departure diameter: it enters rise velocity "
        "quadratically through Stokes, so a 2× sizing error moves void fraction ~4×. "
        "This is the cheapest thing to measure and the first thing to measure."
    )
    return out


def main(quick: bool = False) -> dict:
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    geom = ChannelGeometry()

    sweep = current_density_sweep(
        j_values_mA_cm2=(50.0, 100.0, 200.0, 300.0, 400.0, 500.0),
        current_efficiency=0.85, geometry=geom,
    )
    heights = height_scaling_screen(j_mA_cm2=J_KILL_MA_CM2, base_geometry=geom,
                                    uniformity_floor=UNIFORMITY_FLOOR)

    coupled: List[Dict[str, Any]] = []
    if not quick:
        for j in (100.0, 300.0):
            coupled.append(solve_coupled(j_mean_mA_cm2=j, geometry=geom,
                                         n_segments=4, max_iterations=6).to_dict())

    safety_ref = hydrogen_safety(current_A=3.0, current_efficiency=0.0,
                                 headspace_L=0.5).to_dict()
    safety_table = [
        hydrogen_safety(current_A=I, current_efficiency=fe, headspace_L=0.5).to_dict()
        for I, fe in ((1.0, 0.85), (3.0, 0.85), (3.0, 0.70), (3.0, 0.0))
    ]

    fig1 = _fig_axial_profiles()
    fig2 = _fig_scaling_and_safety(sweep, heights)
    figs = [fig1, fig2]
    if coupled:
        figs.append(_fig_coupling(coupled))

    findings = _findings(sweep, heights, coupled, safety_ref, geom)

    report = {
        "title": "Cathode-channel gas hold-up and two-phase coupling report",
        "purpose": (
            "Model cathodic hydrogen generation, axial void fraction, effective "
            "conductivity, current redistribution, bubble microconvection and "
            "hydrogen safety margin for the RC-1 vertical channel — the "
            "hydrodynamics gap left open by docs/NEXT_STEPS.md section 3.3."
        ),
        "model_scope": model_scope(),
        "geometry": {
            "height_mm": geom.height_m * 1000.0,
            "width_mm": geom.width_m * 1000.0,
            "depth_mm": geom.depth_m * 1000.0,
            "interelectrode_gap_mm": geom.interelectrode_gap_m * 1000.0,
            "liquid_flow_L_min": geom.liquid_flow_L_min,
            "superficial_liquid_velocity_m_s": geom.superficial_liquid_velocity_m_s,
            "electrode_area_cm2": geom.electrode_area_cm2,
            "source": "processes/reference_cell_rc1.yaml",
        },
        "current_density_sweep": sweep,
        "height_scaling_screen": heights,
        "coupled_operating_points": coupled,
        "hydrogen_safety": safety_table,
        "measurement_protocol": measurement_protocol(),
        "findings": findings,
        "figures": [f"docs/figures/{p.name}" for p in figs],
    }

    out = DATA / "gas_holdup_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("Findings:")
    for f in findings:
        print(f"  - {f}")
    print()
    for p in figs:
        print(f"Wrote {p.relative_to(ROOT)}")
    print(f"Wrote {out.relative_to(ROOT)}")
    return report


def cli() -> None:
    ap = argparse.ArgumentParser(description="Cathode-channel gas hold-up model")
    ap.add_argument("--quick", action="store_true",
                    help="skip the coupled FE-engine fixed-point solve")
    args = ap.parse_args()
    main(quick=args.quick)


if __name__ == "__main__":
    cli()
