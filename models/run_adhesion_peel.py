"""Deposit adhesion / peel screen: figures + JSON report.

Answers the question ``models/cell_architecture.py`` flagged and refused to
compute — *does iron peel from a drum?* — and returns the go/no-go for the
continuous-foil architecture branch, plus the coupon experiment that would
replace the estimate with a measurement.

Run::

    python -m models.run_adhesion_peel
    aq-steel-adhesion
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from .adhesion_peel import (
    MAX_WINDER_TENSION_N_PER_M,
    MIN_CONTROLLABLE_TENSION_N_PER_M,
    SUBSTRATES,
    PeelConditions,
    amplification_robustness,
    comparison_table,
    conditions_from_deposition,
    coupon_test_protocol,
    evaluate_peel,
    foil_route_verdict,
    grain_size_sweep,
    hydrogen_sweep,
    model_scope,
    screen_substrates,
    thickness_sweep,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments" / "data"
FIGURES = ROOT / "docs" / "figures"

COLORS = {
    "ti_passive_tio2": "tab:blue",
    "ti_bare_etched": "tab:cyan",
    "stainless_316_passive": "tab:orange",
    "chromium_plated": "tab:green",
    "copper_substrate": "tab:red",
    "ptfe_release_coating": "tab:purple",
}
SHORT = {
    "ti_passive_tio2": "Ti /\nTiO₂",
    "ti_bare_etched": "Ti\netched",
    "stainless_316_passive": "316L\npassive",
    "chromium_plated": "Hard\nCr",
    "copper_substrate": "Cu\n(control)",
    "ptfe_release_coating": "PTFE\n(insul.)",
}
HATCH = {"commercial": "", "pilot": "//", "lab": "xx", "concept": ".."}

OUTCOME_COLORS = {
    "clean_peel": "#2e7d32",
    "marginal_peel": "#f9a825",
    "spontaneous_delamination": "#6a1b9a",
    "tears_before_peel": "#c62828",
    "cohesive_failure_in_film": "#8d6e63",
    "bonded_no_release": "#37474f",
}


# ═════════════════════════════════════════════════════════════════════
#  Figure 1 — substrate screen
# ═════════════════════════════════════════════════════════════════════

def _fig_substrate_screen(results, conditions: PeelConditions) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), constrained_layout=True)
    ids = [r.substrate_id for r in results]
    labels = [SHORT[i] for i in ids]
    colors = [COLORS[i] for i in ids]
    hatches = [HATCH[r.evidence_level] for r in results]
    x = np.arange(len(results))

    # (0,0) Energy balance: driving force vs toughness vs film tearing
    ax = axes[0, 0]
    G = [r.driving_force_J_m2 for r in results]
    gam = [r.toughness_J_m2 for r in results]
    wf = [r.film_tearing_energy_J_m2 for r in results]
    ax.bar(x - 0.26, G, 0.26, label="Driving force $G$", color="#1874b4",
           edgecolor="black")
    ax.bar(x, gam, 0.26, label=r"Interfacial toughness $\Gamma$",
           color="#d95f02", edgecolor="black")
    ax.bar(x + 0.26, wf, 0.26, label=r"Film tearing $W_{film}$",
           color="#7f7f7f", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("Energy per unit area (J/m²)")
    ax.set_title(f"Peel energy balance at {conditions.thickness_um:.0f} µm")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    # (0,1) Peel force against the machine window
    ax = axes[0, 1]
    P = [min(r.peel_force_N_per_m, 1e6) for r in results]
    bars = ax.bar(labels, P, color=colors, edgecolor="black")
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
    ax.axhspan(conditions.min_controllable_tension_N_per_m,
               conditions.max_winder_tension_N_per_m,
               color="green", alpha=0.10)
    ax.axhline(conditions.max_winder_tension_N_per_m, color="crimson",
               linestyle="--", linewidth=1.5,
               label=f"Winder ceiling ({conditions.max_winder_tension_N_per_m:,.0f} N/m)")
    ax.axhline(conditions.min_controllable_tension_N_per_m, color="navy",
               linestyle=":", linewidth=1.5,
               label=f"Control floor ({conditions.min_controllable_tension_N_per_m:,.0f} N/m)")
    for xi, r in zip(x, results):
        if r.peel_force_N_per_m <= 0:
            ax.annotate("self-\nreleases", xy=(xi, 0), xytext=(0, 12),
                        textcoords="offset points", ha="center", fontsize=7,
                        color="purple")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylabel("Peel force per width (N/m)")
    ax.set_title("Required peel force vs the winding-line window")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(alpha=0.25, axis="y")

    # (1,0) Residual stress decomposition
    ax = axes[1, 0]
    intr = [r.stress_breakdown["intrinsic_MPa"] for r in results]
    hyd = [r.stress_breakdown["hydrogen_MPa"] for r in results]
    thm = [r.stress_breakdown["thermal_MPa"] for r in results]
    ax.bar(x - 0.26, intr, 0.26, label="Intrinsic (Hoffman)", color="#4daf4a",
           edgecolor="black")
    ax.bar(x, hyd, 0.26, label="Hydrogen effusion", color="#984ea3",
           edgecolor="black")
    ax.bar(x + 0.26, thm, 0.26, label="Thermal mismatch", color="#ff7f00",
           edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    # Symlog: the PTFE thermal outlier is ~-850 MPa and would otherwise
    # flatten every metal substrate into an invisible line at zero.
    ax.set_yscale("symlog", linthresh=10.0)
    ax.set_ylabel("Residual stress (MPa, tensile +, symlog)")
    ax.set_title("Where the stored energy comes from")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    # (1,1) Outcome map
    ax = axes[1, 1]
    ratios = [min(r.self_release_ratio, 1e3) for r in results]
    cohes = [min(r.cohesive_ratio, 1e3) for r in results]
    for xi, r in zip(x, results):
        ax.scatter(ratios[int(xi)], cohes[int(xi)], s=260,
                   color=OUTCOME_COLORS[r.outcome], edgecolor="black",
                   marker="o" if r.conductive else "X", zorder=5)
        ax.annotate(SHORT[r.substrate_id].replace("\n", " "),
                    xy=(ratios[int(xi)], cohes[int(xi)]),
                    xytext=(8, 6), textcoords="offset points", fontsize=8)
    ax.axvline(1.0, color="purple", linestyle="--", linewidth=1.4)
    ax.axhline(1.0, color="brown", linestyle="--", linewidth=1.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$G/\Gamma$  (>1 → self-releases)")
    ax.set_ylabel(r"$\Gamma/W_{film}$  (>1 → crack runs in the film)")
    ax.set_title("Failure-mode map (X = not electrically conductive)")
    ax.grid(alpha=0.25)
    ax.legend(
        handles=[Patch(facecolor=c, edgecolor="black", label=k.replace("_", " "))
                 for k, c in OUTCOME_COLORS.items()],
        fontsize=7, loc="best",
    )

    fig.suptitle(
        "Iron deposit adhesion screen — hatching marks evidence level "
        "(solid=commercial, xx=lab, ..=concept). Screening model; no iron "
        "peel data exists in this repository.",
        fontweight="bold", fontsize=10,
    )
    path = FIGURES / "adhesion_substrate_screen.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


# ═════════════════════════════════════════════════════════════════════
#  Figure 2 — the foil window
# ═════════════════════════════════════════════════════════════════════

def _fig_foil_window(conditions: PeelConditions) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    ref = SUBSTRATES["ti_passive_tio2"]

    # (0) Thickness: bounded from both sides
    ax = axes[0]
    sweep = thickness_sweep(ref, conditions=conditions)
    t = np.array(sweep["thickness_um"])
    ax.plot(t, sweep["driving_force_J_m2"], color="#1874b4", linewidth=2,
            label="Driving force $G$")
    ax.plot(t, sweep["toughness_J_m2"], color="#d95f02", linewidth=2,
            label=r"Toughness $\Gamma$")
    outcomes = sweep["outcome"]
    for i in range(len(t) - 1):
        ax.axvspan(t[i], t[i + 1], color=OUTCOME_COLORS[outcomes[i]],
                   alpha=0.13, linewidth=0)
    lo, hi = sweep["viable_thickness_min_um"], sweep["viable_thickness_max_um"]
    if lo is not None:
        ax.axvline(hi, color="purple", linestyle="--", linewidth=1.4,
                   label=f"Upper bound ≈ {hi:,.0f} µm")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Deposit thickness (µm)")
    ax.set_ylabel("Energy per unit area (J/m²)")
    ax.set_title("Thickness: $G \\propto h$ but $\\Gamma$ saturates")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # (1) Hydrogen: helps release, hurts the foil
    ax = axes[1]
    hsw = hydrogen_sweep(ref, conditions=conditions)
    ch = np.array(hsw["C_H_ppm"])
    ax.plot(ch, hsw["self_release_ratio"], color="#984ea3", linewidth=2)
    ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1.4,
               label="Spontaneous release")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Diffusible hydrogen (ppm)")
    ax.set_ylabel(r"$G/\Gamma$")
    ax.set_title("Hydrogen raises $G$ and lowers $\\Gamma$ — both toward release")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(ch, hsw["toughness_J_m2"], color="#d95f02", linewidth=1.6,
             linestyle=":")
    ax2.set_ylabel(r"$\Gamma$ (J/m²)", color="#d95f02")
    ax2.tick_params(axis="y", labelcolor="#d95f02")

    # (2) Grain size: the conflict with mechanical properties
    ax = axes[2]
    gsw = grain_size_sweep(ref, conditions=conditions)
    d = np.array(gsw["grain_size_um"])
    ax.plot(d, gsw["self_release_ratio"], color="#4daf4a", linewidth=2)
    ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1.4,
               label="Spontaneous release")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Grain size (µm)")
    ax.set_ylabel(r"$G/\Gamma$")
    ax.set_title("Fine grain strengthens the foil and unsticks it")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle(
        f"Continuous-foil window on {ref.name} — the foil route is bounded "
        f"from both sides, not just by adhesion",
        fontweight="bold", fontsize=11,
    )
    path = FIGURES / "adhesion_foil_window.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


# ═════════════════════════════════════════════════════════════════════
#  Figure 3 — robustness and the operating map
# ═════════════════════════════════════════════════════════════════════

def _fig_robustness(conditions: PeelConditions) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), constrained_layout=True)
    ref = SUBSTRATES["ti_passive_tio2"]

    # (0) Verdict vs the least-known parameter
    ax = axes[0]
    rob = amplification_robustness(ref, conditions)
    phi = np.array(rob["amplification"])
    outs = rob["outcome"]
    for i in range(len(phi) - 1):
        ax.axvspan(phi[i], phi[i + 1], color=OUTCOME_COLORS[outs[i]],
                   alpha=0.55, linewidth=0)
    ax.axvline(ref.plastic_amplification, color="black", linestyle="--",
               linewidth=1.8,
               label=f"Library estimate ({ref.plastic_amplification:.0f}×)")
    ax.set_xscale("log")
    ax.set_xlim(phi.min(), phi.max())
    ax.set_yticks([])
    ax.set_xlabel("Plastic amplification factor $\\phi$ (peel work / $W_{adh}$)")
    ax.set_title(
        "The verdict across the plausible range of the\n"
        "least-constrained parameter"
    )
    ax.legend(
        handles=[Patch(facecolor="black", label=f"estimate = {ref.plastic_amplification:.0f}×")]
        + [Patch(facecolor=OUTCOME_COLORS[k], label=f"{k.replace('_',' ')} "
                 f"({v:.0%})") for k, v in sorted(rob["outcome_fractions"].items(),
                                                  key=lambda kv: -kv[1])],
        fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
    )

    # (1) Operating map: thickness × hydrogen
    ax = axes[1]
    thick = np.logspace(0, np.log10(300), 34)
    hyd = np.logspace(-1, 2, 34)
    grid = np.zeros((len(hyd), len(thick)))
    codes = {
        "bonded_no_release": 0,
        "cohesive_failure_in_film": 1,
        "tears_before_peel": 2,
        "clean_peel": 3,
        "marginal_peel": 4,
        "spontaneous_delamination": 5,
    }
    for i, c in enumerate(hyd):
        for j, t in enumerate(thick):
            r = evaluate_peel(
                ref,
                PeelConditions(**{**conditions.__dict__,
                                  "thickness_um": float(t),
                                  "C_H_ppm": float(c)}),
            )
            grid[i, j] = codes[r.outcome]
    from matplotlib.colors import BoundaryNorm, ListedColormap

    order = sorted(codes, key=codes.get)
    cmap = ListedColormap([OUTCOME_COLORS[k] for k in order])
    norm = BoundaryNorm(np.arange(-0.5, len(order) + 0.5), cmap.N)
    ax.pcolormesh(thick, hyd, grid, cmap=cmap, norm=norm, shading="nearest")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Deposit thickness (µm)")
    ax.set_ylabel("Diffusible hydrogen (ppm)")
    ax.set_title("Operating map on the reference TiO₂ drum surface")
    ax.legend(
        handles=[Patch(facecolor=OUTCOME_COLORS[k], edgecolor="black",
                       label=k.replace("_", " ")) for k in order],
        fontsize=7, loc="upper left", framealpha=0.9,
    )

    fig.suptitle(
        "Adhesion screen robustness — a verdict that moves with $\\phi$ is a "
        "request for the coupon test, not a result",
        fontweight="bold", fontsize=11,
    )
    path = FIGURES / "adhesion_robustness.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


# ═════════════════════════════════════════════════════════════════════
#  Driver
# ═════════════════════════════════════════════════════════════════════

def main() -> dict:
    DATA.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # Reference case: the drum-and-strip target thickness from
    # cell_architecture.ARCHITECTURES["drum_and_strip"] (25 µm).
    conditions = PeelConditions()

    results = screen_substrates(conditions)
    print(comparison_table(results))
    print()

    verdict = foil_route_verdict(conditions)
    print(f"Foil-route verdict: {verdict['verdict'].upper()}")
    print(f"  {verdict['interpretation']}")
    print()

    # An operating point derived end-to-end from plating conditions.
    derived = conditions_from_deposition(
        j_mA_cm2=100.0,
        current_efficiency_percent=85.0,
        deposition_time_s=900.0,
    )
    derived_result = evaluate_peel(
        SUBSTRATES["ti_passive_tio2"], derived["conditions"]
    )
    print("Operating point derived from plating conditions "
          "(100 mA/cm², 85% FE, 15 min):")
    print(derived_result.summary())
    print()

    fig1 = _fig_substrate_screen(results, conditions)
    fig2 = _fig_foil_window(conditions)
    fig3 = _fig_robustness(conditions)

    protocol = coupon_test_protocol(conditions)

    report = {
        "title": "Deposit adhesion and peel screen",
        "purpose": (
            "Resolve the gating unknown cell_architecture.py flagged and "
            "declined to compute: whether electrodeposited iron peels from a "
            "drum cathode, and therefore whether the continuous-foil "
            "architecture branch survives."
        ),
        "model_scope": model_scope(),
        "conditions": {
            k: v for k, v in conditions.__dict__.items()
        },
        "substrates": [r.to_dict() for r in results],
        "foil_route_verdict": verdict,
        "derived_operating_point": {
            "inputs": {
                "j_mA_cm2": 100.0,
                "current_efficiency_percent": 85.0,
                "deposition_time_s": 900.0,
            },
            "derived": derived["derived"],
            "sources": derived["sources"],
            "result": derived_result.to_dict(),
        },
        "coupon_test_protocol": protocol,
        "findings": _findings(results, verdict, derived, derived_result),
        "figures": [
            f"docs/figures/{fig1.name}",
            f"docs/figures/{fig2.name}",
            f"docs/figures/{fig3.name}",
        ],
    }

    out = DATA / "adhesion_peel_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("Findings:")
    for f in report["findings"]:
        print(f"  - {f}")
    print()
    for p in (fig1, fig2, fig3):
        print(f"Wrote {p.relative_to(ROOT)}")
    print(f"Wrote {out.relative_to(ROOT)}")
    return report


def _findings(results, verdict, derived, derived_result) -> list:
    """Derive the narrative conclusions from the computed numbers."""
    findings = []
    ref = next(r for r in results if r.substrate_id == "ti_passive_tio2")

    findings.append(
        f"Reference drum surface (passive TiO₂) at "
        f"{ref.residual_stress_MPa:,.0f} MPa residual stress returns "
        f"'{ref.outcome}': peel force {ref.peel_force_N_per_m:,.1f} N/m "
        f"against a {MIN_CONTROLLABLE_TENSION_N_PER_M:,.0f}–"
        f"{MAX_WINDER_TENSION_N_PER_M:,.0f} N/m machine window."
    )

    peelable = [r for r in results if r.peelable and r.conductive]
    if peelable:
        findings.append(
            "Conductive surfaces inside the peel window: "
            + ", ".join(r.substrate_id for r in peelable) + "."
        )
    else:
        findings.append(
            "No conductive surface screened falls inside the controlled-peel "
            "window at these conditions."
        )

    bonded = [r for r in results if r.outcome == "bonded_no_release"]
    if bonded:
        findings.append(
            "Rejected as too strongly bonded to strip: "
            + ", ".join(r.substrate_id for r in bonded)
            + " — including the deliberate metallic negative controls, which "
            "is the check that the screen discriminates at all."
        )

    insulating = [r for r in results if not r.conductive]
    if insulating:
        findings.append(
            "Rejected on physics rather than omission: "
            + ", ".join(r.substrate_id for r in insulating)
            + " release beautifully and cannot pass current, so they cannot "
            "be a cathode."
        )

    findings.append(
        f"Verdict for the drum-and-strip branch: {verdict['verdict']} "
        f"(robust to the plastic-amplification uncertainty: "
        f"{verdict['verdict_robust_to_amplification']})."
    )

    d = derived["derived"]
    findings.append(
        f"Propagating a real operating point (100 mA/cm², 85% FE, 15 min) "
        f"through the existing models gives {d['thickness_um']:.0f} µm of "
        f"deposit carrying {d['C_H_diffusible_ppm']:.0f} ppm diffusible "
        f"hydrogen, and the deposit then returns "
        f"'{derived_result.outcome}' — hydrogen, not the substrate, dominates "
        f"the stress state at "
        f"{derived_result.stress_breakdown['hydrogen_MPa']:,.0f} MPa."
    )

    findings.append(
        f"Critical self-delamination thickness on the reference surface is "
        f"{ref.critical_thickness_um:,.0f} µm at the screen's baseline "
        f"conditions, which bounds foil thickness from above independently of "
        f"any winder capability."
    )

    findings.append(
        "The screen does not settle the question and is not meant to: it "
        "converts an unpriced unknown into a $1,750, 3-day coupon test with "
        "explicit kill and confirm rules (coupon_test_protocol)."
    )
    return findings


if __name__ == "__main__":
    main()
