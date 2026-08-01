"""Deposit internal stress (Stoney / bent-strip) model: figures + JSON report.

Answers the Missing Physics gap item 7 from ``docs/RESEARCH_PROGRAM.md``:
predicts internal residual stress from deposition conditions (intrinsic,
hydrogen, and thermal mismatch), evaluates bent-strip cantilever deflection and
its GUM uncertainty budget, and connects the stress state to the drum-and-strip
adhesion and peel model.

Run::

    python -m models.run_internal_stress
    aq-steel-stress
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from .internal_stress import (
    COUPON_E_GPA,
    COUPON_LENGTH_MM,
    COUPON_NU,
    COUPON_THICKNESS_MM,
    DIAL_GAUGE_RESOLUTION_UM,
    PROFILOMETER_RESOLUTION_UM,
    cantilever_deflection_m,
    coupon_curvature_protocol,
    deposit_stress_from_conditions,
    finite_thickness_correction,
    model_scope,
    peel_verdict_from_conditions,
    stoney_stress_finite_thickness_MPa,
    stress_evolution,
    stress_profile,
    stress_uncertainty_MPa,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments" / "data"
FIGURES = ROOT / "docs" / "figures"

COLORS = {
    "intrinsic": "#2b5c8f",
    "hydrogen": "#d95f02",
    "thermal": "#7570b3",
    "total": "#1b9e77",
    "dc": "#d95f02",
    "pre": "#1b9e77",
    "saccharin": "#7570b3",
}

OUTCOME_COLORS = {
    "bonded_no_release": "#d73027",
    "cohesive_failure_in_film": "#fc8d59",
    "tears_before_peel": "#fee090",
    "clean_peel": "#1a9850",
    "marginal_peel": "#91cf60",
    "spontaneous_delamination": "#4575b4",
}


def _fig_mechanism_decomposition() -> Path:
    """Figure 1: Mechanism decomposition vs current density and additives."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    # Left panel: Stacked stress contributions across current densities
    ax = axes[0]
    js = np.linspace(50, 400, 36)
    intrinsic_vals = []
    hydrogen_vals = []
    thermal_vals = []
    total_vals = []
    for j in js:
        res = deposit_stress_from_conditions(
            j_mA_cm2=float(j),
            current_efficiency_percent=85.0,
            deposition_time_s=900.0,
        )
        comp = res["components"]
        intrinsic_vals.append(comp["intrinsic_MPa"])
        hydrogen_vals.append(comp["hydrogen_MPa"])
        thermal_vals.append(comp["thermal_MPa"])
        total_vals.append(comp["total_MPa"])

    ax.plot(js, total_vals, color="black", lw=2, label="Total stress")
    ax.stackplot(
        js,
        intrinsic_vals,
        hydrogen_vals,
        thermal_vals,
        labels=["Intrinsic (Hoffman)", "Hydrogen (effusion)", "Thermal mismatch"],
        colors=[COLORS["intrinsic"], COLORS["hydrogen"], COLORS["thermal"]],
        alpha=0.8,
    )
    ax.set_xlabel("Current density (mA/cm²)")
    ax.set_ylabel("Residual stress (MPa)")
    ax.set_title("Stress decomposition (85% FE, 60 °C, TiO₂ substrate)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    # Right panel: Additive relief (saccharin & chloride)
    ax = axes[1]
    sac_concs = np.linspace(0, 3.0, 50)
    dc_stress = []
    chloride_stress = []
    for s in sac_concs:
        res_dc = deposit_stress_from_conditions(
            j_mA_cm2=100.0,
            saccharin_g_L=float(s),
            chloride_bath=False,
        )
        res_cl = deposit_stress_from_conditions(
            j_mA_cm2=100.0,
            saccharin_g_L=float(s),
            chloride_bath=True,
        )
        dc_stress.append(res_dc["components"]["total_MPa"])
        chloride_stress.append(res_cl["components"]["total_MPa"])

    ax.plot(sac_concs, dc_stress, color=COLORS["intrinsic"], lw=2, label="Sulfate bath (baseline)")
    ax.plot(sac_concs, chloride_stress, color=COLORS["hydrogen"], lw=2, linestyle="--", label="Chloride bath (−30 MPa shift)")
    ax.axhline(0, color="gray", linestyle=":", lw=1)
    ax.set_xlabel("Saccharin concentration (g/L)")
    ax.set_ylabel("Total residual stress (MPa)")
    ax.set_title("Additive stress relief at 100 mA/cm²")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Internal stress mechanism decomposition and chemical relief",
        fontweight="bold",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIGURES / "internal_stress_mechanism_decomposition.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _fig_coupon_measurability() -> Path:
    """Figure 2: Cantilever deflection measurability and GUM uncertainty."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    # Left panel: Deflection vs film thickness for 0.2 mm and 0.4 mm shims
    ax = axes[0]
    h_film_um = np.linspace(5, 60, 50)
    for stress in [100.0, 200.0, 400.0]:
        def_04 = [
            cantilever_deflection_m(
                stress, film_thickness_m=h * 1e-6, substrate_thickness_m=0.4e-3
            )
            * 1e6
            for h in h_film_um
        ]
        def_02 = [
            cantilever_deflection_m(
                stress, film_thickness_m=h * 1e-6, substrate_thickness_m=0.2e-3
            )
            * 1e6
            for h in h_film_um
        ]
        ax.plot(h_film_um, def_04, lw=1.8, label=f"{stress:.0f} MPa (0.4 mm shim)")
        ax.plot(h_film_um, def_02, lw=1.3, linestyle="--", label=f"{stress:.0f} MPa (0.2 mm shim)")

    ax.axhline(DIAL_GAUGE_RESOLUTION_UM, color="red", linestyle=":", label=f"Dial gauge floor ({DIAL_GAUGE_RESOLUTION_UM:.0f} µm)")
    ax.axhline(PROFILOMETER_RESOLUTION_UM, color="purple", linestyle=":", label=f"Stylus profilometer floor ({PROFILOMETER_RESOLUTION_UM:.0f} µm)")
    ax.set_yscale("log")
    ax.set_xlabel("Deposit film thickness (µm)")
    ax.set_ylabel("Cantilever free-end deflection δ (µm)")
    ax.set_title("Stoney cantilever deflection (L = 60 mm, 316L shim)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=7, ncol=2)

    # Right panel: GUM relative uncertainty budget vs thickness
    ax = axes[1]
    rel_unc_04 = []
    rel_unc_02 = []
    for h in h_film_um:
        d_04 = cantilever_deflection_m(
            250.0, film_thickness_m=h * 1e-6, substrate_thickness_m=0.4e-3
        )
        u_04 = stress_uncertainty_MPa(
            d_04,
            u_deflection_m=10e-6,
            film_thickness_m=h * 1e-6,
            substrate_thickness_m=0.4e-3,
        )
        rel_unc_04.append(u_04["relative_uncertainty"] * 100.0)

        d_02 = cantilever_deflection_m(
            250.0, film_thickness_m=h * 1e-6, substrate_thickness_m=0.2e-3
        )
        u_02 = stress_uncertainty_MPa(
            d_02,
            u_deflection_m=10e-6,
            film_thickness_m=h * 1e-6,
            substrate_thickness_m=0.2e-3,
        )
        rel_unc_02.append(u_02["relative_uncertainty"] * 100.0)

    ax.plot(h_film_um, rel_unc_04, color="black", lw=2, label="0.4 mm shim (dial gauge ±10 µm)")
    ax.plot(h_film_um, rel_unc_02, color="tab:blue", lw=2, linestyle="--", label="0.2 mm shim (dial gauge ±10 µm)")
    ax.axhline(5.0, color="green", linestyle=":", label="5% target uncertainty threshold")
    ax.set_xlabel("Deposit film thickness (µm)")
    ax.set_ylabel("Combined standard uncertainty u(σ) / σ (%)")
    ax.set_title("GUM standard uncertainty budget (250 MPa nominal stress)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Bent-strip experimental measurability and GUM error propagation",
        fontweight="bold",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIGURES / "internal_stress_coupon_measurability.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _fig_evolution_and_peel() -> Path:
    """Figure 3: Stress evolution profile and peel-verdict map."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    # Left panel: Stress profile local vs Stoney-averaged
    ax = axes[0]
    prof = stress_profile(
        plateau_stress_MPa=400.0,
        h_max_um=50.0,
        n_points=200,
        interface_stress_MPa=-40.0,
        characteristic_thickness_um=10.0,
    )
    ax.plot(prof["h_um"], prof["local_MPa"], color="tab:red", lw=2, label="Local stress σ_loc(h)")
    ax.plot(prof["h_um"], prof["average_MPa"], color="tab:blue", lw=2, linestyle="--", label="Stoney measured average ⟨σ⟩(h)")
    ax.axhline(400.0, color="gray", linestyle=":", label="Plateau stress (+400 MPa)")
    ax.set_xlabel("Deposit film thickness (µm)")
    ax.set_ylabel("Stress (MPa)")
    ax.set_title("Thickness evolution (interfacial transient −40 MPa)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    # Right panel: Peel verdict comparison across plating regimes
    ax = axes[1]
    cases = [
        ("DC baseline\n(100 mA/cm²)", {}),
        ("PRE pulse\n(30% duty)", {"waveform": "pre", "duty_cycle": 0.3, "j_peak_mA_cm2": 333.3, "current_efficiency_percent": 95.0}),
        ("Saccharin\n(1.5 g/L)", {"saccharin_g_L": 1.5}),
        ("Chloride + Sac.\n(2.0 g/L)", {"chloride_bath": True, "saccharin_g_L": 2.0}),
    ]

    labels = []
    stresses = []
    colors = []
    outcomes = []
    for label, kw in cases:
        pv = peel_verdict_from_conditions(**kw)
        labels.append(label)
        stresses.append(pv["stress"]["components"]["total_MPa"])
        outcome = pv["peel"]["outcome"]
        outcomes.append(outcome)
        colors.append(OUTCOME_COLORS.get(outcome, "gray"))

    bars = ax.bar(labels, stresses, color=colors, edgecolor="black", width=0.55)
    ax.set_ylabel("Total deposit residual stress (MPa)")
    ax.set_title("Plating regime impact on deposit stress and peel outcome")
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate bars with outcome text
    for bar, outcome in zip(bars, outcomes):
        height = bar.get_height()
        ax.annotate(
            outcome.replace("_", " "),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )

    # Extend Y-axis limit for annotations
    ax.set_ylim(0, max(stresses) * 1.18)

    fig.suptitle(
        "Stress evolution through deposit thickness and drum peel integration",
        fontweight="bold",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIGURES / "internal_stress_evolution_and_peel.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _findings(
    ref_stress: dict,
    verdicts: dict,
    protocol: dict,
    unc_ref: dict,
) -> list[str]:
    """Derive narrative findings from the computed internal stress results."""
    findings = []
    comp = ref_stress["components"]
    findings.append(
        f"Baseline DC electrowinning (100 mA/cm², 85% FE, 60 °C) produces a total "
        f"tensile residual stress of {comp['total_MPa']:,.1f} MPa on reference TiO₂ "
        f"drum cathodes, with hydrogen effusion contributing "
        f"{comp['hydrogen_MPa']:,.1f} MPa ({comp['hydrogen_MPa']/comp['total_MPa']:.0%} "
        f"of the total stress)."
    )

    findings.append(
        f"Because hydrogen dominance ({comp['hydrogen_MPa']:,.1f} MPa) overwhelms "
        f"the interfacial fracture toughness of passive oxides, the drum-and-strip "
        f"verdict for DC baseline electrowinning is "
        f"'{verdicts['dc']['peel']['outcome']}'."
    )

    pre = verdicts["pre"]["stress"]["components"]
    findings.append(
        f"Pulse Reverse Electrowinning (PRE) at 95% efficiency reduces hydrogen "
        f"effusion stress to {pre['hydrogen_MPa']:,.1f} MPa, lowering total deposit "
        f"stress to {pre['total_MPa']:,.1f} MPa — demonstrating that pulse waveforms "
        f"can mitigate hydrogen embrittlement and internal stress."
    )

    sac = verdicts["saccharin"]["stress"]["components"]
    findings.append(
        f"Saccharin additive (1.5 g/L) relieves intrinsic grain-boundary stress, "
        f"reducing the intrinsic contribution from {comp['intrinsic_MPa']:,.1f} MPa "
        f"to {sac['intrinsic_MPa']:,.1f} MPa, while chloride baths introduce a "
        f"−30 MPa compressive shift."
    )

    findings.append(
        f"On a standard 0.4 mm 316L coupon (L = {COUPON_LENGTH_MM:.0f} mm), a 25 µm "
        f"deposit at 250 MPa induces a free-end cantilever deflection of "
        f"{unc_ref['deflection_um']:.1f} µm, well above the {DIAL_GAUGE_RESOLUTION_UM:.0f} µm "
        f"dial-gauge resolution floor and yielding a relative standard "
        f"uncertainty of {unc_ref['relative_uncertainty']*100:.1f}%."
    )

    findings.append(
        f"The coupon curvature protocol ({protocol['title']}) specifies a "
        f"${protocol['budget_usd']['total']:.0f} experiment across "
        f"{len(protocol['coupons'])} coupons with explicit kill/confirm rules to "
        f"replace screening estimates with measured σ(h) data."
    )
    return findings


def main() -> dict:
    """Run deposit internal stress model, print findings, and write report."""
    print("=== Deposit Internal Stress (Stoney / Bent-Strip) Model ===")
    print()

    ref_stress = deposit_stress_from_conditions(
        j_mA_cm2=100.0,
        current_efficiency_percent=85.0,
        deposition_time_s=900.0,
    )
    print("Baseline DC stress decomposition (100 mA/cm², 85% FE, 15 min):")
    for k, v in ref_stress["components"].items():
        print(f"  {k}: {v:,.1f} MPa")
    print(f"  Dominant mechanism: {ref_stress['dominant_mechanism']}")
    print()

    verdicts = {
        "dc": peel_verdict_from_conditions(
            j_mA_cm2=100.0, current_efficiency_percent=85.0, deposition_time_s=900.0
        ),
        "pre": peel_verdict_from_conditions(
            waveform="pre",
            duty_cycle=0.3,
            j_peak_mA_cm2=333.3,
            current_efficiency_percent=95.0,
            deposition_time_s=900.0,
        ),
        "saccharin": peel_verdict_from_conditions(
            j_mA_cm2=100.0,
            current_efficiency_percent=85.0,
            deposition_time_s=900.0,
            saccharin_g_L=1.5,
        ),
    }

    print("Peel verdict comparison:")
    for key, val in verdicts.items():
        print(
            f"  {key.upper()}: stress = {val['stress']['components']['total_MPa']:,.1f} MPa "
            f"-> {val['peel']['outcome']}"
        )
    print()

    # Reference uncertainty evaluation at 250 MPa, 25 um film on 0.4 mm shim
    d_ref = cantilever_deflection_m(
        250.0, film_thickness_m=25e-6, substrate_thickness_m=0.4e-3
    )
    unc_ref = stress_uncertainty_MPa(
        d_ref,
        u_deflection_m=10e-6,
        film_thickness_m=25e-6,
        substrate_thickness_m=0.4e-3,
    )
    unc_ref["deflection_um"] = d_ref * 1e6

    protocol = coupon_curvature_protocol()

    fig1 = _fig_mechanism_decomposition()
    fig2 = _fig_coupon_measurability()
    fig3 = _fig_evolution_and_peel()

    findings = _findings(ref_stress, verdicts, protocol, unc_ref)

    report = {
        "title": "Deposit internal stress (Stoney / bent-strip) report",
        "purpose": (
            "Predict internal residual stress from deposition conditions "
            "(intrinsic, hydrogen, thermal), evaluate bent-strip cantilever "
            "deflection and GUM uncertainty budget, and connect stress state to "
            "drum-and-strip adhesion and peel."
        ),
        "model_scope": model_scope(),
        "reference_case": ref_stress,
        "peel_verdicts": verdicts,
        "uncertainty_budget_reference": unc_ref,
        "coupon_curvature_protocol": protocol,
        "findings": findings,
        "figures": [
            f"docs/figures/{fig1.name}",
            f"docs/figures/{fig2.name}",
            f"docs/figures/{fig3.name}",
        ],
    }

    out = DATA / "internal_stress_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("Findings:")
    for f in findings:
        print(f"  - {f}")
    print()
    for p in (fig1, fig2, fig3):
        print(f"Wrote {p.relative_to(ROOT)}")
    print(f"Wrote {out.relative_to(ROOT)}")
    return report


if __name__ == "__main__":
    main()
