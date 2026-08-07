"""Driver: surface-state HER chemistry — coverage, Frumkin, site-blocking.

This driver produces a screening report comparing the three
program-relevant baths (sulfate, AWARE chloride, mixed) under the
surface-state HER model (``models.surface_state``).  It is the
mechanism layer behind the "i0,H" screening knob — what the
empirical :class:`DepositionKinetics` calls ``her_i0`` is here
decomposed into a Temkin H-coverage factor, a Frumkin IHP
potential factor, and a site-blocking term that depends on the
competitive Langmuir adsorption of bath anions.

The headline i₀,H suppression ratio (sulfate / AWARE) is presented
as a *sensitivity band* rather than a single number.  The robust
prediction is the site-blocking component (~14x, calibration-free);
the Frumkin amplification depends on the screening parameter
``eta_screening`` and varies from ~2x (eta_screening=0.01) to
~17x (eta_screening=0.05, the screening central) to ~290x
(eta_screening=0.10, above the cited experimental range).  The
total ratio at eta_screening=0.05 is ~238x; at eta_screening=0.02
(in the cited range) it is ~44x.  See
``frumkin_sensitivity_sweep()`` and the PR discussion at
https://github.com/sblaplace/aqueous_electrowinning_steel/pull/50.

Generates:
    experiments/data/surface_state_report.json
    docs/figures/surface_state_coverage_vs_eta.png
    docs/figures/surface_state_bath_comparison.png
    docs/figures/surface_state_site_blocking.png
    docs/figures/surface_state_frumkin_sensitivity.png

Usage:
    python -m models.run_surface_state
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .kinetics import DepositionKinetics
from .surface_state import (
    SurfaceStateKinetics,
    chloride_aware_default,
    diagnostic_table,
    frumkin_sensitivity_sweep,
)

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "docs" / "figures"
DATA_DIR = ROOT / "experiments" / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 10})


def _eta_grid(V_min: float = 0.0, V_max: float = 0.5, n: int = 80) -> np.ndarray:
    return np.linspace(V_min, V_max, n)


def run_aware_vs_sulfate(base: DepositionKinetics) -> dict:
    """Compare three program-relevant baths at fixed T and base kinetics."""
    baths = {}
    for kind in ("sulfate", "aware", "mixed"):
        facets, anions = chloride_aware_default(kind)
        w = SurfaceStateKinetics(base=base, facets=facets, anion_coverages=anions)
        etas = _eta_grid()
        tab = diagnostic_table(base, etas, facets=facets, anion_coverages=anions)
        i0_eff = np.array([
            w.her_i0_corrected(float(e)) for e in etas
        ])
        baths[kind] = {
            "eta_V": tab["eta_V"].tolist(),
            "theta_H": tab["theta_H"].tolist(),
            "psi_1_V": tab["psi_1_V"].tolist(),
            "frumkin_factor": tab["frumkin_factor"].tolist(),
            "i0_H_effective_A_m2": i0_eff.tolist(),
            "facet_summary": facets.summary,
        }
    return baths


def _plot_bath_comparison(baths: dict, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    colors = {"sulfate": "#1f77b4", "aware": "#d62728", "mixed": "#2ca02c"}
    for kind, data in baths.items():
        c = colors[kind]
        eta = np.array(data["eta_V"])
        axes[0, 0].plot(eta, data["theta_H"], color=c, label=kind)
        axes[0, 1].plot(eta, data["psi_1_V"], color=c, label=kind)
        axes[1, 0].plot(eta, data["frumkin_factor"], color=c, label=kind)
        axes[1, 1].semilogy(eta, np.maximum(data["i0_H_effective_A_m2"], 1e-30),
                            color=c, label=kind)
    axes[0, 0].set_ylabel(r"$\theta_H$ (H coverage)")
    axes[0, 0].set_xlabel(r"$\eta$ (V)")
    axes[0, 0].set_title("H coverage (Temkin, mixed facets)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 1].set_ylabel(r"$\psi_1$ (V, IHP shift)")
    axes[0, 1].set_xlabel(r"$\eta$ (V)")
    axes[0, 1].set_title("Inner-Helmholtz potential (anion-down)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[1, 0].set_ylabel(r"Frumkin factor $e^{\alpha F \psi_1 / RT}$")
    axes[1, 0].set_xlabel(r"$\eta$ (V)")
    axes[1, 0].set_title("Frumkin HER suppression factor")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 1].set_ylabel(r"$i_{0,H}^{eff}$ (A/m²)")
    axes[1, 1].set_xlabel(r"$\eta$ (V)")
    axes[1, 1].set_title("Effective HER exchange current")
    axes[1, 1].grid(True, alpha=0.3, which="both")
    axes[1, 1].legend()
    fig.suptitle("Surface-state HER chemistry: bath comparison (60 °C)")
    fig.savefig(path)
    plt.close(fig)


def _plot_coverage_vs_eta(base: DepositionKinetics, path: Path) -> None:
    """Plot the H coverage, IHP potential, and i0 ratio for the sulfate
    bath only (the most-studied reference case)."""
    facets, anions = chloride_aware_default("sulfate")
    etas = _eta_grid()
    tab = diagnostic_table(base, etas, facets=facets, anion_coverages=anions)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(etas, tab["theta_H"], "b-", lw=2)
    axes[0].set_xlabel(r"$\eta$ (V)")
    axes[0].set_ylabel(r"$\theta_H$")
    axes[0].set_title("H coverage (Temkin)")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)
    axes[1].plot(etas, tab["psi_1_V"], "r-", lw=2)
    axes[1].set_xlabel(r"$\eta$ (V)")
    axes[1].set_ylabel(r"$\psi_1$ (V)")
    axes[1].set_title(r"Frumkin IHP potential, $\psi_1 < 0$")
    axes[1].grid(True, alpha=0.3)
    axes[2].semilogy(etas, np.maximum(tab["i0_H_effective_ratio"], 1e-30),
                     "k-", lw=2)
    axes[2].set_xlabel(r"$\eta$ (V)")
    axes[2].set_ylabel(r"$\theta_H (1-\theta_H)(1-\theta_{block}) e^{\alpha F \psi_1/RT}$")
    axes[2].set_title("Net effective $i_{0,H}$ ratio")
    axes[2].grid(True, alpha=0.3, which="both")
    fig.suptitle("Surface-state decomposition: sulfate bath (60 °C, screening L1)")
    fig.savefig(path)
    plt.close(fig)


def _plot_site_blocking(base: DepositionKinetics, path: Path) -> None:
    """Plot competitive site-blocking coverage for each bath.

    Bar chart of per-anion θ_block under the competitive-Langmuir
    isotherm for each of the three screening bath recipes.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    # Get the union of all anion short-names so the x-axis is the same
    # across the three bath sub-figures (use the 'mixed' bath — it
    # has all four anions).
    _, mixed_anions = chloride_aware_default("mixed")
    all_names = [a.anion.name.split(" ")[0] for a in mixed_anions]
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    width = 0.25
    offsets = {"sulfate": -0.25, "aware": 0.0, "mixed": 0.25}
    for kind, c in (("sulfate", "#1f77b4"), ("aware", "#d62728"),
                    ("mixed", "#2ca02c")):
        facets, anions = chloride_aware_default(kind)
        Kc_list = [a.K_eq_M_inv * a.c_bulk_M for a in anions]
        names = [a.anion.name.split(" ")[0] for a in anions]
        denom = 1.0 + sum(Kc_list)
        thetas = {n: kc / denom for n, kc in zip(names, Kc_list)}
        for n in all_names:
            t = thetas.get(n, 0.0)
            x = name_to_idx[n]
            ax.bar(x + offsets[kind], t, width=width, color=c, alpha=0.7)
            if t > 0.01:
                ax.text(x + offsets[kind], t + 0.01, f"{t:.2f}",
                        ha="center", fontsize=7, color=c)
    ax.set_xticks(np.arange(len(all_names)))
    ax.set_xticklabels(all_names, fontsize=8)
    ax.set_ylabel(r"Competitive $\theta_{anion}$ (per-species)")
    ax.set_title("Anion site-blocking coverage (competitive Langmuir, 60 °C)")
    ax.legend([plt.Rectangle((0,0), 1, 1, color=c) for c in
               ("#1f77b4", "#d62728", "#2ca02c")],
              ["sulfate", "aware", "mixed"])
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(path)
    plt.close(fig)


def _plot_frumkin_sensitivity(base: DepositionKinetics, path: Path) -> None:
    """Plot the sulfate/AWARE i0 ratio vs eta_screening.

    The figure has two panels: the *total* ratio on a log scale (with
    the cited psi_1 range marked) and the *decomposition* into
    site-blocking and Frumkin components.
    """
    sweep = frumkin_sensitivity_sweep(
        base, "sulfate", "aware", eta=0.2,
        eta_screening_values=(0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.20),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    es = sweep["eta_screening"]
    rt = sweep["ratio_total"]
    rs = sweep["ratio_site_blocking_only"]
    rf = sweep["ratio_frumkin_only"]
    # Panel 1: total ratio
    axes[0].semilogy(es, np.maximum(rt, 1e-30), "k-o", lw=2,
                     label="Total ratio (sulfate / AWARE)")
    axes[0].semilogy(es, np.maximum(rs, 1e-30), "b--s",
                     label=r"Site-blocking only (robust $\sim$14x)")
    axes[0].semilogy(es, np.maximum(rf, 1e-30), "r--^",
                     label="Frumkin factor ratio only")
    # Mark the cited experimental range.
    axes[0].axvspan(0.005, 0.06, alpha=0.15, color="green",
                    label="Cited $\\psi_1$ range (-0.05 to -0.3 V)")
    axes[0].axvline(0.05, color="green", linestyle=":", alpha=0.5,
                    label="Screening central (eta=0.05)")
    axes[0].set_xlabel(r"$\eta_{screening}$ (Frumkin screening parameter)")
    axes[0].set_ylabel(r"Ratio $i_{0,H}^{sulfate} / i_{0,H}^{AWARE}$")
    axes[0].set_title("Sensitivity band (eta=0.2 V, 60 °C)")
    axes[0].grid(True, alpha=0.3, which="both")
    axes[0].legend(loc="upper left", fontsize=8)
    # Panel 2: psi_1 values for each bath vs eta_screening
    axes[1].plot(es, sweep["psi_1_bath_a"], "b-o", label="sulfate")
    axes[1].plot(es, sweep["psi_1_bath_b"], "r-s", label="AWARE")
    axes[1].axhspan(-0.30, -0.05, alpha=0.15, color="green",
                    label="Cited -0.05 to -0.3 V")
    axes[1].axhline(-0.30, color="green", linestyle=":", alpha=0.5)
    axes[1].axhline(-0.05, color="green", linestyle=":", alpha=0.5)
    axes[1].set_xlabel(r"$\eta_{screening}$")
    axes[1].set_ylabel(r"$\psi_1$ (V)")
    axes[1].set_title(r"$\psi_1$ vs screening parameter (both baths)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower left", fontsize=8)
    fig.suptitle("Frumkin sensitivity: site-blocking (robust) vs Frumkin (calibration-sensitive)")
    fig.savefig(path)
    plt.close(fig)


def main() -> dict:
    base = DepositionKinetics(
        pH=2.0, temperature_C=60.0,
        fe_i0=1.0e-2, her_i0=1.0e-3,
    )
    baths = run_aware_vs_sulfate(base)
    sensitivity = frumkin_sensitivity_sweep(
        base, "sulfate", "aware", eta=0.2,
        eta_screening_values=(0.0, 0.01, 0.02, 0.05, 0.10),
    )
    _plot_coverage_vs_eta(base, FIG_DIR / "surface_state_coverage_vs_eta.png")
    _plot_bath_comparison(baths, FIG_DIR / "surface_state_bath_comparison.png")
    _plot_site_blocking(base, FIG_DIR / "surface_state_site_blocking.png")
    _plot_frumkin_sensitivity(base, FIG_DIR / "surface_state_frumkin_sensitivity.png")
    print(f"  Saved {FIG_DIR / 'surface_state_coverage_vs_eta.png'}")
    print(f"  Saved {FIG_DIR / 'surface_state_bath_comparison.png'}")
    print(f"  Saved {FIG_DIR / 'surface_state_site_blocking.png'}")
    print(f"  Saved {FIG_DIR / 'surface_state_frumkin_sensitivity.png'}")

    # The honest headline: the site-blocking component is robust;
    # the Frumkin amplification is a sensitivity band.
    sb_only = float(sensitivity["ratio_site_blocking_only"].mean())
    rt_at_005 = float(
        sensitivity["ratio_total"][
            int(np.argmin(np.abs(sensitivity["eta_screening"] - 0.05)))
        ]
    )
    rt_at_002 = float(
        sensitivity["ratio_total"][
            int(np.argmin(np.abs(sensitivity["eta_screening"] - 0.02)))
        ]
    )
    headline = {
        "site_blocking_ratio": sb_only,
        "total_ratio_at_eta_screening_0p05": rt_at_005,
        "total_ratio_at_eta_screening_0p02": rt_at_002,
        "frumkin_amplification_band_2_to_17x": True,
        "robust_prediction": (
            f"Site-blocking component is the robust prediction: "
            f"i0,H(sulfate) / i0,H(AWARE) ~ {sb_only:.0f}x, "
            f"independent of the Frumkin screening parameter."
        ),
        "calibration_caveat": (
            "Total ratio is the site-blocking component times the Frumkin "
            "amplification exp(alpha*F*Delta_psi/RT), where Delta_psi is "
            "the difference in psi_1 between the two baths.  The cited "
            "experimental range for psi_1 on Fe(110) is -0.05 to -0.3 V; "
            "the screening central value (eta_screening=0.05) sits at the "
            "upper end of this range.  See the sensitivity figure for the "
            "full band."
        ),
    }

    # Report
    report = {
        "method_scope": {
            "model": "SurfaceStateKinetics wrapper over DepositionKinetics",
            "scope": "Decomposes apparent i0,H into theta_H*(1-theta_H) (Temkin), "
                     "(1-theta_block) (competitive Langmuir), and exp(alpha*F*psi_1/RT) "
                     "(Frumkin).  Screening central values for bath-anion adsorption "
                     "thermodynamics (DG_ads, mu_z, C_ihp).  Tier-1.1 chemistry add.",
            "validity": "All screening central values; production code should pin "
                        "each adsorption constant against a fitted isotherm.",
            "baths": "sulfate (1 M FeSO4 + 0.5 M Na2SO4 + 0.4 M H3BO3), "
                     "aware (1 M FeCl2 + 10 M LiCl, pH 2), mixed.",
        },
        "base_kinetics": {
            "pH": base.pH,
            "T_C": base.temperature_C,
            "fe_i0_A_m2": base.fe_i0,
            "her_i0_A_m2": base.her_i0,
        },
        "headline": headline,
        "bath_comparison": baths,
        "frumkin_sensitivity": {
            "eta_screening": sensitivity["eta_screening"].tolist(),
            "ratio_total": sensitivity["ratio_total"].tolist(),
            "ratio_site_blocking_only": sensitivity["ratio_site_blocking_only"].tolist(),
            "ratio_frumkin_only": sensitivity["ratio_frumkin_only"].tolist(),
            "psi_1_sulfate": sensitivity["psi_1_bath_a"].tolist(),
            "psi_1_aware": sensitivity["psi_1_bath_b"].tolist(),
            "eta_V": sensitivity["eta_V"],
        },
        "screening_flag": "unvalidated (L1)",
    }
    report_path = DATA_DIR / "surface_state_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"  Saved {report_path}")
    print("✅ Surface-state HER driver complete.")
    return report


if __name__ == "__main__":
    main()
