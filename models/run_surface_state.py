"""Driver: surface-state HER chemistry — coverage, Frumkin, site-blocking.

This driver produces a screening report comparing the three
program-relevant baths (sulfate, AWARE chloride, mixed) under the
surface-state HER model (``models.surface_state``).  It is the
mechanism layer behind the "i0,H" screening knob — what the
empirical :class:`DepositionKinetics` calls ``her_i0`` is here
decomposed into a Temkin H-coverage factor, a Frumkin IHP
potential factor, and a site-blocking term that depends on the
competitive Langmuir adsorption of bath anions.

Generates:
    experiments/data/surface_state_report.json
    docs/figures/surface_state_coverage_vs_eta.png
    docs/figures/surface_state_bath_comparison.png
    docs/figures/surface_state_site_blocking.png

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


def main() -> dict:
    base = DepositionKinetics(
        pH=2.0, temperature_C=60.0,
        fe_i0=1.0e-2, her_i0=1.0e-3,
    )
    baths = run_aware_vs_sulfate(base)
    _plot_coverage_vs_eta(base, FIG_DIR / "surface_state_coverage_vs_eta.png")
    _plot_bath_comparison(baths, FIG_DIR / "surface_state_bath_comparison.png")
    _plot_site_blocking(base, FIG_DIR / "surface_state_site_blocking.png")
    print(f"  Saved {FIG_DIR / 'surface_state_coverage_vs_eta.png'}")
    print(f"  Saved {FIG_DIR / 'surface_state_bath_comparison.png'}")
    print(f"  Saved {FIG_DIR / 'surface_state_site_blocking.png'}")

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
        "bath_comparison": baths,
        "screening_flag": "unvalidated (L1)",
    }
    report_path = DATA_DIR / "surface_state_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"  Saved {report_path}")
    print("✅ Surface-state HER driver complete.")
    return report


if __name__ == "__main__":
    main()
