"""Driver: Fe2+/Cl- speciation — AWARE chloride-route bath solver.

This driver produces a screening report and figures comparing the
sulfate, AWARE, and historical Chinese chloride baths under the
Fe-Cl speciation model (``models.fe_chloride_speciation``).  The
report covers the four program-relevant diagnostics:

  1. γ±(FeCl₂) vs [LiCl] at the AWARE operating point.
  2. Bjerrum FeCl+/FeCl₂(aq)/FeCl₃⁻ species distribution vs
     bulk [Cl⁻] (the AWARE paper's central speciation claim).
  3. Bath conductivity vs [LiCl] (the AWARE paper's >99% FE
     depends on a high-conductivity cell).
  4. Nernst E_rev(Fe) vs [Cl⁻] (the chloride-bath penalty that
     the AWARE paper's high V_cell accounts for).

Generates:
    experiments/data/fe_chloride_speciation_report.json
    docs/figures/chloride_gamma_vs_LiCl.png
    docs/figures/chloride_species_distribution.png
    docs/figures/chloride_conductivity.png
    docs/figures/chloride_nernst.png

Usage:
    python -m models.run_fe_chloride_speciation
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .fe_chloride_speciation import (
    aware_default_bath,
    historical_chinese_iron_bath,
    solve_chloride_speciation,
    ChlorideBathComposition,
)

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "docs" / "figures"
DATA_DIR = ROOT / "experiments" / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 10})


def sweep_liCl(c_FeCl2: float = 1.0, c_HCl: float = 0.01,
               T_C: float = 60.0,
               c_LiCl_values: list = None) -> list:
    if c_LiCl_values is None:
        c_LiCl_values = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    out = []
    for c_LiCl in c_LiCl_values:
        comp = ChlorideBathComposition(
            c_FeCl2=c_FeCl2, c_LiCl=c_LiCl, c_NaCl=0.0,
            c_HCl=c_HCl, T_C=T_C,
        )
        s = solve_chloride_speciation(comp)
        out.append({
            "c_LiCl_M": c_LiCl,
            "ionic_strength_molal": s["ionic_strength_molal"],
            "gamma_pm_FeCl2": s["gamma_pm_FeCl2"],
            "a_Fe2": s["a_Fe2"],
            "a_Cl": s["a_Cl"],
            "c_FeCl_plus_M": s["c_FeCl_plus_M"],
            "c_FeCl2_aq_M": s["c_FeCl2_aq_M"],
            "c_FeCl3_minus_M": s["c_FeCl3_minus_M"],
            "fecl_plus_fraction": s["fecl_plus_fraction"],
            "fecl2_aq_fraction": s["fecl2_aq_fraction"],
            "fecl3_minus_fraction": s["fecl3_minus_fraction"],
            "water_activity": s["water_activity"],
            "conductivity_S_m": s["conductivity_S_m"],
            "E_rev_Fe_V_SHE": s["E_rev_Fe_V_SHE"],
            "pH_activity": s["pH_activity"],
        })
    return out


def _plot_gamma_vs_LiCl(sweep: list, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    c_LiCl = [s["c_LiCl_M"] for s in sweep]
    gamma = [s["gamma_pm_FeCl2"] for s in sweep]
    I = [s["ionic_strength_molal"] for s in sweep]
    axes[0].plot(c_LiCl, gamma, "b-o")
    axes[0].set_xlabel("[LiCl] (M)")
    axes[0].set_ylabel(r"$\gamma_\pm$(FeCl₂)")
    axes[0].set_title("Pitzer mean ionic activity coefficient")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(c_LiCl, I, "k-d")
    axes[1].set_xlabel("[LiCl] (M)")
    axes[1].set_ylabel("Ionic strength (molal)")
    axes[1].set_title("Ionic strength vs LiCl loading")
    axes[1].grid(True, alpha=0.3)
    # Mark the AWARE bath point.
    for ax in axes:
        ax.axvline(10.0, color="r", linestyle="--", alpha=0.5, label="AWARE")
        ax.legend()
    fig.suptitle("Fe²⁺/Cl⁻ Pitzer: 1 M FeCl₂, 60 °C, screening L1")
    fig.savefig(path)
    plt.close(fig)


def _plot_species_distribution(sweep: list, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    c_LiCl = [s["c_LiCl_M"] for s in sweep]
    fecl_plus = [s["fecl_plus_fraction"] for s in sweep]
    fecl2_aq = [s["fecl2_aq_fraction"] for s in sweep]
    fecl3_minus = [s["fecl3_minus_fraction"] for s in sweep]
    fe2_free = [1.0 - s["fecl_plus_fraction"] - s["fecl2_aq_fraction"]
                - s["fecl3_minus_fraction"] for s in sweep]
    ax.stackplot(c_LiCl, fe2_free, fecl_plus, fecl2_aq, fecl3_minus,
                 labels=[r"Fe$^{2+}$ free", r"FeCl$^+$",
                         r"FeCl$_2$(aq)", r"FeCl$_3^-$"],
                 colors=["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c"], alpha=0.8)
    ax.set_xlabel("[LiCl] (M)")
    ax.set_ylabel("Fraction of total Fe")
    ax.set_title("Bjerrum speciation vs [LiCl] (1 M FeCl₂, 60 °C)")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.savefig(path)
    plt.close(fig)


def _plot_conductivity(sweep: list, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    c_LiCl = [s["c_LiCl_M"] for s in sweep]
    cond = [s["conductivity_S_m"] for s in sweep]
    ax.plot(c_LiCl, cond, "g-s", lw=2, label="AWARE-type bath")
    # Reference value from AWARE paper.
    ax.axhline(20.0, color="r", linestyle="--", alpha=0.5,
               label="AWARE paper reported value (~20 S/m)")
    ax.set_xlabel("[LiCl] (M)")
    ax.set_ylabel("Conductivity (S/m)")
    ax.set_title("Bath conductivity vs [LiCl] (1 M FeCl₂, 60 °C)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def _plot_nernst(sweep: list, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    c_LiCl = [s["c_LiCl_M"] for s in sweep]
    E = [s["E_rev_Fe_V_SHE"] for s in sweep]
    ax.plot(c_LiCl, E, "m-^", lw=2)
    ax.set_xlabel("[LiCl] (M)")
    ax.set_ylabel(r"$E_{rev}$(Fe²⁺/Fe) vs SHE (V)")
    ax.set_title("Chloride-bath Nernst penalty (1 M FeCl₂, 60 °C)")
    ax.grid(True, alpha=0.3)
    fig.savefig(path)
    plt.close(fig)


def main() -> dict:
    sweep = sweep_liCl()

    _plot_gamma_vs_LiCl(sweep, FIG_DIR / "chloride_gamma_vs_LiCl.png")
    _plot_species_distribution(sweep, FIG_DIR / "chloride_species_distribution.png")
    _plot_conductivity(sweep, FIG_DIR / "chloride_conductivity.png")
    _plot_nernst(sweep, FIG_DIR / "chloride_nernst.png")
    print(f"  Saved {FIG_DIR / 'chloride_gamma_vs_LiCl.png'}")
    print(f"  Saved {FIG_DIR / 'chloride_species_distribution.png'}")
    print(f"  Saved {FIG_DIR / 'chloride_conductivity.png'}")
    print(f"  Saved {FIG_DIR / 'chloride_nernst.png'}")

    aware = solve_chloride_speciation(aware_default_bath())
    historical = solve_chloride_speciation(historical_chinese_iron_bath())

    report = {
        "method_scope": {
            "model": "Fe²⁺/Cl⁻ Pitzer multicomponent + Bjerrum speciation",
            "scope": "Pitzer binary pairs: (Fe2+, Cl-), (H+, Cl-), (Li+, Cl-), "
                     "(Li+, SO4(2-)).  Bjerrum FeCl+/FeCl2(aq)/FeCl3- with "
                     "van't Hoff T-correction and SIT-style I-correction.  "
                     "Tier-1.4 chemistry add.",
            "validity": "Pitzer valid 0-50 °C and I < 6 m.  AWARE bath at I=14 m "
                        "is OUT OF PITZER RANGE (output dict flags this).  "
                        "Production code should pin against Christov & Moller "
                        "Fe-Cl-water T-functions.",
        },
        "liCl_sweep_1M_FeCl2_60C": sweep,
        "aware_bath_screening": aware,
        "historical_chinese_bath_screening": historical,
        "screening_flag": "unvalidated (L1)",
    }
    report_path = DATA_DIR / "fe_chloride_speciation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                       default=lambda o: float(o)) + "\n")
    print(f"  Saved {report_path}")
    print("✅ Fe²⁺/Cl⁻ speciation driver complete.")
    return report


if __name__ == "__main__":
    main()
