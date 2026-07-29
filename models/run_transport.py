#!/usr/bin/env python3
"""
Nernst-Planck transport driver: migration effects in the cathode film.

Generates:
    docs/figures/nernst_planck_profiles.png
    docs/figures/migration_enhancement.png
    docs/figures/transport_model_comparison.png
    experiments/data/transport_report.json

Usage:
    python -m models.run_transport
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.boundary_layer import CathodeBoundaryLayer  # noqa: E402
from models.transport import NernstPlanckFilm, compare_support_levels  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "docs" / "figures"
DATA_DIR = ROOT / "experiments" / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 10})


# ─── Figure 1: species and potential profiles ─────────────────────────
def plot_profiles(j_mA_cm2: float = 100.0):
    """Concentration, pH and potential profiles with and without support."""
    bare = NernstPlanckFilm(bulk_pH=2.0, her_i0=1e-4, support_conc_M=0.0)
    salty = NernstPlanckFilm(bulk_pH=2.0, her_i0=1e-4, support_conc_M=2.0)
    s_bare = bare.solve(j_mA_cm2)
    s_salty = salty.solve(j_mA_cm2)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    for state, film, color, label in (
        (s_bare, bare, "#c62828", "unsupported"),
        (s_salty, salty, "#1565c0", "2 M Na₂SO₄"),
    ):
        p = state.profile
        x_um = p.x_m * 1e6
        axes[0].plot(x_um, p.fe_M, color=color, lw=2, label=label)
        axes[1].plot(x_um, p.pH, color=color, lw=2, label=label)
        axes[2].plot(x_um, p.potential_V * 1000, color=color, lw=2, label=label)

    axes[0].axhline(bare.fe_conc_M, color="#616161", ls=":", lw=1,
                    label="bulk Fe²⁺")
    axes[0].set_ylabel("Fe²⁺ (M)")
    axes[0].set_title("Iron depletion")

    axes[1].axhline(2.0, color="#616161", ls=":", lw=1, label="bulk pH")
    axes[1].set_ylabel("pH")
    axes[1].set_title("Local alkalinisation")

    axes[2].axhline(0.0, color="#616161", ls=":", lw=1)
    axes[2].set_ylabel("φ − φ_bulk (mV)")
    axes[2].set_title("Diffusion (junction) potential")

    for ax in axes:
        ax.set_xlabel("Distance from cathode (µm)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Nernst–Planck cathode film at {j_mA_cm2:.0f} mA/cm² "
        "(diffusion + migration, pH 2, 60 °C)",
        fontsize=11,
    )
    out = FIG_DIR / "nernst_planck_profiles.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✅ Saved: {out}")
    return s_bare, s_salty


# ─── Figure 2: migration enhancement vs. supporting electrolyte ───────
def plot_migration_enhancement():
    support = np.array([0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
    ratios, t_fe = [], []
    for c_s in support:
        film = NernstPlanckFilm(bulk_pH=7.0, support_conc_M=float(c_s))
        ratios.append(film.transport_limit_A_m2() / film.diffusion_limit_A_m2)
        t_fe.append(film.fe_transference_number)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.semilogx(np.maximum(support, 0.02), ratios, "o-", color="#2e7d32", lw=2,
                label="Nernst–Planck  i_lim / i_Levich")
    ax.axhline(2.0, color="#c62828", ls="--", lw=1.3,
               label="binary-salt limit (2×)")
    ax.axhline(1.0, color="#616161", ls=":", lw=1.3,
               label="pure diffusion (Levich)")
    ax.set_xlabel("Supporting electrolyte, Na₂SO₄ (M)   [leftmost point = 0]")
    ax.set_ylabel("Limiting-current enhancement (×)")
    ax.set_title("Migration boosts the Fe²⁺ transport limit\n"
                 "in weakly supported baths")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)

    ax2 = ax.twinx()
    ax2.semilogx(np.maximum(support, 0.02), t_fe, "s--", color="#6a1b9a",
                 lw=1.4, ms=4, alpha=0.8)
    ax2.set_ylabel("Fe²⁺ transference number  t₊", color="#6a1b9a")
    ax2.tick_params(axis="y", labelcolor="#6a1b9a")

    out = FIG_DIR / "migration_enhancement.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✅ Saved: {out}")
    return support, ratios, t_fe


# ─── Figure 3: Nernst-Planck vs. the linear film model ────────────────
def plot_model_comparison(j_values=(5.0, 20.0, 50.0, 100.0, 200.0)):
    """Surface pH and current efficiency: NP vs. the stagnant-film model."""
    js = np.array(j_values, dtype=float)
    np_pH, np_ce, film_pH, film_ce = [], [], [], []
    for j in js:
        s = NernstPlanckFilm(bulk_pH=2.0, her_i0=1e-4).solve(float(j))
        b = CathodeBoundaryLayer(bulk_pH=2.0, her_i0=1e-4).solve(float(j))
        np_pH.append(s.surface_pH)
        np_ce.append(s.current_efficiency * 100)
        film_pH.append(b.surface_pH)
        film_ce.append(b.current_efficiency * 100)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    axes[0].plot(js, np_pH, "o-", color="#1565c0", lw=2,
                 label="Nernst–Planck (diffusion + migration)")
    axes[0].plot(js, film_pH, "s--", color="#ef6c00", lw=2,
                 label="linear film (diffusion only)")
    axes[0].axhline(2.0, color="#616161", ls=":", lw=1, label="bulk pH")
    axes[0].set_ylabel("Surface pH")
    axes[0].set_title("Local pH: proton transport is the missing buffer",
                      fontsize=10)

    axes[1].plot(js, np_ce, "o-", color="#1565c0", lw=2, label="Nernst–Planck")
    axes[1].plot(js, film_ce, "s--", color="#ef6c00", lw=2, label="linear film")
    axes[1].set_ylabel("Current efficiency (%)")
    axes[1].set_title("Predicted current efficiency", fontsize=10)

    for ax in axes:
        ax.set_xlabel("Applied current density (mA/cm²)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5)

    fig.suptitle("Transport model comparison — pH 2, 1 M Fe²⁺, 60 °C", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIG_DIR / "transport_model_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✅ Saved: {out}")
    return js, np_pH, np_ce, film_pH, film_ce


# ─── Report ───────────────────────────────────────────────────────────
def build_report(support, ratios, t_fe, comparison) -> dict:
    js, np_pH, np_ce, film_pH, film_ce = comparison

    operating_points = {}
    for label, kw, j in [
        ("Unsupported acidic (pH 2)", dict(bulk_pH=2.0, her_i0=1e-4), 100.0),
        ("Supported acidic (2 M Na₂SO₄)",
         dict(bulk_pH=2.0, her_i0=1e-4, support_conc_M=2.0), 100.0),
        ("HER-inhibited acidic", dict(bulk_pH=2.0, her_i0=1e-6), 100.0),
        ("Dilute + stagnant",
         dict(bulk_pH=3.0, fe_conc_M=0.1, boundary_layer_m=2e-4, her_i0=1e-4), 50.0),
        ("Alkaline (pH 9)",
         dict(bulk_pH=9.0, fe_conc_M=0.05, her_i0=1e-4), 20.0),
    ]:
        operating_points[label] = NernstPlanckFilm(**kw).summary(j)

    return {
        "model": {
            "description": "Steady 1-D Nernst-Planck film: diffusion + migration, "
                           "electroneutral, fast water autoprotolysis",
            "species": ["Fe2+", "H+", "OH-", "Na+", "SO4^2-"],
            "closure": "electroneutrality differentiated to give dphi/dx",
        },
        "migration_enhancement": [
            {
                "support_M": float(c),
                "i_lim_ratio_vs_Levich": round(float(r), 4),
                "t_Fe": round(float(t), 4),
            }
            for c, r, t in zip(support, ratios, t_fe)
        ],
        "support_sweep_at_100mA_cm2": [
            {k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}
            for row in compare_support_levels(j_mA_cm2=100.0, bulk_pH=2.0, her_i0=1e-4)
        ],
        "model_comparison": [
            {
                "j_mA_cm2": float(j),
                "nernst_planck_surface_pH": round(float(a), 3),
                "linear_film_surface_pH": round(float(b), 3),
                "nernst_planck_CE_pct": round(float(c), 2),
                "linear_film_CE_pct": round(float(d), 2),
            }
            for j, a, b, c, d in zip(js, np_pH, film_pH, np_ce, film_ce)
        ],
        "operating_points": operating_points,
    }


def main():
    print("=" * 70)
    print("NERNST–PLANCK TRANSPORT — MIGRATION IN THE CATHODE FILM")
    print("=" * 70)

    print("\nGenerating figures...")
    s_bare, s_salty = plot_profiles()
    support, ratios, t_fe = plot_migration_enhancement()
    comparison = plot_model_comparison()

    report = build_report(support, ratios, t_fe, comparison)

    print("\nMigration enhancement of the Fe²⁺ transport limit:")
    hdr = f"{'Na₂SO₄ (M)':>12}{'t_Fe':>9}{'i_lim/i_Levich':>17}"
    print(hdr)
    print("─" * len(hdr))
    for c, r, t in zip(support, ratios, t_fe):
        print(f"{c:>12.2f}{t:>9.3f}{r:>17.3f}")

    print("\nOperating points:")
    hdr = f"{'Case':<32}{'CE':>8}{'pH_s':>8}{'i_lim×':>9}{'Δφ mV':>9}"
    print(hdr)
    print("─" * len(hdr))
    for name, s in report["operating_points"].items():
        print(f"{name:<32}{s['Current efficiency (%)']:>7.1f}%"
              f"{s['Surface pH']:>8.2f}"
              f"{s['Migration enhancement (×)']:>9.2f}"
              f"{s['Film potential drop (mV)']:>9.2f}")

    print("\nNernst–Planck vs. linear film (pH 2 bath):")
    hdr = f"{'j (mA/cm²)':>12}{'pH_s NP':>10}{'pH_s film':>11}{'CE NP':>8}{'CE film':>9}"
    print(hdr)
    print("─" * len(hdr))
    for row in report["model_comparison"]:
        print(f"{row['j_mA_cm2']:>12.0f}{row['nernst_planck_surface_pH']:>10.2f}"
              f"{row['linear_film_surface_pH']:>11.2f}"
              f"{row['nernst_planck_CE_pct']:>7.1f}%"
              f"{row['linear_film_CE_pct']:>8.1f}%")

    out = DATA_DIR / "transport_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n  ✅ Saved: {out}")
    print("\n✅ Nernst–Planck transport analysis complete!")


if __name__ == "__main__":
    main()
