#!/usr/bin/env python3
"""
Electrochemical analysis driver: Pourbaix diagram + HER-competition kinetics.

Generates:
    docs/figures/pourbaix_fe_h2o.png
    docs/figures/polarization_curves.png
    docs/figures/current_efficiency_map.png
    experiments/data/electrochemistry_report.json

Usage:
    python -m models.run_electrochemistry
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.kinetics import DepositionKinetics  # noqa: E402
from models.pourbaix import FePourbaix, her_line, oer_line  # noqa: E402
from models.electrochemistry import specific_energy_kWh_per_t  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "docs" / "figures"
DATA_DIR = ROOT / "experiments" / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 10})


# ─── Figure 1: Pourbaix diagram ───────────────────────────────────────
def plot_pourbaix(activity: float = 1.0, temperature_C: float = 60.0):
    p = FePourbaix(activity=activity, temperature_C=temperature_C)
    pH = np.linspace(0, 16, 400)
    fig, ax = plt.subplots(figsize=(8, 6))

    ph1, ph2 = p.pH_Fe2_FeOH2, p.pH_FeOH2_HFeO2

    # Immunity (Fe metal) boundary, piecewise across pH regimes
    E_dep = np.array([p.deposition_potential(x) for x in pH])
    ax.plot(pH, E_dep, color="#1a237e", lw=2.2, label="Fe(s) stability boundary")
    ax.fill_between(pH, -1.8, E_dep, color="#1a237e", alpha=0.10)

    # Fe2+/Fe3+ and hydroxide boundaries
    m_ac = pH <= ph1
    ax.plot(pH[m_ac], p.E_Fe3_Fe2(pH[m_ac]), color="#b71c1c", lw=1.4,
            label="Fe³⁺/Fe²⁺")
    m_ox = (pH > p.pH_Fe3_FeOH3) & (pH <= ph1)
    ax.plot(pH[m_ox], p.E_FeOH3_Fe2(pH[m_ox]), color="#e65100", lw=1.4,
            label="Fe(OH)₃/Fe²⁺")
    m_alk = (pH > ph1) & (pH <= ph2)
    ax.plot(pH[m_alk], p.E_FeOH3_FeOH2(pH[m_alk]), color="#e65100", lw=1.4, ls="--",
            label="Fe(OH)₃/Fe(OH)₂")

    # Vertical hydrolysis boundaries
    for x, lbl in ((p.pH_Fe3_FeOH3, "Fe³⁺→Fe(OH)₃"), (ph1, "Fe²⁺→Fe(OH)₂")):
        ax.axvline(x, color="#616161", lw=1.0, ls=":")
        ax.text(x + 0.12, 1.45, lbl, rotation=90, fontsize=7.5, va="top", color="#424242")

    # Water stability window
    ax.plot(pH, her_line(pH, p.T), "k--", lw=1.3, label="H₂O/H₂ (HER)")
    ax.plot(pH, oer_line(pH, p.T), "k-.", lw=1.3, label="O₂/H₂O (OER)")

    # Annotate the HER penalty at a representative alkaline point
    pH_ann = 11.0
    ax.annotate(
        "",
        xy=(pH_ann, p.deposition_potential(pH_ann)),
        xytext=(pH_ann, float(her_line(pH_ann, p.T))),
        arrowprops=dict(arrowstyle="<->", color="#00695c", lw=1.6),
    )
    ax.text(pH_ann + 0.3, (p.deposition_potential(pH_ann) + float(her_line(pH_ann, p.T))) / 2 - 0.18,
            f"HER penalty {p.her_margin(pH_ann)*1000:.0f} mV",
            fontsize=8, color="#00695c", va="center")

    ax.text(1.0, -1.5, "Fe(s) — immunity / deposition domain",
            fontsize=9, color="#1a237e", fontweight="bold")
    ax.text(2.9, -0.15, "Fe²⁺", fontsize=11, color="#b71c1c", fontweight="bold")
    ax.text(0.25, 1.05, "Fe³⁺", fontsize=11, color="#b71c1c", fontweight="bold")
    ax.text(8.0, 0.0, "Fe(OH)₂", fontsize=10, color="#424242")
    ax.text(8.0, 0.85, "Fe(OH)₃", fontsize=10, color="#e65100")

    ax.set_xlim(0, 16)
    ax.set_ylim(-1.8, 1.8)
    ax.set_xlabel("pH")
    ax.set_ylabel("Potential (V vs. SHE)")
    ax.set_title(f"Fe–H₂O Pourbaix Diagram  (a$_{{Fe}}$ = {activity:g} M, {temperature_C:.0f} °C)",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    ax.grid(alpha=0.25)
    out = FIG_DIR / "pourbaix_fe_h2o.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✅ Saved: {out}")
    return p


# ─── Figure 2: Polarization curves ────────────────────────────────────
def plot_polarization():
    cases = [
        ("Active cathode (Fe substrate, pH 2)", dict(pH=2.0, her_i0=1e-2), "#e53935"),
        ("HER-suppressed additive (pH 2)", dict(pH=2.0, her_i0=1e-5), "#fb8c00"),
        ("Mildly acidic, complexed (pH 5)", dict(pH=5.0, her_i0=1e-5), "#43a047"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    E = np.linspace(-1.6, -0.4, 500)

    ax = axes[0]
    # The Fe branch is common to all cases (same bath chemistry / transport);
    # the cases differ only in the HER branch, which is the design lever.
    k0 = DepositionKinetics(**cases[0][1])
    _, i_fe0, _, _, _ = k0.polarization_curve(E)
    ax.semilogy(E, i_fe0, color="#1a237e", lw=2.6, label="Fe²⁺ + 2e⁻ → Fe (all cases)")
    ax.axhline(k0.i_lim, color="#1a237e", ls=":", lw=1.0)
    ax.text(-1.58, k0.i_lim * 1.15, "i$_{lim,Fe}$", fontsize=8, color="#1a237e")
    for label, kw, c in cases:
        k = DepositionKinetics(**kw)
        _, _, i_h, _, _ = k.polarization_curve(E)
        ax.semilogy(E, i_h, color=c, lw=1.6, ls="--", label=f"HER — {label}")
    ax.axhline(1000, color="gray", ls=":", lw=1.2)
    ax.text(-1.58, 1150, "100 mA/cm² operating target", fontsize=8, color="gray")
    ax.set_xlabel("Cathode potential (V vs. SHE)")
    ax.set_ylabel("Partial current density (A/m²)")
    ax.set_ylim(1e-2, 1e5)
    ax.set_title("Partial-current Tafel behaviour", fontweight="bold")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.25, which="both")

    ax = axes[1]
    js = np.logspace(0, 2.8, 60)
    for label, kw, c in cases:
        k = DepositionKinetics(**kw)
        _, ce = k.efficiency_sweep(js)
        ax.semilogx(js, ce * 100, color=c, lw=2, label=label)
    ax.axhline(95, color="gray", ls="--", lw=1.2)
    ax.text(1.2, 96, "95% CE target", fontsize=8, color="gray")
    ax.set_xlabel("Applied current density (mA/cm²)")
    ax.set_ylabel("Current efficiency (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Current efficiency vs. current density", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25, which="both")

    out = FIG_DIR / "polarization_curves.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✅ Saved: {out}")


# ─── Figure 3: Efficiency map ─────────────────────────────────────────
def plot_efficiency_map():
    js = np.logspace(0.3, 2.7, 45)          # 2 - 500 mA/cm2
    her_i0s = np.logspace(-6, -2, 45)       # HER exchange current density
    Z = np.zeros((len(her_i0s), len(js)))
    for i, h in enumerate(her_i0s):
        k = DepositionKinetics(pH=3.0, her_i0=float(h))
        for j_idx, j in enumerate(js):
            Z[i, j_idx] = k.efficiency_at_current(float(j)) * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    cf = ax.contourf(js, her_i0s, Z, levels=np.linspace(0, 100, 21), cmap="viridis")
    cs = ax.contour(js, her_i0s, Z, levels=[50, 80, 90, 95, 99], colors="white", linewidths=1.1)
    ax.clabel(cs, fmt="%.0f%%", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Current density (mA/cm²)")
    ax.set_ylabel("HER exchange current density i₀,H (A/m²)")
    ax.set_title("Current efficiency map (pH 3, 60 °C)", fontweight="bold")
    fig.colorbar(cf, ax=ax, label="Current efficiency (%)")

    # Energy consequence
    ax = axes[1]
    for h, c in [(1e-2, "#e53935"), (1e-4, "#fb8c00"), (1e-6, "#43a047")]:
        k = DepositionKinetics(pH=3.0, her_i0=h)
        _, ce = k.efficiency_sweep(js)
        energy = np.array([specific_energy_kWh_per_t(2.6, max(c_, 1e-3)) for c_ in ce])
        ax.loglog(js, energy, color=c, lw=2, label=f"i₀,H = {h:g} A/m²")
    ax.axhline(1500, color="gray", ls="--", lw=1.2)
    ax.text(2.5, 1600, "1,500 kWh/t target (EAF-competitive)", fontsize=8, color="gray")
    ax.set_xlabel("Current density (mA/cm²)")
    ax.set_ylabel("Specific energy at V$_{cell}$ = 2.6 V (kWh/t Fe)")
    ax.set_title("Energy penalty of HER competition", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")

    out = FIG_DIR / "current_efficiency_map.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✅ Saved: {out}")


# ─── Report ───────────────────────────────────────────────────────────
def build_report(p: FePourbaix) -> dict:
    operating_points = {}
    for label, kw, j in [
        ("Acidic active cathode", dict(pH=2.0, her_i0=1e-2), 100.0),
        ("Acidic + HER inhibitor", dict(pH=2.0, her_i0=1e-5), 100.0),
        ("Mildly acidic complexed", dict(pH=5.0, her_i0=1e-5), 150.0),
        ("Transport-limited (stagnant)", dict(pH=3.0, her_i0=1e-5, fe_conc_M=0.1,
                                              boundary_layer_m=2e-4), 100.0),
        ("Transport-limited (agitated)", dict(pH=3.0, her_i0=1e-5, fe_conc_M=0.1,
                                              boundary_layer_m=2e-5), 100.0),
    ]:
        k = DepositionKinetics(**kw)
        s = k.summary(j)
        s["specific energy @2.6 V (kWh/t Fe)"] = round(
            specific_energy_kWh_per_t(2.6, k.efficiency_at_current(j)), 0
        )
        operating_points[label] = s

    return {
        "pourbaix": {
            "activity_M": p.activity,
            "temperature_C": p.temperature_C,
            "pH_Fe2_to_FeOH2": round(p.pH_Fe2_FeOH2, 2),
            "pH_Fe3_to_FeOH3": round(p.pH_Fe3_FeOH3, 2),
            "pH_FeOH2_to_HFeO2": round(p.pH_FeOH2_HFeO2, 2),
            "by_pH": {str(k): v for k, v in p.summary().items()},
        },
        "kinetics": operating_points,
    }


def main():
    print("=" * 70)
    print("ELECTROCHEMICAL ANALYSIS — Fe–H₂O THERMODYNAMICS & HER COMPETITION")
    print("=" * 70)

    print("\nGenerating figures...")
    p = plot_pourbaix()
    plot_polarization()
    plot_efficiency_map()

    report = build_report(p)

    print("\nPourbaix boundaries (a = 1 M, 60 °C):")
    for k, v in report["pourbaix"].items():
        if k.startswith("pH_"):
            print(f"  {k:22s} {v}")

    print("\nOperating points:")
    hdr = f"{'Case':<32}{'j':>7}{'CE':>8}{'rate':>10}{'kWh/t':>9}"
    print(hdr)
    print("─" * len(hdr))
    for name, s in report["kinetics"].items():
        print(f"{name:<32}{s['j applied (mA/cm²)']:>7.0f}"
              f"{s['Current efficiency (%)']:>7.1f}%"
              f"{s['Deposition rate (µm/hr)']:>9.1f}µ"
              f"{s['specific energy @2.6 V (kWh/t Fe)']:>9.0f}")

    out = DATA_DIR / "electrochemistry_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n  ✅ Saved: {out}")
    print("\n✅ Electrochemical analysis complete!")


if __name__ == "__main__":
    main()
