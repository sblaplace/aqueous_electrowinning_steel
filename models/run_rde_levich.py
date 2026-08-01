"""Rotating disk electrode (RDE) kinetics/transport separation: figures + JSON
report.

Closes the last unbuilt Tier 1 desk model named in ``docs/RESEARCH_PROGRAM.md``
("RDE + Levich for kinetics/transport separation -- not started; the
measurement that makes ``diffusion_layer_1d`` calibratable. Highest-value
remaining Tier 1 model+experiment pair.").

Demonstrates the full method on a synthetic Fe/HER RDE polarization set:
Levich D extraction at the transport-limited plateau, Koutecky-Levich-corrected
Fe Tafel, and HER Tafel by subtracting i_lim. The report prints the recovered
vs. true kinetic/transport parameters so the method is visibly validated before
it is ever pointed at wet-lab data.

Run::

    python -m models.run_rde_levich
    aq-steel-rde
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .rde_levich import (
    KIN_VISC_WATER_25,
    RDEBranch,
    analyze_rde_polarization,
    diffusivity_from_levich_B,
    koutecky_levich_kinetic,
    levich_limiting_current,
    model_scope,
    recommend_windows_from_polarization,
    rpm_to_rad_per_s,
    rde_experiment_design,
    simulate_rde_polarization,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments" / "data"
FIGURES = ROOT / "docs" / "figures"

# "True" parameters used to generate the synthetic dataset.
TRUE = {
    "D_m2_s": 7.2e-10,
    "C_fe_M": 1.0,
    "nu_m2_s": KIN_VISC_WATER_25,
    "fe_i0_A_m2": 500.0,
    "fe_tafel_V": 0.120,
    "fe_E_eq_V": -0.440,
    "her_i0_A_m2": 0.02,
    "her_tafel_V": 0.150,
    "pH": 2.0,
}

OMEGAS_RPM = [400.0, 900.0, 1600.0, 2500.0]
E_GRID_V = np.linspace(-0.50, -1.05, 96)


def _fig_levich() -> Path:
    """Figure 1: Levich (i_lim vs omega^1/2) and Koutecky-Levich (1/i vs 1/sqrt(omega))."""
    fe = RDEBranch(TRUE["fe_i0_A_m2"], TRUE["fe_tafel_V"], TRUE["fe_E_eq_V"])
    her = RDEBranch(TRUE["her_i0_A_m2"], TRUE["her_tafel_V"], -0.1184)
    data = simulate_rde_polarization(
        E_GRID_V, np.array(OMEGAS_RPM), fe=fe, her=her,
        D_m2_s=TRUE["D_m2_s"], C_fe_M=TRUE["C_fe_M"], nu_m2_s=TRUE["nu_m2_s"],
    )
    df = data["frame"]
    rec = recommend_windows_from_polarization(df)
    plateau = rec["plateau_E_V"]
    omegas = np.array(OMEGAS_RPM)
    i_plateau = np.array(
        [df[(df["potential_V"] == plateau) & (df["omega_rpm"] == o)]["i_total_A_m2"].mean()
         for o in omegas]
    )
    # recover B / D from the plateau (through-origin Levich fit)
    x = np.sqrt(rpm_to_rad_per_s(omegas))
    B = float(np.sum(x * i_plateau) / np.sum(x * x))
    D = diffusivity_from_levich_B(B, z=2, C_bulk_M=TRUE["C_fe_M"], nu_m2_s=TRUE["nu_m2_s"])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=150)
    # Left: Levich plot
    ax = axes[0]
    ax.plot(x, i_plateau, "o-", color="#2b5c8f", ms=5)
    ax.plot(x, B * x, "--", color="#d95f02", lw=1.2, label=f"fit B={B:,.0f} A/m²·s^{{-1/2}}")
    ax.set_xlabel("√ω  (rad/s)$^{1/2}$")
    ax.set_ylabel("i_lim (A/m²)")
    ax.set_title(f"Levich plot @ plateau E={plateau:.2f} V\n→ D = {D:.3g} m²/s")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Right: Koutecky-Levich at a mid-potential kinetic point
    ax = axes[1]
    # pick a potential inside the kinetic window
    kw = rec["kinetic_window_V"]
    probe_E = kw[1] if kw is not None else plateau + 0.03
    probe = np.array([
        df[(df["potential_V"] == probe_E) & (df["omega_rpm"] == o)]["i_total_A_m2"].mean()
        for o in omegas
    ])
    xkl = 1.0 / np.sqrt(rpm_to_rad_per_s(omegas))
    ykl = 1.0 / probe
    slope, intercept = np.polyfit(xkl, ykl, 1)
    i_k = 1.0 / intercept
    xs = np.linspace(xkl.min(), xkl.max(), 50)
    ax.plot(xkl, ykl, "o", color="#1b9e77", ms=5)
    ax.plot(xs, slope * xs + intercept, "--", color="#d95f02", lw=1.2)
    ax.set_xlabel("1/√ω  (s$^{1/2}$·rad$^{-1/2}$)")
    ax.set_ylabel("1/i  (m²/A)")
    ax.set_title(f"Koutecký–Levich @ {probe_E:.2f} V\n1/i_k = {1e3*i_k:,.1f} m²/kA")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = FIGURES / "rde_levich_levich_kl.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _fig_polarization() -> Path:
    """Figure 2: RDE polarization curves at four rotation rates + windows."""
    fe = RDEBranch(TRUE["fe_i0_A_m2"], TRUE["fe_tafel_V"], TRUE["fe_E_eq_V"])
    her = RDEBranch(TRUE["her_i0_A_m2"], TRUE["her_tafel_V"], -0.1184)
    data = simulate_rde_polarization(
        E_GRID_V, np.array(OMEGAS_RPM), fe=fe, her=her,
        D_m2_s=TRUE["D_m2_s"], C_fe_M=TRUE["C_fe_M"], nu_m2_s=TRUE["nu_m2_s"],
    )
    df = data["frame"]
    rec = recommend_windows_from_polarization(df)

    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=150)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(OMEGAS_RPM)))
    for o, c in zip(OMEGAS_RPM, colors):
        sub = df[df["omega_rpm"] == o]
        ax.plot(sub["potential_V"], sub["i_total_A_m2"] / 1e4, "-", color=c,
                lw=1.4, label=f"{int(o)} rpm")
    # shade the three windows
    p = rec["plateau_E_V"]
    kw, hw = rec["kinetic_window_V"], rec["her_window_V"]
    if kw:
        ax.axvspan(kw[0], kw[1], color="#2b5c8f", alpha=0.10, label="kinetic (K–L → Fe Tafel)")
    ax.axvline(p, color="#d95f02", ls="--", lw=1.2, label=f"plateau {p:.2f} V (Levich D)")
    if hw:
        ax.axvspan(hw[0], hw[1], color="#1b9e77", alpha=0.12, label="deep-cathodic (HER Tafel)")
    ax.set_xlabel("Potential (V vs. SHE)")
    ax.set_ylabel("Total cathodic current (10⁴ A/m²)")
    ax.set_title("RDE polarization: rotation rate lifts the Fe transport limit")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out = FIGURES / "rde_levich_polarization.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _fig_tafel_extraction() -> Path:
    """Figure 3: Fe kinetic Tafel (K-L corrected) and HER Tafel (transport-subtracted)."""
    fe = RDEBranch(TRUE["fe_i0_A_m2"], TRUE["fe_tafel_V"], TRUE["fe_E_eq_V"])
    her = RDEBranch(TRUE["her_i0_A_m2"], TRUE["her_tafel_V"], -0.1184)
    data = simulate_rde_polarization(
        E_GRID_V, np.array(OMEGAS_RPM), fe=fe, her=her,
        D_m2_s=TRUE["D_m2_s"], C_fe_M=TRUE["C_fe_M"], nu_m2_s=TRUE["nu_m2_s"],
    )
    df = data["frame"]
    res = analyze_rde_polarization(df, C_fe_M=TRUE["C_fe_M"], pH=TRUE["pH"],
                                   D_ref_m2_s=TRUE["D_m2_s"])
    rec = res["windows"]
    omegas = np.array(OMEGAS_RPM)
    iL = levich_limiting_current(omegas, z=2, D_m2_s=res["levich"]["D_m2_s"],
                                 C_bulk_M=TRUE["C_fe_M"], nu_m2_s=TRUE["nu_m2_s"])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=150)
    # Left: Fe kinetic Tafel
    ax = axes[0]
    kw = rec["kinetic_window_V"]
    pots = np.array(sorted(set(df["potential_V"].tolist())))
    kpots = [p for p in pots if kw[0] <= p <= kw[1]]
    ik_all = []
    for p in kpots:
        vals = []
        for o, lim in zip(omegas, iL):
            itot = float(df[(df["potential_V"] == p) & (df["omega_rpm"] == o)]["i_total_A_m2"].mean())
            if itot < 0.9 * lim:
                vals.append(koutecky_levich_kinetic(np.array([itot]), np.array([lim]))[0])
        if vals:
            ik_all.append((p, float(np.nanmean(vals))))
    ikp = np.array([v[0] for v in ik_all])
    iki = np.array([v[1] for v in ik_all])
    eta = TRUE["fe_E_eq_V"] - ikp
    ax.plot(eta, iki / 100.0, "o", color="#2b5c8f", ms=4, label="K–L corrected")
    fit = res["fe_tafel"]
    etafit = np.linspace(eta.min(), eta.max(), 50)
    ax.plot(etafit, (fit["i0_A_m2"] * 10 ** (etafit / fit["tafel_slope_V_decade"])) / 100.0,
            "--", color="#d95f02", lw=1.2,
            label=f"b={fit['tafel_slope_V_decade']:.3f} V/dec")
    ax.set_yscale("log")
    ax.set_xlabel("η (V)")
    ax.set_ylabel("Fe kinetic current i_k (10² A/m²)")
    ax.set_title(f"Fe Tafel (true b={TRUE['fe_tafel_V']:.3f}, i0={TRUE['fe_i0_A_m2']:.0f}) → rec. b={fit['tafel_slope_V_decade']:.3f}, "
                 f"i0={fit['i0_A_m2']:.0f}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    # Right: HER Tafel (total - i_lim)
    ax = axes[1]
    hw = rec["her_window_V"]
    hpots = [p for p in pots if hw[0] <= p <= hw[1]]
    ih_all = []
    for p in hpots:
        vals = []
        for o, lim in zip(omegas, iL):
            itot = float(df[(df["potential_V"] == p) & (df["omega_rpm"] == o)]["i_total_A_m2"].mean())
            v = itot - lim
            if v > 1.0e-3:
                vals.append(v)
        if vals:
            ih_all.append((p, float(np.nanmean(vals))))
    ihp = np.array([v[0] for v in ih_all])
    ihi = np.array([v[1] for v in ih_all])
    eta_h = TRUE["pH"] and (res["E_eq_her_V"] - ihp)
    ax.plot(eta_h, ihi, "o", color="#1b9e77", ms=4, label="i_total − i_lim")
    hfit = res["her_tafel"]
    etafit_h = np.linspace(eta_h.min(), eta_h.max(), 50)
    ax.plot(etafit_h, hfit["i0_A_m2"] * 10 ** (etafit_h / hfit["tafel_slope_V_decade"]),
            "--", color="#d95f02", lw=1.2,
            label=f"b={hfit['tafel_slope_V_decade']:.3f} V/dec")
    ax.set_yscale("log")
    ax.set_xlabel("η (V)")
    ax.set_ylabel("HER current (A/m²)")
    ax.set_title(f"HER Tafel (true b={TRUE['her_tafel_V']:.3f}, i0={TRUE['her_i0_A_m2']:.2f}) → rec. b={hfit['tafel_slope_V_decade']:.3f}, "
                 f"i0={hfit['i0_A_m2']:.4f}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    out = FIGURES / "rde_levich_tafel.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _findings(res: dict, design: dict) -> list:
    f = []
    D = res["levich"]["D_m2_s"]
    f.append(
        f"Levich plateau @ {res['plateau_E_V']:.2f} V recovers D = {D:.3g} m²/s "
        f"(true {TRUE['D_m2_s']:.3g}, {res['recovery_checks']['relative_error_pct']:+.1f}%)"
    )
    f.append(
        f"Nernst film thickness at 1600 rpm = {res['nernst_layer_m_at_1600rpm']*1e6:.1f} µm "
        f"— the value to feed diffusion_layer_1d as its boundary layer."
    )
    ft = res["fe_tafel"]
    f.append(
        f"Fe Tafel b = {ft['tafel_slope_V_decade']:.3f} V/dec, i0 = {ft['i0_A_m2']:.0f} A/m² "
        f"(true {TRUE['fe_tafel_V']:.3f}, {TRUE['fe_i0_A_m2']:.0f}; R²={ft['r_squared']:.3f})"
    )
    ht = res["her_tafel"]
    f.append(
        f"HER Tafel b = {ht['tafel_slope_V_decade']:.3f} V/dec, i0 = {ht['i0_A_m2']:.4f} A/m² "
        f"(true {TRUE['her_tafel_V']:.3f}, {TRUE['her_i0_A_m2']:.2f}; R²={ht['r_squared']:.3f})"
    )
    f.append(
        f"Design: {design['rotation_matrix_rpm'][0]:.0f}–{design['rotation_matrix_rpm'][-1]:.0f} rpm "
        f"gives an i_lim spread of {design['i_lim_spread_ratio']:.1f}×; current meter should "
        f"resolve ~{design['current_precision_requirement_pct']:.1f}%."
    )
    return f


def main() -> dict:
    """Run the RDE/Levich model, print findings, and write the report."""
    print("=== RDE Kinetics/Transport Separation (Levich + Koutecky-Levich) ===")
    print()

    fe = RDEBranch(TRUE["fe_i0_A_m2"], TRUE["fe_tafel_V"], TRUE["fe_E_eq_V"])
    her = RDEBranch(TRUE["her_i0_A_m2"], TRUE["her_tafel_V"], -0.1184)
    data = simulate_rde_polarization(
        E_GRID_V, np.array(OMEGAS_RPM), fe=fe, her=her,
        D_m2_s=TRUE["D_m2_s"], C_fe_M=TRUE["C_fe_M"], nu_m2_s=TRUE["nu_m2_s"],
    )
    df = data["frame"]
    print(f"Synthetic polarization set: {len(df)} rows "
          f"({len(OMEGAS_RPM)} rotation rates x {len(E_GRID_V)} potentials).")
    print(f"Rotation rates: {OMEGAS_RPM} rpm")
    print()

    res = analyze_rde_polarization(df, C_fe_M=TRUE["C_fe_M"], pH=TRUE["pH"],
                                   D_ref_m2_s=TRUE["D_m2_s"])
    design = rde_experiment_design()
    findings = _findings(res, design)

    fig1 = _fig_levich()
    fig2 = _fig_polarization()
    fig3 = _fig_tafel_extraction()

    report = {
        "title": "RDE kinetics/transport separation (Levich + Koutecky-Levich)",
        "purpose": (
            "Extract D, the Nernst film thickness, and the Fe/HER Tafel kinetics "
            "from one RDE polarization set, calibrating diffusion_layer_1d's "
            "free boundary-layer parameter from measurement rather than assumption."
        ),
        "model_scope": model_scope(),
        "true_parameters": TRUE,
        "experiment": {"omega_rpm": OMEGAS_RPM, "potential_min_V": float(E_GRID_V.min()),
                       "potential_max_V": float(E_GRID_V.max())},
        "analysis": res,
        "experiment_design": design,
        "findings": findings,
        "figures": [f"docs/figures/{fig1.name}", f"docs/figures/{fig2.name}",
                    f"docs/figures/{fig3.name}"],
    }

    out = DATA / "rde_levich_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("Recovered vs. true:")
    chk = res["recovery_checks"]
    print(f"  D: {chk['D_recovered_m2_s']:.4g} vs {chk['D_ref_m2_s']:.4g} m²/s "
          f"({chk['relative_error_pct']:+.2f}%)")
    print(f"  delta @1600rpm: {res['nernst_layer_m_at_1600rpm']*1e6:.2f} µm")
    print(f"  Fe Tafel: b={res['fe_tafel']['tafel_slope_V_decade']:.4f} "
          f"i0={res['fe_tafel']['i0_A_m2']:.3g} (R²={res['fe_tafel']['r_squared']:.4f})")
    print(f"  HER Tafel: b={res['her_tafel']['tafel_slope_V_decade']:.4f} "
          f"i0={res['her_tafel']['i0_A_m2']:.5f} (R²={res['her_tafel']['r_squared']:.4f})")
    print()
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
