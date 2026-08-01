#!/usr/bin/env python3
"""Inverse Hull-cell driver: measured thickness profile → local FE(j) calibration.

Turns a gate-2 Hull-panel thickness profile (Day-1 packet, R1/R2-style runs)
into a strip-by-strip apparent Fe Faradaic-efficiency map, a binned FE(j)
calibration table, and a Tafel-consistent logit fit that downstream
calibration (``diffusion_layer_1d`` / Bayesian pipeline) can consume.

This driver produces a clearly labelled synthetic example; it does not
represent wet-lab deposition or profilometry data.

Generates:
    experiments/data/hull_cell_inverse_report.json
    docs/figures/hull_cell_inverse_profile.png

Usage:
    python -m models.run_hull_cell_inverse
    python -m models.run_hull_cell_inverse --current_A 6.0   # probe higher j
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models.hull_cell import (
    HullCellGeometry,
    hull_current_distribution,
    summarize_hull_distribution,
)
from models.hull_cell_inverse import (
    analyze_hull_panel,
    logit_fe,
    synthesize_thickness_profile,
)

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "docs" / "figures"
DATA_DIR = ROOT / "experiments" / "data"

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 10})

# Truth model for the synthetic example: FE = 85 % at 100 mA/cm², declining
# with slope b = 1 - α_H/α_Fe = -0.35 (HER takes an increasing share at high j).
TRUTH_FE_AT_100 = 0.85
TRUTH_LOGIT_SLOPE = -0.35


def _truth_logit_params() -> tuple[float, float]:
    logit_at_100 = np.log(TRUTH_FE_AT_100 / (1.0 - TRUTH_FE_AT_100))
    a = logit_at_100 - TRUTH_LOGIT_SLOPE * np.log(100.0)
    return float(a), TRUTH_LOGIT_SLOPE


def _plot_profile_and_recovery(profile, local, binned, fit, path: Path) -> None:
    """Write the two-panel synthetic recovery figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))

    ax = axes[0]
    x = profile["position_cm_from_near_edge"]
    ax.plot(x, profile["true_thickness_um"], color="#1874b4", linewidth=2,
            label="True thickness (known FE model)")
    ax.plot(x, profile["measured_thickness_um"], color="#d95f02", marker="o",
            linewidth=1.2, markersize=4, label="'Measured' thickness (noisy)")
    ax.set(xlabel="Distance from near panel edge (cm)", ylabel="Deposit thickness (µm)",
           title="Synthetic thickness profile across the angled panel")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    j_axis = ax.twinx()
    j_axis.plot(x, profile["current_density_mA_cm2"], color="0.45",
                linestyle="--", linewidth=1.2, label="Primary j")
    j_axis.set_ylabel("Primary current density (mA cm$^{-2}$)")
    j_axis.grid(alpha=0.0)

    ax = axes[1]
    ax.axhline(1.0, color="0.7", linestyle=":", linewidth=1)
    ax.scatter(local["current_density_mA_cm2"], local["apparent_faradaic_efficiency"],
               color="#1874b4", s=22, alpha=0.85, label="Recovered strip FE")
    if np.isfinite(fit.a) and np.isfinite(fit.b):
        j_smooth = np.geomspace(
            local["current_density_mA_cm2"].min() * 0.8,
            local["current_density_mA_cm2"].max() * 1.25, 200,
        )
        ax.plot(j_smooth, logit_fe(j_smooth, fit.a, fit.b), color="#33a02c",
                linewidth=1.8, label=f"logit fit (b = {fit.b:.2f})")
    ax.plot(binned["j_geometric_mean_mA_cm2"], binned["fe_current_weighted_mean"],
            color="#d95f02", marker="s", markersize=6, linewidth=1.4,
            label="Binned current-weighted mean")
    j_truth = np.geomspace(
        local["current_density_mA_cm2"].min() * 0.8,
        local["current_density_mA_cm2"].max() * 1.25, 200,
    )
    a, b = _truth_logit_params()
    ax.plot(j_truth, logit_fe(j_truth, a, b), color="0.35", linestyle="--",
            linewidth=1.4, label="Truth model")
    ax.set(xlabel="Primary current density (mA cm$^{-2}$)",
           ylabel="Apparent Fe Faradaic efficiency",
           title="Inverse Hull-cell FE(j) recovery", xscale="log")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, loc="best")

    fig.suptitle("Inverse Hull-cell analysis — synthetic example (not wet-lab data)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """Generate reproducible synthetic inverse Hull-cell outputs and a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--current_A", type=float, default=2.0,
                        help="Applied panel current (A); Day-1 R1/R2 runs use 2 A")
    parser.add_argument("--duration_s", type=float, default=3600.0,
                        help="Plating duration (s). Day-1 R1/R2 are 5-10 min appearance "
                             "screens at 2 A; the 1 h default just gives the synthetic "
                             "profile a ~20-100 µm thickness range to demonstrate")
    parser.add_argument("--noise_um", type=float, default=1.0,
                        help="Thickness measurement noise σ (µm); 1 µm ~ profilometry, "
                             "2 µm ~ point micrometer")
    parser.add_argument("--seed", type=int, default=7,
                        help="RNG seed for the synthetic thickness noise")
    args = parser.parse_args(argv)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("INVERSE HULL CELL — THICKNESS PROFILE → LOCAL FE(j) CALIBRATION")
    print("=" * 70)

    # Day-1 panel: 10 × 5 cm, 1.5 → 9.0 cm gap (48.6°), 2 A (FIRST_LAB_DAY.md §3).
    geometry = HullCellGeometry(
        panel_length_cm=10.0,
        panel_width_cm=5.0,
        near_edge_gap_cm=1.5,
        far_edge_gap_cm=9.0,
    )
    n_segments = 10  # one thickness reading per run-sheet strip
    distribution = hull_current_distribution(geometry, args.current_A, n_segments=n_segments)
    dist_summary = summarize_hull_distribution(distribution)
    print(f"\nPanel geometry: {geometry.panel_length_cm:.1f} × {geometry.panel_width_cm:.1f} cm "
          f"at {geometry.panel_angle_deg:.1f}°; applied {args.current_A:.2f} A")
    print(f"Primary j range: {dist_summary['near_edge_current_density_mA_cm2']:.1f} → "
          f"{dist_summary['far_edge_current_density_mA_cm2']:.1f} mA/cm² (near → far edge)")

    a, b = _truth_logit_params()
    print(f"\nSynthetic truth FE model: logit(FE) = {a:.3f} + ({b:.3f})·ln(j) "
          f"(FE = {TRUTH_FE_AT_100:.0%} @ 100 mA/cm²)")
    print(f"Thickness measurement noise: σ = {args.noise_um:.1f} µm (point micrometer/profilometry)")

    rng = np.random.default_rng(args.seed)
    profile = synthesize_thickness_profile(
        distribution, args.duration_s, a, b,
        noise_sigma_um=args.noise_um, rng=rng,
    )

    # The 'weighing' of the same panel: integrate the measured profile.  The
    # lab would weigh the real panel; the closure check lives in the report.
    measured = profile["measured_thickness_um"].to_numpy(float)
    gravimetric_g = float(
        (measured * distribution["segment_area_cm2"].to_numpy(float)).sum()
    ) * 7.874e-4

    result = analyze_hull_panel(
        distribution,
        measured,
        args.duration_s,
        gravimetric_mass_gain_g=gravimetric_g,
        thickness_uncertainty_um=args.noise_um,
        n_bins=5,
    )
    local = result["local_faradaic_efficiency"]
    binned = result["binned_fe_curve"]
    fit = result["logit_fit"]
    closure = result["mass_closure"]

    print("\nStrip-by-strip inverse FE (10 strips):")
    for row in local.itertuples():
        print(f"  x={row.position_cm_from_near_edge:4.1f} cm  "
              f"j={row.current_density_mA_cm2:6.1f} mA/cm²  "
              f"h={row.deposit_thickness_um:6.1f} µm  "
              f"FE={row.apparent_faradaic_efficiency:5.1%}  [{row.fe_qa_flag}]")

    print("\nBinned FE(j) calibration table (current-weighted):")
    for row in binned.itertuples():
        print(f"  j∈[{row.j_min_mA_cm2:5.1f}, {row.j_max_mA_cm2:5.1f}] mA/cm²  "
              f"FE={row.fe_current_weighted_mean:5.1%}  "
              f"({row.n_strips} strips, {row.current_fraction:4.1%} of current)")

    if np.isfinite(fit.a) and np.isfinite(fit.b):
        print(f"\nLogit fit: logit(FE) = {fit.a:.3f} + ({fit.b:.3f})·ln(j), "
              f"R²={fit.r_squared:.3f} on {fit.n_points} strips")
        print(f"  Truth:  logit(FE) = {a:.3f} + ({b:.3f})·ln(j)  "
              f"(recovered b error {(fit.b - b) / abs(b):+.1%})")
        print("  Reliability (Monte-Carlo, 10 strips, ~20-100 µm profiles): FE at "
              "reference j to ±~1% at this noise; slope to ±~0.1 (1 µm) / ±~0.2 "
              "(2 µm). More strips, profilometry, or repeated panels pin the slope.")
    else:
        print(f"\nLogit fit: {fit.note}")

    print(f"\nMass closure: profile {closure['integrated_mass_g']:.3f} g vs "
          f"gravimetric {closure['gravimetric_mass_g']:.3f} g "
          f"(ratio {closure['closure_ratio']:.3f}, "
          f"{'balanced' if closure['mass_balanced'] else 'NOT BALANCED'})")
    print(f"Implied panel FE (current-weighted local FE) = "
          f"{result['implied_panel_faradaic_efficiency']:.1%}")

    figure_path = FIG_DIR / "hull_cell_inverse_profile.png"
    _plot_profile_and_recovery(profile, local, binned, fit, figure_path)
    print(f"\n  Saved: {figure_path}")

    report = {
        "title": "Inverse Hull-cell analysis — synthetic example",
        "disclaimer": (
            "Synthetic data generated from a known FE model + measurement noise; "
            "not wet-lab deposition or profilometry data. Local FE is apparent Fe "
            "FE pending deposit composition verification."
        ),
        "method": (
            "strip FE = thickness·ρ·nF/(M·t) / primary_j(strip), primary map normalized "
            "to applied current; logit(FE) = a + b·ln(j) fit excludes QA-flagged strips; "
            "current-weighted mean of local FE must equal whole-panel gravimetric FE."
        ),
        "inputs": {
            "applied_current_A": args.current_A,
            "duration_s": args.duration_s,
            "n_segments": n_segments,
            "thickness_noise_sigma_um": args.noise_um,
            "seed": args.seed,
            "density_iron_g_cm3": 7.874,
        },
        "hull_geometry": {
            **geometry.__dict__,
            "panel_area_cm2": geometry.panel_area_cm2,
            "panel_angle_deg": geometry.panel_angle_deg,
        },
        "hull_distribution_summary": dist_summary,
        "synthetic_truth_model": {
            "fe_at_100_mA_cm2": TRUTH_FE_AT_100,
            "logit_a": a,
            "logit_b": b,
        },
        "local_faradaic_efficiency_table": local.to_dict(orient="records"),
        "binned_fe_curve_table": binned.to_dict(orient="records"),
        "logit_fit": fit.to_dict(),
        "implied_panel_faradaic_efficiency": result["implied_panel_faradaic_efficiency"],
        "fit_reliability_note": (
            "Monte-Carlo round trip on a 10-strip panel: FE at the reference j "
            "is recovered to ~±1-2 % even at point-micrometer noise (2 µm); the "
            "slope b is only resolved to ~±0.1 (1 µm profilometry) to ~±0.2 "
            "(2 µm). Pin the slope with profilometry, more strips, or repeated "
            "panels; FE(j) points from the binned table are robust."
        ),
        "mass_closure": closure,
        "qa_flag_counts": {
            flag: int((local["fe_qa_flag"] == flag).sum())
            for flag in ("ok", "above_100", "zero_deposit")
        },
        "usage_note": (
            "For real panels: measure thickness per run-sheet strip, rebuild the "
            "distribution with the same n_segments and measured geometry, pass the "
            "measured panel mass gain for the closure check, and re-run "
            "python -m models.run_hull_cell_inverse --current_A <A> with real inputs."
        ),
    }
    report_path = DATA_DIR / "hull_cell_inverse_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"  Saved: {report_path}")
    print("\n✅ Inverse Hull-cell driver complete!")


if __name__ == "__main__":
    main()
