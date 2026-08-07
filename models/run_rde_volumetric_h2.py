"""Q3 runner: RDE + volumetric H2 to separate Fe from HER kinetics.

Uses :mod:`models.rde_volumetric_h2` to demonstrate the full two-step +
independent-closure procedure on synthetic data: measure the HER branch FIRST
on the Fe-free supporting electrolyte (the #34 dominant uncertainty), then fit
the Fe branch with HER held fixed on the Fe-bath RDE, then confirm the split by
volumetric H2 charge-ledger closure. Writes a JSON report and one figure.

Run::

    python -m models.run_rde_volumetric_h2
    aq-steel-rde-volumetric-h2
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .rde_volumetric_h2 import (
    her_equilibrium_potential,
    model_scope,
    measurement_spec,
    self_test,
    simulate_her_free_bath_polarization,
    simulate_fe_bath_rde_polarization,
    fit_her_from_free_bath,
    fit_fe_kinetics_given_her,
)
from .electrochemistry import E0_FE

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments" / "data"
FIGURES = ROOT / "docs" / "figures"

TRUE = {
    "b_her": 0.140, "i0_her": 2.0e-3,
    "b_fe": 0.120, "i0_fe": 50.0,
    "fe_E_eq": E0_FE, "pH": 2.0, "T_C": 25.0, "fe_conc_M": 1.0, "D": 7.2e-10,
}


def _fig_her_first() -> Path:
    """Figure: Fe-free HER branch fit + Fe-bath Fe fit given fixed HER."""
    E_eq_her = her_equilibrium_potential(TRUE["pH"], TRUE["T_C"])
    E_free = np.linspace(E_eq_her - 0.80, E_eq_her - 0.05, 60)
    free = simulate_her_free_bath_polarization(
        E_free, i0_her_A_m2=TRUE["i0_her"], b_her_V_dec=TRUE["b_her"],
        E_eq_her_V=E_eq_her)
    fit_h = fit_her_from_free_bath(free["potentials_V"], free["i_her_A_m2"],
                                   E_eq_her_V=E_eq_her)

    E_fe = np.linspace(-0.50, -1.05, 80)
    omegas = [400.0, 1600.0, 2500.0]
    fe_bath = simulate_fe_bath_rde_polarization(
        E_fe, np.array(omegas), fe_i0_A_m2=TRUE["i0_fe"], fe_tafel_V=TRUE["b_fe"],
        fe_E_eq_V=E0_FE, b_her_V_dec=TRUE["b_her"], i0_her_A_m2=TRUE["i0_her"],
        E_eq_her_V=E_eq_her, fe_conc_M=TRUE["fe_conc_M"], D_m2_s=TRUE["D"])
    df = fe_bath["frame"]
    fit_fe = fit_fe_kinetics_given_her(
        df["potential_V"].to_numpy(float), df["i_total_A_m2"].to_numpy(float),
        i_lim_A_m2=df["i_lim_A_m2"].to_numpy(float),
        b_her_V_dec=fit_h["b_her_V_dec"], i0_her_A_m2=fit_h["i0_her_A_m2"],
        E_eq_her_V=E_eq_her, E_eq_fe_V=E0_FE)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=150)

    # Left: Fe-free HER branch (measured first)
    ax = axes[0]
    ax.plot(free["potentials_V"], free["i_her_A_m2"], "o",
            color="#1b9e77", ms=4, label="Fe-free bath (HER only)")
    Esel = np.linspace(E_eq_her - 0.80, E_eq_her - 0.05, 100)
    ax.plot(Esel, fit_h["i0_her_A_m2"] * 10 ** ((E_eq_her - Esel) / fit_h["b_her_V_dec"]),
            "--", color="#d95f02", lw=1.3,
            label=f"fit b={fit_h['b_her_V_dec']:.3f}, i0={fit_h['i0_her_A_m2']:.3g}")
    ax.set_yscale("log")
    ax.set_xlabel("E (V vs SHE)")
    ax.set_ylabel("i_HER (A/m²)")
    ax.set_title(f"Step 1 — HER FIRST (no Fe): b={fit_h['b_her_V_dec']:.3f} V/dec "
                 f"(true {TRUE['b_her']:.3f})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    # Right: Fe-bath total curves + Fe fit (HER fixed)
    ax = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(omegas)))
    for o, c in zip(omegas, colors):
        sub = df[df["omega_rpm"] == o]
        ax.plot(sub["potential_V"], sub["i_total_A_m2"], "-", color=c,
                lw=1.3, label=f"{int(o)} rpm (sim)")
    Esel2 = np.linspace(-0.50, -1.02, 100)
    i_lim_1600 = float(fe_bath["i_lim_A_m2"][1])
    fe_model = fit_fe["fe_i0_A_m2"] * 10 ** ((E0_FE - Esel2) / fit_fe["fe_tafel_V_dec"])
    fe_cap = fe_model * i_lim_1600 / (fe_model + i_lim_1600)
    her_fit = fit_h["i0_her_A_m2"] * 10 ** ((E_eq_her - Esel2) / fit_h["b_her_V_dec"])
    ax.plot(Esel2, fe_cap + her_fit, "k--", lw=1.6,
            label=f"Fe+HER model (b_fe={fit_fe['fe_tafel_V_dec']:.3f})")
    ax.set_yscale("log")
    ax.set_xlabel("E (V vs SHE)")
    ax.set_ylabel("i_total (A/m²)")
    ax.set_title("Step 3 — Fe fit with HER FIXED from step 1")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    out = FIGURES / "rde_volumetric_h2_her_first.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> dict:
    print("=== Q3 — RDE + volumetric H2: separate Fe from HER (measure HER FIRST) ===")
    print()
    res = self_test(seed=7, verbose=True)

    fig = _fig_her_first()
    report = {
        "title": "Q3 — RDE + volumetric H2: separate Fe from HER (measure HER Tafel first)",
        "resolves": "#34 dominant uncertainty: her_tafel_V",
        "purpose": (
            "Resolve the single dominant L0 unknown (the HER Tafel slope) by "
            "measuring the HER branch FIRST in the Fe-free bath, then fitting Fe "
            "with HER held fixed on the Fe-bath RDE, then confirming the split by "
            "volumetric charge-ledger closure."
        ),
        "true_parameters": TRUE,
        "self_test": res,
        "measurement_spec": measurement_spec(),
        "model_scope": model_scope(),
        "figure": f"docs/figures/{fig.name}",
        "status": "L0 synthetic validation — NOT gate evidence",
    }

    out = DATA / "rde_volumetric_h2_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\nWrote {fig.relative_to(ROOT)}")
    print(f"Wrote {out.relative_to(ROOT)}")
    return report


if __name__ == "__main__":
    main()
