"""
Driver for leveler / additive Langmuir adsorption kinetics (CHEM_PHYS_REVIEW §2.6).

Replaces the single ``saccharin_g_L`` stress knob with mechanism-level additive
kinetics and answers the morphology question the card reframes: not "coarse or
fine?" but "which additive package reaches structural grade?".

Prints, for a set of named additive packages (id → g/L):
  * per-additive Langmuir coverage θ and areal density Γ (mol/m²);
  * the aggregate nucleation-rate multiplier (Γ-dependent grain refinement);
  * the H-recombination fraction / overpotential reduction (Γ-dependent);
  * intrinsic-stress relief fraction and carbon-incorporation blocking;
  * a structural-grade score + verdict, ranked best-first.

Run::

    python -m models.run_leveler_kinetics
    aq-steel-leveler
"""

from __future__ import annotations

from models.leveler_kinetics import (
    compare_packages,
    resolve_package,
    structural_grade_score,
)

# Named candidate packages (bath additive concentrations in g/L) to screen.
PACKAGES = {
    "baseline-sac-only": {"saccharin": 1.5},
    "sac+chloride": {"saccharin": 1.5, "chloride": 2.0},
    "sac+thiourea+PEG": {"saccharin": 1.5, "thiourea": 0.05, "peg": 0.2},
    "full-package": {
        "saccharin": 1.5,
        "thiourea": 0.05,
        "peg": 0.2,
        "coumarin": 0.05,
        "chloride": 1.0,
    },
    "over-leveled": {"thiourea": 0.3, "coumarin": 0.3, "peg": 0.5},
}


def main() -> None:
    print("=" * 76)
    print("ADDITIVE / LEVELER LANGMUIR KINETICS  (CHEM_PHYS_REVIEW §2.6)")
    print("=" * 76)

    # Per-additive isotherm detail on the full package.
    full = PACKAGES["full-package"]
    pkg = resolve_package(full)
    print("\nFull-package Langmuir adsorption surface state:")
    print(f"  {'additive':<10}{'c(g/L)':>8}{'θ':>8}{'Γ (mol/m²)':>14}")
    for aid, c in sorted(full.items()):
        theta = pkg.theta_by_id[aid]
        gamma = pkg.gamma_by_id[aid]
        print(f"  {aid:<10}{c:>8.2f}{theta:>8.3f}{gamma:>14.2e}")
    print(f"  joint organic coverage θ_org = {pkg.theta_organic:.3f}")
    print(f"  nucleation multiplier      = {pkg.nucleation_multiplier:.2f}")
    print(f"  H-recomb fraction          = {pkg.h_recomb_fraction:.2f}")
    print(f"  H-recomb overpot reduction = {pkg.h_recomb_overpotential_reduction_V*1e3:.1f} mV")
    print(f"  intrinsic-stress relief    = {pkg.stress_relief_fraction:.2f}")
    print(f"  carbon-incorporation block = {pkg.carbon_incorporation_blocking:.2f}")

    print("\nStructural-grade ranking (which package gets us to structural?):")
    print(f"  {'package':<22}{'relief':>8}{'grain':>8}{'H':>6}{'score':>8}  verdict")
    for row in compare_packages(PACKAGES)["ranked"]:
        ax = row["axes"]
        print(
            f"  {row['name']:<22}{ax['relief']:>8.2f}{ax['grain_refinement']:>8.2f}"
            f"{ax['h_removal']:>6.2f}{row['score']:>8.3f}  {row['verdict']}"
        )

    best = compare_packages(PACKAGES)["ranked"][0]
    print("\nRecommended first package to try in the reference cell:")
    det = structural_grade_score(best["package"])
    print(f"  {best['name']}  →  {det['package']}")
    print(f"  score {det['score']:.3f} ({det['verdict']}); "
          f"axes {det['axes']}")


if __name__ == "__main__":
    main()
