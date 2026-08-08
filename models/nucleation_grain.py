"""
Electrochemical nucleation density -> grain size -> Hall-Petch yield strength.

Why this module exists
----------------------
``mechanical_properties.py`` takes **grain size as an input** to the Hall-Petch
relation. But the grain size of an electrodeposit is set *in the cell* by
nucleation: high overpotential and additive coverage raise the nucleation-site
density, giving finer grains and higher yield strength. This module (Round 5,
C1) turns grain size into an **output of the plating recipe** (current density /
overpotential, temperature, additive coverage), using classical 3-D nucleation,
and feeds it into the same Hall-Petch law the mechanical model uses.

Screening flag
--------------
L1. Nucleation pre-factors and site densities are screening anchors; calibrate
the B (surface-energy / supersaturation) term against measured grain size vs
overpotential on the reference cell (SEM/EBSD).

References
----------
* Scharifker & Hills (1983), "Theoretical and experimental studies of multiple
  nucleation." Electrochim. Acta.
* Gamburg (2011), "Theory of Metal Electrodeposition."
* Hall-Petch for bcc Fe: sigma0 ~ 70-150 MPa, k_HP ~ 0.3-0.74 MPa·m^0.5
  (matches mechanical_properties.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

SCREENING_FLAG = "unvalidated (L1)"


@dataclass
class NucleationParams:
    """Screening parameters for 3-D nucleation and grain size of electrodeposited Fe."""

    # Atomistic 3-D nucleation: J_nuc = A0 * exp(-B / eta^2)  (1/m²·s)
    # B (V²) combines surface energy and supersaturation; screening for Fe.
    a0_1_m2_s: float = 1.0e14
    b_nucleation_V2: float = 0.020
    # Maximum nucleation-site density (1/m²); substrate + surface defects.
    n0_max_1_m2: float = 1.0e16
    # Additive (leveler/brightener) refinement: effective
    # N0 = N0_max * (1 + theta_additive * additive_refinement_factor).
    # Levelers poison growth of existing grains and promote fresh nucleation,
    # so higher coverage refines the grain (matches saccharin/thiourea behavior).
    additive_refinement_factor: float = 20.0
    # Grain size from site density: d_grain ~ N0^(-1/3), times a packing factor.
    packing_factor: float = 1.2
    # Temperature coarsening during growth (grain growth at higher T).
    t_ref_C: float = 60.0
    temp_coarse_per_C: float = 0.010
    # Hall-Petch for bcc Fe.
    sigma0_MPa: float = 90.0
    k_hp_MPa_sqrt_m: float = 0.50
    # Lower bound grain size (nm) to avoid unphysical Hall-Petch blow-up.
    d_min_nm: float = 10.0


def nucleation_density_1_m2(
    cathodic_overpotential_V: float,
    temperature_C: float,
    additive_coverage_fraction: float = 0.0,
    params: Optional[NucleationParams] = None,
) -> float:
    """
    Areal nucleation-site density (1/m²) at a given cathodic overpotential.

    Higher overpotential -> higher nucleation rate -> higher site density.
    Additives block sites, reducing the density (but per-site refinement and
    surface-smoothing effects are handled by other modules).
    """
    p = params or NucleationParams()
    eta = max(float(cathodic_overpotential_V), 0.0)

    # Nucleation rate; exponentially suppressed at low overpotential.
    if eta > 1e-6:
        j_nuc = p.a0_1_m2_s * math.exp(-p.b_nucleation_V2 / (eta * eta))
    else:
        j_nuc = 0.0

    # Site density: nucleation rate saturates toward N0_max.
    n0 = p.n0_max_1_m2 * (1.0 - math.exp(-j_nuc / max(p.a0_1_m2_s, 1e-9)))
    # Additive (leveler) refinement of the effective nucleation-site density.
    theta = min(max(float(additive_coverage_fraction), 0.0), 0.999)
    n0 *= (1.0 + theta * p.additive_refinement_factor)
    return float(max(n0, 1e4))


def grain_size_um(
    cathodic_overpotential_V: float,
    temperature_C: float,
    additive_coverage_fraction: float = 0.0,
    params: Optional[NucleationParams] = None,
) -> float:
    """
    Mean grain size (µm) predicted from nucleation density.

    d_grain ~ packing_factor * N0^(-1/3), coarsened with temperature.
    """
    p = params or NucleationParams()
    n0 = nucleation_density_1_m2(
        cathodic_overpotential_V, temperature_C, additive_coverage_fraction, p)
    d_m = p.packing_factor * n0 ** (-1.0 / 3.0)
    d_m *= math.exp(p.temp_coarse_per_C * (temperature_C - p.t_ref_C))
    d_nm = d_m * 1e9
    d_nm = max(d_nm, p.d_min_nm)
    return float(d_nm / 1e3)  # nm -> µm


def hall_petch_yield_MPa(
    grain_size_um: float,
    params: Optional[NucleationParams] = None,
) -> float:
    """Hall-Petch yield strength (MPa) from grain size, matching mechanical_properties."""
    p = params or NucleationParams()
    d_m = max(float(grain_size_um) * 1e-6, 1e-9)
    return float(p.sigma0_MPa + p.k_hp_MPa_sqrt_m / math.sqrt(d_m) / 1e6)


def recipe_to_grain_and_strength(
    cathodic_overpotential_V: float,
    temperature_C: float,
    additive_coverage_fraction: float = 0.0,
    params: Optional[NucleationParams] = None,
) -> dict:
    """
    One-stop prediction: overpotential + additive coverage -> grain size -> YS.

    This is the bridge a cell operator actually uses (j / eta, T, additive) to
    the mechanical outcome that currently starts from an assumed grain size.
    """
    d_um = grain_size_um(cathodic_overpotential_V, temperature_C,
                         additive_coverage_fraction, params)
    sigma_y = hall_petch_yield_MPa(d_um, params)
    n0 = nucleation_density_1_m2(cathodic_overpotential_V, temperature_C,
                                 additive_coverage_fraction, params)
    return {
        "nucleation_density_1_m2": n0,
        "grain_size_um": d_um,
        "hall_petch_yield_MPa": sigma_y,
    }


def main() -> None:
    """CLI entrypoint for nucleation -> grain size -> strength."""
    print("=" * 70)
    print(" Nucleation -> Grain Size -> Hall-Petch YS (Round 5, C1)")
    print("=" * 70)
    print(f" Screening flag : {SCREENING_FLAG}")
    for eta in (0.05, 0.15, 0.30, 0.5):
        for theta in (0.0, 0.5):
            res = recipe_to_grain_and_strength(eta, 60.0, additive_coverage_fraction=theta)
            print(f" eta={eta:4.2f}V  theta_add={theta:3.1f}  ->  "
                  f"grain={res['grain_size_um']:8.4f} µm  "
                  f"YS={res['hall_petch_yield_MPa']:6.1f} MPa")
        print()


if __name__ == "__main__":
    main()
