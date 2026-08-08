"""
Non-equilibrium point-defect (vacancy/interstitial) intrinsic stress.

Why this module exists
----------------------
The prior reviews cover intrinsic stress from **crystallite coalescence**
(V4 ``coalescence_stress.py``) and from **hydrogen** (``internal_stress.py``).
A third, distinct electrocrystallization source is missing: the
**non-equilibrium supersaturation of point defects (excess vacancies / trapped
interstitials)** injected at high overpotential. These coalesce or get trapped
in the growing deposit and contribute residual stress — often in the opposite
sense to coalescence stress and strongly potential-dependent.

The physics (Round 5, C2): at high deposition overpotential, atoms deposit
faster than they reach equilibrium sites, injecting a point-defect
supersaturation. As the deposit thickens these anneal/coalesce, generating a
thickness- and overpotential-dependent stress term:

    σ_pt(η, t) ≈ g(η) · [1 − exp(−t/τ_anneal)]

This feeds directly into the peel/harvest and crack-initiation questions that
``internal_stress.py`` / ``adhesion_peel.py`` answer.

Screening flag
--------------
L1. The overpotential->defect-injection gain and the anneal time constant are
screening; calibrate against bent-strip curvature vs current density
(internal_stress.py coupon protocol).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

SCREENING_FLAG = "unvalidated (L1)"


@dataclass
class PointDefectStressParams:
    """Screening parameters for point-defect-induced intrinsic stress."""

    # Maximum defect-stress magnitude at high overpotential (MPa).
    sigma_defect_max_MPa: float = 350.0
    # Overpotential at which the defect stress reaches ~half max (V).
    eta_half_V: float = 0.25
    # Overpotential sensitivity exponent (steeper -> more potential-sensitive).
    eta_exponent: float = 1.5
    # Anneal time constant (s): how fast injected defects relax/coalesce.
    tau_anneal_s: float = 300.0
    # Thermal anneal: higher temperature shortens tau (screening Arrhenius).
    tau_ref_C: float = 60.0
    e_anneal_J_mol: float = 40.0e3
    r_gas: float = 8.314
    # Additives (brighteners) increase defect incorporation -> higher stress.
    additive_sensitivity: float = 0.5


def defect_injection_stress_MPa(
    cathodic_overpotential_V: float,
    temperature_C: float,
    deposition_time_s: float,
    additive_coverage_fraction: float = 0.0,
    params: Optional[PointDefectStressParams] = None,
) -> dict:
    """
    Point-defect intrinsic stress (MPa) at a given overpotential and thickness.

    Returns
    -------
    dict with steady_stress_MPa (full-injection value), time_constant_s,
      fractional_relaxation (0..1), and net_stress_MPa.
    """
    p = params or PointDefectStressParams()
    eta = max(float(cathodic_overpotential_V), 0.0)
    t_k = temperature_C + 273.15

    # Injection gain rises steeply with overpotential (saturating sigmoid-like).
    gain = (eta / p.eta_half_V) ** p.eta_exponent
    steady = p.sigma_defect_max_MPa * (gain / (1.0 + gain))

    # Additive sensitivity increases defect incorporation.
    theta = min(max(float(additive_coverage_fraction), 0.0), 1.0)
    steady *= (1.0 + p.additive_sensitivity * theta)

    # Thermal anneal time constant.
    tau = p.tau_anneal_s * math.exp(
        p.e_anneal_J_mol / p.r_gas * (1.0 / t_k - 1.0 / (p.tau_ref_C + 273.15)))

    frac = 1.0 - math.exp(-max(float(deposition_time_s), 0.0) / max(tau, 1e-12))
    net = steady * frac

    return {
        "steady_stress_MPa": float(steady),
        "time_constant_s": float(tau),
        "fractional_relaxation": float(min(max(frac, 0.0), 1.0)),
        "net_stress_MPa": float(net),
    }


def main() -> None:
    """CLI entrypoint for point-defect intrinsic stress."""
    print("=" * 70)
    print(" Point-Defect Intrinsic Stress (Round 5, C2)")
    print("=" * 70)
    print(f" Screening flag : {SCREENING_FLAG}")
    for eta in (0.1, 0.25, 0.4, 0.6):
        res = defect_injection_stress_MPa(eta, 60.0, deposition_time_s=900.0)
        print(f"  eta={eta:4.2f}V -> steady={res['steady_stress_MPa']:6.1f} MPa "
              f"net(t=900s)={res['net_stress_MPa']:6.1f} MPa")


if __name__ == "__main__":
    main()
