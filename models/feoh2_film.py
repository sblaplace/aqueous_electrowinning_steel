"""
Fe(OH)₂ passivation-film thickness ODE for iron electrowinning (screening, L1).

The Nernst–Planck solve in :mod:`models.diffusion_layer_1d` treats the cathode
as bare Fe.  Its ``precipitation_sink`` diagnostic computes a volumetric
Fe(OH)₂ precipitation flux but then discards it — it is a supersaturation
*flag* ("is the surface about to precipitate?"), not a *film*.  This module
turns that flux into a real coupled film: an ODE for the Fe(OH)₂ film thickness
whose growth is the precipitation flux and whose removal is dissolution, and
which then feeds a surface-overpotential term (10 s of mV) back into the cell
voltage.

For a screening engine this is deliberately closed-form separable: at each
operating point the flux is treated as quasi-steady against the much slower
film-thickness dynamics, giving an equilibrium film thickness
``δ_ss = Q_precip / k_diss_eff``.  The ODE integrator is provided for
transient/start-up use (e.g. the pulse-reverse "anodic strip the film" branch
in :mod:`models.pulse`), and the steady-state thickness is what
``DiffusionLayer1D`` consumes for the overpotential.

Chemistry
---------
Growth (from the precipitation-sink flux):
    Fe²⁺ + 2 OH⁻ → Fe(OH)₂(s),     Q_precip (mol/m²/s) → dδ/dt = Q_precip·M/ρ

Dissolution — two removal routes, each first-order in δ:

* Acid dissolution:   Fe(OH)₂ + 2H⁺ → Fe²⁺ + 2H₂O
      Q_acid = k_acid · δ · (c_H,surf / c_H,ref)
* Fe²⁺-promoted reductive dissolution (the cathode reducing the film back
  to Fe metal; promoted by the local Fe²⁺ present at the metal/film edge):
      Fe(OH)₂(s) + 2H⁺ + 2e⁻ → Fe(s) + 2H₂O
      Q_red = k_red · δ · (c_Fe²⁺,surf / c_Fe²⁺,ref)

Film overpotential (ohm drop across the oxide):
    η_film = j · R_film,   R_film = δ / κ_film   (Ω·m²)

Quick estimate at the reference passivating point (see ``steady_state``): with
κ_film ≈ 1e-2 S/m (a wet, porous Fe(OH)₂ film, ~3 orders below bulk electrolyte
conductivity) and δ ≈ 0.1–1 µm at j = 100 mA/cm², the overpotential lands in the
10 s-of-mV range the CHEM_PHYS_REVIEW Tier 1.3 asks for.

SCREENING HONESTY
-----------------
Every constant here is a fitted screening value, not a measured one — there is
no wet-lab passivation data yet (the twin is L0/L1).  The two fitted constants
(κ_film and the dissolution-rate ratio k_red/k_acid) were tuned so that the
reference precipitating operating point produces a physically plausible
passivating film (δ ~ 0.1–1 µm) and 10 s-of-mV surface overpotential.  They are
NOT gate evidence; they exist to make the peel/stress/current-efficiency story
mechanistic rather than a bare "S > 1" flag.  Tune them against first
polarization/EIS film data when it arrives.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

# ─── Screening constants (see SCREENING HONESTY above) ──────────────
# Molar mass and density of Fe(OH)₂.
M_FEOH2_KG_MOL = 89.86e-3          # kg/mol
RHO_FEOH2_KG_M3 = 3400.0           # kg/m³ (screening, ~3.4 g/cm³)

# Film ionic/electronic conductivity (S/m), screening central value.
# Bulk concentrated electrolyte is ~10 S/m; a wet porous hydrous-oxide film is
# roughly 3 orders lower.  This is the primary fitted knob locating the
# 10 s-of-mV overpotential at sub-µm thicknesses.
FEOH2_KAPPA_S_M = 1.0e-2

# Fraction of the volumetric precipitation-sink flux that lands as a coherent
# surface film.  The precipitation flux in :mod:`models.diffusion_layer_1d` is
# the TOTAL Fe(OH)₂ precipitating in the boundary layer (bulk sludge included);
# only a small share deposits as the thin passivating layer at the metal/oxide
# interface, the rest stays suspended as sludge.  Treating the whole flux as
# film growth overstates δ by orders of magnitude (drives it to 100 s of µm and
# hundreds of volts).  Screening fit 1e-3 so the reference precipitating point
# lands at δ ~ 0.1–1 µm and η_film ~ 10 s-of-mV (see SCREENING HONESTY).
FILM_DEPOSITION_FRACTION = 1.0e-3

# Dissolution rate constants (1/s at the species reference) and reference
# concentrations (mol/m³).  Fe²⁺-promoted reductive dissolution dominates at the
# neutral/high-pH surface (where c_H is negligible); acid dissolution is the
# cleanup route when the surface is still acid.
FILM_K_ACID_S = 0.5                # per-unit c_H/c_H_ref
FILM_K_RED_S = 2.0e-2              # per-unit c_Fe2/c_Fe2_ref
C_H_REF_MOL_M3 = 100.0             # pH 1 reference for the acid term
C_FE2_REF_MOL_M3 = 1000.0          # 1 M Fe²⁺ reference for the reductive term

FLAG = "unvalidated (L1)"


def film_growth_rate_m_s(
    precip_flux_mol_m2_s: float,
    deposition_fraction: float = FILM_DEPOSITION_FRACTION,
) -> float:
    """Thickness growth rate (m/s) from an Fe(OH)₂ precipitation flux.

    Only ``deposition_fraction`` of the volumetric precipitation flux lands as
    a coherent surface film; the rest is bulk sludge (see FILM_DEPOSITION_FRACTION).
    """
    frac = min(max(float(deposition_fraction), 0.0), 1.0)
    return float(precip_flux_mol_m2_s) * frac * M_FEOH2_KG_MOL / RHO_FEOH2_KG_M3


def acid_dissolution_m_s(
    thickness_m: float,
    c_h_surf_mol_m3: float,
    k_acid_s: float = FILM_K_ACID_S,
    c_h_ref_mol_m3: float = C_H_REF_MOL_M3,
) -> float:
    """Acid-dissolution film removal rate (m/s), first order in δ and c_H."""
    if k_acid_s <= 0.0:
        return 0.0
    return float(max(thickness_m, 0.0)) * k_acid_s * max(c_h_surf_mol_m3, 0.0) / c_h_ref_mol_m3


def reductive_dissolution_m_s(
    thickness_m: float,
    c_fe2_surf_mol_m3: float,
    k_red_s: float = FILM_K_RED_S,
    c_fe2_ref_mol_m3: float = C_FE2_REF_MOL_M3,
) -> float:
    """Fe²⁺-promoted reductive-dissolution removal rate (m/s), ord in δ and c_Fe2."""
    if k_red_s <= 0.0:
        return 0.0
    return float(max(thickness_m, 0.0)) * k_red_s * max(c_fe2_surf_mol_m3, 0.0) / c_fe2_ref_mol_m3


def film_dissolution_m_s(
    thickness_m: float,
    c_h_surf_mol_m3: float,
    c_fe2_surf_mol_m3: float,
    k_acid_s: float = FILM_K_ACID_S,
    k_red_s: float = FILM_K_RED_S,
) -> float:
    """Total film-removal rate (m/s) = acid + reductive dissolution."""
    return (
        acid_dissolution_m_s(thickness_m, c_h_surf_mol_m3, k_acid_s)
        + reductive_dissolution_m_s(thickness_m, c_fe2_surf_mol_m3, k_red_s)
    )


def film_ode(
    _t: float,
    thickness_m: float,
    precip_flux_mol_m2_s: float,
    c_h_surf_mol_m3: float,
    c_fe2_surf_mol_m3: float,
    k_acid_s: float = FILM_K_ACID_S,
    k_red_s: float = FILM_K_RED_S,
    deposition_fraction: float = FILM_DEPOSITION_FRACTION,
) -> float:
    """dδ/dt = growth − (acid + reductive) dissolution (m/s)."""
    g = film_growth_rate_m_s(precip_flux_mol_m2_s, deposition_fraction)
    d = film_dissolution_m_s(
        thickness_m, c_h_surf_mol_m3, c_fe2_surf_mol_m3, k_acid_s, k_red_s
    )
    return g - d


def steady_state_thickness_m(
    precip_flux_mol_m2_s: float,
    c_h_surf_mol_m3: float,
    c_fe2_surf_mol_m3: float,
    k_acid_s: float = FILM_K_ACID_S,
    k_red_s: float = FILM_K_RED_S,
    deposition_fraction: float = FILM_DEPOSITION_FRACTION,
) -> float:
    """Equilibrium δ = Q_precip / k_diss_eff.

    The dissolution removal is linear in δ (both terms), so at steady state
    ``Q_precip = δ·(k_acid·c_H/c_H_ref + k_red·c_Fe2/c_Fe2_ref)`` and δ solves
    directly.  Zero when there is no net precipitation.
    """
    g_m_s = film_growth_rate_m_s(max(precip_flux_mol_m2_s, 0.0), deposition_fraction)
    if g_m_s <= 0.0:
        return 0.0
    k_eff = (
        k_acid_s * max(c_h_surf_mol_m3, 0.0) / C_H_REF_MOL_M3
        + k_red_s * max(c_fe2_surf_mol_m3, 0.0) / C_FE2_REF_MOL_M3
    )
    if k_eff <= 1e-30:
        return float("inf")  # no removal: film grows without bound (flag honesty)
    return float(g_m_s / k_eff)


def film_overpotential_V(
    current_density_A_m2: float,
    thickness_m: float,
    kappa_S_m: float = FEOH2_KAPPA_S_M,
) -> float:
    """Ohmic overpotential across the film: η = j·δ/κ (V, non-negative)."""
    if kappa_S_m <= 0.0:
        return float("nan")
    return float(max(current_density_A_m2, 0.0) * max(thickness_m, 0.0) / kappa_S_m)


def integrate_film(
    precip_flux_fn,
    c_h_surf_fn,
    c_fe2_surf_fn,
    thickness_0_m: float = 0.0,
    t_span_s=(0.0, 3600.0),
    k_acid_s: float = FILM_K_ACID_S,
    k_red_s: float = FILM_K_RED_S,
    deposition_fraction: float = FILM_DEPOSITION_FRACTION,
    rtol: float = 1e-6,
    atol: float = 1e-12,
):
    """Time-step the film-thickness ODE for transient/start-up use.

    Parameters are callables of time (s) returning the precip flux
    (mol/m²/s), surface H⁺ (mol/m³) and surface Fe²⁺ (mol/m³) respectively —
    this is the seam for coupling to a time-varying Nernst–Planck run or a
    pulse waveform.  Returns the full ``solve_ivp`` solution object.
    """
    def rhs(t, y):
        return film_ode(
            t, float(y[0]),
            float(precip_flux_fn(t)), float(c_h_surf_fn(t)), float(c_fe2_surf_fn(t)),
            k_acid_s, k_red_s, deposition_fraction,
        )

    return solve_ivp(
        rhs, tuple(t_span_s), np.array([float(thickness_0_m)]),
        method="LSODA", rtol=rtol, atol=atol, dense_output=False,
    )


@dataclass(frozen=True)
class FilmDiagnostics:
    """Steady-state film picture at one precipitation point."""
    thickness_m: float
    growth_rate_m_s: float
    dissolution_rate_m_s: float
    film_overpotential_V: float
    kappa_S_m: float
    flag: str = FLAG

    @property
    def thickness_um(self) -> float:
        return self.thickness_m * 1e6

    @property
    def film_overpotential_mV(self) -> float:
        return self.film_overpotential_V * 1000.0


def film_diagnostics(
    precip_flux_mol_m2_s: float,
    c_h_surf_mol_m3: float,
    c_fe2_surf_mol_m3: float,
    current_density_A_m2: float | None = None,
    kappa_S_m: float = FEOH2_KAPPA_S_M,
    k_acid_s: float = FILM_K_ACID_S,
    k_red_s: float = FILM_K_RED_S,
    deposition_fraction: float = FILM_DEPOSITION_FRACTION,
) -> FilmDiagnostics:
    """One-stop steady-state film diagnostic for an operating point.

    ``current_density_A_m2`` is optional: when supplied the film overpotential
    is included, otherwise it is reported as 0 for a pure film-chemistry query.
    """
    delta = steady_state_thickness_m(
        precip_flux_mol_m2_s, c_h_surf_mol_m3, c_fe2_surf_mol_m3,
        k_acid_s, k_red_s, deposition_fraction,
    )
    if np.isinf(delta):
        return FilmDiagnostics(
            thickness_m=float("inf"), growth_rate_m_s=0.0, dissolution_rate_m_s=0.0,
            film_overpotential_V=float("inf"), kappa_S_m=kappa_S_m,
        )
    g = film_growth_rate_m_s(max(precip_flux_mol_m2_s, 0.0), deposition_fraction)
    d = film_dissolution_m_s(delta, c_h_surf_mol_m3, c_fe2_surf_mol_m3, k_acid_s, k_red_s)
    eta = film_overpotential_V(current_density_A_m2 or 0.0, delta, kappa_S_m)
    return FilmDiagnostics(
        thickness_m=delta, growth_rate_m_s=g, dissolution_rate_m_s=d,
        film_overpotential_V=eta, kappa_S_m=kappa_S_m,
    )
