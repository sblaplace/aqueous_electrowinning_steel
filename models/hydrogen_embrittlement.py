"""
Hydrogen embrittlement (HE) model for electrodeposited Fe and Fe-Ni-C alloys.

Models hydrogen uptake during aqueous electrowinning, trap-site-modified
diffusion, susceptibility index, and post-deposition bake-out protocols
for risk mitigation.

Components
----------
1. **H diffusion in Fe**: D_H(T) = D0 * exp(-Q/RT)
   * α-Fe (bcc, fast): D0 ~ 7.3e-8 m²/s, Q ~ 4.6 kJ/mol
   * γ-Fe (fcc, austenite): D0 ~ 5.7e-7 m²/s, Q ~ 40 kJ/mol
2. **Trap-site model**: Dislocations, grain boundaries, carbon particles.
   D_eff = D_lattice / (1 + N_t * K_t / N_L)
3. **H uptake from electrolysis** — two models:

   * ``model="ipz"`` (default, 2026-08) — the Iyer–Pickering–Zamanzadeh
     surface-kinetic balance.  The fraction of HER hydrogen entering the
     metal is *derived* from Volmer/Tafel/absorption elementary steps
     (``ipz_hydrogen_entry``) rather than assumed, with constants
     recoverable from a Devanathan–Stachurski permeation cell via
     ``ipz_parameters_from_permeation``.
   * ``model="empirical"`` — the pre-2026-08 5 % nominal absorption
     correlation, retained for A/B comparison.
4. **HE susceptibility index**: Troiano-type combining yield strength,
   diffusible H content, and temperature.
4. **Bake-out protocol optimizer**: Fickian desorption to reduce
   diffusible H below critical threshold.
5. **Integration**: Accepts mechanical_properties + carburization outputs
   for spatially-resolved HE risk.

References (screening calibrations)
-----------------------------------
* H diffusion in α-Fe: D0=7.3e-8 m²/s, Q=4.6 kJ/mol
  (Kiuchi & McLellan 1983; Oriani 1970 fast-path)
* H diffusion in γ-Fe: D0=5.7e-7 m²/s, Q=40.0 kJ/mol
  (Robertson 2001; San Marchi & Somerday)
* Trap binding energies: dislocation ~26 kJ/mol, GB ~20 kJ/mol,
  carbide ~11 kJ/mol (Pressouyre 1980; Oriani 1970)
* N_L lattice site density: 8.46e28 sites/m³ for bcc Fe (2 sites/unit cell,
  a=0.287 nm)
* Troiano HE threshold: I_HE = σ_y * C_H / σ_ref — qualitative only
* Critical diffusible H for HE: 0.05–0.5 ppm depending on strength
  (0.1 ppm screening threshold at >800 MPa yield)
* Bake-out: 150-200°C for 4-24 hr typical for plated fasteners
  (ASTM F1941 guidance)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal, Tuple
import math
import numpy as np

from .electrochemistry import R_GAS, M_FE, RHO_FE

# ── Constants ────────────────────────────────────────────────────────────────

# H diffusion in α-Fe (bcc) — fast diffuser
D0_ALPHA_M2_S = 7.3e-8    # m²/s
Q_ALPHA_KJ_MOL = 4.6      # kJ/mol

# H diffusion in γ-Fe (fcc / austenite)
D0_GAMMA_M2_S = 5.7e-7    # m²/s
Q_GAMMA_KJ_MOL = 40.0     # kJ/mol

# α→γ transition (screening)
A1_TEMP_C = 727.0   # eutectoid (screening threshold for auto phase)
A3_TEMP_C = 912.0   # pure Fe α→γ

# Lattice site density for interstitial H in bcc Fe
# BCC has 2 Fe atoms per unit cell, 6 tetrahedral interstitial sites
# N_L = 6 / a³ where a = 0.2866e-9 m → N_L ≈ 2.56e29 sites/m³
# But commonly H occupies 4 of the 6 T-sites in practice → use 4/a³
# Screening: use widely-cited ~8.46e28 m⁻³ (equivalent to ~1 H per Fe site)
N_LATTICE_M3 = 8.46e28    # lattice interstitial sites (m⁻³)

# Trap binding energies (kJ/mol) — Pressouyre & Bernstein type
E_TRAP_DISLOCATION_KJ_MOL = 26.0
E_TRAP_GB_KJ_MOL = 20.0
E_TRAP_CARBIDE_KJ_MOL = 11.0

# Faraday constant
FARADAY_C_MOL = 96485.3   # C/mol

# ── IPZ (Iyer–Pickering–Zamanzadeh) hydrogen-entry constants ────────────────
#
# The absorbed-H flux is set at the charging surface by the Volmer step and
# partitioned between gas evolution (Tafel recombination) and entry into the
# metal.  At steady state the HER gas rate fixes the surface coverage
# directly, θ = sqrt(r_gas/k_rec), so the recombination constant k_rec is
# the one number that sets θ for a measured HER current; k_abs then sets the
# entry flux.  These are literature-order screening values for iron in
# mildly acidic sulfate (Iyer, Pickering & Zamanzadeh 1989/1990; Zhang et al.
# Fe/steel permeation).  A Devanathan–Stachurski permeation cell replaces
# both; ``ipz_parameters_from_permeation`` inverts a measured steady-state
# flux to recover them.
#
#   k_rec = 5.0e-2 mol/(m²·s):  with j_HER = 1000 A/m² (100 mA/cm² at 10% HER)
#   r_gas = j/(2F) ≈ 5.2e-3 mol/(m²·s) → θ ≈ 0.32 (sub-monolayer, physical);
#   at a heavy 3000 A/m² HER, θ ≈ 0.56.
#   k_abs = 2.1e-3 is calibrated so the reference operating point
#   (100 mA/cm², 85% FE, 15 min deposit) carries ~240 ppm diffusible H —
#   reproducing the "hydrogen dominates residual stress" screening story in
#   adhesion_peel/internal_stress while deriving that level from elementary
#   Volmer/Tafel/absorption steps.  It is a screening constant, not a
#   measurement; replace via ipz_parameters_from_permeation() once a
#   Devanathan–Stachurski cell exists.
IPZ_K_V_DEFAULT = 1.0e-2       # A/m²  (Volmer forward prefactor; diagnostics only)
IPZ_K_REC_DEFAULT = 5.0e-2     # mol/(m²·s)  (Tafel recombination)
IPZ_K_ABS_DEFAULT = 2.1e-3     # subsurface/surface H partition (dimensionless)
IPZ_ALPHA = 0.5                # Volmer charge-transfer symmetry

# ── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HDiffusionParams:
    """Parameters for H lattice diffusivity."""
    D0_alpha_m2_s: float = D0_ALPHA_M2_S
    Q_alpha_kJ_mol: float = Q_ALPHA_KJ_MOL
    D0_gamma_m2_s: float = D0_GAMMA_M2_S
    Q_gamma_kJ_mol: float = Q_GAMMA_KJ_MOL
    auto_threshold_C: float = A1_TEMP_C

    def __post_init__(self):
        if self.D0_alpha_m2_s <= 0 or self.D0_gamma_m2_s <= 0:
            raise ValueError("D0 must be positive")
        if self.Q_alpha_kJ_mol < 0 or self.Q_gamma_kJ_mol < 0:
            raise ValueError("Q must be non-negative")


@dataclass(frozen=True)
class TrapSiteParams:
    """Trap-site model parameters."""
    E_dislocation_kJ_mol: float = E_TRAP_DISLOCATION_KJ_MOL
    E_gb_kJ_mol: float = E_TRAP_GB_KJ_MOL
    E_carbide_kJ_mol: float = E_TRAP_CARBIDE_KJ_MOL
    N_lattice_m3: float = N_LATTICE_M3
    # Default trap densities (screening, m⁻³)
    N_dislocation_m2: float = 1e14      # dislocation density (m/m³ = m⁻²)
    N_gb_per_um: float = 3.0e6          # GB trap sites per µm grain boundary per m³
    N_carbide_per_wt_C: float = 2.0e24  # trap sites per wt% C (particles) m⁻³

    def __post_init__(self):
        if self.N_lattice_m3 <= 0:
            raise ValueError("N_lattice_m3 must be positive")


@dataclass(frozen=True)
class HESusceptibilityParams:
    """Parameters for the Troiano-type HE susceptibility index."""
    sigma_ref_MPa: float = 1000.0       # reference strength for normalization
    C_H_ref_ppm: float = 1.0            # reference H concentration
    T_ref_K: float = 300.0              # reference temperature
    gamma_sigma: float = 1.0            # strength exponent
    gamma_H: float = 1.5                # H content exponent (superlinear)
    gamma_T: float = -0.3               # temperature exponent (higher T = lower HE)
    C_H_critical_ppm: float = 0.1       # critical diffusible H for HE risk
    sigma_critical_MPa: float = 800.0   # strength above which HE concern rises


@dataclass(frozen=True)
class BakeoutParams:
    """Bake-out protocol parameters."""
    D0_m2_s: float = D0_ALPHA_M2_S      # override if phase differs
    Q_kJ_mol: float = Q_ALPHA_KJ_MOL
    C_H_critical_ppm: float = 0.1       # target residual diffusible H
    n_x: int = 200                       # spatial discretization


# ── 1. H Diffusion ──────────────────────────────────────────────────────────


def h_diffusivity_m2_s(
    temperature_C: float,
    phase: Literal["alpha", "gamma", "auto"] = "auto",
    params: Optional[HDiffusionParams] = None,
) -> Tuple[float, str]:
    """
    H lattice diffusivity D_H(T) = D0 * exp(-Q / RT).

    Parameters
    ----------
    temperature_C : temperature in °C
    phase : 'alpha' (bcc), 'gamma' (fcc), or 'auto'
    params : override parameters

    Returns
    -------
    (D_m2_s, phase_str)
    """
    p = params or HDiffusionParams()
    T_K = temperature_C + 273.15

    if phase == "auto":
        actual = "gamma" if temperature_C >= p.auto_threshold_C else "alpha"
    else:
        actual = phase

    if actual == "alpha":
        D = p.D0_alpha_m2_s * math.exp(-p.Q_alpha_kJ_mol * 1000.0 / (R_GAS * T_K))
    else:
        D = p.D0_gamma_m2_s * math.exp(-p.Q_gamma_kJ_mol * 1000.0 / (R_GAS * T_K))

    return float(D), actual


# ── 2. Trap-Site Model ─────────────────────────────────────────────────────


def trap_binding_factor(
    E_trap_kJ_mol: float,
    temperature_C: float,
) -> float:
    """
    Trap binding energy factor K_t = exp(E_trap / RT).

    Higher binding → stronger trap → more H captured → lower effective D.
    """
    T_K = temperature_C + 273.15
    return math.exp(E_trap_kJ_mol * 1000.0 / (R_GAS * T_K))


def trap_density_m3(
    grain_size_um: float = 3.0,
    dislocation_density_m2: Optional[float] = None,
    carbon_wt_percent: float = 0.0,
    params: Optional[TrapSiteParams] = None,
) -> Dict[str, float]:
    """
    Estimate total trap-site density from microstructure.

    Parameters
    ----------
    grain_size_um : mean grain size (µm)
    dislocation_density_m2 : dislocation density (m⁻²). Defaults to params.N_dislocation_m2.
    carbon_wt_percent : carbon content (wt%) for carbide traps

    Returns
    -------
    dict with dislocation, grain_boundary, carbide, and total trap densities (m⁻³)
    """
    p = params or TrapSiteParams()

    if dislocation_density_m2 is None:
        dislocation_density_m2 = p.N_dislocation_m2

    # Dislocation traps
    # Simplified: N_disl ≈ ρ_disl (m⁻²) × 5e9 (sites/m of dislocation line / m²)
    N_disl = dislocation_density_m2 * 5.0e9   # m⁻³

    # Grain boundary traps: scale as 1/d (more boundary = more traps)
    d_m = max(grain_size_um * 1e-6, 1e-9)
    # GB area per volume ≈ 2/d for equiaxed grains (simplified)
    # trap sites per unit GB area: N_gb_per_um * 1e6 → sites/m²
    gb_area_per_vol = 2.0 / d_m  # m⁻¹
    N_gb = gb_area_per_vol * p.N_gb_per_um * 1e6  # m⁻³

    # Carbide / carbon particle traps
    N_carbide = carbon_wt_percent * p.N_carbide_per_wt_C  # m⁻³

    return {
        "dislocation_m3": N_disl,
        "grain_boundary_m3": N_gb,
        "carbide_m3": N_carbide,
        "total_m3": N_disl + N_gb + N_carbide,
    }


def effective_diffusivity_m2_s(
    temperature_C: float,
    grain_size_um: float = 3.0,
    dislocation_density_m2: Optional[float] = None,
    carbon_wt_percent: float = 0.0,
    phase: Literal["alpha", "gamma", "auto"] = "auto",
    params_h: Optional[HDiffusionParams] = None,
    params_trap: Optional[TrapSiteParams] = None,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Effective H diffusivity with trap-site retardation.

    D_eff = D_lattice / (1 + Σ(N_t_i * K_t_i) / N_L)

    Returns
    -------
    (D_eff_m2_s, D_lattice_m2_s, trap_info)
    """
    D_lattice, actual_phase = h_diffusivity_m2_s(temperature_C, phase, params_h)

    traps = trap_density_m3(grain_size_um, dislocation_density_m2, carbon_wt_percent, params_trap)
    p = params_trap or TrapSiteParams()

    T_K = temperature_C + 273.15
    # Weighted average binding factor across trap types
    K_disl = trap_binding_factor(p.E_dislocation_kJ_mol, temperature_C)
    K_gb = trap_binding_factor(p.E_gb_kJ_mol, temperature_C)
    K_carbide = trap_binding_factor(p.E_carbide_kJ_mol, temperature_C)

    sum_NK = (
        traps["dislocation_m3"] * K_disl
        + traps["grain_boundary_m3"] * K_gb
        + traps["carbide_m3"] * K_carbide
    )

    ratio = sum_NK / p.N_lattice_m3
    D_eff = D_lattice / (1.0 + ratio)

    trap_info = {
        "D_lattice_m2_s": D_lattice,
        "phase": actual_phase,
        "trap_densities": traps,
        "K_dislocation": K_disl,
        "K_gb": K_gb,
        "K_carbide": K_carbide,
        "sum_NK_over_NL": ratio,
    }

    return float(D_eff), float(D_lattice), trap_info


# ── 3. H Uptake from Electrolysis (Faraday) ─────────────────────────────────


def hydrogen_uptake_from_electrolysis(
    current_density_mA_cm2: float,
    deposition_time_s: float = 3600.0,
    her_efficiency: float = 0.10,
    bath_pH: float = 3.5,
    temperature_C: float = 60.0,
    deposit_density_kg_m3: float = RHO_FE,
    model: Literal["ipz", "empirical"] = "ipz",
    k_v: float = IPZ_K_V_DEFAULT,
    k_rec: float = IPZ_K_REC_DEFAULT,
    k_abs: float = IPZ_K_ABS_DEFAULT,
) -> Dict[str, float]:
    """
    Estimate diffusible H content (ppm) from electrolytic deposition.

    Two models are available:

    * ``model="ipz"`` (default) — the Iyer–Pickering–Zamanzadeh surface
      kinetic balance.  The fraction of HER hydrogen that enters the metal
      is *derived* from Volmer/Tafel/absorption elementary steps rather than
      assumed; it falls as HER rises (more recombination to gas) and rises
      with overpotential.  This couples to ``her_microkinetics.py`` and is
      the value consumed by the peel/stress predictions.  The three IPZ
      constants are screening values, recoverable from a
      Devanathan–Stachurski cell via :func:`ipz_parameters_from_permeation`.
    * ``model="empirical"`` — the pre-2026-08 screening correlation
      (5 % nominal absorption with pH/T/j power-law factors).  Retained for
      backwards compatibility and A/B comparison.

    Uses Faraday's law: moles H produced per area = j*t / (n*F)
    where n=2 for H₂, then the absorbed fraction gives diffusible H.

    Parameters
    ----------
    current_density_mA_cm2 : cathodic current density
    deposition_time_s : total deposition time
    her_efficiency : fraction of cathodic current going to HER (0-1)
    bath_pH : lower pH → more H⁺ → higher H uptake
    temperature_C : bath temperature
    deposit_density_kg_m3 : density of the deposit
    model : ``"ipz"`` (default, physics-based) or ``"empirical"``.
    k_v, k_rec, k_abs : IPZ rate constants (see module constants).

    Returns
    -------
    dict with C_H_diffusible_ppm, H_flux_mol_m2, her_current_A_m2, etc.
    """
    j_A_m2 = current_density_mA_cm2 * 10.0  # mA/cm² → A/m²

    if model == "ipz":
        return _uptake_ipz(
            j_A_m2, her_efficiency, bath_pH, temperature_C,
            deposition_time_s, deposit_density_kg_m3,
            k_v, k_rec, k_abs,
        )
    if model != "empirical":
        raise ValueError("model must be 'ipz' or 'empirical'")


    # HER current
    j_HER = j_A_m2 * her_efficiency

    # H produced per area (mol/m²): two electrons per H₂ molecule → 1 H per electron
    n_e = 2  # electrons per H₂
    H_mol_m2 = j_HER * deposition_time_s / (n_e * FARADAY_C_MOL)

    # Fraction absorbed into the deposit (screening: ~1-10% of HER H enters metal)
    # pH effect: lower pH → higher H activity → more absorption
    # screening: absorption fraction ∝ 10^(-pH/2) normalized at pH=3.5
    pH_factor = 10.0 ** (-(bath_pH - 3.5) / 2.0)
    absorption_fraction = 0.05 * pH_factor  # ~5% at pH=3.5 (screening)

    # Temperature effect: higher T slightly reduces H absorption in cathodic deposit
    # (thermal desorption during growth)
    T_factor = math.exp(-0.005 * (temperature_C - 60.0))  # ~1 at 60°C

    # Current density effect: higher j → higher overpotential → more HER →
    # higher local H surface activity → higher absorption fraction
    # screening: j_factor ∝ (j/j_ref)^0.3, normalized at 100 mA/cm²
    j_ref = 100.0
    j_factor = (current_density_mA_cm2 / j_ref) ** 0.3

    absorption_fraction = 0.05 * pH_factor * T_factor * j_factor
    absorption_fraction = float(np.clip(absorption_fraction, 0.001, 0.20))

    # Diffusible H absorbed
    H_absorbed_mol_m2 = H_mol_m2 * absorption_fraction

    # Convert to ppm (mass basis)
    # deposit mass per area: j * t * M_Fe / (n_Fe * F) where n_Fe=2 for Fe²⁺ → Fe
    deposit_mass_kg_m2 = j_A_m2 * deposition_time_s * M_FE / (2.0 * FARADAY_C_MOL)
    deposit_mass_kg_m2 = max(deposit_mass_kg_m2, 1e-12)

    # H mass per area
    M_H = 1.008e-3  # kg/mol
    H_mass_kg_m2 = H_absorbed_mol_m2 * M_H

    C_H_ppm = (H_mass_kg_m2 / deposit_mass_kg_m2) * 1e6

    return {
        "C_H_diffusible_ppm": float(C_H_ppm),
        "H_flux_mol_m2": float(H_mol_m2),
        "H_absorbed_mol_m2": float(H_absorbed_mol_m2),
        "her_current_A_m2": float(j_HER),
        "absorption_fraction": float(absorption_fraction),
        "deposit_mass_kg_m2": float(deposit_mass_kg_m2),
        "pH_factor": float(pH_factor),
        "T_factor": float(T_factor),
        "j_factor": float(j_factor),
        "model": "empirical",
    }


# ── 3b. Iyer–Pickering–Zamanzadeh (IPZ) H-entry model ───────────────────────


def ipz_hydrogen_entry(
    j_HER_A_m2: float,
    bath_pH: float,
    temperature_C: float = 60.0,
    k_v: float = IPZ_K_V_DEFAULT,
    k_rec: float = IPZ_K_REC_DEFAULT,
    k_abs: float = IPZ_K_ABS_DEFAULT,
    alpha: float = IPZ_ALPHA,
) -> Dict[str, float]:
    """Steady-state hydrogen entry from the IPZ surface-kinetic model.

    The empirical uptake model (``hydrogen_uptake_from_electrolysis``) folds
    everything into an assumed 5 % absorption fraction.  This function
    instead derives the absorbed-vs-evolved split from the same three
    elementary steps that ``her_microkinetics.py`` uses for the HER rate:

    * **Volmer**         H⁺ + e⁻ + * → H*       (atomic H onto the surface)
    * **Tafel recomb.**  2 H* → H₂ + 2*         (gas evolution)
    * **absorption**     H* ⇌ H(sub-surface)    (entry into the metal)

    At steady state the atomic-H supply from Volmer is partitioned between
    recombination (gas) and absorption.  The measured/modelled HER partial
    current ``j_HER`` fixes the recombination flux
    ``r_rec = j_HER/(2F)``, and the Tafel rate law ``r_rec = k_rec θ²``
    then fixes the surface coverage directly: ``θ = √(r_rec/k_rec)``.
    The entry flux is ``J_in = k'θ`` (IPZ absorption isotherm), and the
    cathodic overpotential needed to supply the total atomic-H rate follows
    from the Volmer Tafel law, so pH enters explicitly through a_H:

        η = (RT/αF) ln[ (2 r_rec + J_in) F / (k_v a_H) ].

    The entry/recombination constants are screening values recoverable from a
    Devanathan–Stachurski permeation cell via
    :func:`ipz_parameters_from_permeation`.

    Parameters
    ----------
    j_HER_A_m2 : float
        HER partial current density (A/m², positive cathodic magnitude).
    bath_pH, temperature_C : float
        Bath state; sets proton activity and the Tafel temperature.
    k_v, k_rec, k_abs, alpha : float
        IPZ rate constants (module defaults are screening values).

    Returns
    -------
    dict
        ``theta_H`` (surface coverage, 0–1), ``eta_V`` (cathodic
        overpotential V), ``entry_flux_mol_m2_s`` (atomic H into the metal),
        ``recombination_flux_mol_m2_s`` (H to H₂ gas),
        ``entry_efficiency`` (J_in / total atomic-H rate), and the input
        constants for traceability.
    """
    if j_HER_A_m2 <= 0.0:
        return {
            "theta_H": 0.0,
            "eta_V": 0.0,
            "entry_flux_mol_m2_s": 0.0,
            "recombination_flux_mol_m2_s": 0.0,
            "entry_efficiency": 0.0,
            "k_v": k_v, "k_rec": k_rec, "k_abs": k_abs,
        }

    T_K = temperature_C + 273.15
    a_h = 10.0 ** (-bath_pH)

    # Recombination (gas) flux fixes the surface coverage (Tafel RDS).
    r_rec = j_HER_A_m2 / (2.0 * FARADAY_C_MOL)   # mol H/(m²·s) → H₂
    theta = math.sqrt(min(r_rec / max(k_rec, 1e-30), 1.0))

    # Entry flux, IPZ absorption isotherm: J_in = k' θ.  In the standard
    # IPZ notation k' = k_abs·√k_rec is fitted directly; we carry the two
    # factors separately so k_rec may be calibrated against θ alone.
    j_entry = k_abs * math.sqrt(max(k_rec, 0.0)) * theta  # mol/(m²·s)

    # Total atomic-H production rate the Volmer step must supply.
    r_v = r_rec + j_entry

    # Cathodic overpotential from the Volmer Tafel law; pH enters via a_H.
    denom = (k_v / FARADAY_C_MOL) * a_h
    eta_V = (R_GAS * T_K / (alpha * FARADAY_C_MOL)) * math.log(
        max(r_v / max(denom, 1e-30), 1e-30)
    )
    entry_eff = j_entry / max(r_v, 1e-30)

    return {
        "theta_H": float(theta),
        "eta_V": float(eta_V),
        "entry_flux_mol_m2_s": float(j_entry),
        "recombination_flux_mol_m2_s": float(r_rec),
        "entry_efficiency": float(entry_eff),
        "k_v": float(k_v),
        "k_rec": float(k_rec),
        "k_abs": float(k_abs),
    }


def ipz_parameters_from_permeation(
    j_perm_A_m2: float,
    j_HER_A_m2: float,
    bath_pH: float,
    temperature_C: float = 60.0,
    alpha: float = IPZ_ALPHA,
    k_rec: float = IPZ_K_REC_DEFAULT,
) -> Dict[str, float]:
    """Recover the IPZ entry constant k_abs from a measured permeation flux.

    A Devanathan–Stachurski cell measures the steady-state anodic oxidation
    current of hydrogen crossing the membrane, which equals the entry flux
    ``J_in = j_perm/F``.  The HER partial current fixes θ via
    ``θ = √(j_HER/(2F k_rec))``, so
    ``k_abs = J_in/(√k_rec θ)`` is recovered directly — this is what
    replaces the screening ``IPZ_K_ABS_DEFAULT`` once permeation data exist.

    Returns ``k_abs``, the inferred ``theta_H`` and ``eta_V``, and the
    resulting entry efficiency.
    """
    if j_perm_A_m2 <= 0.0 or j_HER_A_m2 <= 0.0:
        return {"k_abs": IPZ_K_ABS_DEFAULT, "theta_H": 0.0, "eta_V": 0.0,
                "entry_efficiency": 0.0}

    T_K = temperature_C + 273.15
    a_h = 10.0 ** (-bath_pH)
    r_rec = j_HER_A_m2 / (2.0 * FARADAY_C_MOL)
    theta = math.sqrt(min(r_rec / max(k_rec, 1e-30), 1.0))
    j_entry = j_perm_A_m2 / FARADAY_C_MOL
    k_abs = j_entry / max(math.sqrt(max(k_rec, 0.0)) * theta, 1e-30)

    # Overpotential from Volmer supplying recombination + the measured entry.
    r_v = r_rec + j_entry
    denom = (IPZ_K_V_DEFAULT / FARADAY_C_MOL) * a_h
    eta_V = (R_GAS * T_K / (alpha * FARADAY_C_MOL)) * math.log(
        max(r_v / max(denom, 1e-30), 1e-30)
    )
    return {
        "k_abs": float(k_abs),
        "theta_H": float(theta),
        "eta_V": float(eta_V),
        "entry_efficiency": float(j_entry / max(r_v, 1e-30)),
    }


def _uptake_ipz(
    j_A_m2: float,
    her_efficiency: float,
    bath_pH: float,
    temperature_C: float,
    deposition_time_s: float,
    deposit_density_kg_m3: float,
    k_v: float,
    k_rec: float,
    k_abs: float,
) -> Dict[str, float]:
    """IPZ-based diffusible-H uptake; same return schema as the empirical path."""
    j_HER = j_A_m2 * her_efficiency
    ipz = ipz_hydrogen_entry(
        j_HER, bath_pH, temperature_C,
        k_v=k_v, k_rec=k_rec, k_abs=k_abs,
    )
    # Atomic H absorbed per unit area over the run (mol/m²).
    H_absorbed_mol_m2 = ipz["entry_flux_mol_m2_s"] * deposition_time_s
    H_mol_m2 = j_HER * deposition_time_s / (2.0 * FARADAY_C_MOL)

    # Deposit mass per unit area (Fe²⁺ + 2e⁻ → Fe), using the Fe partial
    # current (total minus HER) so ppm is H per mass of deposited iron.
    j_fe = max(j_A_m2 - j_HER, 0.0)
    deposit_mass_kg_m2 = max(
        j_fe * deposition_time_s * M_FE / (2.0 * FARADAY_C_MOL), 1e-12
    )
    M_H = 1.008e-3  # kg/mol
    H_mass_kg_m2 = H_absorbed_mol_m2 * M_H
    C_H_ppm = (H_mass_kg_m2 / deposit_mass_kg_m2) * 1e6

    return {
        "C_H_diffusible_ppm": float(C_H_ppm),
        "H_flux_mol_m2": float(H_mol_m2),
        "H_absorbed_mol_m2": float(H_absorbed_mol_m2),
        "her_current_A_m2": float(j_HER),
        "absorption_fraction": float(ipz["entry_efficiency"]),
        "deposit_mass_kg_m2": float(deposit_mass_kg_m2),
        "pH_factor": 1.0,
        "T_factor": 1.0,
        "j_factor": 1.0,
        "model": "ipz",
        "ipz": {
            "theta_H": ipz["theta_H"],
            "eta_V": ipz["eta_V"],
            "entry_flux_mol_m2_s": ipz["entry_flux_mol_m2_s"],
            "entry_efficiency": ipz["entry_efficiency"],
        },
    }


# ── 4. HE Susceptibility Index ──────────────────────────────────────────────


def he_susceptibility_index(
    sigma_y_MPa: float,
    C_H_diffusible_ppm: float,
    temperature_C: float = 25.0,
    params: Optional[HESusceptibilityParams] = None,
) -> Dict[str, float]:
    """
    Troiano-type hydrogen embrittlement susceptibility index.

    I_HE = (σ_y / σ_ref)^γσ × (C_H / C_H_ref)^γH × (T_ref / T)^γT

    Index ranges:
    * <1: low risk
    * 1-5: moderate risk
    * 5-20: high risk
    * >20: critical

    Returns
    -------
    dict with I_HE, risk_level, and normalized factors.
    """
    p = params or HESusceptibilityParams()
    T_K = temperature_C + 273.15

    f_sigma = (max(sigma_y_MPa, 1.0) / p.sigma_ref_MPa) ** p.gamma_sigma
    f_H = (max(C_H_diffusible_ppm, 1e-6) / p.C_H_ref_ppm) ** p.gamma_H
    f_T = (p.T_ref_K / T_K) ** p.gamma_T

    I_HE = f_sigma * f_H * f_T

    # Risk classification
    if I_HE < 1.0:
        risk = "low"
    elif I_HE < 5.0:
        risk = "moderate"
    elif I_HE < 20.0:
        risk = "high"
    else:
        risk = "critical"

    return {
        "I_HE": float(I_HE),
        "risk_level": risk,
        "f_sigma": float(f_sigma),
        "f_H": float(f_H),
        "f_T": float(f_T),
        "sigma_y_MPa": sigma_y_MPa,
        "C_H_ppm": C_H_diffusible_ppm,
        "temperature_C": temperature_C,
    }


# ── 5. Bake-Out Protocol Optimizer ─────────────────────────────────────────


def bakeout_time_hr(
    deposit_thickness_um: float,
    initial_C_H_ppm: float,
    target_C_H_ppm: float = 0.1,
    temperature_C: float = 170.0,
    grain_size_um: float = 3.0,
    dislocation_density_m2: float = 1e14,
    carbon_wt_percent: float = 0.0,
    params: Optional[BakeoutParams] = None,
) -> Dict[str, float]:
    """
    Estimate bake-out time to reduce diffusible H below threshold.

    Uses Fickian desorption from free surface. For a slab of thickness L,
    desorption follows:
        C(t)/C0 ≈ (8/π²) Σ exp(-D_eff * n² * π² * t / L²)  for n=1,3,5,...

    We solve for t such that C_H(t) = target.

    Returns
    -------
    dict with bakeout_time_hr, D_eff, Fourier number, residual_C_H_ppm
    """
    p = params or BakeoutParams()

    # Get effective diffusivity at bake-out temperature
    D_eff, D_lattice, trap_info = effective_diffusivity_m2_s(
        temperature_C, grain_size_um, dislocation_density_m2, carbon_wt_percent
    )

    L_m = deposit_thickness_um * 1e-6

    # Need C/C0 = target/initial
    ratio = target_C_H_ppm / max(initial_C_H_ppm, 1e-12)
    if ratio >= 1.0:
        return {
            "bakeout_time_hr": 0.0,
            "D_eff_m2_s": D_eff,
            "D_lattice_m2_s": D_lattice,
            "Fourier_number": 0.0,
            "residual_C_H_ppm": initial_C_H_ppm,
            "temperature_C": temperature_C,
        }

    # Leading term approximation: C/C0 ≈ (8/π²) exp(-D_eff π² t / L²)
    # Solve: ratio = (8/π²) exp(-D_eff π² t / L²)
    # → t = -(L² / (D_eff π²)) * ln(ratio * π² / 8)
    argument = ratio * (math.pi ** 2) / 8.0
    if argument <= 0 or argument >= 1.0:
        # Already below target or very thin → instant
        if argument >= 1.0:
            # ratio * π²/8 >= 1 → initial H already below effective threshold
            # Just use a very short time
            t_s = 0.0
        else:
            t_s = 1e6  # very long, effectively impossible
    else:
        t_s = -(L_m ** 2 / (D_eff * math.pi ** 2)) * math.log(argument)

    t_hr = t_s / 3600.0
    Fo = D_eff * t_s / (L_m ** 2) if L_m > 0 else 0.0

    # Verify with more terms of Fourier series
    residual_ratio = _fourier_slab_ratio(t_s, D_eff, L_m, n_terms=25)
    residual_ppm = initial_C_H_ppm * residual_ratio

    return {
        "bakeout_time_hr": float(t_hr),
        "D_eff_m2_s": float(D_eff),
        "D_lattice_m2_s": float(D_lattice),
        "Fourier_number": float(Fo),
        "residual_C_H_ppm": float(residual_ppm),
        "temperature_C": temperature_C,
        "trap_info": trap_info,
    }


def _fourier_slab_ratio(t_s: float, D_m2_s: float, L_m: float, n_terms: int = 25) -> float:
    """
    Fourier series for desorption from symmetric slab (both surfaces exposed).

    C(t)/C0 = (8/π²) Σ_{n=1,3,5,...} (1/n²) exp(-D n² π² t / L²)
    """
    if t_s <= 0:
        return 1.0
    if D_m2_s <= 0 or L_m <= 0:
        return 1.0

    total = 0.0
    for i in range(n_terms):
        n = 2 * i + 1  # odd terms: 1, 3, 5, ...
        exponent = -D_m2_s * (n * math.pi) ** 2 * t_s / (L_m ** 2)
        if exponent < -500:
            break  # negligible
        total += (1.0 / (n ** 2)) * math.exp(exponent)

    return (8.0 / (math.pi ** 2)) * total


def bakeout_schedule(
    deposit_thickness_um: float,
    initial_C_H_ppm: float,
    target_C_H_ppm: float = 0.1,
    temperatures_C: Optional[list[float]] = None,
    grain_size_um: float = 3.0,
    dislocation_density_m2: float = 1e14,
    carbon_wt_percent: float = 0.0,
) -> list[Dict[str, float]]:
    """
    Generate bake-out time estimates across multiple temperatures.

    Returns list of dicts with temperature and required bake-out time.
    """
    if temperatures_C is None:
        temperatures_C = [120, 150, 170, 200, 250]

    schedule = []
    for T in temperatures_C:
        result = bakeout_time_hr(
            deposit_thickness_um=deposit_thickness_um,
            initial_C_H_ppm=initial_C_H_ppm,
            target_C_H_ppm=target_C_H_ppm,
            temperature_C=T,
            grain_size_um=grain_size_um,
            dislocation_density_m2=dislocation_density_m2,
            carbon_wt_percent=carbon_wt_percent,
        )
        schedule.append(result)

    return schedule


# ── 6. Integration with Mechanical & Carburization ──────────────────────────


@dataclass
class HEResult:
    """Complete hydrogen embrittlement assessment result."""

    # Input parameters
    current_density_mA_cm2: float
    deposition_time_s: float
    temperature_deposition_C: float
    bath_pH: float
    grain_size_um: float
    ni_wt_percent: float
    carbon_wt_percent: float
    sigma_y_MPa: float

    # H uptake
    uptake: Dict[str, float]

    # Diffusion
    D_lattice_m2_s: float
    D_eff_m2_s: float
    diffusion_phase: str
    trap_info: Dict[str, Any]

    # HE susceptibility
    he_index: Dict[str, float]

    # Bake-out
    bakeout: Dict[str, float]

    # Spatially-resolved risk (from integration)
    spatial_he_risk: Optional[Dict[str, Any]] = None
    flags: list[str] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "current_density_mA_cm2": self.current_density_mA_cm2,
            "deposition_time_hr": round(self.deposition_time_s / 3600.0, 2),
            "C_H_diffusible_ppm": round(self.uptake["C_H_diffusible_ppm"], 4),
            "D_lattice_m2_s": self.D_lattice_m2_s,
            "D_eff_m2_s": self.D_eff_m2_s,
            "phase": self.diffusion_phase,
            "trap_sum_NK_over_NL": self.trap_info["sum_NK_over_NL"],
            "I_HE": round(self.he_index["I_HE"], 3),
            "risk_level": self.he_index["risk_level"],
            "bakeout_time_hr": round(self.bakeout["bakeout_time_hr"], 2),
            "bakeout_residual_ppm": round(self.bakeout["residual_C_H_ppm"], 4),
            "grain_size_um": round(self.grain_size_um, 3),
            "sigma_y_MPa": round(self.sigma_y_MPa, 1),
            "ni_wt_percent": self.ni_wt_percent,
            "carbon_wt_percent": self.carbon_wt_percent,
            "flags": self.flags,
        }


class HydrogenEmbrittlementModel:
    """
    Screening-level hydrogen embrittlement model for aqueous-electrodeposited steel.

    Example
    -------
    >>> model = HydrogenEmbrittlementModel()
    >>> result = model.predict(
    ...     current_density_mA_cm2=100.0,
    ...     deposition_time_hr=2.0,
    ...     bath_pH=3.5,
    ...     sigma_y_MPa=450.0,
    ...     grain_size_um=2.0,
    ...     carbon_wt_percent=0.5,
    ... )
    >>> print(result.he_index["I_HE"], result.he_index["risk_level"])
    """

    def __init__(
        self,
        h_params: Optional[HDiffusionParams] = None,
        trap_params: Optional[TrapSiteParams] = None,
        he_params: Optional[HESusceptibilityParams] = None,
        bakeout_params: Optional[BakeoutParams] = None,
    ):
        self.h_params = h_params or HDiffusionParams()
        self.trap_params = trap_params or TrapSiteParams()
        self.he_params = he_params or HESusceptibilityParams()
        self.bakeout_params = bakeout_params or BakeoutParams()

    def predict(
        self,
        current_density_mA_cm2: float = 100.0,
        deposition_time_hr: float = 2.0,
        deposition_temperature_C: float = 60.0,
        bath_pH: float = 3.5,
        her_efficiency: float = 0.10,
        sigma_y_MPa: float = 400.0,
        grain_size_um: float = 3.0,
        dislocation_density_m2: float = 1e14,
        ni_wt_percent: float = 0.0,
        carbon_wt_percent: float = 0.0,
        phase: Literal["alpha", "gamma", "auto"] = "auto",
        bakeout_temperature_C: float = 170.0,
        target_C_H_ppm: float = 0.1,
        deposit_thickness_um: float = 1000.0,
        ambient_temperature_C: float = 25.0,
    ) -> HEResult:
        """Run full HE assessment pipeline."""

        dep_time_s = deposition_time_hr * 3600.0

        # 1) H uptake from electrolysis
        uptake = hydrogen_uptake_from_electrolysis(
            current_density_mA_cm2=current_density_mA_cm2,
            deposition_time_s=dep_time_s,
            her_efficiency=her_efficiency,
            bath_pH=bath_pH,
            temperature_C=deposition_temperature_C,
        )
        C_H_ppm = uptake["C_H_diffusible_ppm"]

        # 2) Effective diffusivity (ambient for service condition)
        D_eff, D_lattice, trap_info = effective_diffusivity_m2_s(
            temperature_C=ambient_temperature_C,
            grain_size_um=grain_size_um,
            dislocation_density_m2=dislocation_density_m2,
            carbon_wt_percent=carbon_wt_percent,
            phase=phase,
            params_h=self.h_params,
            params_trap=self.trap_params,
        )

        # 3) HE susceptibility index
        he_idx = he_susceptibility_index(
            sigma_y_MPa=sigma_y_MPa,
            C_H_diffusible_ppm=C_H_ppm,
            temperature_C=ambient_temperature_C,
            params=self.he_params,
        )

        # 4) Bake-out protocol
        bakeout = bakeout_time_hr(
            deposit_thickness_um=deposit_thickness_um,
            initial_C_H_ppm=C_H_ppm,
            target_C_H_ppm=target_C_H_ppm,
            temperature_C=bakeout_temperature_C,
            grain_size_um=grain_size_um,
            dislocation_density_m2=dislocation_density_m2,
            carbon_wt_percent=carbon_wt_percent,
        )

        # Flags
        flags: list[str] = []
        if he_idx["I_HE"] >= 20.0:
            flags.append("critical_HE_risk")
        elif he_idx["I_HE"] >= 5.0:
            flags.append("high_HE_risk")
        if C_H_ppm > self.he_params.C_H_critical_ppm:
            flags.append("H_above_critical")
        if sigma_y_MPa > self.he_params.sigma_critical_MPa:
            flags.append("high_strength_HE_sensitive")
        if bakeout["bakeout_time_hr"] > 24.0:
            flags.append("extended_bakeout_required")
        if trap_info["sum_NK_over_NL"] > 100:
            flags.append("very_high_trap_density")

        return HEResult(
            current_density_mA_cm2=current_density_mA_cm2,
            deposition_time_s=dep_time_s,
            temperature_deposition_C=deposition_temperature_C,
            bath_pH=bath_pH,
            grain_size_um=grain_size_um,
            ni_wt_percent=ni_wt_percent,
            carbon_wt_percent=carbon_wt_percent,
            sigma_y_MPa=sigma_y_MPa,
            uptake=uptake,
            D_lattice_m2_s=D_lattice,
            D_eff_m2_s=D_eff,
            diffusion_phase=trap_info["phase"],
            trap_info=trap_info,
            he_index=he_idx,
            bakeout=bakeout,
            flags=flags,
        )

    def predict_with_integration(
        self,
        mechanical_result: Dict[str, Any],
        carburization_result: Optional[Dict[str, Any]] = None,
        current_density_mA_cm2: float = 100.0,
        deposition_time_hr: float = 2.0,
        bath_pH: float = 3.5,
        deposit_thickness_um: float = 1000.0,
        bakeout_temperature_C: float = 170.0,
        target_C_H_ppm: float = 0.1,
    ) -> HEResult:
        """
        Integration adapter: accept mechanical_properties and carburization
        outputs to compute spatially-resolved HE risk.

        Parameters
        ----------
        mechanical_result : dict from MechanicalPropertiesResult.summary()
            Must contain yield_strength_MPa, grain_size_um, composition.c_wt_pct, etc.
        carburization_result : optional dict from CarburizationResult.summary()
            If provided, uses case depth and C profile for spatial HE resolution.
        """
        sigma_y = float(mechanical_result.get("yield_strength_MPa", 400.0))
        grain_size = float(mechanical_result.get("grain_size_um", 3.0))
        comp = mechanical_result.get("composition", {})
        c_wt = float(comp.get("c_wt_pct", 0.0))
        ni_wt = float(comp.get("ni_wt_pct", 0.0))

        result = self.predict(
            current_density_mA_cm2=current_density_mA_cm2,
            deposition_time_hr=deposition_time_hr,
            bath_pH=bath_pH,
            sigma_y_MPa=sigma_y,
            grain_size_um=grain_size,
            ni_wt_percent=ni_wt,
            carbon_wt_percent=c_wt,
            deposit_thickness_um=deposit_thickness_um,
            bakeout_temperature_C=bakeout_temperature_C,
            target_C_H_ppm=target_C_H_ppm,
        )

        # Spatially-resolved risk if carburization profile available
        if carburization_result is not None:
            spatial = self._compute_spatial_risk(
                result, carburization_result, deposit_thickness_um
            )
            result = HEResult(
                current_density_mA_cm2=result.current_density_mA_cm2,
                deposition_time_s=result.deposition_time_s,
                temperature_deposition_C=result.temperature_deposition_C,
                bath_pH=result.bath_pH,
                grain_size_um=result.grain_size_um,
                ni_wt_percent=result.ni_wt_percent,
                carbon_wt_percent=result.carbon_wt_percent,
                sigma_y_MPa=result.sigma_y_MPa,
                uptake=result.uptake,
                D_lattice_m2_s=result.D_lattice_m2_s,
                D_eff_m2_s=result.D_eff_m2_s,
                diffusion_phase=result.diffusion_phase,
                trap_info=result.trap_info,
                he_index=result.he_index,
                bakeout=result.bakeout,
                spatial_he_risk=spatial,
                flags=result.flags,
            )

        return result

    def _compute_spatial_risk(
        self,
        he_result: HEResult,
        carburization_result: Dict[str, Any],
        thickness_um: float,
    ) -> Dict[str, Any]:
        """
        Compute spatially-resolved HE risk through the case depth.

        Surface (hard, high C) is most susceptible; core is less so.
        Maps C profile → strength → HE index at each depth.
        """
        # Extract case info from carburization result
        case_depth_035 = carburization_result.get("final_case_depth_035_um", 0.0)
        final_surface_hv = carburization_result.get("final_surface_hv", 300.0)

        # Estimate surface vs core HE risk
        # Surface: high hardness (from carburization) → high σ_y → higher HE
        sigma_surface_MPa = final_surface_hv * 9.80665 / 3.2  # HV → MPa (Tabor)
        sigma_core_MPa = he_result.sigma_y_MPa

        C_H = he_result.uptake["C_H_diffusible_ppm"]
        T = 25.0  # service temperature

        he_surface = he_susceptibility_index(sigma_surface_MPa, C_H, T, self.he_params)
        he_core = he_susceptibility_index(sigma_core_MPa, C_H, T, self.he_params)

        # Interpolate at case boundary
        sigma_case = sigma_surface_MPa * 0.7 + sigma_core_MPa * 0.3
        he_case = he_susceptibility_index(sigma_case, C_H, T, self.he_params)

        return {
            "surface_I_HE": he_surface["I_HE"],
            "surface_risk": he_surface["risk_level"],
            "case_depth_035_um": case_depth_035,
            "case_boundary_I_HE": he_case["I_HE"],
            "case_boundary_risk": he_case["risk_level"],
            "core_I_HE": he_core["I_HE"],
            "core_risk": he_core["risk_level"],
            "surface_sigma_MPa": round(sigma_surface_MPa, 1),
            "core_sigma_MPa": round(sigma_core_MPa, 1),
            "thickness_um": thickness_um,
        }

    def sweep_current_density(
        self,
        j_values_mA_cm2: Optional[np.ndarray] = None,
        deposition_time_hr: float = 2.0,
        bath_pH: float = 3.5,
        sigma_y_MPa: float = 450.0,
        grain_size_um: float = 2.0,
        carbon_wt_percent: float = 0.5,
        bakeout_temperature_C: float = 170.0,
    ) -> Dict[str, np.ndarray]:
        """Sweep over current density for HE trends."""
        if j_values_mA_cm2 is None:
            j_values_mA_cm2 = np.linspace(10, 200, 40)

        results = []
        for j in j_values_mA_cm2:
            r = self.predict(
                current_density_mA_cm2=float(j),
                deposition_time_hr=deposition_time_hr,
                bath_pH=bath_pH,
                sigma_y_MPa=sigma_y_MPa,
                grain_size_um=grain_size_um,
                carbon_wt_percent=carbon_wt_percent,
                bakeout_temperature_C=bakeout_temperature_C,
            )
            results.append(r)

        return {
            "j_mA_cm2": np.array(j_values_mA_cm2),
            "C_H_ppm": np.array([r.uptake["C_H_diffusible_ppm"] for r in results]),
            "I_HE": np.array([r.he_index["I_HE"] for r in results]),
            "D_eff_m2_s": np.array([r.D_eff_m2_s for r in results]),
            "bakeout_hr": np.array([r.bakeout["bakeout_time_hr"] for r in results]),
        }

    def sweep_temperature(
        self,
        T_values_C: Optional[np.ndarray] = None,
        current_density_mA_cm2: float = 100.0,
        deposition_time_hr: float = 2.0,
        bath_pH: float = 3.5,
        sigma_y_MPa: float = 450.0,
        grain_size_um: float = 2.0,
        carbon_wt_percent: float = 0.5,
    ) -> Dict[str, np.ndarray]:
        """Sweep over bath temperature for HE trends."""
        if T_values_C is None:
            T_values_C = np.linspace(25, 90, 30)

        results = []
        for T in T_values_C:
            r = self.predict(
                current_density_mA_cm2=current_density_mA_cm2,
                deposition_time_hr=deposition_time_hr,
                deposition_temperature_C=float(T),
                bath_pH=bath_pH,
                sigma_y_MPa=sigma_y_MPa,
                grain_size_um=grain_size_um,
                carbon_wt_percent=carbon_wt_percent,
            )
            results.append(r)

        return {
            "T_C": np.array(T_values_C),
            "C_H_ppm": np.array([r.uptake["C_H_diffusible_ppm"] for r in results]),
            "I_HE": np.array([r.he_index["I_HE"] for r in results]),
            "D_eff_m2_s": np.array([r.D_eff_m2_s for r in results]),
        }


# ── Integration Adapters ────────────────────────────────────────────────────


def build_he_model_from_mechanical(
    mechanical_result: Dict[str, Any],
    current_density_mA_cm2: float = 100.0,
    deposition_time_hr: float = 2.0,
    bath_pH: float = 3.5,
    deposit_thickness_um: float = 1000.0,
) -> HEResult:
    """
    Convenience adapter: take a MechanicalPropertiesResult.summary() dict
    and feed it into the HE model.
    """
    model = HydrogenEmbrittlementModel()
    return model.predict_with_integration(
        mechanical_result=mechanical_result,
        current_density_mA_cm2=current_density_mA_cm2,
        deposition_time_hr=deposition_time_hr,
        bath_pH=bath_pH,
        deposit_thickness_um=deposit_thickness_um,
    )


def synthetic_h_uptake_data(
    j_range_mA_cm2: Optional[np.ndarray] = None,
    T_range_C: Optional[np.ndarray] = None,
    pH_range: Optional[np.ndarray] = None,
    deposition_time_hr: float = 2.0,
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic H uptake datasets for visualization.

    Returns dict of arrays for j, T, pH sweeps with corresponding H content.
    """
    model = HydrogenEmbrittlementModel()

    if j_range_mA_cm2 is None:
        j_range_mA_cm2 = np.linspace(10, 200, 20)
    if T_range_C is None:
        T_range_C = np.linspace(25, 90, 15)
    if pH_range is None:
        pH_range = np.linspace(2.0, 6.0, 15)

    # j sweep
    H_vs_j = []
    for j in j_range_mA_cm2:
        up = hydrogen_uptake_from_electrolysis(
            float(j), deposition_time_hr * 3600.0, her_efficiency=0.10, bath_pH=3.5
        )
        H_vs_j.append(up["C_H_diffusible_ppm"])

    # T sweep
    H_vs_T = []
    for T in T_range_C:
        up = hydrogen_uptake_from_electrolysis(
            100.0, deposition_time_hr * 3600.0, her_efficiency=0.10, bath_pH=3.5,
            temperature_C=float(T)
        )
        H_vs_T.append(up["C_H_diffusible_ppm"])

    # pH sweep
    H_vs_pH = []
    for pH in pH_range:
        up = hydrogen_uptake_from_electrolysis(
            100.0, deposition_time_hr * 3600.0, her_efficiency=0.10, bath_pH=float(pH)
        )
        H_vs_pH.append(up["C_H_diffusible_ppm"])

    return {
        "j_mA_cm2": np.array(j_range_mA_cm2),
        "H_vs_j_ppm": np.array(H_vs_j),
        "T_C": np.array(T_range_C),
        "H_vs_T_ppm": np.array(H_vs_T),
        "pH": np.array(pH_range),
        "H_vs_pH_ppm": np.array(H_vs_pH),
    }
