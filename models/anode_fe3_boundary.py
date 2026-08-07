"""
Fe³⁺ boundary layer at the OER anode (screening, L1).

:mod:`models.anode` models OER and CER kinetics but treats the anode surface
as if the anolyte were Fe²⁺-free: a DSA oxidising Fe²⁺ to Fe³⁺ is ignored.
This module adds that boundary layer, closing the CHEM_PHYS_REVIEW Tier 1.3
"Fe³⁺ boundary layer at the anode" item.  It is the *anode* end of the same
Fe³⁺/Fe(OH)₃ story that :mod:`models.fe3_shuttle` carries on the *cathode*
side: whatever Fe³⁺ the anode makes is the shuttle source that eventually
steals cathode current efficiency and precipitates Fe(OH)₃ sludge.

Chemistry / transport (closed-form screening)
---------------------------------------------
1. Anolyte Fe²⁺ diffuses through the anode Nernst film and is oxidised:
       Fe²⁺ → Fe³⁺ + e⁻,   E₀(Fe³⁺/Fe²⁺) = 0.771 V vs SHE
   At the operating OER potential (>1.2 V) this is strongly anodic and
   thermodynamically downhill, so the surface Fe²⁺ → ~0 and the oxidation is
   mass-transfer limited: Q = k_m,Fe²⁺ · [Fe²⁺]_bulk.  The associated anodic
   current i_ox = F·Q (1 e⁻ per Fe²⁺) is a parasitic current that competes
   with OER/CER for the applied current.
2. Fe³⁺ hydrolyses at the surface (Fe³⁺ + 3H₂O → Fe(OH)₃(s) + 3H⁺).  A
   screening fraction f_hyd of the produced Fe³⁺ precipitates as Fe(OH)₃
   sludge, releasing 3 H⁺ per Fe³⁺.  This acid efflux lowers the local surface
   pH below bulk.
3. A lower surface pH raises the OER equilibrium potential
   (E_eq = 1.229 − 0.0591·pH) by 0.0591·ΔpH, i.e. a raised OER overpotential
   (10 s of mV at a ~1-unit pH drop).
4. Fe³⁺ not precipitated leaves the film into the anolyte as the shuttle /
   bulk-Fe³⁺ source.

The surface pH is closed with a single-film mass balance on H⁺:
    [H⁺]_surf = [H⁺]_bulk + (3·f_hyd·Q) / k_m,H
(the OER's own H⁺ production is not double-counted here — it is the
background acidity the Hittorf/salt-polarization term in `anode.py` already
carries; this is the *incremental* acid from Fe³⁺ hydrolysis).

SCREENING HONESTY
-----------------
All constants are fitted screening values, not measurements (the twin is
L0/L1; there is no wet anolyte-Fe³⁺ data yet).  The two fitted knobs —
``fraction_hydrolysed`` and the choice to omit OER-H⁺ from the pH closure —
were tuned so the reference 1 M Fe²⁺ anolyte gives a ~1-unit local pH drop and
a 10 s-of-mV OER overpotential raise, matching CHEM_PHYS_REVIEW Tier 1.3.
They are NOT gate evidence; tune against polarization / pH-probe-in-film data
when it arrives.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .electrochemistry import FARADAY, R_GAS
from .fe3_shuttle import LOGKSP_FEOH3

# ─── Screening constants (see SCREENING HONESTY above) ──────────────
# Fe³⁺/Fe²⁺ redox equilibrium (V vs SHE):  Fe²⁺ → Fe³⁺ + e⁻.
E0_FE3_FE2 = 0.771

# Aqueous diffusivities (m²/s), 25 °C anchor, same screening family as the
# catholyte Fe³⁺ value in fe3_shuttle.  H⁺ is ~1-1.5 orders faster than Fe³⁺.
D_FE2_REF_M2_S = 5.5e-10
D_FE3_REF_M2_S = 5.5e-10
D_H_REF_M2_S = 9.3e-9

# Fraction of anode-produced Fe³⁺ that hydrolyses to Fe(OH)₃ sludge in the
# film (the rest leaves as the shuttle source).  Screening central value.
FRACTION_HYDROLYSED = 0.5

# Cap on the local pH drop so the screening closure cannot drive pH_surf
# negative (physically the hydrolysis acid is buffered; this keeps η finite).
MAX_PH_DROP = 6.0

SCREENING_FLAG = "unvalidated (L1)"


def fe3_solubility_cap_M(pH: float) -> float:
    """[Fe³⁺] solubility cap from Fe(OH)₃ hydrolysis at a given pH (mol/L).

    cap = Ksp/[OH⁻]³ with pOH = 14 − pH (25 °C water autoprotolysis) — same
    form as :func:`models.fe3_shuttle.fe3_solubility_cap_M`.
    """
    pOH = 14.0 - pH
    return 10.0 ** (LOGKSP_FEOH3 + 3.0 * pOH)


@dataclass(frozen=True)
class Fe3BoundaryResult:
    """Anode Fe³⁺ boundary-layer picture at one operating point (opt-in).

    All fields are zero when the flag is off or the anolyte has no Fe²⁺,
    since the model is strictly a perturbation on the bare-OER anode.
    """

    fe2_oxidation_flux_mol_m2_s: float   # Q: anolyte Fe²⁺ oxidised (mol/m²/s)
    fe2_oxidation_current_A_m2: float    # parasitic anodic current F·Q (A/m²)
    fe3_surface_M: float                 # surface Fe³⁺ (mol/L), screening
    fe3_solubility_cap_M: float          # Fe(OH)₃ cap at the surface pH
    ph_drop: float                       # pH_bulk − pH_surf (≥ 0)
    surface_pH: float                    # local anode-surface pH
    oer_overpotential_raise_V: float     # Δη from the ΔpH (10 s of mV)
    feoh3_sludge_flux_mol_m2_s: float    # f_hyd·Q — anode Fe(OH)₃ sludge
    shuttle_source_flux_mol_m2_s: float  # (1−f_hyd)·Q — Fe³⁺ leaving the film
    flag: str = SCREENING_FLAG

    @property
    def active(self) -> bool:
        return self.fe2_oxidation_flux_mol_m2_s > 0.0


def mass_transfer_coeff(diffusivity_m2_s: float, boundary_layer_m: float) -> float:
    """Nernst-film mass-transfer coefficient k_m = D/δ (m/s)."""
    return float(diffusivity_m2_s) / max(float(boundary_layer_m), 1e-12)


def ph_from_h_mol_m3(h_mol_m3: float) -> float:
    """Convert [H⁺] (mol/m³) to pH, hugging 0 below 1e-12 mol/m³ (~pH 15)."""
    return float(-math.log10(max(h_mol_m3, 0.0) / 1000.0))


def fe3_boundary_analysis(
    fe2_bulk_M: float,
    ph_bulk: float,
    boundary_layer_m: float = 5e-5,
    temperature_C: float = 60.0,
    fraction_hydrolysed: float = FRACTION_HYDROLYSED,
    d_fe2_m2_s: float = D_FE2_REF_M2_S,
    d_fe3_m2_s: float = D_FE3_REF_M2_S,
    d_h_m2_s: float = D_H_REF_M2_S,
    max_ph_drop: float = MAX_PH_DROP,
) -> Fe3BoundaryResult:
    """Closed-form Fe³⁺ boundary-layer analysis for one anolyte operating point.

    Parameters
    ----------
    fe2_bulk_M : float
        Bulk anolyte Fe²⁺ concentration (mol/L).  0 → an inert result (all
        fields zero), so wiring this in is safe for a bare-OER bath.
    ph_bulk : float
        Bulk anolyte pH.
    boundary_layer_m : float
        Anode Nernst film thickness (m), same as `AnodeKinetics.boundary_layer_m`.
    temperature_C : float
        Temperature (°C) for the RT/F·ln10 OER-shift prefactor.
    fraction_hydrolysed : float
        Fraction of produced Fe³⁺ precipitating to Fe(OH)₃ in the film (0–1);
        the complement is the shuttle source.

    Returns
    -------
    Fe3BoundaryResult
        The screening boundary-layer picture (all-zero for a Fe²⁺-free anolyte).
    """
    if fe2_bulk_M <= 0.0:
        return Fe3BoundaryResult(
            fe2_oxidation_flux_mol_m2_s=0.0,
            fe2_oxidation_current_A_m2=0.0,
            fe3_surface_M=0.0,
            fe3_solubility_cap_M=fe3_solubility_cap_M(ph_bulk),
            ph_drop=0.0,
            surface_pH=ph_bulk,
            oer_overpotential_raise_V=0.0,
            feoh3_sludge_flux_mol_m2_s=0.0,
            shuttle_source_flux_mol_m2_s=0.0,
        )

    f_hyd = min(max(float(fraction_hydrolysed), 0.0), 1.0)

    # 1. Mass-transfer-limited Fe²⁺ oxidation (surface Fe²⁺ → 0 at OER pot.).
    c_fe2_bulk_mol_m3 = fe2_bulk_M * 1000.0
    km_fe2 = mass_transfer_coeff(d_fe2_m2_s, boundary_layer_m)
    q_flux = km_fe2 * c_fe2_bulk_mol_m3               # mol/m²/s
    i_ox = FARADAY * q_flux                           # 1 e⁻ per Fe²⁺ (A/m²)

    # 2. Hydrolysis → acid + sludge.
    q_sludge = f_hyd * q_flux
    q_shuttle = (1.0 - f_hyd) * q_flux

    # 3. Surface pH from a single-film H⁺ mass balance (hydrolysis acid only;
    #    OER-H⁺ is the background the Hittorf term already carries).
    km_h = mass_transfer_coeff(d_h_m2_s, boundary_layer_m)
    h_bulk_mol_m3 = 10.0 ** (-ph_bulk) * 1000.0
    h_surf_mol_m3 = h_bulk_mol_m3 + (3.0 * q_sludge) / km_h
    if h_surf_mol_m3 <= 0.0 or h_bulk_mol_m3 <= 0.0:
        ph_drop = 0.0
    else:
        ph_drop = min(max(ph_bulk - ph_from_h_mol_m3(h_surf_mol_m3), 0.0), max_ph_drop)
    surface_ph = ph_bulk - ph_drop

    # 4. OER overpotential raise from ΔpH (E_eq = 1.229 − (RT/F)ln10·pH).
    T = temperature_C + 273.15
    prefactor = (R_GAS * T / FARADAY) * 2.302585093
    oer_raise = prefactor * ph_drop

    # 5. Surface Fe³⁺ diagnostic + solubility cap at the surface pH.
    fe3_surface_M = q_shuttle / max(km_fe2, 1e-30) / 1000.0 if q_shuttle > 0.0 else 0.0
    cap = fe3_solubility_cap_M(surface_ph)

    return Fe3BoundaryResult(
        fe2_oxidation_flux_mol_m2_s=q_flux,
        fe2_oxidation_current_A_m2=i_ox,
        fe3_surface_M=fe3_surface_M,
        fe3_solubility_cap_M=cap,
        ph_drop=ph_drop,
        surface_pH=surface_ph,
        oer_overpotential_raise_V=oer_raise,
        feoh3_sludge_flux_mol_m2_s=q_sludge,
        shuttle_source_flux_mol_m2_s=q_shuttle,
    )
