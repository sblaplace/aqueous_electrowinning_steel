"""
Chemical osmosis and water activity gradient transport across cation exchange membranes.

Physics and Chemistry
---------------------
In divided-cell iron electrowinning (:mod:`models.membrane_transport`, :mod:`models.closed_loop`),
the cation exchange membrane (Nafion N117, Fumasep FKE-50) separates two electrolytes
with fundamentally different chemical compositions and water activities:

1. **Catholyte**: 1.5 M FeSO₄ + 0.5 M Na₂SO₄ (Ionic strength I ≈ 6.5 m, water activity a_w,cath ≈ 0.925).
2. **Anolyte**: 0.5–1.0 M H₂SO₄ (Ionic strength I ≈ 1.0–2.0 m, water activity a_w,ano ≈ 0.965).

The resulting thermodynamic **water activity difference Δa_w** generates a large
osmotic pressure gradient across the membrane:
    Δπ = - (R T / V_w) · ln(a_w,cath / a_w,ano) ≈ 40–70 bar

Coupled Transmembrane Water Transport
-------------------------------------
The total net volumetric water flux J_w (m/s or L/(m²·hr)) is the vector sum of two
opposing driving forces:

1. **Electro-osmotic drag (EOD)**: Protons migrating through the membrane carry a
   hydration shell (n_w ≈ 2.0–3.0 H₂O per H⁺), transporting water from **catholyte to anolyte**:
     J_w,eod = n_w · (j / F) · V_w

2. **Chemical osmosis (driven by Δa_w)**: Water spontaneously diffuses down its chemical
   potential gradient through the membrane channels from the high-a_w anolyte into the
   concentrated catholyte (**anolyte to catholyte**):
     J_w,osm = - L_p · σ_refl · Δπ = + L_p · σ_refl · (R T / V_w) · ln(a_w,ano / a_w,cath)

3. **Hydraulic pressure difference**:
     J_w,hyd = L_p · (P_cath - P_ano)

Net Water Transfer Balance
--------------------------
    J_w,net = J_w,eod - J_w,osm + J_w,hyd

At moderate current densities (j ≈ 14–25 mA/cm² under balanced hydraulic head,
or higher under positive anolyte overpressure), **chemical osmosis balances
electro-osmotic drag**, establishing an isovolemic zero-net-flux operating window.
At low j, chemical osmosis dominates, diluting the catholyte.  At high j, EOD dominates,
drying the catholyte.

References
----------
* Kedem, O., & Katchalsky, A. (1958). "Thermodynamic analysis of the permeability
  of biological membranes to non-electrolytes." Biochim. Biophys. Acta, 27, 229–246.
* Yeager, H. L., & Steck, A. (1981). "Cation and water diffusion in Nafion ion
  exchange membranes." J. Electrochem. Soc., 128(9), 1880–1884.
* Robinson, R. A., & Stokes, R. H. (2002). "Electrolyte Solutions." Dover Publications.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


# Physical constants
FARADAY = 96485.33212      # C/mol
R_GAS = 8.314462618        # J/(mol·K)
T_REF = 298.15             # K
V_WATER_MOLAR_M3_MOL = 18.015e-6  # Partial molar volume of liquid water (m³/mol)


@dataclass(frozen=True)
class MembraneWaterParams:
    """Hydraulic and transport properties of the cation exchange membrane."""

    hydraulic_permeability_m_per_Pa_s: float = 1.2e-14  # L_p (m/(Pa·s))
    osmotic_reflection_coefficient: float = 0.85       # σ_refl (0 = permeable, 1 = ideal semipermeable)
    electro_osmotic_drag_n_w: float = 2.5              # H2O molecules dragged per H+ ion
    membrane_thickness_m: float = 175e-6               # Nafion N117 thickness (m)
    temperature_K: float = 333.15                      # 60 °C default


@dataclass
class TransmembraneWaterResult:
    """Net water transport rates and volume drift in divided electrowinning cells."""

    current_density_mA_cm2: float
    water_activity_catholyte: float
    water_activity_anolyte: float
    osmotic_pressure_bar: float
    electro_osmotic_flux_L_m2_h: float   # Catholyte -> Anolyte
    chemical_osmotic_flux_L_m2_h: float # Anolyte -> Catholyte
    net_water_flux_L_m2_h: float        # Positive = Net flux from Catholyte -> Anolyte
    zero_net_flux_current_density_mA_cm2: float  # Current density where J_net = 0
    transport_regime: str               # "Osmosis-dominated (catholyte swelling)", etc.


def estimate_water_activity(
    fe_sulfate_mol_L: float,
    na_sulfate_mol_L: float = 0.0,
    h2_sulfate_mol_L: float = 0.0,
) -> float:
    """
    Estimate the osmotic water activity a_w from solute molalities.

    Uses the Robinson & Stokes osmotic coefficient formulation:
      ln(a_w) = - (M_w / 1000) * sum(nu_i * m_i * phi_i)
    """
    m_fe = max(float(fe_sulfate_mol_L), 0.0)
    m_na = max(float(na_sulfate_mol_L), 0.0)
    m_h = max(float(h2_sulfate_mol_L), 0.0)

    # Effective ionic strength sum
    # FeSO4: nu=2, phi~0.65; Na2SO4: nu=3, phi~0.75; H2SO4: nu=3, phi~0.80
    sum_m = (2.0 * m_fe * 0.65) + (3.0 * m_na * 0.75) + (3.0 * m_h * 0.80)
    ln_aw = -0.018015 * sum_m
    return max(min(math.exp(ln_aw), 0.999), 0.75)


def solve_transmembrane_water_flux(
    current_density_mA_cm2: float = 200.0,
    water_activity_catholyte: float = 0.925,
    water_activity_anolyte: float = 0.965,
    delta_p_hydraulic_bar: float = 0.0,
    params: Optional[MembraneWaterParams] = None,
) -> TransmembraneWaterResult:
    """
    Solve the coupled chemical osmosis and electro-osmotic drag water flux.

    Parameters
    ----------
    current_density_mA_cm2 : float
        Operating cathodic current density (mA/cm²).
    water_activity_catholyte : float
        Water activity in the concentrated catholyte.
    water_activity_anolyte : float
        Water activity in the anolyte.
    delta_p_hydraulic_bar : float
        Hydraulic overpressure (P_cath - P_ano) in bar.
    params : MembraneWaterParams, optional
        Membrane permeability and drag coefficients.

    Returns
    -------
    TransmembraneWaterResult
        Detailed breakdown of water flux vectors and steady-state regime.
    """
    if params is None:
        params = MembraneWaterParams()

    j_si = max(float(current_density_mA_cm2), 0.0) * 10.0  # mA/cm² -> A/m²
    t_k = params.temperature_K
    a_cath = max(min(float(water_activity_catholyte), 0.999), 0.50)
    a_ano = max(min(float(water_activity_anolyte), 0.999), 0.50)

    # Thermodynamic chemical osmotic pressure (Pa):
    # Δπ = - (R T / V_w) * ln(a_cath / a_ano)
    rt_vw = (R_GAS * t_k) / V_WATER_MOLAR_M3_MOL  # Pa
    delta_pi_pa = -rt_vw * math.log(a_cath / a_ano)
    delta_pi_bar = delta_pi_pa / 1e5

    # 1. Electro-osmotic drag flux: J_eod (m/s) = n_w * (j / F) * V_w
    # Directed from Catholyte -> Anolyte (positive sign)
    j_eod_m_s = params.electro_osmotic_drag_n_w * (j_si / FARADAY) * V_WATER_MOLAR_M3_MOL
    j_eod_L_m2_h = j_eod_m_s * 1e3 * 3600.0  # L/(m²·hr)

    # 2. Chemical osmotic flux: J_osm (m/s) = L_p * σ_refl * Δπ
    # Directed from Anolyte -> Catholyte
    j_osm_m_s = params.hydraulic_permeability_m_per_Pa_s * params.osmotic_reflection_coefficient * delta_pi_pa
    j_osm_L_m2_h = j_osm_m_s * 1e3 * 3600.0  # L/(m²·hr)

    # 3. Hydraulic flux
    delta_p_pa = float(delta_p_hydraulic_bar) * 1e5
    j_hyd_m_s = params.hydraulic_permeability_m_per_Pa_s * delta_p_pa
    j_hyd_L_m2_h = j_hyd_m_s * 1e3 * 3600.0

    # Net flux: positive = net water leaves catholyte to anolyte
    net_flux_L_m2_h = j_eod_L_m2_h - j_osm_L_m2_h + j_hyd_L_m2_h

    # Zero net flux current density: where J_eod = J_osm - J_hyd
    # n_w * (j_0 / F) * V_w * 3.6e6 = J_osm_L - J_hyd_L
    flux_target = max(j_osm_L_m2_h - j_hyd_L_m2_h, 0.0)
    coeff = (params.electro_osmotic_drag_n_w * 10.0 / FARADAY) * V_WATER_MOLAR_M3_MOL * 3.6e6
    j_zero_mA_cm2 = flux_target / max(coeff, 1e-15)

    if net_flux_L_m2_h > 0.20:
        regime = "EOD-dominated (catholyte water depletion / anolyte accumulation)"
    elif net_flux_L_m2_h < -0.20:
        regime = "Osmosis-dominated (catholyte volume swelling / dilution)"
    else:
        regime = "Near-isovolemic balance (stable continuous electrolyte volumes)"

    return TransmembraneWaterResult(
        current_density_mA_cm2=current_density_mA_cm2,
        water_activity_catholyte=a_cath,
        water_activity_anolyte=a_ano,
        osmotic_pressure_bar=delta_pi_bar,
        electro_osmotic_flux_L_m2_h=j_eod_L_m2_h,
        chemical_osmotic_flux_L_m2_h=j_osm_L_m2_h,
        net_water_flux_L_m2_h=net_flux_L_m2_h,
        zero_net_flux_current_density_mA_cm2=j_zero_mA_cm2,
        transport_regime=regime,
    )


def main() -> None:
    """CLI entrypoint for chemical osmosis and water transport."""
    print("=================================================================")
    print(" Transmembrane Chemical Osmosis & Electro-Osmotic Water Flux")
    print("=================================================================")
    print("Current density sweep (Catholyte aw = 0.925, Anolyte aw = 0.965):")
    for j in [10.0, 50.0, 100.0, 200.0, 350.0]:
        res = solve_transmembrane_water_flux(j, 0.925, 0.965)
        print(f"  j = {j:5.1f} mA/cm² | J_net = {res.net_water_flux_L_m2_h:+6.3f} L/(m²·hr) | {res.transport_regime}")


if __name__ == "__main__":
    main()
