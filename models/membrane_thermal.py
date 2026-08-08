"""
Localized membrane Ohmic heating and trans-membrane temperature profiles.

Physics and Chemistry
---------------------
In divided-cell electrowinning, the membrane (e.g., Nafion) has a finite area resistance
R_mem (typically 1.0–4.0 Ω·cm²). At high industrial current densities (j ≈ 100–300 mA/cm²),
passing current through the membrane core dissipates substantial Ohmic heat:
     P_mem = j² * R_mem  (W/m²)

Since this heat is generated within a very thin polymer film (Δx ≈ 50–150 µm), it
establishes a temperature profile governed by 1D steady-state heat conduction:
     κ_mem * d²T/dx² + q_gen = 0
where:
     - q_gen = P_mem / Δx is the volumetric heat source (W/m³)
     - κ_mem is the thermal conductivity of the hydrated membrane (~0.2 W/m·K)

Boundary conditions at the catholyte interface (x = -Δx/2) and anolyte interface (x = Δx/2)
are set by convective cooling to the flowing bulk electrolytes:
     -κ_mem * dT/dx |_{-Δx/2} = h_c * (T_cath,bulk - T_surface,cath)
      κ_mem * dT/dx |_{Δx/2}  = h_a * (T_surface,ano - T_anolyte,bulk)
where h_c, h_a are the convective heat transfer coefficients (W/m²·K).

Because ferric crossover (Fe³⁺ diffusion) obeys Arrhenius kinetics, the localized core
temperature spike exponentially accelerates crossover, degrading Faradaic Efficiency (FE)
far more than predicted by bulk-isothermal models.

References
----------
* Incropera, F. P., et al. (2007). "Fundamentals of Heat and Mass Transfer." John Wiley & Sons.
* Weber, A. Z., & Newman, J. (2004). "Transport in Polymer-Electrolyte Membranes."
  Journal of The Electrochemical Society, 151(2), A311.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class MembraneThermalParams:
    """Parameters for membrane heat transfer and thermal coupling."""

    thickness_um: float = 120.0          # Membrane hydrated thickness (µm)
    r_mem_ohm_cm2: float = 3.0           # Membrane area resistance (Ω·cm²)
    thermal_cond_W_mK: float = 0.21      # Thermal conductivity of hydrated polymer (W/m·K)
    h_cath_W_m2K: float = 1200.0         # Convective cooling coefficient, catholyte side (W/m²·K)
    h_ano_W_m2K: float = 1200.0          # Convective cooling coefficient, anolyte side (W/m²·K)
    ea_crossover_J_mol: float = 25e3     # Activation energy of ferric crossover diffusion (J/mol)

    @property
    def thickness_m(self) -> float:
        """Membrane thickness in meters (m)."""
        return self.thickness_um * 1e-6

    @property
    def r_mem_ohm_m2(self) -> float:
        """Membrane area resistance in SI units (Ω·m²)."""
        return self.r_mem_ohm_cm2 * 1e-4


@dataclass
class MembraneTemperatureProfile:
    """Temperature distribution and ferric crossover acceleration metrics."""

    j_A_m2: float
    p_mem_W_m2: float
    t_cath_bulk_C: float
    t_ano_bulk_C: float
    t_cath_surface_C: float
    t_ano_surface_C: float
    t_peak_core_C: float
    mean_membrane_temp_C: float
    crossover_acceleration_factor: float  # Integrated Arrhenius acceleration vs bulk isothermal reference
    is_thermally_safe: bool               # True if peak core temperature is below Nafion limit (~85 °C)


def solve_membrane_temperature_profile(
    j_mA_cm2: float,
    t_cath_bulk_C: float,
    t_ano_bulk_C: float = 60.0,
    params: Optional[MembraneThermalParams] = None,
) -> MembraneTemperatureProfile:
    """
    Solve the 1D steady-state heat equation across the membrane.
    Returns the exact analytical temperature profile and coupled crossover metrics.

    Parameters
    ----------
    j_mA_cm2 : float
        Current density (mA/cm²).
    t_cath_bulk_C : float
        Bulk catholyte temperature (°C).
    t_ano_bulk_C : float, default 60.0
        Bulk anolyte temperature (°C).
    params : MembraneThermalParams, optional
        Physical thermal properties.

    Returns
    -------
    MembraneTemperatureProfile
        Solved profile and thermal metrics.
    """
    if params is None:
        params = MembraneThermalParams()

    j_SI = max(float(j_mA_cm2), 0.0) * 10.0  # mA/cm² -> A/m²
    t_c = float(t_cath_bulk_C)
    t_a = float(t_ano_bulk_C)

    # Volumetric and area heat generation
    p_mem = (j_SI ** 2) * params.r_mem_ohm_m2  # W/m²
    thick = params.thickness_m
    q_gen = p_mem / thick if thick > 0 else 0.0

    k_m = params.thermal_cond_W_mK
    h_c = params.h_cath_W_m2K
    h_a = params.h_ano_W_m2K

    # Analytical derivation of temperature profile: T(x) = -q_gen/(2*k_m) * x² + C1*x + C2
    # Coordinate system: x is from -L to +L, where L = thickness / 2
    L = thick / 2.0

    if q_gen == 0.0:
        # Purely linear temperature gradient if no heat is generated (Ohmic heating = 0)
        # T(-L) = T_surf,c; T(L) = T_surf,a
        # Solve linear system for conduction vs convection:
        # -k_m * dT/dx = h_c * (T_c - T(-L))
        #  k_m * dT/dx = h_a * (T(L) - T_a)
        # dT/dx = (T_ano_surf - T_cath_surf) / thick
        r_tot = (1.0 / h_c) + (thick / k_m) + (1.0 / h_a)
        heat_flux_linear = (t_c - t_a) / r_tot
        t_surf_c = t_c - heat_flux_linear / h_c
        t_surf_a = t_a + heat_flux_linear / h_a
        c1 = (t_surf_a - t_surf_c) / thick
        c2 = (t_surf_c + t_surf_a) / 2.0
    else:
        # Solve for coefficients C1 and C2 in the quadratic heat-conduction equation:
        # We set up two linear equations matching the convective boundary conditions:
        # Eq 1: k_m * C1 * (1 + k_m / (h_c * L)) ...
        # Standard analytical system solution for asymmetric convection:
        alpha = k_m / h_c
        beta = k_m / h_a

        # T(-L) = -q_gen*L²/(2*k_m) - C1*L + C2
        # T'( -L) = q_gen*L/k_m + C1
        # -k_m*(q_gen*L/k_m + C1) = h_c*(T_c - (-q_gen*L²/(2*k_m) - C1*L + C2))
        # Rewrite: -q_gen*L - k_m*C1 = h_c*T_c + h_c*q_gen*L²/(2*k_m) + h_c*C1*L - h_c*C2
        # Similar on anolyte side.
        # Let's solve directly for surface temperatures first:
        # Conduction rate must balance convection at boundaries:
        # Total heat generated P_mem leaves via both boundaries.
        # q_c (out of catholyte side) + q_a (out of anolyte side) = P_mem
        # Standard heat balance yields:
        # C1 = ((t_a - t_c) + (q_gen * L / k_m) * (alpha - beta)) / (2*L + alpha + beta)
        # C2 = (t_c + t_a + q_gen*L*(1/h_c + 1/h_a) + q_gen*L²/k_m - C1*(beta - alpha)) / 2.0
        denom = 2.0 * L + alpha + beta
        c1 = ((t_a - t_c) + (q_gen * L / k_m) * (alpha - beta)) / denom
        c2 = (t_c + t_a + q_gen * L * (1.0 / h_c + 1.0 / h_a) + (q_gen * (L ** 2) / k_m) - c1 * (beta - alpha)) / 2.0

    # Calculate critical locations (temperatures in °C)
    t_surf_c = -q_gen * (L ** 2) / (2.0 * k_m) - c1 * L + c2
    t_surf_a = -q_gen * (L ** 2) / (2.0 * k_m) + c1 * L + c2

    # Peak location: dT/dx = -q_gen/k_m * x + C1 = 0 => x_peak = C1 * k_m / q_gen
    if abs(q_gen) > 1e-6:
        x_peak = c1 * k_m / q_gen
        # Bound peak location within membrane boundaries [-L, L]
        if abs(x_peak) <= L:
            t_peak = -q_gen * (x_peak ** 2) / (2.0 * k_m) + c1 * x_peak + c2
        else:
            t_peak = max(t_surf_c, t_surf_a)
    else:
        t_peak = max(t_surf_c, t_surf_a)

    # Integrated mean membrane temperature: Integral of T(x)/thick from -L to L
    # Mean(T) = -q_gen/(6*k_m) * L² + C2
    mean_temp = -q_gen * (L ** 2) / (6.0 * k_m) + c2

    # Integrated Arrhenius acceleration factor for ferric crossover:
    # Enhancement = 1/thick * Integral_{-L}^{L} exp(-Ea / R * (1/T(x) - 1/T_bulk)) dx
    # We solve this numerically over 20 points across the membrane thickness
    x_eval = np.linspace(-L, L, 21)
    temp_profile = -q_gen * (x_eval ** 2) / (2.0 * k_m) + c1 * x_eval + c2
    temp_profile_K = temp_profile + 273.15

    t_bulk_K = ((t_c + t_a) / 2.0) + 273.15
    r_gas = 8.314

    arrhenius_factors = np.exp(
        -(params.ea_crossover_J_mol / r_gas) * (1.0 / temp_profile_K - 1.0 / t_bulk_K)
    )
    enhancement = float(np.mean(arrhenius_factors))

    # Safely criterion: peak core temp should not exceed polymer thermal limits (~85 °C)
    is_safe = t_peak <= 85.0

    return MembraneTemperatureProfile(
        j_A_m2=j_SI,
        p_mem_W_m2=p_mem,
        t_cath_bulk_C=t_c,
        t_ano_bulk_C=t_a,
        t_cath_surface_C=t_surf_c,
        t_ano_surface_C=t_surf_a,
        t_peak_core_C=t_peak,
        mean_membrane_temp_C=mean_temp,
        crossover_acceleration_factor=enhancement,
        is_thermally_safe=is_safe,
    )


def main() -> None:
    """CLI entrypoint for membrane thermal analysis."""
    print("=================================================================")
    print(" Membrane Localized Ohmic Heating & Ferric Crossover Acceleration")
    print("=================================================================")
    params = MembraneThermalParams()
    print(f"Membrane resistance: {params.r_mem_ohm_cm2:.1f} Ω·cm² ({params.thickness_um:.1f} µm)")
    print(f"Convective cooling : {params.h_cath_W_m2K:.1f} W/m²·K\n")

    print("Current Density vs Thermal Performance (Bulk T = 60°C both sides):")
    for j in [0.0, 50.0, 150.0, 300.0, 400.0]:
        res = solve_membrane_temperature_profile(j, 60.0, 60.0, params)
        print(f"  j = {j:3.0f} mA/cm² | Ohmic: {res.p_mem_W_m2:6.1f} W/m²")
        print(f"    Peak Temp: {res.t_peak_core_C:5.1f} °C | Crossover Accel: {res.crossover_acceleration_factor:.3f}x | {'Safe' if res.is_thermally_safe else 'DANGER (>85°C)'}")


if __name__ == "__main__":
    main()
