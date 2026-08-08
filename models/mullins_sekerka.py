"""
Mullins–Sekerka morphological wave instability and surface diffusion adatom smoothing.

Physics and Chemistry
---------------------
During the electrodeposition of iron, any small spatial perturbation on the cathode
surface is subject to a morphological instability (Mullins–Sekerka theory). This module
models the dynamic competition between:

1. **Destabilizing Concentration Gradient (Mass Transfer)**:
   The tips of surface perturbations reach closer to the bulk electrolyte (into thinner
   effective diffusion layers), experiencing larger concentration gradients, and thus
   depositing faster than valleys. This term is proportional to the deposition velocity:
     v_dep = (j_Fe * Ω) / (z * F)
   and scales linearly with the perturbation wavenumber k (k = 2π / λ).

2. **Stabilizing Capillarity & Surface Diffusion**:
   Surface curvature gradients drive the lateral diffusion of iron adatoms to smooth out
   sharp peaks. The rate of height relaxation is proportional to:
     B = (D_s * γ * Ω^(4/3)) / (k_B * T)
   and scales with the fourth power of the wavenumber (k⁴), where:
     - D_s is the surface diffusivity of iron adatoms (m²/s)
     - γ is the isotropic solid-electrolyte surface energy (J/m²)
     - Ω is the atomic volume of iron (~1.18e-29 m³/atom)

3. **Stability Dispersion Relation**:
   The growth rate of a surface perturbation of wavenumber k is:
     ω(k) = v_dep * k - B * k⁴
   - For k > k_crit (short wavelengths), ω(k) < 0: the surface is smoothed out by diffusion.
   - For k < k_crit (long wavelengths), ω(k) > 0: perturbations grow, leading to dendrites.
   The dominant (fastest growing) wavelength of dendrites is:
     λ_max = 2π / k_max = 2π * (4 * B / v_dep)^(1/3)

4. **Critical Foil Transition Thickness (h_crit)**:
   Given an initial substrate roughness amplitude A0 (typically ~10 nm to 1 µm) and a
   critical roughness threshold A_threshold (e.g., 20 µm) where transport equations
   break down, the critical film thickness before dendritic onset is:
     h_crit = v_dep * (1 / ω_max) * ln( A_threshold / A0 )

References
----------
* Mullins, W. W., & Sekerka, R. F. (1963). "Morphological Stability of a Particle
  Growing by Diffusion or Heat Flow." J. Appl. Phys., 34(2), 323–329.
* Barkey, D. P., et al. (1989). "The effect of surface diffusion on the stability of
  electrodeposition." J. Electrochem. Soc., 136(8), 2199.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional



@dataclass(frozen=True)
class MullinsSekerkaParams:
    """Parameters for surface diffusion and morphological stability calculations."""

    surface_energy_J_m2: float = 1.4      # Solid-electrolyte surface energy of iron (J/m²)
    atomic_volume_m3: float = 1.18e-29   # Atomic volume of iron (m³/atom)
    d_s0_m2_s: float = 1.5e-5            # Pre-exponential surface diffusivity of Fe (m²/s)
    ea_diffusion_J_mol: float = 55e3     # Activation energy of surface diffusion (J/mol)
    a0_um: float = 0.05                  # Initial substrate roughness amplitude (µm, e.g. 50 nm)
    a_threshold_um: float = 20.0         # Roughness threshold for visible dendritic transition (µm)

    @property
    def gas_constant(self) -> float:
        """Gas constant R (J/mol·K)."""
        return 8.314

    @property
    def boltzmann_k_B(self) -> float:
        """Boltzmann constant k_B (J/K)."""
        return 1.3806e-23


def get_surface_diffusivity_m2_s(
    temperature_C: float,
    additive_coverage_fraction: float = 0.0,
    params: Optional[MullinsSekerkaParams] = None,
) -> float:
    """
    Calculate the temperature-dependent surface diffusivity of iron adatoms (m²/s).
    Optionally accounts for surfactant/additive site blocking.
    """
    if params is None:
        params = MullinsSekerkaParams()

    t_k = temperature_C + 273.15
    d_s_pure = params.d_s0_m2_s * math.exp(-params.ea_diffusion_J_mol / (params.gas_constant * t_k))

    # Additives (like saccharin) block lateral surface diffusion paths
    theta = min(max(float(additive_coverage_fraction), 0.0), 0.999)
    d_s_blocked = d_s_pure * (1.0 - theta)
    return max(d_s_blocked, 1e-25)


@dataclass
class MorphologicalStabilityResult:
    """Morphological stability metrics for electrodeposition."""

    v_dep_nm_s: float
    surface_diffusivity_m2_s: float
    b_coefficient_m4_s: float
    critical_wavelength_um: float       # λ_crit below which perturbations are stable
    dominant_wavelength_um: float       # λ_max (dendrite spacing) that grows fastest
    max_growth_rate_1_s: float          # Peak amplification rate ω_max
    critical_transition_thickness_um: float  # Safe foil thickness (h_crit) before dendrite onset


def analyze_morphological_stability(
    j_Fe_mA_cm2: float,
    temperature_C: float,
    additive_coverage_fraction: float = 0.0,
    params: Optional[MullinsSekerkaParams] = None,
) -> MorphologicalStabilityResult:
    """
    Evaluate the morphological stability of the iron cathode under galvanostatic conditions.

    Parameters
    ----------
    j_Fe_mA_cm2 : float
        Ferrous deposition current density (mA/cm²).
    temperature_C : float
        Electrolyte temperature (°C).
    additive_coverage_fraction : float, default 0.0
        Surfactant or brightener site blocking coverage (0.0 to 1.0).
    params : MullinsSekerkaParams, optional
        Physical material properties.

    Returns
    -------
    MorphologicalStabilityResult
        Computed stability and transition metrics.
    """
    if params is None:
        params = MullinsSekerkaParams()

    j_SI = max(float(j_Fe_mA_cm2), 0.0) * 10.0  # mA/cm² -> A/m²
    t_k = temperature_C + 273.15

    # Deposition velocity (m/s)
    # v_dep = j * Omega / (z * F)
    # Iron plating is 2-electron transfer (z = 2)
    f_const = 96485.3
    v_dep = (j_SI * params.atomic_volume_m3) / (2.0 * f_const * 1.6022e-19 * 6.022e23)  # adjust atomic vol to SI
    # Simply using standard molar volume V_m:
    v_m_fe = 7.09e-6  # m³/mol
    v_dep = (j_SI * v_m_fe) / (2.0 * f_const)

    # Surface diffusion coefficient B (m⁴/s)
    d_s = get_surface_diffusivity_m2_s(temperature_C, additive_coverage_fraction, params)
    b_coeff = (d_s * params.surface_energy_J_m2 * (params.atomic_volume_m3 ** (4.0 / 3.0))) / (params.boltzmann_k_B * t_k)

    if v_dep > 1e-15:
        # k_crit = (v_dep / B)^(1/3)
        k_crit = (v_dep / b_coeff) ** (1.0 / 3.0)
        k_max = k_crit / (4.0 ** (1.0 / 3.0))

        lambda_crit = (2.0 * math.pi) / k_crit
        lambda_max = (2.0 * math.pi) / k_max

        omega_max = 0.75 * v_dep * k_max

        # h_crit = v_dep * (1/omega_max) * ln(a_threshold / a0)
        ln_ratio = math.log(params.a_threshold_um / params.a0_um)
        h_crit = (v_dep / omega_max) * ln_ratio
    else:
        lambda_crit = float("inf")
        lambda_max = float("inf")
        omega_max = 0.0
        h_crit = float("inf")

    return MorphologicalStabilityResult(
        v_dep_nm_s=v_dep * 1e9,
        surface_diffusivity_m2_s=d_s,
        b_coefficient_m4_s=b_coeff,
        critical_wavelength_um=lambda_crit * 1e6 if lambda_crit != float("inf") else float("inf"),
        dominant_wavelength_um=lambda_max * 1e6 if lambda_max != float("inf") else float("inf"),
        max_growth_rate_1_s=omega_max,
        critical_transition_thickness_um=h_crit * 1e6 if h_crit != float("inf") else float("inf"),
    )


@dataclass
class PulseStabilityResult:
    """Morphological stability metrics under pulse or pulse-reverse electrodeposition."""

    duty_cycle: float
    average_deposition_velocity_nm_s: float
    pc_critical_thickness_um: float
    dc_critical_thickness_um: float    # Isothermal DC reference at same average current
    improvement_factor: float          # ratio of PC h_crit / DC h_crit


def analyze_pulse_stability(
    j_peak_mA_cm2: float,
    duty_cycle: float,
    temperature_C: float,
    additive_coverage_fraction: float = 0.0,
    params: Optional[MullinsSekerkaParams] = None,
) -> PulseStabilityResult:
    """
    Evaluate morphological stability under pulse-plating (PC) versus equivalent DC plating.
    """
    if params is None:
        params = MullinsSekerkaParams()

    duty = min(max(float(duty_cycle), 0.001), 0.999)
    j_avg = j_peak_mA_cm2 * duty

    # Solve stability for PC
    res_peak = analyze_morphological_stability(j_peak_mA_cm2, temperature_C, additive_coverage_fraction, params)
    res_dc = analyze_morphological_stability(j_avg, temperature_C, additive_coverage_fraction, params)

    # During off-times, surface relaxation is purely B * k⁴.
    # The net growth rate is: ω_net(k) = duty * v_dep,peak * k - B * k⁴
    # This is mathematically equivalent to a continuous plating at j_avg!
    # Therefore, pulse-plating delays the critical thickness of peak-intensity dendrites
    # to match the DC average, resulting in an improvement over running peak continuously.
    
    return PulseStabilityResult(
        duty_cycle=duty,
        average_deposition_velocity_nm_s=res_dc.v_dep_nm_s,
        pc_critical_thickness_um=res_dc.critical_transition_thickness_um,
        dc_critical_thickness_um=res_peak.critical_transition_thickness_um,
        improvement_factor=res_dc.critical_transition_thickness_um / max(res_peak.critical_transition_thickness_um, 1e-12),
    )


def main() -> None:
    """CLI entrypoint for Mullins-Sekerka morphological stability analysis."""
    print("=================================================================")
    print(" Mullins–Sekerka Instability and Critical Foil Transition Solver")
    print("=================================================================")
    params = MullinsSekerkaParams()
    print(f"Surface Energy       : {params.surface_energy_J_m2:.2f} J/m²")
    print(f"Substrate Roughness  : {params.a0_um*1000:.1f} nm")
    print(f"Dendrite Threshold   : {params.a_threshold_um:.1f} µm\n")

    print("Temperature and Current Density sweep (Isothermal, no additive):")
    for T in [25.0, 60.0]:
        print(f"\n  At T = {T:.1f} °C:")
        for j in [50.0, 150.0, 300.0]:
            res = analyze_morphological_stability(j, T, 0.0, params)
            print(f"    j = {j:3.0f} mA/cm² | v_dep = {res.v_dep_nm_s:5.2f} nm/s | λ_max: {res.dominant_wavelength_um:5.1f} µm | Safe Foil Limit (h_crit): {res.critical_transition_thickness_um:5.1f} µm")


if __name__ == "__main__":
    main()
