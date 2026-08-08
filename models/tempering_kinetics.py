"""
Four-stage tempering kinetics, carbide coarsening (LSW Ostwald ripening), and Charpy toughness.

Physics and Metallurgy
----------------------
When electrodeposited and carburized iron/steel (martensite with supersaturated carbon)
is subjected to post-quench thermal tempering (:mod:`models.tempering`, :mod:`models.carburization`),
it transforms through **four distinct metallurgical stages**:

1. **Stage 1 (100–200 °C)**:
   - Precipitation of transition **ε-carbide** (Fe₂.₄C) or **η-carbide** (Fe₂C) in high-carbon regions.
   - Martensite loses its initial c/a tetragonality, relaxing into low-tetragonality/cubic ferrite.
   - High hardness is maintained with slight relaxation of as-quenched transformation stress.

2. **Stage 2 (200–250 °C)**:
   - Decomposition of metastable retained austenite (γ_ret) into lower bainite (ferrite + Fe₃C).
   - Volume expansion during austenite transformation can cause dimensional distortion.

3. **Stage 3 (250–380 °C)**:
   - Transition ε-carbides dissolve and transform into **plate-like orthorhombic cementite** (Fe₃C).
   - This temperature window corresponds to **Tempered Martensite Embrittlement (TME)**:
     interlath carbide films degrade Charpy impact toughness.

4. **Stage 4 (380–700 °C)**:
   - Cementite precipitates spheroidize and coarsen via **Lifshitz–Slyozov–Wagner (LSW) Ostwald ripening**:
       r̄(t)³ - r̄₀³ = K_LSW(T) · t
     where the LSW coarsening rate constant is:
       K_LSW = (8 · γ_α/Fe3C · D_C^α · C_C^α · V_m²) / (9 · R · T)
   - Dislocation substructure recovers and recrystallizes, softening the matrix while restoring high
     ductility and low Ductile-to-Brittle Transition Temperature (DBTT).

Mechanical Consequences
-----------------------
* **Orowan precipitate bypass yield strength**:
    Δσ_Orowan = (M · 0.81 · G · b / (2π · √(1-ν))) · (ln(2 r̄ / b) / (λ - 2 r̄))
* **Charpy impact toughness & DBTT**:
    DBTT = DBTT₀ - k_grain · d_grain^(-1/2) + k_carbide · √(r̄)

References
----------
* Speich, G. R., & Leslie, W. C. (1972). "Tempering of steel." Metall. Trans., 3(5), 1043–1054.
* Lifshitz, I. M., & Slyozov, V. V. (1961). "The kinetics of precipitation from
  supersaturated solid solutions." J. Phys. Chem. Solids, 19(1-2), 35–50.
* Gladman, T. (1997). "The Metallurgy of Carbon Steels." Institute of Materials.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


# Physical and metallurgical constants
R_GAS = 8.314462618        # J/(mol·K)
T_REF = 298.15             # K

# Iron matrix properties
G_FE_PA = 80.0e9           # Shear modulus of ferrite (Pa)
BURGERS_B_M = 0.248e-9     # Burgers vector in bcc Fe (m)
POISSON_NU = 0.29          # Poisson's ratio
TAYLOR_M = 3.0             # Taylor orientation factor for untextured bcc polycrystal

# Cementite Fe3C properties
GAMMA_FE3C_J_M2 = 0.70     # α-Fe / Fe3C interfacial energy (J/m²)
V_MOLAR_FE3C_M3 = 2.34e-5  # Molar volume of Fe3C (m³/mol)

# Carbon diffusion in bcc α-iron (ferrite)
D0_C_FERRITE = 6.2e-7      # Pre-exponential factor (m²/s)
Q_C_FERRITE_J = 80.0e3     # Activation energy (J/mol)


@dataclass(frozen=True)
class SteelMicrostructureSpec:
    """Microstructural state of the as-quenched/carburized electrodeposit."""

    carbon_wt_percent: float = 0.40     # Nominal carbon content (wt%)
    grain_size_um: float = 2.5          # Prior austenite / ferrite subgrain size (µm)
    retained_austenite_fraction: float = 0.08  # Initial volume fraction of retained austenite
    initial_carbide_radius_nm: float = 2.0     # Initial nucleus radius (nm)


@dataclass
class TemperingKineticsResult:
    """Microstructural evolution and mechanical properties after tempering."""

    tempering_temperature_C: float
    tempering_time_hours: float
    tempering_stage: str               # "Stage 1 (ε-carbide)", "Stage 4 (spheroidized)", etc.
    mean_carbide_radius_nm: float
    interparticle_spacing_nm: float
    retained_austenite_remaining: float
    orowan_yield_increment_MPa: float
    estimated_yield_strength_MPa: float
    estimated_charpy_energy_J: float
    dbtt_C: float                      # Ductile-to-brittle transition temperature (°C)
    tme_embrittlement_risk: bool       # True if in Stage 3 embrittlement window


def carbon_diffusivity_ferrite(temperature_C: float) -> float:
    """Compute lattice diffusivity of carbon in bcc ferrite (m²/s)."""
    t_k = max(float(temperature_C) + 273.15, 200.0)
    return D0_C_FERRITE * math.exp(-Q_C_FERRITE_J / (R_GAS * t_k))


def lsw_coarsening_rate_constant(temperature_C: float) -> float:
    """
    Compute the LSW Ostwald ripening rate constant K_LSW (m³/s) at a given temperature.

    K_LSW = (8 * gamma * D_C * C_C * V_m^2) / (9 * R * T)
    """
    t_k = max(float(temperature_C) + 273.15, 200.0)
    d_c = carbon_diffusivity_ferrite(temperature_C)

    # Equilibrium carbon solubility in bcc ferrite C_C (mol/m³)
    # C_C ≈ 0.02 wt% at 727 °C, dropping exponentially at lower T
    # C_C(T) ≈ 4.3e4 * exp(-40000 / (R * T)) mol/m³
    c_c_solubility = max(4.3e4 * math.exp(-40000.0 / (R_GAS * t_k)), 1e-4)

    numerator = 8.0 * GAMMA_FE3C_J_M2 * d_c * c_c_solubility * (V_MOLAR_FE3C_M3 ** 2)
    denominator = 9.0 * R_GAS * t_k
    return numerator / max(denominator, 1e-9)


def simulate_tempering_kinetics(
    spec: Optional[SteelMicrostructureSpec] = None,
    temperature_C: float = 450.0,
    time_hours: float = 2.0,
) -> TemperingKineticsResult:
    """
    Simulate the microstructural state and toughness after thermal tempering.

    Parameters
    ----------
    spec : SteelMicrostructureSpec, optional
        Initial carbon content and grain structure.
    temperature_C : float
        Tempering furnace temperature (°C).
    time_hours : float
        Tempering soak time (hours).

    Returns
    -------
    TemperingKineticsResult
        Carbide size, Orowan strengthening, Charpy toughness, and DBTT.
    """
    if spec is None:
        spec = SteelMicrostructureSpec()

    t_c = float(temperature_C)
    t_sec = max(float(time_hours), 0.001) * 3600.0
    r0_m = spec.initial_carbide_radius_nm * 1e-9
    c_wt = spec.carbon_wt_percent

    # Volume fraction of cementite f_v from carbon mass balance:
    # 1 mol Fe3C = 6.67 wt% C => f_v ≈ (c_wt / 6.67) * (rho_Fe / rho_Fe3C)
    f_v_carbide = max(min((c_wt / 6.67) * 1.02, 0.20), 0.001)

    # Retained austenite decomposition (Stage 2: 200–300 °C)
    if t_c < 180.0:
        gamma_ret = spec.retained_austenite_fraction
    elif t_c < 300.0:
        # Partial decomposition
        frac = (t_c - 180.0) / 120.0
        gamma_ret = spec.retained_austenite_fraction * (1.0 - frac)
    else:
        gamma_ret = 0.0

    # Determine tempering stage & carbide growth
    if t_c < 200.0:
        stage = "Stage 1 (ε-carbide precipitation, tetragonality relaxation)"
        r_mean_m = r0_m * (1.0 + 0.2 * math.log10(1.0 + t_sec / 3600.0))
        tme_risk = False
    elif t_c < 250.0:
        stage = "Stage 2 (Retained austenite decomposition to lower bainite)"
        r_mean_m = r0_m * 1.5
        tme_risk = False
    elif t_c < 380.0:
        stage = "Stage 3 (Cementite plate formation & TME danger window)"
        r_mean_m = r0_m * 2.5
        tme_risk = True
    else:
        stage = "Stage 4 (Spheroidization & LSW Ostwald ripening)"
        tme_risk = False
        k_lsw = lsw_coarsening_rate_constant(t_c)
        r3 = (r0_m ** 3) + (k_lsw * t_sec)
        r_mean_m = r3 ** (1.0 / 3.0)

    r_mean_nm = r_mean_m * 1e9

    # Interparticle spacing lambda (m): lambda ≈ r * sqrt(2*pi / (3*f_v))
    lambda_m = r_mean_m * math.sqrt((2.0 * math.pi) / max(3.0 * f_v_carbide, 1e-6))
    lambda_nm = lambda_m * 1e9

    # Orowan precipitate bypass yield strength increment (Pa):
    # Delta_sigma = (M * 0.81 * G * b / (2*pi * sqrt(1-nu))) * (ln(2*r / b) / (lambda - 2*r))
    prefactor = (TAYLOR_M * 0.81 * G_FE_PA * BURGERS_B_M) / (2.0 * math.pi * math.sqrt(1.0 - POISSON_NU))
    log_term = math.log(max((2.0 * r_mean_m) / BURGERS_B_M, 1.5))
    spacing_term = max(lambda_m - (2.0 * r_mean_m), BURGERS_B_M)
    delta_sigma_orowan_pa = prefactor * (log_term / spacing_term)
    orowan_mpa = delta_sigma_orowan_pa * 1e-6

    # Matrix friction + Hall-Petch base strength
    d_um = max(spec.grain_size_um, 0.1)
    sigma_base = 120.0 + 500.0 / math.sqrt(d_um)  # Hall-Petch base
    yield_strength_mpa = min(sigma_base + orowan_mpa, 2200.0)  # Capped at realistic martensite capacity (2.2 GPa)

    # Charpy upper-shelf energy (J) and DBTT (°C)
    # Coarser carbides lower upper-shelf energy and raise DBTT (Griffith-Orowan crack initiation)
    if tme_risk:
        charpy_j = max(15.0 + 0.02 * (600.0 - yield_strength_mpa), 8.0)
        dbtt_c = 45.0  # Embrittled at room temperature
    else:
        # Stage 4 spheroidized structure has excellent impact toughness
        charpy_j = max(85.0 - (0.04 * yield_strength_mpa) + (0.5 * (t_c - 400.0)), 20.0)
        dbtt_c = -40.0 + (1.2 * math.sqrt(r_mean_nm)) - (15.0 / math.sqrt(d_um))

    return TemperingKineticsResult(
        tempering_temperature_C=t_c,
        tempering_time_hours=time_hours,
        tempering_stage=stage,
        mean_carbide_radius_nm=r_mean_nm,
        interparticle_spacing_nm=lambda_nm,
        retained_austenite_remaining=gamma_ret,
        orowan_yield_increment_MPa=orowan_mpa,
        estimated_yield_strength_MPa=yield_strength_mpa,
        estimated_charpy_energy_J=charpy_j,
        dbtt_C=dbtt_c,
        tme_embrittlement_risk=tme_risk,
    )


def main() -> None:
    """CLI entrypoint for 4-stage tempering kinetics."""
    print("=================================================================")
    print(" 4-Stage Tempering Metallurgy & LSW Carbide Coarsening")
    print("=================================================================")
    spec = SteelMicrostructureSpec(carbon_wt_percent=0.40)
    for t_c in [150.0, 250.0, 320.0, 450.0, 600.0]:
        res = simulate_tempering_kinetics(spec, temperature_C=t_c, time_hours=2.0)
        print(f" T = {t_c:5.1f} °C | Stage: {res.tempering_stage[:28]:28s} | Carbide r = {res.mean_carbide_radius_nm:4.1f} nm | YS = {res.estimated_yield_strength_MPa:5.0f} MPa | Charpy = {res.estimated_charpy_energy_J:4.1f} J | DBTT = {res.dbtt_C:+4.1f} °C")


if __name__ == "__main__":
    main()
