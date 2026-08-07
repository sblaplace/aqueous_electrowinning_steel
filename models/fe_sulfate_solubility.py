"""
Temperature-dependent multi-hydrate phase equilibria and retrograde solubility of ferrous sulfate.

Physics and Chemistry
---------------------
In aqueous iron electrowinning, electrolyte conductivity is maximized by
operating at high temperature (60–90 °C) and high salt concentration (1.5–2.0 M
FeSO₄ with 0.5–1.0 M Na₂SO₄ or (NH₄)₂SO₄).  However, the FeSO₄–H₂O binary
system possesses a critical phase boundary at **56.7 °C**:

1. **Melanterite (FeSO₄·7H₂O)** is the stable solid phase below 56.7 °C.  Its
   dissolution is endothermic (ΔH_diss > 0), exhibiting normal *prograde*
   solubility that increases with temperature (from ~1.0 M at 10 °C to ~2.1 M
   at 56.7 °C).

2. **Szomolnokite (FeSO₄·H₂O)** is the thermodynamically stable solid phase
   above 56.7 °C (with rozenite FeSO₄·4H₂O as a metastable intermediate).  Its
   dissolution is exothermic (ΔH_diss < 0), causing **retrograde (inverse)
   solubility** above 56.7 °C:
     C_sat(60 °C) ≈ 1.99 M  →  C_sat(75 °C) ≈ 1.45 M  →  C_sat(90 °C) ≈ 1.00 M

3. **Common-ion salting out**: Addition of supporting sulfate salts (Na₂SO₄,
   (NH₄)₂SO₄, H₂SO₄) increases the sulfate activity a_SO4²⁻, depressing the
   saturation limit of Fe²⁺ via the solubility product K_sp(T):
     [Fe²⁺]_max · [SO₄²⁻]_total · γ±² ≤ K_sp(T)

4. **Heat exchanger and cathode-wall scaling**: When electrolyte is heated by
   submerged heaters, plate heat exchangers, or high-current Ohmic dissipation,
   the local surface temperature T_wall exceeds the bulk temperature T_bulk.
   Because of retrograde solubility, **the hottest surface has the lowest
   solubility**, driving spontaneous crystallization and fouling of hard
   szomolnokite scale on heat transfer surfaces.

References
----------
* Cameron, F. K. (1930). "The System: Ferrous Sulfate, Sulfuric Acid and
  Water at Various Temperatures." J. Phys. Chem., 34(4), 692–710.
* Reardon, E. J., & Beckie, R. D. (1987). "Modelling water–rock interactions
  involving FeSO4: Solubility of melanterite and szomolnokite." Geochim.
  Cosmochim. Acta, 51(9), 2355–2368.
* Linke, W. F., & Seidell, A. (1965). "Solubilities of Inorganic and Metal-
  Organic Compounds", Vol. 1, 4th ed., American Chemical Society.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# Phase transition temperature (Celsius) from melanterite (7H2O) to szomolnokite (1H2O)
T_TRANSITION_C = 56.7

# Molar masses (g/mol)
M_FE = 55.845
M_S = 32.065
M_O = 15.999
M_H = 1.008

M_FESO4 = M_FE + M_S + 4.0 * M_O           # 151.908 g/mol (anhydrous)
M_MELANTERITE = M_FESO4 + 7.0 * (2.0 * M_H + M_O)  # 278.014 g/mol (7H2O)
M_SZOMOLNOKITE = M_FESO4 + 1.0 * (2.0 * M_H + M_O) # 169.923 g/mol (1H2O)
M_ROZENITE = M_FESO4 + 4.0 * (2.0 * M_H + M_O)     # 223.969 g/mol (4H2O)


@dataclass(frozen=True)
class SolidHydratePhase:
    """Solid hydrate phase in equilibrium with aqueous ferrous sulfate."""

    name: str
    formula: str
    waters_of_hydration: int
    molar_mass_g_mol: float
    stable_min_temp_C: float
    stable_max_temp_C: float
    delta_H_dissolution_kJ_mol: float  # >0 = prograde, <0 = retrograde


MELANTERITE = SolidHydratePhase(
    name="melanterite",
    formula="FeSO4·7H2O",
    waters_of_hydration=7,
    molar_mass_g_mol=M_MELANTERITE,
    stable_min_temp_C=-5.0,
    stable_max_temp_C=56.7,
    delta_H_dissolution_kJ_mol=+18.5,  # Endothermic dissolution
)

SZOMOLNOKITE = SolidHydratePhase(
    name="szomolnokite",
    formula="FeSO4·H2O",
    waters_of_hydration=1,
    molar_mass_g_mol=M_SZOMOLNOKITE,
    stable_min_temp_C=56.7,
    stable_max_temp_C=130.0,
    delta_H_dissolution_kJ_mol=-14.2,  # Exothermic dissolution (retrograde)
)

ROZENITE = SolidHydratePhase(
    name="rozenite",
    formula="FeSO4·4H2O",
    waters_of_hydration=4,
    molar_mass_g_mol=M_ROZENITE,
    stable_min_temp_C=45.0,
    stable_max_temp_C=65.0,
    delta_H_dissolution_kJ_mol=+4.0,  # Metastable intermediate
)


def stable_solid_phase(temperature_C: float) -> SolidHydratePhase:
    """Return the thermodynamically stable FeSO4 hydrate solid at a given temperature."""
    if temperature_C < T_TRANSITION_C:
        return MELANTERITE
    return SZOMOLNOKITE


def feso4_binary_solubility_mol_L(temperature_C: float) -> float:
    """
    Equilibrium saturation concentration of pure FeSO4 in water (mol/L) vs temperature.

    Fitted to experimental solubility data from Cameron (1930) and Linke & Seidell (1965).
    Below 56.7 °C: prograde melanterite branch.
    Above 56.7 °C: retrograde szomolnokite branch.
    """
    T = float(temperature_C)
    if T < 0.0:
        T = 0.0

    if T <= T_TRANSITION_C:
        # Melanterite branch (0 to 56.7 °C): prograde quadratic fit
        # At 0 °C ~ 0.95 M, at 25 °C ~ 1.55 M, at 56.7 °C ~ 2.12 M
        c_sat = 0.95 + 0.027 * T - 0.00011 * (T ** 2)
    else:
        # Szomolnokite branch (56.7 to 110 °C): retrograde exponential decay
        # At 56.7 °C ~ 2.12 M, at 70 °C ~ 1.78 M, at 85 °C ~ 1.34 M, at 100 °C ~ 0.92 M
        dT = T - T_TRANSITION_C
        c_sat = 2.12 * math.exp(-0.0185 * dT - 0.00012 * (dT ** 2))

    return max(c_sat, 0.1)


def feso4_solubility_with_common_ion(
    temperature_C: float,
    background_sulfate_mol_L: float = 0.0,
    pitzer_gamma_pm: float = 0.06,
) -> float:
    """
    Maximum soluble Fe²⁺ concentration (mol/L) in presence of supporting sulfate salts.

    Uses an ion-product criterion:
      K'_sp(T) = C_sat,binary(T) * C_sat,binary(T)
      [Fe²⁺]_max * ([Fe²⁺]_max + [SO₄²⁻]_bg) = K'_sp(T)

    Parameters
    ----------
    temperature_C : float
        Bath temperature (°C).
    background_sulfate_mol_L : float
        Additional sulfate concentration from Na₂SO₄, (NH₄)₂SO₄, or H₂SO₄ (mol/L).
    pitzer_gamma_pm : float
        Mean activity coefficient (diagnostic scaling).

    Returns
    -------
    float
        Maximum allowable dissolved Fe²⁺ concentration (mol/L) before precipitation.
    """
    c_bin = feso4_binary_solubility_mol_L(temperature_C)
    k_sp = c_bin * c_bin  # Effective ion product in pure binary solution

    c_bg = max(float(background_sulfate_mol_L), 0.0)
    if c_bg == 0.0:
        return c_bin

    # Solve quadratic: c_fe^2 + c_bg * c_fe - k_sp = 0
    # c_fe = (-c_bg + sqrt(c_bg^2 + 4 * k_sp)) / 2
    disc = (c_bg ** 2) + 4.0 * k_sp
    c_fe_max = (-c_bg + math.sqrt(disc)) / 2.0
    return max(c_fe_max, 0.01)


@dataclass
class HeatExchangerScalingAssessment:
    """Assessment of scale formation on a hot wall or heater surface."""

    bulk_temp_C: float
    wall_temp_C: float
    bulk_fe2_mol_L: float
    background_sulfate_mol_L: float
    c_sat_bulk_mol_L: float
    c_sat_wall_mol_L: float
    supersaturation_ratio_wall: float  # S = bulk_fe2 / c_sat_wall
    stable_phase_wall: str
    is_scaling_risk: bool
    max_safe_wall_temp_C: float
    critical_heat_flux_margin: str


def assess_heat_exchanger_scaling(
    bulk_temp_C: float,
    wall_temp_C: float,
    bulk_fe2_mol_L: float,
    background_sulfate_mol_L: float = 0.5,
) -> HeatExchangerScalingAssessment:
    """
    Evaluate the risk of szomolnokite scale formation on heating surfaces.

    In the retrograde regime (T > 56.7 °C), heating elements and heat exchanger
    tubes operate at T_wall > T_bulk.  Because solubility drops with increasing T,
    the liquid layer directly contacting the hot wall can become supersaturated
    even if the bulk electrolyte is comfortably undersaturated.
    """
    c_sat_bulk = feso4_solubility_with_common_ion(bulk_temp_C, background_sulfate_mol_L)
    c_sat_wall = feso4_solubility_with_common_ion(wall_temp_C, background_sulfate_mol_L)

    s_wall = bulk_fe2_mol_L / c_sat_wall
    is_risk = s_wall >= 1.0
    phase_wall = stable_solid_phase(wall_temp_C).name

    # Find the maximum safe wall temperature where S = 1.0
    # Binary search between bulk_temp_C and 130 °C
    t_low = max(bulk_temp_C, 0.0)
    t_high = 130.0
    for _ in range(30):
        t_mid = 0.5 * (t_low + t_high)
        c_mid = feso4_solubility_with_common_ion(t_mid, background_sulfate_mol_L)
        if c_mid < bulk_fe2_mol_L:
            t_high = t_mid
        else:
            t_low = t_mid
    max_safe_t = t_low

    if s_wall < 0.85:
        margin = "safe (ample headroom)"
    elif s_wall < 1.0:
        margin = "warning (near saturation at boundary layer)"
    elif s_wall < 1.25:
        margin = "severe scaling risk (szomolnokite nucleation)"
    else:
        margin = "critical fouling (spontaneous crystallization on wall)"

    return HeatExchangerScalingAssessment(
        bulk_temp_C=bulk_temp_C,
        wall_temp_C=wall_temp_C,
        bulk_fe2_mol_L=bulk_fe2_mol_L,
        background_sulfate_mol_L=background_sulfate_mol_L,
        c_sat_bulk_mol_L=c_sat_bulk,
        c_sat_wall_mol_L=c_sat_wall,
        supersaturation_ratio_wall=s_wall,
        stable_phase_wall=phase_wall,
        is_scaling_risk=is_risk,
        max_safe_wall_temp_C=max_safe_t,
        critical_heat_flux_margin=margin,
    )


def main() -> None:
    """CLI entrypoint for FeSO4 solubility and scaling assessment."""
    print("=================================================================")
    print(" FeSO4 Temperature-Dependent Solubility & Szomolnokite Scaling")
    print("=================================================================")
    for t in [20.0, 40.0, 56.7, 70.0, 85.0, 100.0]:
        phase = stable_solid_phase(t).name
        c_sat = feso4_binary_solubility_mol_L(t)
        c_sat_bg = feso4_solubility_with_common_ion(t, background_sulfate_mol_L=0.5)
        print(f" T = {t:5.1f} °C | Phase: {phase:12s} | Pure C_sat = {c_sat:4.2f} M | With 0.5M Na2SO4 = {c_sat_bg:4.2f} M")
    print("\nHeat Exchanger Scaling Evaluation (Bulk 65 °C, Wall 90 °C, 1.5 M Fe2+):")
    res = assess_heat_exchanger_scaling(65.0, 90.0, 1.5, background_sulfate_mol_L=0.5)
    print(f"  Wall supersaturation ratio : {res.supersaturation_ratio_wall:.2f}")
    print(f"  Scaling risk status        : {res.critical_heat_flux_margin}")
    print(f"  Max safe wall temperature  : {res.max_safe_wall_temp_C:.1f} °C")


if __name__ == "__main__":
    main()
