"""
Fe³⁺ shuttle and O₂-driven bath aging during electrolysis (screening, L0).
The startup module (``models/bath_startup.py``) covers homogeneous Fe²⁺
autoxidation chemistry in a still bath.  **This** module closes the loop
during electrolysis: whatever Fe³⁺ the bath accumulates is reduced back to
Fe²⁺ at the flowing cathode — a parasitic "shuttle" current that steals
current efficiency — while any Fe³⁺ above the Fe(OH)₃ hydrolysis cap drops
out as sludge, bleeding the iron inventory.  The steady state of that
triangle (production → shuttle | sludge) is what is computed here.
Chemistry / transport
---------------------
1.  Production (homogeneous autoxidation), inherited from
    :func:`models.bath_startup.fe2_oxidation_rate`:
        rate = k_eff(T) · [Fe²⁺] · [O₂] · ([OH⁻]²-relative)   (4 Fe²⁺ + O₂ + 4H⁺ → 4 Fe³⁺ + 2H₂O)
    Dissolved O₂ is pinned at ``o2_fraction_of_sat`` of the Weiss (1970)
    air-saturation value — a screening stand-in for headspace mass
    transfer (sealed cell ≈ 0, open headspace ≈ 1).
2.  Optional anolyte/headspace crossover fault: an explicit O₂ molar flux
    reaching the catholyte (e.g. membrane defect), expressed as a
    fraction of the anode's O₂ generation rate j/(4F).
3.  Shuttle sink: mass-transfer-limited cathodic reduction
        Fe³⁺ + e⁻ → Fe²⁺,   flux = k_m · [Fe³⁺],  k_m = D_Fe3/δ,
    shuttle current density i_sh = F · k_m · [Fe³⁺] (A/m² of cathode).
4.  Hydrolysis cap: [Fe³⁺] ≤ Ksp/[OH⁻]³ (Fe(OH)₃, log Ksp ≈ −38.7);
    production in excess of the shuttle sink at the cap precipitates —
    that flux is reported as the iron inventory loss.
Steady state (well-mixed catholyte, area A, volume V)::
    [Fe³⁺]_ss = min(cap, r_ox / (k_m · A/V) + crossover contribution)
    i_shuttle,ss = F · k_m · [Fe³⁺]_ss ≤ F · (V/A) · r_ox
so the shuttle current is bounded by the oxidation rate times V/A and is
INDEPENDENT of k_m at steady state — the classic divided-cell CE leak.
All values are screening estimates; the homogeneous rate constant and O₂
ingress fractions are the dominant uncertainties (see the parameter
registry) and nothing here is gate evidence.
References
----------
* Sung & Morgan (1980) / Singer & Stumm (1970) — Fe²⁺ autoxidation law
  and its low-pH behaviour (as parameterised in bath_startup.py).
* Weiss (1970) — O₂ solubility.
* Huang & Zhang (2004) and refs. in bath_startup.py — Fe(OH)₃ solubility.
"""
from __future__ import annotations
from dataclasses import dataclass
from .bath_startup import dissolved_o2_saturation_mol_L, fe2_oxidation_rate
from .electrochemistry import FARADAY, M_FE
# Fe(OH)3 solubility product (25 °C): [Fe³⁺][OH⁻]³ ≈ 10^-38.7.
# Screening value; the literature spread is ~1 decade.
LOGKSP_FEOH3 = -38.7
# Fe³⁺ diffusivity, 25 °C anchor (same screening family as D_FE).
D_FE3_REF_M2_S = 5.5e-10
SCREENING_FLAG = "unvalidated (L0)"
def fe3_solubility_cap_M(pH: float) -> float:
    """[Fe³⁺] solubility cap from Fe(OH)3 hydrolysis (mol/L).
    cap = Ksp / [OH⁻]³ with pOH = 14 − pH (25 °C water autoprotolysis).
    """
    pOH = 14.0 - pH
    return 10.0 ** (LOGKSP_FEOH3 + 3.0 * pOH)
@dataclass(frozen=True)
class ShuttleScenario:
    """How oxygen reaches the catholyte during electrolysis."""
    name: str
    o2_fraction_of_sat: float      # dissolved O₂ level as fraction of air saturation
    crossover_o2_flux_mol_m2_s: float = 0.0  # extra O₂ crossing the membrane
    ascorbic_acid_M: float = 0.0   # sacrificial stabilizer in catholyte
def sealed_divided_cell() -> ShuttleScenario:
    """Intact membrane, closed headspace: only residual trace ingress."""
    return ShuttleScenario("sealed_divided_cell", o2_fraction_of_sat=0.005)
def open_headspace() -> ShuttleScenario:
    """Open beaker/tank headspace: catholyte near air saturation."""
    return ShuttleScenario("open_headspace", o2_fraction_of_sat=1.0)
def anolyte_crossover_fault(
    j_mA_cm2: float = 300.0, leak_fraction: float = 0.01
) -> ShuttleScenario:
    """Membrane fault: ``leak_fraction`` of the anode O₂ flux reaches the bath.
    O₂ generation at the anode is j/(4F) mol/m²/s; a 1 % leak on a
    300 mA/cm² anode delivers ~7.8 µmol/m²/s.
    """
    j_A_m2 = j_mA_cm2 * 10.0
    flux = leak_fraction * j_A_m2 / (4.0 * FARADAY)
    return ShuttleScenario(
        f"anolyte_crossover_fault_{leak_fraction:g}", 1.0,
        crossover_o2_flux_mol_m2_s=flux,
    )
@dataclass(frozen=True)
class ShuttleParams:
    """Catholyte state + cell bookkeeping for the shuttle balance."""
    temperature_C: float = 50.0
    pH: float = 2.0
    fe2_M: float = 1.0
    cathode_area_m2: float = 1.0e-3     # RC-1: 10 cm²
    catholyte_volume_L: float = 0.5
    boundary_layer_m: float = 50e-6
    d_fe3_m2_s: float = D_FE3_REF_M2_S
    k_ox_ref: float = 1.0e-4            # M⁻¹ s⁻¹ (bath_startup screening value)
    Ea_ox_J_mol: float = 50_000.0
def _km_fe3(p: ShuttleParams) -> float:
    """Mass-transfer coefficient for Fe³⁺→cathode (m/s)."""
    return p.d_fe3_m2_s / p.boundary_layer_m
def _production_rate_M_s(p: ShuttleParams, s: ShuttleScenario) -> float:
    """Fe³⁺ production rate in the catholyte (mol/L/s).
    Homogeneous autoxidation at the scenario-pinned O₂ level plus the
    area/volume-scaled crossover flux.
    """
    o2_M = s.o2_fraction_of_sat * dissolved_o2_saturation_mol_L(p.temperature_C)
    r_ox = fe2_oxidation_rate(
        p.fe2_M, o2_M, p.pH, p.temperature_C, p.k_ox_ref, p.Ea_ox_J_mol
    )
    v_m3 = p.catholyte_volume_L / 1000.0
    r_cross = s.crossover_o2_flux_mol_m2_s * p.cathode_area_m2 / v_m3  # O₂ → 4 Fe³⁺
    return r_ox + 4.0 * r_cross
def steady_state(p: ShuttleParams, s: ShuttleScenario) -> dict:
    """Steady-state Fe³⁺ shuttle picture for one catholyte/scenario pair."""
    r_prod = _production_rate_M_s(p, s)          # mol/L/s
    area_per_vol_m = p.cathode_area_m2 / (p.catholyte_volume_L / 1000.0)
    km = _km_fe3(p)
    cap = fe3_solubility_cap_M(p.pH)
    fe3_uncapped = r_prod / (km * area_per_vol_m)
    fe3_ss = min(fe3_uncapped, cap)
    precipitated = fe3_uncapped > cap
    i_shuttle = FARADAY * km * fe3_ss * 1000.0     # A/m² (fe3_ss M → mol/m³)
    # Iron inventory loss to Fe(OH)3 sludge, mol Fe /m²/s of cathode:
    if precipitated:
        sludge_flux = (r_prod - km * area_per_vol_m * fe3_ss) / area_per_vol_m
    else:
        sludge_flux = 0.0
    return {
        "scenario": s.name,
        "o2_M": s.o2_fraction_of_sat * dissolved_o2_saturation_mol_L(p.temperature_C),
        "fe3_production_M_s": r_prod,
        "fe3_ss_M": fe3_ss,
        "fe3_solubility_cap_M": cap,
        "feoh3_precipitation_active": precipitated,
        "i_shuttle_A_m2": i_shuttle,
        "iron_sludge_loss_mol_m2_s": sludge_flux,
        # mol/L/s × kg/mol × s/day × 1000 g/kg → g/L/day
        "iron_sludge_loss_g_L_day": (r_prod - km * area_per_vol_m * fe3_ss) * M_FE * 86400.0 * 1000.0,
        "flag": SCREENING_FLAG,
    }
def ce_penalty_at_j(i_shuttle_A_m2: float, j_mA_cm2: float) -> float:
    """Fraction of galvanostatic current lost to the Fe³⁺ shuttle.
    The shuttle current rides on top of the intended partial currents, so
    for an applied j the FE loss is ≈ i_sh/j at first order (exact in the
    small-shuttle limit this screening model lives in).
    """
    return float(i_shuttle_A_m2 / (j_mA_cm2 * 10.0))
def scenario_table(
    p: ShuttleParams, scenarios=None, j_mA_cm2: float = 300.0
) -> dict:
    """Compare the standard scenarios at one operating current density."""
    if scenarios is None:
        scenarios = (sealed_divided_cell(), open_headspace(),
                     anolyte_crossover_fault(j_mA_cm2, 0.01))
    rows = []
    for s in scenarios:
        ss = steady_state(p, s)
        ss["ce_loss_fraction_at_j"] = ce_penalty_at_j(ss["i_shuttle_A_m2"], j_mA_cm2)
        rows.append(ss)
    return {"j_mA_cm2": j_mA_cm2, "rows": rows, "flag": SCREENING_FLAG}
def main() -> None:
    """Print the RC-1-bath shuttle summary (screening, L0)."""
    p = ShuttleParams(pH=2.35)   # RC-1 catholyte activity-scale pH
    table = scenario_table(p, j_mA_cm2=300.0)
    print("Fe³⁺ SHUTTLE / O₂ BATH AGING — unvalidated (L0)")
    for row in table["rows"]:
        print(
            f"{row['scenario']:>32}: Fe³⁺ss {row['fe3_ss_M']:.2e} M, "
            f"i_shuttle {row['i_shuttle_A_m2']:.2e} A/m², "
            f"CE loss {row['ce_loss_fraction_at_j'] * 100:.3f}%"
            + (", Fe(OH)₃ sludge" if row["feoh3_precipitation_active"] else "")
        )
    print("NOT gate evidence; gates are measurement-only (models/process_gates.py).")
if __name__ == "__main__":
    main()
