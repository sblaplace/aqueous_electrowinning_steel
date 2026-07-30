"""
Mechanical properties model for electrodeposited iron/steel.

Bridges Phase III co-deposition predictions (composition, carbon content)
and Phase I/II transport/pulse dynamics (grain size, porosity) to
structural-grade mechanical metrics: yield strength, UTS, hardness,
elongation, and ASTM/AISI grade mapping.

This is a *screening-level* empirical model, not a full CPFEM or
micromechanics solver. All coefficients are labeled as assumptions and
must be calibrated with real Vickers, tensile, and EBSD data.

Mechanisms implemented
----------------------
* Hall-Petch grain-size strengthening (pulse-refined grains)
* Solid-solution strengthening from Ni (and optional Cr/Mn placeholders)
* Dispersion / load-transfer strengthening from incorporated carbon particles
* Porosity / HER embrittlement penalty (from current efficiency and HER flux)
* Grain-size estimation from deposition waveform (DC vs PE vs PRE)

References (screening calibrations)
-----------------------------------
* Hall-Petch for bcc Fe: sigma0 ~ 70-150 MPa, k_HP ~ 0.3-0.74 MPa·m^0.5
  (Takaki 2002; Rajagopalan & Vaidya compilation)
* Electrodeposited Fe grain size: DC 1-5 µm, pulse 0.2-1 µm, PRE 0.1-0.5 µm
  (typical for saccharin-leveled baths and pulse-reverse)
* Ni solid-solution in ferrite: ~ 40-70 MPa per wt% Ni at low % (Leslie 1981;
  Hall-Petch enriched datasets), with ~ (wt%)^0.75 saturation
* Dispersion strengthening by carbon / carbides: Orowan + load-transfer
  Screening fit from Fe-CNT and Fe-graphene composite plating literature
* Tabor relation HV ≈ 3σ_y (in consistent units) is used for hardness estimate
* UTS/YS ratio and elongation trends from cold-worked / electrodeposited Fe
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal
import math
import numpy as np


# -------------------------------------------------------------------------
# Constants and empirical calibration defaults (screening)
# -------------------------------------------------------------------------

# Iron
G_FE_GPA = 80.0           # Shear modulus (GPa)
BURGERS_B_M = 0.25e-9     # Burgers vector (m)

# Hall-Petch defaults for electrodeposited bcc Fe (calibrated to 1-10 µm data)
SIGMA0_FE_MPA = 100.0     # Friction stress / lattice resistance (MPa)
K_HP_MPA_SQRT_M = 0.50    # Hall-Petch slope MPa·m^0.5 (mean of 0.35-0.65 range)
K_HP_MPA_SQRT_UM = K_HP_MPA_SQRT_M * 1e3  # converted: 1 m = 1e6 µm, sqrt = 1e3

# Solid-solution strengthening coefficient for Ni in Fe (screening)
K_SS_NI_MPA_PER_WT = 38.0   # MPa per wt% Ni in linear regime
K_SS_NI_EXP = 0.75         # saturation exponent
SS_NI_SAT_WT = 20.0        # wt% where saturation becomes strong

# Carbon particle strengthening (empirical composite literature)
K_CARBON_MPA_PER_WT = 180.0   # MPa per wt% C at 1 µm, 1 wt% loading (dispersion)
CARBON_NL_EXP = 0.6           # sub-linear with loading (clustering limits)
CARBON_SIZE_REF_UM = 1.5      # reference particle size (µm)
CARBON_SIZE_EXP = -0.25       # smaller particles stronger (Orowan λ^-1)

# Load-transfer contribution fraction
LOAD_TRANSFER_FRAC = 0.15

# Porosity / HER penalty
POROSITY_PENALTY_EXP = 1.8
POROSITY_MAX = 0.30            # cap at 30% equivalent porosity for FE→0

# Hardness conversion (Tabor)
# HV (kgf/mm2) ≈ σ_y (MPa) / 3.0-3.3 for work-hardened metals; use 3.2 with spread
TABOR_FACTOR = 3.2
HV_MPA = 9.80665               # 1 kgf/mm2 = 9.80665 MPa

# UTS and elongation heuristics
UTS_OVER_YS_BASE = 1.25
UTS_OVER_YS_HIGH_CARBON = 1.45
ELONGATION_BASE_PCT = 22.0     # pure coarse-grained Fe (annealed) ~ 30-40%, as-deposited lower


@dataclass(frozen=True)
class GrainSizeParams:
    """Parameters governing grain-size estimation from plating conditions."""

    d0_dc_ref_um: float = 3.5       # reference DC grain size at 100 mA/cm², 60°C
    j_ref_mA_cm2: float = 100.0
    j_exponent: float = 0.30        # grain refines with increasing j: d ∝ (j_ref/j)^n
    temp_coeff_per_C: float = 0.015 # grain growth per °C above reference
    t_ref_C: float = 60.0

    # Pulse bonuses (empirical multipliers vs DC)
    pe_factor_base: float = 0.65    # base reduction for 50% duty PE vs DC
    pe_duty_exp: float = 0.40       # further refinement at low duty
    pe_j_ratio_exp: float = 0.25    # refinement with peak/avg ratio

    pre_factor_base: float = 0.35   # base reduction for PRE vs DC (extra renucleation)
    pre_duty_exp: float = 0.50
    pre_j_ratio_exp: float = 0.32

    d_min_um: float = 0.08          # physical floor (nanocrystalline limit ~80 nm)
    d_max_um: float = 10.0          # cap for anomalous low-j DC

    def __post_init__(self):
        if self.d0_dc_ref_um <= 0 or self.j_ref_mA_cm2 <= 0:
            raise ValueError("reference grain size and current must be positive")
        if not 0 < self.d_min_um < self.d_max_um:
            raise ValueError("must have 0 < d_min < d_max")


@dataclass(frozen=True)
class MechanicalPropertiesParams:
    """Calibration constants for mechanical-properties model (all screening)."""

    sigma0_MPa: float = SIGMA0_FE_MPA
    k_hp_MPa_sqrt_m: float = K_HP_MPA_SQRT_M

    k_ss_ni_MPa_per_wt: float = K_SS_NI_MPA_PER_WT
    ss_ni_exp: float = K_SS_NI_EXP
    ss_ni_sat_wt: float = SS_NI_SAT_WT

    k_carbon_MPa_per_wt: float = K_CARBON_MPA_PER_WT
    carbon_nl_exp: float = CARBON_NL_EXP
    carbon_size_ref_um: float = CARBON_SIZE_REF_UM
    carbon_size_exp: float = CARBON_SIZE_EXP

    load_transfer_frac: float = LOAD_TRANSFER_FRAC
    porosity_penalty_exp: float = POROSITY_PENALTY_EXP
    porosity_max: float = POROSITY_MAX
    tabor_factor: float = TABOR_FACTOR
    uts_over_ys_base: float = UTS_OVER_YS_BASE
    elongation_base_pct: float = ELONGATION_BASE_PCT

    def __post_init__(self):
        if self.sigma0_MPa < 0 or self.k_hp_MPa_sqrt_m <= 0:
            raise ValueError("Hall-Petch params invalid")
        if self.k_ss_ni_MPa_per_wt < 0 or self.k_carbon_MPa_per_wt < 0:
            raise ValueError("strengthening coefficients must be non-negative")


def estimate_grain_size_um(
    j_avg_mA_cm2: float = 100.0,
    j_peak_mA_cm2: Optional[float] = None,
    duty_cycle: float = 1.0,
    waveform: Literal["dc", "pe", "pre"] = "dc",
    temperature_C: float = 60.0,
    params: Optional[GrainSizeParams] = None,
) -> float:
    """
    Screening-level grain-size estimator for electrodeposited iron.

    Parameters
    ----------
    j_avg_mA_cm2 : average (or DC) current density
    j_peak_mA_cm2 : peak cathodic j (for PE/PRE). If None, assumes DC with peak=avg.
    duty_cycle : cathodic duty (0-1); 1.0 for DC
    waveform : 'dc' | 'pe' | 'pre'  (pulse-reverse gives finest grains)
    temperature_C : bath temperature (grain growth at higher T)
    params : GrainSizeParams or defaults

    Returns
    -------
    float
        Estimated mean linear-intercept grain size (µm)
    """
    p = params or GrainSizeParams()
    if j_avg_mA_cm2 <= 0:
        raise ValueError("j_avg_mA_cm2 must be positive")
    j_peak = j_peak_mA_cm2 if j_peak_mA_cm2 is not None else j_avg_mA_cm2
    if j_peak <= 0:
        raise ValueError("j_peak must be positive")
    if not 0 < duty_cycle <= 1.0:
        raise ValueError("duty_cycle must be in (0,1]")

    # Base DC trend: higher j → higher overpotential → higher nucleation rate → finer grains
    d_dc = p.d0_dc_ref_um * (p.j_ref_mA_cm2 / j_avg_mA_cm2) ** p.j_exponent
    # Temperature coarsening
    d_dc *= math.exp(p.temp_coeff_per_C * (temperature_C - p.t_ref_C))
    d_dc = float(np.clip(d_dc, p.d_min_um, p.d_max_um))

    if waveform == "dc":
        return d_dc

    j_ratio = j_avg_mA_cm2 / max(j_peak, 1e-12)  # avg/peak ≤1; smaller ratio = more pulsing bonus
    # For PE/PRE, low duty + high peak/avg ratio refines grains
    if waveform == "pe":
        # Blend: interpolates between pe_factor_base (at duty=0) and 1.0 (at duty=1),
        # normalized at 50% duty, scaled by peak/avg current ratio.
        factor = p.pe_factor_base + (1 - p.pe_factor_base) * duty_cycle
        factor *= (j_ratio ** p.pe_j_ratio_exp) / (0.5 ** p.pe_j_ratio_exp)  # normalize at 50% duty
        factor = max(factor, 0.18)
    else:  # pre
        factor = p.pre_factor_base + (1 - p.pre_factor_base) * duty_cycle * 0.6
        factor *= (j_ratio ** p.pre_j_ratio_exp) / (0.5 ** p.pre_j_ratio_exp)
        factor = max(factor, 0.10)

    d = d_dc * factor
    return float(np.clip(d, p.d_min_um, p.d_max_um))


def solid_solution_strengthening_MPa(
    ni_wt_percent: float = 0.0,
    mn_wt_percent: float = 0.0,
    cr_wt_percent: float = 0.0,
    params: Optional[MechanicalPropertiesParams] = None,
) -> float:
    """
    Solid-solution strengthening contribution (MPa).

    Screening model calibrated to Fe-Ni data; Mn and Cr use smaller coefficients.
    """
    p = params or MechanicalPropertiesParams()
    ni = max(ni_wt_percent, 0.0)
    mn = max(mn_wt_percent, 0.0)
    cr = max(cr_wt_percent, 0.0)

    # Ni term with saturation: Δσ = k * wt^n / (1 + wt/ wt_sat)
    ni_term = p.k_ss_ni_MPa_per_wt * (ni ** p.ss_ni_exp) / (1.0 + ni / p.ss_ni_sat_wt) if ni > 0 else 0.0
    # Mn and Cr placeholder coefficients (weaker for Mn, stronger for Cr)
    mn_term = 25.0 * (mn ** 0.75) / (1.0 + mn / 15.0) if mn > 0 else 0.0
    cr_term = 45.0 * (cr ** 0.70) / (1.0 + cr / 12.0) if cr > 0 else 0.0

    return float(ni_term + mn_term + cr_term)


def carbon_dispersion_strengthening_MPa(
    carbon_wt_percent: float,
    particle_size_um: float = 1.5,
    particle_density_kg_m3: float = 2200.0,
    matrix_density_kg_m3: float = 7874.0,
    params: Optional[MechanicalPropertiesParams] = None,
) -> tuple[float, float, float]:
    """
    Dispersion + load-transfer strengthening from incorporated carbon particles.

    Returns (delta_orowan, delta_load_transfer, total_delta) in MPa.

    Screening treatment combining:
    * Orowan bypass: Δσ ≈ 0.4Gb/πλ · ln(r/b) with λ interparticle spacing
    * Load-transfer: Δσ_LT ≈ 0.5 f_v σ_matrix  (shear-lag approximation)
    * Empirically anchored to composite-plating literature via k_carbon factor.
    """
    p = params or MechanicalPropertiesParams()
    c_wt = max(carbon_wt_percent, 0.0)
    if c_wt <= 1e-9:
        return 0.0, 0.0, 0.0

    d_um = max(particle_size_um, 0.05)
    r_m = (d_um * 1e-6) / 2.0

    # Volume fraction from wt% (rule: f_v = (wt_c/ρ_c) / (wt_c/ρ_c + wt_m/ρ_m))
    c_mass_frac = c_wt / 100.0
    m_mass_frac = 1.0 - c_mass_frac
    vol_c = c_mass_frac / particle_density_kg_m3
    vol_m = m_mass_frac / matrix_density_kg_m3
    f_v = vol_c / (vol_c + vol_m) if (vol_c + vol_m) > 0 else 0.0
    f_v = min(f_v, 0.25)  # physical cap for aggregated particles

    # Interparticle spacing λ ≈ r * (2π / (3 f_v))^0.5 (equilateral array approx)
    if f_v <= 1e-12:
        lambda_m = 1e-3
    else:
        lambda_m = r_m * math.sqrt(2.0 * math.pi / (3.0 * f_v))

    # Orowan term (theoretical upper bound)
    G_Pa = G_FE_GPA * 1e9
    b_m = BURGERS_B_M
    ln_term = math.log(max(r_m / b_m, 2.0))
    orowan_theor_MPa = (0.4 * G_Pa * b_m / (math.pi * max(lambda_m, 1e-9)) * ln_term) / 1e6

    # Empirical dispersion term that maps to measured composite data:
    # Δσ_empirical = k_c * (wt%)^n * (size_ref/size)^exp_size * sqrt(f_v) factor
    size_factor = (p.carbon_size_ref_um / d_um) ** (-p.carbon_size_exp) if d_um > 0 else 1.0
    # Actually carbon_size_exp = -0.25, so smaller d → larger factor; correct orientation:
    size_factor = (p.carbon_size_ref_um / d_um) ** abs(p.carbon_size_exp)
    disp_empirical = p.k_carbon_MPa_per_wt * (c_wt ** p.carbon_nl_exp) * size_factor

    # Use the smaller of theoretical bound and empirical to avoid absurd values
    delta_orowan = min(orowan_theor_MPa, disp_empirical) if f_v > 1e-6 else disp_empirical * 0.3
    delta_orowan = max(delta_orowan, 0.0)

    # Load-transfer (shear lag)
    # Approximate σ_matrix ~ 250 MPa baseline; fraction matters more than absolute
    sigma_matrix_ref_MPa = 300.0
    delta_lt = p.load_transfer_frac * f_v * sigma_matrix_ref_MPa

    total = delta_orowan + delta_lt
    return float(delta_orowan), float(delta_lt), float(total)


def hall_petch_yield_MPa(
    grain_size_um: float,
    params: Optional[MechanicalPropertiesParams] = None,
) -> float:
    """Hall-Petch yield strength from grain size (MPa)."""
    p = params or MechanicalPropertiesParams()
    d_m = max(grain_size_um * 1e-6, 1e-9)
    return float(p.sigma0_MPa + p.k_hp_MPa_sqrt_m / math.sqrt(d_m))


def porosity_factor(
    current_efficiency_percent: float = 100.0,
    her_flux_mol_m2_hr: Optional[float] = None,
    params: Optional[MechanicalPropertiesParams] = None,
) -> tuple[float, float]:
    """
    Estimate porosity-induced knockdown factor (0-1) for strength.

    Returns (porosity_estimate_0-1, strength_knockdown_factor_0-1)
    where factor multiplies ideal strength.

    Low current efficiency → high HER → more hydrogen pits, voids, inclusions.
    """
    p = params or MechanicalPropertiesParams()
    ce = float(np.clip(current_efficiency_percent, 0.0, 100.0)) / 100.0
    # Map CE to porosity: CE=1 → 0% porosity, CE=0 → porosity_max
    # Non-linear because small HER is mostly escaping, large HER traps
    porosity = p.porosity_max * (1.0 - ce) ** 1.2
    if her_flux_mol_m2_hr is not None:
        # Extra HER flux increases porosity beyond CE prediction
        her_factor = min(her_flux_mol_m2_hr / 20.0, 1.0)  # 20 mol/m2/hr is large
        porosity = max(porosity, porosity * 0.6 + her_factor * p.porosity_max * 0.4)
    porosity = float(np.clip(porosity, 0.0, p.porosity_max))

    # Strength knockdown: (1-P)^n  (Gibson-Ashby type)
    knockdown = (1.0 - porosity) ** p.porosity_penalty_exp
    return porosity, float(np.clip(knockdown, 0.25, 1.0))


@dataclass
class MechanicalPropertiesResult:
    """Complete mechanical properties prediction."""

    grain_size_um: float
    ce_percent: float
    ni_wt_percent: float
    carbon_wt_percent: float
    particle_size_um: float
    waveform: str

    # Intermediate contributions
    porosity: float
    porosity_factor: float
    sigma_hp_MPa: float
    delta_ss_MPa: float
    delta_orowan_MPa: float
    delta_lt_MPa: float
    delta_carbon_total_MPa: float

    # Final properties
    sigma_y_MPa: float
    uts_MPa: float
    elongation_pct: float
    vickers_hv: float
    vickers_hv_MPa: float
    specific_energy_impact: str
    grade_estimate: str
    flags: list[str]

    def summary(self) -> dict[str, Any]:
        return {
            "grain_size_um": round(self.grain_size_um, 3),
            "porosity": round(self.porosity, 4),
            "porosity_factor": round(self.porosity_factor, 3),
            "hall_petch_yield_MPa": round(self.sigma_hp_MPa, 1),
            "solid_solution_MPa": round(self.delta_ss_MPa, 1),
            "carbon_orowan_MPa": round(self.delta_orowan_MPa, 1),
            "carbon_load_transfer_MPa": round(self.delta_lt_MPa, 1),
            "carbon_total_MPa": round(self.delta_carbon_total_MPa, 1),
            "yield_strength_MPa": round(self.sigma_y_MPa, 1),
            "uts_MPa": round(self.uts_MPa, 1),
            "elongation_percent": round(self.elongation_pct, 1),
            "vickers_hv_kgf_mm2": round(self.vickers_hv, 1),
            "vickers_hv_MPa": round(self.vickers_hv_MPa, 1),
            "grade_estimate": self.grade_estimate,
            "waveform": self.waveform,
            "composition": {
                "ni_wt_pct": self.ni_wt_percent,
                "c_wt_pct": self.carbon_wt_percent,
            },
            "flags": self.flags,
        }


class MechanicalPropertiesModel:
    """
    Screening-level mechanical properties predictor for aqueous-electrodeposited steel.

    Example
    -------
    >>> model = MechanicalPropertiesModel()
    >>> result = model.predict(
    ...     j_avg_mA_cm2=100, j_peak_mA_cm2=200, duty_cycle=0.5,
    ...     waveform='pre', ni_wt_percent=2.5, carbon_wt_percent=0.8,
    ...     current_efficiency_percent=93.0, temperature_C=60
    ... )
    >>> print(result.sigma_y_MPa, result.vickers_hv, result.grade_estimate)
    """

    def __init__(
        self,
        grain_params: Optional[GrainSizeParams] = None,
        mech_params: Optional[MechanicalPropertiesParams] = None,
    ):
        self.grain_params = grain_params or GrainSizeParams()
        self.mech_params = mech_params or MechanicalPropertiesParams()

    def predict(
        self,
        j_avg_mA_cm2: float = 100.0,
        j_peak_mA_cm2: Optional[float] = None,
        duty_cycle: float = 1.0,
        waveform: Literal["dc", "pe", "pre"] = "dc",
        temperature_C: float = 60.0,
        ni_wt_percent: float = 0.0,
        mn_wt_percent: float = 0.0,
        cr_wt_percent: float = 0.0,
        carbon_wt_percent: float = 0.0,
        particle_size_um: float = 1.5,
        current_efficiency_percent: float = 95.0,
        her_flux_mol_m2_hr: Optional[float] = None,
        grain_size_override_um: Optional[float] = None,
    ) -> MechanicalPropertiesResult:
        """Run full prediction pipeline."""

        # Grain size
        if grain_size_override_um is not None:
            d_um = float(grain_size_override_um)
            if not 0.05 <= d_um <= 50:
                raise ValueError("grain_size_override_um out of physical range")
        else:
            d_um = estimate_grain_size_um(
                j_avg_mA_cm2=j_avg_mA_cm2,
                j_peak_mA_cm2=j_peak_mA_cm2,
                duty_cycle=duty_cycle,
                waveform=waveform,
                temperature_C=temperature_C,
                params=self.grain_params,
            )

        # Hall-Petch
        sigma_hp = hall_petch_yield_MPa(d_um, self.mech_params)

        # Solid solution
        delta_ss = solid_solution_strengthening_MPa(
            ni_wt_percent, mn_wt_percent, cr_wt_percent, self.mech_params
        )

        # Carbon dispersion
        delta_orowan, delta_lt, delta_c_total = carbon_dispersion_strengthening_MPa(
            carbon_wt_percent, particle_size_um, params=self.mech_params
        )

        # Porosity penalty
        porosity, porosity_factor_val = porosity_factor(
            current_efficiency_percent, her_flux_mol_m2_hr, self.mech_params
        )

        # Combined yield (adds quadratically for independent mechanisms: sqrt sum squares)
        # Here use linear sum for conservative upper bound, then apply porosity knockdown
        sigma_ideal = sigma_hp + delta_ss + delta_c_total
        sigma_y = sigma_ideal * porosity_factor_val

        # UTS estimate: increased ratio for high-carbon deposits (strain hardening from particles)
        if carbon_wt_percent > 1.0:
            uts_ratio = self.mech_params.uts_over_ys_base + 0.15 * min(carbon_wt_percent / 2.0, 1.0)
        else:
            uts_ratio = self.mech_params.uts_over_ys_base + 0.05 * min(ni_wt_percent / 10.0, 1.0)
        uts_ratio = min(uts_ratio, UTS_OVER_YS_HIGH_CARBON)
        uts = sigma_y * uts_ratio

        # Elongation: base minus penalties for strength, carbon, porosity, grain refinement
        # High strength → lower elongation; carbon particles → embrittlement; porosity → premature failure
        elong = self.mech_params.elongation_base_pct
        elong -= 0.025 * max(sigma_y - 200.0, 0.0)  # -2.5% per 100 MPa above 200 MPa
        elong -= 1.5 * carbon_wt_percent  # carbon embrittlement
        elong -= 30.0 * porosity  # porosity penalty
        # Fine grains recover some ductility vs coarse if porosity low
        if d_um < 0.5 and porosity < 0.05:
            elong += 2.0
        elong = float(np.clip(elong, 0.5, 40.0))

        # Hardness via Tabor: HV (MPa) ≈ Tabor * σ_y ; HV kgf/mm2 = HV MPa / 9.81
        hv_MPa = self.mech_params.tabor_factor * sigma_y
        hv = hv_MPa / HV_MPA

        # Grade estimate (simplified mapping)
        grade = self._estimate_grade(sigma_y, uts, carbon_wt_percent, ni_wt_percent, elong)

        # QA flags
        flags: list[str] = []
        if porosity > 0.10:
            flags.append("high_porosity")
        if current_efficiency_percent < 80:
            flags.append("low_current_efficiency")
        if carbon_wt_percent > 5.0:
            flags.append("excessive_carbon")
        if d_um < 0.15:
            flags.append("nanocrystalline_grain_size")
        if sigma_y > 800:
            flags.append("very_high_strength_screening")
        if elong < 3.0:
            flags.append("low_ductility_risk")

        specific_energy_note = (
            f"Mechanical properties screening; specific energy not recalculated here. "
            f"Use electrochemistry model for kWh/t. Grain size = {d_um:.2f} µm ({waveform})"
        )

        return MechanicalPropertiesResult(
            grain_size_um=d_um,
            ce_percent=current_efficiency_percent,
            ni_wt_percent=ni_wt_percent,
            carbon_wt_percent=carbon_wt_percent,
            particle_size_um=particle_size_um,
            waveform=waveform,
            porosity=porosity,
            porosity_factor=porosity_factor_val,
            sigma_hp_MPa=sigma_hp,
            delta_ss_MPa=delta_ss,
            delta_orowan_MPa=delta_orowan,
            delta_lt_MPa=delta_lt,
            delta_carbon_total_MPa=delta_c_total,
            sigma_y_MPa=float(sigma_y),
            uts_MPa=float(uts),
            elongation_pct=float(elong),
            vickers_hv=float(hv),
            vickers_hv_MPa=float(hv_MPa),
            specific_energy_impact=specific_energy_note,
            grade_estimate=grade,
            flags=flags,
        )

    @staticmethod
    def _estimate_grade(
        sigma_y: float,
        uts: float,
        c_wt: float,
        ni_wt: float,
        elong: float,
    ) -> str:
        """Simplified ASTM/AISI grade screening."""

        # Low-carbon, low-alloy structural grades
        if c_wt < 0.3 and ni_wt < 1.0:
            if sigma_y < 250 and elong > 15:
                return "AISI 1005-1010 / ASTM A36-like (low-C structural, ductile)"
            elif sigma_y < 350:
                return "AISI 1018-like (low-Cstructural, good weldability)"
            elif sigma_y < 500:
                return "AISI 1020-1025 / HSLA-like (moderate strength, fine-grained)"
            else:
                return "AISI 1030-like / fine-grained high-strength low-C"

        # Carburized / composite
        if 0.3 <= c_wt < 0.8:
            if ni_wt < 2.0:
                return "AISI 1035-1045-like (medium-C, heat-treatable, composite)"
            else:
                return "Fe-Ni-C composite, AISI 4340/O1 tool-steel-range screening"

        if c_wt >= 0.8:
            return "High-C composite / tool steel screening (requires tempering verification)"

        if ni_wt >= 2.0:
            if ni_wt < 6.0:
                return f"Fe-{ni_wt:.1f}Ni low-alloy (AISI 4320-4340 family screening)"
            else:
                return f"Fe-{ni_wt:.1f}Ni high-Ni alloy (FCC-stabilized, austenitic screening)"

        return "Unclassified electrodeposit — verify via XRD/tensile"

    def sweep_current_density(
        self,
        j_values_mA_cm2: Optional[np.ndarray] = None,
        waveform: Literal["dc", "pe", "pre"] = "pe",
        duty_cycle: float = 0.5,
        ni_wt_percent: float = 0.0,
        carbon_wt_percent: float = 0.5,
        current_efficiency_percent: float = 93.0,
    ) -> Dict[str, np.ndarray]:
        """Sweep over current density for property trends (useful for plotting)."""

        if j_values_mA_cm2 is None:
            j_values_mA_cm2 = np.linspace(20.0, 400.0, 40)

        results = []
        for j in j_values_mA_cm2:
            # For screening, assume j_peak = j/duty for pulsed, j itself for DC
            j_peak = j / max(duty_cycle, 0.05) if waveform != "dc" else j
            r = self.predict(
                j_avg_mA_cm2=float(j),
                j_peak_mA_cm2=float(j_peak),
                duty_cycle=duty_cycle,
                waveform=waveform,
                ni_wt_percent=ni_wt_percent,
                carbon_wt_percent=carbon_wt_percent,
                current_efficiency_percent=current_efficiency_percent,
            )
            results.append(r)

        return {
            "j_mA_cm2": np.array(j_values_mA_cm2),
            "grain_size_um": np.array([x.grain_size_um for x in results]),
            "yield_MPa": np.array([x.sigma_y_MPa for x in results]),
            "uts_MPa": np.array([x.uts_MPa for x in results]),
            "hv": np.array([x.vickers_hv for x in results]),
            "elongation_pct": np.array([x.elongation_pct for x in results]),
            "porosity": np.array([x.porosity for x in results]),
        }


def build_mechanical_model_from_phase3_result(
    phase3_res: Dict[str, Any],
    j_avg_mA_cm2: float = 100.0,
    j_peak_mA_cm2: Optional[float] = None,
    duty_cycle: float = 0.5,
    waveform: Literal["dc", "pe", "pre"] = "pe",
    temperature_C: float = 60.0,
) -> MechanicalPropertiesResult:
    """
    Convenience adapter: take a PhaseIIICoDeposition run_at_current() result
    and feed it directly into the mechanical properties predictor.
    """

    alloy = phase3_res.get("alloy_kinetics", {})
    carbon = phase3_res.get("carbon_incorporation", {})

    ni_wt = float(alloy.get("ni_wt_percent", 0.0))
    c_wt = float(carbon.get("predicted_carbon_wt_percent", 0.0))
    ce = float(carbon.get("adjusted_ce_percent", alloy.get("current_efficiency_percent", 93.0)))

    model = MechanicalPropertiesModel()
    return model.predict(
        j_avg_mA_cm2=j_avg_mA_cm2,
        j_peak_mA_cm2=j_peak_mA_cm2,
        duty_cycle=duty_cycle,
        waveform=waveform,
        temperature_C=temperature_C,
        ni_wt_percent=ni_wt,
        carbon_wt_percent=c_wt,
        current_efficiency_percent=ce,
    )
