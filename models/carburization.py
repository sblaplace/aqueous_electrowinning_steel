"""
Post-deposition gaseous carburization model for electrodeposited iron/steel.

Models carbon diffusion into aqueous-electrowon iron sheet/strip at
elevated temperature (plasma, gas, or pack carburizing) as an alternative
to in-situ carbon-particle co-deposition.

This is a screening Fickian model with:
* error-function / finite-slab diffusion (Crank solutions)
* temperature-dependent diffusivity D = D0 exp(-Q/RT) for bcc α-Fe and fcc γ-Fe
* surface carbon potential (gas carburizing potential) as boundary condition
* effective case depth defined at 0.35-0.50 wt% C (typical AISI case criterion)
* as-quenched hardness from Maynier-type empirical C→HV correlation
* Hollomon-Jaffe tempering correction (optional)
* Carbon uptake mass, energy for heating, and mechanical case-core composite estimate

All parameters are screening assumptions to be calibrated with combustion,
hardness traverses, and XRD retained-austenite.

References (screening)
----------------------
* Fick's 2nd law semi-infinite solution: (C-C0)/(Cs-C0)=1-erf(x/2√Dt)=erfc(...)
* C diffusion in α-Fe (bcc, <912°C): D0≈6.2e-7 m²/s, Q≈80 kJ/mol (Wert & Zener range 0.02-2e-6)
* C diffusion in γ-Fe (fcc, 912-1394°C): D0≈15e-6–23e-6 m²/s, Q≈135-145 kJ/mol
* Typical gas carburizing: 850-950°C, Cs≈1.0-1.3 wt% surface, t=0.5-10 hr, case 0.3-1.5 mm
* Low-temp plasma carburizing: 400-550°C for stainless; for Fe 350-500°C gives shallow ~10-100 µm
* As-quenched hardness: Maynier et al., HV≈127+949C+27Si+... for C<0.8 wt%, saturates ~850-900 HV
* Hardness conversion HV≈5.1*HRC for 30-65 HRC approx
* Hollomon-Jaffe tempering parameter P = T(C+log t) with C≈20 for steels

Integration with existing repo
------------------------------
* Input can be pure electrodeposited Fe (C0≈0.02 wt%) or Fe-Ni-C composite with initial C
* Grain-size model from mechanical_properties may be used for prior austenite grain size → hardenability
* Mechanical case-core composite uses rule-of-mixtures: σ_composite ≈ f_case·σ_case + f_core·σ_core
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal, Tuple
import math
import numpy as np
from scipy.special import erfc
from scipy.optimize import brentq

# Constants
from .electrochemistry import R_GAS, RHO_FE

M_C = 12.011e-3
CP_FE_J_KG_K = 449.0  # specific heat around 300-900°C average (J/kg/K)

# Diffusion defaults (screening literature means)
# bcc α-Fe (ferrite) low-T
D0_FERRITE_M2_S = 6.2e-7   # m²/s
Q_FERRITE_KJ_MOL = 80.0    # kJ/mol

# fcc γ-Fe (austenite) high-T
D0_AUSTENITE_M2_S = 2.3e-5  # m²/s
Q_AUSTENITE_KJ_MOL = 148.0  # kJ/mol

# Transition
A3_TEMP_C = 912.0  # α→γ approx for pure Fe, lowers with C

# Hardness model constants
HV_BASE = 127.0
HV_PER_C_WT = 949.0
HV_SAT = 900.0  # cap for martensite ~ 64 HRC


@dataclass(frozen=True)
class CarburizationParams:
    """Carburizing operating conditions and material coefficients."""

    temperature_C: float = 900.0
    surface_carbon_wt_percent: float = 1.10   # carburizing potential (wt% at surface)
    initial_carbon_wt_percent: float = 0.02   # as-deposited
    sheet_thickness_um: float = 1000.0        # total thickness (µm), 1 mm typical electrodeposit
    D0_m2_s: Optional[float] = None           # if None, auto from phase
    Q_kJ_mol: Optional[float] = None
    phase: Literal["auto", "ferrite", "austenite"] = "auto"
    austenite_grain_size_um: float = 25.0     # prior austenite grain size (for hardenability)
    quench_rate_C_s: float = 200.0            # cooling rate for martensite formation (>50 C/s typical)
    tempering_temperature_C: Optional[float] = None
    tempering_time_hr: Optional[float] = None
    furnace_efficiency: float = 0.60          # thermal efficiency
    gas: str = "endo+enriched-CH4"           # free-form label

    def __post_init__(self):
        if self.temperature_C <= 0 or self.temperature_C > 1300:
            raise ValueError("temperature_C must be (0,1300]")
        if not 0 <= self.surface_carbon_wt_percent <= 6.67:
            raise ValueError("surface carbon must be 0-6.67 wt% (Fe3C limit)")
        if not 0 <= self.initial_carbon_wt_percent < self.surface_carbon_wt_percent:
            raise ValueError("initial C must be < surface C and >=0")
        if self.sheet_thickness_um <= 10:
            raise ValueError("thickness must be >10 µm")
        if self.furnace_efficiency <= 0 or self.furnace_efficiency > 1:
            raise ValueError("furnace_efficiency in (0,1]")


@dataclass
class CarburizationProfile:
    """Carbon and hardness profile at a given time."""

    x_um: np.ndarray           # depth from surface (µm), 0 = surface
    c_wt_percent: np.ndarray   # C wt% profile
    hv_predicted: np.ndarray   # HV predicted as-quenched (or tempered if applied)
    time_hr: float
    temperature_C: float


@dataclass
class CarburizationResult:
    """Time series + final profile output."""

    time_hr: np.ndarray
    effective_case_depth_035_um: np.ndarray  # depth where C >=0.35 wt%
    effective_case_depth_050_um: np.ndarray  # depth where C >=0.50 wt%
    carbon_uptake_g_m2: np.ndarray
    surface_hv: np.ndarray
    core_c_wt: np.ndarray
    profiles: list[CarburizationProfile] = field(default_factory=list)
    params: CarburizationParams = field(default_factory=CarburizationParams)  # type: ignore
    flags: list[list[str]] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "final_time_hr": float(self.time_hr[-1]) if len(self.time_hr) else 0.0,
            "final_case_depth_035_um": float(self.effective_case_depth_035_um[-1]) if len(self.time_hr) else 0.0,
            "final_case_depth_050_um": float(self.effective_case_depth_050_um[-1]) if len(self.time_hr) else 0.0,
            "final_carbon_uptake_g_m2": float(self.carbon_uptake_g_m2[-1]) if len(self.time_hr) else 0.0,
            "final_surface_hv": float(self.surface_hv[-1]) if len(self.time_hr) else 0.0,
            "final_core_c_wt": float(self.core_c_wt[-1]) if len(self.time_hr) else 0.0,
            "sheet_thickness_um": self.params.sheet_thickness_um,
            "temperature_C": self.params.temperature_C,
            "surface_carbon_wt": self.params.surface_carbon_wt_percent,
        }


def carbon_diffusivity_m2_s(
    temperature_C: float,
    phase: Literal["auto", "ferrite", "austenite"] = "auto",
    D0: Optional[float] = None,
    Q_kJ_mol: Optional[float] = None,
) -> Tuple[float, str]:
    """
    Return diffusivity D (m²/s) and phase used.

    If D0/Q supplied, they override. Otherwise auto selects ferrite (<A3) or austenite (>=A3).
    For carburization, A3 lowered by carbon: screening uses 800°C as austenite threshold
    (Ac1 ~727°C eutectoid, Ac3 ~850°C at low C). For pure Fe metallurgy use explicit phase.
    """
    if phase == "auto":
        # For general metallurgy: pure Fe α→γ at 912°C
        # For carburization (Fe-C sheet with C surface), austenite region starts ~727-850°C.
        # Use 800°C as pragmatic auto threshold for carburization screening;
        # caller can force "ferrite" or "austenite" to override.
        auto_thresh = 800.0
        actual_phase = "austenite" if temperature_C >= auto_thresh else "ferrite"
        # Also if temperature_C >= A3_TEMP_C always austenite for pure Fe
        if temperature_C >= A3_TEMP_C:
            actual_phase = "austenite"
    else:
        actual_phase = phase

    if D0 is not None and Q_kJ_mol is not None:
        d0 = D0
        q = Q_kJ_mol * 1000.0
    else:
        if actual_phase == "ferrite":
            d0 = D0_FERRITE_M2_S
            q = Q_FERRITE_KJ_MOL * 1000.0
        else:
            d0 = D0_AUSTENITE_M2_S
            q = Q_AUSTENITE_KJ_MOL * 1000.0

    T_K = temperature_C + 273.15
    D = d0 * math.exp(-q / (R_GAS * T_K))
    return float(D), actual_phase


def hardness_from_carbon_wt(c_wt_percent: np.ndarray | float, quench_rate_C_s: float = 200.0) -> np.ndarray | float:
    """
    As-quenched martensite hardness (HV) from C wt% (Maynier simplified).

    Screening: HV = 127+949*C for C≤0.6, then asymptotic to 900 HV.
    Quench-rate correction: if slow (<20 C/s), fraction of bainite/pearlite reduces hardness.
    """
    # Convert input to array for unified handling
    single = np.isscalar(c_wt_percent)
    c = np.atleast_1d(np.asarray(c_wt_percent, dtype=float))
    c = np.clip(c, 0.0, 6.67)

    # Base martensite hardness (as-quenched)
    hv = HV_BASE + HV_PER_C_WT * c
    # Saturation logistic
    # blend linear up to 0.7 wt% then saturate
    hv_sat_blend = HV_SAT * (1.0 - np.exp(-2.5 * c))
    hv = np.minimum(hv, hv_sat_blend)
    hv = np.minimum(hv, HV_SAT)
    hv = np.maximum(hv, 80.0)  # ferrite floor

    # Hardenability / quench-rate knockdown screening
    # Critical rate ~ 50 C/s for plain C steel to get full martensite
    if quench_rate_C_s < 30.0:
        # fraction martensite ≈ 1 - exp(-quench/30)
        f_mart = 1.0 - np.exp(-quench_rate_C_s / 20.0)
        # bainite hardness ≈ 300-500 HV depending on C
        hv_bainite = 200.0 + 200.0 * c
        hv = f_mart * hv + (1.0 - f_mart) * hv_bainite

    if single:
        return float(hv[0])
    return hv


def tempered_hardness(hv_as_quenched: np.ndarray | float,
                      temper_T_C: float,
                      temper_t_hr: float) -> np.ndarray | float:
    """
    Hollomon-Jaffe tempering correction: HV_tempered = HV_quenched * exp(-k * P^n)
    Simplified screening with parameter P = T*(20+log10(t)).
    """
    single = np.isscalar(hv_as_quenched)
    hv = np.atleast_1d(np.asarray(hv_as_quenched, dtype=float))
    T_K = temper_T_C + 273.15
    t_s = temper_t_hr * 3600.0
    # Hollomon-Jaffe P = T*(C + log10(t)) with C~19.5-20 for alloy steels
    C_HJ = 19.5
    P = T_K * (C_HJ + np.log10(max(t_s, 1.0)))  # K*(log)
    # Empirical softening: ΔHV ≈ 0.35*P/1000 for P in 8000-20000
    # Use exp decay: HV_t = HV_q * (1 - 0.8*(1 - exp(-(P-8000)/6000)))
    P_norm = np.clip((P - 8000.0) / 6000.0, 0.0, 5.0)
    softening_factor = 1.0 - 0.75 * (1.0 - np.exp(-P_norm))
    softening_factor = np.clip(softening_factor, 0.35, 1.0)
    hv_t = hv * softening_factor
    hv_t = np.maximum(hv_t, 150.0)  # tempered martensite floor
    if single:
        return float(hv_t[0])
    return hv_t


def carbon_profile_semi_infinite(
    x_m: np.ndarray,
    t_s: float,
    D_m2_s: float,
    Cs_wt: float,
    C0_wt: float,
) -> np.ndarray:
    """
    Semi-infinite solution with constant surface concentration.

    C(x,t) = Cs - (Cs-C0)*erf(x / 2√Dt)  = C0 + (Cs-C0)*erfc(x/2√Dt)
    """
    if t_s <= 0:
        return np.full_like(x_m, C0_wt, dtype=float)
    sqrtDt = math.sqrt(D_m2_s * t_s)
    if sqrtDt < 1e-12:
        # no diffusion
        c = np.full_like(x_m, C0_wt, dtype=float)
        c[x_m == 0] = Cs_wt
        return c
    arg = x_m / (2.0 * sqrtDt)
    # use erfc for numerical stability at large x
    c = C0_wt + (Cs_wt - C0_wt) * erfc(arg)
    return c


def carbon_profile_finite_slab(
    x_m: np.ndarray,
    t_s: float,
    D_m2_s: float,
    Cs_wt: float,
    C0_wt: float,
    thickness_m: float,
) -> np.ndarray:
    """
    Finite sheet with both sides carburized (symmetric). For thickness L,
    we superpose two semi-infinite solutions from each surface (first order)
    or use Fourier series for better accuracy at long times.

    Here implement Fourier series for slab -180 ≤? Actually for sheet of thickness L,
    with both surfaces at Cs, solution is Crank's:

    (C-C0)/(Cs-C0) = 1 - 4/π Σ_{n odd} (1/n) sin(nπx/L) exp(-D n² π² t / L²) ??? Wait.

    For sheet with surface at Cs, initial uniform C0, the solution for plane sheet is:
    (C - C0)/(Cs - C0) = 1 - Σ_{n=0}∞ 4*(-1)^n/((2n+1)π) cos((2n+1)π x / L) exp(-D(2n+1)² π² t / L²)
    where x measured from center? Let's implement for x from surface (0) to L/2 midplane, symmetric.

    Simpler: use superposition of semi-infinite from both sides for short times (Fo <0.2),
    and series for long times. Blend.

    For screening, we use semi-infinite superposition as conservative estimate, then
    for Fo >0.2 switch to series with 20 terms.
    """
    if t_s <= 0:
        return np.full_like(x_m, C0_wt, dtype=float)

    Fo = D_m2_s * t_s / (thickness_m ** 2)

    if Fo < 0.2:
        # superposition of two semi-infinite: C = C0 + (Cs-C0)[erfc(x/2√Dt) + erfc((L-x)/2√Dt)]
        # but second term for far surface contribution to point x from left surface: distance from right surface = L - x
        sqrtDt = math.sqrt(D_m2_s * t_s)
        arg1 = x_m / (2.0 * sqrtDt)
        arg2 = (thickness_m - x_m) / (2.0 * sqrtDt)
        c = C0_wt + (Cs_wt - C0_wt) * (erfc(arg1) + erfc(arg2))
        # cap at Cs
        c = np.minimum(c, Cs_wt)
        return c
    else:
        # Fourier series — compute up to N=50 odd terms
        # x measured from 0 at left surface, L thickness. Midplane at L/2.
        # Solution using dimensionless: use Crank eq 4.18 for slab with surface conc Cs
        # (Cs - C)/(Cs - C0) = Σ ... Let's implement symmetrical.
        # For plane sheet -L/2 to L/2 with surface at ±L/2 at Cs:
        # (Cs - C)/(Cs - C0) = 4/π Σ_{n odd?}?
        # Use x' = x - L/2 centered coordinate, easier use series:
        # C = Cs - (Cs-C0) * (4/π) Σ_{k=0}∞ [(-1)^k/(2k+1)] cos((2k+1)π x'/L) exp(-D(2k+1)²π²t/L²) ????
        # Let's implement centered.

        x_centered = x_m - thickness_m / 2.0  # -L/2 .. +L/2
        sum_series = np.zeros_like(x_m, dtype=float)
        N = 50
        for k in range(N):
            n = 2 * k + 1
            term = (4.0 / (math.pi * n)) * np.sin(n * math.pi / 2.0) * np.cos(n * math.pi * x_centered / thickness_m) * math.exp(-D_m2_s * (n * math.pi / thickness_m) ** 2 * t_s)
            # Actually sin(nπ/2) = (-1)^((n-1)/2) for odd n
            sum_series += term
        # sum_series = Σ ...
        # Then (Cs - C)/(Cs - C0) approximation?
        # For initial uniform C0 and surface Cs, the solution is:
        # (C - C0)/(Cs - C0) = 1 - (4/π) Σ_{k=0}∞ (1/(2k+1)) sin((2k+1)π/2) cos((2k+1)π x'/L) exp(...)
        # Our sum_series already is that Σ, so C = C0 + (Cs-C0)*(1 - sum_series)
        # Check limit at t=0, sum=1 → C=C0, good. At t→∞ sum→0 → C→Cs.

        # But our term calculation includes sin factor; to match, compute after.

        c = C0_wt + (Cs_wt - C0_wt) * (1.0 - sum_series)
        c = np.clip(c, C0_wt, Cs_wt)
        return c


class CarburizationModel:
    """
    Screening gaseous/plasma carburization model for electrodeposited iron.

    Example
    -------
    model = CarburizationModel(CarburizationParams(temperature_C=900, surface_carbon_wt_percent=1.1))
    result = model.simulate(duration_hr=4.0, dt_hr=0.2)
    profile = model.profile_at_time(t_hr=2.0)
    """

    def __init__(self, params: Optional[CarburizationParams] = None):
        self.params = params or CarburizationParams()
        self.D, self.phase_used = carbon_diffusivity_m2_s(
            self.params.temperature_C,
            phase=self.params.phase,
            D0=self.params.D0_m2_s,
            Q_kJ_mol=self.params.Q_kJ_mol,
        )

    def profile_at_time(
        self,
        t_hr: float,
        n_points: int = 200,
        method: Literal["finite", "semi"] = "finite",
    ) -> CarburizationProfile:
        """
        Compute carbon and hardness profile at time t_hr.
        """
        if t_hr < 0:
            raise ValueError("t_hr must be >=0")
        t_s = t_hr * 3600.0
        L_m = self.params.sheet_thickness_um * 1e-6
        # x from surface 0 to L (full thickness) but for case depth we only care 0 to L/2
        x_m = np.linspace(0.0, L_m, n_points)

        if method == "semi":
            c_wt = carbon_profile_semi_infinite(x_m, t_s, self.D, self.params.surface_carbon_wt_percent, self.params.initial_carbon_wt_percent)
        else:
            c_wt = carbon_profile_finite_slab(x_m, t_s, self.D, self.params.surface_carbon_wt_percent, self.params.initial_carbon_wt_percent, L_m)

        hv = hardness_from_carbon_wt(c_wt, self.params.quench_rate_C_s)
        if self.params.tempering_temperature_C is not None and self.params.tempering_time_hr is not None:
            hv = tempered_hardness(hv, self.params.tempering_temperature_C, self.params.tempering_time_hr)

        x_um = x_m * 1e6
        return CarburizationProfile(
            x_um=x_um,
            c_wt_percent=c_wt,
            hv_predicted=hv,
            time_hr=t_hr,
            temperature_C=self.params.temperature_C,
        )

    def _effective_case_depth(self, c_profile: np.ndarray, x_um: np.ndarray, threshold_wt: float) -> float:
        """Depth where C >= threshold, via interpolation. Returns µm, 0 if none."""
        # c_profile decreases from surface; find last index where c >= thresh going inward
        # Since x increases inward, we want first x where c < thresh, interpolate
        mask = c_profile >= threshold_wt
        if not np.any(mask):
            return 0.0
        # find transition index
        # find first False after True
        idx = np.where(mask)[0][-1]  # last true
        if idx == len(c_profile) - 1:
            # whole thickness above threshold (saturated)
            return float(x_um[-1] / 2.0)  # cap at half thickness (symmetric)
        # interpolate between idx and idx+1
        x1, x2 = x_um[idx], x_um[idx + 1]
        c1, c2 = c_profile[idx], c_profile[idx + 1]
        if c1 == c2:
            return float(x1)
        # linear interpolation for depth where c = thresh
        frac = (threshold_wt - c1) / (c2 - c1)  # negative because c decreasing
        # Actually c1 >= thresh > c2, so c2<c1, so frac positive if thresh between
        # but c2 < c1, so denominator negative, numerator negative => positive
        x_thresh = x1 + frac * (x2 - x1)
        return float(np.clip(x_thresh, 0.0, x_um[-1] / 2.0))

    def _carbon_uptake_g_m2(self, profile: CarburizationProfile) -> float:
        """Integrate extra carbon (g/m2) over half thickness (one side)."""
        # total C mass per area: integral of ρ_Fe * (C wt%) over thickness
        # For symmetric both sides, integrate from 0 to L/2 and double? But profile already includes both sides superposition.
        # For screening, integrate extra C (C - C0) * ρ over full thickness /2? Let's integrate over 0..L (full)
        x_m = profile.x_um * 1e-6
        c_extra_wt = profile.c_wt_percent - self.params.initial_carbon_wt_percent
        c_extra_wt = np.maximum(c_extra_wt, 0.0)
        # ρ * C_mass_frac integrated over x
        # mass per area (kg/m²) = ∫ ρ * wt_frac dx
        dx = np.diff(x_m)
        avg = (c_extra_wt[:-1] + c_extra_wt[1:]) / 2.0 / 100.0
        rho = RHO_FE  # approx constant
        mass_kg_m2 = np.sum(avg * rho * dx)
        # But this counts whole thickness, which includes both sides; for one-side reporting double counts?
        # Our x is full thickness with both surfaces enriched; the integral already total. So return total extra for one sheet (both sides).
        return float(mass_kg_m2 * 1000.0)  # g/m2

    def simulate(
        self,
        duration_hr: float = 4.0,
        dt_hr: float = 0.1,
        n_x: int = 250,
        save_profiles_every_hr: Optional[float] = 1.0,
    ) -> CarburizationResult:
        """
        Simulate carburization over time.

        Returns time series of case depths, uptake, surface HV, core C.
        """
        if duration_hr <= 0 or dt_hr <= 0:
            raise ValueError("duration and dt must be positive")
        t_grid = np.arange(0.0, duration_hr + dt_hr * 0.5, dt_hr)
        n_t = len(t_grid)

        case035 = np.zeros(n_t)
        case050 = np.zeros(n_t)
        uptake = np.zeros(n_t)
        surf_hv = np.zeros(n_t)
        core_c = np.zeros(n_t)
        flags: list[list[str]] = []

        profiles: list[CarburizationProfile] = []

        L_m = self.params.sheet_thickness_um * 1e-6
        x_full_m = np.linspace(0.0, L_m, n_x)
        x_full_um = x_full_m * 1e6

        for i, t_hr in enumerate(t_grid):
            t_s = t_hr * 3600.0
            c_wt = carbon_profile_finite_slab(
                x_full_m, t_s, self.D,
                self.params.surface_carbon_wt_percent,
                self.params.initial_carbon_wt_percent,
                L_m,
            )
            hv = hardness_from_carbon_wt(c_wt, self.params.quench_rate_C_s)
            if self.params.tempering_temperature_C and self.params.tempering_time_hr:
                hv = tempered_hardness(hv, self.params.tempering_temperature_C, self.params.tempering_time_hr)

            # Case depths (use profile from surface side)
            cd035 = self._effective_case_depth(c_wt, x_full_um, 0.35)
            cd050 = self._effective_case_depth(c_wt, x_full_um, 0.50)

            # uptake integral
            # quick trapz
            c_extra = np.maximum(c_wt - self.params.initial_carbon_wt_percent, 0.0) / 100.0
            # quick trapz with fallback
            trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
            if trapz is None:
                # manual
                mass_kg_m2 = np.sum((c_extra[:-1] + c_extra[1:]) * 0.5 * RHO_FE * np.diff(x_full_m))
            else:
                mass_kg_m2 = trapz(c_extra * RHO_FE, x_full_m)
            uptake[i] = mass_kg_m2 * 1000.0
            surf_hv[i] = hv[0]
            # core is midplane
            mid_idx = n_x // 2
            core_c[i] = c_wt[mid_idx]

            case035[i] = cd035
            case050[i] = cd050

            # flags
            flag_i: list[str] = []
            if cd035 > 0.45 * self.params.sheet_thickness_um / 2.0:
                flag_i.append("through_carburized")
            if c_wt[mid_idx] > 0.30:
                flag_i.append("core_high_carbon")
            if surf_hv[i] > 850:
                flag_i.append("very_high_surface_hardness")
            flags.append(flag_i)

            # Save profiles sparsely
            if save_profiles_every_hr is not None and save_profiles_every_hr > 0:
                if t_hr == 0 or (t_hr % save_profiles_every_hr) < dt_hr * 0.6 or i == n_t - 1:
                    profiles.append(CarburizationProfile(
                        x_um=x_full_um,
                        c_wt_percent=c_wt,
                        hv_predicted=hv,
                        time_hr=t_hr,
                        temperature_C=self.params.temperature_C,
                    ))

        return CarburizationResult(
            time_hr=t_grid,
            effective_case_depth_035_um=case035,
            effective_case_depth_050_um=case050,
            carbon_uptake_g_m2=uptake,
            surface_hv=surf_hv,
            core_c_wt=core_c,
            profiles=profiles,
            params=self.params,
            flags=flags,
        )

    def energy_estimate_kWh_per_kg(self, mass_kg: float = 1.0) -> Dict[str, float]:
        """
        Estimate thermal energy to heat Fe sheet to carburizing T.
        Does not include soak losses or quench — screening only.
        """
        dT = self.params.temperature_C - 25.0
        Q_J = mass_kg * CP_FE_J_KG_K * dT
        Q_kWh = Q_J / 3.6e6
        Q_with_eff = Q_kWh / max(self.params.furnace_efficiency, 0.1)
        # soak hold ~ furnace power ~ 0.5 kW per kg at temp? placeholder
        return {
            "heat_energy_kWh_kg": Q_kWh,
            "with_furnace_efficiency_kWh_kg": Q_with_eff,
            "dT_C": dT,
            "furnace_efficiency": self.params.furnace_efficiency,
        }

    def composite_strength_estimate(
        self,
        case_depth_um: float,
        core_yield_MPa: float = 300.0,
        sheet_thickness_um: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Simple rule-of-mixtures for case-core sandwich (both sides carburized).

        σ_comp ≈ f_case·σ_case + f_core·σ_core

        σ_case estimated from surface hardness (Tabor) or from C content.
        For screening, σ_case ≈ HV_case * 9.81 /3.2
        """

        L = sheet_thickness_um or self.params.sheet_thickness_um
        # case fraction: 2*case_depth / thickness (both sides)
        f_case = min(2.0 * case_depth_um / L, 1.0)
        f_core = 1.0 - f_case

        # case yield from surface HV (use last surface HV approx)
        # HV ~ 500-800 for case, so σ_y ≈ HV*9.81/3.2 ≈ 3*HV
        # Take representative HV at case depth? For now use surface hardness from high-C martensite
        # Assume case at 0.5%C → HV ~ 600, σ≈1840 MPa? Actually Tabor overestimates for martensite due to brittleness,
        # so use reduced factor 2.5 for martensite.
        # We'll use σ_case ≈ 1500 MPa for 0.5%C hardened case, saturating at 1800 MPa.

        # For screening, compute from C at case depth threshold:
        # At 0.35%C, HV≈460; 0.5%C≈600

        def sigma_case_from_c(c_wt: float) -> float:
            # Empirical: YS of as-quenched martensite ~ 2171*C% + ... from literature
            # Use σ_y ≈ 300 + 2000*C (MPa) for C <0.6
            return 300.0 + 2000.0 * min(c_wt, 0.8)

        sigma_case_035 = sigma_case_from_c(0.35)
        sigma_case_050 = sigma_case_from_c(0.50)

        sigma_comp_035 = f_case * sigma_case_035 + f_core * core_yield_MPa
        sigma_comp_050 = f_case * sigma_case_050 + f_core * core_yield_MPa

        return {
            "case_fraction": f_case,
            "core_fraction": f_core,
            "sigma_case_035_MPa": sigma_case_035,
            "sigma_case_050_MPa": sigma_case_050,
            "sigma_composite_035_MPa": sigma_comp_035,
            "sigma_composite_050_MPa": sigma_comp_050,
            "core_yield_MPa": core_yield_MPa,
        }


def estimate_carburizing_time_for_case_depth(
    target_case_depth_um: float,
    temperature_C: float = 900.0,
    surface_c_wt: float = 1.1,
    threshold_c_wt: float = 0.35,
    initial_c_wt: float = 0.02,
    D_m2_s: Optional[float] = None,
) -> float:
    """
    Inverse estimate: time (hr) needed to reach case depth defined at C >= threshold
    for semi-infinite solid, using erfc inversion.

    (C_thresh - C0)/(Cs - C0) = erfc(x / 2√Dt) → solve for t
    """

    if target_case_depth_um <= 0:
        raise ValueError("target depth must be positive")
    if D_m2_s is None:
        D_m2_s, _ = carbon_diffusivity_m2_s(temperature_C)

    x_m = target_case_depth_um * 1e-6
    ratio = (threshold_c_wt - initial_c_wt) / (surface_c_wt - initial_c_wt)
    if not 0 < ratio < 1:
        raise ValueError("threshold must be between initial and surface C")

    # erfc(z) = ratio → z = erfc⁻¹(ratio)
    # Inverse: we need z where erfc(z)=ratio
    # Use numerical inversion via brentq

    def f(z):
        return erfc(z) - ratio

    # z bracket: erfc(0)=1, erfc(3)≈0.000022, so between 0 and 3 for typical ratios 0.01-0.9
    z_root = brentq(f, 0.0, 3.5, xtol=1e-8)
    # t = x² / (4 D z²)
    t_s = x_m ** 2 / (4.0 * D_m2_s * z_root ** 2)
    return t_s / 3600.0  # hr
