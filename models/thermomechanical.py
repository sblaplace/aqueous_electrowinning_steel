"""
Thermomechanical processing of electrodeposited iron/steel foil.

Closes the gap between the as-deposited foil and a structural-grade steel
sheet.  A deposit straight off the drum/strip is fine-grained, hydrogen
affected and comparatively brittle; structural sheet does not come off the
electrode, it is **cold rolled** and **recrystallization annealed**.  This
module models that step:

* cold-rolling pass schedule -> total true strain
* Johnson-Mehl-Avrami-Kolmogorov (JMAK) recrystallization kinetics
  X(t,T,ε) = 1 - exp(-(k t)^n), k(T,ε) = k0·εᵐ·exp(-Qᵣₓ/RT)
* recovery of the deformed subgrain structure
* normal grain growth D²-Dᵣₓ² = K_gg·t·exp(-Q_gg/RT)
* annealed mechanical properties by reusing the repository's Hall-Petch /
  solid-solution / dispersion machinery (`mechanical_properties.py`), and the
  strength / grade routing
* annealing energy for the techno-economic and site (dark-mill) balances

This is a **screening** model.  Activation energies, the Avrami exponent and
the recrystallized-grain-size coefficients are literature-order estimates that
must be calibrated with real rolling trials, EBSD orientation mapping and
Vickers/tensile traverses.  See `docs/figures/README.md` notes.

References (screening)
----------------------
* JMAK recrystallization: Humphreys & Hatherly, "Recrystallization and Related
  Annealing Phenomena" (activation energy Q≈180-240 kJ/mol for low-carbon Fe)
* Normal grain growth: D^n - D₀^n = K t with n≈2, Q_gg ≈ 240 kJ/mol (grain
  boundary self-diffusion regime)
* Recrystallized grain size decreases with strain (higher stored energy) and
  increases weakly with starting grain size / anneal temperature
* Electroformed foil / strip product route: cold-roll + recrystallize to
  recover ductility lost to the fine as-deposited grain (Hall-Petch)

Integration with existing repo
------------------------------
* Deposit grain size input feeds `estimate_grain_size_um` / `MechanicalPropertiesModel`
  (the same grain -> strength machinery used for the as-deposited screening)
* As-deposited vs annealed property contrast quantifies the benefit of the
  thermomechanical step
* Annealing energy output feeds `technoeconomic.py` and `dark_mill.py`
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Constants
from .electrochemistry import R_GAS
from .mechanical_properties import (
    MechanicalPropertiesModel,
    MechanicalPropertiesParams,
    estimate_grain_size_um,
    GrainSizeParams,
)

# ── Thermomechanical screening constants ────────────────────────────────
# Recrystallization (JMAK) kinetics
Q_RX_KJ_MOL = 210.0            # activation energy for recrystallization (kJ/mol)
K0_RX_1_S = 1.0e9              # pre-exponential (s^-1) so t50 ≈ 5 min at 700 °C, ε=0.5
AVRAMI_N = 2.0                 # Avrami exponent (site-saturated, growth-controlled sheet)
STRAIN_EXP_M = 1.0             # rate constant k ∝ ε^m (higher strain -> faster RX)

# Normal grain growth
Q_GG_KJ_MOL = 240.0            # grain-growth activation energy (kJ/mol)
K_GG_UM2_S = 1.0e10            # such that ~12->~30 µm after 1 h at 900 °C

# Recrystallized grain size: D_rx = D_rx_ref·(D0/D0_ref)^a·(ε_ref/ε)^b·(T/T_ref)^c
D_RX_REF_UM = 12.0             # reference recrystallized grain size (µm)
D0_RX_REF_UM = 1.0             # reference as-deposited grain size (µm)
EPS_RX_REF = 0.5               # reference true strain (ln(1/(1-r)))
T_RX_REF_K = 973.0             # reference anneal temperature (700 °C)
D0_EXP = 0.35                  # starting-grain-size exponent
EPS_EXP = 0.55                 # strain exponent (larger strain -> finer RX grains)
T_RX_EXP = 0.60                # mild temperature dependence of D_rx
SUB_GRAIN_FRAC = 0.40          # recovered subgrain size as fraction of D_rx

# Anneal energy
CP_FE_J_KG_K = 449.0           # specific heat of Fe (~300-900 °C mean)
FURNACE_EFFICIENCY = 0.60      # thermal efficiency of the anneal furnace

D_MIN_UM = 0.1
D_MAX_UM = 120.0


@dataclass(frozen=True)
class RollingSchedule:
    """Cold-rolling pass schedule.

    Either give ``total_reduction`` and ``n_passes`` (equal reduction per pass)
    or an explicit per-pass ``reductions`` list.  ``total_reduction`` is the
    fractional thickness loss (0.5 = 50% thinner).
    """

    total_reduction: float = 0.50
    n_passes: int = 2
    reductions: Optional[list[float]] = None  # explicit per-pass reductions

    def __post_init__(self):
        if self.reductions is None:
            if not 0.0 < self.total_reduction < 0.95:
                raise ValueError("total_reduction must be in (0, 0.95)")
            if self.n_passes < 1:
                raise ValueError("n_passes must be >= 1")
        else:
            red = self.reductions
            if len(red) < 1:
                raise ValueError("reductions must be non-empty")
            if not all(0.0 < r < 0.95 for r in red):
                raise ValueError("each pass reduction must be in (0, 0.95)")

    @property
    def total_true_strain(self) -> float:
        """Cumulative true strain ε = Σ ln(1/(1-r_i)) = ln(h0/hf)."""
        if self.reductions is not None:
            return sum(math.log(1.0 / (1.0 - r)) for r in self.reductions)
        hf_over_h0 = 1.0 - self.total_reduction
        return math.log(1.0 / hf_over_h0)

    @property
    def per_pass_reductions(self) -> list[float]:
        if self.reductions is not None:
            return list(self.reductions)
        per_pass = 1.0 - (1.0 - self.total_reduction) ** (1.0 / self.n_passes)
        return [per_pass] * self.n_passes


@dataclass(frozen=True)
class ThermomechanicalParams:
    """Thermomechanical operating conditions and material coefficients.

    ``deposit_grain_size_um`` is the as-deposited mean grain size (from
    ``mechanical_properties.estimate_grain_size_um`` or a direct input).  Alloy
    and carbon composition are carried through to the annealed strength.
    """

    deposit_grain_size_um: float = 1.0     # as-deposited mean grain (µm)
    rolling: RollingSchedule = field(default_factory=RollingSchedule)
    anneal_temperature_C: float = 700.0
    anneal_time_min: float = 60.0
    ni_wt_percent: float = 0.0
    mn_wt_percent: float = 0.0
    cr_wt_percent: float = 0.0
    carbon_wt_percent: float = 0.0
    current_efficiency_percent: float = 95.0
    # screening kinetic coefficients (override only to calibrate)
    q_rx_kJ_mol: float = Q_RX_KJ_MOL
    k0_rx_1_s: float = K0_RX_1_S
    avrami_n: float = AVRAMI_N
    strain_exp_m: float = STRAIN_EXP_M
    q_gg_kJ_mol: float = Q_GG_KJ_MOL
    k_gg_um2_s: float = K_GG_UM2_S
    d_rx_ref_um: float = D_RX_REF_UM
    d0_rx_ref_um: float = D0_RX_REF_UM
    eps_rx_ref: float = EPS_RX_REF
    t_rx_ref_K: float = T_RX_REF_K
    d0_exp: float = D0_EXP
    eps_exp: float = EPS_EXP
    t_rx_exp: float = T_RX_EXP
    sub_grain_frac: float = SUB_GRAIN_FRAC
    furnace_efficiency: float = FURNACE_EFFICIENCY

    def __post_init__(self):
        if not 0.05 <= self.deposit_grain_size_um <= 100:
            raise ValueError("deposit_grain_size_um out of physical range")
        if not 400.0 <= self.anneal_temperature_C <= 1000.0:
            raise ValueError("anneal_temperature_C must be in [400, 1000] for Fe")
        if self.anneal_time_min <= 0:
            raise ValueError("anneal_time_min must be positive")
        if not 0.0 < self.furnace_efficiency <= 1.0:
            raise ValueError("furnace_efficiency in (0,1]")


def jmak_rate_constant_1_s(
    temperature_C: float,
    true_strain: float,
    params: ThermomechanicalParams,
) -> float:
    """JMAK rate constant k = k0·ε^m·exp(-Q_rx/RT) [1/s]."""
    T_K = temperature_C + 273.15
    strain = max(true_strain, 1e-4)
    return params.k0_rx_1_s * (strain ** params.strain_exp_m) * math.exp(
        -params.q_rx_kJ_mol * 1e3 / (R_GAS * T_K)
    )


def jmak_fraction_recrystallized(
    time_s: np.ndarray,
    temperature_C: float,
    true_strain: float,
    params: ThermomechanicalParams,
) -> np.ndarray:
    """JMAK fraction recrystallized X = 1 - exp(-(k t)^n) over time grid."""
    k = jmak_rate_constant_1_s(temperature_C, true_strain, params)
    t = np.asarray(time_s, dtype=float)
    x = 1.0 - np.exp(-(k * t) ** params.avrami_n)
    return np.clip(x, 0.0, 1.0)


def time_for_fraction(
    x_target: float,
    temperature_C: float,
    true_strain: float,
    params: ThermomechanicalParams,
) -> float:
    """Time (s) to reach fraction ``x_target`` recrystallized."""
    if not 0.0 < x_target < 1.0:
        raise ValueError("x_target must be in (0,1)")
    k = jmak_rate_constant_1_s(temperature_C, true_strain, params)
    if k <= 0:
        return float("inf")
    return (-math.log(1.0 - x_target)) ** (1.0 / params.avrami_n) / k


def recrystallized_grain_size_um(
    deposit_grain_um: float,
    true_strain: float,
    temperature_C: float,
    params: ThermomechanicalParams,
) -> float:
    """Fully-recrystallized grain size D_rx (µm) from cold work.

    Higher strain (stored energy) -> finer RX grains; coarser starting grains
    and higher anneal temperature coarsen the result.
    """
    T_K = temperature_C + 273.15
    d0 = max(deposit_grain_um, 1e-3)
    strain = max(true_strain, 1e-4)
    d_rx = (
        params.d_rx_ref_um
        * (d0 / params.d0_rx_ref_um) ** params.d0_exp
        * (params.eps_rx_ref / strain) ** params.eps_exp
        * (T_K / params.t_rx_ref_K) ** params.t_rx_exp
    )
    return float(np.clip(d_rx, D_MIN_UM, D_MAX_UM))


def grain_growth_um(
    d_rx_um: float,
    time_s: float,
    temperature_C: float,
    params: ThermomechanicalParams,
) -> float:
    """Normal grain growth D² = D_rx² + K_gg·t·exp(-Q_gg/RT)."""
    T_K = temperature_C + 273.15
    growth = (
        params.k_gg_um2_s
        * time_s
        * math.exp(-params.q_gg_kJ_mol * 1e3 / (R_GAS * T_K))
    )
    d = math.sqrt(max(d_rx_um ** 2 + growth, d_rx_um ** 2))
    return float(np.clip(d, d_rx_um, D_MAX_UM))


@dataclass
class ThermomechanicalResult:
    """Output of a thermomechanical processing simulation."""

    # inputs echoed
    deposit_grain_um: float
    total_reduction: float
    true_strain: float
    n_passes: int
    anneal_temperature_C: float
    anneal_time_min: float

    # microstructural predictions
    recrystallized_grain_um: float
    final_grain_um: float
    fraction_recrystallized: float
    t_full_rx_min: float          # time to 99% recrystallized (min)

    # mechanical (annealed, via mechanical_properties machinery)
    annealed_yield_MPa: float
    annealed_uts_MPa: float
    annealed_hv: float
    annealed_elongation_pct: float
    annealed_grade: str
    # as-deposited contrast (same machinery, no anneal)
    deposit_yield_MPa: float
    deposit_uts_MPa: float
    deposit_hv: float
    deposit_elongation_pct: float
    deposit_grade: str

    annealing_energy_kWh_per_kg: float

    # time series
    time_s: np.ndarray
    fraction_recrystallized_series: np.ndarray
    grain_size_series_um: np.ndarray

    flags: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "deposit_grain_um": round(self.deposit_grain_um, 2),
            "total_reduction_frac": round(self.total_reduction, 3),
            "true_strain": round(self.true_strain, 3),
            "n_passes": self.n_passes,
            "anneal_temperature_C": self.anneal_temperature_C,
            "anneal_time_min": self.anneal_time_min,
            "recrystallized_grain_um": round(self.recrystallized_grain_um, 2),
            "final_grain_um": round(self.final_grain_um, 2),
            "fraction_recrystallized": round(self.fraction_recrystallized, 3),
            "time_to_full_rx_min": round(self.t_full_rx_min, 2),
            "annealed_yield_MPa": round(self.annealed_yield_MPa, 1),
            "annealed_uts_MPa": round(self.annealed_uts_MPa, 1),
            "annealed_hv": round(self.annealed_hv, 1),
            "annealed_elongation_pct": round(self.annealed_elongation_pct, 1),
            "annealed_grade": self.annealed_grade,
            "deposit_yield_MPa": round(self.deposit_yield_MPa, 1),
            "deposit_uts_MPa": round(self.deposit_uts_MPa, 1),
            "deposit_elongation_pct": round(self.deposit_elongation_pct, 1),
            "annealing_energy_kWh_per_kg": round(self.annealing_energy_kWh_per_kg, 3),
            "flags": self.flags,
        }


class ThermomechanicalModel:
    """Cold-roll + recrystallization-anneal screening model for Fe foil.

    Example
    -------
    >>> model = ThermomechanicalModel()
    >>> res = model.predict()
    >>> print(res.annealed_grade, res.final_grain_um)
    """

    def __init__(
        self,
        params: Optional[ThermomechanicalParams] = None,
        mech_params: Optional[MechanicalPropertiesParams] = None,
        grain_params: Optional[GrainSizeParams] = None,
    ):
        self.params = params or ThermomechanicalParams()
        self.mech_params = mech_params or MechanicalPropertiesParams()
        self.grain_params = grain_params or GrainSizeParams()
        self._mech_model = MechanicalPropertiesModel(
            grain_params=self.grain_params, mech_params=self.mech_params
        )

    # ------------------------------------------------------------------
    def _mechanical(
        self, grain_um: float
    ) -> tuple[float, float, float, float, str]:
        r = self._mech_model.predict(
            grain_size_override_um=grain_um,
            ni_wt_percent=self.params.ni_wt_percent,
            mn_wt_percent=self.params.mn_wt_percent,
            cr_wt_percent=self.params.cr_wt_percent,
            carbon_wt_percent=self.params.carbon_wt_percent,
            current_efficiency_percent=self.params.current_efficiency_percent,
        )
        return (
            r.sigma_y_MPa,
            r.uts_MPa,
            r.vickers_hv,
            r.elongation_pct,
            r.grade_estimate,
        )

    def anneal_energy_kWh_per_kg(self) -> float:
        """Heating energy to bring 1 kg of deposit to the anneal temperature."""
        T_ambient_C = 25.0
        dT = max(self.params.anneal_temperature_C - T_ambient_C, 0.0)
        heat_J_kg = CP_FE_J_KG_K * dT
        heat_J_kg /= self.params.furnace_efficiency
        return heat_J_kg / 3.6e6  # J/kg -> kWh/kg

    # ------------------------------------------------------------------
    def predict(self) -> ThermomechanicalResult:
        p = self.params
        roll = p.rolling
        eps = roll.total_true_strain
        T_K = p.anneal_temperature_C + 273.15

        d_rx = recrystallized_grain_size_um(
            p.deposit_grain_size_um, eps, p.anneal_temperature_C, p
        )
        t_anneal_s = p.anneal_time_min * 60.0
        d_final = grain_growth_um(d_rx, t_anneal_s, p.anneal_temperature_C, p)

        x_final = float(
            jmak_fraction_recrystallized(
                np.array([t_anneal_s]), p.anneal_temperature_C, eps, p
            )[0]
        )
        t_full = time_for_fraction(
            0.99, p.anneal_temperature_C, eps, p
        )

        # As-deposited vs annealed contrast (same strength machinery)
        deposit = self._mechanical(p.deposit_grain_size_um)
        annealed = self._mechanical(d_final)

        energy = self.anneal_energy_kWh_per_kg()

        flags: list[str] = []
        if x_final < 0.99:
            flags.append("incomplete_recrystallization")
        if p.anneal_temperature_C < 550 and t_full > p.anneal_time_min:
            flags.append("anneal_too_cold_short")
        if annealed[0] > 700:
            flags.append("annealed_still_high_strength")

        # Time series
        n = max(int(math.ceil(t_anneal_s / 30.0)), 2)  # ~30 s resolution
        t_grid = np.linspace(0.0, t_anneal_s, n)
        x_series = jmak_fraction_recrystallized(
            t_grid, p.anneal_temperature_C, eps, p
        )
        d_sub = d_rx * p.sub_grain_frac
        d_eff = d_sub + (d_rx - d_sub) * x_series
        d_series = np.sqrt(
            d_eff ** 2
            + p.k_gg_um2_s * t_grid * math.exp(-p.q_gg_kJ_mol * 1e3 / (R_GAS * T_K))
        )
        d_series = np.clip(d_series, D_MIN_UM, D_MAX_UM)

        return ThermomechanicalResult(
            deposit_grain_um=p.deposit_grain_size_um,
            total_reduction=roll.total_reduction,
            true_strain=eps,
            n_passes=len(roll.per_pass_reductions),
            anneal_temperature_C=p.anneal_temperature_C,
            anneal_time_min=p.anneal_time_min,
            recrystallized_grain_um=d_rx,
            final_grain_um=d_final,
            fraction_recrystallized=x_final,
            t_full_rx_min=t_full / 60.0,
            annealed_yield_MPa=annealed[0],
            annealed_uts_MPa=annealed[1],
            annealed_hv=annealed[2],
            annealed_elongation_pct=annealed[3],
            annealed_grade=annealed[4],
            deposit_yield_MPa=deposit[0],
            deposit_uts_MPa=deposit[1],
            deposit_hv=deposit[2],
            deposit_elongation_pct=deposit[3],
            deposit_grade=deposit[4],
            annealing_energy_kWh_per_kg=energy,
            time_s=t_grid,
            fraction_recrystallized_series=x_series,
            grain_size_series_um=d_series,
            flags=flags,
        )

    # ------------------------------------------------------------------
    def sweep_temperature(self, temps_C=None, time_min: float = 60.0) -> dict:
        """Final grain size / fraction RX / yield vs anneal temperature."""
        if temps_C is None:
            temps_C = np.arange(500.0, 951.0, 50.0)
        out = {"T_C": [], "D_final_um": [], "frac_rx": [], "yield_MPa": [],
               "energy_kWh_kg": []}
        for T in temps_C:
            params = replace_anneal(self.params, T, time_min)
            model = ThermomechanicalModel(params, self.mech_params, self.grain_params)
            res = model.predict()
            out["T_C"].append(float(T))
            out["D_final_um"].append(res.final_grain_um)
            out["frac_rx"].append(res.fraction_recrystallized)
            out["yield_MPa"].append(res.annealed_yield_MPa)
            out["energy_kWh_kg"].append(res.annealing_energy_kWh_per_kg)
        for k, v in out.items():
            out[k] = np.asarray(v)
        return out

    def sweep_reduction(self, reductions=None, time_min: float = 60.0) -> dict:
        """Recrystallized grain size & final yield vs cold reduction."""
        if reductions is None:
            reductions = np.arange(0.10, 0.91, 0.10)
        out = {"reduction": [], "D_rx_um": [], "D_final_um": [],
               "frac_rx": [], "yield_MPa": []}
        for r in reductions:
            roll = RollingSchedule(total_reduction=float(r), n_passes=2)
            params = replace_rolling(self.params, roll)
            model = ThermomechanicalModel(params, self.mech_params, self.grain_params)
            res = model.predict()
            out["reduction"].append(float(r))
            out["D_rx_um"].append(res.recrystallized_grain_um)
            out["D_final_um"].append(res.final_grain_um)
            out["frac_rx"].append(res.fraction_recrystallized)
            out["yield_MPa"].append(res.annealed_yield_MPa)
        for k, v in out.items():
            out[k] = np.asarray(v)
        return out

    def sweep_time(self, times_min=None, temperature_C: float = 700.0) -> dict:
        """Final grain size & yield vs anneal time at a fixed temperature."""
        if times_min is None:
            times_min = np.array([1.0, 5, 15, 30, 60, 120, 240])
        out = {"time_min": [], "D_final_um": [], "frac_rx": [], "yield_MPa": []}
        for t in times_min:
            params = replace_anneal(self.params, temperature_C, t)
            model = ThermomechanicalModel(params, self.mech_params, self.grain_params)
            res = model.predict()
            out["time_min"].append(float(t))
            out["D_final_um"].append(res.final_grain_um)
            out["frac_rx"].append(res.fraction_recrystallized)
            out["yield_MPa"].append(res.annealed_yield_MPa)
        for k, v in out.items():
            out[k] = np.asarray(v)
        return out


def replace_anneal(
    params: ThermomechanicalParams, temperature_C: float, time_min: float
) -> ThermomechanicalParams:
    """Return a copy of ``params`` with new anneal conditions."""
    return ThermomechanicalParams(
        **{**params.__dict__, "anneal_temperature_C": temperature_C,
           "anneal_time_min": time_min}
    )


def replace_rolling(
    params: ThermomechanicalParams, rolling: RollingSchedule
) -> ThermomechanicalParams:
    """Return a copy of ``params`` with a new rolling schedule."""
    return ThermomechanicalParams(
        **{**params.__dict__, "rolling": rolling}
    )


def deposit_grain_size_um(
    j_avg_mA_cm2: float = 100.0,
    waveform: str = "dc",
    temperature_C: float = 60.0,
    duty_cycle: float = 1.0,
) -> float:
    """Convenience wrapper over mechanical_properties grain-size estimator."""
    return estimate_grain_size_um(
        j_avg_mA_cm2=j_avg_mA_cm2, waveform=waveform,  # type: ignore[arg-type]
        temperature_C=temperature_C, duty_cycle=duty_cycle,
    )
