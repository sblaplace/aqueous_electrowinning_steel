"""Feedstock purification model — Cu/Ni/Zn removal before electrowinning.

Nobler metals (Cu > Ni > Zn > Fe by standard potential) co-deposit
preferentially at the cathode.  Cu above ~0.1 wt% causes hot shortness
in the deposited steel.  This screening model couples four unit
operations that progressively strip contaminants from the recycle loop:

1. **Cementation** — Cu/Ni/Zn displace metallic Fe on powder bed.
2. **Hydrolysis** — Fe³⁺ precipitates as Fe(OH)₃ at pH 3–4 while Fe²⁺
   remains soluble; carries co-precipitated contaminants.
3. **Selective electrowinning** — contaminants plate at a controlled
   cathode potential in a side-stream cell.
4. **Ion exchange** — chelating resin polishes residual Cu/Ni.

Each stage is a first-order / equilibrium screen — calibrate kinetics
and selectivity with bench experiments before design use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ── Physical constants (from electrochemistry.py) ─────────────────
from .electrochemistry import FARADAY, R_GAS, M_FE_G as M_FE

# Molar masses (g/mol)
M_CU = 63.546
M_NI = 58.693
M_ZN = 65.38

# Standard reduction potentials (V vs SHE) at 25 °C
E0_CU = +0.342
E0_NI = -0.257
E0_ZN = -0.762
E0_FE = -0.440


# ── Parameter classes ─────────────────────────────────────────────────

@dataclass(frozen=True)
class CementationParams:
    """Parameters for cementation of Cu/Ni/Zn on iron powder."""

    cu_rate_const_per_hr: float = 2.5        # first-order rate constant for Cu
    ni_rate_const_per_hr: float = 0.40       # first-order rate constant for Ni
    zn_rate_const_per_hr: float = 0.60       # first-order rate constant for Zn
    activation_energy_kJ_mol: float = 35.0   # Arrhenius Ea (same for all three)
    reference_temperature_C: float = 25.0
    iron_powder_dose_g_per_L: float = 10.0   # gram Fe powder per litre of solution
    iron_powder_cost_per_kg: float = 0.80    # USD/kg
    ph_optimum: float = 2.0                  # best pH for cementation rate
    ph_sensitivity: float = 0.5              # rate penalty per pH unit away from optimum

    def __post_init__(self) -> None:
        for name in ("cu_rate_const_per_hr", "ni_rate_const_per_hr", "zn_rate_const_per_hr"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class HydrolysisParams:
    """Parameters for Fe(OH)₃ precipitation / co-removal stage."""

    target_pH: float = 3.5
    fe3_solubility_at_target_M: float = 1.0e-5   # mol/L at target pH
    precipitation_rate_per_hr: float = 5.0        # first-order approach to equilibrium
    co_precipitation_fraction_cu: float = 0.30    # fraction of Cu co-removed
    co_precipitation_fraction_ni: float = 0.20
    co_precipitation_fraction_zn: float = 0.25
    naoh_cost_per_kg: float = 0.40                # USD/kg NaOH
    naoh_mol_per_pH_unit_per_L: float = 0.01      # mol NaOH / (L · pH unit)

    def __post_init__(self) -> None:
        if not 1.0 <= self.target_pH <= 7.0:
            raise ValueError("target_pH must lie between 1 and 7")


@dataclass(frozen=True)
class ElectrowinningParams:
    """Parameters for a selective side-stream electrowinning cell."""

    cathode_potential_V: float = -0.10    # vs SHE — between Cu and Ni deposition
    cu_exchange_current_A_m2: float = 1.0e-2
    ni_exchange_current_A_m2: float = 1.0e-4
    tafel_slope_V: float = 0.060         # ~60 mV/decade for metal deposition
    cathode_area_m2: float = 0.5
    flow_rate_L_hr: float = 100.0
    electricity_cost_per_kWh: float = 0.04
    cell_voltage_V: float = 2.0          # total cell voltage (anode + cathode + IR)

    def __post_init__(self) -> None:
        if self.cathode_area_m2 <= 0:
            raise ValueError("cathode_area_m2 must be positive")


@dataclass(frozen=True)
class IonExchangeParams:
    """Parameters for chelating resin polishing step."""

    cu_capacity_meq_per_mL: float = 1.2    # resin capacity for Cu
    ni_capacity_meq_per_mL: float = 0.8
    selectivity_cu_over_fe: float = 1000.0
    selectivity_ni_over_fe: float = 500.0
    resin_volume_L: float = 10.0
    resin_cost_per_L: float = 80.0
    resin_lifetime_cycles: float = 2000.0
    regeneration_acid_cost_per_cycle: float = 2.0  # USD per cycle


@dataclass(frozen=True)
class PurificationFeedstock:
    """Incoming electrolyte composition before purification."""

    cu_M: float = 5.0e-4     # ~0.003 wt% in 1M Fe solution
    ni_M: float = 3.0e-4
    zn_M: float = 2.0e-4
    fe2_M: float = 1.0       # Fe²⁺ background
    fe3_M: float = 0.05      # Fe³⁺ (minor oxidised fraction)
    pH: float = 2.0
    temperature_C: float = 50.0
    volume_L: float = 1000.0

    def __post_init__(self) -> None:
        for name in ("cu_M", "ni_M", "zn_M", "fe2_M", "fe3_M"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


# ── Result classes ────────────────────────────────────────────────────

@dataclass
class CementationResult:
    """Output from cementation stage."""

    time_hr: np.ndarray
    cu_M: np.ndarray
    ni_M: np.ndarray
    zn_M: np.ndarray
    fe_consumed_M: float
    fe_released_M: float  # Fe²⁺ released by displacement
    powder_consumed_kg: float
    cost_USD: float

    def removal_fractions(self) -> dict[str, float]:
        return {
            "cu": float(1.0 - self.cu_M[-1] / self.cu_M[0]) if self.cu_M[0] > 0 else 0.0,
            "ni": float(1.0 - self.ni_M[-1] / self.ni_M[0]) if self.ni_M[0] > 0 else 0.0,
            "zn": float(1.0 - self.zn_M[-1] / self.zn_M[0]) if self.zn_M[0] > 0 else 0.0,
        }


@dataclass
class HydrolysisResult:
    """Output from hydrolysis / precipitation stage."""

    fe3_remaining_M: float
    cu_removed_M: float
    ni_removed_M: float
    zn_removed_M: float
    fe2_unchanged_M: float
    precipitate_mass_kg: float  # Fe(OH)₃ mass
    naoh_consumed_kg: float
    cost_USD: float


@dataclass
class ElectrowinningResult:
    """Output from selective electrowinning stage."""

    cu_plated_M: float
    ni_plated_M: float
    cu_remaining_M: float
    ni_remaining_M: float
    energy_kWh: float
    cost_USD: float


@dataclass
class IonExchangeResult:
    """Output from ion exchange polishing stage."""

    cu_removed_M: float
    ni_removed_M: float
    cu_final_M: float
    ni_final_M: float
    resin_cost_per_cycle: float
    cost_USD: float


@dataclass
class PurificationResult:
    """Aggregated output from all four purification stages."""

    cementation: CementationResult
    hydrolysis: HydrolysisResult
    electrowinning: ElectrowinningResult
    ion_exchange: IonExchangeResult
    cu_initial_M: float
    cu_final_M: float
    cu_removal_fraction: float
    cu_final_wt_pct: float        # wt% relative to Fe²⁺
    ni_initial_M: float
    ni_final_M: float
    zn_initial_M: float
    zn_final_M: float
    total_cost_per_t_fe: float    # USD per tonne of Fe in feedstock
    stage_costs: dict[str, float]
    impurity_buildup_rate_M_per_hr: float  # in recycle loop

    def cu_meets_spec(self, limit_wt_pct: float = 0.01) -> bool:
        return bool(self.cu_final_wt_pct < limit_wt_pct)

    def summary(self) -> dict:
        return {
            "cu_initial_M": float(self.cu_initial_M),
            "cu_final_M": float(self.cu_final_M),
            "cu_removal_fraction": float(self.cu_removal_fraction),
            "cu_final_wt_pct": float(self.cu_final_wt_pct),
            "cu_meets_spec_0_01wt_pct": self.cu_meets_spec(),
            "ni_final_M": float(self.ni_final_M),
            "zn_final_M": float(self.zn_final_M),
            "total_cost_per_t_fe": float(self.total_cost_per_t_fe),
            "stage_costs": {k: float(v) for k, v in self.stage_costs.items()},
            "impurity_buildup_rate_M_per_hr": float(self.impurity_buildup_rate_M_per_hr),
        }


# ── Model classes ─────────────────────────────────────────────────────

class CementationModel:
    """First-order cementation of Cu²⁺, Ni²⁺, Zn²⁺ on Fe powder.

    Reaction: M²⁺ + Fe⁰ → M⁰ + Fe²⁺  (selective displacement).

    Rate: d[M]/dt = -k_eff · [M] · θ_Fe
    where θ_Fe is the fractional surface coverage of active Fe sites,
    approximated by the remaining Fe powder dose.
    """

    def __init__(
        self,
        params: Optional[CementationParams] = None,
        feedstock: Optional[PurificationFeedstock] = None,
    ) -> None:
        self.params = params or CementationParams()
        self.feedstock = feedstock or PurificationFeedstock()

    def _effective_rate(self, k_ref: float, temperature_C: float, pH: float) -> float:
        """Arrhenius + pH-adjusted rate constant."""
        p = self.params
        dT = temperature_C - p.reference_temperature_C
        arrhenius = math.exp(p.activation_energy_kJ_mol * 1000.0 / R_GAS
                             * (1.0 / (p.reference_temperature_C + 273.15)
                                - 1.0 / (temperature_C + 273.15)))
        pH_factor = math.exp(-p.ph_sensitivity * abs(pH - p.ph_optimum))
        return float(k_ref * arrhenius * pH_factor)

    def simulate(self, duration_hr: float = 4.0, dt_hr: float = 0.05) -> CementationResult:
        p = self.params
        f = self.feedstock

        k_cu = self._effective_rate(p.cu_rate_const_per_hr, f.temperature_C, f.pH)
        k_ni = self._effective_rate(p.ni_rate_const_per_hr, f.temperature_C, f.pH)
        k_zn = self._effective_rate(p.zn_rate_const_per_hr, f.temperature_C, f.pH)

        n_steps = int(duration_hr / dt_hr) + 1
        t = np.linspace(0.0, duration_hr, n_steps)
        cu = np.empty(n_steps)
        ni = np.empty(n_steps)
        zn = np.empty(n_steps)
        cu[0], ni[0], zn[0] = f.cu_M, f.ni_M, f.zn_M

        # Fe powder dose gives a surface-site proxy (consumed as metals cement)
        dose_remaining = p.iron_powder_dose_g_per_L
        fe_consumed = 0.0

        for i in range(n_steps - 1):
            theta = max(dose_remaining / p.iron_powder_dose_g_per_L, 0.0)
            dcu = -k_cu * cu[i] * theta * dt_hr
            dni = -k_ni * ni[i] * theta * dt_hr
            dzn = -k_zn * zn[i] * theta * dt_hr

            cu[i + 1] = max(0.0, cu[i] + dcu)
            ni[i + 1] = max(0.0, ni[i] + dni)
            zn[i + 1] = max(0.0, zn[i] + dzn)

            # Fe consumed = moles of metals displaced (stoichiometry 1:1)
            displaced_mol = abs(dcu) + abs(dni) + abs(dzn)
            fe_consumed += displaced_mol
            dose_remaining = max(0.0, dose_remaining - displaced_mol * M_FE)

        # Cost: iron powder consumed + regeneration
        powder_kg = fe_consumed * M_FE / 1000.0 * f.volume_L
        cost = powder_kg * p.iron_powder_cost_per_kg

        return CementationResult(
            time_hr=t,
            cu_M=cu,
            ni_M=ni,
            zn_M=zn,
            fe_consumed_M=fe_consumed,
            fe_released_M=fe_consumed,  # 1:1 displacement
            powder_consumed_kg=powder_kg,
            cost_USD=cost,
        )


class HydrolysisModel:
    """Fe(OH)₃ precipitation with co-removal of Cu/Ni/Zn.

    Raises pH to *target_pH* where Fe³⁺ solubility is far below its
    initial concentration.  Fe²⁺ remains in solution (its solubility
    limit is ~pH 8–9).  A fraction of Cu/Ni/Zn co-precipitates or
    adsorbs on the Fe(OH)₃ flocs.
    """

    def __init__(
        self,
        params: Optional[HydrolysisParams] = None,
        feedstock: Optional[PurificationFeedstock] = None,
    ) -> None:
        self.params = params or HydrolysisParams()
        self.feedstock = feedstock or PurificationFeedstock()

    def simulate(self) -> HydrolysisResult:
        p = self.params
        f = self.feedstock
        volume = f.volume_L

        # Fe³⁺ precipitation: assume complete removal to solubility limit
        fe3_precipitated = max(0.0, f.fe3_M - p.fe3_solubility_at_target_M)

        # Fe(OH)₃ mass (g → kg): Fe(OH)₃ molar mass = 106.87 g/mol
        M_FEOH3 = 106.87
        precipitate_mass_kg = fe3_precipitated * volume * M_FEOH3 / 1000.0

        # Co-precipitation of contaminants
        cu_removed = f.cu_M * p.co_precipitation_fraction_cu
        ni_removed = f.ni_M * p.co_precipitation_fraction_ni
        zn_removed = f.zn_M * p.co_precipitation_fraction_zn

        # NaOH consumption for pH adjustment
        delta_pH = max(0.0, p.target_pH - f.pH)
        naoh_mol = delta_pH * p.naoh_mol_per_pH_unit_per_L * volume
        naoh_kg = naoh_mol * 40.0 / 1000.0  # NaOH = 40 g/mol
        naoh_cost = naoh_kg * p.naoh_cost_per_kg

        return HydrolysisResult(
            fe3_remaining_M=p.fe3_solubility_at_target_M,
            cu_removed_M=cu_removed,
            ni_removed_M=ni_removed,
            zn_removed_M=zn_removed,
            fe2_unchanged_M=f.fe2_M,
            precipitate_mass_kg=precipitate_mass_kg,
            naoh_consumed_kg=naoh_kg,
            cost_USD=naoh_cost,
        )


class SelectiveElectrowinningModel:
    """Side-stream electrowinning cell for Cu/Ni removal at controlled potential.

    Uses a simplified Butler-Volmer expression at low overpotential to
    estimate deposition current for each metal.  The cathode potential
    is set between Cu⁰ (plate) and Fe⁰ (don't plate).
    """

    def __init__(
        self,
        params: Optional[ElectrowinningParams] = None,
        feedstock: Optional[PurificationFeedstock] = None,
    ) -> None:
        self.params = params or ElectrowinningParams()
        self.feedstock = feedstock or PurificationFeedstock()

    def simulate(
        self,
        cu_M: float,
        ni_M: float,
        duration_hr: float = 1.0,
    ) -> ElectrowinningResult:
        p = self.params
        f = self.feedstock

        # Overpotential for each metal relative to its equilibrium potential
        # η = E_cathode - E_eq(M) where E_eq ≈ E0 + (RT/2F) ln([M²⁺])
        def eta(metal_E0: float, conc_M: float) -> float:
            if conc_M <= 0:
                return 0.0
            eq = metal_E0 + (R_GAS * (f.temperature_C + 273.15)) / (2.0 * FARADAY) * math.log(max(conc_M, 1e-30))
            return p.cathode_potential_V - eq

        eta_cu = eta(E0_CU, cu_M)
        eta_ni = eta(E0_NI, ni_M)

        # Deposition current density: i = i0 * exp(η / b) for cathodic (η < 0)
        # For Cu: η is very negative (strong deposition)
        # For Ni: η may be slightly positive (no deposition at this potential)
        def current_density(eta_V: float, i0: float) -> float:
            if eta_V >= 0:
                return 0.0  # no deposition if potential is above equilibrium
            return float(i0 * math.exp(-abs(eta_V) / p.tafel_slope_V))

        j_cu = current_density(eta_cu, p.cu_exchange_current_A_m2)
        j_ni = current_density(eta_ni, p.ni_exchange_current_A_m2)

        # Total deposition rate: i * A / (n * F) in mol/s → mol/hr
        cu_rate = j_cu * p.cathode_area_m2 / (2.0 * FARADAY) * 3600.0  # mol/hr
        ni_rate = j_ni * p.cathode_area_m2 / (2.0 * FARADAY) * 3600.0

        # Concentration removed in duration (limited by available metal)
        cu_removed = min(cu_M, cu_rate * duration_hr / f.volume_L * duration_hr)
        ni_removed = min(ni_M, ni_rate * duration_hr / f.volume_L * duration_hr)

        # Energy: V_cell * I_total * t
        I_total = (j_cu + j_ni) * p.cathode_area_m2  # Amperes
        energy_kWh = p.cell_voltage_V * I_total * duration_hr / 1000.0
        cost = energy_kWh * p.electricity_cost_per_kWh

        return ElectrowinningResult(
            cu_plated_M=cu_removed,
            ni_plated_M=ni_removed,
            cu_remaining_M=cu_M - cu_removed,
            ni_remaining_M=ni_M - ni_removed,
            energy_kWh=energy_kWh,
            cost_USD=cost,
        )


class IonExchangeModel:
    """Chelating resin polishing for residual Cu/Ni.

    Uses a simple capacity model: the resin removes metal until its
    exchange capacity is exhausted.  Selectivity over Fe is high
    (chelating iminodiacetic / aminophosphonic resins).
    """

    def __init__(
        self,
        params: Optional[IonExchangeParams] = None,
        feedstock: Optional[PurificationFeedstock] = None,
    ) -> None:
        self.params = params or IonExchangeParams()
        self.feedstock = feedstock or PurificationFeedstock()

    def simulate(self, cu_M: float, ni_M: float) -> IonExchangeResult:
        p = self.params
        f = self.feedstock
        volume = f.volume_L

        # Capacity in mol: meq/mL → meq/L → mol/L (div by charge = 2)
        cu_capacity_M = p.cu_capacity_meq_per_mL * 1000.0 / 2.0 / 1000.0
        ni_capacity_M = p.ni_capacity_meq_per_mL * 1000.0 / 2.0 / 1000.0

        # Scale by resin volume relative to batch volume
        scale = p.resin_volume_L / volume
        cu_removable = min(cu_M, cu_capacity_M * scale)
        ni_removable = min(ni_M, ni_capacity_M * scale)

        # Assume >99% removal when capacity allows (strong chelating resin)
        cu_removed = cu_removable * 0.99
        ni_removed = ni_removable * 0.99

        # Cost per cycle: fraction of resin replacement + regeneration
        resin_cost_per_cycle = (
            p.resin_volume_L * p.resin_cost_per_L / p.resin_lifetime_cycles
            + p.regeneration_acid_cost_per_cycle
        )

        return IonExchangeResult(
            cu_removed_M=cu_removed,
            ni_removed_M=ni_removed,
            cu_final_M=cu_M - cu_removed,
            ni_final_M=ni_M - ni_removed,
            resin_cost_per_cycle=resin_cost_per_cycle,
            cost_USD=resin_cost_per_cycle,
        )


class PurificationModel:
    """Full purification train: cementation → hydrolysis → electrowinning → IX.

    Chains the four unit operations and computes integrated removal
    efficiency, cost per tonne of Fe, and impurity build-up rate in
    the closed-loop recycle.
    """

    def __init__(
        self,
        feedstock: Optional[PurificationFeedstock] = None,
        cementation: Optional[CementationParams] = None,
        hydrolysis: Optional[HydrolysisParams] = None,
        electrowinning: Optional[ElectrowinningParams] = None,
        ion_exchange: Optional[IonExchangeParams] = None,
    ) -> None:
        self.feedstock = feedstock or PurificationFeedstock()
        self._cementation_params = cementation or CementationParams()
        self._hydrolysis_params = hydrolysis or HydrolysisParams()
        self._electrowinning_params = electrowinning or ElectrowinningParams()
        self._ion_exchange_params = ion_exchange or IonExchangeParams()

    def simulate(
        self,
        cementation_duration_hr: float = 4.0,
        electrowinning_duration_hr: float = 1.0,
    ) -> PurificationResult:
        f = self.feedstock

        # ── Stage 1: Cementation ──────────────────────────────────────
        cem_model = CementationModel(self._cementation_params, f)
        cem = cem_model.simulate(cementation_duration_hr)

        cu_after_cem = float(cem.cu_M[-1])
        ni_after_cem = float(cem.ni_M[-1])
        zn_after_cem = float(cem.zn_M[-1])

        # ── Stage 2: Hydrolysis ───────────────────────────────────────
        # Update feedstock composition for hydrolysis (Fe²⁺ increases from cementation)
        hydro_feed = PurificationFeedstock(
            cu_M=cu_after_cem,
            ni_M=ni_after_cem,
            zn_M=zn_after_cem,
            fe2_M=f.fe2_M + cem.fe_released_M,
            fe3_M=f.fe3_M,
            pH=f.pH,
            temperature_C=f.temperature_C,
            volume_L=f.volume_L,
        )
        hydro_model = HydrolysisModel(self._hydrolysis_params, hydro_feed)
        hydro = hydro_model.simulate()

        cu_after_hydro = cu_after_cem - hydro.cu_removed_M
        ni_after_hydro = ni_after_cem - hydro.ni_removed_M
        zn_after_hydro = zn_after_cem - hydro.zn_removed_M

        # ── Stage 3: Selective Electrowinning ─────────────────────────
        ew_model = SelectiveElectrowinningModel(self._electrowinning_params, hydro_feed)
        ew = ew_model.simulate(cu_after_hydro, ni_after_hydro, electrowinning_duration_hr)

        cu_after_ew = ew.cu_remaining_M
        ni_after_ew = ew.ni_remaining_M

        # ── Stage 4: Ion Exchange ─────────────────────────────────────
        ix_model = IonExchangeModel(self._ion_exchange_params, hydro_feed)
        ix = ix_model.simulate(cu_after_ew, ni_after_ew)

        cu_final = ix.cu_final_M
        ni_final = ix.ni_final_M
        zn_final = max(0.0, zn_after_hydro)  # IX does not target Zn

        # ── Cost aggregation ──────────────────────────────────────────
        M_FE_KG = M_FE / 1000.0  # kg/mol
        fe_mass_t = f.fe2_M * f.volume_L * M_FE_KG / 1000.0  # tonnes Fe in feedstock

        stage_costs = {
            "cementation_USD": cem.cost_USD,
            "hydrolysis_USD": hydro.cost_USD,
            "electrowinning_USD": ew.cost_USD,
            "ion_exchange_USD": ix.cost_USD,
        }
        total_cost = sum(stage_costs.values())
        cost_per_t = total_cost / max(fe_mass_t, 1e-12)

        # Cu wt% relative to Fe
        cu_removal = 1.0 - cu_final / f.cu_M if f.cu_M > 0 else 1.0
        # wt% = (cu_M / fe2_M) * (M_CU / M_FE) * 100
        cu_wt_pct = cu_final / max(f.fe2_M, 1e-12) * M_CU / M_FE * 100.0

        # Impurity build-up rate in recycle loop (closed_loop integration)
        # Each pass through purification removes a fraction; if impurity
        # enters at feed_impurity_M and is removed at efficiency η,
        # steady-state accumulation = feed_M / η in the worst case.
        # Build-up rate = impurity feed rate - removal rate
        total_removal = cu_removal  # dominant contaminant
        impurity_rate = f.cu_M * (1.0 - total_removal)  # M per pass (residual)

        return PurificationResult(
            cementation=cem,
            hydrolysis=hydro,
            electrowinning=ew,
            ion_exchange=ix,
            cu_initial_M=f.cu_M,
            cu_final_M=cu_final,
            cu_removal_fraction=cu_removal,
            cu_final_wt_pct=cu_wt_pct,
            ni_initial_M=f.ni_M,
            ni_final_M=ni_final,
            zn_initial_M=f.zn_M,
            zn_final_M=zn_final,
            total_cost_per_t_fe=cost_per_t,
            stage_costs=stage_costs,
            impurity_buildup_rate_M_per_hr=impurity_rate,
        )


# ── Integration helper ────────────────────────────────────────────────

def purification_from_closed_loop_result(
    closed_loop_result,
    cementation: Optional[CementationParams] = None,
    hydrolysis: Optional[HydrolysisParams] = None,
    electrowinning: Optional[ElectrowinningParams] = None,
    ion_exchange: Optional[IonExchangeParams] = None,
) -> PurificationResult:
    """Run purification on the impurity profile from a PhaseIVClosedLoop result.

    Extracts the final impurity concentration and Fe²⁺ level from the
    closed-loop CSTR simulation, then runs the purification train.
    The impurity from the closed-loop model (anode coating fragments,
    generic contaminants) is used as a proxy for the Cu equivalent that
    must be removed.
    """
    impurity_M = float(closed_loop_result.impurity_M[-1])
    fe_M = float(closed_loop_result.fe_M[-1])

    feed = PurificationFeedstock(
        cu_M=impurity_M,       # treat total impurity as Cu-equivalent
        ni_M=impurity_M * 0.3, # assume 30% Ni fraction
        zn_M=impurity_M * 0.2, # assume 20% Zn fraction
        fe2_M=fe_M,
    )
    model = PurificationModel(feed, cementation, hydrolysis, electrowinning, ion_exchange)
    return model.simulate()
