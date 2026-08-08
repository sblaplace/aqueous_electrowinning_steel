"""Phase IV anode-durability and closed-loop electrolyte screening model.

The module couples three deliberately transparent engineering balances:

* charge-throughput-based anode coating wear and voltage drift;
* a constant-volume CSTR balance for Fe, ligand, chloride and impurities; and
* cell-voltage, energy, purge and consumables metrics.

It is a screening model, not a corrosion-life qualification or an electrolyte
speciation package.  Wear coefficients, precipitation kinetics and impurity
limits must be calibrated with accelerated-life and closed-loop experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from .anode import AnodeKinetics
from .electrochemistry import FARADAY, M_FE, CellVoltageModel, specific_energy_kWh_per_t


@dataclass(frozen=True)
class AnodeDurabilityParams:
    """Empirical coating-wear parameters for an accelerated-life screen."""

    coating_loading_g_m2: float = 12.0
    base_wear_mg_per_kAh: float = 0.35
    temperature_acceleration_per_C: float = 0.025
    chloride_acceleration_per_M: float = 0.10
    acidity_acceleration_per_pH_below_2: float = 0.20
    cer_acceleration: float = 2.0
    activity_exponent: float = 1.5
    resistance_growth_ohm_m2: float = 4.0e-4
    end_of_life_fraction: float = 0.20
    replacement_cost_per_m2: float = 150.0

    def __post_init__(self) -> None:
        if self.coating_loading_g_m2 <= 0 or self.base_wear_mg_per_kAh < 0:
            raise ValueError("coating loading must be positive and wear non-negative")
        if not 0 < self.end_of_life_fraction < 1:
            raise ValueError("end_of_life_fraction must lie between 0 and 1")


@dataclass(frozen=True)
class ClosedLoopParams:
    """Operating and balance parameters for a constant-volume CSTR."""

    volume_L: float = 1000.0
    feed_flow_L_hr: float = 20.0
    purge_flow_L_hr: float = 20.0
    fe_feed_M: float = 1.20
    ligand_feed_M: float = 1.50
    chloride_feed_M: float = 0.0
    impurity_feed_M: float = 2.0e-4
    fe_initial_M: float = 1.00
    ligand_initial_M: float = 1.30
    chloride_initial_M: float = 0.0
    impurity_initial_M: float = 2.0e-4
    fe_solubility_M: float = 1.50
    precipitation_rate_per_hr: float = 0.5
    ligand_decay_per_hr: float = 1.0e-4
    impurity_removal_fraction: float = 0.0
    impurity_limit_M: float = 0.010
    minimum_fe_M: float = 0.20
    minimum_ligand_to_fe: float = 1.0
    electrolyte_makeup_cost_per_mol_ligand: float = 0.15
    purge_treatment_cost_per_m3: float = 8.0

    def __post_init__(self) -> None:
        positive = (self.volume_L, self.feed_flow_L_hr, self.fe_feed_M)
        if any(x <= 0 for x in positive) or self.purge_flow_L_hr < 0:
            raise ValueError("volume, feed flow and Fe feed must be positive; purge cannot be negative")
        if abs(self.feed_flow_L_hr - self.purge_flow_L_hr) > 1e-12:
            raise ValueError("feed and purge flows must match in the constant-volume model")
        if self.precipitation_rate_per_hr < 0 or self.ligand_decay_per_hr < 0:
            raise ValueError("rate constants cannot be negative")
        if not 0 <= self.impurity_removal_fraction <= 1:
            raise ValueError("impurity_removal_fraction must lie between 0 and 1")


@dataclass(frozen=True)
class PhaseIVOperatingPoint:
    """Electrolyzer conditions shared by the durability and CSTR models."""

    current_density_mA_cm2: float = 100.0
    anode_area_m2: float = 1.0
    current_efficiency: float = 0.95
    eta_cathode_V: float = 0.30
    ir_drop_V: float = 0.20
    electricity_price_per_kWh: float = 0.04

    def __post_init__(self) -> None:
        if self.current_density_mA_cm2 <= 0 or self.anode_area_m2 <= 0:
            raise ValueError("current density and area must be positive")
        if not 0 < self.current_efficiency <= 1:
            raise ValueError("current_efficiency must lie in (0, 1]")

    @property
    def current_A(self) -> float:
        return self.current_density_mA_cm2 * 10.0 * self.anode_area_m2

    @property
    def fe_removal_mol_hr(self) -> float:
        return self.current_A * self.current_efficiency * 3600.0 / (2.0 * FARADAY)


@dataclass
class PhaseIVResult:
    """Time-series output from :class:`PhaseIVClosedLoop`."""

    time_hr: np.ndarray
    fe_M: np.ndarray
    ligand_M: np.ndarray
    chloride_M: np.ndarray
    impurity_M: np.ndarray
    precipitated_fe_mol: np.ndarray
    coating_remaining_fraction: np.ndarray
    anode_overpotential_V: np.ndarray
    cell_voltage_V: np.ndarray
    specific_energy_kWh_t: np.ndarray
    cer_fraction: np.ndarray
    end_of_life_fraction: float = 0.20
    flags: list[list[str]] = field(default_factory=list)

    def summary(self) -> dict:
        eol = np.flatnonzero(self.coating_remaining_fraction <= self.end_of_life_fraction)
        flagged = sum(bool(x) for x in self.flags)
        return {
            "duration_hr": float(self.time_hr[-1]),
            "final_fe_M": float(self.fe_M[-1]),
            "final_ligand_M": float(self.ligand_M[-1]),
            "final_impurity_M": float(self.impurity_M[-1]),
            "final_coating_remaining_fraction": float(self.coating_remaining_fraction[-1]),
            "initial_cell_voltage_V": float(self.cell_voltage_V[0]),
            "final_cell_voltage_V": float(self.cell_voltage_V[-1]),
            "final_specific_energy_kWh_t": float(self.specific_energy_kWh_t[-1]),
            "cumulative_precipitated_fe_mol": float(self.precipitated_fe_mol[-1]),
            "flagged_time_points": int(flagged),
            "end_of_life_hr": None if eol.size == 0 else float(self.time_hr[eol[0]]),
        }

    def as_columns(self) -> dict[str, np.ndarray]:
        """Return numeric columns suitable for a DataFrame or CSV writer."""
        return {
            "time_hr": self.time_hr,
            "fe_M": self.fe_M,
            "ligand_M": self.ligand_M,
            "chloride_M": self.chloride_M,
            "impurity_M": self.impurity_M,
            "precipitated_fe_mol": self.precipitated_fe_mol,
            "coating_remaining_fraction": self.coating_remaining_fraction,
            "anode_overpotential_V": self.anode_overpotential_V,
            "cell_voltage_V": self.cell_voltage_V,
            "specific_energy_kWh_t": self.specific_energy_kWh_t,
            "cer_fraction": self.cer_fraction,
        }


class PhaseIVClosedLoop:
    """Coupled durability/CSTR simulator with explicit, auditable balances."""

    def __init__(
        self,
        anode: AnodeKinetics,
        loop: Optional[ClosedLoopParams] = None,
        durability: Optional[AnodeDurabilityParams] = None,
        operating: Optional[PhaseIVOperatingPoint] = None,
    ) -> None:
        self.anode = anode
        self.loop = loop or ClosedLoopParams()
        self.durability = durability or AnodeDurabilityParams()
        self.operating = operating or PhaseIVOperatingPoint()

    def wear_rate_mg_per_kAh(self, chloride_M: float, cer_fraction: float) -> float:
        """Condition-adjusted coating wear; calibration is experiment-specific."""
        d = self.durability
        temperature = self.anode.material.temperature_C
        factor = np.exp(d.temperature_acceleration_per_C * max(temperature - 25.0, 0.0))
        factor *= 1.0 + d.chloride_acceleration_per_M * max(chloride_M, 0.0)
        factor *= 1.0 + d.acidity_acceleration_per_pH_below_2 * max(2.0 - self.anode.pH, 0.0)
        factor *= 1.0 + d.cer_acceleration * np.clip(cer_fraction, 0.0, 1.0)
        return float(d.base_wear_mg_per_kAh * factor)

    def degraded_anode(
        self, remaining_fraction: float, chloride_M: Optional[float] = None
    ) -> AnodeKinetics:
        """Construct an anode with composition-coupled activity and resistance."""
        fraction = float(np.clip(remaining_fraction, 1e-6, 1.0))
        material = replace(
            self.anode.material,
            oer_i0=self.anode.material.oer_i0 * fraction ** self.durability.activity_exponent,
            cer_i0=(None if self.anode.material.cer_i0 is None else
                    self.anode.material.cer_i0 * fraction ** self.durability.activity_exponent),
        )
        return replace(
            self.anode,
            material=material,
            a_Cl_molar=(self.anode.a_Cl_molar if chloride_M is None else max(chloride_M, 0.0)),
            electrolyte_resistivity_ohm_m2=(
                self.anode.electrolyte_resistivity_ohm_m2
                + self.durability.resistance_growth_ohm_m2 * (1.0 - fraction)
            ),
        )

    def _flags(self, fe: float, ligand: float, impurity: float, remaining: float) -> list[str]:
        flags: list[str] = []
        if fe < self.loop.minimum_fe_M:
            flags.append("low_fe")
        if ligand / max(fe, 1e-12) < self.loop.minimum_ligand_to_fe:
            flags.append("low_ligand_ratio")
        if impurity > self.loop.impurity_limit_M:
            flags.append("high_impurity")
        if fe > self.loop.fe_solubility_M:
            flags.append("supersaturated_fe")
        if remaining <= self.durability.end_of_life_fraction:
            flags.append("anode_end_of_life")
        return flags

    def simulate(self, duration_hr: float = 4000.0, dt_hr: float = 1.0) -> PhaseIVResult:
        """Integrate balances with a positive-preserving forward time step."""
        if duration_hr <= 0 or dt_hr <= 0:
            raise ValueError("duration_hr and dt_hr must be positive")
        # CSTR stability guard; keeps the transparent Euler integration well behaved.
        if dt_hr * max(self.loop.feed_flow_L_hr, self.loop.purge_flow_L_hr) / self.loop.volume_L > 0.25:
            raise ValueError("dt_hr is too large relative to the CSTR residence time")

        t = np.arange(0.0, duration_hr + 0.5 * dt_hr, dt_hr)
        n = len(t)
        arrays = [np.empty(n) for _ in range(10)]
        fe, ligand, chloride, impurity, precip, remaining, eta, voltage, energy, cer = arrays
        fe[0], ligand[0], chloride[0], impurity[0] = (
            self.loop.fe_initial_M, self.loop.ligand_initial_M,
            self.loop.chloride_initial_M, self.loop.impurity_initial_M,
        )
        precip[0], remaining[0] = 0.0, 1.0
        flags: list[list[str]] = []

        coating_mg_m2 = self.durability.coating_loading_g_m2 * 1000.0
        qin, qout, volume = self.loop.feed_flow_L_hr, self.loop.purge_flow_L_hr, self.loop.volume_L
        current_density_A_m2 = self.operating.current_density_mA_cm2 * 10.0

        for i in range(n):
            active = self.degraded_anode(remaining[i], chloride[i])
            ar = active.overpotential_at_current(self.operating.current_density_mA_cm2)
            eta[i], cer[i] = ar["total_V"], ar["cer_fraction"]
            cv = CellVoltageModel(
                eta_cathode=self.operating.eta_cathode_V,
                ir_drop=self.operating.ir_drop_V,
                anode=active,
                j_operating_mA_cm2=self.operating.current_density_mA_cm2,
            )
            voltage[i] = cv.V_cell
            energy[i] = specific_energy_kWh_per_t(voltage[i], self.operating.current_efficiency)
            flags.append(self._flags(fe[i], ligand[i], impurity[i], remaining[i]))
            if i == n - 1:
                break

            wear = self.wear_rate_mg_per_kAh(chloride[i], cer[i])
            charge_kAh_m2 = current_density_A_m2 * dt_hr / 1000.0
            coating_loss_mg_m2 = wear * charge_kAh_m2
            remaining[i + 1] = max(0.0, remaining[i] - coating_loss_mg_m2 / coating_mg_m2)

            # Concentrations are mol/L. Flow terms (L/hr * mol/L) and electrode
            # terms (mol/hr) divide directly by volume in L.
            supersaturation = max(fe[i] - self.loop.fe_solubility_M, 0.0)
            precipitation_mol_hr = self.loop.precipitation_rate_per_hr * supersaturation * volume
            dfe = (
                qin * self.loop.fe_feed_M - qout * fe[i]
                - self.operating.fe_removal_mol_hr - precipitation_mol_hr
            ) / volume
            dligand = (
                qin * self.loop.ligand_feed_M - qout * ligand[i]
                - self.loop.ligand_decay_per_hr * ligand[i] * volume
            ) / volume
            dchloride = (qin * self.loop.chloride_feed_M - qout * chloride[i]) / volume
            # Coating loss is represented as a conservative one-mole-per-metal-
            # atom impurity source with a 0.1 kg/mol placeholder molar mass.
            # Replace with measured coating chemistry for design work.
            anode_impurity_mol_hr = coating_loss_mg_m2 * self.operating.anode_area_m2 / 100_000.0 / dt_hr
            impurity_out = qout * impurity[i] * (1.0 + self.loop.impurity_removal_fraction)
            dimpurity = (qin * self.loop.impurity_feed_M + anode_impurity_mol_hr - impurity_out) / volume

            fe[i + 1] = max(0.0, fe[i] + dt_hr * dfe)
            ligand[i + 1] = max(0.0, ligand[i] + dt_hr * dligand)
            chloride[i + 1] = max(0.0, chloride[i] + dt_hr * dchloride)
            impurity[i + 1] = max(0.0, impurity[i] + dt_hr * dimpurity)
            precip[i + 1] = precip[i] + precipitation_mol_hr * dt_hr

        return PhaseIVResult(
            time_hr=t,
            fe_M=fe,
            ligand_M=ligand,
            chloride_M=chloride,
            impurity_M=impurity,
            precipitated_fe_mol=precip,
            coating_remaining_fraction=remaining,
            anode_overpotential_V=eta,
            cell_voltage_V=voltage,
            specific_energy_kWh_t=energy,
            cer_fraction=cer,
            end_of_life_fraction=self.durability.end_of_life_fraction,
            flags=flags,
        )

    def process_metrics(self, result: PhaseIVResult) -> dict:
        """Return integrated production, reagent, purge, energy and cost metrics."""
        duration = float(result.time_hr[-1])
        production_t = self.operating.fe_removal_mol_hr * duration * M_FE / 1000.0
        integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        energy_kWh = float(integrate(
            self.operating.current_A * result.cell_voltage_V / 1000.0,
            result.time_hr,
        ))
        purge_m3 = self.loop.purge_flow_L_hr * duration / 1000.0
        ligand_makeup_mol = self.loop.feed_flow_L_hr * self.loop.ligand_feed_M * duration
        coating_used_fraction = max(0.0, 1.0 - result.coating_remaining_fraction[-1])
        anode_cost = coating_used_fraction * self.durability.replacement_cost_per_m2 * self.operating.anode_area_m2
        treatment_cost = purge_m3 * self.loop.purge_treatment_cost_per_m3
        ligand_cost = ligand_makeup_mol * self.loop.electrolyte_makeup_cost_per_mol_ligand
        return {
            "iron_produced_t": production_t,
            "electricity_kWh": energy_kWh,
            "average_specific_energy_kWh_t": energy_kWh / max(production_t, 1e-30),
            "electricity_cost": energy_kWh * self.operating.electricity_price_per_kWh,
            "purge_volume_m3": purge_m3,
            "purge_treatment_cost": treatment_cost,
            "ligand_makeup_mol": ligand_makeup_mol,
            "ligand_makeup_cost": ligand_cost,
            "anode_consumption_cost": anode_cost,
            "modeled_variable_cost_per_t": (
                energy_kWh * self.operating.electricity_price_per_kWh
                + treatment_cost + ligand_cost + anode_cost
            ) / max(production_t, 1e-30),
        }

    # ── Stress-relaxation screen (additive, default-off) ─────────────────────
    # Reuses the already-integrated PhaseIVResult to report a stress-driven
    # defect rate per step (CHEM_PHYS_REVIEW Tier 3.5).  Does not alter
    # ``simulate`` — callers opt in explicitly; default behaviour is unchanged.

    def stress_relaxation_screen(
        self,
        result: PhaseIVResult,
        temperature_C: float = 60.0,
        bath_pH: float = 3.0,
        ambient_temperature_C: float = 25.0,
        substrate="ti_passive_tio2",
        saccharin_g_L: float = 0.0,
        chloride_bath: bool = False,
        C_H_ppm: float | None = None,
        params=None,
        sigma_survival_threshold_MPa: float = 150.0,
    ) -> dict:
        """Optional: per-step stress-mechanism defect rate over a closed-loop run.

        Every closed-loop time step gets the ``internal_stress`` snapshot
        re-derived at the operating j/FE (Faraday thickness growth), then the
        ``stress_relaxation`` log-linear closure is applied against elapsed
        time to yield retained stress, a stress-mechanism defect rate, and a
        drum-winding survival verdict.  Returns per-step arrays plus the
        terminal state.  Off by default — callers must opt in explicitly.
        """
        from .stress_relaxation import (
            seed_stress_snapshot_Mpa,
            sigma_relaxation_series,
            sigma_relaxed,
            stress_defect_rate,
            survives_drum_winding,
        )

        t_hr = np.asarray(result.time_hr, dtype=float)
        j = self.operating.current_density_mA_cm2
        fe_percent = self.operating.current_efficiency * 100.0

        # Re-derive the deposition-time at each step from the run's Faraday
        # throughput so sigma0 grows with thickness, matching how the deposit
        # actually accumulates on the drum.
        snap = seed_stress_snapshot_Mpa(
            j_mA_cm2=j,
            current_efficiency_percent=fe_percent,
            deposition_time_s=max(float(t_hr[-1]) * 3600.0, 1e-6),
            bath_pH=bath_pH,
            temperature_C=temperature_C,
            substrate=substrate,
            saccharin_g_L=saccharin_g_L,
            chloride_bath=chloride_bath,
        )
        sigma0 = float(snap["sigma0_MPa"])
        c_h = C_H_ppm if C_H_ppm is not None else float(
            snap["derived"]["C_H_diffusible_ppm"]
        )

        sigma_t = sigma_relaxation_series(
            sigma0, t_hr, temperature_C=temperature_C, C_H_ppm=float(c_h), params=params
        )
        # Defect rate per step from the retained stress.
        rates = np.array(
            [
                stress_defect_rate(float(s), sigma0, params)["defect_rate_per_hr"]
                for s in sigma_t
            ]
        )
        terminal = sigma_relaxed(
            sigma0, float(t_hr[-1]),
            temperature_C=temperature_C, C_H_ppm=float(c_h), params=params,
        )
        terminal_life = survives_drum_winding(
            terminal["sigma_MPa"], sigma_survival_threshold_MPa, params
        )
        return {
            "time_hr": t_hr,
            "sigma0_MPa": float(sigma0),
            "sigma_MPa": sigma_t,
            "retained_fraction": sigma_t / max(sigma0, 1e-30),
            "defect_rate_per_hr": rates,
            "C_H_ppm": float(c_h),
            "tau_hr": float(terminal["tau_hr"]),
            "terminal_sigma_MPa": float(terminal["sigma_MPa"]),
            "terminal_defect_rate_per_hr": float(stress_defect_rate(
                terminal["sigma_MPa"], sigma0, params
            )["defect_rate_per_hr"]),
            "winding_verdict": terminal_life,
            "default_unchanged": True,  # simulate() is untouched by this screen
        }
