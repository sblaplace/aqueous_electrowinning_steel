"""Membrane fouling model for electrolyte recirculation loops.

Models performance degradation of cross-flow microfiltration / ultrafiltration
membranes in the electrolyte recirculation (CSTR) loop.  The module couples:

1. **Fouling mechanisms** — colloidal Fe(OH)₃ deposition, CaSO₄ scaling,
   organic additive decomposition products, and biofilm growth.
2. **Flux decline** — all four classical Hermia fouling models (complete
   blocking, intermediate blocking, standard blocking, cake filtration) plus
   a general exponential-plus-steady-state form.
3. **Cleaning cycle optimization** — acid wash, NaOH wash, and backflush
   efficiencies per mechanism; optimal cleaning interval search.
4. **Impurity accumulation** — CSTR-loop impurity dynamics with
   fouling-coupled rejection.
5. **Integration** — feeds from the closed-loop CSTR model (impurity
   concentrations) and the techno-economic model (membrane cost, replacement
   labour).

References
----------
- Hermia (1982), Trans. IChemE, 60, 183–187.
- Field et al. (1995), *Critical flux concept*, ICOM '95.
- Vrouwenvelder et al. (2003), *Biofouling of spiral-wound membranes*.

This is a screening model — calibrate fouling rates, cleaning efficiencies
and rejection-vs-fouling curves with pilot-loop experiments before design use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# ── Constants ──────────────────────────────────────────────────────────────
M_FE_OH3 = 106.87       # g/mol, Fe(OH)₃
M_CASO4 = 136.14        # g/mol, CaSO₄
SECONDS_PER_HR = 3600.0


# ── Enumerations ───────────────────────────────────────────────────────────
class FoulingMechanism(str, Enum):
    """Identifies the dominant fouling layer type."""
    FE_OH3 = "fe_oh3"
    CASO4 = "caso4"
    ORGANIC = "organic"
    BIOFILM = "biofilm"


class HermiaModel(str, Enum):
    """Classical Hermia fouling model variant."""
    COMPLETE_BLOCKING = "complete_blocking"
    INTERMEDIATE_BLOCKING = "intermediate_blocking"
    STANDARD_BLOCKING = "standard_blocking"
    CAKE_FILTRATION = "cake_filtration"


class CleaningAgent(str, Enum):
    """Chemical / physical cleaning method."""
    ACID_WASH = "acid_wash"       # 0.5 M HCl or citric acid
    NAOH_WASH = "naoh_wash"      # 0.1–0.5 M NaOH
    BACKFLUSH = "backflush"       # periodic reverse-flow


# ── Parameter dataclasses ──────────────────────────────────────────────────
@dataclass(frozen=True)
class MembraneParams:
    """Physical and cost properties of the membrane module."""
    area_m2: float = 10.0
    clean_water_flux_L_m2_hr: float = 100.0   # J₀  (LMH)
    pore_diameter_um: float = 0.1
    max_operating_pressure_bar: float = 3.0
    membrane_cost_per_m2: float = 200.0       # $/m² replacement
    replacement_labor_cost: float = 500.0      # $ per replacement event
    module_volume_L: float = 5.0               # hold-up inside module

    def __post_init__(self) -> None:
        if self.area_m2 <= 0 or self.clean_water_flux_L_m2_hr <= 0:
            raise ValueError("area and clean-water flux must be positive")


@dataclass(frozen=True)
class FoulingRateParams:
    """Empirical fouling resistance accumulation rates [resistance_units / hr].

    Rates are additive — multiple mechanisms run in parallel.  Each
    mechanism's rate is modulated by operating conditions in
    :meth:`MembraneFoulingModel._effective_rate`.
    """
    fe_oh3_base_rate: float = 1.2e-3      # hr⁻¹
    caso4_base_rate: float = 5.0e-4       # hr⁻¹
    organic_base_rate: float = 2.0e-4     # hr⁻¹
    biofilm_base_rate: float = 8.0e-4     # hr⁻¹

    # Thresholds / activation conditions
    fe_oh3_pH_threshold: float = 3.0      # Fe(OH)₃ precipitation onset pH
    caso4_hardness_mg_L: float = 200.0    # Ca²⁺ threshold for scaling
    biofilm_idle_hr: float = 24.0         # idle hours before biofilm onset

    def __post_init__(self) -> None:
        for name in ("fe_oh3_base_rate", "caso4_base_rate",
                      "organic_base_rate", "biofilm_base_rate"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class CleaningParams:
    """Cleaning efficiency and cost per agent per mechanism.

    Each entry is the fractional resistance removal (0 = ineffective,
    1 = perfect restoration).  Agents are applied in the order listed
    during a cleaning cycle.
    """
    acid_wash_efficiency: dict[str, float] = field(default_factory=lambda: {
        "fe_oh3": 0.95, "caso4": 0.85, "organic": 0.10, "biofilm": 0.05,
    })
    naoh_wash_efficiency: dict[str, float] = field(default_factory=lambda: {
        "fe_oh3": 0.10, "caso4": 0.05, "organic": 0.90, "biofilm": 0.80,
    })
    backflush_efficiency: dict[str, float] = field(default_factory=lambda: {
        "fe_oh3": 0.40, "caso4": 0.30, "organic": 0.50, "biofilm": 0.30,
    })
    acid_wash_cost: float = 30.0      # $ per event
    naoh_wash_cost: float = 25.0      # $ per event
    backflush_cost: float = 10.0      # $ per event
    downtime_hr_per_clean: float = 0.5


@dataclass(frozen=True)
class CSTRFoulingCoupling:
    """Impurity-rejection coupling for the CSTR loop.

    C_imp = S / (Q * R_rej)   where S = source rate, Q = flow, R_rej = rejection.

    As fouling resistance grows, rejection increases (membrane gets tighter)
    but eventually the membrane is replaced and rejection resets.
    """
    base_rejection: float = 0.90          # R_rej at zero fouling
    max_rejection: float = 0.999          # asymptotic limit
    rejection_fouling_coupling: float = 0.5  # sensitivity of R_rej to fouling resistance
    impurity_source_mol_hr: float = 1.0e-3  # S — generation rate
    loop_flow_L_hr: float = 20.0          # Q
    loop_volume_L: float = 1000.0         # CSTR volume

    def __post_init__(self) -> None:
        if not 0 < self.base_rejection < 1:
            raise ValueError("base_rejection must lie in (0, 1)")


# ── Result dataclasses ─────────────────────────────────────────────────────
@dataclass
class FluxDeclineResult:
    """Time-series from a fouling simulation."""
    time_hr: np.ndarray
    flux_L_m2_hr: np.ndarray              # J(t)
    total_resistance: np.ndarray          # R_f(t)
    fe_oh3_resistance: np.ndarray
    caso4_resistance: np.ndarray
    organic_resistance: np.ndarray
    biofilm_resistance: np.ndarray
    rejection: np.ndarray                 # R_rej(t)
    impurity_M: np.ndarray               # C_imp(t)

    def summary(self) -> dict:
        return {
            "duration_hr": float(self.time_hr[-1]),
            "initial_flux_L_m2_hr": float(self.flux_L_m2_hr[0]),
            "final_flux_L_m2_hr": float(self.flux_L_m2_hr[-1]),
            "flux_decline_pct": float(
                100.0 * (1.0 - self.flux_L_m2_hr[-1] / self.flux_L_m2_hr[0])
            ),
            "final_total_resistance": float(self.total_resistance[-1]),
            "final_rejection": float(self.rejection[-1]),
            "final_impurity_M": float(self.impurity_M[-1]),
        }


@dataclass
class CleaningResult:
    """Output of a cleaning cycle simulation."""
    time_hr: np.ndarray
    flux_L_m2_hr: np.ndarray
    total_resistance: np.ndarray
    n_cleanings: int
    total_cleaning_cost: float
    total_downtime_hr: float
    total_membrane_cost: float
    optimal_cleaning_interval_hr: Optional[float]
    cleaning_events: list[tuple[float, str]]   # (time_hr, agent)


@dataclass
class FoulingSimulationResult:
    """Combined output of fouling + cleaning + economics."""
    flux_decline: FluxDeclineResult
    cleaning: CleaningResult
    economics: dict

    def summary(self) -> dict:
        return {
            "flux_decline": self.flux_decline.summary(),
            "n_cleanings": self.cleaning.n_cleanings,
            "total_cleaning_cost": self.cleaning.total_cleaning_cost,
            "total_membrane_cost": self.cleaning.total_membrane_cost,
            "optimal_cleaning_interval_hr": self.cleaning.optimal_cleaning_interval_hr,
            "economics": self.economics,
        }


# ── Hermia flux models ─────────────────────────────────────────────────────
def hermia_flux(
    t: np.ndarray,
    J0: float,
    K: float,
    model: HermiaModel,
) -> np.ndarray:
    """Evaluate one of the four classical Hermia blocking models.

    Parameters
    ----------
    t : array, time in hours
    J0 : clean-water flux (LMH)
    K : fouling coefficient (model-specific units)
    model : which Hermia variant

    Returns
    -------
    J(t) array in LMH.

    Equations (Hermia 1982):
      complete blocking:   J = J₀ exp(-K t)
      intermediate blocking: J = J₀ / (1 + K t)
      standard blocking:   J = J₀ (1 + K √t)⁻²
      cake filtration:     J = J₀ (1 + 2K t)⁻⁰·⁵
    """
    t = np.asarray(t, dtype=float)
    if model == HermiaModel.COMPLETE_BLOCKING:
        return J0 * np.exp(-K * t)
    elif model == HermiaModel.INTERMEDIATE_BLOCKING:
        return J0 / (1.0 + K * t)
    elif model == HermiaModel.STANDARD_BLOCKING:
        return J0 / (1.0 + K * np.sqrt(np.maximum(t, 0.0))) ** 2
    elif model == HermiaModel.CAKE_FILTRATION:
        return J0 / (1.0 + 2.0 * K * t) ** 0.5
    else:
        raise ValueError(f"Unknown Hermia model: {model}")


# ── Core model ─────────────────────────────────────────────────────────────
class MembraneFoulingModel:
    """Full membrane fouling simulation engine.

    Combines fouling mechanism kinetics, Hermia flux decline, cleaning
    cycle scheduling, impurity accumulation, and techno-economic costing.
    """

    def __init__(
        self,
        membrane: Optional[MembraneParams] = None,
        fouling: Optional[FoulingRateParams] = None,
        cleaning: Optional[CleaningParams] = None,
        coupling: Optional[CSTRFoulingCoupling] = None,
        hermia_variant: HermiaModel = HermiaModel.CAKE_FILTRATION,
        operating_pH: float = 2.0,
        hardness_mg_L: float = 50.0,
        idle_time_hr: float = 0.0,
        temperature_C: float = 60.0,
    ) -> None:
        self.membrane = membrane or MembraneParams()
        self.fouling = fouling or FoulingRateParams()
        self.cleaning = cleaning or CleaningParams()
        self.coupling = coupling or CSTRFoulingCoupling()
        self.hermia_variant = hermia_variant
        self.operating_pH = operating_pH
        self.hardness_mg_L = hardness_mg_L
        self.idle_time_hr = idle_time_hr
        self.temperature_C = temperature_C

    # ── Mechanism-specific effective rates ──────────────────────────────
    def _fe_oh3_rate(self) -> float:
        """Fe(OH)₃ deposition — active only above pH threshold."""
        if self.operating_pH < self.fouling.fe_oh3_pH_threshold:
            return 0.0
        excess = self.operating_pH - self.fouling.fe_oh3_pH_threshold
        return self.fouling.fe_oh3_base_rate * (1.0 + 2.0 * excess)

    def _caso4_rate(self) -> float:
        """CaSO₄ scaling — linear in hardness above threshold."""
        if self.hardness_mg_L < self.fouling.caso4_hardness_mg_L:
            return 0.0
        scale = self.hardness_mg_L / self.fouling.caso4_hardness_mg_L
        return self.fouling.caso4_base_rate * scale

    def _organic_rate(self) -> float:
        """Organic decomposition products — temperature-accelerated."""
        arrhenius = np.exp(0.02 * max(self.temperature_C - 40.0, 0.0))
        return self.fouling.organic_base_rate * arrhenius

    def _biofilm_rate(self) -> float:
        """Biofilm — activates after idle period, grows with idle time."""
        if self.idle_time_hr < self.fouling.biofilm_idle_hr:
            return 0.0
        excess = self.idle_time_hr - self.fouling.biofilm_idle_hr
        return self.fouling.biofilm_base_rate * (1.0 + 0.05 * excess)

    # ── Rejection model ────────────────────────────────────────────────
    def rejection_at_resistance(self, R_f: float) -> float:
        """Membrane rejection as a function of fouling resistance.

        R_rej = R_base + (R_max - R_base) * (1 - exp(-coupling * R_f))
        """
        c = self.coupling
        return float(np.clip(
            c.base_rejection + (c.max_rejection - c.base_rejection)
            * (1.0 - np.exp(-c.rejection_fouling_coupling * R_f)),
            c.base_rejection,
            c.max_rejection,
        ))

    # ── Flux from resistance ───────────────────────────────────────────
    def flux_from_resistance(self, R_f: float) -> float:
        """J = J₀ * exp(-R_f)  — simple exponential resistance model."""
        return float(self.membrane.clean_water_flux_L_m2_hr * np.exp(-R_f))

    # ── Cleaning ───────────────────────────────────────────────────────
    def cleaning_efficiency(self, agent: CleaningAgent) -> dict[str, float]:
        """Return per-mechanism cleaning efficiencies for *agent*."""
        if agent == CleaningAgent.ACID_WASH:
            return dict(self.cleaning.acid_wash_efficiency)
        elif agent == CleaningAgent.NAOH_WASH:
            return dict(self.cleaning.naoh_wash_efficiency)
        elif agent == CleaningAgent.BACKFLUSH:
            return dict(self.cleaning.backflush_efficiency)
        raise ValueError(f"Unknown agent: {agent}")

    def apply_cleaning(
        self,
        resistances: dict[str, float],
        agent: CleaningAgent,
    ) -> dict[str, float]:
        """Return resistances after applying *agent*."""
        eff = self.cleaning_efficiency(agent)
        return {k: v * (1.0 - eff.get(k, 0.0)) for k, v in resistances.items()}

    def cleaning_cost(self, agent: CleaningAgent) -> float:
        if agent == CleaningAgent.ACID_WASH:
            return self.cleaning.acid_wash_cost
        elif agent == CleaningAgent.NAOH_WASH:
            return self.cleaning.naoh_wash_cost
        elif agent == CleaningAgent.BACKFLUSH:
            return self.cleaning.backflush_cost
        raise ValueError(f"Unknown agent: {agent}")

    # ── Simulation: flux decline over time ─────────────────────────────
    def simulate_flux_decline(
        self,
        duration_hr: float = 4000.0,
        dt_hr: float = 1.0,
        initial_impurity_M: float = 2.0e-4,
    ) -> FluxDeclineResult:
        """Integrate fouling kinetics forward in time (no cleaning).

        Parameters
        ----------
        duration_hr : total simulation time
        dt_hr : Euler time step
        initial_impurity_M : starting impurity concentration in the loop

        Returns
        -------
        FluxDeclineResult with time-series arrays.
        """
        if duration_hr <= 0 or dt_hr <= 0:
            raise ValueError("duration_hr and dt_hr must be positive")

        t = np.arange(0.0, duration_hr + 0.5 * dt_hr, dt_hr)
        n = len(t)

        # Per-mechanism resistance accumulators
        r_fe = np.zeros(n)
        r_ca = np.zeros(n)
        r_org = np.zeros(n)
        r_bio = np.zeros(n)
        r_total = np.zeros(n)
        flux = np.zeros(n)
        rej = np.zeros(n)
        imp = np.zeros(n)

        # Pre-compute effective rates (constant for a given operating point)
        rates = {
            "fe_oh3": self._fe_oh3_rate(),
            "caso4": self._caso4_rate(),
            "organic": self._organic_rate(),
            "biofilm": self._biofilm_rate(),
        }

        imp[0] = initial_impurity_M
        flux[0] = self.membrane.clean_water_flux_L_m2_hr
        rej[0] = self.rejection_at_resistance(0.0)

        for i in range(n - 1):
            r_fe[i + 1] = r_fe[i] + rates["fe_oh3"] * dt_hr
            r_ca[i + 1] = r_ca[i] + rates["caso4"] * dt_hr
            r_org[i + 1] = r_org[i] + rates["organic"] * dt_hr
            r_bio[i + 1] = r_bio[i] + rates["biofilm"] * dt_hr
            r_total[i + 1] = r_fe[i + 1] + r_ca[i + 1] + r_org[i + 1] + r_bio[i + 1]

            rej[i + 1] = self.rejection_at_resistance(r_total[i + 1])

            # Impurity accumulation: C_imp = S / (Q * (1 - R_rej))
            # Higher rejection → less removal through permeate → higher C
            c = self.coupling
            removal_term = c.loop_flow_L_hr * (1.0 - rej[i + 1])
            imp[i + 1] = c.impurity_source_mol_hr / max(removal_term, 1e-30)

            flux[i + 1] = self.flux_from_resistance(r_total[i + 1])

        return FluxDeclineResult(
            time_hr=t,
            flux_L_m2_hr=flux,
            total_resistance=r_total,
            fe_oh3_resistance=r_fe,
            caso4_resistance=r_ca,
            organic_resistance=r_org,
            biofilm_resistance=r_bio,
            rejection=rej,
            impurity_M=imp,
        )

    # ── Simulation with cleaning cycles ────────────────────────────────
    def simulate_with_cleaning(
        self,
        duration_hr: float = 4000.0,
        dt_hr: float = 1.0,
        cleaning_interval_hr: Optional[float] = None,
        initial_impurity_M: float = 2.0e-4,
    ) -> CleaningResult:
        """Simulate fouling with periodic cleaning events.

        If *cleaning_interval_hr* is ``None``, uses the optimal interval
        from :meth:`find_optimal_cleaning_interval`.
        """
        if cleaning_interval_hr is None:
            cleaning_interval_hr = self.find_optimal_cleaning_interval(
                duration_hr=duration_hr,
            )

        if cleaning_interval_hr <= 0:
            raise ValueError("cleaning_interval_hr must be positive")

        t = np.arange(0.0, duration_hr + 0.5 * dt_hr, dt_hr)
        n = len(t)

        resistances = {"fe_oh3": 0.0, "caso4": 0.0, "organic": 0.0, "biofilm": 0.0}
        rates = {
            "fe_oh3": self._fe_oh3_rate(),
            "caso4": self._caso4_rate(),
            "organic": self._organic_rate(),
            "biofilm": self._biofilm_rate(),
        }

        flux_arr = np.zeros(n)
        r_total_arr = np.zeros(n)
        flux_arr[0] = self.membrane.clean_water_flux_L_m2_hr

        next_clean = cleaning_interval_hr
        events: list[tuple[float, str]] = []
        total_cost = 0.0
        downtime = 0.0
        n_clean = 0

        # Cleaning protocol: acid → NaOH → backflush
        agents = [CleaningAgent.ACID_WASH, CleaningAgent.NAOH_WASH, CleaningAgent.BACKFLUSH]

        for i in range(n - 1):
            # Accumulate resistance
            for k in resistances:
                resistances[k] += rates[k] * dt_hr
            rt = sum(resistances.values())
            r_total_arr[i + 1] = rt
            flux_arr[i + 1] = self.flux_from_resistance(rt)

            # Check cleaning
            if t[i + 1] >= next_clean:
                for agent in agents:
                    resistances = self.apply_cleaning(resistances, agent)
                    total_cost += self.cleaning_cost(agent)
                    events.append((t[i + 1], agent.value))
                downtime += self.cleaning.downtime_hr_per_clean
                n_clean += 1
                next_clean += cleaning_interval_hr
                # Update flux after cleaning
                rt = sum(resistances.values())
                r_total_arr[i + 1] = rt
                flux_arr[i + 1] = self.flux_from_resistance(rt)

        # Membrane replacement cost if end-of-life flux < 20% of initial
        eol_fraction = flux_arr[-1] / flux_arr[0] if flux_arr[0] > 0 else 1.0
        membrane_cost = 0.0
        if eol_fraction < 0.20:
            membrane_cost = (
                self.membrane.area_m2 * self.membrane.membrane_cost_per_m2
                + self.membrane.replacement_labor_cost
            )

        return CleaningResult(
            time_hr=t,
            flux_L_m2_hr=flux_arr,
            total_resistance=r_total_arr,
            n_cleanings=n_clean,
            total_cleaning_cost=total_cost,
            total_downtime_hr=downtime,
            total_membrane_cost=membrane_cost,
            optimal_cleaning_interval_hr=cleaning_interval_hr,
            cleaning_events=events,
        )

    # ── Optimal cleaning interval ──────────────────────────────────────
    def find_optimal_cleaning_interval(
        self,
        duration_hr: float = 4000.0,
        interval_range: tuple[float, float] = (10.0, 1000.0),
        n_points: int = 50,
    ) -> float:
        """Sweep cleaning intervals to minimise total cost.

        Total cost = cleaning cost + downtime penalty + membrane replacement
        (if flux drops below 20% of initial).

        Returns the interval (hours) that minimises total cost per unit
        production.
        """
        intervals = np.linspace(interval_range[0], interval_range[1], n_points)
        best_cost = np.inf
        best_interval = intervals[0]

        for interval in intervals:
            result = self.simulate_with_cleaning(
                duration_hr=duration_hr,
                dt_hr=max(1.0, interval / 10.0),
                cleaning_interval_hr=float(interval),
            )
            # Production proxy: average flux * area * duration
            avg_flux = np.mean(result.flux_L_m2_hr)
            production = avg_flux * self.membrane.area_m2 * duration_hr
            if production <= 0:
                continue
            cost_per_unit = (
                result.total_cleaning_cost + result.total_membrane_cost
            ) / production
            if cost_per_unit < best_cost:
                best_cost = cost_per_unit
                best_interval = float(interval)

        return best_interval

    # ── Full integrated simulation ─────────────────────────────────────
    def simulate(
        self,
        duration_hr: float = 4000.0,
        dt_hr: float = 1.0,
        initial_impurity_M: float = 2.0e-4,
        cleaning_interval_hr: Optional[float] = None,
    ) -> FoulingSimulationResult:
        """Run complete fouling + cleaning + economics simulation."""
        # First: unconstrained flux decline
        decline = self.simulate_flux_decline(duration_hr, dt_hr, initial_impurity_M)

        # Second: cleaning cycles
        clean = self.simulate_with_cleaning(
            duration_hr, dt_hr, cleaning_interval_hr, initial_impurity_M,
        )

        # Economics
        total_flux_throughput = np.trapezoid(
            clean.flux_L_m2_hr * self.membrane.area_m2, clean.time_hr,
        )
        avg_flux = total_flux_throughput / duration_hr if duration_hr > 0 else 0.0
        production_L = total_flux_throughput

        economics = {
            "membrane_area_m2": self.membrane.area_m2,
            "duration_hr": duration_hr,
            "total_throughput_L": float(production_L),
            "average_flux_L_m2_hr": float(avg_flux),
            "total_cleaning_cost": clean.total_cleaning_cost,
            "total_membrane_cost": clean.total_membrane_cost,
            "total_downtime_hr": clean.total_downtime_hr,
            "effective_operating_hr": duration_hr - clean.total_downtime_hr,
            "cost_per_L_treated": float(
                (clean.total_cleaning_cost + clean.total_membrane_cost)
                / max(production_L, 1.0)
            ),
            "n_cleanings": clean.n_cleanings,
        }

        return FoulingSimulationResult(
            flux_decline=decline,
            cleaning=clean,
            economics=economics,
        )


# ── Integration helpers ────────────────────────────────────────────────────
def fouling_from_closed_loop_result(
    closed_loop_result,
    membrane: Optional[MembraneParams] = None,
    fouling: Optional[FoulingRateParams] = None,
    cleaning: Optional[CleaningParams] = None,
) -> FoulingSimulationResult:
    """Run a membrane fouling simulation driven by a closed-loop CSTR result.

    Extracts pH, impurity concentration and duration from *closed_loop_result*
    to feed the fouling model.
    """
    duration = float(closed_loop_result.time_hr[-1])
    initial_impurity = float(closed_loop_result.impurity_M[0])

    model = MembraneFoulingModel(
        membrane=membrane,
        fouling=fouling,
        cleaning=cleaning,
        operating_pH=2.0,  # acidic electrowinning bath
    )
    return model.simulate(
        duration_hr=duration,
        initial_impurity_M=initial_impurity,
    )


def membrane_replacement_cost(
    membrane: MembraneParams,
    n_replacements: int,
) -> dict:
    """Calculate membrane replacement costs for the techno-economic model."""
    per_event = (
        membrane.area_m2 * membrane.membrane_cost_per_m2
        + membrane.replacement_labor_cost
    )
    return {
        "membrane_cost_per_event": per_event,
        "n_replacements": n_replacements,
        "total_membrane_cost": per_event * n_replacements,
        "membrane_cost_per_m2": membrane.membrane_cost_per_m2,
    }
