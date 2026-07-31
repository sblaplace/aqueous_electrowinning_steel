"""
Steel grade specification and post-processing route selection.

Maps a target steel grade (AISI/SAE designation or custom composition) to
the post-processing steps needed to get there from electrodeposited iron.

Two routes:
  - "carburize": deposit pure Fe sheet, then gas/plasma carburize
  - "codeposit": co-deposit carbon particles during electrodeposition
  - "none": pure iron (no carbon addition)

The route determines what equipment the dark mill needs:
  - carburize → batch furnace + gas supply + quench tank
  - codeposit → particle suspension system + filtration loop
  - none → nothing extra

All screening — real grade mapping needs combustion analysis and hardness
traverses for calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Literal

from .carburization import (
    CarburizationModel, CarburizationParams,
    estimate_carburizing_time_for_case_depth, carbon_diffusivity_m2_s,
)
from .carbon_potential import (
    carbon_potential_summary, carbon_wt_from_activity,
    austenite_max_carbon_wt_percent,
)


# ─── Grade Definitions ────────────────────────────────────────────

@dataclass(frozen=True)
class SteelGradeSpec:
    """Target steel grade specification."""

    name: str                          # e.g. "AISI 1040", "AISI 8620"
    c_wt_percent_target: float         # target carbon (wt%)
    c_wt_percent_tolerance: float = 0.05  # acceptable deviation

    # Alloying elements (wt%, 0 = not present)
    ni_wt_percent: float = 0.0
    cr_wt_percent: float = 0.0
    mn_wt_percent: float = 0.0
    si_wt_percent: float = 0.0
    mo_wt_percent: float = 0.0

    # Classification
    category: str = "carbon"           # "carbon", "alloy", "tool", "stainless"
    aisi_series: str = "10xx"          # AISI/SAE series

    # Processing hints
    needs_hardening: bool = False      # quench + temper
    case_hardened: bool = False        # surface hardening only
    notes: str = ""

    @property
    def is_plain_carbon(self) -> bool:
        return self.ni_wt_percent == 0 and self.cr_wt_percent == 0

    @property
    def is_alloy(self) -> bool:
        return self.ni_wt_percent > 0 or self.cr_wt_percent > 0

    @property
    def total_alloy_wt_percent(self) -> float:
        return (self.ni_wt_percent + self.cr_wt_percent +
                self.mn_wt_percent + self.si_wt_percent + self.mo_wt_percent)


# ─── Pre-defined Grades ───────────────────────────────────────────

STEEL_GRADES: Dict[str, SteelGradeSpec] = {
    # Pure iron
    "pure_iron": SteelGradeSpec(
        name="Pure Iron (EW grade)",
        c_wt_percent_target=0.02,
        c_wt_percent_tolerance=0.02,
        category="pure",
        aisi_series="---",
        notes="As-deposited electrodeposited iron. C from bath impurities only.",
    ),

    # Plain carbon steels (10xx series)
    "AISI_1008": SteelGradeSpec(
        name="AISI 1008 — Extra-low carbon",
        c_wt_percent_target=0.10,
        c_wt_percent_tolerance=0.03,
        category="carbon",
        aisi_series="10xx",
        notes="Welding grade. Minimal hardening.",
    ),
    "AISI_1010": SteelGradeSpec(
        name="AISI 1010 — Low carbon",
        c_wt_percent_target=0.10,
        c_wt_percent_tolerance=0.03,
        category="carbon",
        aisi_series="10xx",
    ),
    "AISI_1018": SteelGradeSpec(
        name="AISI 1018 — Low carbon, general purpose",
        c_wt_percent_target=0.18,
        c_wt_percent_tolerance=0.03,
        category="carbon",
        aisi_series="10xx",
        notes="Most common general-purpose steel. Good weldability.",
    ),
    "AISI_1040": SteelGradeSpec(
        name="AISI 1040 — Medium carbon",
        c_wt_percent_target=0.40,
        c_wt_percent_tolerance=0.05,
        category="carbon",
        aisi_series="10xx",
        needs_hardening=True,
        notes="Shafts, axles. Heat-treatable.",
    ),
    "AISI_1045": SteelGradeSpec(
        name="AISI 1045 — Medium carbon",
        c_wt_percent_target=0.45,
        c_wt_percent_tolerance=0.05,
        category="carbon",
        aisi_series="10xx",
        needs_hardening=True,
    ),
    "AISI_1060": SteelGradeSpec(
        name="AISI 1060 — High carbon",
        c_wt_percent_target=0.60,
        c_wt_percent_tolerance=0.05,
        category="carbon",
        aisi_series="10xx",
        needs_hardening=True,
        notes="Springs, hand tools.",
    ),
    "AISI_1080": SteelGradeSpec(
        name="AISI 1080 — High carbon",
        c_wt_percent_target=0.80,
        c_wt_percent_tolerance=0.05,
        category="carbon",
        aisi_series="10xx",
        needs_hardening=True,
        notes="Near-eutectoid. Knife blades, music wire.",
    ),
    "AISI_1095": SteelGradeSpec(
        name="AISI 1095 — Very high carbon",
        c_wt_percent_target=0.95,
        c_wt_percent_tolerance=0.05,
        category="tool",
        aisi_series="10xx",
        needs_hardening=True,
        notes="Spring steel, knives. Maximum hardness ~64 HRC.",
    ),

    # Case-hardening steels (carburizing grades)
    "AISI_1018_case": SteelGradeSpec(
        name="AISI 1018 — Case hardened",
        c_wt_percent_target=0.18,
        c_wt_percent_tolerance=0.03,
        category="carbon",
        aisi_series="10xx",
        case_hardened=True,
        notes="Low-C core, high-C case. Gears, pins.",
    ),
    "AISI_8620": SteelGradeSpec(
        name="AISI 8620 — Ni-Cr-Mo case hardening",
        c_wt_percent_target=0.20,
        c_wt_percent_tolerance=0.03,
        ni_wt_percent=0.55,
        cr_wt_percent=0.50,
        mo_wt_percent=0.20,
        category="alloy",
        aisi_series="86xx",
        case_hardened=True,
        notes="Premium case-hardening grade. Requires Ni/Cr co-deposition.",
    ),
}


# ─── Route Selection ──────────────────────────────────────────────

PostProcessingRoute = Literal["none", "carburize", "codeposit"]


def select_route(grade: SteelGradeSpec) -> PostProcessingRoute:
    """
    Select the best post-processing route for a target grade.

    Decision logic:
    - Pure iron (C < 0.05%) → "none"
    - Plain carbon steel → "carburize" (well-understood, gas atmosphere control)
    - Alloy steel with Ni/Cr → "codeposit" (need in-situ alloying)
    - Case-hardened → "carburize" (gradient is the point)
    """
    if grade.c_wt_percent_target < 0.05 and grade.total_alloy_wt_percent < 0.1:
        return "none"

    if grade.is_alloy:
        # Need Ni, Cr, etc. in the deposit — must co-deposit
        return "codeposit"

    # Plain carbon steel — carburize is simpler and better understood
    return "carburize"


# ─── Post-Processing Sizing ───────────────────────────────────────

@dataclass
class CarburizationSizing:
    """Sized carburization furnace parameters."""
    temperature_C: float
    surface_carbon_wt_percent: float
    duration_hr: float
    case_depth_um: float              # 0 if through-hardening
    furnace_length_mm: float
    furnace_width_mm: float
    furnace_height_mm: float
    furnace_power_kW: float
    gas_flow_m3_hr: float
    quench_tank_volume_L: float
    energy_kWh_per_t_Fe: float
    batch_time_hr: float              # total cycle (heat + soak + quench + cool)
    notes: str = ""

    @property
    def furnace_volume_m3(self) -> float:
        return (self.furnace_length_mm * self.furnace_width_mm *
                self.furnace_height_mm / 1e9)


@dataclass
class CoDepositionSizing:
    """Sized co-deposition system parameters."""
    carbon_loading_g_L: float         # particle loading in bath
    ni_loading_M: float               # NiSO4 concentration for alloy grades
    particle_size_um: float
    filtration_rate_L_min: float
    particle_hopper_volume_L: float
    suspension_power_kW: float        # for agitation/ultrasonics
    energy_kWh_per_t_Fe: float
    notes: str = ""


@dataclass
class PostProcessingResult:
    """Complete post-processing sizing for a target grade."""
    route: PostProcessingRoute
    grade: SteelGradeSpec
    carburization: Optional[CarburizationSizing] = None
    codeposition: Optional[CoDepositionSizing] = None

    @property
    def total_energy_kWh_per_t(self) -> float:
        if self.carburization:
            return self.carburization.energy_kWh_per_t_Fe
        if self.codeposition:
            return self.codeposition.energy_kWh_per_t_Fe
        return 0.0

    @property
    def additional_capex_fraction(self) -> float:
        """Rough fraction of base CAPEX for post-processing equipment."""
        if self.carburization:
            return 0.15  # furnace + gas + quench ~15% of cell CAPEX
        if self.codeposition:
            return 0.08  # particle system + filtration ~8%
        return 0.0

    def summary(self) -> str:
        lines = [
            f"POST-PROCESSING: {self.grade.name}",
            f"  Route: {self.route}",
            f"  Target C: {self.grade.c_wt_percent_target:.2f} wt%",
        ]
        if self.carburization:
            c = self.carburization
            lines.extend([
                f"  Furnace: {c.temperature_C:.0f}°C, {c.duration_hr:.1f} hr soak",
                f"  Surface C: {c.surface_carbon_wt_percent:.2f}%",
                f"  Case depth: {c.case_depth_um:.0f} µm" if c.case_depth_um > 0
                    else "  Through-hardening (1mm sheet)",
                f"  Cycle time: {c.batch_time_hr:.1f} hr",
                f"  Energy: {c.energy_kWh_per_t_Fe:.0f} kWh/t Fe",
                f"  Furnace: {c.furnace_length_mm:.0f} × {c.furnace_width_mm:.0f} × {c.furnace_height_mm:.0f} mm",
                f"  Power: {c.furnace_power_kW:.1f} kW",
                f"  Quench tank: {c.quench_tank_volume_L:.0f} L",
            ])
        if self.codeposition:
            c = self.codeposition
            lines.extend([
                f"  Carbon loading: {c.carbon_loading_g_L:.1f} g/L",
                f"  Particle size: {c.particle_size_um:.1f} µm",
                f"  Energy: {c.energy_kWh_per_t_Fe:.0f} kWh/t Fe",
            ])
        if self.total_energy_kWh_per_t == 0:
            lines.append("  No post-processing needed")
        return "\n".join(lines)


def size_post_processing(
    grade: SteelGradeSpec,
    route: Optional[PostProcessingRoute] = None,
    sheet_thickness_um: float = 1000.0,
    annual_production_t: float = 1000.0,
) -> PostProcessingResult:
    """
    Size post-processing equipment for a target steel grade.

    If route is None, auto-selects based on grade composition.
    """
    if route is None:
        route = select_route(grade)

    if route == "none":
        return PostProcessingResult(route=route, grade=grade)

    if route == "carburize":
        return _size_carburization(grade, sheet_thickness_um, annual_production_t)
    elif route == "codeposit":
        return _size_codeposition(grade, annual_production_t)
    else:
        raise ValueError(f"Unknown route: {route}")


def _size_carburization(
    grade: SteelGradeSpec,
    sheet_thickness_um: float,
    annual_production_t: float,
) -> PostProcessingResult:
    """Size a batch carburization furnace for the target grade."""

    target_C = grade.c_wt_percent_target
    initial_C = 0.02  # as-deposited

    # Determine carburizing temperature (austenite region for carbon dissolution)
    if target_C <= 0.25:
        temperature_C = 880.0
    elif target_C <= 0.50:
        temperature_C = 900.0
    elif target_C <= 0.80:
        temperature_C = 920.0
    else:
        temperature_C = 930.0

    # Surface carbon potential: slightly above target for driving force
    # For through-hardening, surface must be near target
    # For case hardening, surface is high (~1.1-1.3%)
    if grade.case_hardened:
        surface_C = 1.10
        # Case depth: typical 0.5-1.5 mm for gears/bearings
        case_depth_um = 500.0
        duration_hr = estimate_carburizing_time_for_case_depth(
            case_depth_um, temperature_C, surface_C,
        )
    else:
        # Through-hardening: need uniform C at target through 1mm sheet
        # Use finite-slab model to find time where midplane reaches target
        surface_C = min(target_C * 1.05, austenite_max_carbon_wt_percent(temperature_C) * 0.95)
        case_depth_um = 0.0  # not applicable

        model = CarburizationModel(CarburizationParams(
            temperature_C=temperature_C,
            surface_carbon_wt_percent=surface_C,
            initial_carbon_wt_percent=initial_C,
            sheet_thickness_um=sheet_thickness_um,
        ))
        # Find time for midplane to reach 90% of target
        target_mid_C = target_C * 0.90
        for t_hr in [0.5, 1, 2, 4, 8, 12, 16, 24]:
            profile = model.profile_at_time(t_hr)
            mid_C = profile.c_wt_percent[len(profile.c_wt_percent) // 2]
            if mid_C >= target_mid_C:
                duration_hr = t_hr
                break
        else:
            duration_hr = 24.0  # cap

    # Furnace sizing: batch capacity for one day's production
    # Assume 2 batches/day for through-hardening, 3 for case-hardening
    batches_per_day = 3 if grade.case_hardened else 2
    daily_production_kg = annual_production_t * 1000.0 / 365.0
    batch_mass_kg = daily_production_kg / batches_per_day

    # Furnace dimensions: rectangular batch furnace
    # Stack sheet on trays, ~500 kg/m² floor loading
    floor_area_m2 = batch_mass_kg / 500.0
    furnace_length_mm = max(1200.0, (floor_area_m2 ** 0.5) * 1000 * 1.5)
    furnace_width_mm = max(800.0, furnace_length_mm * 0.7)
    furnace_height_mm = max(1000.0, sheet_thickness_um / 1000 * 20 + 400)  # trays + clearance

    # Furnace power: ~50 kW/m³ for batch, plus soak hold
    furnace_volume_m3 = furnace_length_mm * furnace_width_mm * furnace_height_mm / 1e9
    furnace_power_kW = furnace_volume_m3 * 50.0 + batch_mass_kg * 0.01  # rough

    # Gas flow: endothermic carrier + enriching gas, ~5-15 m³/hr per m²
    gas_flow_m3_hr = floor_area_m2 * 10.0

    # Quench tank: oil or water, ~2x batch volume
    quench_tank_volume_L = batch_mass_kg / 7.874 * 2 * 1000  # 2x steel volume in L

    # Energy: heating + soak + quench
    # Heating: Q = m * Cp * dT / efficiency
    dT = temperature_C - 25.0
    heating_energy_kWh = batch_mass_kg * 449.0 * dT / 3.6e6 / 0.60  # 60% efficiency
    soak_energy_kWh = furnace_power_kW * 0.3 * duration_hr  # 30% hold power
    total_cycle_energy_kWh = heating_energy_kWh + soak_energy_kWh
    # Per tonne of annual production
    cycles_per_year = batches_per_day * 365
    annual_energy_kWh = total_cycle_energy_kWh * cycles_per_year
    energy_per_t = annual_energy_kWh / annual_production_t

    # Total cycle time: heat-up + soak + quench + cool
    heatup_hr = dT / 200.0  # ~200°C/hr heating rate
    quench_hr = 0.5
    cool_hr = 2.0
    batch_time_hr = heatup_hr + duration_hr + quench_hr + cool_hr

    carburization = CarburizationSizing(
        temperature_C=temperature_C,
        surface_carbon_wt_percent=surface_C,
        duration_hr=duration_hr,
        case_depth_um=case_depth_um,
        furnace_length_mm=furnace_length_mm,
        furnace_width_mm=furnace_width_mm,
        furnace_height_mm=furnace_height_mm,
        furnace_power_kW=furnace_power_kW,
        gas_flow_m3_hr=gas_flow_m3_hr,
        quench_tank_volume_L=quench_tank_volume_L,
        energy_kWh_per_t_Fe=energy_per_t,
        batch_time_hr=batch_time_hr,
        notes=f"{'Case-hardening' if grade.case_hardened else 'Through-hardening'} "
              f"at {temperature_C}°C, {duration_hr:.1f}hr soak, {batches_per_day} batches/day",
    )

    return PostProcessingResult(
        route="carburize",
        grade=grade,
        carburization=carburization,
    )


def _size_codeposition(
    grade: SteelGradeSpec,
    annual_production_t: float,
) -> PostProcessingResult:
    """Size a co-deposition particle suspension system."""

    target_C = grade.c_wt_percent_target

    # Carbon loading in bath: empirical mapping from target C wt%
    # Guglielmi model: ~0.1-0.5 wt% C at 1 g/L, ~0.5-2 wt% at 5 g/L
    # Screening: loading (g/L) ≈ target_C * 5
    carbon_loading_g_L = max(0.5, target_C * 5.0)

    # Ni loading for alloy grades
    ni_loading_M = grade.ni_wt_percent / 100.0 * 5.0 if grade.ni_wt_percent > 0 else 0.0

    # Particle size: smaller = stronger dispersion hardening, harder to suspend
    particle_size_um = 1.5  # default fine carbon

    # Filtration: continuous bath filtration to remove agglomerates
    # ~10% of bath volume per minute
    bath_volume_L = annual_production_t * 0.02 * 1000  # ~20 L/t rough
    filtration_rate_L_min = bath_volume_L * 0.10 / 60.0

    # Particle hopper: 1-day supply
    daily_consumption_kg = annual_production_t * target_C / 100.0 / 365.0
    hopper_volume_L = daily_consumption_kg / 2.2 * 1000 * 3  # 3 days, bulk density ~2.2 g/cm³

    # Suspension power: ultrasonics + pump
    suspension_power_kW = max(0.5, bath_volume_L * 0.001)

    # Energy: suspension + filtration (small compared to electrolysis)
    energy_per_t = suspension_power_kW * 8000.0 / annual_production_t  # kWh/t

    codeposition = CoDepositionSizing(
        carbon_loading_g_L=carbon_loading_g_L,
        ni_loading_M=ni_loading_M,
        particle_size_um=particle_size_um,
        filtration_rate_L_min=filtration_rate_L_min,
        particle_hopper_volume_L=hopper_volume_L,
        suspension_power_kW=suspension_power_kW,
        energy_kWh_per_t_Fe=energy_per_t,
        notes=f"Carbon particles {carbon_loading_g_L:.1f} g/L in bath, "
              f"continuous filtration at {filtration_rate_L_min:.1f} L/min",
    )

    return PostProcessingResult(
        route="codeposit",
        grade=grade,
        codeposition=codeposition,
    )
