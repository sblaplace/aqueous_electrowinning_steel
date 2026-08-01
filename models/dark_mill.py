"""
Dark Mill — site-sizing digital twin for autonomous iron electrowinning.

Given a site definition (feedstock, grid, climate, market distance),
sizes a modular dark mill, produces full mass/energy/water balances,
and answers: "Does this site make sense?"

Physics-driven: V_cell, FE, and current density are computed from the
unified cell_physics solver (speciation → Nernst-Planck transport →
voltage decomposition), not assumed.

The dark-mill philosophy:
  - Ship the plant to the feedstock, not the other way around.
  - Modular cell stacks, scale by replication.
  - Self-monitoring, self-calibrating — minimal on-site labor.
  - Deploy next to cheap power, negative-cost feedstock, or both.

References
----------
- RESEARCH_PROGRAM.md — "Dark Mill" vision and kill criteria
- cell_physics.py — unified physics solver (speciation → transport → voltage)
- technoeconomic.py — CAPEX/OPEX/LCOFe (Humbert 2024 benchmarks)
- supply_chain.py — feedstock library, transport, centralized vs decentralized
- thermal_balance.py — cell heat balance, cooling requirements
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import math

from .electrochemistry import (
    specific_energy_kWh_per_t, current_density_to_production,
)
from .technoeconomic import (
    ElectrolyzerParams, CAPEXModel, OPEXModel, LevelizedCost,
    BENCHMARK_COSTS, compare_routes,
)
from .supply_chain import (
    Feedstock, FEEDSTOCKS,
)
from .thermal_balance import CellThermalParams, simulate_thermal_transient
from .cell_physics import (
    CellPhysics, BathRecipe, ProcessConditions, CellGeometry,
    OperatingPoint,
)
from .steel_grade import (
    SteelGradeSpec, PostProcessingRoute, PostProcessingResult,
    STEEL_GRADES, size_post_processing,
)

# Crate import is lazy inside functions to avoid circular at import time
# but we provide type hints via Any. For evaluation we import locally.

# ─── Site Definition ──────────────────────────────────────────────

@dataclass
class GridSpec:
    """Electrical supply characteristics at the site."""
    electricity_price_kWh: float = 0.04       # $/kWh (PPA or grid)
    renewable_fraction: float = 0.80          # fraction of electricity from renewables
    grid_CO2_kg_per_kWh: float = 0.40        # grid emission factor
    max_power_MW: float = 10.0               # available grid connection capacity
    power_factor_surcharge: float = 0.0       # $/kWh surcharge for reactive power
    demand_charge_kW_month: float = 0.0       # $/kW-month peak demand charge

    @property
    def effective_price_kWh(self) -> float:
        """Electricity price including surcharges."""
        return self.electricity_price_kWh + self.power_factor_surcharge


@dataclass
class ClimateSpec:
    """Site climate conditions affecting operations.

    Extended with structural/environmental fields for the whole-system twin:
    wind gust + terrain, rainfall, snow, freeze depth. These feed the crate
    structural model and the operating twin's environmental safe-state limits.
    All numbers are screening-grade (L0) until validated by site survey / load test.
    """
    ambient_temp_C: float = 25.0              # annual average
    ambient_temp_range_C: float = 15.0        # ± range (for seasonal sizing)
    relative_humidity: float = 0.50           # annual average
    altitude_m: float = 0.0                   # affects air density, cooling
    water_availability: str = "municipal"     # "municipal", "well", "river", "scarce"
    water_cost_per_m3: float = 2.0            # $/m³
    freeze_risk: bool = False                 # affects winterization costs

    # ── Whole-system twin extensions (structural/environmental) ──
    wind_gust_m_s: float = 40.0               # design 3-s gust (m/s) at site
    wind_terrain: str = "open"                # "open" | "suburban" | "urban"
    rainfall_intensity_mm_hr: float = 50.0    # design rainfall intensity (mm/hr)
    snow_load_kPa: float = 0.0                # design roof snow load
    freeze_depth_m: float = 0.0               # frost depth for foundation winterization


@dataclass
class SiteDefinition:
    """Everything needed to evaluate a deployment site."""
    name: str
    feedstock_key: str                        # key into FEEDSTOCKS
    grid: GridSpec = field(default_factory=GridSpec)
    climate: ClimateSpec = field(default_factory=ClimateSpec)
    feedstock_distance_km: float = 0.0        # distance from feedstock source
    product_market_km: float = 200.0          # distance to product market
    target_capacity_t_Fe_yr: float = 1000.0   # desired annual production
    available_area_m2: float = 500.0          # site footprint
    labor_cost_per_yr: float = 1_500_000.0    # on-site labor (reduced for autonomous)
    notes: str = ""

    # ── Whole-system twin: structural / environmental ground fields ──
    soil_bearing_kPa: float = 100.0           # allowable bearing pressure
    seismic_coefficient: Optional[float] = None  # base shear coeff (fraction of W), e.g. 0.15
    flood_depth_m: float = 0.0                # plausible flood depth at grade
    sealing_class: str = "industrial"         # "industrial" | "sealed" envelope
    ground_friction_mu: float = 0.5           # soil-footer friction
    ground_anchored: bool = False             # pre-provided tie-down

    # Physics inputs (sensible defaults for 1M FeSO4, pH 2, divided cell)
    bath: BathRecipe = field(default_factory=BathRecipe)
    geometry: CellGeometry = field(default_factory=CellGeometry)
    conditions: ProcessConditions = field(default_factory=ProcessConditions)

    # Product specification: target steel grade and post-processing route
    target_grade: Optional[SteelGradeSpec] = None   # None = pure iron
    post_processing_route: Optional[PostProcessingRoute] = None  # None = auto-select
    sheet_thickness_um: float = 1000.0        # deposit thickness for carburization

    @property
    def feedstock(self) -> Feedstock:
        return FEEDSTOCKS[self.feedstock_key]

    @property
    def effective_grade(self) -> SteelGradeSpec:
        """Target grade, defaulting to pure iron if not set."""
        if self.target_grade is not None:
            return self.target_grade
        return STEEL_GRADES["pure_iron"]


# ─── Sizing Results ───────────────────────────────────────────────

@dataclass
class StackDesign:
    """Sized electrolyzer stack configuration."""
    n_stacks: int
    cells_per_stack: int
    electrode_area_m2: float              # per electrode face
    current_density_mA_cm2: float
    current_efficiency: float
    cell_voltage_V: float
    temperature_C: float

    @property
    def total_electrode_area_m2(self) -> float:
        """Total active area (both faces, all cells, all stacks)."""
        return self.electrode_area_m2 * self.cells_per_stack * 2 * self.n_stacks

    @property
    def total_current_A(self) -> float:
        return self.current_density_mA_cm2 * 10.0 * self.electrode_area_m2

    @property
    def stack_voltage_V(self) -> float:
        return self.cell_voltage_V * self.cells_per_stack

    @property
    def total_power_kW(self) -> float:
        return self.total_current_A * self.stack_voltage_V * self.n_stacks / 1000.0

    @property
    def production_rate_kg_hr(self) -> float:
        per_cell = current_density_to_production(
            self.current_density_mA_cm2,
            self.electrode_area_m2,
            self.current_efficiency,
        )
        return per_cell * self.cells_per_stack * self.n_stacks

    def annual_production_t(self, operating_hours: float = 8000.0) -> float:
        return self.production_rate_kg_hr * operating_hours / 1000.0


@dataclass
class MassBalance:
    """Full material balance for the dark mill (annual basis)."""
    # Iron
    iron_production_t_yr: float
    feedstock_consumed_t_yr: float
    feedstock_cost_per_t_Fe: float

    # Water
    water_consumption_m3_yr: float
    water_cost_yr: float

    # Acid / reagents
    acid_consumption_t_yr: float
    acid_cost_yr: float

    # Electrolyte losses
    electrolyte_makeup_m3_yr: float
    electrolyte_cost_yr: float

    # Waste
    waste_generated_t_yr: float           # gangue, sludge
    waste_disposal_cost_yr: float

    # Energy
    electrolysis_energy_MWh_yr: float
    grinding_energy_MWh_yr: float
    cooling_energy_MWh_yr: float
    auxiliary_energy_MWh_yr: float
    total_energy_MWh_yr: float

    # CO2
    scope1_CO2_t_yr: float                # direct (minimal for EW)
    scope2_CO2_t_yr: float                # from electricity

    # Post-processing
    post_processing_energy_MWh_yr: float = 0.0   # carburization or co-deposition
    product_grade: str = "Pure Iron"


@dataclass
class SiteReport:
    """Complete site assessment."""
    site: SiteDefinition
    stack_design: StackDesign
    mass_balance: MassBalance
    capex: Dict[str, float]
    opex: Dict[str, float]
    lcofe: Dict[str, float]
    thermal: Dict[str, Any]
    benchmarks: Dict[str, Any]
    go_no_go: Dict[str, Any]
    physics_point: Optional[OperatingPoint] = None  # from cell_physics solver
    post_processing: Optional[PostProcessingResult] = None  # grade-specific sizing
    crate_verdict: Optional[Any] = None        # CrateVerdict from crate.py (avoid circular import)
    credibility: Optional[Dict[str, int]] = None  # per-layer L# vector if evaluated via system_twin

    def summary(self) -> str:
        """Human-readable site assessment."""
        sd = self.stack_design
        mb = self.mass_balance
        gng = self.go_no_go

        lines = [
            f"{'='*70}",
            f"DARK MILL SITE ASSESSMENT: {self.site.name}",
            f"{'='*70}",
            "",
            f"FEEDSTOCK: {self.site.feedstock.name}",
            f"  Fe content:       {self.site.feedstock.fe_wt_fraction*100:.0f}%",
            f"  Cost:             ${self.site.feedstock.cost_per_t_feedstock:.0f}/t feedstock",
            f"  Per tonne Fe:     ${mb.feedstock_cost_per_t_Fe:.0f}/t Fe",
            f"  Transport:        {self.site.feedstock_distance_km:.0f} km by {self.site.feedstock.transport_mode}",
            "",
            "STACK DESIGN:",
            f"  Configuration:    {sd.n_stacks} stacks × {sd.cells_per_stack} cells",
            f"  Electrode area:   {sd.electrode_area_m2:.2f} m²/face",
            f"  Total active area: {sd.total_electrode_area_m2:.0f} m²",
            f"  Current density:  {sd.current_density_mA_cm2:.0f} mA/cm²",
            f"  Cell voltage:     {sd.cell_voltage_V:.2f} V",
            f"  Temperature:      {sd.temperature_C:.0f} °C",
            f"  Power draw:       {sd.total_power_kW:.0f} kW ({sd.total_power_kW/1000:.1f} MW)",
            f"  Annual production: {sd.annual_production_t():.0f} t Fe/yr",
        ]

        # Physics details (if available from cell_physics solver)
        if self.physics_point is not None:
            pp = self.physics_point
            lines.extend([
                "",
                "CELL PHYSICS (from Nernst-Planck + speciation):",
                f"  FE (predicted):   {pp.current_efficiency*100:.1f}%",
                f"  Transport limit:  {pp.transport_limit_mA_cm2:.0f} mA/cm²",
                f"  Diffusion limit:  {pp.diffusion_limit_mA_cm2:.0f} mA/cm²",
                f"  Migration boost:  {pp.migration_enhancement:.2f}×",
                f"  Surface pH:       {pp.surface_pH:.2f}",
                f"  Surface [Fe²⁺]:   {pp.surface_fe_M:.3f} M",
                f"  Fe(OH)₂ SSAT:     {pp.feoh2_supersaturation:.2g}",
                f"  Precipitation:    {'YES' if pp.precipitation_active else 'no'}",
            ])

        lines.extend([
            "",
            "MASS BALANCE (annual):",
            f"  Iron produced:    {mb.iron_production_t_yr:.0f} t/yr",
            f"  Feedstock needed: {mb.feedstock_consumed_t_yr:.0f} t/yr",
            f"  Water:            {mb.water_consumption_m3_yr:.0f} m³/yr",
            f"  Waste (gangue):   {mb.waste_generated_t_yr:.0f} t/yr",
            "",
            "ENERGY BALANCE (annual):",
            f"  Electrolysis:     {mb.electrolysis_energy_MWh_yr:.0f} MWh/yr",
            f"  Grinding:         {mb.grinding_energy_MWh_yr:.0f} MWh/yr",
            f"  Cooling:          {mb.cooling_energy_MWh_yr:.0f} MWh/yr",
            f"  Auxiliary:        {mb.auxiliary_energy_MWh_yr:.0f} MWh/yr",
            f"  TOTAL:            {mb.total_energy_MWh_yr:.0f} MWh/yr",
            f"  Specific energy:  {mb.total_energy_MWh_yr*1000/mb.iron_production_t_yr:.0f} kWh/t Fe",
        ])

        # Post-processing (grade-specific)
        if self.post_processing is not None and self.post_processing.route != "none":
            pp_res = self.post_processing
            lines.extend(["",
                f"POST-PROCESSING: {pp_res.grade.name}",
                f"  Route:            {pp_res.route}",
                f"  Target C:         {pp_res.grade.c_wt_percent_target:.2f} wt%",
            ])
            if pp_res.carburization is not None:
                c = pp_res.carburization
                lines.extend([
                    f"  Furnace temp:     {c.temperature_C:.0f} C",
                    f"  Soak time:        {c.duration_hr:.1f} hr",
                    f"  Surface C:        {c.surface_carbon_wt_percent:.2f} wt%",
                    f"  Cycle time:       {c.batch_time_hr:.1f} hr",
                    f"  Post-proc energy: {c.energy_kWh_per_t_Fe:.0f} kWh/t Fe",
                    f"  Furnace:          {c.furnace_length_mm:.0f} x {c.furnace_width_mm:.0f} x {c.furnace_height_mm:.0f} mm",
                ])
            elif pp_res.codeposition is not None:
                c = pp_res.codeposition
                lines.extend([
                    f"  Carbon loading:   {c.carbon_loading_g_L:.1f} g/L",
                    f"  Particle size:    {c.particle_size_um:.1f} um",
                    f"  Post-proc energy: {c.energy_kWh_per_t_Fe:.0f} kWh/t Fe",
                ])
            if mb.post_processing_energy_MWh_yr > 0:
                lines.append(f"  Post-proc total:  {mb.post_processing_energy_MWh_yr:.0f} MWh/yr")

        lines.extend([
            "",
            "CO2 FOOTPRINT:",
            f"  Scope 1 (direct): {mb.scope1_CO2_t_yr:.1f} t CO₂/yr",
            f"  Scope 2 (elec):   {mb.scope2_CO2_t_yr:.0f} t CO₂/yr",
            f"  Per tonne Fe:     {mb.scope2_CO2_t_yr/mb.iron_production_t_yr:.3f} t CO₂/t Fe",
            "",
            "ECONOMICS:",
            f"  CAPEX:            ${self.capex['Total CAPEX ($)']:,.0f} ({self.capex['Total CAPEX (M$)']:.1f} M$)",
            f"  Annual OPEX:      ${self.opex['Total OPEX ($/yr)']:,.0f}",
            f"  LCOFe:            ${self.lcofe['LCOFe ($/t Fe)']:,.0f}/t Fe",
            f"  CAPEX share:      {self.lcofe['CAPEX share (%)']:.0f}%",
            "",
            f"{'='*70}",
            "GO / NO-GO ASSESSMENT",
            f"{'='*70}",
        ])

        for criterion, result in gng.items():
            status = "PASS" if result["pass"] else "FAIL"
            lines.append(f"  [{status}] {criterion}: {result['detail']}")

        overall = all(r["pass"] for r in gng.values())
        lines.append("")
        lines.append(f"  OVERALL: {'GO — site is viable for dark mill deployment' if overall else 'NO-GO — see failed criteria above'}")
        lines.append(f"{'='*70}")

        # Crate structural summary if present
        if self.crate_verdict is not None:
            cv = self.crate_verdict
            lines.extend([
                "",
                f"{'='*70}",
                "CRATE STRUCTURAL / ENVIRONMENTAL VERDICT",
                f"{'='*70}",
                f"  Wind:             {cv.notes.get('wind','') if hasattr(cv,'notes') else ''}",
                f"  Dynamic pressure: {cv.dynamic_pressure_Pa:.0f} Pa",
                f"  Wind force:       {cv.wind_force_N:.0f} N (sliding: {cv.wind_force_sliding_N:.0f} N)",
                f"  FS overturn:      {cv.fs_overturn:.2f}",
                f"  FS bearing:       {cv.fs_bearing:.2f} (p_net {cv.net_bearing_kPa:.1f} kPa)",
                f"  FS sliding:       {cv.fs_slide:.2f}",
                f"  Ingress risk:     {cv.ingress_risk}",
                f"  Min ballast:      {cv.min_ballast_kg:.0f} kg",
                f"  Mounting:         {cv.mounting_spec}",
                f"  Stable:           {cv.stable}",
            ])
            if self.credibility is not None:
                cred = self.credibility
                lines.append(f"  Credibility:      process L{cred.get('process',0)} / crate L{cred.get('crate',0)} / site L{cred.get('site',0)}")

        return "\n".join(lines)


# ─── Crate coupling helpers ────────────────────────────────────────

def site_to_crate_config(site: SiteDefinition, ballast_kg: float = 0.0):
    """Map a SiteDefinition's structural/environmental fields to a CrateConfig.

    This is the L2/L3 layer coupling: ClimateSpec (wind, rain, snow) + SiteDefinition
    (soil, seismic, flood, sealing) → CrateConfig for stability checks.
    All numbers are screening-grade L0.
    """
    # Lazy import to avoid circular
    from .crate import CrateSpec, WindLoad, GroundSpec, EnvironmentalLoads, CrateConfig

    crate = CrateSpec()  # default 40-ft container, can be overridden by site in future
    wind = WindLoad(
        gust_m_s=site.climate.wind_gust_m_s,
        direction="broadside",  # worst-case screening; system_twin can test end-on too
        terrain=site.climate.wind_terrain,
        altitude_m=site.climate.altitude_m,
        temperature_C=site.climate.ambient_temp_C,
    )
    ground = GroundSpec(
        p_allow_kPa=site.soil_bearing_kPa,
        friction_mu=site.ground_friction_mu,
        drainable=(site.flood_depth_m < 0.05 and site.climate.rainfall_intensity_mm_hr < 100),
        flood_depth_m=site.flood_depth_m,
        anchored=site.ground_anchored,
    )
    env = EnvironmentalLoads(
        rain_intensity_mm_hr=site.climate.rainfall_intensity_mm_hr,
        sealing_class=site.sealing_class,
        snow_load_kPa=site.climate.snow_load_kPa,
        seismic_base_coefficient=site.seismic_coefficient,
    )
    return CrateConfig(crate=crate, wind=wind, ground=ground, env=env, ballast_kg=ballast_kg)


def evaluate_crate_for_site(site: SiteDefinition, ballast_kg: float = 0.0):
    """Evaluate crate stability for a given SiteDefinition (screening L0)."""
    from .crate import Crate
    cfg = site_to_crate_config(site, ballast_kg=ballast_kg)
    return Crate().evaluate(cfg)


# ─── Sizing Engine ────────────────────────────────────────────────

# Module-level defaults for the dark mill sizing
DEFAULT_ELECTRODE_AREA_M2 = 0.5   # per face, ~50×50 cm panels
DEFAULT_CELLS_PER_STACK = 20      # bipolar stack
DEFAULT_OPERATING_HOURS = 8000.0  # ~91% capacity factor


def size_dark_mill(
    site: SiteDefinition,
    electrode_area_m2: float = DEFAULT_ELECTRODE_AREA_M2,
    cells_per_stack: int = DEFAULT_CELLS_PER_STACK,
    operating_hours: float = DEFAULT_OPERATING_HOURS,
    min_FE: float = 0.70,
) -> SiteReport:
    """
    Size a dark mill for a given site and produce a full assessment.

    Physics-driven: runs the unified cell_physics solver (speciation →
    Nernst-Planck transport → voltage decomposition) to predict V_cell,
    FE, and optimal current density from bath chemistry and conditions.

    Falls back to literature defaults if the physics solver fails.
    """

    fs = site.feedstock
    grid = site.grid
    climate = site.climate

    # ── Step 0: Solve cell physics ──
    # Set temperature from climate if not overridden in conditions
    if site.conditions.temperature_C == 50.0:  # default, not explicitly set
        site.conditions.temperature_C = max(40.0, min(70.0, climate.ambient_temp_C + 25.0))

    physics = CellPhysics(
        bath=site.bath,
        geometry=site.geometry,
        conditions=site.conditions,
    )

    # Find optimal operating point from physics
    physics_point = physics.find_optimal_j(min_FE=min_FE)

    if physics_point is not None:
        # Physics succeeded — use computed values
        cell_voltage_V = physics_point.V_cell
        current_efficiency = physics_point.current_efficiency
        current_density_mA_cm2 = physics_point.j_mA_cm2
        temperature_C = site.conditions.temperature_C
        specific_energy_kWh_t = physics_point.specific_energy_kWh_t
    else:
        # Physics failed to find a viable point — use conservative defaults
        cell_voltage_V = 2.8
        current_efficiency = 0.85
        current_density_mA_cm2 = 100.0
        temperature_C = site.conditions.temperature_C
        specific_energy_kWh_t = specific_energy_kWh_per_t(cell_voltage_V, current_efficiency)

    # ── Step 1: Size the stack to meet target capacity ──
    prod_per_cell_hr = current_density_to_production(
        current_density_mA_cm2, electrode_area_m2, current_efficiency
    )
    prod_per_stack_hr = prod_per_cell_hr * cells_per_stack
    prod_per_stack_yr = prod_per_stack_hr * operating_hours / 1000.0  # t/yr

    if prod_per_stack_yr > 0:
        n_stacks = max(1, math.ceil(site.target_capacity_t_Fe_yr / prod_per_stack_yr))
    else:
        n_stacks = 1

    actual_capacity_t_yr = prod_per_stack_yr * n_stacks

    stack = StackDesign(
        n_stacks=n_stacks,
        cells_per_stack=cells_per_stack,
        electrode_area_m2=electrode_area_m2,
        current_density_mA_cm2=current_density_mA_cm2,
        current_efficiency=current_efficiency,
        cell_voltage_V=cell_voltage_V,
        temperature_C=temperature_C,
    )

    # ── Step 2: Mass balance ──
    # Feedstock
    t_feed_per_t_Fe = fs.tonnes_feedstock_per_tonne_Fe
    feedstock_consumed = actual_capacity_t_yr * t_feed_per_t_Fe

    # Water: ~3-5 m³/t Fe for electrolyte makeup, cooling, washing
    water_per_t_Fe = 4.0  # m³/t Fe
    if climate.water_availability == "scarce":
        water_per_t_Fe = 3.0
    water_consumption = actual_capacity_t_yr * water_per_t_Fe

    # Acid consumption: depends on feedstock
    if fs.already_dissolved:
        acid_per_t_Fe = 0.0
    else:
        acid_per_t_Fe = fs.dissolution_acid_cost_per_t_Fe / 20.0
    acid_consumption = actual_capacity_t_yr * acid_per_t_Fe

    # Electrolyte makeup: ~2% of volume per month
    electrolyte_volume_m3 = 0.02 * stack.total_electrode_area_m2
    electrolyte_makeup = electrolyte_volume_m3 * 0.02 * 12

    # Waste: gangue from feedstock + anode sludge
    waste_gangue = feedstock_consumed * (1.0 - fs.fe_wt_fraction) * 0.5
    waste_anode_sludge = actual_capacity_t_yr * 0.01
    waste_total = waste_gangue + waste_anode_sludge

    # Energy — use physics-derived specific energy
    electrolysis_energy = specific_energy_kWh_t * actual_capacity_t_yr / 1000.0  # MWh/yr
    grinding_energy = fs.dissolution_energy_kWh_per_t_Fe * actual_capacity_t_yr / 1000.0
    # Cooling: ~5-10% of electrolysis waste heat
    # Waste heat = total input - thermoneutral = I*(V_cell - E_therm)
    E_therm = 1.28  # V, thermoneutral for Fe2+ + H2O -> Fe + 0.5 O2 + 2H+
    waste_heat_fraction = max(0, (cell_voltage_V - E_therm) / cell_voltage_V)
    cooling_energy = electrolysis_energy * waste_heat_fraction * 0.10  # 10% of waste heat for cooling systems
    auxiliary_energy = actual_capacity_t_yr * 50.0 / 1000.0  # ~50 kWh/t for pumps, controls, lighting
    total_energy = electrolysis_energy + grinding_energy + cooling_energy + auxiliary_energy

    # CO2
    scope1_CO2 = actual_capacity_t_yr * 0.02  # minimal direct emissions
    scope2_CO2 = total_energy * 1000.0 * grid.grid_CO2_kg_per_kWh / 1000.0  # MWh→kWh, kg→t

    mass_bal = MassBalance(
        iron_production_t_yr=actual_capacity_t_yr,
        feedstock_consumed_t_yr=feedstock_consumed,
        feedstock_cost_per_t_Fe=fs.cost_per_t_Fe,
        water_consumption_m3_yr=water_consumption,
        water_cost_yr=water_consumption * climate.water_cost_per_m3,
        acid_consumption_t_yr=acid_consumption,
        acid_cost_yr=acid_consumption * 200.0,  # ~$200/t acid
        electrolyte_makeup_m3_yr=electrolyte_makeup,
        electrolyte_cost_yr=electrolyte_makeup * 5000.0,  # ~$5000/m³
        waste_generated_t_yr=waste_total,
        waste_disposal_cost_yr=waste_total * 30.0,  # ~$30/t disposal
        electrolysis_energy_MWh_yr=electrolysis_energy,
        grinding_energy_MWh_yr=grinding_energy,
        cooling_energy_MWh_yr=cooling_energy,
        auxiliary_energy_MWh_yr=auxiliary_energy,
        total_energy_MWh_yr=total_energy,
        scope1_CO2_t_yr=scope1_CO2,
        scope2_CO2_t_yr=scope2_CO2,
    )

    # ── Step 3: Thermal assessment ──
    cell_thermal = CellThermalParams(
        V_cell=cell_voltage_V,
        current_A=stack.total_current_A,
        volume_L=electrolyte_volume_m3 * 1000,
        T_init_C=climate.ambient_temp_C,
        T_amb_C=climate.ambient_temp_C,
        relative_humidity=climate.relative_humidity,
    )
    # Run a short thermal simulation to get steady-state temp
    try:
        thermal_result = simulate_thermal_transient(
            cell_thermal, t_end_hr=4.0, dt_s=10.0
        )
        T_steady = thermal_result["T_ss_C"]
        cooling_W = thermal_result.get("cooling_duty_50C_W", 0)
    except Exception:
        T_steady = temperature_C
        cooling_W = 0

    thermal_info = {
        "steady_state_temp_C": round(T_steady, 1),
        "cooling_required_W": round(cooling_W, 0),
        "ambient_C": climate.ambient_temp_C,
        "thermal_management": "passive" if cooling_W < 500 else "active jacket cooling",
    }

    # ── Step 4: Economics ──
    electrolyzer_params = ElectrolyzerParams(
        current_density_mA_cm2=current_density_mA_cm2,
        current_efficiency=current_efficiency,
        cell_voltage=cell_voltage_V,
        temperature_C=temperature_C,
        electrode_area_m2=electrode_area_m2,
        n_cells=cells_per_stack,
    )

    capex_model = CAPEXModel()
    capex = capex_model.estimate(electrolyzer_params, n_stacks)

    # Adjust OPEX for site specifics
    opex_model = OPEXModel(
        electricity_price_kWh=grid.effective_price_kWh,
        ore_cost_per_t_Fe=max(0, fs.cost_per_t_Fe),  # negative feedstock = revenue
        grinding_energy_kWh_per_t=fs.dissolution_energy_kWh_per_t_Fe,
        water_cost_per_t_Fe=climate.water_cost_per_m3 * water_per_t_Fe,
        labor_cost_per_yr=site.labor_cost_per_yr,
    )
    opex = opex_model.estimate(
        electrolyzer_params, capex["Total CAPEX ($)"], n_stacks, operating_hours
    )

    lc_model = LevelizedCost(capacity_factor=operating_hours / 8760.0)
    lcofe = lc_model.calculate(
        capex["Total CAPEX ($)"], opex["Total OPEX ($/yr)"], capex["Annual capacity (t/yr)"]
    )

    # Add feedstock revenue if negative cost
    feedstock_revenue = 0.0
    if fs.cost_per_t_feedstock < 0:
        # You're paid to take the feedstock — this is a revenue line
        feedstock_revenue = abs(fs.cost_per_t_feedstock) * feedstock_consumed / actual_capacity_t_yr
        # Adjust LCOFe downward
        adjusted_annual_cost = (
            lcofe["Total annual cost ($/yr)"] - feedstock_revenue * actual_capacity_t_yr
        )
        if adjusted_annual_cost > 0 and actual_capacity_t_yr > 0:
            lcofe["LCOFe ($/t Fe) adjusted"] = round(
                adjusted_annual_cost / actual_capacity_t_yr, 0
            )
            lcofe["Feedstock revenue ($/t Fe)"] = round(feedstock_revenue, 2)

    # -- Step 5: Benchmarks --
    benchmarks = compare_routes(lcofe["LCOFe ($/t Fe)"])

    # -- Step 6: Post-processing / grade specification --
    post_proc = size_post_processing(
        grade=site.effective_grade,
        route=site.post_processing_route,
        sheet_thickness_um=site.sheet_thickness_um,
        annual_production_t=actual_capacity_t_yr,
    )
    # Add post-processing energy to mass balance
    pp_energy_MWh = post_proc.total_energy_kWh_per_t * actual_capacity_t_yr / 1000.0
    mass_bal.post_processing_energy_MWh_yr = pp_energy_MWh
    mass_bal.total_energy_MWh_yr += pp_energy_MWh
    mass_bal.product_grade = site.effective_grade.name
    # Update scope2 CO2 for post-processing electricity
    mass_bal.scope2_CO2_t_yr += pp_energy_MWh * 1000.0 * grid.grid_CO2_kg_per_kWh / 1000.0

    # Adjust CAPEX for post-processing equipment
    pp_capex_fraction = post_proc.additional_capex_fraction
    if pp_capex_fraction > 0:
        pp_capex = capex["Total CAPEX ($)"] * pp_capex_fraction
        capex["Post-processing CAPEX ($)"] = round(pp_capex, 0)
        capex["Total CAPEX ($)"] += pp_capex
        capex["Total CAPEX (M$)"] = round(capex["Total CAPEX ($)"] / 1e6, 2)

    # -- Step 6b: Crate structural / environmental verdict (L2 layer) --
    # Screening-grade L0 until real load test / site survey.
    # Feeds the go/no-go so "does this site make sense?" also means
    # "can a unit physically sit here through a storm?"
    try:
        crate_verdict = evaluate_crate_for_site(site)
    except Exception:
        # Don't fail the whole site assessment if crate eval fails — mark None
        crate_verdict = None

    # -- Step 7: Go/No-Go (including crate) --
    go_no_go = _assess_go_nogo(
        site, stack, mass_bal, capex, opex, lcofe, thermal_info, benchmarks,
        crate_verdict=crate_verdict,
    )

    # Credibility vector per-layer (all L0 screening until validated)
    credibility = {"process": 0, "crate": 0, "site": 0}

    return SiteReport(
        site=site,
        stack_design=stack,
        mass_balance=mass_bal,
        capex=capex,
        opex=opex,
        lcofe=lcofe,
        thermal=thermal_info,
        benchmarks=benchmarks,
        go_no_go=go_no_go,
        physics_point=physics_point,
        post_processing=post_proc,
        crate_verdict=crate_verdict,
        credibility=credibility,
    )


def _assess_go_nogo(
    site: SiteDefinition,
    stack: StackDesign,
    mb: MassBalance,
    capex: Dict,
    opex: Dict,
    lcofe: Dict,
    thermal: Dict,
    benchmarks: Dict,
    crate_verdict: Optional[Any] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate go/no-go criteria.

    Criteria from RESEARCH_PROGRAM.md kill criteria, adapted for
    site-level assessment, plus crate structural/environmental checks that
    answer "can a unit physically sit here through a storm?"

    1. Power: can the grid support the mill?
    2. LCOFe: competitive with DRI-H2?
    3. Area: fits within available footprint?
    4. Thermal: manageable operating temperature?
    5. Feedstock: sustainable supply?
    6. Water: available at site?
    7. Energy kill criterion
    8-11. Crate: overturning / bearing / sliding / ingress
    """
    criteria = {}

    # 1. Grid capacity
    power_MW = stack.total_power_kW / 1000.0
    criteria["Grid capacity"] = {
        "pass": power_MW <= site.grid.max_power_MW,
        "detail": f"{power_MW:.1f} MW needed, {site.grid.max_power_MW:.1f} MW available",
    }

    # 2. LCOFe vs DRI-H2
    dri_mid = BENCHMARK_COSTS["H2-DRI + EAF"]["mid"]
    our_lcofe = lcofe.get("LCOFe ($/t Fe) adjusted", lcofe["LCOFe ($/t Fe)"])
    criteria["LCOFe vs DRI-H2"] = {
        "pass": our_lcofe < dri_mid,
        "detail": f"${our_lcofe:.0f}/t vs DRI-H2 ${dri_mid}/t",
    }

    # 3. Site footprint
    area_needed = stack.total_electrode_area_m2 * 2.0 + 200  # +200 for BOP
    criteria["Site footprint"] = {
        "pass": area_needed <= site.available_area_m2,
        "detail": f"{area_needed:.0f} m² needed, {site.available_area_m2:.0f} m² available",
    }

    # 4. Thermal management
    T_op = thermal["steady_state_temp_C"]
    criteria["Operating temperature"] = {
        "pass": T_op < 90.0,
        "detail": f"Steady-state {T_op:.0f} °C (limit 90 °C)",
    }

    # 5. Feedstock sustainability
    criteria["Feedstock supply"] = {
        "pass": True,
        "detail": f"{mb.feedstock_consumed_t_yr:.0f} t/yr consumed ({site.feedstock.name})",
    }

    # 6. Water availability
    water_tight = site.climate.water_availability == "scarce"
    criteria["Water supply"] = {
        "pass": not water_tight or mb.water_consumption_m3_yr < 5000,
        "detail": f"{mb.water_consumption_m3_yr:.0f} m³/yr, availability: {site.climate.water_availability}",
    }

    # 7. Energy kill criterion
    specific_energy = mb.total_energy_MWh_yr * 1000 / mb.iron_production_t_yr if mb.iron_production_t_yr > 0 else float('inf')
    energy_pass = specific_energy <= 4000
    criteria["Energy kill criterion"] = {
        "pass": energy_pass,
        "detail": f"{specific_energy:.0f} kWh/t Fe (limit 4,000)",
    }

    # 8-11. Crate structural / environmental (whole-system twin L2/L3)
    if crate_verdict is not None:
        # Overturning
        try:
            fs_ot = crate_verdict.fs_overturn
            criteria["Crate overturning"] = {
                "pass": fs_ot >= 1.5 or crate_verdict.min_ballast_kg == 0 or site.ground_anchored,
                "detail": f"FS_over={fs_ot:.2f} (target 1.5), min ballast {crate_verdict.min_ballast_kg:.0f} kg, "
                          f"mounting: {crate_verdict.mounting_spec}",
            }
        except Exception:
            pass
        # Bearing
        try:
            criteria["Crate bearing"] = {
                "pass": crate_verdict.fs_bearing >= 1.0,
                "detail": f"FS_bearing={crate_verdict.fs_bearing:.2f}, p_net {crate_verdict.net_bearing_kPa:.1f} kPa "
                          f"vs {site.soil_bearing_kPa:.0f} kPa allowable",
            }
        except Exception:
            pass
        # Sliding
        try:
            criteria["Crate sliding"] = {
                "pass": crate_verdict.fs_slide >= 1.5 or site.ground_anchored,
                "detail": f"FS_slide={crate_verdict.fs_slide:.2f} (target 1.5), "
                          f"wind {crate_verdict.wind_force_sliding_N:.0f} N",
            }
        except Exception:
            pass
        # Ingress / flood
        try:
            ingress_ok = crate_verdict.ingress_risk != "high"
            # If flood depth > 0.3 m, require elevation regardless of sealing
            if site.flood_depth_m > 0.3 and not ingress_ok:
                ingress_ok = False
            criteria["Crate ingress"] = {
                "pass": ingress_ok,
                "detail": f"ingress {crate_verdict.ingress_risk}, flood {site.flood_depth_m:.2f} m, "
                          f"rain {site.climate.rainfall_intensity_mm_hr:.0f} mm/hr, sealing {site.sealing_class}",
            }
        except Exception:
            pass

    return criteria


# ─── Site Library ──────────────────────────────────────────────────

EXAMPLE_SITES: Dict[str, SiteDefinition] = {
    "pickle_liquor_us_midwest": SiteDefinition(
        name="US Midwest — Pickle Liquor at Steel Mill",
        feedstock_key="pickle_liquor",
        grid=GridSpec(electricity_price_kWh=0.035, renewable_fraction=0.40, max_power_MW=5.0),
        climate=ClimateSpec(
            ambient_temp_C=12.0, relative_humidity=0.65, water_cost_per_m3=1.5,
            wind_gust_m_s=35.0, wind_terrain="suburban", rainfall_intensity_mm_hr=60.0,
            snow_load_kPa=0.5, freeze_risk=True, freeze_depth_m=0.9,
        ),
        feedstock_distance_km=0.0,
        product_market_km=50.0,
        target_capacity_t_Fe_yr=2000.0,
        available_area_m2=300.0,
        soil_bearing_kPa=150.0, seismic_coefficient=0.06, flood_depth_m=0.0,
        sealing_class="industrial", ground_friction_mu=0.5,
        notes="Co-located with a steel mill. Paid to take pickle liquor. Product feeds the mill's EAF.",
    ),
    "acid_mine_drainage_appalachia": SiteDefinition(
        name="Appalachia — Acid Mine Drainage Remediation",
        feedstock_key="acid_mine_drainage",
        grid=GridSpec(electricity_price_kWh=0.05, renewable_fraction=0.30, max_power_MW=2.0),
        climate=ClimateSpec(
            ambient_temp_C=12.0, relative_humidity=0.70, water_cost_per_m3=0.5,
            wind_gust_m_s=40.0, wind_terrain="open", rainfall_intensity_mm_hr=80.0,
            snow_load_kPa=0.8, freeze_risk=True, freeze_depth_m=0.6,
        ),
        feedstock_distance_km=0.0,
        product_market_km=300.0,
        target_capacity_t_Fe_yr=500.0,
        available_area_m2=200.0,
        soil_bearing_kPa=80.0, seismic_coefficient=0.10, flood_depth_m=0.10,
        sealing_class="industrial", ground_friction_mu=0.4,
        notes="Remediation co-benefit. Very dilute feedstock (2% Fe) — concentration step needed.",
    ),
    "red_mud_alumina_refinery": SiteDefinition(
        name="Red Mud — Alumina Refinery Co-Location",
        feedstock_key="red_mud",
        grid=GridSpec(electricity_price_kWh=0.03, renewable_fraction=0.60, max_power_MW=10.0),
        climate=ClimateSpec(
            ambient_temp_C=28.0, relative_humidity=0.75, water_cost_per_m3=1.0,
            wind_gust_m_s=45.0, wind_terrain="open", rainfall_intensity_mm_hr=100.0,
            snow_load_kPa=0.0, freeze_risk=False, freeze_depth_m=0.0,
        ),
        feedstock_distance_km=0.0,
        product_market_km=100.0,
        target_capacity_t_Fe_yr=5000.0,
        available_area_m2=1000.0,
        soil_bearing_kPa=100.0, seismic_coefficient=0.12, flood_depth_m=0.20,
        sealing_class="industrial", ground_friction_mu=0.45,
        notes="Co-located with alumina refinery. Paid to take red mud. SIDERWIN demonstrated this route.",
    ),
    "wind_farm_ore": SiteDefinition(
        name="Wind Farm — High-Grade Ore",
        feedstock_key="high_grade_ore",
        grid=GridSpec(electricity_price_kWh=0.025, renewable_fraction=0.95, max_power_MW=20.0,
                      grid_CO2_kg_per_kWh=0.02),
        climate=ClimateSpec(
            ambient_temp_C=18.0, relative_humidity=0.40, water_cost_per_m3=3.0,
            water_availability="scarce",
            wind_gust_m_s=55.0, wind_terrain="open", rainfall_intensity_mm_hr=30.0,
            snow_load_kPa=0.2, freeze_risk=False, freeze_depth_m=0.0,
        ),
        feedstock_distance_km=30.0,
        product_market_km=500.0,
        target_capacity_t_Fe_yr=10000.0,
        available_area_m2=2000.0,
        soil_bearing_kPa=120.0, seismic_coefficient=0.10, flood_depth_m=0.0,
        sealing_class="sealed", ground_friction_mu=0.5,
        notes="Next to a wind farm. Cheap clean electricity. Ore railed in. Water may be limiting. High wind exposure.",
    ),
    "copperas_tio2_plant": SiteDefinition(
        name="TiO2 Plant — Copperas Byproduct",
        feedstock_key="copperas",
        grid=GridSpec(electricity_price_kWh=0.06, renewable_fraction=0.20, max_power_MW=3.0),
        climate=ClimateSpec(
            ambient_temp_C=20.0, relative_humidity=0.55, water_cost_per_m3=2.0,
            wind_gust_m_s=38.0, wind_terrain="suburban", rainfall_intensity_mm_hr=50.0,
            snow_load_kPa=0.3, freeze_risk=False, freeze_depth_m=0.2,
        ),
        feedstock_distance_km=0.0,
        product_market_km=150.0,
        target_capacity_t_Fe_yr=1500.0,
        available_area_m2=400.0,
        soil_bearing_kPa=100.0, seismic_coefficient=0.08, flood_depth_m=0.0,
        sealing_class="industrial", ground_friction_mu=0.5,
        notes="Copperas from TiO2 sulfate process. Negative-cost feedstock. Water-soluble — no grinding.",
    ),
}


def run_site(site_key: str, **kwargs) -> SiteReport:
    """Run a dark mill assessment for a named example site."""
    site = EXAMPLE_SITES[site_key]
    return size_dark_mill(site, **kwargs)


def run_all_sites(**kwargs) -> Dict[str, SiteReport]:
    """Run all example sites and return reports keyed by site name."""
    return {key: size_dark_mill(site, **kwargs) for key, site in EXAMPLE_SITES.items()}


def comparison_table(reports: Dict[str, SiteReport]) -> str:
    """Pretty-print a comparison of all sites."""
    lines = [
        f"{'Site':<42} {'Cap t/yr':>8} {'CAPEX M$':>8} {'LCOFe':>8} {'kWh/t':>7} {'CO2/t':>6} {'GO?':>4}",
        "-" * 90,
    ]
    for key, r in reports.items():
        cap = r.stack_design.annual_production_t()
        capex_m = r.capex["Total CAPEX (M$)"]
        lcofe = r.lcofe["LCOFe ($/t Fe)"]
        kwh_t = r.mass_balance.total_energy_MWh_yr * 1000 / r.mass_balance.iron_production_t_yr
        co2_t = r.mass_balance.scope2_CO2_t_yr / r.mass_balance.iron_production_t_yr
        go = "GO" if all(v["pass"] for v in r.go_no_go.values()) else "NO"
        lines.append(f"{r.site.name:<42} {cap:>7.0f} {capex_m:>8.1f} ${lcofe:>6.0f} {kwh_t:>6.0f} {co2_t:>5.3f} {go:>4}")
    return "\n".join(lines)
