"""
Supply chain model for aqueous electrowinning steel.

Covers feedstock chemical material balance, electrolyte recycling economics,
and geographic siting analysis for plant location.

All costs in USD (2024 basis).

References
----------
- USGS Mineral Commodity Summaries (2024): FeSO4, NiSO4, H2SO4, NaOH, BaCO3
- ICIS Chemical Pricing (2024): carbon black, industrial chemicals
- IEA Electricity Prices (2024): industrial electricity by region
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ─── Raw Material Definitions ─────────────────────────────────────────

@dataclass
class RawMaterial:
    """Single feedstock chemical with price range and supply notes."""
    name: str
    formula: str
    unit: str                           # typically "$/t"
    price_low: float                    # $/t low estimate
    price_mid: float                    # $/t midpoint
    price_high: float                   # $/t high estimate
    primary_sources: str                # where it comes from
    notes: str = ""


# Feedstock chemicals for aqueous electrowinning + post-processing
RAW_MATERIALS: List[RawMaterial] = [
    RawMaterial(
        "Ferrous sulfate", "FeSO4·7H2O", "$/t",
        0, 75, 150,
        "Byproduct of steel pickling, TiO2 production (sulfate process)",
        "Often negative price (waste disposal credit); price varies by region",
    ),
    RawMaterial(
        "Nickel sulfate", "NiSO4·6H2O", "$/t",
        3000, 4000, 5000,
        "Battery-grade supply chain (Class 1 Ni → NiSO4); also plating waste",
        "Battery EV demand driving price volatility",
    ),
    RawMaterial(
        "Carbon black", "C", "$/t",
        1000, 1250, 1500,
        "Commodity (furnace process); petroleum/coking byproduct",
        "Used in co-deposition and conductive additives",
    ),
    RawMaterial(
        "Sulfuric acid", "H2SO4", "$/t",
        50, 75, 100,
        "Commodity; contact process; also smelter acid byproduct",
        "Bulk industrial; large price breaks above 1000 t/yr",
    ),
    RawMaterial(
        "Sodium hydroxide", "NaOH (50% soln)", "$/t",
        250, 375, 500,
        "Chlor-alkali process; commodity",
        "Used for pH control, Cl2 scrubbing, neutralization",
    ),
    RawMaterial(
        "Barium carbonate", "BaCO3", "$/t",
        400, 600, 800,
        "Witherite mining (natural); also chemical precipitation",
        "Used in pack carburizing as energizer",
    ),
]


# ─── Design Point ─────────────────────────────────────────────────────

@dataclass
class DesignPoint:
    """Operating parameters that determine material consumption."""
    # Electrolyte composition (mol/L)
    Fe2_mol_L: float = 1.0              # ferrous ion concentration
    Ni2_mol_L: float = 0.0              # optional nickel co-deposition
    H2SO4_mol_L: float = 0.5           # supporting electrolyte

    # Process parameters
    current_density_A_m2: float = 200.0  # A/m²
    current_efficiency: float = 0.90     # Faradaic efficiency
    cell_voltage_V: float = 2.5          # total cell voltage
    electrode_area_m2: float = 1.0       # per cell
    n_cells: int = 10                    # cells in series

    # Post-processing
    carburize_with_BaCO3: bool = True    # pack carburizing energizer
    BaCO3_kg_per_kg_Fe: float = 0.05     # consumption ratio

    # Electrolyte volume
    electrolyte_volume_L: float = 1000.0  # total system volume

    def production_rate_kg_hr(self) -> float:
        """Fe deposition rate (kg/hr) from Faraday's law."""
        # Molar mass Fe = 55.845 g/mol, z = 2
        M_FE = 55.845e-3  # kg/mol
        Z_FE = 2
        F = 96485.0  # C/mol
        I = self.current_density_A_m2 * self.electrode_area_m2  # A per cell
        per_cell = I * self.current_efficiency * M_FE / (Z_FE * F) * 3600  # kg/hr
        return per_cell * self.n_cells

    def energy_kWh_per_kg(self) -> float:
        """Specific energy consumption (kWh/kg Fe)."""
        rate = self.production_rate_kg_hr()
        if rate <= 0:
            return float('inf')
        power_kW = (self.current_density_A_m2 * self.electrode_area_m2
                     * self.cell_voltage_V * self.n_cells / 1000.0)
        return power_kW / rate


# ─── Material Balance ─────────────────────────────────────────────────

@dataclass
class MaterialLineItem:
    """Single chemical in the material balance."""
    name: str
    formula: str
    consumption_kg_per_day: float       # kg consumed per day
    consumption_t_per_year: float       # tonnes per year
    unit_cost_per_t: float              # $/t (midpoint)
    daily_cost: float                   # $/day
    annual_cost: float                  # $/year
    fraction_of_total: float            # share of total feedstock cost


@dataclass
class MaterialBalance:
    """Full material balance for a production scale."""
    production_rate_kg_day: float
    production_rate_t_year: float
    design_point: DesignPoint
    items: List[MaterialLineItem]
    total_annual_feedstock_cost: float
    specific_feedstock_cost_per_kg: float  # $/kg Fe


def material_balance(
    design_point: DesignPoint,
    production_rate_kg_day: float,
    materials: Optional[List[RawMaterial]] = None,
    operating_days_per_year: float = 300.0,
) -> MaterialBalance:
    """Compute material balance for all feedstock chemicals.

    Parameters
    ----------
    design_point : DesignPoint
        Process operating parameters.
    production_rate_kg_day : float
        Target Fe production (kg/day). Used to scale all consumptions.
    materials : list of RawMaterial, optional
        Override default feedstock list.
    operating_days_per_year : float
        Days per year of operation.

    Returns
    -------
    MaterialBalance
    """
    if materials is None:
        materials = RAW_MATERIALS

    kg_day = production_rate_kg_day
    t_year = kg_day * operating_days_per_year / 1000.0

    items: List[MaterialLineItem] = []

    # FeSO4 — 1 mol FeSO4 per mol Fe deposited
    # FeSO4·7H2O MW = 278.01 g/mol; Fe = 55.845 g/mol
    feso4_ratio = 278.01 / 55.845  # kg FeSO4·7H2O per kg Fe
    feso4_kg_day = kg_day * feso4_ratio
    feso4_cost_t = _get_mid_price(materials, "Ferrous sulfate")
    items.append(MaterialLineItem(
        "Ferrous sulfate", "FeSO4·7H2O",
        feso4_kg_day, feso4_kg_day * operating_days_per_year / 1000,
        feso4_cost_t, feso4_kg_day / 1000 * feso4_cost_t,
        feso4_kg_day / 1000 * feso4_cost_t * operating_days_per_year,
        0.0,  # filled below
    ))

    # NiSO4 — if Ni co-deposition enabled
    ni_mol_L = design_point.Ni2_mol_L
    if ni_mol_L > 0:
        niso4_ratio = 262.85 / 58.69  # NiSO4·6H2O / Ni molar mass
        # Ni deposition fraction (rough: proportional to mol ratio)
        ni_frac = ni_mol_L / (design_point.Fe2_mol_L + ni_mol_L + 1e-12)
        ni_kg_day = kg_day * ni_frac * niso4_ratio
        niso4_cost_t = _get_mid_price(materials, "Nickel sulfate")
        items.append(MaterialLineItem(
            "Nickel sulfate", "NiSO4·6H2O",
            ni_kg_day, ni_kg_day * operating_days_per_year / 1000,
            niso4_cost_t, ni_kg_day / 1000 * niso4_cost_t,
            ni_kg_day / 1000 * niso4_cost_t * operating_days_per_year,
            0.0,
        ))

    # H2SO4 — makeup for acid consumed/depleted in electrolyte
    # Rough: 0.5 kg H2SO4 per kg Fe (pH maintenance + anode reaction)
    h2so4_kg_day = kg_day * 0.5
    h2so4_cost_t = _get_mid_price(materials, "Sulfuric acid")
    items.append(MaterialLineItem(
        "Sulfuric acid", "H2SO4",
        h2so4_kg_day, h2so4_kg_day * operating_days_per_year / 1000,
        h2so4_cost_t, h2so4_kg_day / 1000 * h2so4_cost_t,
        h2so4_kg_day / 1000 * h2so4_cost_t * operating_days_per_year,
        0.0,
    ))

    # NaOH — for pH control, Cl2 scrubbing, neutralization
    # Rough: 0.3 kg NaOH per kg Fe
    naoh_kg_day = kg_day * 0.3
    naoh_cost_t = _get_mid_price(materials, "Sodium hydroxide")
    items.append(MaterialLineItem(
        "Sodium hydroxide", "NaOH",
        naoh_kg_day, naoh_kg_day * operating_days_per_year / 1000,
        naoh_cost_t, naoh_kg_day / 1000 * naoh_cost_t,
        naoh_kg_day / 1000 * naoh_cost_t * operating_days_per_year,
        0.0,
    ))

    # BaCO3 — pack carburizing energizer (optional)
    if design_point.carburize_with_BaCO3:
        baco3_kg_day = kg_day * design_point.BaCO3_kg_per_kg_Fe
        baco3_cost_t = _get_mid_price(materials, "Barium carbonate")
        items.append(MaterialLineItem(
            "Barium carbonate", "BaCO3",
            baco3_kg_day, baco3_kg_day * operating_days_per_year / 1000,
            baco3_cost_t, baco3_kg_day / 1000 * baco3_cost_t,
            baco3_kg_day / 1000 * baco3_cost_t * operating_days_per_year,
            0.0,
        ))

    # Carbon black — co-deposition additive (minor)
    cb_kg_day = kg_day * 0.02  # 2% of Fe mass
    cb_cost_t = _get_mid_price(materials, "Carbon black")
    items.append(MaterialLineItem(
        "Carbon black", "C",
        cb_kg_day, cb_kg_day * operating_days_per_year / 1000,
        cb_cost_t, cb_kg_day / 1000 * cb_cost_t,
        cb_kg_day / 1000 * cb_cost_t * operating_days_per_year,
        0.0,
    ))

    # Compute fractions
    total_cost = sum(it.annual_cost for it in items)
    for it in items:
        it.fraction_of_total = it.annual_cost / total_cost if total_cost > 0 else 0

    specific = total_cost / (kg_day * operating_days_per_year) if kg_day > 0 else 0

    return MaterialBalance(
        production_rate_kg_day=kg_day,
        production_rate_t_year=t_year,
        design_point=design_point,
        items=items,
        total_annual_feedstock_cost=total_cost,
        specific_feedstock_cost_per_kg=specific,
    )


def _get_mid_price(materials: List[RawMaterial], name: str) -> float:
    for m in materials:
        if m.name == name:
            return m.price_mid
    return 0.0


# ─── Electrolyte Recycling ────────────────────────────────────────────

@dataclass
class ImpurityLimit:
    """Maximum tolerable impurity concentration."""
    name: str
    max_g_L: float                      # g/L in electrolyte
    source: str                         # how it accumulates


DEFAULT_IMPURITY_LIMITS: List[ImpurityLimit] = [
    ImpurityLimit("Ni", 5.0, "Anode dissolution, NiSO4 makeup"),
    ImpurityLimit("S (as SO4)", 50.0, "H2SO4 makeup, anode oxidation"),
    ImpurityLimit("Organic", 2.0, "Additive decomposition products"),
    ImpurityLimit("Cl", 10.0, "Chloride impurity from water"),
]


@dataclass
class RecyclingEconomics:
    """Electrolyte recycling cost analysis."""
    fe2_depletion_rate_kg_day: float     # Fe2+ consumed by deposition
    fe2_makeup_rate_kg_day: float        # FeSO4 needed to replenish
    impurity_buildup: Dict[str, float]   # name -> accumulation rate (g/L/day)
    purge_rate_L_day: float              # electrolyte purge to maintain limits
    purge_fraction_per_day: float        # fraction of total volume purged daily
    makeup_chemical_cost_per_day: float  # cost of makeup chemicals
    makeup_chemical_cost_per_kg_fe: float
    purge_treatment_cost_per_day: float  # cost to neutralize/treat purge
    total_recycling_cost_per_kg_fe: float


def electrolyte_recycling(
    balance: MaterialBalance,
    impurity_limits: Optional[List[ImpurityLimit]] = None,
    purge_treatment_cost_per_L: float = 0.50,  # $/L of purge treated
) -> RecyclingEconomics:
    """Model electrolyte recycling and impurity management.

    Parameters
    ----------
    balance : MaterialBalance
        From material_balance().
    impurity_limits : list of ImpurityLimit, optional
        Override default impurity limits.
    purge_treatment_cost_per_L : float
        Cost to treat each litre of purge (neutralization, waste disposal).

    Returns
    -------
    RecyclingEconomics
    """
    if impurity_limits is None:
        impurity_limits = DEFAULT_IMPURITY_LIMITS

    dp = balance.design_point
    kg_day = balance.production_rate_kg_day
    V = dp.electrolyte_volume_L

    # Fe2+ depletion: 1 mol Fe per mol deposited
    fe2_depletion_mol_day = (kg_day * 1000) / 55.845  # mol/day
    fe2_depletion_g_day = fe2_depletion_mol_day * 55.845  # g/day Fe
    fe2_depletion_kg_day = fe2_depletion_g_day / 1000

    # Makeup: FeSO4·7H2O (278.01 g/mol Fe)
    fe2_makeup_kg_day = fe2_depletion_kg_day * (278.01 / 55.845)

    # Impurity accumulation rates (g/L/day)
    impurity_rates: Dict[str, float] = {}
    for imp in impurity_limits:
        if imp.name == "Ni":
            # From NiSO4 makeup + anode
            rate = dp.Ni2_mol_L * 58.69 * 0.001  # very rough: 0.1% leakage
        elif imp.name == "S (as SO4)":
            rate = 0.5  # g/L/day from H2SO4 additions
        elif imp.name == "Organic":
            rate = 0.05  # g/L/day from additive decomposition
        elif imp.name == "Cl":
            rate = 0.02  # g/L/day from water impurity
        else:
            rate = 0.0
        impurity_rates[imp.name] = rate

    # Purge rate: determined by worst-case impurity
    max_purge_fraction = 0.0
    for imp in impurity_limits:
        rate = impurity_rates.get(imp.name, 0)
        if rate > 0 and imp.max_g_L > 0:
            # At steady state: rate * V = purge * C_max  =>  purge = rate * V / C_max
            needed_purge = rate * V / imp.max_g_L  # L/day
            fraction = needed_purge / V
            if fraction > max_purge_fraction:
                max_purge_fraction = fraction

    purge_L_day = max_purge_fraction * V

    # Makeup chemical cost (from the material balance, scaled to daily)
    # Use the FeSO4 line item cost as the primary makeup cost
    feso4_item = next((it for it in balance.items if "Ferrous" in it.name), None)
    daily_fe_cost = feso4_item.daily_cost if feso4_item else 0.0
    h2so4_item = next((it for it in balance.items if "Sulfuric" in it.name), None)
    daily_acid_cost = h2so4_item.daily_cost if h2so4_item else 0.0
    makeup_cost_day = daily_fe_cost + daily_acid_cost

    purge_treat_day = purge_L_day * purge_treatment_cost_per_L

    total_day = makeup_cost_day + purge_treat_day
    total_per_kg = total_day / kg_day if kg_day > 0 else 0

    return RecyclingEconomics(
        fe2_depletion_rate_kg_day=fe2_depletion_kg_day,
        fe2_makeup_rate_kg_day=fe2_makeup_kg_day,
        impurity_buildup=impurity_rates,
        purge_rate_L_day=purge_L_day,
        purge_fraction_per_day=max_purge_fraction,
        makeup_chemical_cost_per_day=makeup_cost_day,
        makeup_chemical_cost_per_kg_fe=makeup_cost_day / kg_day if kg_day > 0 else 0,
        purge_treatment_cost_per_day=purge_treat_day,
        total_recycling_cost_per_kg_fe=total_per_kg,
    )


# ─── Siting Analysis ──────────────────────────────────────────────────

@dataclass
class LocationData:
    """Candidate plant location with scoring inputs."""
    name: str
    country: str
    # Electricity
    electricity_cost_kWh: float         # $/kWh (industrial)
    renewable_fraction: float           # 0-1 (fraction of grid that is renewable)
    # Water
    water_cost_per_m3: float            # $/m³ industrial water
    water_stress_index: float           # 0-1 (0 = abundant, 1 = severe stress)
    # Feedstock proximity
    distance_to_FeSO4_source_km: float  # nearest steel mill / TiO2 plant
    distance_to_acid_supplier_km: float
    # Regulatory & labor
    environmental_regulatory_score: float  # 0-1 (1 = most favorable / streamlined)
    labor_cost_per_hour: float          # $/hr (semi-skilled operator)
    # Infrastructure
    grid_reliability: float             # 0-1 (uptime fraction)
    logistics_score: float              # 0-1 (port/rail/road access)


@dataclass
class LocationWeight:
    """Weights for multi-criteria site scoring (must sum to ~1.0)."""
    electricity_cost: float = 0.25
    renewable_fraction: float = 0.10
    water_availability: float = 0.10
    feedstock_proximity: float = 0.20
    regulatory: float = 0.10
    labor_cost: float = 0.10
    grid_reliability: float = 0.10
    logistics: float = 0.05

    def total(self) -> float:
        return (self.electricity_cost + self.renewable_fraction
                + self.water_availability + self.feedstock_proximity
                + self.regulatory + self.labor_cost
                + self.grid_reliability + self.logistics)


@dataclass
class LocationScore:
    """Scored result for a single location."""
    name: str
    raw_scores: Dict[str, float]        # criterion -> 0-1 score
    weighted_scores: Dict[str, float]   # criterion -> weighted score
    total_score: float                  # sum of weighted scores
    annual_electricity_cost: float      # $/yr at given production rate


@dataclass
class LocationRanking:
    """Comparison of multiple candidate locations."""
    locations: List[LocationScore]
    design_point: DesignPoint
    production_rate_kg_day: float
    weights: LocationWeight


def site_score(
    location: LocationData,
    weights: Optional[LocationWeight] = None,
) -> LocationScore:
    """Score a single location against criteria.

    All raw scores normalized to 0-1 where 1 is best.

    Parameters
    ----------
    location : LocationData
    weights : LocationWeight, optional

    Returns
    -------
    LocationScore
    """
    if weights is None:
        weights = LocationWeight()

    raw: Dict[str, float] = {}

    # Electricity cost: lower is better. Normalize: 0 $/kWh → 1, 0.30 → 0
    raw["electricity_cost"] = max(0, 1.0 - location.electricity_cost_kWh / 0.30)

    # Renewable fraction: higher is better (direct)
    raw["renewable_fraction"] = location.renewable_fraction

    # Water availability: lower stress + lower cost is better
    water_avail = (1.0 - location.water_stress_index) * 0.6 + \
                  max(0, 1.0 - location.water_cost_per_m3 / 5.0) * 0.4
    raw["water_availability"] = water_avail

    # Feedstock proximity: shorter distance is better
    # Normalize: 0 km → 1, 2000 km → 0
    dist_combined = (location.distance_to_FeSO4_source_km +
                     location.distance_to_acid_supplier_km) / 2
    raw["feedstock_proximity"] = max(0, 1.0 - dist_combined / 2000)

    # Regulatory: direct score
    raw["regulatory"] = location.environmental_regulatory_score

    # Labor cost: lower is better. Normalize: 0 → 1, 80 $/hr → 0
    raw["labor_cost"] = max(0, 1.0 - location.labor_cost_per_hour / 80.0)

    # Grid reliability: direct
    raw["grid_reliability"] = location.grid_reliability

    # Logistics: direct
    raw["logistics"] = location.logistics_score

    # Weighted scores
    weighted: Dict[str, float] = {}
    for criterion in raw:
        w = getattr(weights, criterion, 0)
        weighted[criterion] = raw[criterion] * w

    total = sum(weighted.values())

    return LocationScore(
        name=location.name,
        raw_scores=raw,
        weighted_scores=weighted,
        total_score=total,
        annual_electricity_cost=0.0,  # filled by compare_locations
    )


def compare_locations(
    locations: List[LocationData],
    design_point: DesignPoint,
    production_rate_kg_day: float,
    weights: Optional[LocationWeight] = None,
    operating_days_per_year: float = 300.0,
) -> LocationRanking:
    """Score and rank all candidate locations.

    Parameters
    ----------
    locations : list of LocationData
    design_point : DesignPoint
    production_rate_kg_day : float
    weights : LocationWeight, optional
    operating_days_per_year : float

    Returns
    -------
    LocationRanking
        Locations sorted by total_score (highest first).
    """
    if weights is None:
        weights = LocationWeight()

    energy_kWh_per_kg = design_point.energy_kWh_per_kg()
    annual_kg = production_rate_kg_day * operating_days_per_year

    scored: List[LocationScore] = []
    for loc in locations:
        s = site_score(loc, weights)
        # Fill electricity cost
        annual_kWh = energy_kWh_per_kg * annual_kg
        s.annual_electricity_cost = annual_kWh * loc.electricity_cost_kWh
        scored.append(s)

    # Sort descending by total score
    scored.sort(key=lambda s: s.total_score, reverse=True)

    return LocationRanking(
        locations=scored,
        design_point=design_point,
        production_rate_kg_day=production_rate_kg_day,
        weights=weights,
    )


# ─── Candidate Locations ──────────────────────────────────────────────

CANDIDATE_LOCATIONS: List[LocationData] = [
    LocationData("Pittsburgh, PA", "USA", 0.06, 0.15, 2.0, 0.40,
                 50, 50, 0.7, 30.0, 0.95, 0.80),
    LocationData("Shandong Province", "China", 0.08, 0.25, 1.0, 0.20,
                 20, 30, 0.5, 8.0, 0.90, 0.70),
    LocationData("Duisburg, Germany", "Germany", 0.20, 0.45, 2.5, 0.60,
                 30, 40, 0.8, 40.0, 0.99, 0.90),
    LocationData("Jamshedpur, India", "India", 0.08, 0.20, 1.5, 0.30,
                 10, 20, 0.5, 5.0, 0.75, 0.60),
    LocationData("Vitória, Brazil", "Brazil", 0.07, 0.45, 1.8, 0.25,
                 40, 60, 0.6, 10.0, 0.80, 0.65),
    LocationData("Jeddah, Saudi Arabia", "Saudi Arabia", 0.05, 0.05, 3.0, 0.80,
                 500, 300, 0.6, 12.0, 0.85, 0.75),
    LocationData("Newcastle, Australia", "Australia", 0.10, 0.30, 2.0, 0.35,
                 80, 100, 0.8, 35.0, 0.97, 0.85),
    LocationData("Recife, Brazil", "Brazil", 0.07, 0.60, 2.0, 0.30,
                 200, 150, 0.5, 8.0, 0.70, 0.55),
]


# ─── Sensitivity ──────────────────────────────────────────────────────

@dataclass
class SensitivityResult:
    """Electricity price sensitivity analysis."""
    electricity_prices: List[float]     # $/kWh
    feedstock_costs: List[float]        # $/kg Fe
    recycling_costs: List[float]        # $/kg Fe
    total_costs: List[float]            # $/kg Fe (feedstock + recycling + electricity)
    electricity_costs_per_kg: List[float]


def electricity_sensitivity(
    design_point: DesignPoint,
    production_rate_kg_day: float,
    price_range: Tuple[float, float] = (0.02, 0.30),
    n_points: int = 15,
) -> SensitivityResult:
    """Sensitivity of total cost to electricity price.

    Parameters
    ----------
    design_point : DesignPoint
    production_rate_kg_day : float
    price_range : tuple of (low, high) $/kWh
    n_points : int

    Returns
    -------
    SensitivityResult
    """
    prices = [price_range[0] + (price_range[1] - price_range[0]) * i / (n_points - 1)
              for i in range(n_points)]

    balance = material_balance(design_point, production_rate_kg_day)
    recycling = electrolyte_recycling(balance)
    feedstock_per_kg = balance.specific_feedstock_cost_per_kg
    recycling_per_kg = recycling.total_recycling_cost_per_kg_fe
    energy_per_kg = design_point.energy_kWh_per_kg()

    feedstock_costs = [feedstock_per_kg] * n_points
    recycling_costs = [recycling_per_kg] * n_points
    elec_costs = [energy_per_kg * p for p in prices]
    total_costs = [f + r + e for f, r, e in zip(feedstock_costs, recycling_costs, elec_costs)]

    return SensitivityResult(
        electricity_prices=prices,
        feedstock_costs=feedstock_costs,
        recycling_costs=recycling_costs,
        total_costs=total_costs,
        electricity_costs_per_kg=elec_costs,
    )
