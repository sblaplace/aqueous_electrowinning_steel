"""
Life Cycle Assessment (LCA) for aqueous electrowinning of iron/steel.

Cradle-to-gate assessment: ore/chemicals → electrolyte prep → electrowinning
→ heat treatment → product steel.

Impact categories:
  1. Global warming potential  (kg CO₂-eq / kg steel)
  2. Acidification             (kg SO₂-eq / kg steel)
  3. Eutrophication            (kg PO₄-eq / kg steel)
  4. Water consumption         (L / kg steel)
  5. Land use                  (m² / kg steel, for renewable electricity)

References
----------
- IPCC AR6 (2021) — GWP100 characterisation factors.
- Hischier et al. (2010), ecoinvent v3 — background inventory data.
- BOF/EAF/DRI ranges: World Steel Assn (2021), IEA Iron & Steel Tech. Roadmap.
- Electrowinning energy: derived from electrochemistry model in this project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# ── Characterisation factors (midpoint, per unit material) ─────────────
# Electricity generation (global average mixes)
ELECTRICITY_MIX_CF = {
    "coal": {
        "gwp_kgCO2eq_per_kWh": 0.95,
        "acidification_kgSO2eq_per_kWh": 6.0e-3,
        "eutrophication_kgPO4eq_per_kWh": 1.2e-4,
        "water_L_per_kWh": 2.2,
        "land_use_m2_per_kWh": 0.0,
    },
    "natural_gas": {
        "gwp_kgCO2eq_per_kWh": 0.45,
        "acidification_kgSO2eq_per_kWh": 1.5e-3,
        "eutrophication_kgPO4eq_per_kWh": 4.0e-5,
        "water_L_per_kWh": 0.7,
        "land_use_m2_per_kWh": 0.0,
    },
    "grid_avg": {
        "gwp_kgCO2eq_per_kWh": 0.475,
        "acidification_kgSO2eq_per_kWh": 3.0e-3,
        "eutrophication_kgPO4eq_per_kWh": 7.0e-5,
        "water_L_per_kWh": 1.5,
        "land_use_m2_per_kWh": 0.0,
    },
    "renewable": {
        "gwp_kgCO2eq_per_kWh": 0.011,
        "acidification_kgSO2eq_per_kWh": 4.0e-5,
        "eutrophication_kgPO4eq_per_kWh": 6.0e-6,
        "water_L_per_kWh": 0.05,
        "land_use_m2_per_kWh": 0.005,
    },
}

# Chemical inputs per kg of iron produced (electrolyte prep + makeup)
CHEMICAL_CF = {
    "iron_ore_per_kg_fe": 1.6,          # kg ore / kg Fe (stoichiometric + losses)
    "acid_per_kg_fe": 0.15,             # kg H₂SO₄ / kg Fe (electrolyte makeup)
    "additives_per_kg_fe": 0.02,        # kg boric acid + surfactant / kg Fe
    "water_process_L_per_kg_fe": 8.0,   # L / kg Fe (electrolyte volume, rinsing)
}

CHEMICAL_GWP = {
    "iron_ore_kgCO2eq_per_kg": 0.035,
    "acid_kgCO2eq_per_kg": 0.55,
    "additives_kgCO2eq_per_kg": 1.2,
}

CHEMICAL_ACIDIFICATION = {
    "iron_ore_kgSO2eq_per_kg": 2.0e-4,
    "acid_kgSO2eq_per_kg": 3.0e-3,
    "additives_kgSO2eq_per_kg": 5.0e-3,
}

CHEMICAL_EUTROPHICATION = {
    "iron_ore_kgPO4eq_per_kg": 5.0e-5,
    "acid_kgPO4eq_per_kg": 8.0e-5,
    "additives_kgPO4eq_per_kg": 1.5e-4,
}

# Heat treatment (furnace gas, natural gas fired)
HEAT_TREATMENT_GWP_per_kg = 0.15       # kg CO₂-eq / kg steel (natural gas furnace)
HEAT_TREATMENT_ACID_per_kg = 4.0e-4    # kg SO₂-eq / kg steel
HEAT_TREATMENT_EUTRO_per_kg = 8.0e-6   # kg PO₄-eq / kg steel
HEAT_TREATMENT_WATER_L_per_kg = 0.5    # L / kg (quenching)

# Waste treatment baseline
WASTE_TREATMENT_GWP_per_kg = 0.02      # kg CO₂-eq / kg steel
WASTE_TREATMENT_WATER_L_per_kg = 1.0   # L / kg


# ── Reference routes (literature ranges) ───────────────────────────────
REFERENCE_ROUTES = {
    "BOF (primary)": {
        "gwp_low": 1.8,
        "gwp_mid": 2.0,
        "gwp_high": 2.2,
        "acidification": 0.012,
        "eutrophication": 3.0e-4,
        "water_L_per_kg": 45.0,
        "land_use_m2_per_kg": 0.0,
        "notes": "Blast furnace + basic oxygen furnace; coal-based reduction",
    },
    "EAF (scrap)": {
        "gwp_low": 0.4,
        "gwp_mid": 0.55,
        "gwp_high": 0.7,
        "acidification": 4.0e-3,
        "eutrophication": 1.5e-4,
        "water_L_per_kg": 12.0,
        "land_use_m2_per_kg": 0.0,
        "notes": "Electric arc furnace, scrap-based; depends on grid mix",
    },
    "DRI-EAF (natural gas)": {
        "gwp_low": 0.6,
        "gwp_mid": 0.9,
        "gwp_high": 1.2,
        "acidification": 5.0e-3,
        "eutrophication": 2.0e-4,
        "water_L_per_kg": 25.0,
        "land_use_m2_per_kg": 0.0,
        "notes": "Midrex/HYL process with natural gas + EAF melting",
    },
    "DRI-EAF (green H₂)": {
        "gwp_low": 0.05,
        "gwp_mid": 0.10,
        "gwp_high": 0.20,
        "acidification": 1.5e-3,
        "eutrophication": 5.0e-5,
        "water_L_per_kg": 35.0,
        "land_use_m2_per_kg": 0.002,
        "notes": "Green H₂-DRI + EAF; low GWP but high water for H₂ electrolysis",
    },
}


# ── Data classes ───────────────────────────────────────────────────────
@dataclass
class ElectricityMix:
    """Weighted electricity source mix."""
    coal: float = 0.0
    natural_gas: float = 0.0
    grid_avg: float = 0.0
    renewable: float = 1.0

    def __post_init__(self):
        total = self.coal + self.natural_gas + self.grid_avg + self.renewable
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Mix fractions must sum to 1.0, got {total:.4f}")

    def cf(self, key: str) -> float:
        """Weighted characterisation factor."""
        return (
            self.coal * ELECTRICITY_MIX_CF["coal"][key]
            + self.natural_gas * ELECTRICITY_MIX_CF["natural_gas"][key]
            + self.grid_avg * ELECTRICITY_MIX_CF["grid_avg"][key]
            + self.renewable * ELECTRICITY_MIX_CF["renewable"][key]
        )


@dataclass
class ChemicalSources:
    """Sourcing options for chemical inputs."""
    ore_origin: str = "average"         # "average", "high_grade", "recycled"
    acid_recycling_fraction: float = 0.0  # 0–1

    def ore_factor(self) -> float:
        return {"average": 1.0, "high_grade": 0.85, "recycled": 0.30}.get(
            self.ore_origin, 1.0
        )

    def acid_factor(self) -> float:
        return max(0.0, 1.0 - self.acid_recycling_fraction)


@dataclass
class LCAResult:
    """Complete LCA result for 1 kg of product steel."""
    gwp_kgCO2eq: float          # Global warming potential
    acidification_kgSO2eq: float
    eutrophication_kgPO4eq: float
    water_L: float
    land_use_m2: float
    # Breakdowns
    electricity_gwp: float = 0.0
    chemicals_gwp: float = 0.0
    heat_treatment_gwp: float = 0.0
    waste_gwp: float = 0.0
    # Metadata
    specific_energy_kWh_per_kg: float = 0.0
    electricity_mix: Optional[ElectricityMix] = None

    def to_dict(self) -> dict:
        return {
            "GWP (kg CO₂-eq/kg)": round(self.gwp_kgCO2eq, 4),
            "Acidification (kg SO₂-eq/kg)": round(self.acidification_kgSO2eq, 6),
            "Eutrophication (kg PO₄-eq/kg)": round(self.eutrophication_kgPO4eq, 8),
            "Water consumption (L/kg)": round(self.water_L, 2),
            "Land use (m²/kg)": round(self.land_use_m2, 6),
            "Electricity GWP share": round(self.electricity_gwp / max(self.gwp_kgCO2eq, 1e-12), 4),
            "Specific energy (kWh/kg)": round(self.specific_energy_kWh_per_kg, 2),
        }


@dataclass
class ComparisonRow:
    """Single row in a route comparison table."""
    route: str
    gwp_low: float
    gwp_mid: float
    gwp_high: float
    acidification: float
    eutrophication: float
    water_L_per_kg: float
    land_use_m2_per_kg: float


@dataclass
class ComparisonTable:
    """Comparison of electrowinning LCA against reference steelmaking routes."""
    electrowinning: LCAResult
    routes: List[ComparisonRow] = field(default_factory=list)

    def to_dict(self) -> dict:
        out = {}
        out["Aqueous Electrowinning"] = self.electrowinning.to_dict()
        for r in self.routes:
            out[r.route] = {
                "GWP (kg CO₂-eq/kg)": f"{r.gwp_low:.2f}–{r.gwp_high:.2f} (mid {r.gwp_mid:.2f})",
                "Acidification (kg SO₂-eq/kg)": r.acidification,
                "Eutrophication (kg PO₄-eq/kg)": r.eutrophication,
                "Water consumption (L/kg)": r.water_L_per_kg,
                "Land use (m²/kg)": r.land_use_m2_per_kg,
            }
        return out


# ── Core functions ─────────────────────────────────────────────────────
def compute_lca(
    specific_energy_kWh_per_kg: float,
    electricity_mix: Optional[ElectricityMix] = None,
    chemical_sources: Optional[ChemicalSources] = None,
    current_efficiency: float = 0.90,
) -> LCAResult:
    """
    Compute cradle-to-gate LCA for 1 kg of electrowinning steel.

    Parameters
    ----------
    specific_energy_kWh_per_kg : float
        Electrical energy input per kg of iron produced (from electrochemistry model).
    electricity_mix : ElectricityMix
        Weighted electricity source mix. Defaults to 100% renewable.
    chemical_sources : ChemicalSources
        Sourcing options for ore and chemicals.
    current_efficiency : float
        Faradaic efficiency — affects effective chemical consumption.
    """
    if electricity_mix is None:
        electricity_mix = ElectricityMix()
    if chemical_sources is None:
        chemical_sources = ChemicalSources()

    # Effective energy per kg product (accounting for current efficiency losses)
    eff_energy = specific_energy_kWh_per_kg / max(current_efficiency, 0.5)

    # ── Electricity impacts ────────────────────────────────────────────
    elec_gwp = eff_energy * electricity_mix.cf("gwp_kgCO2eq_per_kWh")
    elec_acid = eff_energy * electricity_mix.cf("acidification_kgSO2eq_per_kWh")
    elec_eutro = eff_energy * electricity_mix.cf("eutrophication_kgPO4eq_per_kWh")
    elec_water = eff_energy * electricity_mix.cf("water_L_per_kWh")
    elec_land = eff_energy * electricity_mix.cf("land_use_m2_per_kWh")

    # ── Chemical / ore impacts ─────────────────────────────────────────
    ore_f = chemical_sources.ore_factor()
    acid_f = chemical_sources.acid_factor()

    chem_gwp = (
        CHEMICAL_CF["iron_ore_per_kg_fe"] * ore_f * CHEMICAL_GWP["iron_ore_kgCO2eq_per_kg"]
        + CHEMICAL_CF["acid_per_kg_fe"] * acid_f * CHEMICAL_GWP["acid_kgCO2eq_per_kg"]
        + CHEMICAL_CF["additives_per_kg_fe"] * CHEMICAL_GWP["additives_kgCO2eq_per_kg"]
    )
    chem_acid = (
        CHEMICAL_CF["iron_ore_per_kg_fe"] * ore_f * CHEMICAL_ACIDIFICATION["iron_ore_kgSO2eq_per_kg"]
        + CHEMICAL_CF["acid_per_kg_fe"] * acid_f * CHEMICAL_ACIDIFICATION["acid_kgSO2eq_per_kg"]
        + CHEMICAL_CF["additives_per_kg_fe"] * CHEMICAL_ACIDIFICATION["additives_kgSO2eq_per_kg"]
    )
    chem_eutro = (
        CHEMICAL_CF["iron_ore_per_kg_fe"] * ore_f * CHEMICAL_EUTROPHICATION["iron_ore_kgPO4eq_per_kg"]
        + CHEMICAL_CF["acid_per_kg_fe"] * acid_f * CHEMICAL_EUTROPHICATION["acid_kgPO4eq_per_kg"]
        + CHEMICAL_CF["additives_per_kg_fe"] * CHEMICAL_EUTROPHICATION["additives_kgPO4eq_per_kg"]
    )
    chem_water = CHEMICAL_CF["water_process_L_per_kg_fe"] * ore_f

    # ── Heat treatment ─────────────────────────────────────────────────
    ht_gwp = HEAT_TREATMENT_GWP_per_kg
    ht_acid = HEAT_TREATMENT_ACID_per_kg
    ht_eutro = HEAT_TREATMENT_EUTRO_per_kg
    ht_water = HEAT_TREATMENT_WATER_L_per_kg

    # ── Waste treatment ────────────────────────────────────────────────
    wt_gwp = WASTE_TREATMENT_GWP_per_kg
    wt_water = WASTE_TREATMENT_WATER_L_per_kg

    # ── Aggregate ──────────────────────────────────────────────────────
    gwp = elec_gwp + chem_gwp + ht_gwp + wt_gwp
    acid = elec_acid + chem_acid + ht_acid
    eutro = elec_eutro + chem_eutro + ht_eutro
    water = elec_water + chem_water + ht_water + wt_water
    land = elec_land

    return LCAResult(
        gwp_kgCO2eq=gwp,
        acidification_kgSO2eq=acid,
        eutrophication_kgPO4eq=eutro,
        water_L=water,
        land_use_m2=land,
        electricity_gwp=elec_gwp,
        chemicals_gwp=chem_gwp,
        heat_treatment_gwp=ht_gwp,
        waste_gwp=wt_gwp,
        specific_energy_kWh_per_kg=eff_energy,
        electricity_mix=electricity_mix,
    )


def compare_routes(lca_result: LCAResult, reference_routes: Optional[dict] = None) -> ComparisonTable:
    """Build a comparison table of electrowinning vs reference routes."""
    if reference_routes is None:
        reference_routes = REFERENCE_ROUTES

    rows = []
    for name, data in reference_routes.items():
        rows.append(ComparisonRow(
            route=name,
            gwp_low=data["gwp_low"],
            gwp_mid=data["gwp_mid"],
            gwp_high=data["gwp_high"],
            acidification=data["acidification"],
            eutrophication=data["eutrophication"],
            water_L_per_kg=data["water_L_per_kg"],
            land_use_m2_per_kg=data.get("land_use_m2_per_kg", 0.0),
        ))

    return ComparisonTable(electrowinning=lca_result, routes=rows)


def sensitivity_to_electricity(
    specific_energy_kWh_per_kg: float,
    mixes: Dict[str, ElectricityMix],
    chemical_sources: Optional[ChemicalSources] = None,
    current_efficiency: float = 0.90,
) -> Dict[str, LCAResult]:
    """
    Compute LCA under different electricity mixes.

    Returns dict mapping mix name → LCAResult.
    """
    if chemical_sources is None:
        chemical_sources = ChemicalSources()

    results = {}
    for name, mix in mixes.items():
        results[name] = compute_lca(
            specific_energy_kWh_per_kg,
            electricity_mix=mix,
            chemical_sources=chemical_sources,
            current_efficiency=current_efficiency,
        )
    return results


def breakeven_renewable_fraction(
    specific_energy_kWh_per_kg: float,
    target_co2_kg_per_kg: float,
    chemical_sources: Optional[ChemicalSources] = None,
    current_efficiency: float = 0.90,
) -> float:
    """
    Find the minimum renewable fraction in a coal/renewable mix
    to achieve the target GWP.

    Returns fraction in [0, 1]. Returns 0.0 if already below target
    with 100% coal; returns 1.0 if even 100% renewable exceeds target.
    """
    if chemical_sources is None:
        chemical_sources = ChemicalSources()

    # Check endpoints
    coal_mix = ElectricityMix(coal=1.0, renewable=0.0)
    renew_mix = ElectricityMix(renewable=1.0)

    coal_result = compute_lca(
        specific_energy_kWh_per_kg, electricity_mix=coal_mix,
        chemical_sources=chemical_sources, current_efficiency=current_efficiency,
    )
    renew_result = compute_lca(
        specific_energy_kWh_per_kg, electricity_mix=renew_mix,
        chemical_sources=chemical_sources, current_efficiency=current_efficiency,
    )

    if coal_result.gwp_kgCO2eq <= target_co2_kg_per_kg:
        return 0.0
    if renew_result.gwp_kgCO2eq >= target_co2_kg_per_kg:
        return 1.0

    # Binary search
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        mix = ElectricityMix(coal=1.0 - mid, renewable=mid)
        result = compute_lca(
            specific_energy_kWh_per_kg, electricity_mix=mix,
            chemical_sources=chemical_sources, current_efficiency=current_efficiency,
        )
        if result.gwp_kgCO2eq > target_co2_kg_per_kg:
            lo = mid
        else:
            hi = mid

    return round((lo + hi) / 2.0, 4)
