"""
Supply-chain input model for aqueous electrowinning of iron.

Compares delivered cost of iron across feedstock types and deployment
architectures (centralized vs. decentralized at mine/waste site).

Key insight: iron ore is ~60% Fe by weight — 40% is gangue you'd
otherwise transport.  Low-grade ores and waste streams have even worse
ratios.  Electrowinning can be small and modular, so shipping the
*plant* to the *feedstock* may beat shipping the feedstock to the plant.

Feedstock categories:
  - Ore (high-grade, low-grade, tailings)
  - Waste streams (pickle liquor, copperas, acid mine drainage, red mud)
  - Derived (spray-roasted Fe2O3, precipitated FeOOH)

Each feedstock carries:
  - Acquisition cost ($/t feedstock) — negative = paid to take it
  - Fe content (wt fraction)
  - Dissolution energy (kWh/t Fe) — grinding + leaching
  - Impurity burden (relative) — affects purification cost
  - Transport mode and cost ($/t-km)
  - Whether it's already dissolved

References
----------
- Humbert et al. (2024) — AHE CAPEX/OPEX benchmarks
- Kempler/Shekhar (2025) — porosity affects dissolution kinetics
- Copper SX-EW industry practice — mine-site deployment model
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import numpy as np


# ─── Feedstock definitions ─────────────────────────────────────────

@dataclass(frozen=True)
class Feedstock:
    """A candidate iron feedstock with its economic and technical profile."""

    name: str
    # Economics
    cost_per_t_feedstock: float      # $/t of feedstock as-received (negative = you get paid)
    fe_wt_fraction: float            # mass fraction Fe in feedstock (0-1)
    # Processing
    already_dissolved: bool = False  # True for pickle liquor, AMD
    dissolution_energy_kWh_per_t_Fe: float = 0.0  # grinding + leaching electricity
    dissolution_acid_cost_per_t_Fe: float = 0.0   # $/t Fe for acid consumption
    # Impurities
    impurity_burden: float = 0.0     # 0 (clean) to 1 (heavily contaminated)
    # Transport
    transport_mode: str = "truck"    # "truck", "rail", "pipeline", "onsite"
    transport_cost_per_t_km: float = 0.0  # $/t-km (of feedstock)
    # Location
    typical_location: str = "mine"   # "mine", "steel_mill", "waste_site", "refinery"
    notes: str = ""

    @property
    def cost_per_t_Fe(self) -> float:
        """Feedstock cost per tonne of contained iron."""
        if self.fe_wt_fraction <= 0:
            return float('inf')
        return self.cost_per_t_feedstock / self.fe_wt_fraction

    @property
    def tonnes_feedstock_per_tonne_Fe(self) -> float:
        """How many tonnes of feedstock needed per tonne of iron."""
        if self.fe_wt_fraction <= 0:
            return float('inf')
        return 1.0 / self.fe_wt_fraction


# ─── Default feedstock library ─────────────────────────────────────

FEEDSTOCKS: Dict[str, Feedstock] = {
    "high_grade_ore": Feedstock(
        name="High-grade hematite ore (62% Fe)",
        cost_per_t_feedstock=100.0,     # ~$100/t ore (2024 benchmark)
        fe_wt_fraction=0.62,
        dissolution_energy_kWh_per_t_Fe=64.0,   # Humbert 2024
        dissolution_acid_cost_per_t_Fe=30.0,
        impurity_burden=0.1,
        transport_mode="rail",
        transport_cost_per_t_km=0.02,
        typical_location="mine",
        notes="Standard iron ore. Bulk rail transport.",
    ),
    "low_grade_ore": Feedstock(
        name="Low-grade ore (30% Fe)",
        cost_per_t_feedstock=15.0,      # much cheaper per tonne
        fe_wt_fraction=0.30,
        dissolution_energy_kWh_per_t_Fe=80.0,   # harder to dissolve
        dissolution_acid_cost_per_t_Fe=50.0,
        impurity_burden=0.4,
        transport_mode="truck",
        transport_cost_per_t_km=0.05,
        typical_location="mine",
        notes="Sub-economic ore or waste dump material. 70% gangue.",
    ),
    "tailings": Feedstock(
        name="Iron tailings (15% Fe)",
        cost_per_t_feedstock=-10.0,     # paid to remediate
        fe_wt_fraction=0.15,
        dissolution_energy_kWh_per_t_Fe=100.0,
        dissolution_acid_cost_per_t_Fe=60.0,
        impurity_burden=0.6,
        transport_mode="truck",
        transport_cost_per_t_km=0.05,
        typical_location="mine",
        notes="Mine tailings. Very high gangue ratio. Remediation co-benefit.",
    ),
    "pickle_liquor": Feedstock(
        name="Spent pickle liquor (12% Fe as FeCl2/FeSO4)",
        cost_per_t_feedstock=-50.0,     # steel mills pay to dispose
        fe_wt_fraction=0.12,
        already_dissolved=True,
        dissolution_energy_kWh_per_t_Fe=0.0,
        dissolution_acid_cost_per_t_Fe=0.0,
        impurity_burden=0.2,
        transport_mode="truck",
        transport_cost_per_t_km=0.08,   # liquid, heavier
        typical_location="steel_mill",
        notes="Already dissolved. Negative cost. Available at steel mills.",
    ),
    "copperas": Feedstock(
        name="Copperas (FeSO4.7H2O) from TiO2 production",
        cost_per_t_feedstock=-20.0,     # negative cost
        fe_wt_fraction=0.20,
        already_dissolved=False,        # crystals, but water-soluble
        dissolution_energy_kWh_per_t_Fe=5.0,   # just dissolve in water
        dissolution_acid_cost_per_t_Fe=0.0,
        impurity_burden=0.3,
        transport_mode="truck",
        transport_cost_per_t_km=0.04,
        typical_location="refinery",
        notes="Water-soluble crystals. Byproduct of TiO2 sulfate process.",
    ),
    "acid_mine_drainage": Feedstock(
        name="Acid mine drainage (2% Fe)",
        cost_per_t_feedstock=-80.0,     # environmental remediation value
        fe_wt_fraction=0.02,
        already_dissolved=True,
        dissolution_energy_kWh_per_t_Fe=0.0,
        dissolution_acid_cost_per_t_Fe=0.0,
        impurity_burden=0.5,
        transport_mode="pipeline",
        transport_cost_per_t_km=0.01,   # pipeline, very cheap per tonne
        typical_location="mine",
        notes="Very dilute but free and environmentally beneficial. "
              "Concentration step needed before electrowinning.",
    ),
    "red_mud": Feedstock(
        name="Red mud / bauxite residue (30% Fe)",
        cost_per_t_feedstock=-30.0,     # alumina refineries pay
        fe_wt_fraction=0.30,
        dissolution_energy_kWh_per_t_Fe=70.0,
        dissolution_acid_cost_per_t_Fe=40.0,
        impurity_burden=0.5,
        transport_mode="truck",
        transport_cost_per_t_km=0.05,
        typical_location="refinery",
        notes="Highly alkaline, contains Al/Ti/Na impurities. "
              "SIDERWIN demonstrated electrolysis from red mud.",
    ),
    "spray_roasted_fe2o3": Feedstock(
        name="Spray-roasted Fe2O3 from pickle liquor",
        cost_per_t_feedstock=80.0,      # processed product
        fe_wt_fraction=0.70,
        dissolution_energy_kWh_per_t_Fe=40.0,   # porous, dissolves easily
        dissolution_acid_cost_per_t_Fe=20.0,
        impurity_burden=0.1,
        transport_mode="rail",
        transport_cost_per_t_km=0.02,
        typical_location="steel_mill",
        notes="Ruthner process product. Already porous (Kempler-friendly). "
              "Concentrated Fe form from pickle liquor.",
    ),
    "mill_scale": Feedstock(
        name="Mill scale (72% Fe as Fe3O4/FeO)",
        cost_per_t_feedstock=40.0,
        fe_wt_fraction=0.72,
        dissolution_energy_kWh_per_t_Fe=50.0,
        dissolution_acid_cost_per_t_Fe=25.0,
        impurity_burden=0.1,
        transport_mode="truck",
        transport_cost_per_t_km=0.04,
        typical_location="steel_mill",
        notes="Oxide scale from hot rolling. Clean, high Fe. "
              "Available at steel mills.",
    ),
}


# ─── Transport model ───────────────────────────────────────────────

@dataclass
class TransportModel:
    """Simple transport cost calculator."""

    # Cost per tonne-km by mode ($/t-km, 2024 approximate)
    cost_truck: float = 0.05           # $/t-km (short-haul)
    cost_rail: float = 0.02            # $/t-km (long-haul bulk)
    cost_pipeline: float = 0.01        # $/t-km (liquid/slurry)
    cost_onsite: float = 0.0           # no transport

    def cost_per_t_km(self, mode: str) -> float:
        return {
            "truck": self.cost_truck,
            "rail": self.cost_rail,
            "pipeline": self.cost_pipeline,
            "onsite": self.cost_onsite,
        }.get(mode, self.cost_truck)

    def transport_cost(
        self,
        tonnes: float,
        distance_km: float,
        mode: str = "truck",
    ) -> float:
        """Total transport cost ($) for given tonnage and distance."""
        return tonnes * distance_km * self.cost_per_t_km(mode)


# ─── Deployment architecture ───────────────────────────────────────

@dataclass
class DeploymentScenario:
    """
    Compares centralized vs decentralized deployment.

    Centralized: ship feedstock to a large central plant.
    Decentralized: ship a small modular plant to the feedstock source.
    """

    # Plant economics
    plant_capacity_t_Fe_yr: float = 1000.0      # annual iron production
    plant_capex_centralized: float = 3_000_000   # $ for centralized plant
    plant_capex_modular: float = 1_500_000       # $ for small modular plant
    modular_overhead_factor: float = 1.15        # cost penalty for small scale

    # Operating costs (per t Fe, excluding feedstock and transport)
    opex_per_t_Fe: float = 200.0       # electricity + labor + consumables

    # Transport
    transport: TransportModel = field(default_factory=TransportModel)

    def analyze(
        self,
        feedstock: Feedstock,
        distance_to_central_plant_km: float = 500.0,
        distance_product_to_market_km: float = 200.0,
        feedstock_concentration_factor: float = 1.0,
        plant_lifetime_yr: int = 20,
    ) -> Dict[str, Any]:
        """
        Compare centralized vs decentralized for a given feedstock.

        Parameters
        ----------
        feedstock : Feedstock
            The feedstock to evaluate.
        distance_to_central_plant_km : float
            Distance from feedstock source to centralized plant.
        distance_product_to_market_km : float
            Distance from either plant to the product market.
        feedstock_concentration_factor : float
            For dilute feedstocks (AMD), how much pre-concentration
            is needed on-site before electrowinning (1.0 = no concentration).
        plant_lifetime_yr : int
            Plant lifetime for annualized CAPEX.
        """
        annual_prod = self.plant_capacity_t_Fe_yr
        t_feed_per_t_Fe = feedstock.tonnes_feedstock_per_tonne_Fe

        # ── Centralized: ship feedstock to plant ──
        feed_transport_central = self.transport.transport_cost(
            t_feed_per_t_Fe,
            distance_to_central_plant_km,
            feedstock.transport_mode,
        )
        product_transport_central = self.transport.transport_cost(
            1.0,
            distance_product_to_market_km,
            "rail",
        )
        capex_annual_central = self.plant_capex_centralized / plant_lifetime_yr
        total_centralized = (
            feedstock.cost_per_t_Fe
            + feed_transport_central
            + product_transport_central
            + self.opex_per_t_Fe
            + feedstock.dissolution_energy_kWh_per_t_Fe * 0.04  # electricity
            + feedstock.dissolution_acid_cost_per_t_Fe
            + capex_annual_central / annual_prod
        )

        # ── Decentralized: ship plant to feedstock, ship product out ──
        # Plant goes to the feedstock, so feedstock transport = 0
        # But modular plant has cost penalty
        mod_capex = self.plant_capex_modular * self.modular_overhead_factor
        capex_annual_mod = mod_capex / plant_lifetime_yr

        # For dilute feedstocks, pre-concentration cost
        concentration_cost = 0.0
        if feedstock_concentration_factor > 1.0:
            # Rough estimate: evaporation/pumping cost for dilute streams
            concentration_cost = (feedstock_concentration_factor - 1.0) * 20.0  # $/t Fe

        # Product still needs to get to market
        product_transport_decentral = self.transport.transport_cost(
            1.0,
            distance_product_to_market_km,
            "rail",
        )
        total_decentralized = (
            feedstock.cost_per_t_Fe
            + 0.0  # no feedstock transport (plant is there)
            + product_transport_decentral
            + self.opex_per_t_Fe * self.modular_overhead_factor  # small scale penalty
            + feedstock.dissolution_energy_kWh_per_t_Fe * 0.04
            + feedstock.dissolution_acid_cost_per_t_Fe
            + concentration_cost
            + capex_annual_mod / annual_prod
        )

        savings = total_centralized - total_decentralized

        return {
            "feedstock": feedstock.name,
            "fe_content_pct": feedstock.fe_wt_fraction * 100,
            "t_feed_per_t_Fe": round(t_feed_per_t_Fe, 1),
            "already_dissolved": feedstock.already_dissolved,
            "centralized": {
                "feedstock_cost": round(feedstock.cost_per_t_Fe, 2),
                "feed_transport": round(feed_transport_central, 2),
                "product_transport": round(product_transport_central, 2),
                "processing": round(self.opex_per_t_Fe, 2),
                "capex_annualized": round(capex_annual_central / annual_prod, 2),
                "total_per_t_Fe": round(total_centralized, 2),
            },
            "decentralized": {
                "feedstock_cost": round(feedstock.cost_per_t_Fe, 2),
                "feed_transport": 0.0,
                "product_transport": round(product_transport_decentral, 2),
                "processing": round(self.opex_per_t_Fe * self.modular_overhead_factor, 2),
                "capex_annualized": round(capex_annual_mod / annual_prod, 2),
                "concentration": round(concentration_cost, 2),
                "total_per_t_Fe": round(total_decentralized, 2),
            },
            "savings_per_t_Fe": round(savings, 2),
            "decentral_wins": savings > 0,
            "transport_sensitivity": self._transport_sensitivity(
                feedstock, distance_product_to_market_km, feedstock_concentration_factor
            ),
        }

    def _transport_sensitivity(
        self,
        feedstock: Feedstock,
        product_dist: float,
        conc_factor: float,
    ) -> Dict[str, Any]:
        """At what distance does decentralized become cheaper?"""
        t_feed = feedstock.tonnes_feedstock_per_tonne_Fe
        feed_mode_cost = self.transport.cost_per_t_km(feedstock.transport_mode)

        # Centralized cost increases with feedstock distance
        # Decentralized cost is constant (no feedstock transport)
        # Break-even: feed_transport_central = capex_diff + processing_penalty
        mod_capex = self.plant_capex_modular * self.modular_overhead_factor
        capex_annual_mod = mod_capex / 20.0
        capex_annual_central = self.plant_capex_centralized / 20.0
        capex_diff = (capex_annual_mod - capex_annual_central) / self.plant_capacity_t_Fe_yr
        processing_penalty = self.opex_per_t_Fe * (self.modular_overhead_factor - 1.0)
        conc_cost = (conc_factor - 1.0) * 20.0 if conc_factor > 1.0 else 0.0

        # Break-even distance: t_feed * dist * feed_mode_cost = capex_diff + processing_penalty + conc_cost
        rhs = capex_diff + processing_penalty + conc_cost
        if t_feed * feed_mode_cost > 0:
            break_even_km = rhs / (t_feed * feed_mode_cost)
        else:
            break_even_km = float('inf')

        return {
            "break_even_feedstock_distance_km": round(break_even_km, 0),
            "interpretation": (
                f"Decentralized wins if feedstock is >{break_even_km:.0f} km "
                f"from a centralized plant"
            ),
        }


def run_full_comparison(
    distance_km: float = 500.0,
    product_market_km: float = 200.0,
) -> List[Dict[str, Any]]:
    """
    Run the centralized vs decentralized comparison for all feedstocks.

    Returns sorted list (best decentralized candidate first).
    """
    scenario = DeploymentScenario()
    results = []

    for key, fs in FEEDSTOCKS.items():
        # For AMD, need concentration (50x)
        conc = 50.0 if fs.name.startswith("Acid mine") else 1.0
        r = scenario.analyze(
            fs,
            distance_to_central_plant_km=distance_km,
            distance_product_to_market_km=product_market_km,
            feedstock_concentration_factor=conc,
        )
        results.append(r)

    results.sort(key=lambda x: x["savings_per_t_Fe"], reverse=True)
    return results


def print_comparison_table(results: List[Dict[str, Any]]) -> str:
    """Pretty-print the comparison as a table."""
    lines = []
    lines.append(f"{'Feedstock':<42} {'Fe%':>4} {'Central':>10} {'Decentral':>10} {'Savings':>8} {'Wins?':>5}")
    lines.append("-" * 85)
    for r in results:
        c = r["centralized"]["total_per_t_Fe"]
        d = r["decentralized"]["total_per_t_Fe"]
        s = r["savings_per_t_Fe"]
        w = "YES" if r["decentral_wins"] else "no"
        lines.append(
            f"{r['feedstock']:<42} {r['fe_content_pct']:>3.0f}% "
            f"${c:>8.0f} ${d:>8.0f} ${s:>6.0f} {w:>5}"
        )
    return "\n".join(lines)
