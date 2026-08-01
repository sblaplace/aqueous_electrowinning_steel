"""
Crate — structural & environmental model for the autonomous production unit.

The cell twin answers "is the chemistry right?"; this module answers
"What does the crate do in the wind, the rain, and on this ground, and how
does it need to be mounted?"  It is the **unit/crate layer** of the
whole-system twin (see docs/SYSTEM_TWIN.md): the containerized autonomous
unit treated as a physical object sitting on a real site under real weather.

It is deliberately screening-grade, in the same spirit as the rest of the
repository:

* wind loading uses a simplified dynamic-pressure + drag-coefficient model
  (not ASCE finite-element) over the crate's projected area;
* stability is checked by overturning, sliding, bearing-pressure and uplift
  factors of safety using rigid-body statics;
* rain/ingress is a *risk* assessment driving a sealing/drainage/mounting
  recommendation, not a CFD roof model.

Every number here is an unvalidated screening estimate until a real load test
or site survey checks it (per-layer Level 0 on the credibility ladder).  The
target factors of safety are screening values to be superseded by the
jurisdiction civil code at the named-beachhead stage.

Layer data flow:  Site (ground + climate + loads) -> Crate -> StabilityVerdict
The verdict feeds the dark_mill site go/no-go and the operating twin's
environmental safe-state limits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, Optional

G = 9.81
RHO_AIR_SEA_LEVEL = 1.225  # kg/m^3 at 15 C, sea level
# Simplified terrain exposure multiplier on dynamic pressure (ASCE-style):
#   1.00 = open / flat / water (cat C), 0.85 = suburban (cat B), 0.70 = dense urban
_TERRAIN_MULT = {"open": 1.00, "suburban": 0.85, "urban": 0.70}
_TERRAIN_LABEL = {
    "open": "open (category C)", "suburban": "suburban (category B)", "urban": "dense urban",
}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrateSpec:
    """Physical envelope of the autonomous unit (default ~ a 40-ft container)."""

    length_m: float = 12.19
    width_m: float = 2.44
    height_m: float = 2.59
    mass_kg: float = 4500.0      # total non-ballast mass (frame + equipment)
    drag_coefficient: float = 1.2   # box / shipping-container shape

    @property
    def footprint_m2(self) -> float:
        return self.length_m * self.width_m

    @property
    def empty_weight_N(self) -> float:
        return self.mass_kg * G

    def projected_area_m2(self, wind: "WindLoad") -> float:
        """Projected area normal to the wind direction (broadside vs end-on)."""
        if wind.direction == "end":
            return self.length_m * self.height_m
        return self.width_m * self.height_m  # broadside (default, worst)

    def pivot_arm_m(self, wind: "WindLoad") -> float:
        """Lever arm about the downwind edge for the restoring moment."""
        if wind.direction == "end":
            return self.width_m / 2.0
        return self.length_m / 2.0


@dataclass(frozen=True)
class WindLoad:
    """Design wind at the site (3-second gust) + exposure."""

    gust_m_s: float = 40.0              # design 3-s gust (m/s) at the site
    direction: str = "broadside"         # "broadside" (worst) or "end"
    terrain: str = "open"                # "open" | "suburban" | "urban"
    altitude_m: float = 0.0
    temperature_C: float = 25.0

    @property
    def air_density_kg_m3(self) -> float:
        return RHO_AIR_SEA_LEVEL * math.exp(-self.altitude_m / 8500.0)

    @property
    def terrain_multiplier(self) -> float:
        return _TERRAIN_MULT.get(self.terrain, 1.0)

    def dynamic_pressure_Pa(self) -> float:
        return (
            0.5 * self.air_density_kg_m3 * self.gust_m_s ** 2 * self.terrain_multiplier
        )

    def load_label(self) -> str:
        return (
            f"{self.gust_m_s} m/s gust, {_TERRAIN_LABEL.get(self.terrain, self.terrain)}"
        )


@dataclass(frozen=True)
class GroundSpec:
    """Ground / foundation conditions at the deployment site."""

    p_allow_kPa: float = 100.0          # allowable bearing pressure
    friction_mu: float = 0.5            # soil-footer friction coefficient
    drainable: bool = True              # site drains away from the unit
    flood_depth_m: float = 0.0          # plausible flood depth at grade
    anchored: bool = False              # pre-provided tie-down/anchorage


@dataclass(frozen=True)
class EnvironmentalLoads:
    """Additional environmental loads (rain, snow, seismic) at the site."""

    rain_intensity_mm_hr: float = 50.0   # design rainfall intensity
    sealing_class: str = "industrial"    # "industrial" | "sealed" (IP/NEMA)
    snow_load_kPa: float = 0.0           # design rooftop snow load
    seismic_base_coefficient: Optional[float] = None  # e.g. 0.15 (fraction of W)


@dataclass(frozen=True)
class CrateConfig:
    """Full state for a crate stability evaluation."""

    crate: CrateSpec = field(default_factory=CrateSpec)
    wind: WindLoad = field(default_factory=WindLoad)
    ground: GroundSpec = field(default_factory=GroundSpec)
    env: EnvironmentalLoads = field(default_factory=EnvironmentalLoads)
    ballast_kg: float = 0.0
    target_fs_overturn: float = 1.5
    target_fs_slide: float = 1.5

    def with_(self, **kw: float) -> "CrateConfig":
        return replace(self, **kw)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    name: str
    value: float
    limit: float
    ok: bool
    note: str = ""


@dataclass
class CrateVerdict:
    """Structural / environmental verdict for the crate at one site."""

    dynamic_pressure_Pa: float
    wind_force_N: float
    overturning_moment_Nm: float
    restoring_moment_Nm: float
    fs_overturn: float
    net_bearing_kPa: float
    fs_bearing: float
    wind_force_sliding_N: float
    fs_slide: float
    uplift_active: bool
    ingress_risk: str            # "low" | "medium" | "high"
    min_ballast_kg: float        # to reach targets given current mount
    stable: bool
    checks: Dict[str, CheckResult]
    mounting_spec: str
    notes: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "dynamic_pressure_Pa": round(self.dynamic_pressure_Pa, 1),
            "wind_force_N": round(self.wind_force_N, 1),
            "fs_overturn": round(self.fs_overturn, 2),
            "net_bearing_kPa": round(self.net_bearing_kPa, 2),
            "fs_bearing": round(self.fs_bearing, 2),
            "fs_slide": round(self.fs_slide, 2),
            "ingress_risk": self.ingress_risk,
            "min_ballast_kg": round(self.min_ballast_kg, 0),
            "stable": self.stable,
            "mounting_spec": self.mounting_spec,
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Crate:
    """Rigid-body structural + environmental check for the autonomous unit."""

    def evaluate(self, cfg: CrateConfig) -> CrateVerdict:
        c, w, g, e = cfg.crate, cfg.wind, cfg.ground, cfg.env
        total_mass = c.mass_kg + cfg.ballast_kg
        N_self = c.empty_weight_N + cfg.ballast_kg * G

        # Vertical accessories: snow adds weight (helps overturn, hurts bearing)
        snow_W = e.snow_load_kPa * 1000.0 * c.footprint_m2
        N = N_self + snow_W

        # ---- Wind force & overturning ------------------------------------
        q = w.dynamic_pressure_Pa()
        A_proj = c.projected_area_m2(w)
        F_wind = q * c.drag_coefficient * A_proj

        # lateral force for stability = max(wind, seismic)
        F_lat = F_wind
        if e.seismic_base_coefficient:
            F_seis = e.seismic_base_coefficient * N
            F_lat = max(F_wind, F_seis)
        self._F_lat = F_lat

        M_overturn = F_lat * (c.height_m / 2.0)   # force at mid-height
        M_restore = N * c.pivot_arm_m(w)
        fs_overturn = M_restore / M_overturn if M_overturn > 0 else math.inf

        # ---- Bearing pressure -------------------------------------------
        net_bearing_kPa = N / c.footprint_m2 / 1000.0
        fs_bearing = g.p_allow_kPa / net_bearing_kPa if net_bearing_kPa > 0 else math.inf

        # ---- Sliding -----------------------------------------------------
        resist_slide = g.friction_mu * N + (0.0 if not g.anchored else g.friction_mu * N)
        fs_slide = resist_slide / F_lat if F_lat > 0 else math.inf

        # ---- Uplift (net vertical > 0 stays planted; wind lift neglected) -
        uplift_active = fs_overturn < 1.0 or (F_wind > N * 0.2)

        # ---- Ingress / rain risk -----------------------------------------
        ingress_risk = self._ingress_risk(e, g)

        # ---- Minimum ballast to meet target FS ---------------------------
        min_ballast = self._required_ballast(cfg)

        checks = {
            "overturn": CheckResult(
                "overturn", fs_overturn, cfg.target_fs_overturn,
                fs_overturn >= cfg.target_fs_overturn,
                f"M_r={M_restore:.0f} / M_o={M_overturn:.0f} Nm"),
            "bearing": CheckResult(
                "bearing", fs_bearing, 1.0, fs_bearing >= 1.0,
                f"p_net={net_bearing_kPa:.1f} kPa vs {g.p_allow_kPa:.0f} kPa allow"),
            "slide": CheckResult(
                "slide", fs_slide, cfg.target_fs_slide, fs_slide >= cfg.target_fs_slide,
                f"resist={resist_slide:.0f} / F={F_lat:.0f} N"),
        }
        stable = all(cr.ok for cr in checks.values()) and not uplift_active

        mounting = self._mounting_spec(checks, min_ballast, ingress_risk, g)

        return CrateVerdict(
            dynamic_pressure_Pa=q,
            wind_force_N=F_wind,
            overturning_moment_Nm=M_overturn,
            restoring_moment_Nm=M_restore,
            fs_overturn=fs_overturn,
            net_bearing_kPa=net_bearing_kPa,
            fs_bearing=fs_bearing,
            wind_force_sliding_N=F_lat,
            fs_slide=fs_slide,
            uplift_active=uplift_active,
            ingress_risk=ingress_risk,
            min_ballast_kg=min_ballast,
            stable=stable,
            checks=checks,
            mounting_spec=mounting,
            notes={
                "wind": w.load_label(),
                "projected_area_m2": f"{A_proj:.1f}",
                "lateral_force_N": f"{F_lat:.0f}",
            },
        )

    def _ingress_risk(self, e: EnvironmentalLoads, g: GroundSpec) -> str:
        score = 0
        if e.rain_intensity_mm_hr >= 100:
            score += 2
        elif e.rain_intensity_mm_hr >= 50:
            score += 1
        if e.sealing_class != "sealed":
            score += 1
        if not g.drainable:
            score += 1
        if g.flood_depth_m > 0:
            score += 2
        if score >= 4:
            return "high"
        if score >= 2:
            return "medium"
        return "low"

    def _required_ballast(self, cfg: CrateConfig) -> float:
        """Ballast mass needed to hit the overturn target at this mount."""
        c, w, g, e = cfg.crate, cfg.wind, cfg.ground, cfg.env
        q = w.dynamic_pressure_Pa()
        A = c.projected_area_m2(w)
        F = q * c.drag_coefficient * A
        if e.seismic_base_coefficient:
            F = max(F, e.seismic_base_coefficient * c.empty_weight_N)
        M_o = F * (c.height_m / 2.0)
        pivot = c.pivot_arm_m(w)
        N_req = cfg.target_fs_overturn * (M_o / pivot) if pivot > 0 else 0.0
        ballast = max(0.0, (N_req / G) - c.mass_kg)
        return 0.0 if ballast < 0.0 else ballast

    def _mounting_spec(
        self, checks: Dict[str, CheckResult], ballast: float,
        ingress: str, g: GroundSpec,
    ) -> str:
        parts = []
        if ballast > 0:
            parts.append(f"{ballast:.0f} kg ballast")
        elif not checks["overturn"].ok:
            parts.append("ballast / lowered CG / fixed anchorage")
        else:
            parts.append("no ballast required")
        if not checks["slide"].ok:
            parts.append("tie-down anchors")
        if not checks["bearing"].ok:
            parts.append("load-spreading pad / cribbing")
        if not checks["overturn"].ok:
            parts.append("lowered CG or fixed anchorage")
        if ingress != "low":
            parts.append(f"{ingress}-grade sealing + drainage")
        if g.flood_depth_m > 0:
            parts.append("elevate above flood level")
        return ", ".join(parts) if parts else "concrete pad, no ballast required"


# Convenience ---------------------------------------------------------


def evaluate_crate(
    crate: CrateSpec = CrateSpec(),
    wind: WindLoad = WindLoad(),
    ground: GroundSpec = GroundSpec(),
    env: EnvironmentalLoads = EnvironmentalLoads(),
    ballast_kg: float = 0.0,
) -> CrateVerdict:
    """Evaluate crate stability at a site with defaults."""
    return Crate().evaluate(CrateConfig(crate, wind, ground, env, ballast_kg=ballast_kg))
