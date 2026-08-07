"""
Site layer (L3) — civil / ground / drainage / layout design for the crate.

The crate model (`models/crate.py`, D4) answers *is the unit stable*: it reports
overturning / bearing / sliding factors of safety, an ingress risk band, and a
required ballast mass.  This module turns those *risks* into a *design* for the
**site** (the ground the crate sits on, the water that must drain away from it,
the wind it is exposed to, and the trucks/power/water/product it must connect
to).  It is the L3 complement to the D4 crate layer, and the layer that answers
"will the container sink / flood / get wind-exposed" for a real deployment.

It is *purely additive* and deliberately screening-grade (L0 on the credibility
ladder): it reuses `dark_mill.SiteDefinition`/`ClimateSpec` and the crate verdict
and does **not** modify them.  Every number here is an unvalidated screening
estimate until a real soil test / hydraulic survey / wind campaign checks it at
the named-beachhead stage (docs/SYSTEM_TWIN.md § 4-7).

Layer data flow (acyclic):

    Site (ground + climate + feed/power/water/product)
        -> Crate verdict (D4): FS_overturn, p_net/p_allow, FS_slide, ingress
        -> SiteLayer design: foundation/ballast, drainage, wind exposure,
                             layout/access
        -> dark_mill go/no-go + operating-twin safe-state (SystemTwin report)

Four design blocks:

1. Foundation & ballast   — pad area/pressure from FS_overturn bearing needs,
   ballast mass to meet the overturn target, frost depth consideration, and
   the resulting net bearing check on the *designed* pad.
2. Flood & drainage       — freeboard above the plausible flood depth, minimum
   grade away from the unit, and a roof-rainfall runoff / soffit-vs-drain
   estimate that flags overflow; returns a drain spec and a clear FLOOD verdict.
3. Terrain wind exposure  — refines the crate's simplified terrain multiplier
   into an exposure category (roughness) that scales design gust pressure, so a
   high-drag / high-altitude site is not under-rated; returns the design gusts
   that the crate check should actually use.
4. Layout & access        — feedstock, power, water and product offtake each get
   an access check (distance, transport mode, required service capacity) against
   the site, producing a layout verdict and a recommended site footprint.

The class produces a single `SiteLayerVerdict` whose `.flat()` is JSON-serialisable
for the SystemTwin report.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, Optional

from .crate import Crate, CrateConfig, CrateSpec, GroundSpec, EnvironmentalLoads, WindLoad

G = 9.81

# Roughness/exposure classes (simplified ASCE 7-16-style, screening only).
# Each is a power-law exponent `alpha` such that gust pressure scales as
# (v_gust)^2 and the terrain factor Kz ~ (z/zg)^(2/alpha)·2.01-style correction
# is replaced here by a single relative exposure factor vs the crate's "open"
# baseline at the reference height.  Screening-grade; real code supersedes.
_EXPOSURE = {
    # (1/z0-ish label, roughness length z0 (m), gust-factor multiplier vs open)
    "open":     {"z0": 0.03, "mult": 1.00},   # ASCE C / flat, ocean
    "suburban": {"z0": 0.35, "mult": 0.88},   # ASCE B / scattered low-rise
    "urban":    {"z0": 1.00, "mult": 0.78},   # ASCE B-dense / city
}

# Transport mode -> recommended min service capacity (truck loads/day is handled
# separately; these are the "is the road/rail/pipeline physically there" gates).
_MODE_HAS_CAPACITY = {"onsite": True, "pipeline": True, "rail": True, "truck": True}

# Feedstock release-discharge coefficients (screening; rational-method C).
_RUNOFF_C = {"roof": 0.95, "pad": 0.85, "gravel": 0.70, "drained": 0.85}


# ---------------------------------------------------------------------------
# Inputs (all flat, defaults match a 40-ft crate on marginal ground)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SiteLayerSpec:
    """Design parameters for the L3 site layer (additive, screening)."""

    # Foundation / ballast
    allow_pad_bearing_kPa: float = 100.0   # allowable bearing on the *pad*
    frost_depth_m: float = 0.0             # frost line; >0 forces a deeper footer
    pad_margin_m: float = 0.60             # pad overhang beyond crate footprint (m)
    design_fs_overturn: float = 1.5        # overturn target the *design* must meet

    # Flood & drainage
    freeboard_m: float = 0.30              # min freeboard above flood depth (m)
    min_grade_pct: float = 2.0             # min fall away from the unit (%)
    design_storm_mm_hr: float = 50.0       # roof-rainfall design intensity (mm/hr)

    # Wind exposure
    reference_height_m: float = 10.0       # gust reported at this height
    exposure_height_m: float = 2.6         # crate mid-height (pressure reference)


@dataclass(frozen=True)
class SiteAccess:
    """What the site must provide; each gate is a yes/no with a driver."""

    feedstock_distance_km: float = 0.0
    feedstock_mode: str = "truck"          # "truck" | "rail" | "pipeline" | "onsite"
    power_available_MW: float = 2.0
    power_required_MW: float = 1.0
    water_available: bool = True
    water_required_m3_yr: float = 5000.0
    product_road_access: bool = True
    product_distance_km: float = 50.0


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


@dataclass
class FoundationDesign:
    """Designed pad + ballast to make the crate stable and bearing-safe."""

    pad_length_m: float
    pad_width_m: float
    pad_area_m2: float
    pad_bearing_kPa: float               # net bearing across the *designed* pad
    bearing_ok: bool
    ballast_kg: float
    ballast_ok: bool                     # meets FS_overturn target
    footer_depth_m: float                # incl. frost relief
    frost_ok: bool
    foundation_type: str                 # e.g. "compacted pad", "concrete slab"
    checks: Dict[str, bool]

    def flat(self) -> Dict[str, object]:
        return {
            "pad_length_m": round(self.pad_length_m, 2),
            "pad_width_m": round(self.pad_width_m, 2),
            "pad_area_m2": round(self.pad_area_m2, 2),
            "pad_bearing_kPa": round(self.pad_bearing_kPa, 2),
            "bearing_ok": self.bearing_ok,
            "ballast_kg": round(self.ballast_kg, 0),
            "ballast_ok": self.ballast_ok,
            "footer_depth_m": round(self.footer_depth_m, 2),
            "frost_ok": self.frost_ok,
            "foundation_type": self.foundation_type,
        }


@dataclass
class DrainageDesign:
    """Flood / drainage design and the "will it flood" verdict."""

    freeboard_m: float
    flood_depth_m: float
    flood_clearance_m: float             # = freeboard - flood_depth (negative => flooded)
    min_grade_pct: float
    roof_runoff_m3_hr: float             # design-storm roof rainfall off the crate
    drainage_verdict: str                # "low" | "medium" | "high"
    spec: str                            # recommended drain arrangement
    checks: Dict[str, bool]

    def flat(self) -> Dict[str, object]:
        return {
            "freeboard_m": round(self.freeboard_m, 2),
            "flood_depth_m": round(self.flood_depth_m, 2),
            "flood_clearance_m": round(self.flood_clearance_m, 2),
            "min_grade_pct": self.min_grade_pct,
            "roof_runoff_m3_hr": round(self.roof_runoff_m3_hr, 2),
            "drainage_verdict": self.drainage_verdict,
            "spec": self.spec,
        }


@dataclass
class WindExposureDesign:
    """Refined design gusts from terrain roughness + height, feeding the crate."""

    exposure_category: str               # "open" | "suburban" | "urban"
    terrain_multiplier: float            # vs crate "open" baseline
    design_gust_m_s: float               # gust at exposure height (roughness-adjusted)
    geo_gust_m_s: float                  # gust at reference height (as-reported)
    gust_pressure_Pa: float              # dynamic pressure at exposure height
    checks: Dict[str, bool]

    def flat(self) -> Dict[str, object]:
        return {
            "exposure_category": self.exposure_category,
            "terrain_multiplier": round(self.terrain_multiplier, 3),
            "design_gust_m_s": round(self.design_gust_m_s, 2),
            "geo_gust_m_s": round(self.geo_gust_m_s, 2),
            "gust_pressure_Pa": round(self.gust_pressure_Pa, 1),
        }


@dataclass
class LayoutDesign:
    """Site layout / offtake access verdict."""

    access: Dict[str, bool]              # feed / power / water / product
    feedstock_ok: bool
    power_ok: bool
    water_ok: bool
    product_ok: bool
    layout_fit: bool                     # spatial arrangement fits available_area
    footprint_total_m2: float
    available_area_m2: float
    spec: str
    checks: Dict[str, bool]

    def flat(self) -> Dict[str, object]:
        return {
            "feedstock_ok": self.feedstock_ok,
            "power_ok": self.power_ok,
            "water_ok": self.water_ok,
            "product_ok": self.product_ok,
            "layout_fit": self.layout_fit,
            "footprint_total_m2": round(self.footprint_total_m2, 0),
            "available_area_m2": round(self.available_area_m2, 0),
            "spec": self.spec,
        }


@dataclass
class SiteLayerVerdict:
    """Combined L3 site design answer."""

    foundation: FoundationDesign
    drainage: DrainageDesign
    wind: WindExposureDesign
    layout: LayoutDesign

    # The three questions the task asks
    def will_sink(self) -> bool:
        return not (self.foundation.bearing_ok and self.foundation.ballast_ok
                    and self.foundation.frost_ok)

    def will_flood(self) -> bool:
        return self.drainage.flood_clearance_m < 0

    def will_get_wind_exposed(self) -> bool:
        """Screening: the crate is 'exposed' if design gust drives FS_overturn
        below the target.  We surface the design gust + bearing/ballast outcome;
        the numeric FS is computed by the crate check (D4) at that gust."""
        return self.wind.design_gust_m_s >= 55.0

    def flat(self) -> Dict[str, object]:
        return {
            "will_sink": self.will_sink(),
            "will_flood": self.will_flood(),
            "will_get_wind_exposed": self.will_get_wind_exposed(),
            "foundation": self.foundation.flat(),
            "drainage": self.drainage.flat(),
            "wind": self.wind.flat(),
            "layout": self.layout.flat(),
        }

    def summary(self) -> str:
        fdn = self.foundation
        drn = self.drainage
        wnd = self.wind
        lay = self.layout
        return (
            f"SITE LAYER: sink={'YES' if self.will_sink() else 'no'}"
            f" flood={'YES' if self.will_flood() else 'no'}"
            f" wind-exposed={'YES' if self.will_get_wind_exposed() else 'no'}\n"
            f"  foundation: {fdn.foundation_type}, pad {fdn.pad_area_m2:.0f} m² @ "
            f"{fdn.pad_bearing_kPa:.0f} kPa (ok={fdn.bearing_ok}), ballast "
            f"{fdn.ballast_kg:.0f} kg (ok={fdn.ballast_ok})\n"
            f"  drainage: {drn.drainage_verdict} ({drn.spec}); flood clearance "
            f"{drn.flood_clearance_m:+.2f} m\n"
            f"  wind: {wnd.exposure_category}, design gust {wnd.design_gust_m_s:.0f} m/s "
            f"@ {wnd.gust_pressure_Pa:.0f} Pa\n"
            f"  layout: feed={lay.feedstock_ok} power={lay.power_ok} water={lay.water_ok} "
            f"product={lay.product_ok} fit={lay.layout_fit} ({lay.footprint_total_m2:.0f} of "
            f"{lay.available_area_m2:.0f} m²)\n"
            f"  => {lay.spec}"
        )


# ---------------------------------------------------------------------------
# SiteLayer — the L3 design engine
# ---------------------------------------------------------------------------


class SiteLayer:
    """Produces the L3 site-layer design from a `dark_mill.SiteDefinition`."""

    def evaluate(
        self,
        site,
        crate_verdict=None,
        spec: Optional[SiteLayerSpec] = None,
        access: Optional[SiteAccess] = None,
    ) -> SiteLayerVerdict:
        """Design the site for the crate sitting on it.

        Parameters
        ----------
        site : dark_mill.SiteDefinition (duck-typed)
            Ground + climate + feed/power/water/product fields.
        crate_verdict : CrateVerdict, optional
            Pre-computed D4 crate verdict.  If given, its designed ballast and
            bearing are carried through; otherwise a nominal 40-ft crate is
            assumed for foundation/ballast sizing.
        spec, access : SiteLayerSpec / SiteAccess
            Design assumptions.  Defaults screen a nominal autonomous unit.
        """
        spec = spec or SiteLayerSpec()
        access = access or self._default_access(site)

        foundation = self._design_foundation(site, crate_verdict, spec)
        drainage = self._design_drainage(site, spec)
        wind = self._design_wind(site, spec, crate_verdict)
        layout = self._design_layout(site, access)

        return SiteLayerVerdict(foundation, drainage, wind, layout)

    # ---- foundation / ballast ------------------------------------------

    def _design_foundation(
        self, site, crate_verdict, spec: SiteLayerSpec,
    ) -> FoundationDesign:
        # Crate footprint (40-ft default if we don't have an explicit one).
        crate = CrateSpec()
        length_m, width_m = crate.length_m, crate.width_m

        # Mass available for restoring moment: non-ballast crate + any ballast an
        # upstream check already required.
        crate_mass_kg = crate.mass_kg
        ballast_from_verdict = (
            crate_verdict.min_ballast_kg if crate_verdict is not None else 0.0
        )
        ballast_kg = max(ballast_from_verdict, 0.0)

        # Pad: extend beyond the crate by `pad_margin_m` on each side.
        pad_l = length_m + 2 * spec.pad_margin_m
        pad_w = width_m + 2 * spec.pad_margin_m
        pad_area = pad_l * pad_w

        # Net weight bearing on the pad: crate + ballast (+ snow if modeled).
        snow_W = 0.0
        if getattr(site.climate, "snow_load_kPa", 0.0):
            snow_W = site.climate.snow_load_kPa * 1000.0 * (length_m * width_m)
        total_kg = crate_mass_kg + ballast_kg + snow_W / G
        pad_bearing_kPa = total_kg * G / pad_area / 1000.0
        bearing_ok = pad_bearing_kPa <= spec.allow_pad_bearing_kPa

        # Overturn ballast need (recompute from a pure wind basis if no verdict).
        if crate_verdict is None:
            need = self._nominal_ballast(site, spec)
            ballast_kg = max(ballast_kg, need)
        ballast_ok = ballast_kg >= 0 and (
            crate_verdict is None or ballast_kg >= 0.999 * ballast_from_verdict
        )

        # Frost: footer below the frost line.
        footer_depth_m = 0.5 + max(0.0, getattr(site.climate, "freeze_depth_m", 0.0))
        frost_ok = footer_depth_m >= 0.5  # always placed below local frost line

        foundation_type = (
            "compacted gravel pad + concrete footing"
            if pad_bearing_kPa > 0.5 * spec.allow_pad_bearing_kPa
            else "concrete slab"
        )
        # If we're over allowable, the pad must spread more; flag it.
        checks = {
            "pad_bearing_ok": bearing_ok,
            "ballast_ok": ballast_ok,
            "frost_ok": frost_ok,
            "pad_area_m2": pad_area >= (length_m * width_m),
        }
        return FoundationDesign(
            pad_length_m=pad_l, pad_width_m=pad_w, pad_area_m2=pad_area,
            pad_bearing_kPa=pad_bearing_kPa, bearing_ok=bearing_ok,
            ballast_kg=ballast_kg, ballast_ok=ballast_ok,
            footer_depth_m=footer_depth_m, frost_ok=frost_ok,
            foundation_type=foundation_type, checks=checks,
        )

    def _nominal_ballast(self, site, spec: SiteLayerSpec) -> float:
        """Ballast to make a nominal 40-ft crate meet the overturn target, using
        the site's geotechnical wind exactly as the crate check would."""
        wind = WindLoad(
            gust_m_s=getattr(site.climate, "wind_gust_m_s", 40.0),
            terrain=getattr(site.climate, "wind_terrain", "open"),
            altitude_m=getattr(site.climate, "altitude_m", 0.0),
        )
        env = EnvironmentalLoads(
            rain_intensity_mm_hr=getattr(site.climate, "rainfall_intensity_mm_hr", 50.0),
            seismic_base_coefficient=getattr(site, "seismic_coefficient", None),
        )
        cfg = CrateConfig(CrateSpec(), wind, GroundSpec(), env)
        cfg = replace(cfg, target_fs_overturn=spec.design_fs_overturn)
        return Crate()._required_ballast(cfg)

    # ---- flood / drainage ----------------------------------------------

    def _design_drainage(self, site, spec: SiteLayerSpec) -> DrainageDesign:
        flood_depth_m = getattr(site, "flood_depth_m", 0.0)
        clearance = spec.freeboard_m - flood_depth_m

        # Roof runoff at the design storm off the crate roof (40-ft default <=> CrateSpec).
        crate = CrateSpec()
        rain_hr = spec.design_storm_mm_hr / 1000.0  # m/hr
        roof_area = crate.length_m * crate.width_m
        c = _RUNOFF_C.get("roof", 0.95)
        runoff_m3_hr = c * rain_hr * roof_area

        if clearance < 0:
            verdict, spec_txt = "high", f"flood depth {flood_depth_m:.2f} m exceeds {spec.freeboard_m:.2f} m freeboard — elevate crate on {abs(clearance)+0.15:.2f} m voussoirs/piers"
        elif clearance < 0.15:
            verdict, spec_txt = "medium", f"marginal freeboard ({clearance:.2f} m) — monitor + storm drain"
        elif not getattr(site, "ground_anchored", False) and getattr(site, "ground_friction_mu", 0.5) < 0.35:
            verdict, spec_txt = "medium", "slippery footprint — add grade + keyed anchor"
        else:
            verdict, spec_txt = "low", f"drain away at ≥{spec.min_grade_pct:.0f}% grade, perimeter swale"

        checks = {
            "freeboard_ok": clearance >= 0.15,
            "grade_ok": spec.min_grade_pct >= 1.0,
            "runoff_handled": runoff_m3_hr <= 15.0,  # screening roof capacity
        }
        return DrainageDesign(
            freeboard_m=spec.freeboard_m, flood_depth_m=flood_depth_m,
            flood_clearance_m=clearance, min_grade_pct=spec.min_grade_pct,
            roof_runoff_m3_hr=runoff_m3_hr, drainage_verdict=verdict,
            spec=spec_txt, checks=checks,
        )

    # ---- terrain wind exposure ------------------------------------------

    def _design_wind(self, site, spec: SiteLayerSpec, crate_verdict) -> WindExposureDesign:
        terrain = getattr(site.climate, "wind_terrain", "open")
        exp = _EXPOSURE.get(terrain, _EXPOSURE["open"])
        reported_gust = getattr(site.climate, "wind_gust_m_s", 40.0)

        # Height correction: gusts are stronger higher up; the crate mid-height
        # is lower than the reporting anemometer, so a crude planetary-boundary
        # -layer log profile reduces the effective gust toward the ground.
        # Screening only (z0 from the exposure table).
        z0 = exp["z0"]
        h = max(spec.exposure_height_m, 0.5)
        ref_h = max(spec.reference_height_m, h)
        try:
            height_factor = math.log(h / z0) / math.log(ref_h / z0)
        except (ValueError, ZeroDivisionError):
            height_factor = 1.0
        height_factor = max(0.5, min(1.2, height_factor))

        # Open terrain is the crate's baseline multiplier (1.0); other classes
        # derate because roughness slows the near-ground wind.
        terrain_mult = exp["mult"]
        design_gust = reported_gust * height_factor * terrain_mult
        design_gust = max(design_gust, reported_gust * 0.4)

        rho = 1.225 * math.exp(-getattr(site.climate, "altitude_m", 0.0) / 8500.0)
        gust_pressure = 0.5 * rho * design_gust ** 2

        checks = {
            "below_55": design_gust < 55.0,
            "reported_preserved_order": design_gust <= reported_gust * 1.15,
        }
        return WindExposureDesign(
            exposure_category=terrain, terrain_multiplier=terrain_mult,
            design_gust_m_s=design_gust, geo_gust_m_s=reported_gust,
            gust_pressure_Pa=gust_pressure, checks=checks,
        )

    # ---- layout / access ------------------------------------------------

    def _default_access(self, site) -> SiteAccess:
        return SiteAccess(
            feedstock_distance_km=getattr(site, "feedstock_distance_km", 0.0),
            feedstock_mode=getattr(site.feedstock, "transport_mode", "truck")
            if hasattr(site, "feedstock") else "truck",
            power_available_MW=getattr(site.grid, "max_power_MW", 2.0)
            if hasattr(site, "grid") else 2.0,
            power_required_MW=access_power(getattr(site, "target_capacity_t_Fe_yr", 1500.0)),
            water_available=(getattr(site.climate, "water_availability", "municipal")
                             != "scarce"),
            water_required_m3_yr=_water_need(getattr(site, "target_capacity_t_Fe_yr", 1500.0)),
            product_road_access=True,
            product_distance_km=getattr(site, "product_market_km", 150.0),
        )

    def _design_layout(self, site, access: SiteAccess) -> LayoutDesign:
        crate = CrateSpec()
        crate_fp = crate.length_m * crate.width_m
        # Function of scale: feedstock + ballast laydown + access + crane + offtake.
        turnover = access.feedstock_distance_km  # km -> bigger laydown demand
        footprint_total = crate_fp * (2.0 + 0.5 * max(0, 5.0 - turnover))
        # Available area.
        available = getattr(site, "available_area_m2", 1000.0)

        # Feedstock access
        feed_ok = (access.feedstock_distance_km <= 250.0
                   and access.feedstock_mode in _MODE_HAS_CAPACITY)
        # Power: required fits in available grid capacity (with margin).
        power_ok = access.power_required_MW <= access.power_available_MW
        # Water
        water_ok = access.water_available
        # Product offtake: reasonable market distance or road access.
        product_ok = access.product_road_access and access.product_distance_km <= 1000.0

        layout_fit = footprint_total <= available
        parts = []
        if not feed_ok:
            parts.append("reposition near feedstock / secure rail-or-pipeline")
        if not power_ok:
            parts.append(f"grid upsize to {access.power_required_MW:.1f} MW")
        if not water_ok:
            parts.append("secure water supply / truck-in make-up")
        if not product_ok:
            parts.append("open product offtake / nearer market")
        if not layout_fit:
            parts.append(f"site too small ({footprint_total:.0f} > {available:.0f} m²)")
        spec = ", ".join(parts) if parts else "layout fits; direct feed/power/water/product access"

        checks = {
            "feedstock_ok": feed_ok, "power_ok": power_ok,
            "water_ok": water_ok, "product_ok": product_ok, "layout_fit": layout_fit,
        }
        return LayoutDesign(
            access=checks, feedstock_ok=feed_ok, power_ok=power_ok,
            water_ok=water_ok, product_ok=product_ok, layout_fit=layout_fit,
            footprint_total_m2=footprint_total, available_area_m2=available,
            spec=spec, checks=checks,
        )


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def access_power(capacity_t_Fe_yr: float) -> float:
    """Screening power requirement (MW) for a target Fe/yr at ~2.6 MWh/t Fe
    (mid-case from the economics) with ~85% load factor."""
    kwh_yr = capacity_t_Fe_yr * 1e3 * 2.6
    return kwh_yr / 8760.0 / 0.85 / 1000.0


def _water_need(capacity_t_Fe_yr: float) -> float:
    """Screening make-up water (m³/yr): ~6 m³/t Fe lost to H₂ + dragout."""
    return capacity_t_Fe_yr * 6.0


def evaluate_site_layer(site, crate_verdict=None, **kw) -> SiteLayerVerdict:
    """One-call convenience: design the site for a `SiteDefinition`."""
    return SiteLayer().evaluate(site, crate_verdict=crate_verdict, **kw)
