"""
Whole-system twin — compositional driver wiring process, crate/structural,
and site layers into one end-to-end site assessment with an explicit
per-layer credibility vector.

The whole-system twin answers "does this site make sense?" as:
  site (climate + ground + flood + seismic + feedstock + grid) →
  crate (wind, rain, bearing, sliding, ingress, ballast/mounting) →
  cell process (thermo, kinetics, transport, FE/V) →
  combined GO/NO-GO + required ballast/mounting + environmental safe-state action.

All numbers are screening-grade (per-layer L0) until real data / load tests
validate them. The twin is therefore a vector of layer-credibilities,
not a single label: process L0/L1, crate L0, site L0.

Architecture
------------
- L1 Process: cell_physics → FE/V → stack sizing (dark_mill)
- L2 Crate: Crate structural/environmental model (crate.py)
- L3 Site: SiteDefinition + ClimateSpec extended with wind gust/terrain,
  rainfall, snow, soil bearing, seismic, flood, freeze.

Data flow is acyclic: site → crate → cell, with the site assessment
aggregating all layers into the final verdict. The crate verdict feeds the
dark_mill go/no-go, and the environmental limits feed the operating twin's
storm-mode safe-state.

References
----------
- docs/SYSTEM_TWIN.md — vision / three-layer stack / credibility ladder
- docs/NEXT_STEPS.md — L0→L5 ladder
- dark_mill.py — site sizing + crate coupling
- crate.py — overturning / bearing / sliding / ingress + min_ballast
- operating_twin.py — environmental safe-state limits
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import math

from .dark_mill import (
    SiteDefinition,
    SiteReport,
    ClimateSpec,
    GridSpec,
    EXAMPLE_SITES,
    size_dark_mill,
    site_to_crate_config,
    evaluate_crate_for_site,
)
from .crate import Crate, CrateVerdict, CrateSpec, WindLoad
from .operating_twin import TwinConfig, SensorSnapshot, OperatingTwin, TwinMode


# ---------------------------------------------------------------------------
# Credibility vector
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CredibilityVector:
    """Per-layer credibility on the L0→L5 ladder.

    All screening estimates are L0 until validated by real data:
    - process: calibrated vs measured FE/V/j, transport
    - crate: wind/load test, site survey
    - site: soil test, flood survey, wind measurement campaign

    The vector is explicit on every report so we never say "system twin is L5"
    — we say "process L0 / crate L0 / site L0" and drive each independently.
    """

    process_level: int = 0  # L0 screening
    crate_level: int = 0
    site_level: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "process": self.process_level,
            "crate": self.crate_level,
            "site": self.site_level,
        }

    def label(self) -> str:
        return f"process L{self.process_level} / crate L{self.crate_level} / site L{self.site_level}"

    @classmethod
    def screening(cls) -> "CredibilityVector":
        """All L0 screening — the default until validation data arrives."""
        return cls(process_level=0, crate_level=0, site_level=0)


# ---------------------------------------------------------------------------
# System twin report
# ---------------------------------------------------------------------------

@dataclass
class SystemTwinReport:
    """End-to-end site assessment: process + crate + site."""

    site_key: str
    site: SiteDefinition
    site_report: SiteReport
    crate_verdict: CrateVerdict
    crate_verdict_end_on: Optional[CrateVerdict]  # worst-case wind direction
    credibility: CredibilityVector
    combined_stable: bool
    overall_go: bool
    required_ballast_kg: float
    mounting_spec: str
    environmental_safe_state: str
    environmental_snapshot: SensorSnapshot
    operating_twin_modes: Dict[str, str]  # reasons → mode
    go_no_go: Dict[str, Dict[str, Any]]
    notes: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable dict for report persistence."""
        sd = self.site_report.stack_design
        mb = self.site_report.mass_balance

        def _py_bool(v):
            # Convert numpy bool_ to Python bool recursively
            if isinstance(v, dict):
                return {kk: _py_bool(vv) for kk, vv in v.items()}
            if isinstance(v, (list, tuple)):
                return [_py_bool(x) for x in v]
            try:
                import numpy as np
                if isinstance(v, (np.bool_, np.integer, np.floating)):
                    return bool(v) if isinstance(v, np.bool_) else float(v) if isinstance(v, np.floating) else int(v)
            except Exception:
                pass
            return v

        return {
            "site_key": self.site_key,
            "site_name": self.site.name,
            "credibility_vector": self.credibility.to_dict(),
            "credibility_label": self.credibility.label(),
            "combined_stable": bool(self.combined_stable),
            "overall_go": bool(self.overall_go),
            "required_ballast_kg": round(float(self.required_ballast_kg), 0),
            "mounting_spec": self.mounting_spec,
            "environmental_safe_state": self.environmental_safe_state,
            "environmental_snapshot": {
                "wind_gust_m_s": self.environmental_snapshot.wind_gust_m_s,
                "flood_depth_m": self.environmental_snapshot.flood_depth_m,
                "rain_mm_hr": self.environmental_snapshot.rain_intensity_mm_hr,
                "snow_kPa": self.environmental_snapshot.snow_load_kPa,
                "ingress_detected": bool(self.environmental_snapshot.ingress_detected),
            },
            "crate_verdict": _py_bool(self.crate_verdict.to_dict()),
            "crate_verdict_end_on": _py_bool(self.crate_verdict_end_on.to_dict()) if self.crate_verdict_end_on else None,
            "stack": {
                "n_stacks": sd.n_stacks,
                "cells_per_stack": sd.cells_per_stack,
                "power_kW": round(float(sd.total_power_kW), 1),
                "production_t_yr": round(float(sd.annual_production_t()), 1),
                "FE": round(float(sd.current_efficiency), 3),
                "V_cell": round(float(sd.cell_voltage_V), 3),
                "j_mA_cm2": round(float(sd.current_density_mA_cm2), 1),
            },
            "LCOFe": self.site_report.lcofe.get("LCOFe ($/t Fe)"),
            "LCOFe_adjusted": self.site_report.lcofe.get("LCOFe ($/t Fe) adjusted"),
            "go_no_go": _py_bool(self.go_no_go),
            "notes": self.notes,
        }

    def summary(self) -> str:
        sd = self.site_report.stack_design
        gng = self.go_no_go
        lines = [
            f"{'='*72}",
            f"SYSTEM TWIN — End-to-End Site Assessment: {self.site.name}",
            f"Site key: {self.site_key}",
            f"{'='*72}",
            "",
            f"Credibility vector: {self.credibility.label()} — all screening-grade L0 until validated",
            f"  process: cell physics (FE/V) from Nernst-Planck + speciation — L{self.credibility.process_level}",
            f"  crate: structural (overturning/bearing/sliding/ingress) screening — L{self.credibility.crate_level}",
            f"  site: climatology + soil bearing + flood + seismic screening — L{self.credibility.site_level}",
            "",
            f"FEEDSTOCK: {self.site.feedstock.name} ({self.site.feedstock_key})",
            f"GRID: {self.site.grid.electricity_price_kWh*1000:.0f} $/MWh, {self.site.grid.renewable_fraction*100:.0f}% RE, {self.site.grid.max_power_MW:.1f} MW max",
            f"CLIMATE: {self.site.climate.ambient_temp_C:.0f} C av, wind {self.site.climate.wind_gust_m_s:.0f} m/s gust {self.site.climate.wind_terrain}, "
            f"rain {self.site.climate.rainfall_intensity_mm_hr:.0f} mm/hr, snow {self.site.climate.snow_load_kPa:.1f} kPa",
            f"GROUND: {self.site.soil_bearing_kPa:.0f} kPa allowable, μ={self.site.ground_friction_mu:.2f}, "
            f"flood {self.site.flood_depth_m:.2f} m, seismic {self.site.seismic_coefficient}, freeze {self.site.climate.freeze_depth_m:.1f} m",
            "",
            f"PROCESS (L{self.credibility.process_level}):",
            f"  FE {sd.current_efficiency*100:.1f}%, V {sd.cell_voltage_V:.2f} V, j {sd.current_density_mA_cm2:.0f} mA/cm²",
            f"  Power {sd.total_power_kW:.0f} kW, production {sd.annual_production_t():.0f} t/yr, LCOFe ${self.site_report.lcofe.get('LCOFe ($/t Fe)',0):.0f}/t",
            "",
            f"CRATE (L{self.credibility.crate_level}) — broadside:",
            f"  q={self.crate_verdict.dynamic_pressure_Pa:.0f} Pa, F_wind={self.crate_verdict.wind_force_N:.0f} N, FS_over={self.crate_verdict.fs_overturn:.2f}, "
            f"FS_bear={self.crate_verdict.fs_bearing:.2f}, FS_slide={self.crate_verdict.fs_slide:.2f}",
            f"  Ingress {self.crate_verdict.ingress_risk}, min ballast {self.crate_verdict.min_ballast_kg:.0f} kg, stable={self.crate_verdict.stable}",
        ]
        if self.crate_verdict_end_on:
            lines.append(
                f"  End-on worst: FS_over={self.crate_verdict_end_on.fs_overturn:.2f}, "
                f"min ballast {self.crate_verdict_end_on.min_ballast_kg:.0f} kg, stable={self.crate_verdict_end_on.stable}"
            )
        lines.extend([
            f"  Mounting spec: {self.mounting_spec}",
            "",
            f"SITE (L{self.credibility.site_level}) + ENVIRONMENTAL SAFE-STATE:",
            f"  Env snapshot: wind {self.environmental_snapshot.wind_gust_m_s} m/s, "
            f"flood {self.environmental_snapshot.flood_depth_m} m, rain {self.environmental_snapshot.rain_intensity_mm_hr} mm/hr",
            f"  Ingress detected: {self.environmental_snapshot.ingress_detected}, snow {self.environmental_snapshot.snow_load_kPa} kPa",
            f"  Safe-state action: {self.environmental_safe_state}",
            f"  Operating twin mode under env: {self.operating_twin_modes}",
            "",
            f"{'='*72}",
            f"COMBINED STABILITY VERDICT & GO/NO-GO",
            f"{'='*72}",
        ])
        for k, v in gng.items():
            status = "PASS" if v["pass"] else "FAIL"
            lines.append(f"  [{status}] {k}: {v['detail']}")
        lines.extend([
            "",
            f"  Required ballast: {self.required_ballast_kg:.0f} kg",
            f"  Mounting: {self.mounting_spec}",
            f"  Credibility: {self.credibility.label()}",
            f"  Combined stable (crate): {self.combined_stable}",
            f"  Overall GO: {self.overall_go} — {'GO — site viable' if self.overall_go else 'NO-GO — see failures'}",
            f"{'='*72}",
            "",
            "All numbers screening-grade (per-layer L0) until real data/load tests validate them.",
        ])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Environmental mapping — site → operating twin
# ---------------------------------------------------------------------------

def _site_to_env_snapshot(site: SiteDefinition, ingress_detected: bool = False) -> SensorSnapshot:
    """Create an environmental SensorSnapshot from a SiteDefinition.

    Uses plausible process sensor values for the cell part (V=2.8, etc.)
    because the focus here is the environmental coupling. The process part
    is valid but not the point — the wind/flood/ingress are.
    """
    # Freeze is a winterization concern, not a storm-mode trip unless ambient is below freezing.
    freeze_now = site.climate.freeze_risk and site.climate.ambient_temp_C < 0.0 and site.climate.freeze_depth_m > 0.3
    return SensorSnapshot(
        timestamp_s=0.0,
        current_A=1.0,
        voltage_V=2.8,
        temperature_C=site.climate.ambient_temp_C,
        pH=2.0,
        fe2_M=1.0,
        cathode_area_cm2=100.0,
        sensor_quality={},
        source_run_id=f"system-twin-{site.name}",
        wind_gust_m_s=site.climate.wind_gust_m_s,
        flood_depth_m=site.flood_depth_m,
        rain_intensity_mm_hr=site.climate.rainfall_intensity_mm_hr,
        snow_load_kPa=site.climate.snow_load_kPa,
        ingress_detected=ingress_detected,
        freeze_detected=freeze_now,
    )


def _site_to_twin_config(site: SiteDefinition) -> TwinConfig:
    """Map site structural/environmental fields to an OperatingTwin config
    with storm-mode limits.

    Thresholds are screening-grade: modest flood (0.1 m) and wind (40 m/s)
    trip to storm mode, reflecting that the autonomous unit should hold
    production in a storm rather than risk damage.
    """
    # Screening thresholds — site-specific limits could come from hardware qual
    return TwinConfig(
        cell_id=f"system-twin-{site.name[:20]}",
        max_current_A=100.0,
        max_current_density_mA_cm2=500.0,
        max_voltage_V=5.0,
        min_temperature_C=-10.0,
        max_temperature_C=90.0,
        min_fe2_M=0.1,
        max_fe2_M=3.0,
        min_pH=0.5,
        max_pH=6.0,
        target_current_A=10.0,
        target_temperature_C=50.0,
        current_ramp_A_per_s=1.0,
        max_wind_gust_m_s=40.0,          # storm mode above 40 m/s gust
        max_flood_depth_m=0.10,          # 10 cm flood → hold
        max_rain_intensity_mm_hr=100.0,  # 100 mm/hr → high rain
        max_snow_load_kPa=1.0,
        freeze_protection_required=site.climate.freeze_risk,
    )


def _evaluate_environmental_safe_state(site: SiteDefinition, crate_verdict) -> Tuple[str, SensorSnapshot, Dict[str, str]]:
    """Evaluate environmental safe-state action using the operating twin.

    Returns (action, snapshot, modes dict).
    """
    # Ingress detected if crate says high risk or flood > 0
    ingress = crate_verdict.ingress_risk == "high" or site.flood_depth_m > 0.2
    snapshot = _site_to_env_snapshot(site, ingress_detected=ingress)
    cfg = _site_to_twin_config(site)
    twin = OperatingTwin(cfg)
    # twin.environmental_safe_state uses _safety_reasons internally
    action = twin.environmental_safe_state(snapshot)
    # Also evaluate full trip evaluation via update+command for demonstration
    state = twin.update(snapshot, now_s=snapshot.timestamp_s)
    cmd = twin.command(now_s=snapshot.timestamp_s)
    modes = {
        "twin_mode": state.mode.value,
        "command_mode": cmd.mode.value,
        "trip_reasons": ",".join(state.trip_reasons) if state.trip_reasons else "none",
        "action": action,
    }
    return action, snapshot, modes


# ---------------------------------------------------------------------------
# Main entry point: evaluate a named site end-to-end
# ---------------------------------------------------------------------------

def evaluate_system_twin(
    site_key: str,
    ballast_kg: float = 0.0,
    credibility: Optional[CredibilityVector] = None,
) -> SystemTwinReport:
    """Evaluate a named site end-to-end from climatology+soil to cell FE/V.

    Reuses the 3 existing dark-mill scenarios (pickle_liquor_us_midwest,
    red_mud_alumina_refinery, wind_farm_ore) plus the full EXAMPLE_SITES library.
    Emits per-layer credibility vector, combined stability verdict,
    required ballast/mounting, and environmental safe-state action.

    Parameters
    ----------
    site_key : str
        Key into EXAMPLE_SITES, e.g. "pickle_liquor_us_midwest"
    ballast_kg : float
        Optional pre-applied ballast (0 = assess bare unit, report required)
    credibility : CredibilityVector | None
        Per-layer credibility; defaults to all L0 screening.

    Returns
    -------
    SystemTwinReport
        Decision-grade verdict with explicit per-layer L#.
    """
    if site_key not in EXAMPLE_SITES:
        raise KeyError(f"Unknown site_key {site_key!r}. Available: {list(EXAMPLE_SITES.keys())}")
    site = EXAMPLE_SITES[site_key]
    if credibility is None:
        credibility = CredibilityVector.screening()

    # ── L1 + L3: process + site economics via dark_mill (includes L2 screening crate) ──
    site_report = size_dark_mill(site)

    # ── L2: crate — explicit broadside + worst-case end-on ──
    # Broadside is the default in site_to_crate_config (long face screening)
    # For the system twin we also evaluate end-on (more limiting for overturning)
    crate_broadside = evaluate_crate_for_site(site, ballast_kg=ballast_kg)
    # End-on worst case
    from .crate import Crate, CrateSpec, WindLoad, GroundSpec, EnvironmentalLoads, CrateConfig
    cfg_broad = site_to_crate_config(site, ballast_kg=ballast_kg)
    cfg_end = CrateConfig(
        crate=cfg_broad.crate,
        wind=WindLoad(
            gust_m_s=site.climate.wind_gust_m_s,
            direction="end",
            terrain=site.climate.wind_terrain,
            altitude_m=site.climate.altitude_m,
            temperature_C=site.climate.ambient_temp_C,
        ),
        ground=cfg_broad.ground,
        env=cfg_broad.env,
        ballast_kg=ballast_kg,
        target_fs_overturn=cfg_broad.target_fs_overturn,
        target_fs_slide=cfg_broad.target_fs_slide,
    )
    crate_end_on = Crate().evaluate(cfg_end)

    # Choose the limiting verdict for combined stability (end-on typically worse)
    limiting = crate_end_on if crate_end_on.fs_overturn < crate_broadside.fs_overturn else crate_broadside

    # Required ballast is the max needed across directions to meet FS targets
    required_ballast = max(crate_broadside.min_ballast_kg, crate_end_on.min_ballast_kg)

    # Mounting spec: combine both, but limiting wins for ballast amount
    # If end-on needs more, its spec dominates; prepend ballast amount
    if crate_end_on.min_ballast_kg > crate_broadside.min_ballast_kg:
        mounting = crate_end_on.mounting_spec
    else:
        mounting = crate_broadside.mounting_spec
    # Ensure mounting mentions worst-case direction
    if "end" in cfg_end.wind.direction and crate_end_on.fs_overturn < 1.5:
        if f"{required_ballast:.0f} kg ballast" not in mounting:
            mounting = f"{required_ballast:.0f} kg ballast (end-on worst) + {mounting}"

    # ── Environmental safe-state via operating_twin ──
    safe_state_action, env_snapshot, op_modes = _evaluate_environmental_safe_state(site, limiting)

    # ── Combined verdict ──
    # Combined stability: can the crate physically sit through a storm with
    # practical ballast / tie-down? This is distinct from the bare-crate stable flag
    # which includes a conservative uplift trip.
    go_criteria = site_report.go_no_go
    crate_keys = ["Crate overturning", "Crate bearing", "Crate sliding", "Crate ingress"]
    crate_go = all(go_criteria.get(k, {"pass": True})["pass"] for k in crate_keys)
    # Practical ballast limit
    stabilisable = required_ballast < 20000
    # Bearing/sliding/ingress must pass independent of overturn (overturn fixed by ballast)
    bearing_pass = go_criteria.get("Crate bearing", {"pass": True})["pass"]
    sliding_pass = go_criteria.get("Crate sliding", {"pass": True})["pass"]
    ingress_pass = go_criteria.get("Crate ingress", {"pass": True})["pass"]
    # Combined stable if bearing/sliding/ingress pass and (crate overturn passes or ballast makes it pass)
    # We treat required_ballast as proof that overturn can be fixed.
    combined_stable = bearing_pass and sliding_pass and ingress_pass and (crate_go or stabilisable)

    # Overall GO = all dark_mill go/no-go PASS (including crate) AND combined stable AND not flood emergency
    dark_mill_go = all(v["pass"] for v in go_criteria.values())
    flood_no_go = site.flood_depth_m > 0.5
    overall_go = dark_mill_go and combined_stable and not flood_no_go

    # Enrich notes
    notes = {
        "process_layer": f"L{credibility.process_level} screening — FE/V from cell_physics, not measured",
        "crate_layer": f"L{credibility.crate_level} screening — overturn/bearing/sliding/ingress screening, no load test",
        "site_layer": f"L{credibility.site_level} screening — soil/wind/flood from ClimateSpec/SiteDefinition, no survey",
        "wind_broadside": crate_broadside.notes.get("wind", ""),
        "wind_end_on": crate_end_on.notes.get("wind", ""),
    }

    report = SystemTwinReport(
        site_key=site_key,
        site=site,
        site_report=site_report,
        crate_verdict=crate_broadside,
        crate_verdict_end_on=crate_end_on,
        credibility=credibility,
        combined_stable=combined_stable,
        overall_go=overall_go,
        required_ballast_kg=required_ballast,
        mounting_spec=mounting,
        environmental_safe_state=safe_state_action,
        environmental_snapshot=env_snapshot,
        operating_twin_modes=op_modes,
        go_no_go=go_criteria,
        notes=notes,
    )
    # Attach credibility to site_report for summary printing
    try:
        site_report.credibility = credibility.to_dict()
    except Exception:
        pass

    return report


def evaluate_all_sites() -> Dict[str, SystemTwinReport]:
    """Evaluate all example sites (including the 3 legacy scenario sites)."""
    return {key: evaluate_system_twin(key) for key in EXAMPLE_SITES.keys()}


# ---------------------------------------------------------------------------
# Backwards compatibility alias for CLI discovery
# ---------------------------------------------------------------------------

# 3 legacy sites referenced in experiments/data (the ones the task mentions)
LEGACY_THREE = ["pickle_liquor_us_midwest", "red_mud_alumina_refinery", "wind_farm_ore"]
