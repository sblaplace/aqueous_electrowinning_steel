"""Position-resolved FE prediction on a Hull cell panel.

Chains the Hull-cell primary-current model (j vs. position) with the 1D
diffusion-layer model (FE vs. j, T, C, δ, pH) to predict Faradaic
efficiency, deposit appearance, and mass gain across the cathode panel.

The key hypothesis this model tests::

    "Position 3-5 cm should show bright deposit at ~85% FE;
     position 7+ cm should show burnt/powdery deposit at ~40% FE."

Usage
-----
>>> from models.hull_cell_fe import hull_cell_fe_prediction, HullCellFEConfig
>>> cfg = HullCellFEConfig(total_current_A=2.0)
>>> result = hull_cell_fe_prediction(cfg)
>>> print(result.summary_df().head())

The diffusion-layer model is the expensive part (~0.5 s per zone), so the
sweep deliberately uses a small number of zones (default 10) to keep the
full prediction under 10 seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .diffusion_layer_1d import DiffusionLayer1D, DiffusionLayerResult
from .hull_cell import HullCellGeometry, hull_current_distribution


# ─── Appearance classification ────────────────────────────────────────

Appearance = Literal["bright", "dull", "burnt", "powdery"]


def appearance_from_fe(fe_fraction: float) -> Appearance:
    """Map Faradaic efficiency (0–1) to expected deposit appearance.

    Thresholds are approximate — real boundaries depend on current density,
    additives, and substrate.  They are calibrated to the Fe-sulphate/borate
    system at 60 °C with no organic brighteners.

    * **bright**   : FE ≥ 0.80  — compact, specular, fine-grained
    * **dull**     : FE 0.60–0.80 — matte, slightly rough
    * **burnt**    : FE 0.40–0.60 — dark, stressed, dendritic onset
    * **powdery**  : FE < 0.40 — loose, non-adherent, black powder
    """
    if not np.isfinite(fe_fraction):
        return "powdery"
    if fe_fraction >= 0.80:
        return "bright"
    if fe_fraction >= 0.60:
        return "dull"
    if fe_fraction >= 0.40:
        return "burnt"
    return "powdery"


# ─── Boundary-layer thickness model ──────────────────────────────────

@dataclass(frozen=True)
class DeltaProfile:
    """Position-dependent diffusion-layer thickness along the Hull panel.

    Stirring is not uniform: the near edge (high-j end, small gap) has
    better convective access than the far edge (low-j end, large gap).
    The default model is a linear interpolation.

    Parameters
    ----------
    delta_near_m : float
        Film thickness at the near (high-j) panel edge (m).
    delta_far_m : float
        Film thickness at the far (low-j) panel edge (m).
    """

    delta_near_m: float = 50.0e-6
    delta_far_m: float = 150.0e-6

    def __post_init__(self) -> None:
        if self.delta_near_m <= 0 or self.delta_far_m <= 0:
            raise ValueError("Delta values must be positive")

    def delta_at_position(
        self, position_cm: float | np.ndarray, panel_length_cm: float = 10.0,
    ) -> np.ndarray:
        """Linear interpolation of δ from near to far edge.

        Parameters
        ----------
        position_cm : float or array
            Distance from the near (high-j) edge in cm.
        panel_length_cm : float
            Total panel length (cm).
        """
        pos = np.asarray(position_cm, dtype=float)
        frac = np.clip(pos / panel_length_cm, 0.0, 1.0)
        return self.delta_near_m + frac * (self.delta_far_m - self.delta_near_m)


# ─── Configuration ───────────────────────────────────────────────────

@dataclass
class HullCellFEConfig:
    """All parameters for a Hull-cell FE prediction sweep.

    Hull cell geometry maps to the primary-current model; diffusion-layer
    parameters pass through to the 1D film solver.
    """

    # ── Hull cell ────────────────────────────────────────────────────
    geometry: HullCellGeometry = field(default_factory=HullCellGeometry)
    total_current_A: float = 2.0
    n_hull_segments: int = 100  # fine grid for current distribution

    # ── Zones (output resolution) ────────────────────────────────────
    n_zones: int = 10

    # ── Diffusion-layer model ────────────────────────────────────────
    fe_conc_M: float = 1.0
    pH_bulk: float = 2.0
    temperature_C: float = 60.0
    buffer_conc_M: float = 0.40
    support_conc_M: float = 0.0

    # ── Boundary-layer thickness ─────────────────────────────────────
    delta: DeltaProfile = field(default_factory=DeltaProfile)

    # ── Diffusion-layer solver kwargs ────────────────────────────────
    fe_i0: float = 10.0
    her_i0: float = 0.010
    fe_tafel_V: float = 0.120
    her_tafel_V: float = 0.140
    grid_points: int = 81  # film solver grid

    def __post_init__(self) -> None:
        if self.total_current_A <= 0:
            raise ValueError("total_current_A must be positive")
        if self.n_zones < 1:
            raise ValueError("n_zones must be at least 1")


# ─── Per-zone result ─────────────────────────────────────────────────

@dataclass(frozen=True)
class ZoneResult:
    """FE prediction for one panel zone."""

    zone_index: int
    position_cm_from_near_edge: float
    zone_width_cm: float
    local_j_mA_cm2: float
    local_delta_m: float
    faradaic_efficiency: float
    fe_percent: float
    appearance: Appearance
    surface_pH: float
    surface_fe_M: float
    precipitation_active: bool
    feoh2_supersaturation: float
    mass_gain_mg_per_min: float
    zone_area_cm2: float
    detail: DiffusionLayerResult = field(repr=False)


# ─── Full result ─────────────────────────────────────────────────────

@dataclass
class HullCellFEResult:
    """Complete Hull-cell FE prediction across all zones."""

    config: HullCellFEConfig
    zones: list[ZoneResult]
    j_distribution: pd.DataFrame

    # ── Derived summaries ────────────────────────────────────────────

    @property
    def fe_curve(self) -> pd.DataFrame:
        """FE(position) as a tidy DataFrame."""
        return pd.DataFrame({
            "position_cm": [z.position_cm_from_near_edge for z in self.zones],
            "j_mA_cm2": [z.local_j_mA_cm2 for z in self.zones],
            "delta_um": [z.local_delta_m * 1e6 for z in self.zones],
            "FE_fraction": [z.faradaic_efficiency for z in self.zones],
            "FE_percent": [z.fe_percent for z in self.zones],
            "appearance": [z.appearance for z in self.zones],
            "surface_pH": [z.surface_pH for z in self.zones],
            "surface_fe_M": [z.surface_fe_M for z in self.zones],
            "precipitation": [z.precipitation_active for z in self.zones],
        })

    @property
    def appearance_map(self) -> dict[float, Appearance]:
        """Position → appearance mapping."""
        return {
            z.position_cm_from_near_edge: z.appearance for z in self.zones
        }

    @property
    def bright_dull_boundary_cm(self) -> float | None:
        """Position where appearance transitions bright → dull."""
        return self._find_boundary("bright", "dull")

    @property
    def dull_burnt_boundary_cm(self) -> float | None:
        """Position where appearance transitions dull → burnt."""
        return self._find_boundary("dull", "burnt")

    @property
    def burnt_powdery_boundary_cm(self) -> float | None:
        """Position where appearance transitions burnt → powdery."""
        return self._find_boundary("burnt", "powdery")

    def _find_boundary(self, left: Appearance, right: Appearance) -> float | None:
        """Find the position of a left→right transition by linear interpolation."""
        for i in range(len(self.zones) - 1):
            if self.zones[i].appearance == left and self.zones[i + 1].appearance == right:
                # Linear interpolation on FE
                z0, z1 = self.zones[i], self.zones[i + 1]
                if z0.fe_percent == z1.fe_percent:
                    return z1.position_cm_from_near_edge
                # Threshold for the right appearance
                thresholds = {"bright": 80.0, "dull": 60.0, "burnt": 40.0, "powdery": 0.0}
                target = thresholds[right]
                frac = (z0.fe_percent - target) / (z0.fe_percent - z1.fe_percent)
                frac = np.clip(frac, 0.0, 1.0)
                return float(
                    z0.position_cm_from_near_edge
                    + frac * (z1.position_cm_from_near_edge - z0.position_cm_from_near_edge)
                )
        return None

    @property
    def best_fe_zone(self) -> ZoneResult:
        """Zone with the highest FE."""
        return max(self.zones, key=lambda z: z.faradaic_efficiency)

    @property
    def worst_fe_zone(self) -> ZoneResult:
        """Zone with the lowest FE."""
        return min(self.zones, key=lambda z: z.faradaic_efficiency)

    @property
    def total_mass_gain_mg_per_min(self) -> float:
        """Sum of mass gain across all zones (mg/min)."""
        return sum(z.mass_gain_mg_per_min for z in self.zones)

    @property
    def area_weighted_fe(self) -> float:
        """Area-weighted average FE across the panel."""
        total_area = sum(z.zone_area_cm2 for z in self.zones)
        if total_area <= 0:
            return 0.0
        return sum(
            z.faradaic_efficiency * z.zone_area_cm2 for z in self.zones
        ) / total_area

    def summary_df(self) -> pd.DataFrame:
        """All zone data in a single DataFrame."""
        return pd.DataFrame({
            "zone": [z.zone_index for z in self.zones],
            "position_cm": [z.position_cm_from_near_edge for z in self.zones],
            "j_mA_cm2": [round(z.local_j_mA_cm2, 1) for z in self.zones],
            "delta_um": [round(z.local_delta_m * 1e6, 1) for z in self.zones],
            "FE_%": [round(z.fe_percent, 1) for z in self.zones],
            "appearance": [z.appearance for z in self.zones],
            "surface_pH": [round(z.surface_pH, 2) for z in self.zones],
            "mass_mg_min": [round(z.mass_gain_mg_per_min, 2) for z in self.zones],
            "precip": [z.precipitation_active for z in self.zones],
        })

    def fe_window(
        self, min_fe_pct: float = 70.0,
    ) -> dict:
        """Panel coverage with FE above a threshold.

        Analogous to ``current_density_window`` in hull_cell.py.
        """
        selected = [z for z in self.zones if z.fe_percent >= min_fe_pct]
        total_area = sum(z.zone_area_cm2 for z in self.zones)
        total_current = sum(
            z.local_j_mA_cm2 * z.zone_area_cm2 for z in self.zones
        )  # proportional
        if not selected:
            return {
                "min_fe_pct": min_fe_pct,
                "n_zones": 0,
                "area_fraction": 0.0,
                "current_fraction": 0.0,
                "position_range_cm": None,
            }
        sel_area = sum(z.zone_area_cm2 for z in selected)
        sel_current = sum(
            z.local_j_mA_cm2 * z.zone_area_cm2 for z in selected
        )
        return {
            "min_fe_pct": min_fe_pct,
            "n_zones": len(selected),
            "area_fraction": sel_area / total_area if total_area > 0 else 0.0,
            "current_fraction": sel_current / total_current if total_current > 0 else 0.0,
            "position_range_cm": (
                selected[0].position_cm_from_near_edge,
                selected[-1].position_cm_from_near_edge,
            ),
        }


# ─── Main prediction ─────────────────────────────────────────────────

def hull_cell_fe_prediction(config: HullCellFEConfig | None = None) -> HullCellFEResult:
    """Run the Hull-cell FE prediction sweep.

    Chains the primary-current distribution from ``hull_cell.py`` with the
    1D diffusion-layer model from ``diffusion_layer_1d.py`` to produce a
    position-resolved FE prediction across the panel.

    Parameters
    ----------
    config : HullCellFEConfig, optional
        Full parameter set.  Uses defaults if omitted.

    Returns
    -------
    HullCellFEResult
        Per-zone FE, appearance, mass gain, and summary statistics.
    """
    if config is None:
        config = HullCellFEConfig()

    # 1. Primary current distribution (fine grid)
    j_dist = hull_current_distribution(
        config.geometry, config.total_current_A, config.n_hull_segments,
    )

    # 2. Bin into zones — each zone spans n_hull_segments/n_zones segments
    n = config.n_zones
    seg_per_zone = max(len(j_dist) // n, 1)
    zones: list[ZoneResult] = []

    for zone_idx in range(n):
        start = zone_idx * seg_per_zone
        end = start + seg_per_zone if zone_idx < n - 1 else len(j_dist)
        zone_df = j_dist.iloc[start:end]
        if zone_df.empty:
            continue

        # Area-weighted average j in this zone
        area = float(zone_df["segment_area_cm2"].sum())
        j_avg = float(
            (zone_df["current_density_mA_cm2"] * zone_df["segment_area_cm2"]).sum()
            / area
        )
        pos = float(zone_df["position_cm_from_near_edge"].mean())
        zone_width = float(zone_df["segment_width_cm"].sum())

        # 3. Local delta at this position
        delta_m = float(config.delta.delta_at_position(
            pos, config.geometry.panel_length_cm,
        ))

        # 4. Solve diffusion-layer model
        model = DiffusionLayer1D(
            fe_conc_M=config.fe_conc_M,
            pH_bulk=config.pH_bulk,
            temperature_C=config.temperature_C,
            delta_m=delta_m,
            buffer_conc_M=config.buffer_conc_M,
            support_conc_M=config.support_conc_M,
            fe_i0=config.fe_i0,
            her_i0=config.her_i0,
            fe_tafel_V=config.fe_tafel_V,
            her_tafel_V=config.her_tafel_V,
            grid_points=config.grid_points,
        )
        result = model.solve(j_avg)

        fe = result.current_efficiency

        # 5. Mass gain:  mg/min = j(A/cm²) × area(cm²) × FE × M(g/mol) / (nF) × 60(s/min) × 1000(mg/g)
        j_A_cm2 = j_avg / 1000.0
        M_FE = 55.845  # g/mol
        FARADAY = 96485.33212  # C/mol
        mass_gain_mg_per_min = (
            j_A_cm2 * area * fe * M_FE / (2.0 * FARADAY) * 60.0 * 1000.0
        )

        zones.append(ZoneResult(
            zone_index=zone_idx,
            position_cm_from_near_edge=pos,
            zone_width_cm=zone_width,
            local_j_mA_cm2=j_avg,
            local_delta_m=delta_m,
            faradaic_efficiency=fe,
            fe_percent=fe * 100.0,
            appearance=appearance_from_fe(fe),
            surface_pH=result.surface_pH,
            surface_fe_M=result.surface_fe_M,
            precipitation_active=result.precipitation_active,
            feoh2_supersaturation=result.feoh2_supersaturation,
            mass_gain_mg_per_min=mass_gain_mg_per_min,
            zone_area_cm2=area,
            detail=result,
        ))

    return HullCellFEResult(
        config=config,
        zones=zones,
        j_distribution=j_dist,
    )


# ─── Sensitivity analysis ────────────────────────────────────────────

def fe_sensitivity_to_delta(
    config: HullCellFEConfig | None = None,
    multipliers: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0),
) -> pd.DataFrame:
    """How much does delta variation affect the FE(position) curve?

    Runs the prediction at several delta scaling factors and returns a
    comparison DataFrame.

    Parameters
    ----------
    config : HullCellFEConfig, optional
        Base configuration (delta is scaled from this).
    multipliers : tuple of float
        Scaling factors applied uniformly to both near and far delta.

    Returns
    -------
    pd.DataFrame
        Columns: multiplier, position_cm, FE_percent, delta_um, appearance
    """
    if config is None:
        config = HullCellFEConfig()

    rows = []
    for m in multipliers:
        cfg = HullCellFEConfig(
            geometry=config.geometry,
            total_current_A=config.total_current_A,
            n_hull_segments=config.n_hull_segments,
            n_zones=config.n_zones,
            fe_conc_M=config.fe_conc_M,
            pH_bulk=config.pH_bulk,
            temperature_C=config.temperature_C,
            buffer_conc_M=config.buffer_conc_M,
            support_conc_M=config.support_conc_M,
            delta=DeltaProfile(
                delta_near_m=config.delta.delta_near_m * m,
                delta_far_m=config.delta.delta_far_m * m,
            ),
            fe_i0=config.fe_i0,
            her_i0=config.her_i0,
            fe_tafel_V=config.fe_tafel_V,
            her_tafel_V=config.her_tafel_V,
            grid_points=config.grid_points,
        )
        result = hull_cell_fe_prediction(cfg)
        for z in result.zones:
            rows.append({
                "multiplier": m,
                "position_cm": z.position_cm_from_near_edge,
                "j_mA_cm2": z.local_j_mA_cm2,
                "FE_percent": z.fe_percent,
                "delta_um": z.local_delta_m * 1e6,
                "appearance": z.appearance,
                "surface_pH": z.surface_pH,
            })

    return pd.DataFrame(rows)


# ─── FE window comparison ────────────────────────────────────────────

def compare_fe_windows(
    config: HullCellFEConfig | None = None,
    fe_threshold_pct: float = 70.0,
    j_sweep_points: int = 50,
) -> dict:
    """Compare the j-window from the Hull-panel FE(position) to standalone FE(j).

    The hypothesis: if the panel-predicted j-window matches the j-window
    from FE(j) alone, then the Hull cell experiment can be interpreted
    directly from the 1D model without coupling to geometry.

    Parameters
    ----------
    config : HullCellFEConfig, optional
        Hull cell configuration.
    fe_threshold_pct : float
        FE threshold (%) for the "useful" window.
    j_sweep_points : int
        Number of j values in the standalone sweep.

    Returns
    -------
    dict
        panel_j_window_mA_cm2, standalone_j_window_mA_cm2, match
    """
    if config is None:
        config = HullCellFEConfig()

    # Panel: get j range of zones above threshold
    panel_result = hull_cell_fe_prediction(config)
    good_zones = [z for z in panel_result.zones if z.fe_percent >= fe_threshold_pct]
    if good_zones:
        panel_j_min = min(z.local_j_mA_cm2 for z in good_zones)
        panel_j_max = max(z.local_j_mA_cm2 for z in good_zones)
    else:
        panel_j_min = panel_j_max = None

    # Standalone: sweep FE(j) at the panel's midpoint delta
    delta_mid_m = config.delta.delta_at_position(
        5.0, config.geometry.panel_length_cm,
    )
    model = DiffusionLayer1D(
        fe_conc_M=config.fe_conc_M,
        pH_bulk=config.pH_bulk,
        temperature_C=config.temperature_C,
        delta_m=float(delta_mid_m),
        buffer_conc_M=config.buffer_conc_M,
        support_conc_M=config.support_conc_M,
        fe_i0=config.fe_i0,
        her_i0=config.her_i0,
        fe_tafel_V=config.fe_tafel_V,
        her_tafel_V=config.her_tafel_V,
        grid_points=config.grid_points,
    )

    j_range = np.linspace(5.0, 500.0, j_sweep_points)
    standalone_good = []
    for j in j_range:
        s = model.solve(float(j))
        if s.current_efficiency * 100.0 >= fe_threshold_pct:
            standalone_good.append(float(j))

    if standalone_good:
        standalone_j_min = min(standalone_good)
        standalone_j_max = max(standalone_good)
    else:
        standalone_j_min = standalone_j_max = None

    # Match criterion: overlap of j ranges
    if panel_j_min is not None and standalone_j_min is not None:
        overlap_lo = max(panel_j_min, standalone_j_min)
        overlap_hi = min(panel_j_max, standalone_j_max)
        match = overlap_lo < overlap_hi
    else:
        match = False

    return {
        "fe_threshold_pct": fe_threshold_pct,
        "panel_j_window_mA_cm2": (panel_j_min, panel_j_max),
        "standalone_j_window_mA_cm2": (standalone_j_min, standalone_j_max),
        "standalone_delta_um": float(delta_mid_m) * 1e6,
        "match": match,
    }
