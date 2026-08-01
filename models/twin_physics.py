"""
Physics-coupled process model for the digital twin.

The twin's job is to keep a real cell's hidden state consistent with sensor
readings *and* with the repository's electrochemistry.  This module supplies
the physics half of that loop: given an operating point (current density,
temperature, Fe(II) concentration), it returns what the physical cell models
say the observable quantities should be — Faradaic efficiency, cell voltage,
cathode surface pH, deposit rate and transport margin.

The heavy physics (`CellPhysics.solve_at_j`, a self-consistent Nernst–Planck +
voltage solve) is far too slow to call inside an online EKF (~0.3 s per solve).
``CellProcessModel`` therefore precomputes a surrogate of that solve over a
coarse operating grid once (offline) and interpolates in microseconds online.

    model = CellProcessModel()                      # build with defaults (slow, once)
    y = model.predict(j_mA_cm2=150.0, temperature_C=60.0, fe2_M=1.0)

The surrounding twin uses ``predict`` as its measurement model; the difference
between ``predict`` and the real sensors is the signal calibration acts on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .cell_physics import BathRecipe, CellGeometry, CellPhysics, ProcessConditions
from .electrochemistry import FARADAY, M_FE, Z_FE

# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------


@dataclass
class ProcessPrediction:
    """Physics-predicted observables at one operating point."""

    j_mA_cm2: float
    temperature_C: float
    fe2_M: float
    current_efficiency: float
    fe_percent: float
    v_cell_V: float
    v_cathode_V: float
    surface_pH: float
    deposit_rate_um_hr: float
    fe_current_A_m2: float
    transport_margin: float  # transport_limit / applied (>=1 = not limit-bound)

    def to_dict(self) -> Dict[str, float]:
        return {
            "j_mA_cm2": self.j_mA_cm2,
            "temperature_C": self.temperature_C,
            "fe2_M": self.fe2_M,
            "current_efficiency": self.current_efficiency,
            "fe_percent": self.fe_percent,
            "v_cell_V": self.v_cell_V,
            "v_cathode_V": self.v_cathode_V,
            "surface_pH": self.surface_pH,
            "deposit_rate_um_hr": self.deposit_rate_um_hr,
            "fe_current_A_m2": self.fe_current_A_m2,
            "transport_margin": self.transport_margin,
        }


# ---------------------------------------------------------------------------
# Surrogate process model
# ---------------------------------------------------------------------------


def _deposit_rate_um_hr(
    j_mA_cm2: float, current_efficiency: float, temperature_C: float
) -> float:
    """Deposit growth rate in µm/hr from Faraday's law.

    Mass flux kg/(m²·s) = j_A_m2 * FE * M_Fe / (z F).  Converted to a volumetric
    growth rate assuming dense bcc iron (7874 kg/m³).
    """
    j_A_m2 = j_mA_cm2 * 10.0  # mA/cm² -> A/m²
    mass_flux_kg_m2_s = j_A_m2 * current_efficiency * M_FE / (Z_FE * FARADAY)
    # µm/hr = (kg/m²/s) / rho * 1e6 * 3600
    return mass_flux_kg_m2_s / 7874.0 * 1e6 * 3600.0


class CellProcessModel:
    """Fast interpolated surrogate of the self-consistent cell physics.

    Parameters
    ----------
    bath : BathRecipe, optional
        Reference bath (Fe(II), support, buffer, bulk pH).
    geometry : CellGeometry, optional
        Cell geometry / membrane configuration.
    conditions : ProcessConditions, optional
        Kinetic + transport operating conditions.
    j_grid, T_grid, fe2_grid : sequence, optional
        Coarse surrogate grid (defaults cover the realistic electrowinning range).
    cache_path : str | Path, optional
        If set, persist/load the surrogate arrays as JSON so the (slow) build
        and heavy Nernst–Planck solve only run once.
    """

    def __init__(
        self,
        bath: Optional[BathRecipe] = None,
        geometry: Optional[CellGeometry] = None,
        conditions: Optional[ProcessConditions] = None,
        j_grid: Optional[Sequence[float]] = None,
        T_grid: Optional[Sequence[float]] = None,
        fe2_grid: Optional[Sequence[float]] = None,
        cache_path: Optional[str] = None,
    ) -> None:
        self.bath = bath or BathRecipe()
        self.geometry = geometry or CellGeometry()
        self.conditions = conditions or ProcessConditions()

        self.j_grid = np.asarray(j_grid or (50.0, 100.0, 150.0, 200.0, 250.0), dtype=float)
        self.T_grid = np.asarray(T_grid or (40.0, 60.0, 80.0), dtype=float)
        self.fe2_grid = np.asarray(fe2_grid or (0.5, 1.0, 1.5), dtype=float)

        self._load_or_build(cache_path)
        self._make_interpolators()

    # -- build -----------------------------------------------------------------
    def _load_or_build(self, cache_path: Optional[str]) -> None:
        self.cache_path = Path(cache_path) if cache_path else None
        if self.cache_path and self.cache_path.exists():
            data = json.loads(self.cache_path.read_text())
            self.fe_map = np.asarray(data["fe"])
            self.vcell_map = np.asarray(data["vcell"])
            self.surf_ph_map = np.asarray(data["surf_ph"])
            self.dep_map = np.asarray(data["dep"])
            self.j_grid = np.asarray(data["j"])
            self.T_grid = np.asarray(data["T"])
            self.fe2_grid = np.asarray(data["fe2"])
            return
        self._build()
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(
                {
                    "j": self.j_grid.tolist(),
                    "T": self.T_grid.tolist(),
                    "fe2": self.fe2_grid.tolist(),
                    "fe": self.fe_map.tolist(),
                    "vcell": self.vcell_map.tolist(),
                    "surf_ph": self.surf_ph_map.tolist(),
                    "dep": self.dep_map.tolist(),
                }
            ))

    def _build(self) -> None:
        """Solve the full physics on the (T, fe2) x j grid."""
        nj, nT, nf = len(self.j_grid), len(self.T_grid), len(self.fe2_grid)
        fe = np.zeros((nj, nT, nf))
        vcell = np.zeros((nj, nT, nf))
        surf_ph = np.zeros((nj, nT, nf))
        dep = np.zeros((nj, nT, nf))
        for ni, fe2 in enumerate(self.fe2_grid):
            for ti, T in enumerate(self.T_grid):
                # bath with varying Fe(II); conditions with varying T
                bath = BathRecipe(
                    c_FeSO4_M=float(fe2),
                    c_Na2SO4_M=self.bath.c_Na2SO4_M,
                    c_H3BO3_M=self.bath.c_H3BO3_M,
                    pH=self.bath.pH,
                )
                cond = ProcessConditions(
                    temperature_C=float(T),
                    boundary_layer_m=self.conditions.boundary_layer_m,
                    fe_i0=self.conditions.fe_i0,
                    her_i0=self.conditions.her_i0,
                    fe_tafel_V=self.conditions.fe_tafel_V,
                    her_tafel_V=self.conditions.her_tafel_V,
                )
                cp = CellPhysics(bath, self.geometry, cond)
                for ji, j in enumerate(self.j_grid):
                    p = cp.solve_at_j(float(j))
                    fe[ji, ti, ni] = p.current_efficiency
                    vcell[ji, ti, ni] = p.V_cell
                    surf_ph[ji, ti, ni] = p.surface_pH
                    dep[ji, ti, ni] = _deposit_rate_um_hr(
                        float(j), p.current_efficiency, float(T)
                    )
        self.fe_map, self.vcell_map = fe, vcell
        self.surf_ph_map, self.dep_map = surf_ph, dep

    def _make_interpolators(self) -> None:
        # grid order: (j, T, fe2) for RegularGridInterpolator
        self._fe_itp = RegularGridInterpolator(
            (self.j_grid, self.T_grid, self.fe2_grid), self.fe_map,
            bounds_error=False, fill_value=None)
        self._vcell_itp = RegularGridInterpolator(
            (self.j_grid, self.T_grid, self.fe2_grid), self.vcell_map,
            bounds_error=False, fill_value=None)
        self._ph_itp = RegularGridInterpolator(
            (self.j_grid, self.T_grid, self.fe2_grid), self.surf_ph_map,
            bounds_error=False, fill_value=None)
        self._dep_itp = RegularGridInterpolator(
            (self.j_grid, self.T_grid, self.fe2_grid), self.dep_map,
            bounds_error=False, fill_value=None)

    # -- query ----------------------------------------------------------------
    def predict(
        self,
        j_mA_cm2: float,
        temperature_C: float,
        fe2_M: float,
    ) -> ProcessPrediction:
        """Physics-predicted observables at one operating point (fast)."""
        pts = np.array([[j_mA_cm2, temperature_C, fe2_M]])
        fe = float(np.clip(self._fe_itp(pts)[0], 0.0, 1.0))
        vcell = float(self._vcell_itp(pts)[0])
        surf_ph = float(self._ph_itp(pts)[0])
        dep = float(self._dep_itp(pts)[0])

        j_A_m2 = j_mA_cm2 * 10.0
        return ProcessPrediction(
            j_mA_cm2=float(j_mA_cm2),
            temperature_C=float(temperature_C),
            fe2_M=float(fe2_M),
            current_efficiency=fe,
            fe_percent=fe * 100.0,
            v_cell_V=vcell,
            v_cathode_V=vcell,  # surrogate keeps the self-consistent value
            surface_pH=surf_ph,
            deposit_rate_um_hr=dep,
            fe_current_A_m2=j_A_m2 * fe,
            transport_margin=float("nan"),
        )

    def predict_grid(self, j_mA_cm2: float, temperature_C: float, fe2_M: float) -> Dict[str, np.ndarray]:
        """Return the raw surrogate arrays (for inspection / plots)."""
        return {"fe": self.fe_map, "v_cell": self.vcell_map,
                "surface_pH": self.surf_ph_map, "deposit_rate": self.dep_map}

    def fe(self, j_mA_cm2: float, temperature_C: float, fe2_M: float) -> float:
        return float(np.clip(self._fe_itp([[j_mA_cm2, temperature_C, fe2_M]])[0], 0.0, 1.0))

    @property
    def nominal(self) -> Dict[str, float]:
        """A representative operating point from the grid centre."""
        return {
            "temperature_C": float(np.median(self.T_grid)),
            "j_avg_mA_cm2": float(np.median(self.j_grid)),
            "fe2_M": float(np.median(self.fe2_grid)),
            "cell_voltage_V": self.predict(
                float(np.median(self.j_grid)),
                float(np.median(self.T_grid)),
                float(np.median(self.fe2_grid)),
            ).v_cell_V,
        }


def default_process_model(cache_path: Optional[str] = None) -> CellProcessModel:
    """Convenience constructor with the repository default reference bath."""
    return CellProcessModel(cache_path=cache_path)


# ---------------------------------------------------------------------------
# Physics-grounded synthetic sensor stream
# ---------------------------------------------------------------------------

_PHYSICS_SENSOR_SPECS = {
    # quantity -> (tag, noise_std, physical units)           matching digital_twin tags
    "cell_voltage": ("VT-201", 0.01),
    "catholyte_temperature": ("TT-101", 0.5),
    "catholyte_pH": ("pHAT-101", 0.05),
}


def generate_physics_readings(
    model: CellProcessModel,
    j_mA_cm2: float,
    temperature_C: float,
    fe2_M: float,
    rng: np.random.Generator,
    electrode_area_m2: float = 1.0,
    fault: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Sensor readings derived from the physics surrogate plus measurement noise.

    Unlike the heuristic ``generate_synthetic_readings`` in ``digital_twin``,
    the observables here (cell voltage, temperature, pH) are those the physical
    cell models actually predict at the given operating point, so the twin's
    measurement model and its sensor stream come from the same physics.
    """
    p = model.predict(j_mA_cm2, temperature_C, fe2_M)
    readings: Dict[str, float] = {}
    readings["VT-201"] = p.v_cell_V + rng.normal(0, 0.01)
    readings["TT-101"] = temperature_C + rng.normal(0, 0.5)
    readings["TT-201"] = temperature_C + 1.5 + rng.normal(0, 0.5)
    readings["pHAT-101"] = model.bath.pH + rng.normal(0, 0.05)
    readings["AT-202"] = model.bath.pH + 0.15 + rng.normal(0, 0.05)
    readings["FT-201"] = 10.0 + rng.normal(0, 0.2)
    readings["FT-202"] = 10.2 + rng.normal(0, 0.2)
    readings["FT-103"] = 20.0 + rng.normal(0, 0.3)
    readings["AT-201"] = 7.5 + rng.normal(0, 0.1)
    readings["AT-301A"] = 20.9 + rng.normal(0, 0.2)
    readings["AIT-501"] = 420.0 + rng.normal(0, 1.0)
    readings["AIT-502"] = 15.0 + rng.normal(0, 0.3)
    readings["CT-201"] = j_mA_cm2 * electrode_area_m2 * 10.0 + rng.normal(0, 5.0)

    if fault is not None:
        tag = fault.get("tag")
        kind = fault.get("kind")
        mag = fault.get("magnitude", 5.0)
        if tag in readings:
            if kind == "bias":
                readings[tag] += mag
            elif kind == "stuck":
                readings[tag] = fault.get("stuck_value", readings[tag])
            elif kind == "spike" and fault.get("spike_active", True):
                readings[tag] += mag
    return readings
