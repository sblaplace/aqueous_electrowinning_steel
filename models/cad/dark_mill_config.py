"""
Dark Mill — parametric 3D CAD model.

Generates a deployable dark mill unit sized from the physics-driven
dark_mill.py outputs.  Answers deployment questions:
  - Does it fit on a tractor trailer?
  - Where does rainwater accumulate?
  - Can a forklift move it?
  - What are the utility connection points?

All dimensions flow from DarkMillConfig — never hardcode.
CadQuery required (needs libGL on headless systems).

Usage:
    # Via shell (needs LD_LIBRARY_PATH for CadQuery/OCC on NixOS)
    python -m models.cad.dark_mill_cad --site pickle_liquor_us_midwest
    python -m models.cad.dark_mill_cad --export-step docs/cad/dark_mill.step

Origin: center of floor, Z up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import math


# ─── Config ───────────────────────────────────────────────────────

@dataclass
class DarkMillConfig:
    """Parametric config for the dark mill 3D model.  All dims in mm."""

    # Enclosure (standard 20ft container footprint)
    enclosure_length: float = 6058.0     # 20ft container
    enclosure_width: float = 2438.0      # 8ft container
    enclosure_height: float = 2896.0     # 9ft 6in high-cube
    wall_thickness: float = 2.0          # corrugated steel
    frame_member_size: float = 80.0      # square tube cross-section
    roof_slope_deg: float = 3.0          # slope for rainwater runoff
    floor_thickness: float = 3.0         # steel floor plate

    # Forklift pockets (standard 20ft container)
    forklift_pocket_width: float = 200.0
    forklift_pocket_height: float = 100.0
    forklift_pocket_spacing: float = 900.0  # between centers
    forklift_pocket_length: float = 1200.0

    # Cell stacks
    n_stacks: int = 7
    cells_per_stack: int = 20
    electrode_width: float = 707.0       # sqrt(0.5 m²) × 1000
    electrode_height: float = 707.0
    cell_gap: float = 40.0              # interelectrode + membrane + frame
    stack_frame_width: float = 50.0      # frame around each cell
    stack_base_height: float = 200.0     # manifold/base under stack

    # Electrolyte tanks
    feed_tank_length: float = 2000.0
    feed_tank_width: float = 800.0
    feed_tank_height: float = 1200.0
    product_tank_length: float = 1500.0
    product_tank_width: float = 800.0
    product_tank_height: float = 1000.0
    tank_wall_thickness: float = 6.0     # PP or SS

    # Power electronics cabinet
    rectifier_length: float = 1200.0
    rectifier_width: float = 600.0
    rectifier_height: float = 1800.0

    # Heat exchanger
    hx_length: float = 800.0
    hx_width: float = 600.0
    hx_height: float = 1200.0

    # Control cabinet
    control_length: float = 600.0
    control_width: float = 400.0
    control_height: float = 1800.0

    # Piping
    main_pipe_diameter: float = 100.0    # DN100
    pipe_wall_thickness: float = 4.0

    # Door (one end)
    door_width: float = 2200.0
    door_height: float = 2500.0

    # Lifting lugs
    lug_diameter: float = 60.0
    lug_height: float = 100.0

    # Post-processing equipment (carburization furnace)
    post_processing_route: str = "none"  # "none", "carburize", "codeposit"
    furnace_length: float = 1800.0       # batch furnace
    furnace_width: float = 1200.0
    furnace_height: float = 1500.0
    furnace_wall_thickness: float = 150.0  # refractory + insulation
    quench_tank_length: float = 1500.0
    quench_tank_width: float = 1000.0
    quench_tank_height: float = 1200.0
    gas_supply_length: float = 600.0     # gas panel / manifold cabinet
    gas_supply_width: float = 400.0
    gas_supply_height: float = 1200.0

    # Post-processing equipment (co-deposition particle system)
    particle_hopper_diameter: float = 600.0   # cylindrical hopper
    particle_hopper_height: float = 1000.0
    ultrasonic_bath_length: float = 1200.0
    ultrasonic_bath_width: float = 800.0
    ultrasonic_bath_height: float = 600.0

    # Clearance margins
    side_clearance: float = 100.0        # from wall to equipment
    top_clearance: float = 200.0         # from roof to tallest equipment

    @classmethod
    def from_sizing(cls, sizing: dict) -> "DarkMillConfig":
        """Build config from dark_mill.py sizing output."""
        sd = sizing.get("stack_design", {})
        pp = sizing.get("post_processing", {})
        route = pp.get("route", "none") if isinstance(pp, dict) else "none"
        # Carburization furnace sizing from post-processing result
        furnace_kw = {}
        if route == "carburize" and isinstance(pp, dict):
            carb = pp.get("carburization", {})
            if isinstance(carb, dict):
                furnace_kw = {
                    "furnace_length": carb.get("furnace_length_mm", 1800.0),
                    "furnace_width": carb.get("furnace_width_mm", 1200.0),
                    "furnace_height": carb.get("furnace_height_mm", 1500.0),
                    "quench_tank_length": max(1000.0, carb.get("furnace_length_mm", 1800.0) * 0.8),
                    "quench_tank_width": max(800.0, carb.get("furnace_width_mm", 1200.0) * 0.8),
                }
        return cls(
            n_stacks=sd.get("n_stacks", 7),
            cells_per_stack=sd.get("cells_per_stack", 20),
            post_processing_route=route,
            **furnace_kw,
        )

    @property
    def stack_height(self) -> float:
        """Total height of one cell stack (mm)."""
        return (self.stack_base_height
                + self.cells_per_stack * self.cell_gap
                + self.stack_frame_width)

    @property
    def n_stacks_per_row(self) -> int:
        """How many stacks fit side-by-side."""
        stack_total_width = self.electrode_width + 2 * self.stack_frame_width
        available = self.enclosure_length - 2 * self.side_clearance
        return max(1, int(available / (stack_total_width + 100)))

    @property
    def n_stack_rows(self) -> int:
        """How many rows of stacks."""
        return max(1, math.ceil(self.n_stacks / self.n_stacks_per_row))

    def enclosure_inner_length(self) -> float:
        return self.enclosure_length - 2 * self.frame_member_size

    def enclosure_inner_width(self) -> float:
        return self.enclosure_width - 2 * self.frame_member_size

    def enclosure_inner_height(self) -> float:
        return self.enclosure_height - self.frame_member_size - self.floor_thickness

    def to_dict(self) -> dict:
        """Serialize config to dict."""
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith("_")}


# ─── Deployment Queries ───────────────────────────────────────────

# Standard transport limits
TRAILER_MAX_WIDTH = 2600.0     # mm (US standard)
TRAILER_MAX_HEIGHT = 4000.0    # mm (US standard, ground to top)
TRAILER_MAX_LENGTH = 16500.0   # mm (US standard 53ft)
CONTAINER_20FT_LENGTH = 6058.0
CONTAINER_40FT_LENGTH = 12192.0


def check_transportability(cfg: DarkMillConfig) -> Dict[str, Any]:
    """Check if the unit fits on standard transport."""
    return {
        "fits_width": cfg.enclosure_width <= TRAILER_MAX_WIDTH,
        "fits_height": cfg.enclosure_height <= TRAILER_MAX_HEIGHT,
        "fits_length_20ft": cfg.enclosure_length <= CONTAINER_20FT_LENGTH,
        "fits_length_40ft": cfg.enclosure_length <= CONTAINER_40FT_LENGTH,
        "fits_trailer": (cfg.enclosure_width <= TRAILER_MAX_WIDTH
                         and cfg.enclosure_height <= TRAILER_MAX_HEIGHT),
        "enclosure_dims_mm": (cfg.enclosure_length, cfg.enclosure_width, cfg.enclosure_height),
        "trailer_limit_mm": (TRAILER_MAX_LENGTH, TRAILER_MAX_WIDTH, TRAILER_MAX_HEIGHT),
        "weight_estimate_kg": _estimate_weight(cfg),
    }


def _estimate_weight(cfg: DarkMillConfig) -> float:
    """Rough weight estimate for the loaded unit."""
    # Enclosure steel: surface area × wall thickness × density
    surface = 2 * (cfg.enclosure_length * cfg.enclosure_width
                   + cfg.enclosure_length * cfg.enclosure_height
                   + cfg.enclosure_width * cfg.enclosure_height)
    steel_density = 7.85e-6  # kg/mm³
    enclosure_kg = surface * cfg.wall_thickness * steel_density
    # Frame
    frame_perimeter = 4 * (cfg.enclosure_length + cfg.enclosure_width + cfg.enclosure_height)
    frame_kg = frame_perimeter * cfg.frame_member_size**2 * steel_density
    # Tanks (filled): ~1.2 kg/L for electrolyte
    feed_vol_L = cfg.feed_tank_length * cfg.feed_tank_width * cfg.feed_tank_height / 1e9 * 1000
    prod_vol_L = cfg.product_tank_length * cfg.product_tank_width * cfg.product_tank_height / 1e9 * 1000
    tank_kg = (feed_vol_L + prod_vol_L) * 1.2
    # Stacks: rough ~200 kg per stack
    stack_kg = cfg.n_stacks * 200
    # Electronics, piping, etc.
    bop_kg = 2000

    return round(enclosure_kg + frame_kg + tank_kg + stack_kg + bop_kg, 0)


def check_rainwater(cfg: DarkMillConfig) -> Dict[str, Any]:
    """Analyze rainwater behavior on the enclosure."""
    # Roof slope determines runoff direction
    slope_rad = math.radians(cfg.roof_slope_deg)
    height_diff = cfg.enclosure_width * math.tan(slope_rad)

    return {
        "roof_slope_deg": cfg.roof_slope_deg,
        "roof_height_diff_mm": round(height_diff, 1),
        "runoff_direction": "toward low side (width direction)",
        "low_side": "one long side of the enclosure",
        "recommendation": "Install gutter along low side. "
                          "No flat surfaces for water pooling on roof.",
        "door_risk": "Door end is flat — add drip rail above door header",
        "penetrations": "All cable/pipe entries should have boot seals. "
                        "Use weatherproof connectors at utility end.",
        "floor_drain": "Add floor drain at low corner for washdown/spill recovery",
        "puddle_risk_zones": [
            "Door threshold (add raised sill)",
            "Around forklift pockets (seal pocket ends)",
            "Utility connection panel (add canopy)",
        ],
    }


def check_maintenance_access(cfg: DarkMillConfig) -> Dict[str, Any]:
    """Check maintenance access paths."""
    aisle_width = cfg.enclosure_inner_width() - 2 * (cfg.feed_tank_width + cfg.side_clearance)

    return {
        "center_aisle_width_mm": round(aisle_width, 0),
        "aisle_sufficient": aisle_width >= 800,  # OSHA minimum
        "stack_access": "Front access to each stack from center aisle",
        "tank_access": "Top-access hatches on tanks",
        "electrical_access": "Front panel access to rectifier and control cabinets",
        "door_type": "Double swing or roll-up for full-end access",
        "crane_lift_points": 4,
        "forklift_pockets": True,
    }
