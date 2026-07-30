"""
CadQuery builders for the dark mill 3D model.

Each builder returns a CadQuery Workplane. Assembly function composites
them into the full deployable unit.  All dimensions from DarkMillConfig.

Requires CadQuery + OCC (needs libGL on headless systems).
"""

from __future__ import annotations

from typing import Optional
import math

try:
    import cadquery as cq
except ImportError:
    raise ImportError(
        "CadQuery required. Install: pip install cadquery. "
        "On NixOS: set LD_LIBRARY_PATH to include libGL, libX11, etc."
    )

from .dark_mill_config import DarkMillConfig


# ─── Helpers ──────────────────────────────────────────────────────

def _box(x: float, y: float, z: float, dx: float, dy: float, dz: float):
    """Place a box at (x, y, z) corner with dimensions (dx, dy, dz)."""
    return (cq.Workplane("XY")
            .transformed(offset=(x + dx/2, y + dy/2, z + dz/2))
            .box(dx, dy, dz))


def _cylinder(x: float, y: float, z: float, r: float, h: float):
    """Place a cylinder at (x, y, z) with radius r and height h."""
    return (cq.Workplane("XY")
            .transformed(offset=(x, y, z + h/2))
            .circle(r).extrude(h))


# ─── Enclosure ────────────────────────────────────────────────────

def build_frame(cfg: DarkMillConfig):
    """Steel tube frame — corner posts + roof/floor perimeter."""
    L, W, H = cfg.enclosure_length, cfg.enclosure_width, cfg.enclosure_height
    s = cfg.frame_member_size

    result = cq.Workplane("XY")

    # 8 corner posts
    for x in [0, L - s]:
        for y in [0, W - s]:
            post = _box(x, y, 0, s, s, H)
            result = result.union(post)

    # Floor perimeter beams
    for y in [0, W - s]:
        beam = _box(s, y, 0, L - 2*s, s, s)
        result = result.union(beam)
    for x in [0, L - s]:
        beam = _box(x, s, 0, s, W - 2*s, s)
        result = result.union(beam)

    # Roof perimeter beams
    for y in [0, W - s]:
        beam = _box(s, y, H - s, L - 2*s, s, s)
        result = result.union(beam)
    for x in [0, L - s]:
        beam = _box(x, s, H - s, s, W - 2*s, s)
        result = result.union(beam)

    # Mid-height horizontal members (for rigidity)
    mid_z = H / 2
    for y in [0, W - s]:
        beam = _box(s, y, mid_z - s/2, L - 2*s, s, s)
        result = result.union(beam)

    return result


def build_floor(cfg: DarkMillConfig):
    """Steel floor plate."""
    return _box(cfg.frame_member_size, cfg.frame_member_size, 0,
                cfg.enclosure_length - 2*cfg.frame_member_size,
                cfg.enclosure_width - 2*cfg.frame_member_size,
                cfg.floor_thickness)


def build_roof(cfg: DarkMillConfig):
    """Sloped roof panel for rainwater runoff."""
    L = cfg.enclosure_length - 2 * cfg.frame_member_size
    W = cfg.enclosure_width - 2 * cfg.frame_member_size
    slope_rad = math.radians(cfg.roof_slope_deg)
    height_diff = W * math.tan(slope_rad)

    H_base = cfg.enclosure_height - cfg.frame_member_size

    # Simple sloped box (approximation — real roof would be a ruled surface)
    avg_h = cfg.wall_thickness
    result = _box(cfg.frame_member_size, cfg.frame_member_size,
                  H_base, L, W, avg_h)

    return result


def build_walls(cfg: DarkMillConfig):
    """Corrugated wall panels (simplified as flat plates)."""
    L, W, H = cfg.enclosure_length, cfg.enclosure_width, cfg.enclosure_height
    s = cfg.frame_member_size
    t = cfg.wall_thickness
    inner_L = L - 2*s
    inner_W = W - 2*s
    inner_H = H - s - cfg.floor_thickness

    result = cq.Workplane("XY")

    # Long walls (front and back)
    front = _box(s, 0, cfg.floor_thickness, inner_L, t, inner_H)
    result = result.union(front)
    back = _box(s, W - t, cfg.floor_thickness, inner_L, t, inner_H)
    result = result.union(back)

    # Short walls (ends)
    left = _box(0, s, cfg.floor_thickness, t, inner_W, inner_H)
    result = result.union(left)
    # Right wall has door opening — add panels around door
    dw, dh = cfg.door_width, cfg.door_height
    door_z = cfg.floor_thickness
    door_y_start = (W - dw) / 2
    door_y_end = door_y_start + dw

    # Panel above door
    above_door = _box(L - t, door_y_start, door_z + dh,
                       t, dw, inner_H - dh)
    result = result.union(above_door)
    # Panels beside door
    left_of_door = _box(L - t, s, cfg.floor_thickness,
                         t, door_y_start - s, inner_H)
    result = result.union(left_of_door)
    right_of_door = _box(L - t, door_y_end, cfg.floor_thickness,
                          t, W - s - door_y_end, inner_H)
    result = result.union(right_of_door)

    return result


def build_door(cfg: DarkMillConfig):
    """Double swing door (shown open at 90°)."""
    L, W = cfg.enclosure_length, cfg.enclosure_width
    s = cfg.frame_member_size
    dw = cfg.door_width
    dh = cfg.door_height
    t = cfg.wall_thickness

    door_z = cfg.floor_thickness
    door_y_start = (W - dw) / 2

    # Left door panel (open 90° — rotated out)
    half_w = dw / 2
    left_door = (cq.Workplane("XY")
                 .transformed(offset=(L + half_w/2, door_y_start + half_w/2, door_z + dh/2))
                 .box(half_w, t, dh))

    # Right door panel (open 90° — rotated out)
    right_door = (cq.Workplane("XY")
                  .transformed(offset=(L + half_w/2, door_y_start + dw - half_w/2, door_z + dh/2))
                  .box(half_w, t, dh))

    return left_door.union(right_door)


# ─── Process Equipment ────────────────────────────────────────────

def build_cell_stacks(cfg: DarkMillConfig):
    """Cell stacks — bipolar plate assemblies."""
    result = cq.Workplane("XY")

    stack_w = cfg.electrode_width + 2 * cfg.stack_frame_width
    stack_d = cfg.electrode_height + 2 * cfg.stack_frame_width
    stack_h = cfg.stack_height

    # Position stacks in rows
    margin = cfg.side_clearance
    for i in range(cfg.n_stacks):
        row = i // cfg.n_stacks_per_row
        col = i % cfg.n_stacks_per_row

        x = margin + col * (stack_w + 100)
        y = margin + row * (stack_d + 200)
        z = cfg.floor_thickness

        # Stack base/manifold
        base = _box(x, y, z, stack_w, stack_d, cfg.stack_base_height)
        result = result.union(base)

        # Stack cells (simplified as a block)
        cells = _box(x, y, z + cfg.stack_base_height,
                      stack_w, stack_d,
                      cfg.cells_per_stack * cfg.cell_gap)
        result = result.union(cells)

        # Stack top frame
        top = _box(x, y,
                    z + cfg.stack_base_height + cfg.cells_per_stack * cfg.cell_gap,
                    stack_w, stack_d, cfg.stack_frame_width)
        result = result.union(top)

    return result


def build_electrolyte_tanks(cfg: DarkMillConfig):
    """Feed and product electrolyte tanks."""
    result = cq.Workplane("XY")

    L = cfg.enclosure_length
    s = cfg.frame_member_size
    t = cfg.tank_wall_thickness

    # Position tanks behind the stacks (far side from door)
    tank_y = cfg.enclosure_width - cfg.frame_member_size - cfg.feed_tank_width - cfg.side_clearance

    # Feed tank
    feed_x = s + cfg.side_clearance
    feed = _box(feed_x, tank_y, cfg.floor_thickness,
                cfg.feed_tank_length, cfg.feed_tank_width, cfg.feed_tank_height)
    # Hollow it out (tank interior)
    feed_inner = _box(feed_x + t, tank_y + t, cfg.floor_thickness + t,
                       cfg.feed_tank_length - 2*t,
                       cfg.feed_tank_width - 2*t,
                       cfg.feed_tank_height - t)
    feed = feed.cut(feed_inner)
    result = result.union(feed)

    # Product tank (next to feed tank)
    prod_x = feed_x + cfg.feed_tank_length + 200
    prod = _box(prod_x, tank_y, cfg.floor_thickness,
                cfg.product_tank_length, cfg.product_tank_width, cfg.product_tank_height)
    prod_inner = _box(prod_x + t, tank_y + t, cfg.floor_thickness + t,
                       cfg.product_tank_length - 2*t,
                       cfg.product_tank_width - 2*t,
                       cfg.product_tank_height - t)
    prod = prod.cut(prod_inner)
    result = result.union(prod)

    return result


def build_rectifier(cfg: DarkMillConfig):
    """Power electronics / rectifier cabinet."""
    x = cfg.enclosure_length - cfg.frame_member_size - cfg.rectifier_length - cfg.side_clearance
    y = cfg.frame_member_size + cfg.side_clearance
    return _box(x, y, cfg.floor_thickness,
                cfg.rectifier_length, cfg.rectifier_width, cfg.rectifier_height)


def build_heat_exchanger(cfg: DarkMillConfig):
    """Plate heat exchanger for electrolyte cooling."""
    x = cfg.enclosure_length - cfg.frame_member_size - cfg.hx_length - cfg.side_clearance
    y = cfg.frame_member_size + cfg.side_clearance + cfg.rectifier_width + 200
    return _box(x, y, cfg.floor_thickness,
                cfg.hx_length, cfg.hx_width, cfg.hx_height)


def build_control_cabinet(cfg: DarkMillConfig):
    """PLC + instrumentation cabinet."""
    x = cfg.enclosure_length - cfg.frame_member_size - cfg.control_length - cfg.side_clearance
    y = (cfg.enclosure_width - cfg.frame_member_size - cfg.control_width
         - cfg.side_clearance)
    return _box(x, y, cfg.floor_thickness,
                cfg.control_length, cfg.control_width, cfg.control_height)


def build_piping(cfg: DarkMillConfig):
    """Main electrolyte piping (simplified as cylinders)."""
    result = cq.Workplane("XY")

    r = cfg.main_pipe_diameter / 2
    pipe_z = cfg.floor_thickness + cfg.feed_tank_height * 0.3

    # Main run along the stacks (long axis)
    pipe_y = cfg.enclosure_width / 2
    main_pipe = (cq.Workplane("XY")
                 .transformed(offset=(cfg.frame_member_size + cfg.side_clearance,
                                       pipe_y, pipe_z))
                 .circle(r)
                 .extrude(cfg.enclosure_length - 2*cfg.frame_member_size - 2*cfg.side_clearance))
    result = result.union(main_pipe)

    return result


def build_forklift_pockets(cfg: DarkMillConfig):
    """Forklift pocket beams under the floor."""
    result = cq.Workplane("XY")

    pw, ph, pl = cfg.forklift_pocket_width, cfg.forklift_pocket_height, cfg.forklift_pocket_length
    spacing = cfg.forklift_pocket_spacing

    # Two pockets, centered under the unit
    center_y = cfg.enclosure_width / 2
    for offset_y in [-spacing/2, spacing/2]:
        y = center_y + offset_y - pw/2
        x = (cfg.enclosure_length - pl) / 2
        # Pocket beam
        beam = _box(x, y, -ph, pl, pw, ph)
        result = result.union(beam)

    return result


def build_lifting_lugs(cfg: DarkMillConfig):
    """Crane lifting lugs at 4 corners."""
    result = cq.Workplane("XY")
    r = cfg.lug_diameter / 2
    h = cfg.lug_height
    s = cfg.frame_member_size
    L, W = cfg.enclosure_length, cfg.enclosure_height

    for x in [s/2, cfg.enclosure_length - s/2]:
        for y in [s/2, cfg.enclosure_width - s/2]:
            lug = _cylinder(x - r, y - r, cfg.enclosure_height, r, h)
            result = result.union(lug)

    return result


# ─── Assembly ─────────────────────────────────────────────────────

def build_dark_mill(cfg: DarkMillConfig):
    """
    Full dark mill assembly.

    Returns a CadQuery Workplane with all components positioned.
    """
    assembly = cq.Workplane("XY")

    # Structural
    assembly = assembly.union(build_frame(cfg))
    assembly = assembly.union(build_floor(cfg))
    assembly = assembly.union(build_roof(cfg))
    assembly = assembly.union(build_walls(cfg))

    # Process equipment
    assembly = assembly.union(build_cell_stacks(cfg))
    assembly = assembly.union(build_electrolyte_tanks(cfg))
    assembly = assembly.union(build_piping(cfg))

    # Electrical
    assembly = assembly.union(build_rectifier(cfg))
    assembly = assembly.union(build_control_cabinet(cfg))
    assembly = assembly.union(build_heat_exchanger(cfg))

    # Transport
    assembly = assembly.union(build_forklift_pockets(cfg))
    assembly = assembly.union(build_lifting_lugs(cfg))

    return assembly


def build_dark_mill_assembly(cfg: DarkMillConfig):
    """
    Build as a named CadQuery Assembly (for STEP export with part names).
    """
    assy = cq.Assembly(name="DarkMill")

    # Structural
    assy.add(build_frame(cfg), name="frame", color=cq.Color(0.5, 0.5, 0.5, 1))
    assy.add(build_floor(cfg), name="floor", color=cq.Color(0.3, 0.3, 0.3, 1))
    assy.add(build_roof(cfg), name="roof", color=cq.Color(0.6, 0.6, 0.6, 1))
    assy.add(build_walls(cfg), name="walls", color=cq.Color(0.7, 0.7, 0.7, 0.5))

    # Process (blue/green tones)
    assy.add(build_cell_stacks(cfg), name="cell_stacks", color=cq.Color(0.2, 0.4, 0.8, 1))
    assy.add(build_electrolyte_tanks(cfg), name="tanks", color=cq.Color(0.2, 0.7, 0.3, 1))
    assy.add(build_piping(cfg), name="piping", color=cq.Color(0.8, 0.5, 0.2, 1))

    # Electrical (red/yellow tones)
    assy.add(build_rectifier(cfg), name="rectifier", color=cq.Color(0.8, 0.2, 0.2, 1))
    assy.add(build_control_cabinet(cfg), name="control", color=cq.Color(0.9, 0.7, 0.1, 1))
    assy.add(build_heat_exchanger(cfg), name="heat_exchanger", color=cq.Color(0.3, 0.6, 0.9, 1))

    # Transport
    assy.add(build_forklift_pockets(cfg), name="forklift_pockets", color=cq.Color(0.4, 0.4, 0.4, 1))
    assy.add(build_lifting_lugs(cfg), name="lifting_lugs", color=cq.Color(0.9, 0.5, 0.1, 1))

    # Door (shown open)
    assy.add(build_door(cfg), name="door", color=cq.Color(0.6, 0.6, 0.6, 0.7))

    return assy
