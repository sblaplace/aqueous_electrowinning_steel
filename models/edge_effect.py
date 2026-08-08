"""
Terminal / edge-effect current crowding -> thickness non-uniformity.

Why this module exists
----------------------
``hull_cell.py`` handles a 1-D primary current distribution; ``gas_holdup.py``
handles axial redistribution. Missing is the **terminal (edge) effect**: the
primary current crowds at deposit boundaries, so the foil/flake grows thicker at
its edges. That edge thickening drives **edge cracking on cold rolling** — a
central rollability gate in ``oxygen_in_iron.py`` — and non-uniform composition.

The physics (Round 5, D2): near a deposit edge, the current density rises toward
``j_edge/j_center`` from a few percent to ~2x depending on geometry,
electrode/deposit conductivity, and shield placement. A simple secondary-current
edge correction turns "is the foil rollable?" into a statement about the edges,
not just the center.

This module computes an edge current ratio, a thickness profile, and the edge
O/H loading penalty fed into the rollability gate.

Screening flag
--------------
L1. The edge-ratio correlation is a screening secondary-current proxy;
calibrate against segmented-cathode thickness maps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

SCREENING_FLAG = "unvalidated (L1)"


@dataclass
class EdgeEffectParams:
    """Screening parameters for edge/terminal current crowding."""

    # Electrode half-width (m) and edge "fringe" length scale (m).
    half_width_m: float = 0.10
    edge_fringe_m: float = 0.01
    # Ratio of electrolyte to electrode-deposit conductivity; higher ratio
    # (low deposit conductivity) -> stronger edge crowding.
    conductivity_ratio: float = 2.0
    # Base edge-to-center current ratio at high conductivity ratio.
    edge_ratio_max: float = 1.8
    # Edge O/H penalty: co-deposited O and H scale with local current density
    # to this power (higher j at edge -> more O/H).
    edge_oh_exponent: float = 1.2


def edge_current_ratio(
    params: Optional[EdgeEffectParams] = None,
) -> float:
    """Edge-to-center primary current ratio (screening secondary correction)."""
    p = params or EdgeEffectParams()
    # Secondary-distribution proxy: edge ratio rises with conductivity ratio and
    # with sharpness of the fringe (small fringe -> sharper -> more crowding).
    fringe_factor = p.half_width_m / max(p.edge_fringe_m, 1e-9)
    sharp = min(fringe_factor, 10.0)
    ratio = 1.0 + (p.edge_ratio_max - 1.0) * (p.conductivity_ratio / (1.0 + p.conductivity_ratio)) \
        * (sharp / 10.0)
    return float(max(ratio, 1.0))


def thickness_ratio_across_width(
    n_points: int = 21,
    params: Optional[EdgeEffectParams] = None,
) -> dict:
    """
    Normalized thickness profile across the deposit width (edge = ends).

    Returns
    -------
    dict with x_norm (0=center, 1=edge), thickness_ratio (relative to center),
      and max_min_ratio.
    """
    p = params or EdgeEffectParams()
    ratio = edge_current_ratio(p)
    xs = [i / (n_points - 1) for i in range(n_points)]
    # Thickness scales with local current; approximate with a smooth rise to
    # the edge ratio near the boundary.
    thickness = []
    for x in xs:
        # Sigmoid-like rise within the fringe region near x=1.
        dist_from_edge = (1.0 - x) * p.half_width_m
        frac = math.exp(-dist_from_edge / max(p.edge_fringe_m, 1e-9))
        t = 1.0 + (ratio - 1.0) * frac
        thickness.append(t)
    return {
        "x_norm": xs,
        "thickness_ratio": thickness,
        "max_min_ratio": max(thickness) / min(thickness),
        "edge_to_center_ratio": ratio,
    }


def edge_oh_penalty(
    center_O_ppm: float = 400.0,
    center_H_ppm: float = 1.0,
    params: Optional[EdgeEffectParams] = None,
) -> dict:
    """
    Edge O/H loading penalty (higher local j at the edge).

    Returns
    -------
    dict with edge_ratio, edge_O_ppm, edge_H_ppm, and edge_oh_flag.
    """
    p = params or EdgeEffectParams()
    ratio = edge_current_ratio(p)
    penalty = ratio ** p.edge_oh_exponent
    edge_O = center_O_ppm * penalty
    edge_H = center_H_ppm * penalty
    return {
        "edge_current_ratio": ratio,
        "oh_penalty": float(penalty),
        "edge_O_ppm": float(edge_O),
        "edge_H_ppm": float(edge_H),
        "edge_oh_flag": bool(edge_O > 1000.0),  # above cold-roll O ceiling
    }


def main() -> None:
    """CLI entrypoint for edge-effect analysis."""
    print("=" * 70)
    print(" Terminal / Edge-Effect Current Crowding (Round 5, D2)")
    print("=" * 70)
    print(f" Screening flag : {SCREENING_FLAG}")
    r = edge_current_ratio()
    print(f" Edge/center current ratio : {r:.2f}x")
    prof = thickness_ratio_across_width(n_points=11)
    print(f" Thickness max/min         : {prof['max_min_ratio']:.2f}x")
    oh = edge_oh_penalty(center_O_ppm=400.0)
    print(f" Edge O loading            : {oh['edge_O_ppm']:.0f} ppm "
          f"(flag={oh['edge_oh_flag']})")


if __name__ == "__main__":
    main()
