"""
Manifold ionic shunt currents (leakage currents) in multi-cell series stacks.

Physics and Engineering
-----------------------
In modular industrial electrowinning units and crate architectures
(:mod:`models.crate`, :mod:`models.cell_architecture`), N cells (e.g. N = 20 to 60)
are connected electrically in series while sharing common electrolyte supply and
return fluid manifolds.  Because the total stack voltage is large
(V_stack = N · V_cell ≈ 50–180 V) and the concentrated ferrous electrolyte is highly
conductive (κ ≈ 10–25 S/m), **parasitic ionic shunt currents** flow through the
fluid piping headers.

This discrete ladder network (Kaminski–Gileadi & Waha formalism) models:

1. **Stack resistor ladder network**:
   - Manifold channel resistance between cells: R_m = L_m / (κ · A_m)
   - Port connection resistance from cell to manifold: R_p = L_p / (κ · A_p)
   - Cell internal resistance: R_cell = V_cell / I_cell

2. **Faradaic bypass loss**:
   - Current entering port channels bypasses active cathode/anode surfaces.
   - Stack current efficiency loss: ΔFE_shunt = I_shunt,total / (N · I_applied)

3. **Accelerated port corrosion**:
   - At the extreme high-voltage end cells (cells 1 and N), large shunt currents
     exit the electrolyte port into metal piping nozzles or instrumentation ports.
   - High exit current densities (j_port > 50 mA/cm²) cause severe electrolytic
     pitting and rapid manifold tube perforation.

4. **Cell-by-cell current maldistribution**:
   - Center cells operate at nominal current, while end cells experience depressed
     Faradaic current, leading to non-uniform iron foil thickness across the stack.

References
----------
* Kaminski, P. C., & Gileadi, E. (1984). "Shunt currents in electrochemical
  reactors." J. Electrochem. Soc., 131(8), 1804–1809.
* Waha, A., & Euler, K. J. (1981). "Current distribution and shunt currents in
  bipolar and series battery stacks." Electrochim. Acta, 26(10), 1435–1442.
* White, R. E., & Beck, T. R. (1986). "Analysis of shunt currents in a
  chlor-alkali membrane cell stack." J. Electrochem. Soc., 133(12), 2530–2538.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class StackManifoldGeometry:
    """Dimensions and fluid header geometry for a multi-cell stack."""

    n_cells: int = 50                 # Number of cells in series
    cell_voltage_V: float = 2.60      # Operating voltage per cell (V)
    applied_current_A: float = 300.0  # Total rectifier current (A)
    electrolyte_conductivity_S_m: float = 18.0  # Bulk conductivity (S/m)

    # Manifold header geometry
    manifold_diameter_m: float = 0.075  # Main pipe inner diameter (m)
    manifold_pitch_m: float = 0.050     # Spacing between cell inlets (m)

    # Port channel geometry (cell feed nozzle)
    port_inner_diameter_m: float = 0.015 # Port tube diameter (m)
    port_length_m: float = 0.120         # Port tube length (m)

    @property
    def manifold_area_m2(self) -> float:
        """Cross-sectional area of main header (m²)."""
        return 0.25 * math.pi * (self.manifold_diameter_m ** 2)

    @property
    def port_area_m2(self) -> float:
        """Cross-sectional area of single port nozzle (m²)."""
        return 0.25 * math.pi * (self.port_inner_diameter_m ** 2)

    @property
    def r_manifold_ohm(self) -> float:
        """Resistance of manifold segment between two adjacent cells (Ω)."""
        return self.manifold_pitch_m / (self.electrolyte_conductivity_S_m * self.manifold_area_m2)

    @property
    def r_port_ohm(self) -> float:
        """Resistance of single inlet or outlet port tube (Ω)."""
        return self.port_length_m / (self.electrolyte_conductivity_S_m * self.port_area_m2)


@dataclass
class ShuntCurrentResult:
    """Calculated shunt current distribution across the stack."""

    n_cells: int
    stack_voltage_V: float
    total_shunt_current_A: float
    max_single_port_current_A: float
    max_port_current_density_mA_cm2: float
    stack_faradaic_efficiency_loss_percent: float
    stack_ohmic_shunt_power_W: float
    is_corrosion_threat: bool          # True if max port current density > 25 mA/cm²
    recommended_min_port_length_m: float
    cell_faradaic_currents_A: List[float]
    port_currents_A: List[float]


def solve_stack_shunt_currents(
    geometry: Optional[StackManifoldGeometry] = None,
) -> ShuntCurrentResult:
    """
    Solve the Kaminski–Gileadi tridiagonal resistor network for stack shunt currents.

    Parameters
    ----------
    geometry : StackManifoldGeometry, optional
        Stack dimensions and operating point.

    Returns
    -------
    ShuntCurrentResult
        Detailed current distribution and corrosion risk metrics.
    """
    if geometry is None:
        geometry = StackManifoldGeometry()

    N = max(int(geometry.n_cells), 2)
    v_cell = geometry.cell_voltage_V
    i_app = geometry.applied_current_A
    r_m = geometry.r_manifold_ohm
    r_p = geometry.r_port_ohm

    # Cell potentials (cumulative series potential): V_k = (k - 0.5) * v_cell
    # for k = 1..N
    v_cells = np.array([(k - 0.5) * v_cell for k in range(1, N + 1)])

    # Construct tridiagonal system for manifold node potentials phi_m[1..N]:
    # At internal nodes k:
    # (phi_m[k] - phi_m[k-1])/r_m + (phi_m[k] - phi_m[k+1])/r_m + (phi_m[k] - v_cells[k])/r_p = 0
    # => - (1/r_m) phi_m[k-1] + (2/r_m + 1/r_p) phi_m[k] - (1/r_m) phi_m[k+1] = v_cells[k] / r_p

    A = np.zeros((N, N))
    b = np.zeros(N)

    g_m = 1.0 / max(r_m, 1e-9)
    g_p = 1.0 / max(r_p, 1e-9)

    for k in range(N):
        b[k] = v_cells[k] * g_p
        if k == 0:
            # End node: no connection to left
            A[k, k] = g_m + g_p
            A[k, k + 1] = -g_m
        elif k == N - 1:
            # End node: no connection to right
            A[k, k - 1] = -g_m
            A[k, k] = g_m + g_p
        else:
            A[k, k - 1] = -g_m
            A[k, k] = 2.0 * g_m + g_p
            A[k, k + 1] = -g_m

    phi_m = np.linalg.solve(A, b)

    # Shunt currents through ports: I_port[k] = (v_cells[k] - phi_m[k]) / r_p
    i_ports = (v_cells - phi_m) * g_p  # positive = flowing out of cell into manifold

    # Net cell current (Faradaic current passing through active electrode)
    # The cell at position k loses i_ports[k]
    i_faradaic = i_app - np.abs(i_ports)

    total_shunt_current = float(np.sum(np.abs(i_ports)))
    max_port_current = float(np.max(np.abs(i_ports)))

    # Port current density (mA/cm²)
    port_area_cm2 = geometry.port_area_m2 * 1e4
    max_j_port = (max_port_current / max(port_area_cm2, 1e-4)) * 1e3

    fe_loss_pct = (total_shunt_current / max(N * i_app, 1e-6)) * 100.0

    # Ohmic power dissipated in shunt paths (W): sum(I_port^2 * R_p) + sum(I_m^2 * R_m)
    i_manifold = np.zeros(N - 1)
    for k in range(N - 1):
        i_manifold[k] = (phi_m[k] - phi_m[k + 1]) * g_m
    power_shunt = float(np.sum(i_ports ** 2) * r_p + np.sum(i_manifold ** 2) * r_m)

    # Corrosion threshold: > 25 mA/cm² causes accelerated nozzle electrolytic pitting
    is_corrosion = max_j_port > 25.0

    # Calculate recommended port length to drop max port current density below 25 mA/cm²
    # Port resistance scales linearly with length (R_p = L_p / (kappa * A_p))
    if is_corrosion:
        recommended_l_p = geometry.port_length_m * (max_j_port / 25.0)
    else:
        recommended_l_p = geometry.port_length_m

    return ShuntCurrentResult(
        n_cells=N,
        stack_voltage_V=float(N * v_cell),
        total_shunt_current_A=total_shunt_current,
        max_single_port_current_A=max_port_current,
        max_port_current_density_mA_cm2=max_j_port,
        stack_faradaic_efficiency_loss_percent=fe_loss_pct,
        stack_ohmic_shunt_power_W=power_shunt,
        is_corrosion_threat=is_corrosion,
        recommended_min_port_length_m=recommended_l_p,
        cell_faradaic_currents_A=i_faradaic.tolist(),
        port_currents_A=i_ports.tolist(),
    )
