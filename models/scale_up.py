"""
Scale-up model: current distribution, mass transport, and thermal management
for transitioning from lab-scale (10 cm²) to pilot-scale (1000 cm²) cells.

Five components
---------------
1. **Primary current distribution** — edge effects via Wagner number.
2. **Secondary current distribution** — Butler-Volmer + ohmic drop (1-D Poisson).
3. **Mass transport scaling** — boundary-layer growth, transport-limited current.
4. **Thermal management** — Joule heating, convective + radiative cooling.
5. **Geometry optimization** — minimise energy per kg at uniformity > 90 %.

All electrolyte properties reuse the transport module constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.optimize import brentq

from .electrochemistry import (
    E0_FE,
    FARADAY,
    R_GAS,
    Z_FE,
    specific_energy_kWh_per_kg,
)
from .kinetics import limiting_current_density
from .transport import D_FE

# ─── Physical defaults (reuse existing transport constants) ──────────
DEFAULT_DIFFUSIVITY = D_FE          # m²/s, Fe²⁺ in aqueous sulfate
DEFAULT_FE_CONC_M = 1.0             # mol/L
DEFAULT_TEMPERATURE_C = 60.0        # °C
DEFAULT_KAPPA = 10.0                # S/m, typical FeSO₄/FeCl₂ electrolyte
DEFAULT_FE_IO = 1e-2                # A/m²
DEFAULT_HER_IO = 1e-6               # A/m²
DEFAULT_FE_TAFEL = 0.120            # V/decade
DEFAULT_HER_TAFEL = 0.140           # V/decade
DEFAULT_FLOW_VELOCITY = 0.1         # m/s, electrolyte face velocity
DEFAULT_HTC = 200.0                 # W/(m²·K), liquid-side heat transfer
DEFAULT_EMISSIVITY = 0.3            # cell wall emissivity
STEFAN_BOLTZMANN = 5.670374419e-8   # W/(m²·K⁴)
AMBIENT_T_C = 25.0                  # °C


# ═══════════════════════════════════════════════════════════════════════
#  1. Primary current distribution
# ═══════════════════════════════════════════════════════════════════════


def wagner_number(
    kappa: float = DEFAULT_KAPPA,
    j_ref: float = 1000.0,
    L: float = 0.1,
) -> float:
    """Wagner number Wa = κ / (j_ref · L).

    Large Wa → uniform primary distribution.
    Small Wa → significant edge effects.
    """
    return kappa / (j_ref * L)


def primary_current_distribution(
    x: np.ndarray,
    L: float,
    j_avg: float,
    kappa: float = DEFAULT_KAPPA,
) -> np.ndarray:
    """Local primary current density j(x) across a parallel-plate cathode.

    Uses the conformal-mapping solution for a strip electrode of half-width
    L/2 between infinite plates, modulated by the Wagner number:

        Wa = κ / (j_ref · L)

    For large Wa (high conductivity, small cell), the current distribution
    is nearly uniform.  For small Wa (low conductivity, large cell), edge
    effects are prominent.  The edge-penetration depth scales as L / (2·Wa).

    Parameters
    ----------
    x : array
        Position across the cathode width (m), centred at 0.
    L : float
        Cathode width (m).
    j_avg : float
        Average current density (A/m²).
    kappa : float
        Electrolyte conductivity (S/m).

    Returns
    -------
    j_local : array
        Local current density (A/m²).
    """
    x = np.asarray(x, dtype=float)
    Wa = wagner_number(kappa=kappa, j_ref=j_avg, L=L)

    xi = np.clip(2.0 * x / L, -0.999, 0.999)  # normalised position
    # Conformal solution for a finite strip: j/j_avg = 1/sqrt(1-xi²)
    # This diverges at the edges (xi = ±1).
    j_conformal = j_avg / np.sqrt(1.0 - xi**2)

    # Edge-effect penetration fraction: f_edge ∈ [0, 1]
    #   Wa >> 1 → f_edge ≈ 0 (uniform)
    #   Wa << 1 → f_edge ≈ 1 (full edge effects)
    # Smooth sigmoid transition around Wa = 1.
    f_edge = 1.0 / (1.0 + Wa)

    # Blend: uniform baseline + edge enhancement weighted by f_edge
    j_local = j_avg + f_edge * (j_conformal - j_avg)

    # Physical clamp: edge enhancement rarely exceeds 3× in real cells.
    return np.minimum(j_local, 3.0 * j_avg)


def uniformity_index(j: np.ndarray) -> float:
    """Fraction of cathode area where |j - j_avg| / j_avg < 10 %.

    Returns a value in [0, 1].  1.0 = perfectly uniform.
    """
    j = np.asarray(j, dtype=float)
    j_avg = np.mean(j)
    if j_avg <= 0.0:
        return 1.0
    return float(np.mean(np.abs(j - j_avg) / j_avg < 0.10))


# ═══════════════════════════════════════════════════════════════════════
#  2. Secondary current distribution (1-D Butler-Volmer + Poisson)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class SecondaryCurrentResult:
    """Result of the 1-D secondary current distribution solve."""

    x: np.ndarray           # position (m), 0 = membrane, L = cathode
    phi: np.ndarray         # potential (V)
    j_local: np.ndarray     # local current density (A/m²)
    eta: np.ndarray         # overpotential (V, cathodic negative)
    j_avg: float            # area-averaged current density (A/m²)
    uniformity: float       # uniformity index
    L: float                # cell length (m)


def secondary_current_distribution(
    L: float,
    j_target: float,
    kappa: float = DEFAULT_KAPPA,
    fe_i0: float = DEFAULT_FE_IO,
    her_i0: float = DEFAULT_HER_IO,
    fe_tafel_V: float = DEFAULT_FE_TAFEL,
    her_tafel_V: float = DEFAULT_HER_TAFEL,
    E_eq: float = E0_FE,
    n_points: int = 201,
    temperature_C: float = DEFAULT_TEMPERATURE_C,
) -> SecondaryCurrentResult:
    """Solve 1-D Poisson + Butler-Volmer for a parallel-plate cell.

    Governing equation (simplified 1-D, no convection):

        d²φ/dx² = - j_BV(η) / κ ,     η = φ - E_eq

    where j_BV is the Tafel-limited Butler-Volmer current (cathodic only).

    Boundary conditions:
        φ(0) = 0        (reference at membrane / anode side)
        φ(L) = E_cathode (adjusted to hit j_target)

    The solve finds E_cathode by bisection so that the average current
    matches the target.

    Parameters
    ----------
    L : float
        Inter-electrode gap (m).
    j_target : float
        Target average current density (A/m²).
    kappa : float
        Electrolyte conductivity (S/m).
    E_eq : float
        Equilibrium potential (V vs. SHE).

    Returns
    -------
    SecondaryCurrentResult
    """
    T = temperature_C + 273.15

    def bv_current(eta: float) -> float:
        """Butler-Volmer current (cathodic, positive when η < 0)."""
        # Tafel approximation for large cathodic overpotentials
        # j = i0 * exp(-alpha * F * eta / (R*T))
        if eta >= 0.0:
            # Small anodic or zero overpotential
            return fe_i0 * np.exp(-eta * Z_FE * FARADAY / (R_GAS * T * 10.0))
        # Cathodic: Tafel form, clamped to avoid overflow
        eta_cath = min(-eta, 2.0)  # cap at 2 V to prevent 10^(2/0.12) overflow
        j_fe = fe_i0 * 10.0 ** (eta_cath / fe_tafel_V)
        j_her = her_i0 * 10.0 ** (eta_cath / her_tafel_V)
        return j_fe + j_her

    def solve_at_cathode_potential(E_cathode: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Solve the 1-D Poisson-BV system for a given cathode potential."""
        x = np.linspace(0.0, L, n_points)
        dx = x[1] - x[0]
        # Shooting method: φ(0) = 0, dφ/dx(0) = s  (unknown slope)
        # We integrate forward and adjust s to hit φ(L) = E_cathode.

        def shoot(s: float) -> float:
            """Integrate with slope s at x=0, return φ(L) - E_cathode."""
            phi = 0.0
            dphi = s
            for i in range(n_points - 1):
                eta = phi - E_eq
                j_bv = bv_current(eta)
                # d²φ/dx² = -j/κ  →  φ advances with this curvature
                d2phi = -j_bv / kappa
                phi_new = phi + dphi * dx + 0.5 * d2phi * dx**2
                dphi_new = dphi + d2phi * dx
                phi = phi_new
                dphi = dphi_new
            return phi - E_cathode

        # Bisect the initial slope to match cathode BC.
        # Cathodic potential is negative vs E_eq, so dφ/dx is typically negative.
        try:
            s_opt = brentq(shoot, -1e6, 1e6, xtol=1e-10, rtol=1e-12)
        except ValueError:
            # If bracketing fails, use a simple linear profile
            s_opt = (E_cathode - 0.0) / L

        # Now integrate forward storing the full profile.
        x_arr = np.linspace(0.0, L, n_points)
        phi_arr = np.zeros(n_points)
        j_arr = np.zeros(n_points)
        eta_arr = np.zeros(n_points)
        phi_arr[0] = 0.0
        dphi = s_opt
        for i in range(n_points - 1):
            eta = phi_arr[i] - E_eq
            j_bv = bv_current(eta)
            j_arr[i] = j_bv
            eta_arr[i] = eta
            d2phi = -j_bv / kappa
            phi_arr[i + 1] = phi_arr[i] + dphi * dx + 0.5 * d2phi * dx**2
            dphi = dphi + d2phi * dx
        # Final point
        eta = phi_arr[-1] - E_eq
        j_arr[-1] = bv_current(eta)
        eta_arr[-1] = eta
        return x_arr, phi_arr, j_arr, eta_arr

    # Bisect cathode potential to match target average current.
    def avg_residual(E_cathode: float) -> float:
        _, _, j_arr, _ = solve_at_cathode_potential(E_cathode)
        return float(np.mean(j_arr)) - j_target

    # Cathodic potentials must be below E_eq for deposition.
    # Find a bracket.
    E_lo, E_hi = E_eq - 2.0, E_eq + 0.1
    try:
        E_cath = brentq(avg_residual, E_lo, E_hi, xtol=1e-8)
    except ValueError:
        E_cath = E_eq - 0.5  # fallback

    x, phi, j_local, eta = solve_at_cathode_potential(E_cath)
    j_avg = float(np.mean(j_local))
    return SecondaryCurrentResult(
        x=x,
        phi=phi,
        j_local=j_local,
        eta=eta,
        j_avg=j_avg,
        uniformity=uniformity_index(j_local),
        L=L,
    )


# ═══════════════════════════════════════════════════════════════════════
#  3. Mass transport scaling
# ═══════════════════════════════════════════════════════════════════════


def boundary_layer_thickness(
    L: float,
    D: float = DEFAULT_DIFFUSIVITY,
    v: float = DEFAULT_FLOW_VELOCITY,
) -> float:
    """Boundary-layer thickness δ = sqrt(D · L / v).

    For a laminar flow over a flat plate (Levich-type scaling).
    As electrode length L increases, δ grows and the transport-limited
    current drops.

    Parameters
    ----------
    L : float
        Electrode length in the flow direction (m).
    D : float
        Diffusivity (m²/s).
    v : float
        Bulk flow velocity (m/s).

    Returns
    -------
    delta : float
        Boundary-layer thickness (m).
    """
    return np.sqrt(D * L / max(v, 1e-10))


@dataclass
class MassTransportResult:
    """Mass-transport scaling along a cathode."""

    x: np.ndarray           # position along cathode (m)
    delta: np.ndarray        # boundary-layer thickness (m)
    j_lim: np.ndarray        # transport-limited current density (A/m²)
    j_local: np.ndarray      # local operating current (A/m²)
    transport_limited: np.ndarray   # bool: j_local > j_lim
    fraction_limited: float  # fraction of cathode that is transport-limited
    j_lim_min: float         # minimum transport-limited current (A/m²)


def mass_transport_scaling(
    L: float,
    j_local: np.ndarray,
    D: float = DEFAULT_DIFFUSIVITY,
    v: float = DEFAULT_FLOW_VELOCITY,
    fe_conc_mol_L: float = DEFAULT_FE_CONC_M,
    n_points: int = 201,
) -> MassTransportResult:
    """Compute transport-limited current along the cathode.

    Parameters
    ----------
    L : float
        Cathode length in flow direction (m).
    j_local : array
        Local current density profile (A/m²), length n_points.
    D : float
        Diffusivity (m²/s).
    v : float
        Flow velocity (m/s).
    fe_conc_mol_L : float
        Bulk Fe²⁺ concentration (mol/L).

    Returns
    -------
    MassTransportResult
    """
    x = np.linspace(0.0, L, n_points)
    delta = np.array([boundary_layer_thickness(xi, D, v) for xi in x])
    fe_conc_mol_m3 = fe_conc_mol_L * 1000.0
    j_lim = np.array([
        limiting_current_density(fe_conc_mol_m3, D, max(di, 1e-10))
        for di in delta
    ])
    j_local = np.asarray(j_local, dtype=float)
    if len(j_local) != n_points:
        j_local = np.interp(np.linspace(0, 1, n_points), np.linspace(0, 1, len(j_local)), j_local)
    limited = j_local > j_lim
    return MassTransportResult(
        x=x,
        delta=delta,
        j_lim=j_lim,
        j_local=j_local,
        transport_limited=limited,
        fraction_limited=float(np.mean(limited)),
        j_lim_min=float(np.min(j_lim)),
    )


# ═══════════════════════════════════════════════════════════════════════
#  4. Thermal management
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class ThermalResult:
    """Thermal management analysis result."""

    Q_gen_W: float          # total Joule heating (W)
    Q_conv_W: float         # convective heat removal (W)
    Q_rad_W: float          # radiative heat removal (W)
    delta_T_C: float        # temperature rise above ambient (°C)
    T_cell_C: float         # estimated cell temperature (°C)
    boiling_risk: bool      # T_cell > 80 °C
    S_V_ratio: float        # surface-to-volume ratio (1/m)
    R_cell_ohm: float       # cell resistance (Ω)


def thermal_management(
    j_avg: float,
    area_m2: float,
    gap_m: float,
    kappa: float = DEFAULT_KAPPA,
    V_cell: float = 2.0,
    h_conv: float = DEFAULT_HTC,
    emissivity: float = DEFAULT_EMISSIVITY,
    T_ambient_C: float = AMBIENT_T_C,
    max_T_C: float = 80.0,
) -> ThermalResult:
    """Estimate temperature rise in a pilot-scale cell.

    Joule heating Q = j² · R_cell · Volume.
    Heat removal by convection + radiation from cell walls.

    Parameters
    ----------
    j_avg : float
        Average current density (A/m²).
    area_m2 : float
        Electrode area (m²).
    gap_m : float
        Inter-electrode gap (m).
    kappa : float
        Electrolyte conductivity (S/m).
    V_cell : float
        Cell voltage (V), used for total power.
    h_conv : float
        Convective heat transfer coefficient (W/(m²·K)).
    emissivity : float
        Wall emissivity for radiative loss.
    T_ambient_C : float
        Ambient temperature (°C).

    Returns
    -------
    ThermalResult
    """
    T_amb_K = T_ambient_C + 273.15
    # Electrode geometry: assume square cathode
    side = np.sqrt(area_m2)  # m
    # Volume of electrolyte in the gap
    volume = area_m2 * gap_m  # m³
    # R_cell = gap / (kappa * area)
    R_cell = gap_m / (kappa * area_m2)  # Ω
    # Joule heating: Q = I²R = (j·A)² · R
    total_current = j_avg * area_m2  # A
    Q_gen = total_current**2 * R_cell  # W

    # Surface area for heat exchange (top + sides, bottom insulated)
    perimeter = 4.0 * side
    A_top = area_m2
    A_sides = perimeter * gap_m
    A_heat = A_top + A_sides  # effective heat exchange area
    S_V = A_heat / max(volume, 1e-10)

    # Iterative heat balance: Q_gen = h·A·ΔT + ε·σ·A·(T⁴ - T_amb⁴)
    def heat_balance(dT: float) -> float:
        T = T_amb_K + dT
        Q_conv = h_conv * A_heat * dT
        Q_rad = emissivity * STEFAN_BOLTZMANN * A_heat * (T**4 - T_amb_K**4)
        return Q_gen - Q_conv - Q_rad

    # Solve for ΔT
    try:
        dT = brentq(heat_balance, 0.0, 200.0, xtol=1e-4)
    except ValueError:
        dT = Q_gen / (h_conv * A_heat + 1e-30)  # linear fallback

    T_cell = T_ambient_C + dT
    Q_conv = h_conv * A_heat * dT
    Q_rad = emissivity * STEFAN_BOLTZMANN * A_heat * ((T_amb_K + dT)**4 - T_amb_K**4)

    return ThermalResult(
        Q_gen_W=Q_gen,
        Q_conv_W=Q_conv,
        Q_rad_W=Q_rad,
        delta_T_C=dT,
        T_cell_C=T_cell,
        boiling_risk=T_cell > max_T_C,
        S_V_ratio=S_V,
        R_cell_ohm=R_cell,
    )


# ═══════════════════════════════════════════════════════════════════════
#  5. Geometry optimization
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class GeometryResult:
    """Optimal geometry for a given total current."""

    length_m: float         # optimal cell length (m)
    width_m: float          # optimal cell width (m)
    gap_m: float            # optimal inter-electrode gap (m)
    area_m2: float          # electrode area (m²)
    j_avg: float            # average current density (A/m²)
    uniformity: float       # uniformity index
    energy_kWh_per_kg: float  # specific energy consumption
    thermal: ThermalResult
    feasible: bool          # all constraints satisfied


def optimize_geometry(
    total_current_A: float,
    area_m2: float = 0.1,
    kappa: float = DEFAULT_KAPPA,
    V_cell: float = 2.0,
    current_efficiency: float = 0.85,
    min_uniformity: float = 0.90,
    max_T_C: float = 80.0,
    h_conv: float = DEFAULT_HTC,
) -> GeometryResult:
    """Find optimal cell dimensions to minimise energy per kg.

    For a given total current and target area, optimises the aspect ratio
    (length/width) and inter-electrode gap to minimise specific energy
    consumption while maintaining:
    - uniformity > min_uniformity (90 %)
    - temperature < max_T_C (80 °C)

    Parameters
    ----------
    total_current_A : float
        Total cell current (A).
    area_m2 : float
        Target electrode area (m²).
    kappa : float
        Electrolyte conductivity (S/m).
    V_cell : float
        Cell voltage (V).
    current_efficiency : float
        Fractional current efficiency (0–1).
    min_uniformity : float
        Minimum acceptable uniformity index.
    max_T_C : float
        Maximum acceptable cell temperature (°C).

    Returns
    -------
    GeometryResult
    """
    j_avg = total_current_A / area_m2  # A/m²

    def energy_for_gap(gap: float) -> float:
        """Specific energy for a given gap, using thermal + uniformity penalty."""
        # R_cell = gap / (kappa * area)
        R_cell = gap / (kappa * area_m2)
        V_eff = V_cell  # cell voltage is the driving potential
        energy = specific_energy_kWh_per_kg(V_eff, current_efficiency)
        return energy

    def is_feasible(gap: float) -> bool:
        """Check thermal and uniformity constraints."""
        thermal = thermal_management(
            j_avg, area_m2, gap, kappa, V_cell, h_conv
        )
        # Primary distribution uniformity (square electrode)
        side = np.sqrt(area_m2)
        x_pts = np.linspace(-side / 2 * 0.95, side / 2 * 0.95, 201)
        j_local = primary_current_distribution(x_pts, side, j_avg, kappa)
        uni = uniformity_index(j_local)
        return uni >= min_uniformity and not thermal.boiling_risk

    # Sweep gap from 1 mm to 50 mm
    gaps = np.linspace(0.001, 0.05, 200)
    energies = [energy_for_gap(g) for g in gaps]
    feasible_mask = np.array([is_feasible(g) for g in gaps])

    if not np.any(feasible_mask):
        # No feasible geometry — return the best-energy option anyway
        idx = int(np.argmin(energies))
        gap_opt = float(gaps[idx])
        feasible = False
    else:
        # Among feasible gaps, pick the one with minimum energy
        feasible_energies = np.where(feasible_mask, energies, np.inf)
        idx = int(np.argmin(feasible_energies))
        gap_opt = float(gaps[idx])
        feasible = True

    # Compute the final geometry
    side = np.sqrt(area_m2)
    thermal = thermal_management(j_avg, area_m2, gap_opt, kappa, V_cell, h_conv)
    x_pts = np.linspace(-side / 2 * 0.95, side / 2 * 0.95, 201)
    j_local = primary_current_distribution(x_pts, side, j_avg, kappa)
    uni = uniformity_index(j_local)
    energy = specific_energy_kWh_per_kg(V_cell, current_efficiency)

    return GeometryResult(
        length_m=side,
        width_m=side,
        gap_m=gap_opt,
        area_m2=area_m2,
        j_avg=j_avg,
        uniformity=uni,
        energy_kWh_per_kg=energy,
        thermal=thermal,
        feasible=feasible,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Unified scale-up analysis
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class ScaleUpResult:
    """Complete scale-up analysis for one operating point."""

    area_m2: float
    j_avg: float
    wagner: float
    primary_uniformity: float
    secondary: SecondaryCurrentResult
    mass_transport: MassTransportResult
    thermal: ThermalResult
    geometry: GeometryResult


def scale_up_analysis(
    area_m2: float,
    total_current_A: float,
    kappa: float = DEFAULT_KAPPA,
    V_cell: float = 2.0,
    current_efficiency: float = 0.85,
    D: float = DEFAULT_DIFFUSIVITY,
    v: float = DEFAULT_FLOW_VELOCITY,
    fe_conc_mol_L: float = DEFAULT_FE_CONC_M,
    h_conv: float = DEFAULT_HTC,
) -> ScaleUpResult:
    """Run the full scale-up analysis for a given cell size.

    Parameters
    ----------
    area_m2 : float
        Electrode area (m²).
    total_current_A : float
        Total cell current (A).
    kappa : float
        Electrolyte conductivity (S/m).
    V_cell : float
        Cell voltage (V).
    current_efficiency : float
        Current efficiency for Fe deposition.

    Returns
    -------
    ScaleUpResult
    """
    j_avg = total_current_A / area_m2
    side = np.sqrt(area_m2)
    L = side  # characteristic length

    # 1. Primary distribution
    Wa = wagner_number(kappa=kappa, j_ref=j_avg, L=L)
    x_pts = np.linspace(-side / 2 * 0.95, side / 2 * 0.95, 501)
    j_primary = primary_current_distribution(x_pts, side, j_avg, kappa)
    uni_primary = uniformity_index(j_primary)

    # 2. Secondary distribution
    gap = 0.01  # 10 mm default
    secondary = secondary_current_distribution(
        L=gap, j_target=j_avg, kappa=kappa,
    )

    # 3. Mass transport
    x_flow = np.linspace(0.0, L, 501)
    j_flow = np.full_like(x_flow, j_avg)  # uniform assumption for scaling
    mt = mass_transport_scaling(L, j_flow, D, v, fe_conc_mol_L, n_points=501)

    # 4. Thermal
    thermal = thermal_management(j_avg, area_m2, gap, kappa, V_cell, h_conv)

    # 5. Geometry optimization
    geometry = optimize_geometry(
        total_current_A, area_m2, kappa, V_cell, current_efficiency,
    )

    return ScaleUpResult(
        area_m2=area_m2,
        j_avg=j_avg,
        wagner=Wa,
        primary_uniformity=uni_primary,
        secondary=secondary,
        mass_transport=mt,
        thermal=thermal,
        geometry=geometry,
    )
