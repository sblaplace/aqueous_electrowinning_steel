"""Transient pulse and pulse-reverse electrodeposition modeling.

Models time-dependent concentration profiles, surface Fe2+ depletion, local pH
relaxation, and faradaic efficiency under pulsed and pulse-reverse current waveforms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple
import numpy as np

from .electrochemistry import FARADAY, M_FE, RHO_FE

# Standard diffusivities (m^2/s)
DIFFUSIVITY_FE2 = 7.2e-10   # Fe²⁺ infinite-dilution, 25°C (CRC); consistent with transport.py D_FE
DIFFUSIVITY_H = 9.31e-9


@dataclass(frozen=True)
class PulseWaveform:
    """Current density waveform for pulsed / pulse-reverse electrodeposition.

    Sign convention:
      - Cathodic current (iron deposition / HER): positive (mA/cm²).
      - Anodic current (reverse pulse / dissolution): negative (mA/cm²).
      - Off period: current density = 0.
    """

    j_cathodic_mA_cm2: float
    t_cathodic_s: float
    j_anodic_mA_cm2: float = 0.0
    t_anodic_s: float = 0.0
    t_off_s: float = 0.0

    def __post_init__(self) -> None:
        if self.j_cathodic_mA_cm2 <= 0.0:
            raise ValueError("j_cathodic_mA_cm2 must be positive")
        if self.t_cathodic_s <= 0.0:
            raise ValueError("t_cathodic_s must be positive")
        if self.j_anodic_mA_cm2 > 0.0:
            raise ValueError("j_anodic_mA_cm2 must be non-positive (0 for off-period, <0 for reverse)")
        if self.t_anodic_s < 0.0:
            raise ValueError("t_anodic_s must be non-negative")
        if self.t_off_s < 0.0:
            raise ValueError("t_off_s must be non-negative")
        if self.t_cycle_s <= 0.0:
            raise ValueError("Total cycle duration must be positive")

    @property
    def t_cycle_s(self) -> float:
        """Total duration of one complete pulse cycle (s)."""
        return self.t_cathodic_s + self.t_anodic_s + self.t_off_s

    @property
    def frequency_Hz(self) -> float:
        """Pulse repetition frequency (Hz)."""
        return 1.0 / self.t_cycle_s

    @property
    def duty_cycle(self) -> float:
        """Cathodic duty cycle fraction (t_cathodic / t_cycle)."""
        return self.t_cathodic_s / self.t_cycle_s

    @property
    def j_avg_mA_cm2(self) -> float:
        """Cycle-averaged net current density (mA/cm²)."""
        q_net = (
            self.j_cathodic_mA_cm2 * self.t_cathodic_s
            + self.j_anodic_mA_cm2 * self.t_anodic_s
        )
        return q_net / self.t_cycle_s

    def evaluate_current_A_m2(self, t_s: float) -> float:
        """Return applied current density in A/m² at time t_s within cycle."""
        tau = t_s % self.t_cycle_s
        if tau < self.t_cathodic_s:
            j_mA = self.j_cathodic_mA_cm2
        elif tau < (self.t_cathodic_s + self.t_anodic_s):
            j_mA = self.j_anodic_mA_cm2
        else:
            j_mA = 0.0
        return j_mA * 10.0  # convert mA/cm² to A/m²


@dataclass
class PulseResult:
    """Simulation output from a pulse-reverse electrodeposition run."""

    time_s: np.ndarray
    applied_current_A_m2: np.ndarray
    surface_fe_M: np.ndarray
    surface_pH: np.ndarray
    fe_current_A_m2: np.ndarray
    her_current_A_m2: np.ndarray
    instant_efficiency: np.ndarray
    cycle_avg_efficiency: float
    net_fe_deposited_g_m2: float
    plating_rate_um_hr: float
    peak_surface_depletion_ratio: float
    max_surface_pH: float
    waveform: PulseWaveform

    def summary(self) -> Dict[str, Any]:
        """Return unit-labelled summary dictionary."""
        return {
            "Frequency (Hz)": self.waveform.frequency_Hz,
            "Duty cycle (%)": self.waveform.duty_cycle * 100.0,
            "Average current density (mA/cm²)": self.waveform.j_avg_mA_cm2,
            "Cycle-averaged current efficiency (%)": self.cycle_avg_efficiency * 100.0,
            "Net plating rate (µm/hr)": self.plating_rate_um_hr,
            "Net Fe deposited (g/m²)": self.net_fe_deposited_g_m2,
            "Min surface Fe²⁺ ratio (C_surf / C_bulk)": self.peak_surface_depletion_ratio,
            "Max surface pH": self.max_surface_pH,
        }


class PulseDepositionModel:
    """Transient 1D diffusion-kinetics model for pulsed and pulse-reverse electrowinning."""

    def __init__(
        self,
        boundary_layer_m: float = 1.0e-4,
        fe_bulk_M: float = 1.0,
        bulk_pH: float = 2.0,
        diffusivity_fe_m2_s: float = DIFFUSIVITY_FE2,
        diffusivity_h_m2_s: float = DIFFUSIVITY_H,
        her_i0_A_m2: float = 1.0e-4,
        fe_i0_A_m2: float = 10.0,
        grid_points: int = 51,
    ) -> None:
        if boundary_layer_m <= 0.0:
            raise ValueError("boundary_layer_m must be positive")
        if fe_bulk_M <= 0.0:
            raise ValueError("fe_bulk_M must be positive")
        if bulk_pH < 0.0 or bulk_pH > 14.0:
            raise ValueError("bulk_pH must be between 0 and 14")
        if grid_points < 5:
            raise ValueError("grid_points must be at least 5")
        if diffusivity_fe_m2_s <= 0.0 or diffusivity_h_m2_s <= 0.0:
            raise ValueError("Diffusivities must be positive")

        self.boundary_layer_m = boundary_layer_m
        self.fe_bulk_M = fe_bulk_M
        self.bulk_pH = bulk_pH
        self.diffusivity_fe = diffusivity_fe_m2_s
        self.diffusivity_h = diffusivity_h_m2_s
        self.her_i0 = her_i0_A_m2
        self.fe_i0 = fe_i0_A_m2
        self.grid_points = grid_points

        # Spatial grid (0 = cathode surface, boundary_layer_m = bulk)
        self.x_m = np.linspace(0.0, boundary_layer_m, grid_points)
        self.dx = boundary_layer_m / (grid_points - 1)

    def simulate(
        self,
        waveform: PulseWaveform,
        n_cycles: int = 10,
        steps_per_cycle: int = 100,
    ) -> PulseResult:
        """Simulate transient pulse deposition over n_cycles."""
        if n_cycles <= 0:
            raise ValueError("n_cycles must be positive")
        if steps_per_cycle < 10:
            raise ValueError("steps_per_cycle must be at least 10")

        dt = waveform.t_cycle_s / steps_per_cycle
        n_steps = n_cycles * steps_per_cycle
        time_arr = np.linspace(0.0, n_cycles * waveform.t_cycle_s, n_steps + 1)

        # Initial concentration profiles (mol/m^3)
        c_fe_bulk = self.fe_bulk_M * 1000.0
        c_h_bulk = (10.0 ** (-self.bulk_pH)) * 1000.0

        c_fe = np.full(self.grid_points, c_fe_bulk, dtype=float)
        c_h = np.full(self.grid_points, c_h_bulk, dtype=float)

        # Pre-allocate output arrays
        applied_j = np.zeros(n_steps + 1)
        surf_fe = np.zeros(n_steps + 1)
        surf_ph = np.zeros(n_steps + 1)
        j_fe = np.zeros(n_steps + 1)
        j_her = np.zeros(n_steps + 1)
        inst_eff = np.zeros(n_steps + 1)

        # Record initial state
        applied_j[0] = waveform.evaluate_current_A_m2(0.0)
        surf_fe[0] = c_fe[0] / 1000.0
        surf_ph[0] = -np.log10(max(c_h[0] / 1000.0, 1e-14))

        # Build Crank-Nicolson tridiagonal matrices for inner nodes
        A_fe, B_fe = self._build_cn_matrices(self.diffusivity_fe, dt)
        A_h, B_h = self._build_cn_matrices(self.diffusivity_h, dt)

        for step in range(n_steps):
            t_current = time_arr[step]
            j_app = waveform.evaluate_current_A_m2(t_current)
            applied_j[step] = j_app

            # Calculate kinetic split & fluxes based on surface state
            c_fe_surf = max(c_fe[0], 1e-6)
            c_h_surf = max(c_h[0], 1e-12)

            i_fe_step, i_her_step, eff_step = self._kinetic_split(j_app, c_fe_surf, c_h_surf)
            j_fe[step] = i_fe_step
            j_her[step] = i_her_step
            inst_eff[step] = eff_step

            # Surface fluxes (mol / m^2 s)
            # Positive current -> cathodic (depletion of Fe2+, consumption of H+)
            # Negative current -> anodic (dissolution of Fe, no HER)
            flux_fe = -i_fe_step / (2.0 * FARADAY)
            flux_h = -i_her_step / FARADAY

            # Advance 1D profiles using Crank-Nicolson step
            c_fe = self._step_cn(c_fe, c_fe_bulk, A_fe, B_fe, self.diffusivity_fe, dt, flux_fe)
            c_h = self._step_cn(c_h, c_h_bulk, A_h, B_h, self.diffusivity_h, dt, flux_h)

            surf_fe[step + 1] = max(c_fe[0], 0.0) / 1000.0
            surf_ph[step + 1] = -np.log10(max(c_h[0] / 1000.0, 1e-14))

        # Final step recording
        j_last = waveform.evaluate_current_A_m2(time_arr[-1])
        applied_j[-1] = j_last
        i_fe_last, i_her_last, eff_last = self._kinetic_split(j_last, c_fe[0], c_h[0])
        j_fe[-1] = i_fe_last
        j_her[-1] = i_her_last
        inst_eff[-1] = eff_last

        # Calculate integrated net iron deposited (g/m^2)
        # trapezoidal integration of j_fe over time
        dt_arr = np.diff(time_arr)
        q_fe_net = np.sum(0.5 * (j_fe[:-1] + j_fe[1:]) * dt_arr)  # Coulombs/m^2
        net_fe_kg_m2 = (q_fe_net / (2.0 * FARADAY)) * M_FE
        net_fe_g_m2 = net_fe_kg_m2 * 1000.0

        q_total_cathodic = np.sum(0.5 * (np.maximum(applied_j[:-1], 0) + np.maximum(applied_j[1:], 0)) * dt_arr)
        if q_total_cathodic > 0:
            cycle_avg_eff = max(q_fe_net / q_total_cathodic, 0.0)
        else:
            cycle_avg_eff = 0.0

        # Net plating rate in um/hr
        total_time_hr = time_arr[-1] / 3600.0
        if total_time_hr > 0:
            thickness_m = net_fe_kg_m2 / RHO_FE  # kg/m^2 / (kg/m^3) = m
            plating_rate_um_hr = (thickness_m * 1.0e6) / total_time_hr
        else:
            plating_rate_um_hr = 0.0

        peak_depletion = float(np.min(surf_fe) / self.fe_bulk_M)
        max_ph = float(np.max(surf_ph))

        return PulseResult(
            time_s=time_arr,
            applied_current_A_m2=applied_j,
            surface_fe_M=surf_fe,
            surface_pH=surf_ph,
            fe_current_A_m2=j_fe,
            her_current_A_m2=j_her,
            instant_efficiency=inst_eff,
            cycle_avg_efficiency=float(cycle_avg_eff),
            net_fe_deposited_g_m2=float(net_fe_g_m2),
            plating_rate_um_hr=float(plating_rate_um_hr),
            peak_surface_depletion_ratio=peak_depletion,
            max_surface_pH=max_ph,
            waveform=waveform,
        )

    def _kinetic_split(self, j_app: float, c_fe_surf_mol_m3: float, c_h_surf_mol_m3: float) -> Tuple[float, float, float]:
        """Split applied current into Fe and HER partial currents and instant efficiency."""
        if j_app < 0.0:
            # Anodic reverse pulse: iron dissolves back, no HER
            return j_app, 0.0, 0.0
        if j_app == 0.0:
            return 0.0, 0.0, 0.0

        # Mass-transport limited iron current density (A/m^2)
        i_lim_fe = 2.0 * FARADAY * self.diffusivity_fe * max(c_fe_surf_mol_m3, 0.0) / self.dx

        # Kinetic preference ratio based on exchange currents & surface concentrations
        fe_conc_ratio = c_fe_surf_mol_m3 / (self.fe_bulk_M * 1000.0)
        h_conc_M = c_h_surf_mol_m3 / 1000.0

        # Butler-Volmer competitive activation factor
        raw_fe_ratio = (self.fe_i0 * fe_conc_ratio) / (self.fe_i0 * fe_conc_ratio + self.her_i0 * (h_conc_M / 10.0**-self.bulk_pH))
        raw_fe_ratio = np.clip(raw_fe_ratio, 0.01, 0.995)

        i_fe_kinetic = j_app * raw_fe_ratio
        i_fe = min(i_fe_kinetic, i_lim_fe)
        i_her = j_app - i_fe
        eff = i_fe / j_app if j_app > 0 else 0.0

        return i_fe, i_her, eff

    def _build_cn_matrices(self, diffusivity: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Construct Crank-Nicolson system matrices A and B for internal nodes."""
        r = diffusivity * dt / (2.0 * self.dx**2)
        n = self.grid_points

        A = np.zeros((n, n), dtype=float)
        B = np.zeros((n, n), dtype=float)

        A[0, 0] = 1.0 + 2.0 * r
        A[0, 1] = -2.0 * r

        B[0, 0] = 1.0 - 2.0 * r
        B[0, 1] = 2.0 * r

        for i in range(1, n - 1):
            A[i, i - 1] = -r
            A[i, i] = 1.0 + 2.0 * r
            A[i, i + 1] = -r

            B[i, i - 1] = r
            B[i, i] = 1.0 - 2.0 * r
            B[i, i + 1] = r

        A[n - 1, n - 1] = 1.0
        B[n - 1, n - 1] = 1.0

        return A, B

    def _step_cn(
        self,
        c: np.ndarray,
        c_bulk: float,
        A: np.ndarray,
        B: np.ndarray,
        diffusivity: float,
        dt: float,
        flux_surf: float,
    ) -> np.ndarray:
        """Advance concentration profile c by one time step dt using Crank-Nicolson."""
        r = diffusivity * dt / (2.0 * self.dx**2)
        rhs = B @ c

        rhs[0] += 4.0 * r * self.dx * flux_surf / diffusivity
        rhs[-1] = c_bulk

        c_next = np.linalg.solve(A, rhs)
        c_next[-1] = c_bulk
        return np.maximum(c_next, 0.0)


def compare_dc_vs_pulse(
    j_peak_mA_cm2: float = 100.0,
    duty_cycle: float = 0.5,
    frequency_Hz: float = 10.0,
    n_cycles: int = 20,
    fe_bulk_M: float = 1.0,
    bulk_pH: float = 2.0,
) -> Dict[str, Any]:
    """Compare continuous DC electrodeposition against Pulse and Pulse-Reverse plating."""
    t_cycle = 1.0 / frequency_Hz
    t_on = t_cycle * duty_cycle

    pe_waveform = PulseWaveform(
        j_cathodic_mA_cm2=j_peak_mA_cm2,
        t_cathodic_s=t_on,
        j_anodic_mA_cm2=0.0,
        t_anodic_s=0.0,
        t_off_s=t_cycle - t_on,
    )

    t_rev = (t_cycle - t_on) * 0.2
    t_off = (t_cycle - t_on) * 0.8
    pre_waveform = PulseWaveform(
        j_cathodic_mA_cm2=j_peak_mA_cm2,
        t_cathodic_s=t_on,
        j_anodic_mA_cm2=-0.2 * j_peak_mA_cm2,
        t_anodic_s=t_rev,
        t_off_s=t_off,
    )

    dc_peak_waveform = PulseWaveform(
        j_cathodic_mA_cm2=j_peak_mA_cm2,
        t_cathodic_s=t_cycle * n_cycles,
    )

    j_avg = pe_waveform.j_avg_mA_cm2
    dc_avg_waveform = PulseWaveform(
        j_cathodic_mA_cm2=j_avg,
        t_cathodic_s=t_cycle * n_cycles,
    )

    model = PulseDepositionModel(fe_bulk_M=fe_bulk_M, bulk_pH=bulk_pH)

    res_pe = model.simulate(pe_waveform, n_cycles=n_cycles)
    res_pre = model.simulate(pre_waveform, n_cycles=n_cycles)
    res_dc_peak = model.simulate(dc_peak_waveform, n_cycles=1, steps_per_cycle=n_cycles * 100)
    res_dc_avg = model.simulate(dc_avg_waveform, n_cycles=1, steps_per_cycle=n_cycles * 100)

    return {
        "dc_peak": res_dc_peak,
        "dc_avg": res_dc_avg,
        "pulsed": res_pe,
        "pulse_reverse": res_pre,
    }
