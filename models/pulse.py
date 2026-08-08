"""Transient pulse and pulse-reverse electrodeposition modeling.

Models time-dependent concentration profiles, surface Fe2+ depletion, local pH
relaxation, and faradaic efficiency under pulsed and pulse-reverse current waveforms.

Current-split kinetics (2026-08 rewrite, ``kinetics="bv"`` default)
-------------------------------------------------------------------
Per time step the model solves for the ONE cathode potential that delivers
the applied current from the two surface-state Butler–Volmer branches
(``kinetics.surface_bv_branches``, transport-free because the Crank–Nicolson
film already owns transport):

    i_Fe_BV(E; c_Fe,surf) + i_HER_BV(E; pH_surf, T) = j_app(t)

Consequences the old heuristic split could not represent:

* **Open-circuit corrosion during off periods** — at j = 0 the surface sits
  at the mixed potential between E_eq(Fe) and E_eq(HER): a small Fe
  dissolution current runs with equal-and-opposite HER, so rest periods cost
  a little plated iron and generate H₂.
* **Reverse pulses sit at a corrosion potential too** — applied anodic
  charge is carried by Fe dissolution PLUS residual cathodic HER, so more
  iron dissolves per reverse coulomb than the charge suggests, with an H⁺
  flux to match.
* **Mass consistency through deep depletion** — both forward arms carry a
  first-order surface-activity scale (c_surf/c_bulk;
  ``ButlerVolmerBranch.current_scaled``), so each wall flux vanishes as its
  reactant starves instead of pinning phantom current; anodic arms are
  unscaled (their reactants are the solid / adsorbed H).
* Exchange currents follow the gating engine's screening family
  (``diffusion_layer_1d``: fe_i0 = 10 A/m², her_i0 = 0.010 A/m² at the
  50 °C anchor) and diffusivities are Arrhenius-scaled from their 25 °C
  anchors (``kinetics.py`` conventions), so DC pulse results sit in the
  same FE family as the reference FE engine.

Honest limitations (all L0 screening): E_eq(Fe) is the fixed canonical
E⁰(Fe²⁺/Fe) = −0.440 V with the first-order surface-activity scale
carrying the concentration response (the ``DepositionKinetics``
convention; ``diffusion_layer_1d`` instead Nernst-shifts E_eq with fixed
i0 — the forms agree at mild depletion, deviate deep in the starved
regime, and are flagged for calibration to close).  HER is a homogeneous
BV branch, not the full microkinetic model; the HSO₄⁻/water-coupled
proton sources that keep real sulfate baths supplied at pH ≳ 4 are NOT in
this two-species film — surface pH saturating upward is therefore the
model signalling its own invalid envelope (it stays mass-consistent, but
the kinetics there are the water-reduction regime, unmodelled). Waveform
morphology/leveling effects are outcomes of concentration/pH excursions
only.  ``kinetics="heuristic"`` preserves the pre-2026-08 preference-ratio
split verbatim for A/B checks (``docs/SIM_PULSE_BV.md``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np
from scipy.optimize import brentq

from .electrochemistry import FARADAY, M_FE, RHO_FE
from .kinetics import (
    EA_DIFFUSION_J_MOL,
    arrhenius_diffusivity,
    surface_bv_branches,
)
from .pourbaix import E0_FE3_FE2, LOGKSP_FEOH3

# Standard diffusivities (m^2/s), 25 °C anchors (Arrhenius-scaled to the
# model temperature at construction; kinetics.py conventions)
DIFFUSIVITY_FE2 = 7.2e-10   # Fe²⁺ infinite-dilution, 25°C (CRC); consistent with transport.py D_FE
DIFFUSIVITY_H = 9.31e-9

# ─── Fe²⁺/Fe³⁺ anodic-dissolution split (CHEM_PHYS_REVIEW §2.5, L0) ─────
# Equilibrium potential of the Fe²⁺/Fe³⁺ couple (V vs SHE) — the
# thermodynamic anchor of the reverse-pulse oxidation-state split
# (porbaix.py; pH-independent, no H⁺ in the couple).
E0_FE3_FE2_V = E0_FE3_FE2
# Canonical Fe/Fe²⁺ equilibrium potential (V vs SHE) — the reference for the
# reverse-pulse anodic overpotential η = E − E_eq(Fe) that the BV branch spans
# (surface_bv_branches default), not the fixed Fe²⁺/Fe³⁺ couple above.
E_EQ_FE_V = -0.440
# Fe³⁺ hydrolysis precipitation pH (Pourbaix construction, a_Fe = 0.1): above
# this, aqueous Fe³⁺ is hydrolysed to Fe(OH)₃ sludge, so the effective
# Fe²⁺/Fe³⁺ split collapses toward Fe²⁺.  (≈ 1.48 at 25 °C.)
PH_FE3_PPT = 14.0 + (LOGKSP_FEOH3 - math.log10(0.1)) / 3.0
# Anodic overpotential (η = E − E_eq(Fe)) at which the Fe³⁺ fraction reaches
# half its ceiling (V).  Fitted screening value, NOT gate evidence.
FE3_CHAR_V = 0.5
# Strength of the pH suppression of Fe³⁺: the aqueous Fe³⁺ split falls by 10×
# per pH unit above the Fe³⁺ precipitation boundary.
PH_SUPPRESS_PER_UNIT = 1.0


def fe3_fraction_on_anodic(
    anodic_overpotential_V: float,
    ph: float,
    temperature_C: float = 50.0,
    fe3_char_v: float = FE3_CHAR_V,
    ph_ppt: Optional[float] = None,
    ph_suppress_per_unit: float = PH_SUPPRESS_PER_UNIT,
) -> float:
    """Fraction of anodic Fe dissolution released as Fe³⁺ (screening, L0).

    Real reverse-pulse dissolution is not all Fe²⁺: an anodic flux oxidises
    freshly-dissolved iron toward Fe³⁺ (the E0_Fe3/Fe2 = 0.771 V line of
    porbaix.py).  That Fe³⁺ seeds Fe(OH)₃ sludge and the H₂ problem at
    restart.  The aqueous Fe³⁺ fraction

        f_Fe3 = [1 + (η_char/η)²]⁻¹ · 10^(−κ·max(pH − pH_ppt, 0))

    rises monotonically with anodic overpotential η = E − E_eq(Fe) (the
    driving force the resolved BV anodic branch actually spans — the model's
    reverse-pulse E tops out well anodic of E_eq but short of 0.771 V, so η
    is the live signal, referenced to the 0.771 V couple as its asymptotic
    anchor) and FALLS with rising pH as Fe³⁺ is hydrolysed to Fe(OH)₃ sludge.

    It returns 0 at η = 0 and decays to 0 at high pH, so it reproduces the
    legacy 100 %-Fe²⁺ anodic branch at the reference operating point.  Charge
    is NOT conserved here — the caller must apply the charge-weighted split
    (2 e⁻/Fe²⁺ vs 3 e⁻/Fe³⁺) to keep the current balance exact.
    """
    eta = max(float(anodic_overpotential_V), 0.0)
    f_eta = 1.0 / (1.0 + (fe3_char_v / eta) ** 2) if eta > 0.0 else 0.0
    ppt = PH_FE3_PPT if ph_ppt is None else float(ph_ppt)
    excess = max(float(ph) - ppt, 0.0)
    f_ph = 10.0 ** (-ph_suppress_per_unit * excess)
    return min(float(f_eta * f_ph), 1.0)


def fe3_charge_split(fe_total_current_A_m2: float, fe3_fraction: float) -> Tuple[float, float]:
    """Charge-weighted Fe²⁺/Fe³⁺ partial currents for a total Fe-branch current.

    ``fe_total_current_A_m2`` is the resolved Fe-branch current (signed;
    negative on a reverse-pulse anodic branch).  With Fe-atom flux ``N``,
    ``i_total = F·N·(2 + f)``; the Fe²⁺ path (2 e⁻) and the Fe³⁺ path
    (3 e⁻) carry

        i_Fe2 = i_total·2(1−f)/(2+f),   i_Fe3 = i_total·3f/(2+f),

    which conserves charge (``i_Fe2 + i_Fe3 == i_total``) while the Fe³⁺ path
    strips fewer Fe atoms per coulomb (3 e⁻/Fe vs 2 e⁻/Fe).  Returns
    ``(fe2_current, fe3_current)``, both signed like the input.
    """
    f = max(min(float(fe3_fraction), 1.0), 0.0)
    if f <= 0.0:
        return float(fe_total_current_A_m2), 0.0
    d = 2.0 + f
    fe2 = float(fe_total_current_A_m2) * 2.0 * (1.0 - f) / d
    fe3 = float(fe_total_current_A_m2) * 3.0 * f / d
    return fe2, fe3


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
    cathode_potential_V: Optional[np.ndarray] = None
    proton_limited_steps_fraction: float = 0.0
    # Fe²⁺/Fe³⁺ anodic-dissolution split (CHEM_PHYS_REVIEW §2.5, L0).
    # fe3_current_A_m2 is the Fe³⁺-path partial current (negative on a
    # reverse pulse, signed like fe_current); all-zero when fe3_split is off.
    fe3_current_A_m2: Optional[np.ndarray] = None
    # Cycle-and-run-averaged Fe³⁺ production flux from the anodic branch
    # (mol Fe³⁺/m²/s of cathode) — the coupling handle into fe3_shuttle.
    cycle_avg_anodic_fe3_flux_mol_m2_s: float = 0.0
    # Same, as a fraction of the total anodic Fe dissolution charge.
    cycle_avg_anodic_fe3_fraction: float = 0.0

    def summary(self) -> Dict[str, Any]:
        """Return unit-labelled summary dictionary."""
        out = {
            "Frequency (Hz)": self.waveform.frequency_Hz,
            "Duty cycle (%)": self.waveform.duty_cycle * 100.0,
            "Average current density (mA/cm²)": self.waveform.j_avg_mA_cm2,
            "Cycle-averaged current efficiency (%)": self.cycle_avg_efficiency * 100.0,
            "Net plating rate (µm/hr)": self.plating_rate_um_hr,
            "Net Fe deposited (g/m²)": self.net_fe_deposited_g_m2,
            "Min surface Fe²⁺ ratio (C_surf / C_bulk)": self.peak_surface_depletion_ratio,
            "Max surface pH": self.max_surface_pH,
        }
        if self.proton_limited_steps_fraction > 0.0:
            out["Proton-limited steps fraction (outside envelope!)"] = (
                self.proton_limited_steps_fraction)
        return out


class PulseDepositionModel:
    """Transient 1D diffusion-kinetics model for pulsed and pulse-reverse electrowinning.

    Parameters
    ----------
    kinetics : str
        ``"bv"`` (default): per-step cathode-potential solve against signed
        Butler–Volmer surface branches (module docstring).
        ``"heuristic"``: legacy pre-2026-08 concentration preference-ratio
        split, kept verbatim for A/B checks.
    temperature_C : float
        Bath temperature.  Exchange currents are Arrhenius-scaled from the
        50 °C kinetics anchor; diffusivities from their 25 °C anchors
        (``kinetics.py`` conventions, same as ``diffusion_layer_1d``).
    fe_i0_A_m2, her_i0_A_m2 : float
        Exchange current densities at the 50 °C kinetics anchor (A/m²).
        Defaults are the gating engine's (``diffusion_layer_1d``) screening
        values 10.0 / 0.010, so pulse DC results land in the same FE family
        as the reference FE engine.  (``DepositionKinetics``' older demo
        defaults differ: 1e-2 / 1e-3.  The pre-2026-08 heuristic module
        defaults were 10.0 / 1e-4 — see ``docs/SIM_PULSE_BV.md``.)
    diffusivity_fe_m2_s, diffusivity_h_m2_s : float
        25 °C-anchored diffusivities (Arrhenius-scaled to ``temperature_C``).
    """

    def __init__(
        self,
        boundary_layer_m: float = 1.0e-4,
        fe_bulk_M: float = 1.0,
        bulk_pH: float = 2.0,
        diffusivity_fe_m2_s: float = DIFFUSIVITY_FE2,
        diffusivity_h_m2_s: float = DIFFUSIVITY_H,
        her_i0_A_m2: float = 0.010,
        fe_i0_A_m2: float = 10.0,
        fe_tafel_V: float = 0.120,
        her_tafel_V: float = 0.140,
        temperature_C: float = 50.0,
        kinetics: str = "bv",
        grid_points: int = 51,
        fe3_split: bool = False,
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
        if her_i0_A_m2 <= 0.0 or fe_i0_A_m2 <= 0.0:
            raise ValueError("Exchange current densities must be positive")
        if kinetics not in ("bv", "heuristic"):
            raise ValueError("kinetics must be 'bv' or 'heuristic'")

        self.boundary_layer_m = boundary_layer_m
        self.fe_bulk_M = fe_bulk_M
        self.bulk_pH = bulk_pH
        self.c_h_bulk_mol_m3 = (10.0 ** (-bulk_pH)) * 1000.0
        self.temperature_C = temperature_C
        self.kinetics = kinetics
        self.fe3_split = bool(fe3_split)
        T_K = temperature_C + 273.15
        self.diffusivity_fe = arrhenius_diffusivity(
            diffusivity_fe_m2_s, T_K, EA_DIFFUSION_J_MOL)
        self.diffusivity_h = arrhenius_diffusivity(
            diffusivity_h_m2_s, T_K, EA_DIFFUSION_J_MOL)
        self.her_i0 = her_i0_A_m2
        self.fe_i0 = fe_i0_A_m2
        self.fe_tafel_V = fe_tafel_V
        self.her_tafel_V = her_tafel_V
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
        j_fe3 = np.zeros(n_steps + 1)
        inst_eff = np.zeros(n_steps + 1)
        n_proton_limited = 0
        fe3_flux_sum = 0.0      # mol Fe³⁺/m² of anode flux, summed over steps
        anodic_i_sum = 0.0      # Σ of anodic Fe-dissolution current magnitude (A/m²)

        # Record initial state
        applied_j[0] = waveform.evaluate_current_A_m2(0.0)
        surf_fe[0] = c_fe[0] / 1000.0
        surf_ph[0] = -np.log10(max(c_h[0] / 1000.0, 1e-14))
        pot_arr = (np.zeros(n_steps + 1) if self.kinetics == "bv" else None)
        if pot_arr is not None:
            _, _, _, pot_arr[0] = self._kinetic_split(
                applied_j[0], max(c_fe[0], 1e-6), max(c_h[0], 1e-12))

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

            i_fe_step, i_her_step, eff_step, pot_step = self._kinetic_split(j_app, c_fe_surf, c_h_surf)
            # Fe²⁺/Fe³⁺ anodic-dissolution split (opt-in, §2.5).  On an anodic
            # Fe branch (reverse pulse OR open-circuit rest corrosion) the total
            # i_fe is apportioned charge-consistently between the Fe²⁺ path
            # (2 e⁻/Fe) and the Fe³⁺ path (3 e⁻/Fe); the Fe³⁺ produced is
            # reported for fe3_shuttle closure and subtracted from the Fe²⁺
            # released back to the film.
            fe2_current, fe3_current, fe3_flux = self._split_anodic_fe3(
                i_fe_step, pot_step, c_h_surf)
            j_fe[step] = fe2_current
            j_fe3[step] = fe3_current
            j_her[step] = i_her_step
            inst_eff[step] = eff_step
            if pot_arr is not None:
                pot_arr[step] = pot_step
            fe3_flux_sum += fe3_flux
            anodic_i_sum += max(-i_fe_step, 0.0)   # anodic Fe-dissolution current

            # Envelope diagnostic (counted, never rescaled — see helper).
            n_proton_limited += self._proton_limited_step(i_her_step, c_h_surf)

            # Surface fluxes (mol / m^2 s)
            # Positive current -> cathodic (depletion of Fe2+, consumption of H+)
            # Negative current -> anodic (dissolution of Fe, no HER)
            flux_fe = -fe2_current / (2.0 * FARADAY)
            flux_h = -i_her_step / FARADAY

            # Advance 1D profiles using Crank-Nicolson step
            c_fe = self._step_cn(c_fe, c_fe_bulk, A_fe, B_fe, self.diffusivity_fe, dt, flux_fe)
            c_h = self._step_cn(c_h, c_h_bulk, A_h, B_h, self.diffusivity_h, dt, flux_h)

            surf_fe[step + 1] = max(c_fe[0], 0.0) / 1000.0
            surf_ph[step + 1] = -np.log10(max(c_h[0] / 1000.0, 1e-14))

        # Final step recording
        j_last = waveform.evaluate_current_A_m2(time_arr[-1])
        applied_j[-1] = j_last
        i_fe_last, i_her_last, eff_last, pot_last = self._kinetic_split(j_last, c_fe[0], c_h[0])
        fe2_last, fe3_last, _ = self._split_anodic_fe3(i_fe_last, pot_last, c_h[0])
        j_fe[-1] = fe2_last
        j_fe3[-1] = fe3_last
        j_her[-1] = i_her_last
        inst_eff[-1] = eff_last
        if pot_arr is not None:
            pot_arr[-1] = pot_last

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

        # Cycle-and-run-averaged Fe³⁺ production from the anodic branch.
        cycle_avg_fe3_flux = fe3_flux_sum / max(n_steps, 1)     # mol Fe³⁺/m²/s
        # Charge-weighted Fe³⁺ fraction of total anodic Fe dissolution:
        # Σ(-i_Fe3)·dt / Σ(-i_Fe)·dt = fe3_flux_sum·3F / anodic_i_sum (dt cancels,
        # since each step's Fe³⁺ charge is fe3_flux·3F·dt and its 3F·fe3_flux = -i_Fe3).
        cycle_avg_fe3_frac = 0.0
        if anodic_i_sum > 0.0:
            cycle_avg_fe3_frac = fe3_flux_sum * 3.0 * FARADAY / anodic_i_sum

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
            cathode_potential_V=pot_arr,
            proton_limited_steps_fraction=n_proton_limited / max(n_steps, 1),
            fe3_current_A_m2=j_fe3,
            cycle_avg_anodic_fe3_flux_mol_m2_s=float(cycle_avg_fe3_flux),
            cycle_avg_anodic_fe3_fraction=float(cycle_avg_fe3_frac),
        )

    def _kinetic_split(self, j_app: float, c_fe_surf_mol_m3: float, c_h_surf_mol_m3: float) -> Tuple[float, float, float, Optional[float]]:
        """Split applied current into Fe and HER partial currents.

        Returns ``(i_fe, i_her, instant_efficiency, cathode_potential_V)``.
        The potential is only meaningful (and populated) in BV mode.
        Dispatches on ``self.kinetics``.
        """
        if self.kinetics == "heuristic":
            i_fe, i_her, eff = self._kinetic_split_heuristic(
                j_app, c_fe_surf_mol_m3, c_h_surf_mol_m3)
            return i_fe, i_her, eff, None
        return self._kinetic_split_bv(j_app, c_fe_surf_mol_m3, c_h_surf_mol_m3)

    def _split_anodic_fe3(
        self, i_fe_step: float, pot_step: Optional[float], c_h_surf_mol_m3: float
    ) -> Tuple[float, float, float]:
        """Apportion a resolved Fe-branch current into Fe²⁺/Fe³⁺ partial currents.

        Opt-in (``self.fe3_split``) and only meaningful when the resolved Fe
        branch is actually anodic (``i_fe_step < 0``): the total is split
        charge-consistently — Fe²⁺ at 2 e⁻/Fe, Fe³⁺ at 3 e⁻/Fe — keyed on the
        anodic overpotential (pot − E_eq(Fe)) and surface pH.  Returns
        ``(fe2_current, fe3_current, fe3_flux_mol_m2_s)``; with the flag off,
        or on a non-anodic step, the whole current stays Fe²⁺ and the Fe³⁺
        flux is 0 (legacy behaviour).
        """
        if not (self.fe3_split and pot_step is not None and i_fe_step < 0.0):
            return float(i_fe_step), 0.0, 0.0
        ph_surf = -math.log10(max(c_h_surf_mol_m3 / 1000.0, 1e-14))
        f_fe3 = fe3_fraction_on_anodic(
            pot_step - E_EQ_FE_V, ph_surf, self.temperature_C)
        fe2, fe3 = fe3_charge_split(i_fe_step, f_fe3)
        return fe2, fe3, -fe3 / (3.0 * FARADAY)   # mol Fe³⁺/m²/s

    def fe3_shuttle_closure(self, result: PulseResult, shuttle_params=None, scenario=None) -> Dict[str, Any]:
        """Close the pulse-reverse Fe³⁺ production through fe3_shuttle.

        Feeds the run-averaged anodic Fe³⁺ flux carried by ``result`` into
        ``fe3_shuttle.steady_state`` as an extra Fe³⁺ source term, so the
        Fe(OH)₃ sludge and shuttle CE-leak that the CHEM_PHYS_REVIEW §2.5
        split seeds are reported by the same Fe³⁺ triangle the bath dynamics
        already close (models/fe3_shuttle.py).  Defaults to the RC-1
        catholyte and the sealed divided cell; both can be overridden.
        """
        from .fe3_shuttle import ShuttleParams, sealed_divided_cell, steady_state

        p = shuttle_params if shuttle_params is not None else ShuttleParams(
            pH=self.bulk_pH, temperature_C=self.temperature_C)
        s = scenario if scenario is not None else sealed_divided_cell()
        return steady_state(
            p, s,
            anodic_fe3_source_mol_m2_s=result.cycle_avg_anodic_fe3_flux_mol_m2_s,
        )

    def _proton_limited_step(self, i_her_step: float, c_h_surf_mol_m3: float) -> int:
        """Return 1 when this step is outside the two-species film's envelope.

        HER drawing > 2× the film's steady proton supply while the surface is
        proton-starved means the drive has left the model's validity envelope
        (the real bath answers with HSO4⁻/water-coupled HER — physics this
        film does not carry).  Counted and reported, never rescaled.
        """
        if i_her_step > 0.0 and c_h_surf_mol_m3 < 0.25 * self.c_h_bulk_mol_m3:
            i_supply = (FARADAY * self.diffusivity_h
                        * (self.c_h_bulk_mol_m3 - c_h_surf_mol_m3) / self.boundary_layer_m)
            if i_her_step > 2.0 * max(i_supply, 1e-12):
                return 1
        return 0

    def _kinetic_split_heuristic(self, j_app: float, c_fe_surf_mol_m3: float, c_h_surf_mol_m3: float) -> Tuple[float, float, float]:
        """Pre-2026-08 preference-ratio split, preserved verbatim for A/B checks."""
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

    # ─── Butler–Volmer surface-potential solve (2026-08 default) ─────

    def _branches(self, pH_surf: float):
        """Transport-free signed BV branches at the surface pH and model T."""
        return surface_bv_branches(
            pH_surf, self.temperature_C, self.fe_i0, self.her_i0,
            fe_tafel_V=self.fe_tafel_V, her_tafel_V=self.her_tafel_V,
        )

    def _solve_surface_potential(self, j_app: float, fe_branch, her_branch,
                                 fwd_scale_fe: float, fwd_scale_h: float) -> float:
        """Cathode potential E (V vs SHE) where i_Fe + i_HER = j_app.

        Strictly monotone in E for BV branches, so the bracket is a one-line
        guarantee: deep-cathodic at −3 V vs SHE for any cathodic j, and past
        the Fe dissolution ceiling (E_eq + 15 b_a decades) for any anodic j
        short of 1e12 A/m².  Forward arms carry a first-order surface-activity
        scale (c_surf/c_bulk) so wall fluxes stay mass-consistent through
        depletion; anodic arms are unscaled (solid-/gas-phase reactants).
        """
        def f(E):
            return (float(fe_branch.current_scaled(E, fwd_scale_fe))
                    + float(her_branch.current_scaled(E, fwd_scale_h)) - j_app)

        lo, hi = -3.0, fe_branch.E_eq + 0.6
        return float(brentq(f, lo, hi, xtol=1e-10, rtol=1e-12))

    def _kinetic_split_bv(self, j_app: float, c_fe_surf_mol_m3: float, c_h_surf_mol_m3: float) -> Tuple[float, float, float, float]:
        """Signed BV split at the solved surface potential.

        j > 0: ordinary cathodic split (both branches cathodic).
        j = 0: open-circuit corrosion (i_Fe < 0, i_HER > 0, sum zero).
        j < 0: reverse pulse — i_Fe ≤ j (dissolution carries the applied
        reverse charge PLUS the residual corrosion HER, which keeps running
        while E sits below E_eq(HER)).
        """
        pH_surf = -math.log10(max(c_h_surf_mol_m3 / 1000.0, 1e-14))
        # Surface-activity closure (first order, both forward arms): c→0 shuts
        # the branch down.  For HER this is what makes the starved proton
        # regime mass-consistent — surface pH saturating is then the model's
        # (correct) signal that the bath has entered the HSO4⁻/water-coupled
        # regime this reduced two-species film does not carry (see docstring).
        fwd_scale_fe = min(max(c_fe_surf_mol_m3 / (self.fe_bulk_M * 1000.0), 0.0), 1.0)
        fwd_scale_h = min(max(c_h_surf_mol_m3 / self.c_h_bulk_mol_m3, 0.0), 1.0)
        fe_branch, her_branch = self._branches(pH_surf)
        E = self._solve_surface_potential(j_app, fe_branch, her_branch, fwd_scale_fe, fwd_scale_h)
        i_fe = float(fe_branch.current_scaled(E, fwd_scale_fe))
        i_her = float(her_branch.current_scaled(E, fwd_scale_h))
        eff = i_fe / j_app if j_app > 0.0 else 0.0
        return i_fe, i_her, eff, E

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
    **model_kwargs: Any,
) -> Dict[str, Any]:
    """Compare continuous DC electrodeposition against Pulse and Pulse-Reverse plating.

    Extra keyword arguments are forwarded to ``PulseDepositionModel`` (e.g.
    ``kinetics="heuristic"`` for the legacy split, ``temperature_C=...``).
    """
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

    model = PulseDepositionModel(fe_bulk_M=fe_bulk_M, bulk_pH=bulk_pH, **model_kwargs)

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
