"""
Membrane transport model for divided-cell iron electrowinning.

Models the ionic fluxes through a cation exchange membrane (Nafion N117 or
Fumasep FKE-50) that separates the anolyte and catholyte compartments of a
divided-cell electrowinning reactor.  The module couples:

1. **Fe³⁺ crossover flux** — migration-driven (electric field) and
   diffusion-driven (concentration gradient) transport of Fe³⁺ from the
   anolyte through the membrane into the catholyte.  This is a parasitic
   redox shuttle: Fe³⁺ arriving at the cathode is re-reduced to Fe²⁺,
   wasting current without depositing iron.

2. **Acid balance** — the H⁺ transport number through Nafion (t_H⁺ ≈ 0.9)
   means that nearly all ionic current is carried by protons.  In a
   two-compartment cell the net H⁺ flux across the membrane couples the
   anolyte and catholyte pH evolution.

3. **Anolyte composition drift** — Fe²⁺ is oxidised to Fe³⁺ at the soluble
   iron anode.  The anolyte Fe³⁺ fraction grows with charge passed, which
   increases the crossover driving force and eventually demands a purge or
   chemical regeneration step.

4. **Membrane Ohmic drop** — the fixed-charge sites in the membrane give a
   well-defined ionic conductivity (κ ≈ 0.08 S/cm for hydrated Nafion),
   and the resulting IR drop is a computed fraction of the total cell
   voltage rather than a lumped parameter.

Physics
-------
A dilute-solution Nernst-Planck framework is used inside the membrane::

    J_i = -D_i dC_i/dx - z_i D_i (F / RT) C_i dphi/dx

For a thin membrane of thickness L the fluxes are linearised::

    J_Fe3+ ≈ D_Fe3+ ΔC / L  +  t_Fe3+ j / (3 F)

where the first term is diffusion and the second is migration.  The
transference number t_Fe3+ inside the membrane is estimated from the
Nernst-Planck transport number in the membrane phase::

    t_i = z_i² D_i C_i / Σ_j(z_j² D_j C_j)

Species and chemistry
---------------------
- Anode: Fe²⁺ → Fe³⁺ + e⁻   (E° = +0.771 V vs. SHE)
- Cathode: Fe²⁺ + 2e⁻ → Fe  (E° = −0.440 V vs. SHE)
- Parasitic shuttle: Fe³⁺ crosses membrane, is reduced at cathode, returns
  as Fe²⁺ through the electrolyte or through the membrane.

Scope
-----
Screening model: steady-state fluxes at each time step, no Donnan exclusion
model (uses effective diffusivities that already fold in the exclusion), no
water osmotic drag, no membrane swelling dynamics.  The anolyte is treated
as a well-stirred batch (CSTR without feed).

References
----------
- Nafion N117 properties: DuPont technical bulletin; Zawodzinski et al.
  (1993) J. Phys. Chem. 97, 6042–6050.
- Fe³⁺ diffusivity in Nafion: Ogumi et al. (1984) J. Electrochem. Soc.
  131, 769–773.
- Fumasep FKE-50: Fumatech product data sheet.
- Transport in ion exchange membranes: Helfferich (1962) "Ion Exchange".

Units
-----
Concentrations: mol/L (= mol/dm³) throughout.
Current densities: A/m² SI.
Membrane thickness: m.
Fluxes: mol/(m²·s).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .electrochemistry import FARADAY

# ─── Iron redox constants ──────────────────────────────────────────────
E0_FE3_FE2 = 0.771        # V vs. SHE  (Fe³⁺ + e⁻ → Fe²⁺)
Z_FE3 = 3                 # charge on Fe³⁺
Z_FE2 = 2                 # charge on Fe²⁺
Z_H = 1                   # charge on H⁺

# ─── Bulk diffusivities (m²/s) in free solution, 25–60 °C ─────────────
D_FE3_BULK = 6.0e-10      # Fe³⁺ in water (Ogumi et al.)
D_FE2_BULK = 7.2e-10      # Fe²⁺ in water
D_H_BULK   = 9.31e-9      # H⁺ (Grotthuss)

# ─── Nafion N117 defaults ──────────────────────────────────────────────
NAFION_N117_EW = 1100.0    # g/mol equivalent weight
NAFION_N117_THICKNESS = 180e-6  # m  (fully hydrated)
NAFION_N117_CONDUCTIVITY = 8.0   # S/m  (0.08 S/cm)
NAFION_N117_D_FE3 = 1.0e-11     # m²/s  (2 orders slower than bulk)
NAFION_N117_D_H = 5.0e-9        # m²/s  (H⁺ in Nafion, slightly slower than bulk)

# ─── Fumasep FKE-50 defaults ──────────────────────────────────────────
FUMASEP_FKE50_EW = 1000.0       # g/mol
FUMASEP_FKE50_THICKNESS = 50e-6  # m  (thinner PEM)
FUMASEP_FKE50_CONDUCTIVITY = 10.0  # S/m
FUMASEP_FKE50_D_FE3 = 2.0e-11   # m²/s  (slightly higher due to thinner film)
FUMASEP_FKE50_D_H = 6.0e-9      # m²/s


# ─── Membrane specification ────────────────────────────────────────────
@dataclass(frozen=True)
class MembraneSpec:
    """Physical and transport properties of a cation exchange membrane.

    Parameters
    ----------
    name : str
        Membrane identifier (e.g. "Nafion N117").
    equivalent_weight : float
        EW in g/mol — mass of dry polymer per mole of fixed sulfonate sites.
    thickness_m : float
        Membrane thickness (m) in the hydrated state.
    conductivity_S_m : float
        Proton conductivity (S/m) in fully hydrated conditions.
    D_Fe3_m2_s : float
        Effective Fe³⁺ diffusion coefficient inside the membrane (m²/s).
    D_H_m2_s : float
        Effective H⁺ diffusion coefficient inside the membrane (m²/s).
    """

    name: str = "Nafion N117"
    equivalent_weight: float = NAFION_N117_EW
    thickness_m: float = NAFION_N117_THICKNESS
    conductivity_S_m: float = NAFION_N117_CONDUCTIVITY
    D_Fe3_m2_s: float = NAFION_N117_D_FE3
    D_H_m2_s: float = NAFION_N117_D_H


# Pre-defined membrane catalogue
NAFION_N117 = MembraneSpec()
FUMASEP_FKE50 = MembraneSpec(
    name="Fumasep FKE-50",
    equivalent_weight=FUMASEP_FKE50_EW,
    thickness_m=FUMASEP_FKE50_THICKNESS,
    conductivity_S_m=FUMASEP_FKE50_CONDUCTIVITY,
    D_Fe3_m2_s=FUMASEP_FKE50_D_FE3,
    D_H_m2_s=FUMASEP_FKE50_D_H,
)


# ─── Donnan exclusion (optional correction) ───────────────────────────
# Cation-exchange membranes carry a fixed negative charge (sulfonate,
# ~1 M for Nafion).  At equilibrium the Donnan potential between the
# membrane phase and the external electrolyte is
#
#   φ_D = (RT/F) · asinh( X / 2c_s )
#
# where X is the fixed charge concentration and c_s the external salt
# concentration.  This partitions co-ions (anions) and enriches counter-
# ions.  For Fe³⁺ the Donnan partition coefficient is
#
#   K_D = exp(-z·F·φ_D / RT)
#
# At low external concentration K_D >> 1 for cations (enrichment) but
# the product D·K_D still suppresses Fe³⁺ because the membrane-phase
# diffusivity is 10²× lower.  The function below gives the correction
# so that a caller can estimate the *effective* bulk driving concentration
# C_mem = K_D · C_ext.  It is not yet wired into the default flux path
# (which uses the fitted effective D that already folds in the mean
# exclusion) — it is an explicit physics diagnostic and an opt-in
# correction for sensitivity studies.

def donnan_potential_V(
    fixed_charge_M: float = 1.0,
    external_salt_M: float = 1.0,
    temperature_C: float = 60.0,
) -> float:
    """Donnan potential φ_D (V) for a cation-exchange membrane.

    φ_D = (RT/F) · asinh(X / 2c_s)   (>0, membrane positive vs solution
    for a cation exchanger with X>0 defined as positive counter-charge).

    The sign convention here returns the membrane-minus-solution
    potential.  Its magnitude is ~15–40 mV for Nafion at 0.5–2 M salt.
    """
    import math
    R = 8.314462618
    F = 96485.33212
    T = temperature_C + 273.15
    return float((R * T / F) * math.asinh(fixed_charge_M / (2.0 * max(external_salt_M, 1e-6))))


def donnan_partition_coefficient(
    z: int,
    fixed_charge_M: float = 1.0,
    external_salt_M: float = 1.0,
    temperature_C: float = 60.0,
) -> float:
    """Partition coefficient K_D = C_mem / C_ext for ion of charge z.

    K_D = exp(-z·F·φ_D / RT).  For Fe³⁺ (z=+3) in Nafion at 1 M salt,
    K_D ≈ 0.3–0.5 (electrostatic enrichment is overwhelmed by the
    low membrane diffusivity in the net flux).
    """
    import math
    R = 8.314462618
    F = 96485.33212
    T = temperature_C + 273.15
    phi_d = donnan_potential_V(fixed_charge_M, external_salt_M, temperature_C)
    return float(math.exp(-z * F * phi_d / (R * T)))


# ─── Anolyte / catholyte state ────────────────────────────────────────
@dataclass
class AnolyteState:
    """Composition and volume of the anolyte compartment.

    Parameters
    ----------
    volume_L : float
        Anolyte volume (L).
    fe2_M : float
        Fe²⁺ concentration (mol/L).
    fe3_M : float
        Fe³⁺ concentration (mol/L).
    h_M : float
        H⁺ concentration (mol/L).  In a sulfate bath this equals
        2[FeSO₄] + [H₂SO₄] excess; here it is a tracked variable.
    charge_passed_Ah : float
        Cumulative charge passed through the cell (A·h).
    """

    volume_L: float = 10.0
    fe2_M: float = 1.0
    fe3_M: float = 0.0
    h_M: float = 1.0          # ~pH 0 in 1 M H₂SO₄
    charge_passed_Ah: float = 0.0


@dataclass
class CatholyteState:
    """Composition of the catholyte compartment.

    Parameters
    ----------
    volume_L : float
        Catholyte volume (L).
    fe2_M : float
        Fe²⁺ concentration (mol/L) — replenished from dissolution / crossover.
    fe3_M : float
        Fe³⁺ concentration (mol/L) — introduced by crossover shuttle.
    h_M : float
        H⁺ concentration (mol/L).
    """

    volume_L: float = 10.0
    fe2_M: float = 1.0
    fe3_M: float = 0.0
    h_M: float = 1.0


# ─── Time-step result ─────────────────────────────────────────────────
@dataclass
class MembraneTimeStep:
    """Instantaneous fluxes and rates at one operating point.

    Attributes
    ----------
    fe3_crossover_flux : float
        Total Fe³⁺ flux through the membrane (mol/(m²·s)).
    fe3_migration_flux : float
        Migration-driven component of the Fe³⁺ flux.
    fe3_diffusion_flux : float
        Diffusion-driven component of the Fe³⁺ flux.
    h_flux : float
        Net H⁺ flux through the membrane (mol/(m²·s)).
        Positive = anolyte → catholyte.
    membrane_V_drop : float
        Ohmic voltage drop across the membrane (V).
    fe3_crossover_current_A_m2 : float
        Equivalent cathodic current from Fe³⁺ shuttle (A/m²).
    anolyte_fe3_production_mol_s : float
        Rate of Fe³⁺ generation at the anode (mol/s).
    """

    fe3_crossover_flux: float
    fe3_migration_flux: float
    fe3_diffusion_flux: float
    h_flux: float
    membrane_V_drop: float
    fe3_crossover_current_A_m2: float
    anolyte_fe3_production_mol_s: float


# ─── Simulation result ────────────────────────────────────────────────
@dataclass
class MembraneSimulationResult:
    """Full time-series output of a divided-cell membrane simulation.

    Attributes
    ----------
    time_hr : np.ndarray
        Simulation time (h).
    anolyte_fe2_M, anolyte_fe3_M, anolyte_h_M : np.ndarray
        Anolyte composition vs. time.
    catholyte_fe3_M, catholyte_h_M : np.ndarray
        Catholyte composition vs. time.
    fe3_crossover_flux : np.ndarray
        Fe³⁺ crossover flux (mol/(m²·s)) vs. time.
    h_flux : np.ndarray
        H⁺ flux (mol/(m²·s)) vs. time.
    membrane_V_drop : np.ndarray
        Membrane IR drop (V) vs. time.
    crossover_current_A_m2 : np.ndarray
        Equivalent parasitic current (A/m²) vs. time.
    purge_events : list of (time_hr, fe3_M_before)
        Times at which the anolyte was purged.
    fe_crossover_loss_pct : float
        Cumulative Fe lost to crossover as a percentage of Fe deposited.
    """

    time_hr: np.ndarray
    anolyte_fe2_M: np.ndarray
    anolyte_fe3_M: np.ndarray
    anolyte_h_M: np.ndarray
    catholyte_fe3_M: np.ndarray
    catholyte_h_M: np.ndarray
    fe3_crossover_flux: np.ndarray
    h_flux: np.ndarray
    membrane_V_drop: np.ndarray
    crossover_current_A_m2: np.ndarray
    purge_events: list = field(default_factory=list)
    fe_crossover_loss_pct: float = 0.0

    def summary(self) -> dict:
        return {
            "duration_hr": float(self.time_hr[-1]),
            "final_anolyte_fe3_M": float(self.anolyte_fe3_M[-1]),
            "final_catholyte_fe3_M": float(self.catholyte_fe3_M[-1]),
            "final_catholyte_pH": float(-np.log10(max(self.catholyte_h_M[-1], 1e-30))),
            "peak_crossover_flux_mol_m2_s": float(np.max(self.fe3_crossover_flux)),
            "peak_membrane_V_drop": float(np.max(self.membrane_V_drop)),
            "n_purge_events": len(self.purge_events),
            "fe_crossover_loss_pct": round(self.fe_crossover_loss_pct, 2),
        }


# ─── Core model ───────────────────────────────────────────────────────
class MembraneTransportModel:
    """Fe³⁺ crossover, acid balance, and anolyte drift in a divided cell.

    Parameters
    ----------
    membrane : MembraneSpec
        Membrane properties (Nafion N117 or Fumasep FKE-50).
    electrode_area_m2 : float
        Active membrane / electrode area (m²).
    temperature_C : float
        Operating temperature (°C).
    j_mA_cm2 : float
        Applied current density (mA/cm²).
    anolyte : AnolyteState
        Initial anolyte composition.
    catholyte : CatholyteState
        Initial catholyte composition.
    purge_fe3_threshold_M : float
        Anolyte Fe³⁺ concentration (mol/L) that triggers a purge.
    purge_fraction : float
        Fraction of anolyte volume replaced during a purge event.
    """

    def __init__(
        self,
        membrane: MembraneSpec | None = None,
        electrode_area_m2: float = 0.01,
        temperature_C: float = 60.0,
        j_mA_cm2: float = 100.0,
        anolyte: AnolyteState | None = None,
        catholyte: CatholyteState | None = None,
        purge_fe3_threshold_M: float = 0.5,
        purge_fraction: float = 0.2,
    ) -> None:
        self.membrane = membrane or NAFION_N117
        self.electrode_area_m2 = electrode_area_m2
        self.temperature_C = temperature_C
        self.j_mA_cm2 = j_mA_cm2
        self.anolyte = anolyte or AnolyteState()
        self.catholyte = catholyte or CatholyteState()
        self.purge_fe3_threshold_M = purge_fe3_threshold_M
        self.purge_fraction = purge_fraction

        if electrode_area_m2 <= 0:
            raise ValueError("electrode_area_m2 must be positive")
        if j_mA_cm2 < 0:
            raise ValueError("j_mA_cm2 must be non-negative")

    @property
    def T(self) -> float:
        return self.temperature_C + 273.15

    @property
    def j_A_m2(self) -> float:
        return self.j_mA_cm2 * 10.0

    # ─── Membrane transport properties ─────────────────────────────────
    @property
    def membrane_conductivity(self) -> float:
        """Membrane ionic conductivity (S/m)."""
        return self.membrane.conductivity_S_m

    def membrane_ohmic_drop(self, j_A_m2: float | None = None) -> float:
        """Ohmic voltage drop across the membrane (V).

        V = j · L / κ

        Parameters
        ----------
        j_A_m2 : float, optional
            Current density (A/m²).  Defaults to the operating value.
        """
        j = j_A_m2 if j_A_m2 is not None else self.j_A_m2
        kappa = self.membrane.conductivity_S_m
        L = self.membrane.thickness_m
        return float(j * L / kappa)

    def h_transport_number(self) -> float:
        """H⁺ transference number inside the membrane.

        t_H⁺ = D_H C_H / (D_H C_H + D_Fe3 C_Fe3 * z_Fe3² / z_H²)

        Uses the anolyte H⁺ and Fe³⁺ concentrations as the representative
        membrane-phase composition (the membrane is in Donnan equilibrium
        with the anolyte side).
        """
        d_h = self.membrane.D_H_m2_s
        d_fe3 = self.membrane.D_Fe3_m2_s
        c_h = max(self.anolyte.h_M, 1e-15) * 1000.0   # mol/m³
        c_fe3 = max(self.anolyte.fe3_M, 1e-15) * 1000.0
        w_h = Z_H**2 * d_h * c_h
        w_fe3 = Z_FE3**2 * d_fe3 * c_fe3
        return float(w_h / (w_h + w_fe3))

    def fe3_transference_number(self) -> float:
        """Fe³⁺ transference number inside the membrane."""
        return 1.0 - self.h_transport_number()

    # ─── Fe³⁺ crossover flux ──────────────────────────────────────────
    def fe3_crossover_flux(
        self,
        c_anolyte_fe3_M: float | None = None,
        c_catholyte_fe3_M: float | None = None,
        j_A_m2: float | None = None,
    ) -> tuple[float, float, float]:
        """Fe³⁺ flux through the membrane (mol/(m²·s)).

        Returns (total, migration, diffusion).

        Migration term:  J_mig = t_Fe3⁺ · j / (3 · F)
        Diffusion term:  J_diff = D_Fe3⁺ · ΔC / L

        Parameters
        ----------
        c_anolyte_fe3_M : float
            Anolyte Fe³⁺ concentration (mol/L).  Defaults to current state.
        c_catholyte_fe3_M : float
            Catholyte Fe³⁺ concentration (mol/L).  Defaults to current state.
        j_A_m2 : float
            Current density (A/m²).  Defaults to operating value.
        """
        ca = c_anolyte_fe3_M if c_anolyte_fe3_M is not None else self.anolyte.fe3_M
        cc = c_catholyte_fe3_M if c_catholyte_fe3_M is not None else self.catholyte.fe3_M
        j = j_A_m2 if j_A_m2 is not None else self.j_A_m2

        L = self.membrane.thickness_m
        D = self.membrane.D_Fe3_m2_s

        # Migration: Fe³⁺ moves from anolyte → catholyte under anodic field
        t_fe3 = self.fe3_transference_number()
        j_mig = t_fe3 * j / (Z_FE3 * FARADAY)

        # Diffusion: down the concentration gradient
        delta_c = (ca - cc) * 1000.0  # mol/L → mol/m³
        j_diff = D * delta_c / L

        return float(j_mig + j_diff), float(j_mig), float(j_diff)

    def fe3_crossover_current(
        self,
        c_anolyte_fe3_M: float | None = None,
        c_catholyte_fe3_M: float | None = None,
        j_A_m2: float | None = None,
    ) -> float:
        """Equivalent parasitic cathodic current from Fe³⁺ shuttle (A/m²).

        Each mole of Fe³⁺ arriving at the cathode consumes 1 electron:
        Fe³⁺ + e⁻ → Fe²⁺.
        """
        flux, _, _ = self.fe3_crossover_flux(c_anolyte_fe3_M, c_catholyte_fe3_M, j_A_m2)
        return float(flux * FARADAY)

    # ─── H⁺ flux ──────────────────────────────────────────────────────
    def h_flux(self, j_A_m2: float | None = None) -> float:
        """Net H⁺ flux through the membrane (mol/(m²·s)).

        H⁺ carries (t_H⁺) of the ionic current.  Convention: positive =
        anolyte → catholyte (the field drives cations from anode side to
        cathode side in the membrane).
        """
        j = j_A_m2 if j_A_m2 is not None else self.j_A_m2
        t_h = self.h_transport_number()
        return float(t_h * j / (Z_H * FARADAY))

    # ─── Single time-step snapshot ─────────────────────────────────────
    def evaluate(
        self,
        c_anolyte_fe3_M: float | None = None,
        c_catholyte_fe3_M: float | None = None,
    ) -> MembraneTimeStep:
        """Evaluate all fluxes and rates at the current operating point."""
        j = self.j_A_m2
        flux_total, flux_mig, flux_diff = self.fe3_crossover_flux(
            c_anolyte_fe3_M, c_catholyte_fe3_M, j
        )
        h_flux = self.h_flux(j)
        v_drop = self.membrane_ohmic_drop(j)
        j_crossover = flux_total * FARADAY

        # Anolyte Fe²⁺ → Fe³⁺ production at the anode
        # Rate = j * A / F  (mol/s), all current oxidises Fe²⁺
        anolyte_rate = j * self.electrode_area_m2 / FARADAY

        return MembraneTimeStep(
            fe3_crossover_flux=flux_total,
            fe3_migration_flux=flux_mig,
            fe3_diffusion_flux=flux_diff,
            h_flux=h_flux,
            membrane_V_drop=v_drop,
            fe3_crossover_current_A_m2=j_crossover,
            anolyte_fe3_production_mol_s=anolyte_rate,
        )

    # ─── Catholyte pH drift ───────────────────────────────────────────
    def catholyte_pH_drift_rate(
        self,
        j_A_m2: float | None = None,
    ) -> float:
        """Rate of H⁺ concentration change in the catholyte (mol/(L·s)).

        H⁺ arrives from the anolyte through the membrane.  At the cathode,
        HER consumes H⁺.  The net drift is:

            dC_H+/dt = (h_flux * A) / V_cat   (H⁺ arriving)
                     - j_HER / (F * V_cat)     (H⁺ consumed by HER)

        For the acid-balance model we track only the membrane contribution
        since the HER consumption depends on the cathode kinetics (handled
        by the kinetics module).  Here we return the *membrane input* only.

        Positive = H⁺ concentration rising in catholyte.
        """
        j = j_A_m2 if j_A_m2 is not None else self.j_A_m2
        h_f = self.h_flux(j)
        V_cat = self.catholyte.volume_L * 1e-3  # L → m³
        return float(h_f * self.electrode_area_m2 / V_cat / 1000.0)  # mol/m³/s → mol/L/s

    # ─── Time integration ─────────────────────────────────────────────
    def simulate(
        self,
        duration_hr: float = 8.0,
        dt_hr: float = 0.1,
        current_efficiency: float = 0.85,
    ) -> MembraneSimulationResult:
        """Integrate the divided-cell model forward in time.

        Parameters
        ----------
        duration_hr : float
            Simulation duration (h).
        dt_hr : float
            Euler time step (h).
        current_efficiency : float
            Faradaic efficiency for Fe deposition.  The remaining fraction
            goes to HER, which consumes H⁺ at the cathode.
        """
        if duration_hr <= 0 or dt_hr <= 0:
            raise ValueError("duration_hr and dt_hr must be positive")

        dt_s = dt_hr * 3600.0
        j = self.j_A_m2
        A = self.electrode_area_m2
        V_an = self.anolyte.volume_L * 1e-3   # m³
        V_cat = self.catholyte.volume_L * 1e-3
        CE = current_efficiency

        # Clone state to avoid mutating the caller's objects
        fe2_a = self.anolyte.fe2_M
        fe3_a = self.anolyte.fe3_M
        h_a = self.anolyte.h_M
        fe3_c = self.catholyte.fe3_M
        h_c = self.catholyte.h_M
        Q = 0.0  # cumulative charge (A·h)

        n_steps = int(np.ceil(duration_hr / dt_hr)) + 1
        t_arr = np.linspace(0.0, duration_hr, n_steps)
        an_fe2 = np.zeros(n_steps)
        an_fe3 = np.zeros(n_steps)
        an_h = np.zeros(n_steps)
        cat_fe3 = np.zeros(n_steps)
        cat_h = np.zeros(n_steps)
        xover_flux = np.zeros(n_steps)
        h_flux_arr = np.zeros(n_steps)
        v_drop_arr = np.zeros(n_steps)
        j_xover_arr = np.zeros(n_steps)
        purge_events: list[tuple[float, float]] = []

        for i in range(n_steps):
            # Record current state
            an_fe2[i] = fe2_a
            an_fe3[i] = fe3_a
            an_h[i] = h_a
            cat_fe3[i] = fe3_c
            cat_h[i] = h_c

            # Evaluate fluxes at current concentrations
            step = self.evaluate(fe3_a, fe3_c)
            xover_flux[i] = step.fe3_crossover_flux
            h_flux_arr[i] = step.h_flux
            v_drop_arr[i] = step.membrane_V_drop
            j_xover_arr[i] = step.fe3_crossover_current_A_m2

            if i == n_steps - 1:
                break

            # ── Anolyte: Fe²⁺ → Fe³⁺ at the anode ─────────────────
            # Production rate (mol/s): all anodic current oxidises Fe²⁺
            r_anode = j * A / FARADAY

            # Fe³⁺ lost to crossover (mol/s)
            r_xover = step.fe3_crossover_flux * A

            # Fe³⁺ + e⁻ → Fe²⁺ from shuttle at cathode — the Fe²⁺ that
            # forms at the cathode returns as dissolved Fe²⁺; for the
            # anolyte mass balance, Fe³⁺ is removed by crossover.
            # Net anolyte Fe³⁺: produced at anode, lost to crossover
            dfe3_a = (r_anode - r_xover) * dt_s / (V_an * 1000.0)

            # H⁺ produced at the anode: Fe²⁺ → Fe³⁺ + e⁻ does not
            # produce/consume H⁺ directly.  But water oxidation at the
            # anode (in a soluble-anode system) does not occur — all
            # anodic current goes to Fe²⁺ oxidation.  H⁺ change is due
            # to the membrane flux only.
            # dH/dt_anolyte = -h_flux * A / V_an  (H⁺ leaving through membrane)
            dh_a = -step.h_flux * A / (V_an * 1000.0) * dt_s

            # Fe²⁺ consumed by anode reaction and partially replenished
            # by Fe³⁺ reduction at cathode (returns as Fe²⁺ through
            # electrolyte circulation).  In the batch model:
            # dFe2+/dt = -r_anode + r_xover  (Fe²⁺ lost to oxidation,
            #   partially recovered as Fe²⁺ from cathode shuttle)
            # Simplification: the Fe³⁺ reduced at the cathode becomes Fe²⁺
            # in the catholyte; in a loop it recirculates.  For a batch
            # anolyte, we track Fe³⁺ production and crossover separately.
            dfe2_a = -(r_anode - r_xover) * dt_s / (V_an * 1000.0)

            # ── Catholyte: Fe³⁺ arrives from crossover ──────────────
            dfe3_c = step.fe3_crossover_flux * A / (V_cat * 1000.0) * dt_s

            # H⁺ arrives from membrane; consumed by HER (fraction = 1-CE)
            # HER: 2H⁺ + 2e⁻ → H₂   consumes H⁺ proportional to HER current
            j_her = j * (1.0 - CE)
            r_her = j_her * A / (2.0 * FARADAY)  # mol H₂/s, consumes 2 H⁺ per
            dh_c = (step.h_flux * A - 2.0 * r_her) / (V_cat * 1000.0) * dt_s

            # Apply changes
            fe2_a = max(fe2_a + dfe2_a, 0.0)
            fe3_a = max(fe3_a + dfe3_a, 0.0)
            h_a = max(h_a + dh_a, 1e-10)
            fe3_c = max(fe3_c + dfe3_c, 0.0)
            h_c = max(h_c + dh_c, 1e-10)

            # Cumulative charge
            Q += j * A * dt_s / 3600.0  # A·h

            # ── Purge check ─────────────────────────────────────────
            if fe3_a >= self.purge_fe3_threshold_M:
                purge_events.append((t_arr[i + 1], fe3_a))
                fe3_a *= (1.0 - self.purge_fraction)
                # Purge replaces volume with fresh Fe²⁺ solution
                fe2_a = fe2_a * (1.0 - self.purge_fraction) + self.anolyte.fe2_M * self.purge_fraction
                h_a = h_a * (1.0 - self.purge_fraction) + self.anolyte.h_M * self.purge_fraction

        # Fe crossover loss as % of Fe deposited
        total_fe_deposited_mol = j * A * duration_hr * 3600.0 * CE / (2.0 * FARADAY)
        total_fe_crossover_mol = np.sum(xover_flux * A * (dt_hr * 3600.0))
        fe_loss_pct = (
            100.0 * total_fe_crossover_mol / max(total_fe_deposited_mol, 1e-30)
        )

        return MembraneSimulationResult(
            time_hr=t_arr,
            anolyte_fe2_M=an_fe2,
            anolyte_fe3_M=an_fe3,
            anolyte_h_M=an_h,
            catholyte_fe3_M=cat_fe3,
            catholyte_h_M=cat_h,
            fe3_crossover_flux=xover_flux,
            h_flux=h_flux_arr,
            membrane_V_drop=v_drop_arr,
            crossover_current_A_m2=j_xover_arr,
            purge_events=purge_events,
            fe_crossover_loss_pct=fe_loss_pct,
        )

    # ─── Purge criterion ──────────────────────────────────────────────
    def time_to_purge(
        self,
        current_efficiency: float = 0.85,
    ) -> float:
        """Estimate time until anolyte Fe³⁺ reaches the purge threshold (h).

        Assumes a simplified linear accumulation rate (valid for small Fe³⁺
        fractions where crossover is a minor correction).
        """
        j = self.j_A_m2
        A = self.electrode_area_m2
        V_an = self.anolyte.volume_L * 1e-3

        # Net Fe³⁺ accumulation rate (mol/s): production - crossover
        step = self.evaluate(0.0, 0.0)  # at zero Fe³⁺, worst case for production
        r_prod = j * A / FARADAY  # production at the anode
        r_xover = step.fe3_crossover_flux * A
        r_net = r_prod - r_xover  # mol/s

        if r_net <= 0:
            return float("inf")  # crossover exceeds production; never purges

        # Time to reach threshold from zero
        target_mol = self.purge_fe3_threshold_M * V_an * 1000.0  # mol
        t_s = target_mol / r_net
        return t_s / 3600.0

    # ─── Summary ──────────────────────────────────────────────────────
    def summary(self) -> dict:
        """Operating summary at the current conditions."""
        step = self.evaluate()
        t_h = self.h_transport_number()
        t_fe3 = self.fe3_transference_number()
        return {
            "membrane": self.membrane.name,
            "j (mA/cm²)": self.j_mA_cm2,
            "membrane_V_drop (V)": round(step.membrane_V_drop, 4),
            "Fe³⁺ crossover flux (mol/m²/s)": f"{step.fe3_crossover_flux:.3e}",
            "Fe³⁺ migration share (%)": round(100.0 * step.fe3_migration_flux / max(step.fe3_crossover_flux, 1e-30), 1),
            "Fe³⁺ equiv. current (A/m²)": round(step.fe3_crossover_current_A_m2, 4),
            "H⁺ flux (mol/m²/s)": f"{step.h_flux:.3e}",
            "t_H⁺": round(t_h, 4),
            "t_Fe³⁺": round(t_fe3, 4),
            "anolyte Fe³⁺ production (mol/s)": f"{step.anolyte_fe3_production_mol_s:.3e}",
            "time_to_purge (h)": round(self.time_to_purge(), 2),
        }


# ─── Convenience sweep ────────────────────────────────────────────────
def crossover_vs_current_density(
    membranes: list[MembraneSpec] | None = None,
    j_range_mA_cm2: np.ndarray | None = None,
    fe3_M: float = 0.1,
    temperature_C: float = 60.0,
) -> list[dict]:
    """Fe³⁺ crossover flux vs. current density for multiple membrane types.

    Returns a list of dicts with one entry per (membrane, j) combination.
    """
    if membranes is None:
        membranes = [NAFION_N117, FUMASEP_FKE50]
    if j_range_mA_cm2 is None:
        j_range_mA_cm2 = np.array([10.0, 50.0, 100.0, 200.0, 400.0])

    mems: list[MembraneSpec] = membranes if membranes is not None else [NAFION_N117, FUMASEP_FKE50]
    rows = []
    for mem in mems:
        for j in j_range_mA_cm2:
            model = MembraneTransportModel(
                membrane=mem,
                temperature_C=temperature_C,
                j_mA_cm2=float(j),
            )
            model.anolyte.fe3_M = fe3_M
            step = model.evaluate()
            rows.append(
                {
                    "membrane": mem.name,
                    "j_mA_cm2": float(j),
                    "fe3_flux_mol_m2_s": step.fe3_crossover_flux,
                    "fe3_migration_flux": step.fe3_migration_flux,
                    "fe3_diffusion_flux": step.fe3_diffusion_flux,
                    "crossover_current_A_m2": step.fe3_crossover_current_A_m2,
                    "membrane_V_drop": step.membrane_V_drop,
                    "h_transport_number": model.h_transport_number(),
                }
            )
    return rows
