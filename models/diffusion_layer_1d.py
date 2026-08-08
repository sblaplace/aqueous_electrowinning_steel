"""
1D diffusion-layer model for iron electrowinning — the FE prediction engine.

Full Nernst-Planck transport in a stagnant cathode film with:

* **Faradaic reactions** at the cathode:
    Fe²⁺ + 2 e⁻ → Fe(s)        (iron deposition)
    2 H⁺ + 2 e⁻ → H₂(g)        (proton-reduction HER; dominates in acid)
    2 H₂O + 2 e⁻ → H₂ + 2 OH⁻  (water-reduction HER; dominates at
                               high pH — see ``water_reduction_her``)
  The two HER branches share the same Nernst equilibrium (RHE) but have
  separate exchange currents and Tafel slopes, so alkaline/near-neutral
  baths are represented inside the model rather than outside its envelope.
* **Migration + diffusion** for six species:
    Fe²⁺ (z=+2), H⁺ (z=+1), OH⁻ (z=−1),
    HSO₄⁻ (z=−1), SO₄²⁻ (z=−2), H₃BO₃ (z=0), H₂BO₃⁻ (z=−1)
* **Homogeneous fast equilibria** (local, instantaneous):
    HSO₄⁻  ↔  H⁺ + SO₄²⁻      (Ka₂)
    H₃BO₃  ↔  H⁺ + H₂BO₃⁻     (Ka_b)
    H₂O    ↔  H⁺ + OH⁻         (Kw)
* **Arrhenius temperature correction** for all diffusivities
* **Butler-Volmer (Tafel) kinetics** at the electrode surface
* **Surface-state HER kinetics** (``surface_state=True``): the HER i₀
  is corrected for H coverage (Temkin), anion site-blocking (Cl⁻, SO₄²⁻,
  HSO₄⁻, borate), and facet ensemble using models.surface_state — the
  mechanism layer that predicts *why* chloride suppresses HER instead
  of having it as an exogenous scenario knob (Tier 1.1 from
  CHEM_PHYS_REVIEW.md).
* **FeSO₄⁰ contact-pair correction** (``fes04_pair_correction=True``):
  the effective free Fe²⁺ is reduced by the neutral-pair fraction
  (K ≈ 200 L/mol at 25 °C).  The pair does not migrate (z = 0), so
  j_lim is reduced by ~5–15% in 1.5 M FeSO₄ (Tier 2 from V2 review).
* **Surface pH** from proton flux balance
* **Fe(OH)₂ precipitation** criterion ([Fe²⁺][OH⁻]² / Ksp(T));
  Ksp is van 't Hoff corrected to the operating temperature (Fe(OH)₂ is
  more soluble at elevated T) via :func:`pourbaix.ksp_feoh2`.
* **Optional Pitzer activity correction** (``activity_model="pitzer"``):
  the Fe²⁺/Fe Nernst potential uses the multicomponent Pitzer
  activity coefficient γ_Fe²⁺(m, T) (valid at the multi-molal ionic
  strengths of real FeSO₄/Na₂SO₄ baths) instead of the ideal a = [Fe²⁺].
* **Outputs**:  FE(j, T, C_Fe²⁺, δ, pH, buffer), V_cell, surface_pH

Conserved transported variables
-------------------------------
The ODE state is ``[c_fe, c_h, c_s, c_b, phi]`` where
``c_s = C_HSO4 + C_SO4`` (total sulfate) and
``c_b = C_H3BO3 + C_H2BO3`` (total borate).
The proton invariant whose flux equals the HER consumption rate is
``Φ = C_H + C_HSO4 + C_H3BO3 − C_OH``.  When ``transport_na=True`` the
state gains a sixth variable ``c_na`` obeying a zero-flux boundary
(N_Na = 0 at the electrode); Na+ then develops a diffusion-migration
gradient across the film rather than being held at its bulk value.

Electroneutrality closes the system for ``dφ/dx``.

Units
-----
Concentrations are SI (mol/m³) internally, mol/L in the public API.
Current densities are positive cathodic magnitudes (A/m² internally,
mA/cm² in the API).  Potentials are V vs. SHE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .electrochemistry import E0_FE, FARADAY, R_GAS, Z_FE
from .kinetics import (
    EA_FE_DEPOSITION_J_MOL,
    EA_HER_ON_FE_J_MOL,
    FE_ANODIC_SLOPE_V,
    HER_ANODIC_SLOPE_V,
    I0_REF_K,
    arrhenius_i0,
)
from .pourbaix import LOGKSP_FEOH2, ksp_feoh2
from . import pitzer as _pitzer
from . import speciation as _speciation
from .thermodynamic_constants import (
    KA_HSO4_25,
    KA_BORIC_25,
    DH_HSO4_J_MOL,
    DH_BORIC_J_MOL,
    D_FE2_25,
    D_H_25,
    D_OH_25,
    D_HSO4_25,
    D_SO4_25,
    D_H3BO3_25,
    D_H2BO3_25,
    D_NA_25,
    DIFFUSION_EA_J_MOL,
)

# ─── Reference temperature ────────────────────────────────────────────
T_REF = 298.15  # K

# ─── Water autoprotolysis ─────────────────────────────────────────────
# Kw = 1e-14 (mol/L)² = 1e-14 × 1000² = 1e-8 (mol/m³)²
KW_SI = 1.0e-8

# ─── Bisulfate dissociation: HSO₄⁻ ↔ H⁺ + SO₄²⁻ ────────────────────
KA2_25C = KA_HSO4_25       # mol/L at 25 °C  (pKa ≈ 1.98)
KA2_EA_J_MOL = DH_HSO4_J_MOL  # J/mol (exothermic; Ka₂ decreases with T)

# ─── Boric acid: H₃BO₃ ↔ H⁺ + H₂BO₃⁻ ──────────────────────────────
KAB_25C = KA_BORIC_25      # mol/L at 25 °C  (pKa ≈ 9.24)
KAB_EA_J_MOL = DH_BORIC_J_MOL  # J/mol (endothermic; Ka_b increases with T)

# ─── Fe(OH)₂ solubility product ──────────────────────────────────────
KSP_FEOH2 = 10.0 ** LOGKSP_FEOH2  # (mol/L)³ at 25 °C; see Ksp property for T

# ─── Limiting ionic diffusivities at 25 °C (m²/s) ────────────────────
D_FE2 = D_FE2_25
D_H = D_H_25
D_OH = D_OH_25
D_HSO4 = D_HSO4_25
D_SO4 = D_SO4_25
D_H3BO3 = D_H3BO3_25
D_H2BO3 = D_H2BO3_25
D_NA = D_NA_25         # Na⁺ (shared with transport.py)

# Default activation energy for diffusion (J/mol)
DIFF_EA_J_MOL = DIFFUSION_EA_J_MOL

# ─── FeSO₄⁰ contact-pair correction ─────────────────────────────────
# Thermodynamic equilibrium constant for Fe²⁺ + SO₄²⁻ ↔ FeSO₄⁰(aq) at 25 °C.
# From speciation.py (K ≈ 200 L/mol at I = 0; falls to ~50 at I = 2 m).
# The pair is electrically neutral (z = 0), so it does not migrate in
# the Nernst-Planck film; it diffuses at ~60 % of the free Fe²⁺ rate.
K_FESO4_PAIR_25 = getattr(_speciation, 'K_FESO4_PAIR_25', 200.0)  # L/mol
D_FESO4_RATIO = 0.6  # D_FeSO4 / D_Fe2 (Stokes-Einstein, ~20% larger species)

# ─── Composition-dependent diffusivity D(γ, c) slope ──────────────
# Stokes-Einstein viscosity factor: log10(D/D0) = −k·I.  k anchored to the
# Lobo & Quaresma 1989 screening value for Fe²⁺ (0.05 per unit ionic
# strength) already used by fe_chloride_speciation.  See the
# ``composition_diffusivity`` flag docstring on DiffusionLayer1D.
D_FE2_COMPOSITION_SLOPE = 0.05


# ─── Helpers ──────────────────────────────────────────────────────────

def _diffusivity_T(D25: float, T: float, Ea: float = DIFF_EA_J_MOL) -> float:
    """Arrhenius temperature correction for diffusivity.

    D(T) = D(25 °C) · exp[ Ea/R · (1/298.15 − 1/T) ]
    """
    return D25 * np.exp(Ea / R_GAS * (1.0 / T_REF - 1.0 / T))


def _Ka_T(Ka25: float, T: float, Ea: float) -> float:
    """Temperature-corrected equilibrium constant (van 't Hoff)."""
    return Ka25 * np.exp(-Ea / R_GAS * (1.0 / T - 1.0 / T_REF))


# ─── Result data classes ─────────────────────────────────────────────

@dataclass
class FilmProfile:
    """Spatial profiles across the diffusion layer (x = 0 at cathode)."""

    x_m: np.ndarray
    fe_mol_m3: np.ndarray
    h_mol_m3: np.ndarray
    oh_mol_m3: np.ndarray
    hso4_mol_m3: np.ndarray
    so4_mol_m3: np.ndarray
    h3bo3_mol_m3: np.ndarray
    h2bo3_mol_m3: np.ndarray
    potential_V: np.ndarray
    depleted: bool
    na_mol_m3: np.ndarray = None  # only populated when transport_na=True
    Ksp_FeOH2: float = KSP_FEOH2  # (mol/L)³ at the film temperature

    # ── convenience views (mol/L) ───────────────────────────────────
    @property
    def fe_M(self) -> np.ndarray:
        return self.fe_mol_m3 / 1000.0

    @property
    def h_M(self) -> np.ndarray:
        return self.h_mol_m3 / 1000.0

    @property
    def oh_M(self) -> np.ndarray:
        return self.oh_mol_m3 / 1000.0

    @property
    def pH(self) -> np.ndarray:
        return -np.log10(np.maximum(self.h_M, 1e-30))

    @property
    def feoh2_supersaturation(self) -> np.ndarray:
        """[Fe²⁺][OH⁻]² / Ksp along the film; >1 means Fe(OH)₂ unstable."""
        return self.fe_M * self.oh_M ** 2 / self.Ksp_FeOH2

    @property
    def film_potential_drop_V(self) -> float:
        """φ(surface) − φ(bulk): the junction / diffusion potential."""
        return float(self.potential_V[0] - self.potential_V[-1])


@dataclass
class DiffusionLayerResult:
    """Operating point returned by :meth:`DiffusionLayer1D.solve`."""

    j_mA_cm2: float
    current_efficiency: float
    fe_current_A_m2: float
    her_current_A_m2: float
    surface_pH: float
    bulk_pH: float
    surface_fe_M: float
    surface_oh_M: float
    film_potential_drop_V: float
    precipitation_active: bool
    feoh2_supersaturation: float
    transport_limit_A_m2: float
    diffusion_limit_A_m2: float
    V_cathode_V: float
    V_cell: float
    temperature_C: float
    converged: bool
    profile: FilmProfile = field(repr=False)
    # Precipitation sink diagnostics (screening).  Zero when S≤1 everywhere
    # or when precipitation_sink is disabled.  The volumetric sink
    # R(x)=k·max(0,[Fe]−Ksp/[OH]²) is integrated over the film:
    precipitation_flux_mol_m2_s: float = 0.0
    precipitation_fraction: float = 0.0
    sludge_rate_g_m2_s: float = 0.0
    # Surface-state diagnostics (Tier 1.1 from CHEM_PHYS_REVIEW.md).
    # Zero when surface_state=False; otherwise the ratio of surface-state
    # corrected her_i0 to the bare her_i0 (< 1 means HER is suppressed).
    her_i0_surface_state_ratio: float = 1.0
    # FeSO₄⁰ contact-pair diagnostic (Tier 2 from V2 review).
    # Zero when fes04_pair_correction=False; otherwise the bulk pair
    # fraction (typically 0.15–0.25 in 1.5 M FeSO₄ at pH 2).
    fe_pair_fraction_bulk: float = 0.0
    # Composition-dependent diffusivity diagnostics (Tier 2.2).  1.0 when
    # composition_diffusivity=False; otherwise the Stokes-Einstein D factor
    # f = 10^(−k·I) applied to D_Fe²⁺ and the Pitzer bulk γ_Fe²⁺.
    d_composition_factor: float = 1.0
    gamma_fe_bulk: float = 1.0

    @property
    def fe_percent(self) -> float:
        return self.current_efficiency * 100.0

    @property
    def local_pH_rise(self) -> float:
        return self.surface_pH - self.bulk_pH


# ─── Main model ──────────────────────────────────────────────────────

@dataclass
class DiffusionLayer1D:
    """Steady 1D Nernst-Planck diffusion-layer model.

    Solves coupled transport of Fe²⁺, H⁺, total sulfate (HSO₄⁻ + SO₄²⁻),
    and total borate (H₃BO₃ + H₂BO₃⁻) across a cathode film, with
    Butler-Volmer kinetics and fast homogeneous equilibria.

    Parameters
    ----------
    fe_conc_M : float
        Bulk Fe²⁺ concentration (mol/L).
    pH_bulk : float
        Bulk electrolyte pH.
    temperature_C : float
        Electrolyte temperature (°C).
    delta_m : float
        Diffusion layer (Nernst film) thickness (m).
    buffer_conc_M : float
        Boric acid concentration (mol/L). 0 = no buffer.
    support_conc_M : float
        Na₂SO₄ supporting electrolyte (mol/L). Adds Na⁺ for charge balance
        and collapses the migration field at high concentrations.
    fe_i0, her_i0 : float
        Exchange current densities (A/m²).
    fe_tafel_V, her_tafel_V : float
        Cathodic Tafel slopes (V/decade).
    E_anode_eq : float
        Anode equilibrium potential (V vs. SHE).
    eta_anode_V : float
        Anode overpotential (V).
    ir_drop_V : float
        Ohmic drop across electrolyte, membrane, contacts (V).
    grid_points : int
        Spatial grid resolution for the ODE integration.
    fast_mode : bool
        If ``True``, use relaxed solver tolerances (looser ODE rtol, fewer
        Picard iterations, looser brentq).  Yields FE/V_cell accurate to a
        few hundredths of a percent at 10-20x lower cost — intended for
        screening loops (sensitivity / optimisation) where the tight solve
        would be prohibitively slow.  Default ``False`` (unchanged, tight).
    """

    fe_conc_M: float = 1.0
    pH_bulk: float = 2.0
    temperature_C: float = 60.0
    delta_m: float = 50.0e-6
    buffer_conc_M: float = 0.40
    support_conc_M: float = 0.0
    fe_i0: float = 10.0
    her_i0: float = 0.010
    fe_tafel_V: float = 0.120
    her_tafel_V: float = 0.140
    # Water-reduction HER branch (2 H₂O + 2e⁻ → H₂ + 2 OH⁻).  Disabled
    # by default for back-compat; set water_reduction_her=True for
    # alkaline/near-neutral baths.  Its exchange current is per unit
    # water activity (≈1) and its cathodic Tafel slope is typically
    # ~120 mV/dec on Fe.
    water_reduction_her: bool = False
    her_water_i0: float = 1.0e-5
    her_water_tafel_V: float = 0.120
    # Transport Na+ as a zero-flux species (N_Na = 0 at the electrode).
    # When False Na+ is held at its bulk concentration (the prior
    # approximation); when True a concentration gradient develops across
    # the film and Na+ migration contributes to the potential — the
    # correct treatment for a supporting electrolyte in a stagnant film.
    transport_na: bool = False
    E_anode_eq: float = 1.229
    eta_anode_V: float = 0.40
    ir_drop_V: float = 0.20
    grid_points: int = 101
    max_iterations: int = 200
    convergence_tol: float = 1e-9
    fast_mode: bool = False
    _transport_limit_cache: dict = field(default_factory=dict, repr=False, compare=False)
    # Exchange-current densities are anchored at kinetics_ref_K and
    # Arrhenius-scaled to the operating temperature; see models/kinetics.py.
    fe_i0_Ea_J_mol: float = EA_FE_DEPOSITION_J_MOL
    her_i0_Ea_J_mol: float = EA_HER_ON_FE_J_MOL
    kinetics_ref_K: float = I0_REF_K
    # Butler-Volmer anodic-branch slopes (cathodic slopes above retained).
    fe_anodic_slope_V: float = FE_ANODIC_SLOPE_V
    her_anodic_slope_V: float = HER_ANODIC_SLOPE_V
    # Precipitation sink (Fe(OH)₂).  When ``precipitation_sink=True`` the
    # film result carries a first-order sludge estimate: at any position
    # where S=[Fe][OH]²/Ksp>1 the excess Fe is assigned a volumetric
    # sink  R = k_precip · ( [Fe] − Fe_sat )  with Fe_sat=Ksp/[OH]².
    # This is a screening diagnostic that closes the iron ledger outside
    # the ODE (the ODE itself remains precipitation-free) — it gives the
    # inventory loss and the acidifying 2 H⁺ per Fe precipitated without
    # yet solving the fully coupled diffusion-reaction film (see
    # SIM_PHYSICS_UPGRADE.md § next).  Set k_precip=0 to recover the
    # previous diagnostic-only behaviour.
    precipitation_sink: bool = True
    k_precip_s: float = 1.0e1  # 1/s — fast, drives S→1 on the film timescale δ²/D≈3 s
    # Activity-coefficient model for the Fe²⁺/Fe Nernst potential:
    #   "ideal"  — a = [Fe²⁺] (mol/L), the pre-2026-08 default;
    #   "pitzer" — a_Fe²⁺ = γ_Fe²⁺(m, T) · [Fe²⁺], with γ from the
    #              multicomponent Pitzer model (valid to the multi-molal
    #              ionic strengths of real FeSO₄/Na₂SO₄ baths).  At the
    #              reference bath γ_Fe²⁺ ≈ 0.05–0.15, shifting E_eq(Fe) by
    #              −20 to −40 mV — directly comparable to the HER margin
    #              the whole program hinges on.  Gamma is now evaluated
    #              at the *surface* composition in the Picard Nernst
    #              correction (local ionic-strength dependence), with bulk
    #              γ as the seed and a per-composition cache for speed.
    activity_model: str = "ideal"
    # Surface-state HER kinetics (Tier 1.1 from CHEM_PHYS_REVIEW.md).
    # When True, the HER exchange current density is corrected for
    # hydrogen coverage (Temkin), anion site-blocking, and facet
    # ensemble using models.surface_state.SurfaceStateKinetics.  The
    # correction replaces the constant her_i0 with an effective i₀,H
    # that depends on η, bath composition, and facet mix — i.e. it
    # predicts why Cl⁻ suppresses HER instead of having that as an
    # exogenous scenario knob.  Default False for backward compatibility.
    surface_state: bool = False
    bath_type: str = "sulfate"   # "sulfate", "aware", or "mixed"
    # FeSO₄⁰ neutral contact-pair correction (Tier 2 from V2 review).
    # When True, the effective free Fe²⁺ in the Nernst-Planck film is
    # reduced by the FeSO₄⁰ pair fraction (computed from the local SO₄²⁻
    # concentration via speciation.K_FESO4_PAIR_25).  The neutral pair
    # does not migrate (z = 0), so only the free Fe²⁺ contributes to
    # the migration-enhanced transport; the pair diffuses but at a
    # lower diffusivity (D_FeSO4 ≈ 0.6 × D_Fe2).  The net effect is a
    # small reduction in j_lim (typically 5–15% in 1.5 M FeSO₄).
    fes04_pair_correction: bool = False
    # Composition-dependent Fe²⁺ diffusivity D(γ, c) (Tier 2.2 from
    # CHEM_PHYS_REVIEW.md).  When True, the infinite-dilution/Arrhenius
    # D_Fe²⁺ is scaled by a Stokes-Einstein viscosity factor that accounts
    # for the real bath composition — in 1.5 M FeSO₄ + 0.5 M Na₂SO₄ the
    # physical D is 30–50 % below its infinite-dilution value (Lobo &
    # Quaresma).  The factor is applied across the whole Nernst-Planck
    # film (both the diffusion-only limit and the reactive ODE), so it
    # lowers both j_lim and the migration-enhanced transport limit.
    #
    #   D_Fe2+_eff(c) = D_Fe2+_T(T) · 10^(−k · I(c))
    #
    # with I(c) the bulk ionic strength from ``_estimate_ionic_strength``
    # (the same composition signal the FeSO₄⁰ pair correction uses) and
    # k = D_COMPOSITION_SLOPE.  The slope is anchored to the Lobo &
    # Quaresma Fe²⁺/NaClO₄/Cl⁻ screening value already used by
    # ``fe_chloride_speciation.fe2_diffusivity_in_chloride_bath``, so the
    # reference-bath reduction lands at ~46 % (inside the cited 30–50 %).
    # Pitzer γ_Fe²⁺ is surfaced as a companion diagnostic (see
    # ``gamma_fe_bulk``) but the closure itself is viscosity-driven on the
    # ionic strength, not a direct γ power-law.
    #
    # Backward compatible: default False leaves D_Fe²⁺ exactly as before.
    composition_diffusivity: bool = False
    # Stokes-Einstein viscosity slope, log10(D/D0) per unit ionic strength
    # (Lobo & Quaresma 1989 screening value for the Fe²⁺ system).
    d_composition_slope: float = 0.05

    def __post_init__(self) -> None:
        if self.fe_conc_M <= 0.0:
            raise ValueError("fe_conc_M must be positive")
        if self.delta_m <= 0.0:
            raise ValueError("delta_m must be positive")
        if self.grid_points < 3:
            raise ValueError("grid_points must be at least 3")
        if self.activity_model not in ("ideal", "pitzer"):
            raise ValueError(
                f"activity_model must be 'ideal' or 'pitzer', got "
                f"{self.activity_model!r}"
            )
        # Cache the composition-dependent Pitzer γ so the per-iteration
        # Nernst correction does not re-solve Pitzer dozens of times.
        self._gamma_fe_cache: float | None = None
        self._gamma_at_cache: dict = {}
        # Bulk Na+ concentration (mol/m3), used as the boundary value and
        # (when transport_na is False) held constant across the film.
        self._c_na_bulk: float = 2.0 * self.support_conc_M * 1000.0

    # ─── Temperature-dependent properties ───────────────────────────

    @property
    def T(self) -> float:
        return self.temperature_C + 273.15

    @property
    def f_RT(self) -> float:
        """F / RT  (1/V)."""
        return FARADAY / (R_GAS * self.T)

    @property
    def Ka2(self) -> float:
        """Bisulfate Ka₂ in mol/m³ at operating T."""
        return _Ka_T(KA2_25C, self.T, KA2_EA_J_MOL) * 1000.0

    @property
    def Ka_b(self) -> float:
        """Boric acid Ka_b in mol/m³ at operating T."""
        return _Ka_T(KAB_25C, self.T, KAB_EA_J_MOL) * 1000.0

    @property
    def D_fe(self) -> float:
        """Fe²⁺ diffusivity at the operating T (and composition if enabled).

        Base value is the Arrhenius-corrected infinite-dilution D.  When
        ``composition_diffusivity`` is True, it is further scaled by the
        Stokes-Einstein viscosity factor 10^(−k·I) so the film transport
        reflects the real (30–50 % lower) mobility in a concentrated
        sulfate bath.  Returning a *scalar* keeps the ODE solve untouched —
        the same reduced D flows into both the diffusion limit and the
        reactive Nernst-Planck integration.
        """
        base = _diffusivity_T(D_FE2, self.T)
        if self.composition_diffusivity:
            base = base * self._composition_diffusivity_factor()
        return base

    @property
    def D_h(self) -> float:
        return _diffusivity_T(D_H, self.T)

    @property
    def D_oh(self) -> float:
        return _diffusivity_T(D_OH, self.T)

    @property
    def D_hso4(self) -> float:
        return _diffusivity_T(D_HSO4, self.T)

    @property
    def D_so4(self) -> float:
        return _diffusivity_T(D_SO4, self.T)

    @property
    def D_h3bo3(self) -> float:
        return _diffusivity_T(D_H3BO3, self.T)

    @property
    def D_h2bo3(self) -> float:
        return _diffusivity_T(D_H2BO3, self.T)

    @property
    def D_na(self) -> float:
        return _diffusivity_T(D_NA, self.T)

    @property
    def Ksp(self) -> float:
        """Fe(OH)₂ solubility product (mol/L)³ at the operating temperature.

        Fe(OH)₂ is more soluble at elevated T (endothermic dissolution), so
        the precipitation boundary shifts to higher pH as the bath warms.
        See :func:`pourbaix.ksp_feoh2`.
        """
        return float(ksp_feoh2(self.T))

    @property
    def gamma_fe(self) -> float:
        """Bulk Fe²⁺ activity coefficient (Pitzer) at the operating T.

        Returns 1.0 for ``activity_model="ideal"``.  For ``"pitzer"`` the
        multicomponent Pitzer model is evaluated once at the bulk bath
        composition (FeSO₄ + Na₂SO₄ support + H₂SO₄ at pH_bulk), giving the
        strong 2–2-salt non-ideality (γ_Fe²⁺ ≈ 0.05–0.15 at 1 M) that the
        ideal model ignores.  Used in the Fe²⁺/Fe Nernst potential.
        """
        if self.activity_model == "ideal":
            return 1.0
        if self._gamma_fe_cache is not None:
            return self._gamma_fe_cache
        self._gamma_fe_cache = self._gamma_at(self.fe_conc_M, self.pH_bulk)
        return self._gamma_fe_cache

    def _gamma_at(self, fe_M: float, pH: float) -> float:
        """Fe²⁺ activity coefficient at an arbitrary local composition.

        Re-evaluates the Pitzer model at ``fe_M`` and ``pH`` while holding
        supporting electrolyte and temperature fixed.  This upgrades the
        previous bulk-only gamma into a surface-correct Nernst term: as
        Fe²⁺ depletes toward the cathode, ionic strength falls and γ
        rises, shifting E_eq positive by 5–15 mV at 50–70% depletion —
        directly comparable to the HER margin the program hinges on.

        A small per-composition cache keeps the Picard loop cheap (Pitzer
        solve is µs, but called dozens of times per operating point).
        """
        if self.activity_model == "ideal":
            return 1.0
        cache: dict = getattr(self, "_gamma_at_cache", None)
        if cache is None:
            cache = {}
            object.__setattr__(self, "_gamma_at_cache", cache)
        key = (round(float(fe_M), 4), round(float(pH), 2))
        if key in cache:
            return float(cache[key])
        rho = _pitzer.water_density_kg_L(self.temperature_C)
        solute_kg_L = float(fe_M) * 151.908e-3 + self.support_conc_M * 142.04e-3
        kg_water_per_L = max(rho - solute_kg_L, 0.3)
        m_fe = max(float(fe_M), 1e-6) / kg_water_per_L
        m_na = 2.0 * self.support_conc_M / kg_water_per_L
        m_so4 = (float(fe_M) + self.support_conc_M) / kg_water_per_L
        c_h_mol_m3 = max(10.0 ** (-float(pH)) * 1000.0, 1e-10)
        m_h = c_h_mol_m3 / 1000.0 / kg_water_per_L
        composition = {
            "Fe2+": m_fe, "Na+": m_na, "H+": m_h,
            "SO4-2": m_so4 + 0.5 * m_h,
        }
        try:
            sol = _pitzer.solve_pitzer(composition, T_C=self.temperature_C)
            gamma = float(sol.gamma.get("Fe2+", 1.0))
        except Exception:
            gamma = float(self._gamma_fe_cache) if self._gamma_fe_cache is not None else 1.0
        gamma = max(0.02, min(gamma, 1.0))
        cache[key] = gamma
        return gamma

    # ─── FeSO₄⁰ contact-pair fraction ───────────────────────────────

    def _fe_pair_fraction(self, c_so4_mol_m3: float) -> float:
        """Fraction of total Fe²⁺ bound as FeSO₄⁰ neutral pair.

        K_eff = [FeSO₄⁰] / ([Fe²⁺][SO₄²⁻]) where K_eff is the
        *activity-corrected* (concentration-scale) equilibrium constant.
        The thermodynamic K_FESO4_PAIR_25 ≈ 200 L/mol at I = 0; at
        the program's working ionic strength (I ~ 5 M) the activity
        coefficients of Fe²⁺ and SO₄²⁻ are both << 1 (Pitzer: γ_Fe²⁺
        ~ 0.05, γ_SO₄²⁻ ~ 0.1 at the reference bath), so K_eff drops
        by ~3 orders of magnitude to ~0.4 L/mol.  We approximate this
        with an ionic-strength-dependent correction:

            K_eff = K_thermo × exp(-α × I)

        where α = 1.18 is tuned to reproduce the speciation module's
        24 % pair fraction at the reference bath (1.5 M FeSO₄ +
        0.5 M Na₂SO₄, I ≈ 5.4 M from the bulk electroneutrality
        solve; c_SO₄²⁻ ≈ 0.9 M from the HSO₄⁻/SO₄²⁻ equilibrium
        at pH 2, 60 °C).
        """
        if not self.fes04_pair_correction:
            return 0.0
        c_so4_M = c_so4_mol_m3 / 1000.0  # mol/L
        # Ionic strength screening estimate from the bulk composition
        I = self._estimate_ionic_strength()
        K_eff = K_FESO4_PAIR_25 * math.exp(-1.18 * I)
        Kc = K_eff * max(c_so4_M, 1e-10)
        return Kc / (1.0 + Kc)

    def _estimate_ionic_strength(self) -> float:
        """Screening ionic strength (M) from bulk composition.

        I = 0.5 × Σ c_i × z_i².  Includes Fe²⁺, Na⁺, SO₄²⁻, HSO₄⁻,
        H⁺.  Uses the bulk pH to split sulfate into HSO₄⁻/SO₄²⁻.
        """
        c_fe = self.fe_conc_M
        c_na = 2.0 * self.support_conc_M
        c_h = 10.0 ** (-self.pH_bulk)
        c_total_so4 = c_fe + self.support_conc_M  # total sulfate
        ka2 = _Ka_T(KA2_25C, self.T, KA2_EA_J_MOL)
        f_so4 = ka2 / (ka2 + c_h)
        c_so4 = c_total_so4 * f_so4
        c_hso4 = c_total_so4 * (1 - f_so4)
        I = 0.5 * (c_fe * 4 + c_na * 1 + c_so4 * 4 + c_hso4 * 1 + c_h * 1)
        return max(I, 0.01)  # floor at 0.01 to avoid K_eff = K_thermo

    # ─── Composition-dependent diffusivity D(γ, c) ───────────────────

    def _composition_diffusivity_factor(self) -> float:
        """Stokes-Einstein viscosity factor on D_Fe²⁺ (0 < f ≤ 1).

        f = 10^(−k · I), with I the bulk ionic strength and k the
        Stokes-Einstein viscosity slope (Lobo & Quaresma).  Returns 1.0
        when ``composition_diffusivity`` is False (backward compatible),
        or when the ionic strength is negligible (dilute → f → 1).

        This is the gamma/activity-coupled composition signal: Fe²⁺
        mobility in a concentrated FeSO₄/Na₂SO₄ bath is 30–50 % lower
        than the infinite-dilution value (CHEM_PHYS_REVIEW Tier 2.2).
        """
        if not self.composition_diffusivity:
            return 1.0
        k = self.d_composition_slope if self.d_composition_slope > 0 else D_FE2_COMPOSITION_SLOPE
        I = self._estimate_ionic_strength()
        return float(10.0 ** (-k * I))

    @property
    def _bulk_pair_fraction(self) -> float:
        """FeSO₄⁰ pair fraction at bulk composition."""
        if not self.fes04_pair_correction:
            return 0.0
        fr = self._fractions(self._bulk_c_h, self.bulk_c_s, self.buffer_conc_M * 1000.0)
        return self._fe_pair_fraction(fr["c_so4"])

    @property
    def _effective_D_fe(self) -> float:
        """Effective Fe diffusivity accounting for FeSO₄⁰ pair.

        The total Fe flux is the sum of free-Fe²⁺ (D_Fe2, migrates)
        and pair (D_FeSO4, does NOT migrate).  At the transport limit
        (surface Fe → 0), the pair concentration also → 0, so the
        effective diffusivity weighting is:

            D_eff = D_Fe2 · (1 - f_pair) + D_FeSO4 · f_pair
                  = D_Fe2 · (1 - f_pair · (1 - D_FESO4_RATIO))

        This reduces j_lim by f_pair · (1 - 0.6) ≈ 0.4 · f_pair.
        Only the free Fe²⁺ feels migration; the pair's contribution
        to j_lim is pure diffusion.
        """
        if not self.fes04_pair_correction:
            return self.D_fe
        f = self._bulk_pair_fraction
        return self.D_fe * (1.0 - f * (1.0 - D_FESO4_RATIO))

    # ─── Surface-state kinetics wrapper ─────────────────────────────

    def _surface_state_her_i0(self, eta_her_V: float) -> float:
        """HER i₀ corrected by surface-state anion site-blocking.

        Returns the effective exchange current at this HER overpotential.
        Falls back to the bare Arrhenius-corrected value when
        ``surface_state`` is False.

        Scope
        -----
        This is a *mechanism scaffold*, not a calibrated prediction.
        ``models.surface_state`` is flagged ``SCREENING_FLAG =
        "unvalidated (L1)"`` — no data has been fit.  Within that
        sandbox the correction below is the *dominant, robust*
        suppression mechanism: anion site-blocking, ``1 - theta_block``.

        We deliberately do **not** multiply by the Temkin ``θ_H·(1−θ_H)``
        term.  At the program's operating overpotential ``θ_H → 1``
        (strong H binding on Fe(110), ΔG_H* ≈ −0.40 eV) that term
        collapses to ~0, which drives HER to zero and FE to 1 in
        *every* bath — the exact behaviour this wiring rejects (it would
        contradict the H₂ generation the sister h2_safety module models,
        and erase the chloride-vs-sulfate distinction at the cell level).
        Binding i₀,H to the anion-free-site fraction keeps HER
        non-degenerate while preserving the real ordering (chloride
        suppresses more than sulfate).  Calibrate against Phase I
        measurements before quoting FE to a published number.
        """
        if not self.surface_state:
            return self.her_i0_T
        from .surface_state import SurfaceCoverage, chloride_aware_default
        facets, anions = chloride_aware_default(self.bath_type)
        ss = SurfaceCoverage(
            eta_V=eta_her_V, T_K=self.T,
            facets=facets, adsorbed_anions=anions,
        )
        return self.her_i0_T * (1.0 - ss.theta_block)

    # ─── Bulk composition ──────────────────────────────────────────

    @property
    def _bulk_c_h(self) -> float:
        """Bulk H⁺ in mol/m³."""
        return 10.0 ** (-self.pH_bulk) * 1000.0

    @property
    def bulk_c_s(self) -> float:
        """Total sulfate (mol/m³) from FeSO₄ + H₂SO₄, satisfying electroneutrality."""
        c_fe = self.fe_conc_M * 1000.0
        c_h = self._bulk_c_h
        c_oh = KW_SI / c_h
        c_b = self.buffer_conc_M * 1000.0
        c_na = 2.0 * self.support_conc_M * 1000.0
        ka2 = self.Ka2
        kab = self.Ka_b

        f_so4 = ka2 / (ka2 + c_h)
        f_h2bo3 = kab / (kab + c_h)

        # Electroneutrality:  2·Fe + H + Na = c_s·(1 + f_so4) + c_b·f_h2bo3 + OH
        cation = 2.0 * c_fe + c_h + c_na
        anion_fixed = c_b * f_h2bo3 + c_oh
        denom = 1.0 + f_so4  # = f_hso4 + 2·f_so4
        c_s = (cation - anion_fixed) / denom
        return max(c_s, c_fe)  # at least as much sulfate as from FeSO₄

    @property
    def diffusion_limit_A_m2(self) -> float:
        """Pure-diffusion (Levich) limiting current for Fe²⁺ (A/m²).

        When ``fes04_pair_correction=True``, uses the effective Fe
        diffusivity that accounts for the FeSO₄⁰ neutral pair (which
        diffuses but does not migrate).
        """
        D_eff = self._effective_D_fe
        return Z_FE * FARADAY * D_eff * self.fe_conc_M * 1000.0 / self.delta_m

    @property
    def fe_i0_T(self) -> float:
        """Fe exchange current density Arrhenius-scaled to T."""
        return arrhenius_i0(self.fe_i0, self.T, self.fe_i0_Ea_J_mol, self.kinetics_ref_K)

    @property
    def her_i0_T(self) -> float:
        """HER exchange current density Arrhenius-scaled to T."""
        return arrhenius_i0(self.her_i0, self.T, self.her_i0_Ea_J_mol, self.kinetics_ref_K)

    # ─── Equilibrium fractions ─────────────────────────────────────

    def _fractions(self, c_h: float, c_s: float, c_b: float,
                   c_na: float | None = None) -> dict:
        """Individual species concentrations and equilibrium derivatives."""
        ka2 = self.Ka2
        kab = self.Ka_b
        denom_s = ka2 + c_h
        denom_b = kab + c_h

        f_hso4 = c_h / denom_s
        f_so4 = ka2 / denom_s
        f_h3bo3 = c_h / denom_b
        f_h2bo3 = kab / denom_b

        # d(f_hso4)/d(c_h) = −d(f_so4)/d(c_h) = ka2 / (ka2 + c_h)²
        g1 = ka2 / (denom_s * denom_s)
        # d(f_h3bo3)/d(c_h) = −d(f_h2bo3)/d(c_h) = kab / (kab + c_h)²
        g2 = kab / (denom_b * denom_b)

        if c_na is None:
            c_na = self._c_na_bulk
        return {
            "c_hso4": f_hso4 * c_s,
            "c_so4": f_so4 * c_s,
            "c_h3bo3": f_h3bo3 * c_b,
            "c_h2bo3": f_h2bo3 * c_b,
            "c_oh": KW_SI / c_h,
            "c_na": c_na,
            "f_hso4": f_hso4,
            "f_so4": f_so4,
            "f_h3bo3": f_h3bo3,
            "f_h2bo3": f_h2bo3,
            "g1": g1,
            "g2": g2,
        }

    # ─── ODE right-hand side ───────────────────────────────────────

    def _rhs(self, _x: float, y: np.ndarray, n_fe: float, n_prot: float):
        """d/dx of [c_fe, c_h, c_s, c_b, phi] at fixed species fluxes.

        Integrates from bulk (x = δ) to surface (x = 0).

        Parameters
        ----------
        n_fe : float
            Fe²⁺ flux = −i_Fe / (2F)  (< 0, toward electrode).
        n_prot : float
            Proton invariant flux = −i_HER / F  (< 0).
        """
        c_fe = max(y[0], 1e-10)
        c_h = max(y[1], 1e-20)
        c_s = max(y[2], 1e-10)
        c_b = max(y[3], 0.0)

        fr = self._fractions(c_h, c_s, c_b)
        d_fe, d_h, d_oh = self.D_fe, self.D_h, self.D_oh
        d_hso4, d_so4 = self.D_hso4, self.D_so4
        d_h3bo3, d_h2bo3 = self.D_h3bo3, self.D_h2bo3
        f = self.f_RT
        g1, g2 = fr["g1"], fr["g2"]
        kw_c2 = KW_SI / (c_h * c_h)  # d(c_OH)/d(c_H) magnitude

        # ── Build 5×5 linear system  A · u = b ─────────────────────
        #    u = [dc_fe, dc_h, dc_s, dc_b, dphi]
        A = np.zeros((5, 5))
        b = np.zeros(5)

        # 1.  N_Fe = n_fe   (Fe²⁺ flux, z = +2)
        A[0, 0] = -d_fe
        A[0, 4] = -2.0 * d_fe * f * c_fe
        b[0] = n_fe

        # 2.  N_S = N_HSO4 + N_SO4 = 0   (total sulfate conserved)
        A[1, 1] = (d_so4 - d_hso4) * c_s * g1
        A[1, 2] = -(d_hso4 * fr["f_hso4"] + d_so4 * fr["f_so4"])
        A[1, 4] = f * (d_hso4 * fr["c_hso4"] + 2.0 * d_so4 * fr["c_so4"])
        b[1] = 0.0

        # 3.  N_B = N_H3BO3 + N_H2BO3 = 0   (total borate conserved)
        A[2, 1] = (d_h2bo3 - d_h3bo3) * c_b * g2
        A[2, 3] = -(d_h3bo3 * fr["f_h3bo3"] + d_h2bo3 * fr["f_h2bo3"])
        A[2, 4] = f * d_h2bo3 * fr["c_h2bo3"]  # H₃BO₃ is neutral
        b[2] = 0.0

        # 4.  N_Φ = N_H + N_HSO4 + N_H3BO3 − N_OH = n_prot
        #     (proton invariant flux set by HER rate)
        A[3, 1] = (
            -d_h
            - d_hso4 * c_s * g1
            - d_h3bo3 * c_b * g2
            - d_oh * kw_c2
        )
        A[3, 2] = -d_hso4 * fr["f_hso4"]
        A[3, 3] = -d_h3bo3 * fr["f_h3bo3"]
        A[3, 4] = f * (
            -d_h * c_h + d_hso4 * fr["c_hso4"] - d_oh * KW_SI / c_h
        )
        b[3] = n_prot

        # 5.  Electroneutrality (differentiated)
        #     2·c_fe + c_h + c_na = c_hso4 + 2·c_so4 + c_h2bo3 + c_oh
        c_na = fr["c_na"]
        A[4, 0] = 2.0
        A[4, 1] = 1.0 + c_s * g1 + c_b * g2 + kw_c2
        A[4, 2] = fr["f_hso4"] - 2.0  # = −(f_hso4 + 2·f_so4) + f_hso4  ... = -(2 - f_hso4)
        A[4, 3] = -fr["f_h2bo3"]
        A[4, 4] = 0.0
        b[4] = 0.0

        try:
            u = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.zeros(5)
        return u

    # ─── Film integration ──────────────────────────────────────────

    def _rhs_na(self, _x: float, y: np.ndarray, n_fe: float, n_prot: float):
        """6-variable RHS with Na+ as a transported zero-flux species.

        State ``[c_fe, c_h, c_s, c_b, c_na, phi]``.  Na+ obeys
        N_Na = 0 at the electrode (it is not consumed there), so a
        diffusion-migration gradient develops across the film.
        """
        c_fe = max(y[0], 1e-10)
        c_h = max(y[1], 1e-20)
        c_s = max(y[2], 1e-10)
        c_b = max(y[3], 0.0)
        c_na = max(y[4], 0.0)

        fr = self._fractions(c_h, c_s, c_b, c_na)
        d_fe, d_h, d_oh = self.D_fe, self.D_h, self.D_oh
        d_hso4, d_so4 = self.D_hso4, self.D_so4
        d_h3bo3, d_h2bo3, d_na = self.D_h3bo3, self.D_h2bo3, self.D_na
        f = self.f_RT
        g1, g2 = fr["g1"], fr["g2"]
        kw_c2 = KW_SI / (c_h * c_h)

        # 6×6 system, u = [dc_fe, dc_h, dc_s, dc_b, dc_na, dphi]
        A = np.zeros((6, 6))
        b = np.zeros(6)

        # 1. N_Fe = n_fe
        A[0, 0] = -d_fe
        A[0, 5] = -2.0 * d_fe * f * c_fe
        b[0] = n_fe

        # 2. N_S = 0 (total sulfate)
        A[1, 1] = (d_so4 - d_hso4) * c_s * g1
        A[1, 2] = -(d_hso4 * fr["f_hso4"] + d_so4 * fr["f_so4"])
        A[1, 5] = f * (d_hso4 * fr["c_hso4"] + 2.0 * d_so4 * fr["c_so4"])
        b[1] = 0.0

        # 3. N_B = 0 (total borate)
        A[2, 1] = (d_h2bo3 - d_h3bo3) * c_b * g2
        A[2, 3] = -(d_h3bo3 * fr["f_h3bo3"] + d_h2bo3 * fr["f_h2bo3"])
        A[2, 5] = f * d_h2bo3 * fr["c_h2bo3"]
        b[2] = 0.0

        # 4. proton invariant flux = n_prot
        A[3, 1] = -d_h - d_hso4*c_s*g1 - d_h3bo3*c_b*g2 - d_oh*kw_c2
        A[3, 2] = -d_hso4 * fr["f_hso4"]
        A[3, 3] = -d_h3bo3 * fr["f_h3bo3"]
        A[3, 5] = f * (-d_h*c_h + d_hso4*fr["c_hso4"] - d_oh*KW_SI/c_h)
        b[3] = n_prot

        # 5. N_Na = 0 (supporting cation, no electrode reaction)
        A[4, 4] = -d_na
        A[4, 5] = -1.0 * d_na * f * c_na
        b[4] = 0.0

        # 6. differentiated electroneutrality (c_na now variable)
        #    2 c_fe + c_h + c_na = c_hso4 + 2 c_so4 + c_h2bo3 + c_oh
        A[5, 0] = 2.0
        A[5, 1] = 1.0 + c_s*g1 + c_b*g2 + kw_c2
        A[5, 2] = fr["f_hso4"] - 2.0
        A[5, 3] = -fr["f_h2bo3"]
        A[5, 4] = 1.0
        A[5, 5] = 0.0
        b[5] = 0.0

        try:
            u = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.zeros(6)
        return u


    def integrate(self, i_fe_A_m2: float, i_her_A_m2: float,
                  grid_points: int | None = None,
                  rtol: float = 1e-8, atol: float = 1e-10,
                  ) -> FilmProfile | None:
        """Integrate the film from bulk edge to electrode surface.

        Parameters
        ----------
        i_fe_A_m2, i_her_A_m2 : float
            Positive cathodic partial current densities (A/m²).
        grid_points : int, optional
            Override spatial grid resolution (default ``self.grid_points``).
        rtol, atol : float
            ODE solver tolerances (relaxed during Picard iteration).
        """
        if i_fe_A_m2 < 0.0 or i_her_A_m2 < 0.0:
            raise ValueError("partial current densities must be non-negative")

        gp = grid_points if grid_points is not None else self.grid_points

        n_fe = -i_fe_A_m2 / (Z_FE * FARADAY)
        n_prot = -i_her_A_m2 / FARADAY

        c_fe_0 = self.fe_conc_M * 1000.0
        c_h_0 = self._bulk_c_h
        c_s_0 = self.bulk_c_s
        c_b_0 = self.buffer_conc_M * 1000.0

        if self.transport_na:
            y0 = np.array([c_fe_0, c_h_0, c_s_0, c_b_0, self._c_na_bulk, 0.0])
            rhs = self._rhs_na
        else:
            y0 = np.array([c_fe_0, c_h_0, c_s_0, c_b_0, 0.0])
            rhs = self._rhs
        x_eval = np.linspace(self.delta_m, 0.0, gp)

        sol = solve_ivp(
            rhs,
            (self.delta_m, 0.0),
            y0,
            t_eval=x_eval,
            args=(n_fe, n_prot),
            method="LSODA",
            rtol=rtol,
            atol=atol,
        )
        if not sol.success:
            return None

        # Flip so index 0 = electrode surface, index −1 = bulk.
        c_fe = sol.y[0][::-1]
        c_h = sol.y[1][::-1]
        c_s = sol.y[2][::-1]
        c_b = sol.y[3][::-1]
        if self.transport_na:
            c_na = sol.y[4][::-1]
            phi = sol.y[5][::-1]
        else:
            c_na = np.full_like(c_fe, self._c_na_bulk)
            phi = sol.y[4][::-1]

        floor_fe = 1e-8 * c_fe_0
        depleted = bool(np.min(c_fe) <= floor_fe * 1.01)
        c_fe = np.maximum(c_fe, floor_fe)
        c_h = np.maximum(c_h, 1e-20)
        c_oh = KW_SI / c_h

        # Derived species at each grid point
        ka2, kab = self.Ka2, self.Ka_b
        f_hso4 = c_h / (ka2 + c_h)
        f_so4 = ka2 / (ka2 + c_h)
        f_h3bo3 = c_h / (kab + c_h)
        f_h2bo3 = kab / (kab + c_h)

        return FilmProfile(
            x_m=sol.t[::-1],
            fe_mol_m3=c_fe,
            h_mol_m3=c_h,
            oh_mol_m3=c_oh,
            hso4_mol_m3=f_hso4 * c_s,
            so4_mol_m3=f_so4 * c_s,
            h3bo3_mol_m3=f_h3bo3 * c_b,
            h2bo3_mol_m3=f_h2bo3 * c_b,
            potential_V=phi,
            depleted=depleted,
            Ksp_FeOH2=self.Ksp,
            na_mol_m3=c_na,
        )

    # ─── Electrode kinetics ────────────────────────────────────────

    def _fe_equilibrium_potential(self, fe_surface_M: float, surface_pH: float | None = None) -> float:
        # a_Fe²⁺ = γ_Fe²⁺ · [Fe²⁺]; the ideal model uses γ = 1.
        # When the surface pH is known (Picard loop), use the *local*
        # Pitzer γ at the surface composition rather than the bulk γ —
        # the correction is 5–15 mV at the depletion levels that set
        # FE and the precipitation boundary.  Bulk callers (seed
        # potentials) pass surface_pH=None and retain the bulk γ.
        if self.activity_model == "pitzer" and surface_pH is not None:
            gamma = self._gamma_at(fe_surface_M, float(surface_pH))
        else:
            gamma = self.gamma_fe
        a = max(gamma * fe_surface_M, 1e-15)
        return E0_FE + (R_GAS * self.T / (Z_FE * FARADAY)) * np.log(a)

    def _her_equilibrium_potential(self, surface_pH: float) -> float:
        return -(R_GAS * self.T / FARADAY) * np.log(10.0) * surface_pH

    def _bv_current(self, E: float, i0: float, slope_c: float,
                    slope_a: float, E_eq: float) -> float:
        """Full Butler–Volmer branch current (cathodic positive; signed)."""
        return float(i0 * (10.0 ** ((E_eq - E) / slope_c)
                           - 10.0 ** ((E - E_eq) / slope_a)))

    def _her_water_equilibrium_potential(self, surface_pH: float) -> float:
        """E_eq for 2 H₂O + 2e⁻ → H₂ + 2 OH⁻ (V vs SHE).

        Same RHE potential as proton HER (the two reactions are linked by
        water autoionization), so E_eq = −(RT/F) ln10 · pH.  Its kinetics
        are separate (i0, Tafel slope) and it dominates at high pH where
        proton concentration is negligible.
        """
        return self._her_equilibrium_potential(surface_pH)

    def _her_water_current(
        self, E: float, surface_pH: float
    ) -> float:
        """Cathodic current of the water-reduction HER branch (A/m²)."""
        if not self.water_reduction_her:
            return 0.0
        eq = self._her_water_equilibrium_potential(surface_pH)
        i0_T = arrhenius_i0(
            self.her_water_i0, self.T, self.her_i0_Ea_J_mol, self.kinetics_ref_K
        )
        # Tafel cathodic branch only (no meaningful reverse at cathodic E).
        return float(max(i0_T * 10.0 ** ((eq - E) / self.her_water_tafel_V), 0.0))

    # ─── Transport limit ───────────────────────────────────────────

    def _cached_transport_limit(self, i_her_A_m2: float = 0.0) -> float:
        """Return the migration-aware film limit without repeated bisection."""
        key = round(float(np.log10(max(i_her_A_m2, 1e-12))), 1)
        if key not in self._transport_limit_cache:
            self._transport_limit_cache[key] = self.transport_limit_A_m2(i_her_A_m2)
        return float(self._transport_limit_cache[key])

    def transport_limit_A_m2(self, i_her_A_m2: float = 0.0) -> float:
        """Largest Fe deposition current the film can sustain (A/m²).

        Uses bisection on the ODE: find i_fe where surface Fe²⁺ drops
        to 1 % of bulk.
        """
        target = 0.01 * self.fe_conc_M * 1000.0

        def _residual_tl(i_fe: float) -> float:
            profile = self.integrate(i_fe, i_her_A_m2)
            if profile is None:
                return -target
            return float(profile.fe_mol_m3[0]) - target

        lo = 1.0
        hi = self.diffusion_limit_A_m2 * 1.5
        for _ in range(20):
            if _residual_tl(hi) < 0.0:
                break
            hi *= 1.5

        return float(brentq(_residual_tl, lo, hi, xtol=1.0, rtol=1e-6))

    # ─── Galvanostatic solver (Picard iteration) ──────────────────

    def _solve_kinetics_at_E(self, E: float) -> tuple:
        """Solve self-consistent (i_fe, i_her) at cathode potential *E*.

        Damped Picard iteration: integrate film → surface conditions →
        Tafel kinetics → update currents.  Converges in ~10 iterations
        (~10 ODE solves).

        Returns ``(i_fe, i_her, profile, converged)``.
        """
        # Migration is part of the same film closure, so do not cap the
        # cathodic branch with the binary diffusion-only limit.  The cached
        # limit avoids rerunning the bisection at every potential evaluation.
        i_lim = self._cached_transport_limit(0.0)

        # Seed from bulk BV (no transport correction)
        fe_eq_bulk = self._fe_equilibrium_potential(self.fe_conc_M)
        her_eq_bulk = self._her_equilibrium_potential(self.pH_bulk)
        i_fe = max(self._bv_current(E, self.fe_i0_T, self.fe_tafel_V,
                                    self.fe_anodic_slope_V, fe_eq_bulk), 0.0)
        eta_her_seed = her_eq_bulk - E
        her_i0_seed = self._surface_state_her_i0(eta_her_seed)
        i_her = max(self._bv_current(E, her_i0_seed, self.her_tafel_V,
                                     self.her_anodic_slope_V, her_eq_bulk), 0.0)
        i_fe = min(i_fe, i_lim * 0.99)

        converged = False
        damping = 0.4  # under-relaxation for stability

        if self.fast_mode:
            # Screening-mode: loose ODE + coarse brentq, ~10-20x cheaper.
            max_iter = min(self.max_iterations, 40)
            _rtol, _atol = 1e-3, 1e-6
            picard_tol = 1e-3
        else:
            max_iter = self.max_iterations
            _rtol, _atol = 1e-5, 1e-8
            picard_tol = 1e-6

        # Coarse grid + relaxed tolerances for iteration speed
        _gp = max(self.grid_points // 4, 21)

        for _ in range(max_iter):
            i_fe_c = max(i_fe, 1e-15)
            i_her_c = max(i_her, 1e-15)

            profile = self.integrate(i_fe_c, i_her_c,
                                     grid_points=_gp, rtol=_rtol, atol=_atol)
            if profile is None:
                i_fe *= 0.3
                i_her *= 0.3
                continue

            surf_fe_M = float(profile.fe_M[0])
            surf_pH = float(profile.pH[0])

            fe_eq = self._fe_equilibrium_potential(surf_fe_M, surf_pH)
            her_eq = self._her_equilibrium_potential(surf_pH)

            i_fe_kin = max(self._bv_current(E, self.fe_i0_T, self.fe_tafel_V,
                                            self.fe_anodic_slope_V, fe_eq), 0.0)
            # Surface-state correction to HER i₀ (Tier 1.1 from
            # CHEM_PHYS_REVIEW.md).  When enabled, the constant her_i0
            # is replaced by an effective value that accounts for
            # coverage, site-blocking, and facet ensemble.
            eta_her = her_eq - E  # HER overpotential (positive cathodic)
            her_i0_eff = self._surface_state_her_i0(eta_her)
            i_her_kin = max(self._bv_current(E, her_i0_eff, self.her_tafel_V,
                                             self.her_anodic_slope_V, her_eq), 0.0)
            # Water-reduction branch (high pH); its OH⁻ production enters
            # the proton-invariant balance just like proton consumption.
            i_her_water = self._her_water_current(E, surf_pH)

            # Koutecky-Levich: Fe cannot outrun transport
            i_fe_new = 1.0 / (
                1.0 / max(i_fe_kin, 1e-30) + 1.0 / max(i_lim, 1e-30)
            )
            i_her_new = i_her_kin + i_her_water

            # Convergence (relative)
            tol = picard_tol
            if (abs(i_fe_new - i_fe) < tol * max(i_fe, 1.0)
                    and abs(i_her_new - i_her) < tol * max(i_her, 1.0)):
                i_fe, i_her = i_fe_new, i_her_new
                converged = True
                break

            i_fe = damping * i_fe + (1.0 - damping) * i_fe_new
            i_her = damping * i_her + (1.0 - damping) * i_her_new

        # Final integration with converged currents (full grid, tight tol)
        profile = self.integrate(
            max(i_fe, 1e-15), max(i_her, 1e-15),
            rtol=1e-4 if self.fast_mode else 1e-8,
            atol=1e-7 if self.fast_mode else 1e-10,
        )
        return i_fe, i_her, profile, converged

    def solve(self, j_mA_cm2: float) -> DiffusionLayerResult:
        """Solve at an applied current density (mA/cm²).

        Finds the cathode potential that delivers the requested total
        cathodic current via ``brentq``, with self-consistent kinetics
        from Picard iteration at each *E* evaluation.
        """
        if j_mA_cm2 <= 0.0:
            raise ValueError("j_mA_cm2 must be positive")

        target = j_mA_cm2 * 10.0  # A/m²

        def _total_at_E(E: float) -> tuple:
            i_fe, i_her, profile, conv = self._solve_kinetics_at_E(E)
            return i_fe + i_her, i_fe, i_her, profile, conv

        def _residual_E(E: float) -> float:
            total, *_ = _total_at_E(E)
            return total - target

        # Bracket: near 0 V → tiny current; very negative → huge current
        E_lo, E_hi = -2.0, 0.05
        if _residual_E(E_hi) < 0:
            E_hi = -3.0
        if _residual_E(E_lo) > 0:
            E_lo = 0.1

        if self.fast_mode:
            E_sol = brentq(_residual_E, E_lo, E_hi, xtol=1e-4, rtol=1e-3,
                           maxiter=40)
        else:
            E_sol = brentq(_residual_E, E_lo, E_hi, xtol=1e-6, rtol=1e-8)

        total, i_fe, i_her, profile, converged = _total_at_E(E_sol)

        if profile is None:  # pragma: no cover
            raise RuntimeError("Film integration failed at converged solution")

        surf_fe_M = float(profile.fe_M[0])
        surf_pH = float(profile.pH[0])
        surf_oh_M = float(profile.oh_M[0])
        supersat = float(np.max(profile.feoh2_supersaturation))
        # ---- precipitation sink (screening, post-integration) ----
        precip_flux = 0.0
        precip_fraction = 0.0
        sludge_g = 0.0
        if self.precipitation_sink and self.k_precip_s > 0 and supersat > 1.0:
            # Local excess Fe above saturation: [Fe]_ex = max(0, [Fe]-Ksp/[OH]²)
            # R_vol = k · [Fe]_ex  (mol/L/s) → mol/m³/s after ×1000
            fe_ex_M = np.maximum(0.0, profile.fe_M - profile.Ksp_FeOH2 / np.maximum(profile.oh_M, 1e-30)**2)
            # Only where S>1 (fe_ex>0) contributes; clamp negative to 0
            R_vol_M_s = self.k_precip_s * fe_ex_M  # mol/L/s
            R_vol = R_vol_M_s * 1000.0  # mol/m³/s
            # Integrate over film thickness: ∫ R dx  (mol/m²/s)
            # profile.x_m is 0 at cathode, δ at bulk; use trapz sorted by x
            x = profile.x_m
            # x is ascending 0→δ, R is same order after flip-back; already 0=cathode
            precip_flux = float(np.trapezoid(R_vol, x))
            # Screening cap: precipitation cannot outrun diffusion of Fe to
            # the precipitation zone. The diffusion-limited Fe supply is
            # ~ D·(Fe_bulk−Fe_sat)/δ; cap the volumetric estimate there so
            # the diagnostic does not diverge at S≫1 (k·ΔC·δ ≫ D·ΔC/δ).
            fe_diff_limit_mol_m2_s = self.diffusion_limit_A_m2 / (Z_FE * FARADAY)
            precip_flux = float(min(precip_flux, 10.0 * fe_diff_limit_mol_m2_s))
            # Fraction of the Fe flux that precipitates before reaching the electrode
            fe_flux_mol_m2_s = i_fe / (Z_FE * FARADAY) if i_fe > 1e-12 else 0.0
            if fe_flux_mol_m2_s > 0:
                precip_fraction = float(min(precip_flux / (fe_flux_mol_m2_s + precip_flux), 0.99))
            # Sludge as Fe(OH)₂ mass (M=89.86 g/mol)
            M_FEOH2 = 89.86e-3  # kg/mol
            sludge_g = float(precip_flux * M_FEOH2 * 1000.0)  # g/m²/s
        # Report both the binary diffusion limit and the actual film limit.
        # The latter includes migration and concurrent HER through the same
        # reactive ODE; the previous result object silently reported the
        # diffusion value in both fields.
        transport_limit = self.transport_limit_A_m2(i_her_A_m2=i_her)

        V_cathode = E_sol
        V_cell = (self.E_anode_eq + self.eta_anode_V) - V_cathode + self.ir_drop_V

        # Surface-state diagnostic: ratio of corrected to bare her_i0
        her_i0_ratio = 1.0
        if self.surface_state:
            eta_her_diag = self._her_equilibrium_potential(surf_pH) - E_sol
            her_i0_eff = self._surface_state_her_i0(eta_her_diag)
            if self.her_i0_T > 0:
                her_i0_ratio = her_i0_eff / self.her_i0_T

        return DiffusionLayerResult(
            j_mA_cm2=total / 10.0,
            current_efficiency=i_fe / max(total, 1e-30),
            fe_current_A_m2=i_fe,
            her_current_A_m2=i_her,
            surface_pH=surf_pH,
            bulk_pH=self.pH_bulk,
            surface_fe_M=surf_fe_M,
            surface_oh_M=surf_oh_M,
            film_potential_drop_V=profile.film_potential_drop_V,
            precipitation_active=supersat >= 1.0,
            feoh2_supersaturation=supersat,
            transport_limit_A_m2=transport_limit,
            diffusion_limit_A_m2=self.diffusion_limit_A_m2,
            V_cathode_V=V_cathode,
            V_cell=V_cell,
            temperature_C=self.temperature_C,
            converged=converged,
            profile=profile,
            precipitation_flux_mol_m2_s=precip_flux,
            precipitation_fraction=precip_fraction,
            sludge_rate_g_m2_s=sludge_g,
            her_i0_surface_state_ratio=her_i0_ratio,
            fe_pair_fraction_bulk=self._bulk_pair_fraction if self.fes04_pair_correction else 0.0,
            d_composition_factor=self._composition_diffusivity_factor()
            if self.composition_diffusivity else 1.0,
            gamma_fe_bulk=self.gamma_fe,
        )

    # ─── Convenience methods ───────────────────────────────────────

    def efficiency_sweep(
        self, j_values_mA_cm2: Iterable[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Current efficiency vs. applied current density."""
        js = np.asarray(list(j_values_mA_cm2), dtype=float)
        return js, np.array([self.solve(float(j)).current_efficiency for j in js])

    def summary(self, j_mA_cm2: float = 100.0) -> dict:
        """Human-readable operating summary at a given current density."""
        s = self.solve(j_mA_cm2)
        return {
            "j applied (mA/cm²)": j_mA_cm2,
            "E cathode (V vs SHE)": round(s.V_cathode_V, 3),
            "Current efficiency (%)": round(s.fe_percent, 1),
            "Surface pH": round(s.surface_pH, 2),
            "ΔpH (surface − bulk)": round(s.local_pH_rise, 2),
            "Surface Fe²⁺ (M)": round(s.surface_fe_M, 4),
            "i_lim diffusion (A/m²)": round(s.diffusion_limit_A_m2, 1),
            "i_lim transport (A/m²)": round(s.transport_limit_A_m2, 1),
            "Film Δφ (mV)": round(s.film_potential_drop_V * 1000, 2),
            "Fe(OH)₂ supersaturation": float(f"{s.feoh2_supersaturation:.3g}"),
            "V_cell (V)": round(s.V_cell, 3),
        }


# ─── Top-level convenience ───────────────────────────────────────────

def faradaic_efficiency(
    j_mA_cm2: float,
    temperature_C: float = 60.0,
    fe_conc_M: float = 1.0,
    delta_m: float = 50.0e-6,
    pH_bulk: float = 2.0,
    buffer_conc_M: float = 0.40,
    **kwargs,
) -> float:
    """Direct FE(j, T, C, δ, pH, buffer) call.

    Returns current efficiency as a fraction (0–1).
    """
    model = DiffusionLayer1D(
        fe_conc_M=fe_conc_M,
        pH_bulk=pH_bulk,
        temperature_C=temperature_C,
        delta_m=delta_m,
        buffer_conc_M=buffer_conc_M,
        **kwargs,
    )
    return model.solve(j_mA_cm2).current_efficiency
