"""
1D diffusion-layer model for iron electrowinning — the FE prediction engine.

Full Nernst-Planck transport in a stagnant cathode film with:

* **Two Faradaic reactions** at the cathode:
    Fe²⁺ + 2 e⁻ → Fe(s)        (iron deposition)
    2 H⁺ + 2 e⁻ → H₂(g)        (hydrogen evolution)
* **Migration + diffusion** for six species:
    Fe²⁺ (z=+2), H⁺ (z=+1), OH⁻ (z=−1),
    HSO₄⁻ (z=−1), SO₄²⁻ (z=−2), H₃BO₃ (z=0), H₂BO₃⁻ (z=−1)
* **Homogeneous fast equilibria** (local, instantaneous):
    HSO₄⁻  ↔  H⁺ + SO₄²⁻      (Ka₂)
    H₃BO₃  ↔  H⁺ + H₂BO₃⁻     (Ka_b)
    H₂O    ↔  H⁺ + OH⁻         (Kw)
* **Arrhenius temperature correction** for all diffusivities
* **Butler-Volmer (Tafel) kinetics** at the electrode surface
* **Surface pH** from proton flux balance
* **Fe(OH)₂ precipitation** criterion ([Fe²⁺][OH⁻]² / Ksp)
* **Outputs**:  FE(j, T, C_Fe²⁺, δ, pH, buffer), V_cell, surface_pH

Conserved transported variables
-------------------------------
The ODE state is ``[c_fe, c_h, c_s, c_b, phi]`` where
``c_s = C_HSO4 + C_SO4`` (total sulfate) and
``c_b = C_H3BO3 + C_H2BO3`` (total borate).
The proton invariant whose flux equals the HER consumption rate is
``Φ = C_H + C_HSO4 + C_H3BO3 − C_OH``.

Electroneutrality closes the system for ``dφ/dx``.

Units
-----
Concentrations are SI (mol/m³) internally, mol/L in the public API.
Current densities are positive cathodic magnitudes (A/m² internally,
mA/cm² in the API).  Potentials are V vs. SHE.
"""

from __future__ import annotations

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
from .pourbaix import LOGKSP_FEOH2

# ─── Reference temperature ────────────────────────────────────────────
T_REF = 298.15  # K

# ─── Water autoprotolysis ─────────────────────────────────────────────
# Kw = 1e-14 (mol/L)² = 1e-14 × 1000² = 1e-8 (mol/m³)²
KW_SI = 1.0e-8

# ─── Bisulfate dissociation: HSO₄⁻ ↔ H⁺ + SO₄²⁻ ────────────────────
KA2_25C = 1.2e-2        # mol/L at 25 °C  (pKa ≈ 1.92)
KA2_EA_J_MOL = -22.0e3  # J/mol (exothermic; Ka₂ decreases with T)

# ─── Boric acid: H₃BO₃ ↔ H⁺ + H₂BO₃⁻ ──────────────────────────────
KAB_25C = 5.8e-10       # mol/L at 25 °C  (pKa ≈ 9.24)
KAB_EA_J_MOL = 14.0e3   # J/mol (endothermic; Ka_b increases with T)

# ─── Fe(OH)₂ solubility product ──────────────────────────────────────
KSP_FEOH2 = 10.0 ** LOGKSP_FEOH2  # (mol/L)³

# ─── Limiting ionic diffusivities at 25 °C (m²/s) ────────────────────
D_FE2 = 7.2e-10
D_H = 9.31e-9
D_OH = 5.27e-9
D_HSO4 = 1.33e-9
D_SO4 = 1.07e-9
D_H3BO3 = 0.92e-9
D_H2BO3 = 1.00e-9

# Default activation energy for diffusion (J/mol)
DIFF_EA_J_MOL = 18.0e3


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
        return self.fe_M * self.oh_M ** 2 / KSP_FEOH2

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
    E_anode_eq: float = 1.229
    eta_anode_V: float = 0.40
    ir_drop_V: float = 0.20
    grid_points: int = 101
    max_iterations: int = 200
    convergence_tol: float = 1e-9
    fast_mode: bool = False
    # Exchange-current densities are anchored at kinetics_ref_K and
    # Arrhenius-scaled to the operating temperature; see models/kinetics.py.
    fe_i0_Ea_J_mol: float = EA_FE_DEPOSITION_J_MOL
    her_i0_Ea_J_mol: float = EA_HER_ON_FE_J_MOL
    kinetics_ref_K: float = I0_REF_K
    # Butler-Volmer anodic-branch slopes (cathodic slopes above retained).
    fe_anodic_slope_V: float = FE_ANODIC_SLOPE_V
    her_anodic_slope_V: float = HER_ANODIC_SLOPE_V

    def __post_init__(self) -> None:
        if self.fe_conc_M <= 0.0:
            raise ValueError("fe_conc_M must be positive")
        if self.delta_m <= 0.0:
            raise ValueError("delta_m must be positive")
        if self.grid_points < 3:
            raise ValueError("grid_points must be at least 3")

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
        return _diffusivity_T(D_FE2, self.T)

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
        """Pure-diffusion (Levich) limiting current for Fe²⁺ (A/m²)."""
        return Z_FE * FARADAY * self.D_fe * self.fe_conc_M * 1000.0 / self.delta_m

    @property
    def fe_i0_T(self) -> float:
        """Fe exchange current density Arrhenius-scaled to T."""
        return arrhenius_i0(self.fe_i0, self.T, self.fe_i0_Ea_J_mol, self.kinetics_ref_K)

    @property
    def her_i0_T(self) -> float:
        """HER exchange current density Arrhenius-scaled to T."""
        return arrhenius_i0(self.her_i0, self.T, self.her_i0_Ea_J_mol, self.kinetics_ref_K)

    # ─── Equilibrium fractions ─────────────────────────────────────

    def _fractions(self, c_h: float, c_s: float, c_b: float) -> dict:
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

        return {
            "c_hso4": f_hso4 * c_s,
            "c_so4": f_so4 * c_s,
            "c_h3bo3": f_h3bo3 * c_b,
            "c_h2bo3": f_h2bo3 * c_b,
            "c_oh": KW_SI / c_h,
            "c_na": 2.0 * self.support_conc_M * 1000.0,
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

        y0 = np.array([c_fe_0, c_h_0, c_s_0, c_b_0, 0.0])
        x_eval = np.linspace(self.delta_m, 0.0, gp)

        sol = solve_ivp(
            self._rhs,
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
        )

    # ─── Electrode kinetics ────────────────────────────────────────

    def _fe_equilibrium_potential(self, fe_surface_M: float) -> float:
        a = max(fe_surface_M, 1e-15)
        return E0_FE + (R_GAS * self.T / (Z_FE * FARADAY)) * np.log(a)

    def _her_equilibrium_potential(self, surface_pH: float) -> float:
        return -(R_GAS * self.T / FARADAY) * np.log(10.0) * surface_pH

    def _bv_current(self, E: float, i0: float, slope_c: float,
                    slope_a: float, E_eq: float) -> float:
        """Full Butler–Volmer branch current (cathodic positive; signed)."""
        return float(i0 * (10.0 ** ((E_eq - E) / slope_c)
                           - 10.0 ** ((E - E_eq) / slope_a)))

    # ─── Transport limit ───────────────────────────────────────────

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
        i_lim = self.diffusion_limit_A_m2

        # Seed from bulk BV (no transport correction)
        fe_eq_bulk = self._fe_equilibrium_potential(self.fe_conc_M)
        her_eq_bulk = self._her_equilibrium_potential(self.pH_bulk)
        i_fe = max(self._bv_current(E, self.fe_i0_T, self.fe_tafel_V,
                                    self.fe_anodic_slope_V, fe_eq_bulk), 0.0)
        i_her = max(self._bv_current(E, self.her_i0_T, self.her_tafel_V,
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

            fe_eq = self._fe_equilibrium_potential(surf_fe_M)
            her_eq = self._her_equilibrium_potential(surf_pH)

            i_fe_kin = max(self._bv_current(E, self.fe_i0_T, self.fe_tafel_V,
                                            self.fe_anodic_slope_V, fe_eq), 0.0)
            i_her_kin = max(self._bv_current(E, self.her_i0_T, self.her_tafel_V,
                                             self.her_anodic_slope_V, her_eq), 0.0)

            # Koutecky-Levich: Fe cannot outrun transport
            i_fe_new = 1.0 / (
                1.0 / max(i_fe_kin, 1e-30) + 1.0 / max(i_lim, 1e-30)
            )
            i_her_new = i_her_kin

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

        V_cathode = E_sol
        V_cell = (self.E_anode_eq + self.eta_anode_V) - V_cathode + self.ir_drop_V

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
            transport_limit_A_m2=self.diffusion_limit_A_m2,
            diffusion_limit_A_m2=self.diffusion_limit_A_m2,
            V_cathode_V=V_cathode,
            V_cell=V_cell,
            temperature_C=self.temperature_C,
            converged=converged,
            profile=profile,
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
