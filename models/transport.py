"""
Steady one-dimensional Nernst-Planck transport in the cathode film.

This module replaces the linear stagnant-film approximation of
:mod:`models.boundary_layer` with a dilute-solution Nernst-Planck solve that
carries **diffusion and migration** for every ionic species in the film:

    N_i = -D_i dC_i/dx - z_i D_i (F/RT) C_i dphi/dx

Why migration matters
---------------------
Iron electrowinning baths are frequently run with little or no inert
supporting electrolyte (FeSO4/FeCl2 alone, or the chloride-rich AWARE-type
baths).  In that regime the electric field inside the diffusion layer does
real work: it drags Fe2+ toward the cathode, raising the transport-limited
current above the Levich/diffusion-only value by roughly 1/(1 - t_Fe), and it
also sets a diffusion (liquid-junction) potential across the film that is not
captured by an IR-free lumped model.  Conversely, heavy supporting electrolyte
collapses the field and recovers the pure-diffusion answer -- which is exactly
the limit the film model already describes.

Species and chemistry
---------------------
Five species are tracked: Fe2+, H+, OH-, a supporting cation (Na+) and a
supporting anion (SO4^2-).  Water autoprotolysis is treated as a fast local
equilibrium, so H+ and OH- are not independently conserved; the conserved
combination is the *proton excess*

    S = C_H+ - C_OH-,      N_S = N_H+ - N_OH- = -i_HER / F

which holds for both the acidic (2 H+ + 2 e- -> H2) and alkaline
(2 H2O + 2 e- -> H2 + 2 OH-) hydrogen pathways.  C_H+ is recovered from S via
C_H+ = (S + sqrt(S^2 + 4 Kw)) / 2.

Closure
-------
Electroneutrality (sum z_i C_i = 0) is imposed pointwise; differentiating it
and substituting the Nernst-Planck fluxes gives an explicit expression for
dphi/dx, so the system integrates as a plain ODE from the bulk edge of the
film (x = delta) to the electrode surface (x = 0).

Scope
-----
Still a screening model: steady state, no convection inside the film, no
activity coefficients, no homogeneous complexation or bisulfate equilibria,
and precipitation is reported as a supersaturation diagnostic rather than
being allowed to consume Fe2+.

Units: concentrations are SI (mol/m^3) internally and reported in mol/L;
current densities are positive cathodic magnitudes in A/m^2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .electrochemistry import E0_FE, FARADAY, R_GAS
from .kinetics import limiting_current_density
from .pourbaix import LOGKSP_FEOH2, her_line

# Water autoprotolysis constant expressed in (mol/m^3)^2:
# Kw = 1e-14 (mol/L)^2 = 1e-14 * (1000 mol/m^3)^2 = 1e-8 (mol/m^3)^2
KW_SI = 1.0e-8

# Limiting ionic diffusivities at infinite dilution, 25 C (m^2/s).
D_FE = 7.2e-10
D_H = 9.31e-9
D_OH = 5.27e-9
D_NA = 1.33e-9
D_SO4 = 1.07e-9


@dataclass
class FilmProfile:
    """Spatial profiles across the cathode film, x = 0 at the electrode."""

    x_m: np.ndarray
    fe_M: np.ndarray
    h_M: np.ndarray
    oh_M: np.ndarray
    na_M: np.ndarray
    so4_M: np.ndarray
    potential_V: np.ndarray
    depleted: bool

    @property
    def pH(self) -> np.ndarray:
        return -np.log10(np.maximum(self.h_M, 1e-30))

    @property
    def feoh2_supersaturation(self) -> np.ndarray:
        """[Fe2+][OH-]^2 / Ksp along the film; > 1 means Fe(OH)2 is unstable."""
        return self.fe_M * self.oh_M**2 / (10.0**LOGKSP_FEOH2)

    @property
    def film_potential_drop_V(self) -> float:
        """phi(surface) - phi(bulk): the diffusion/junction potential (V)."""
        return float(self.potential_V[0] - self.potential_V[-1])


@dataclass
class NernstPlanckState:
    """Operating point returned by :class:`NernstPlanckFilm.solve`."""

    potential_V: float
    applied_current_A_m2: float
    fe_current_A_m2: float
    her_current_A_m2: float
    current_efficiency: float
    bulk_pH: float
    surface_pH: float
    bulk_fe_M: float
    surface_fe_M: float
    surface_oh_M: float
    film_potential_drop_V: float
    migration_flux_fraction: float
    diffusion_limit_A_m2: float
    transport_limit_A_m2: float
    precipitation_active: bool
    feoh2_supersaturation: float
    converged: bool
    profile: FilmProfile = field(repr=False)

    @property
    def local_pH_rise(self) -> float:
        """Surface pH minus bulk pH."""
        return self.surface_pH - self.bulk_pH

    @property
    def migration_enhancement(self) -> float:
        """Transport-limited current relative to the diffusion-only value."""
        return self.transport_limit_A_m2 / max(self.diffusion_limit_A_m2, 1e-30)


@dataclass
class NernstPlanckFilm:
    """Steady Nernst-Planck film coupled to competing Fe/HER kinetics.

    Parameters
    ----------
    bulk_pH : float
        Bulk electrolyte pH.
    fe_conc_M : float
        Bulk Fe2+ concentration (mol/L).
    support_conc_M : float
        Bulk inert supporting electrolyte, as Na2SO4 (mol/L).  Zero means an
        unsupported binary bath, where migration is strongest.
    boundary_layer_m : float
        Film (Nernst diffusion layer) thickness; agitation reduces it.
    temperature_C : float
        Temperature for the Nernst equation, HER line and the F/RT factor.
    fe_i0, her_i0 : float
        Exchange current densities (A/m^2).
    fe_tafel_V, her_tafel_V : float
        Cathodic Tafel slopes (V/decade).
    """

    bulk_pH: float = 2.0
    fe_conc_M: float = 1.0
    support_conc_M: float = 0.0
    boundary_layer_m: float = 5e-5
    temperature_C: float = 60.0
    fe_i0: float = 1e-2
    her_i0: float = 1e-6
    fe_tafel_V: float = 0.120
    her_tafel_V: float = 0.140
    diffusivity_fe_m2_s: float = D_FE
    diffusivity_h_m2_s: float = D_H
    diffusivity_oh_m2_s: float = D_OH
    diffusivity_na_m2_s: float = D_NA
    diffusivity_so4_m2_s: float = D_SO4
    grid_points: int = 121
    max_iterations: int = 200
    convergence_tol: float = 1e-9

    _limit_cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.fe_conc_M <= 0.0:
            raise ValueError("fe_conc_M must be positive")
        if self.support_conc_M < 0.0:
            raise ValueError("support_conc_M must be non-negative")
        if self.boundary_layer_m <= 0.0:
            raise ValueError("boundary_layer_m must be positive")
        if self.grid_points < 3:
            raise ValueError("grid_points must be at least 3")

    # ─── Bulk composition ─────────────────────────────────────────────
    @property
    def T(self) -> float:
        return self.temperature_C + 273.15

    @property
    def f_RT(self) -> float:
        """F / RT (1/V)."""
        return FARADAY / (R_GAS * self.T)

    @property
    def bulk_oh_M(self) -> float:
        return 10.0 ** (self.bulk_pH - 14.0)

    @property
    def bulk_composition_mol_m3(self) -> dict:
        """Electroneutral bulk composition in mol/m^3.

        Fe2+ enters as FeSO4 and the support as Na2SO4; any residual charge
        (from the imposed pH) is closed with the inert spectator ion of the
        appropriate sign, i.e. extra SO4^2- in acid or extra Na+ in alkali.
        """
        fe = self.fe_conc_M * 1000.0
        h = 10.0 ** (-self.bulk_pH) * 1000.0
        oh = KW_SI / h
        na = 2.0 * self.support_conc_M * 1000.0
        so4 = (self.fe_conc_M + self.support_conc_M) * 1000.0

        charge = 2.0 * fe + h + na - oh - 2.0 * so4
        if charge > 0.0:
            so4 += charge / 2.0
        else:
            na += -charge
        return {"fe": fe, "h": h, "oh": oh, "na": na, "so4": so4}

    @property
    def diffusion_limit_A_m2(self) -> float:
        """Pure-diffusion (Levich) limiting current for Fe2+ (A/m^2)."""
        return limiting_current_density(
            self.fe_conc_M * 1000.0,
            self.diffusivity_fe_m2_s,
            self.boundary_layer_m,
        )

    @property
    def fe_transference_number(self) -> float:
        """Bulk transference number of Fe2+, t = z^2 D C / sum(z^2 D C)."""
        c = self.bulk_composition_mol_m3
        terms = {
            "fe": 4.0 * self.diffusivity_fe_m2_s * c["fe"],
            "h": 1.0 * self.diffusivity_h_m2_s * c["h"],
            "oh": 1.0 * self.diffusivity_oh_m2_s * c["oh"],
            "na": 1.0 * self.diffusivity_na_m2_s * c["na"],
            "so4": 4.0 * self.diffusivity_so4_m2_s * c["so4"],
        }
        return terms["fe"] / sum(terms.values())

    # ─── Nernst-Planck integration ────────────────────────────────────
    def _rhs(self, _x: float, y: np.ndarray, n_fe: float, n_s: float):
        """d/dx of [C_Fe, C_H, C_Na, C_SO4, phi] at fixed species fluxes.

        Fluxes are in the +x direction (away from the electrode), so cathodic
        consumption of Fe2+ corresponds to n_fe < 0.
        """
        floor = 1e-8 * self.fe_conc_M * 1000.0
        c_fe = max(y[0], floor)
        c_h = max(y[1], 1e-20)
        c_na = max(y[2], 0.0)
        c_so4 = max(y[3], 0.0)
        c_oh = KW_SI / c_h

        d_fe = self.diffusivity_fe_m2_s
        d_h = self.diffusivity_h_m2_s
        d_oh = self.diffusivity_oh_m2_s
        d_na = self.diffusivity_na_m2_s
        d_so4 = self.diffusivity_so4_m2_s

        # Proton-pair coupling: C_OH = Kw/C_H, so both derivatives follow C_H.
        k = KW_SI / c_h**2                       # -dC_OH/dC_H
        a = d_h + d_oh * k                       # effective proton diffusivity
        b = (d_h * c_h + d_oh * c_oh) / a        # effective migration weight

        # Electroneutrality derivative -> explicit field.
        numerator = -2.0 * n_fe / d_fe - (1.0 + k) * n_s / a
        denominator = 4.0 * c_fe + (1.0 + k) * b + c_na + 4.0 * c_so4
        f_dphi = numerator / denominator         # = (F/RT) dphi/dx

        dc_fe = -n_fe / d_fe - 2.0 * c_fe * f_dphi
        dc_h = -n_s / a - b * f_dphi
        dc_na = -c_na * f_dphi
        dc_so4 = 2.0 * c_so4 * f_dphi
        dphi = f_dphi / self.f_RT
        return [dc_fe, dc_h, dc_na, dc_so4, dphi]

    def integrate(self, i_fe_A_m2: float, i_her_A_m2: float) -> FilmProfile:
        """Integrate the film from the bulk edge to the electrode surface.

        Parameters
        ----------
        i_fe_A_m2, i_her_A_m2 : float
            Positive cathodic partial current densities.
        """
        if i_fe_A_m2 < 0.0 or i_her_A_m2 < 0.0:
            raise ValueError("partial current densities must be non-negative")

        n_fe = -i_fe_A_m2 / (2.0 * FARADAY)   # Fe2+ consumed at the electrode
        n_s = -i_her_A_m2 / FARADAY           # proton excess consumed by HER

        c = self.bulk_composition_mol_m3
        y0 = [c["fe"], c["h"], c["na"], c["so4"], 0.0]  # phi(bulk) = 0

        x_eval = np.linspace(self.boundary_layer_m, 0.0, self.grid_points)

        from ._transport_jit import has_numba, get_integrate_film_jit

        if has_numba():
            integrate_film = get_integrate_film_jit()
            y_arr = integrate_film(
                np.array(y0, dtype=np.float64),
                n_fe, n_s,
                self.boundary_layer_m, 0.0, x_eval,
                self.diffusivity_fe_m2_s,
                self.diffusivity_h_m2_s,
                self.diffusivity_oh_m2_s,
                self.diffusivity_na_m2_s,
                self.diffusivity_so4_m2_s,
                self.f_RT,
                1e-8 * self.fe_conc_M * 1000.0,
            )
            x = x_eval[::-1]
            fe = y_arr[0][::-1]
            h = y_arr[1][::-1]
            na = y_arr[2][::-1]
            so4 = y_arr[3][::-1]
            phi = y_arr[4][::-1]
        else:
            sol = solve_ivp(
                self._rhs,
                (self.boundary_layer_m, 0.0),
                y0,
                t_eval=x_eval,
                args=(n_fe, n_s),
                method="LSODA",
                rtol=1e-8,
                atol=1e-10,
            )
            if not sol.success:  # pragma: no cover - defensive
                raise RuntimeError(f"Nernst-Planck integration failed: {sol.message}")

            # Flip so index 0 is the electrode surface and index -1 the bulk.
            x = sol.t[::-1]
            fe = sol.y[0][::-1]
            h = sol.y[1][::-1]
            na = sol.y[2][::-1]
            so4 = sol.y[3][::-1]
            phi = sol.y[4][::-1]

        floor = 1e-8 * self.fe_conc_M * 1000.0
        depleted = bool(np.min(fe) <= floor)
        fe = np.maximum(fe, floor)
        h = np.maximum(h, 1e-20)
        oh = KW_SI / h

        return FilmProfile(
            x_m=x,
            fe_M=fe / 1000.0,
            h_M=h / 1000.0,
            oh_M=oh / 1000.0,
            na_M=na / 1000.0,
            so4_M=so4 / 1000.0,
            potential_V=phi,
            depleted=depleted,
        )

    # ─── Electrode kinetics ───────────────────────────────────────────
    def _fe_equilibrium_potential(self, fe_surface_M: float) -> float:
        activity = max(fe_surface_M, 1e-15)
        return E0_FE + (R_GAS * self.T / (2.0 * FARADAY)) * np.log(activity)

    def _tafel_current(self, E: float, i0: float, slope: float, E_eq: float) -> float:
        return float(i0 * 10.0 ** ((E_eq - E) / slope))

    def _kinetic_currents(self, E: float, i_fe: float, i_her: float):
        """Tafel currents evaluated at the surface produced by (i_fe, i_her)."""
        profile = self.integrate(i_fe, i_her)
        her = self._tafel_current(
            E, self.her_i0, self.her_tafel_V,
            float(her_line(float(profile.pH[0]), self.T)),
        )
        fe_kin = self._tafel_current(
            E, self.fe_i0, self.fe_tafel_V,
            self._fe_equilibrium_potential(float(profile.fe_M[0])),
        )
        # Deposition cannot outrun what the film can deliver.  A hard min()
        # would make the residual non-smooth (and the surface Fe2+ collapse
        # onto the numerical floor, destroying the kinetic feedback), so the
        # two resistances are blended in Koutecky-Levich form, exactly as in
        # models.kinetics.TafelBranch.  The migration-aware limit is
        # evaluated at zero HER: its HER dependence is weak, and freezing it
        # keeps the residual a smooth function of the unknowns.
        i_lim = self._cached_transport_limit(0.0)
        fe = 1.0 / (1.0 / max(fe_kin, 1e-30) + 1.0 / i_lim)
        return fe, her, profile

    def _state_at_potential(self, E: float) -> NernstPlanckState:
        """Solve the coupled surface composition / kinetics at fixed potential.

        The two unknowns are the partial currents, solved in log space by
        nested bisection.  Each branch is a monotonically decreasing residual
        (more current -> more depletion / higher local pH -> less kinetic
        driving force), so bisection is unconditionally convergent.  A
        Newton-type solver is *not* usable here: when the film runs out of
        protons the surface pH jumps by ~10 units over a hair's width of HER
        current, and the resulting near-discontinuity in the residual defeats
        derivative-based methods (and makes plain fixed-point iteration
        oscillate).  Bisection simply brackets that front and squeezes in.
        """
        floor = 1e-12
        lo, hi = -12.0, 6.0

        def solve_branch(index: int, other: float) -> float:
            """Bisect log10(current) for one branch holding the other fixed."""

            def g(u: float) -> float:
                args = (10.0**u, other) if index == 0 else (other, 10.0**u)
                kin = self._kinetic_currents(E, *args)[index]
                return np.log10(max(kin, floor)) - u

            if g(lo) <= 0.0:
                return lo
            if g(hi) >= 0.0:
                return hi
            return float(brentq(g, lo, hi, xtol=1e-10, rtol=1e-12))

        # Seed from the transport-free (bulk-composition) kinetic currents.
        fe0, her0, _ = self._kinetic_currents(E, 0.0, 0.0)
        u_fe = np.clip(np.log10(max(fe0, floor)), lo, hi)
        u_her = np.clip(np.log10(max(her0, floor)), lo, hi)

        # Outer loop couples the branches: HER sets the local pH that the Fe
        # branch sees, and Fe depletion shifts the field that moves protons.
        converged = False
        for _ in range(self.max_iterations):
            new_her = solve_branch(1, 10.0**u_fe)
            new_fe = solve_branch(0, 10.0**new_her)
            if max(abs(new_her - u_her), abs(new_fe - u_fe)) < 1e-8:
                u_her, u_fe = new_her, new_fe
                converged = True
                break
            u_her, u_fe = new_her, new_fe

        i_fe = float(10.0**u_fe)
        i_her = float(10.0**u_her)
        profile = self.integrate(i_fe, i_her)
        total = i_fe + i_her
        supersat = float(np.max(profile.feoh2_supersaturation))

        # Fraction of the Fe2+ flux at the surface carried by migration.
        migration_fraction = self._migration_fraction(profile, i_fe)

        return NernstPlanckState(
            potential_V=E,
            applied_current_A_m2=total,
            fe_current_A_m2=i_fe,
            her_current_A_m2=i_her,
            current_efficiency=i_fe / max(total, 1e-30),
            bulk_pH=self.bulk_pH,
            surface_pH=float(profile.pH[0]),
            bulk_fe_M=self.fe_conc_M,
            surface_fe_M=float(profile.fe_M[0]),
            surface_oh_M=float(profile.oh_M[0]),
            film_potential_drop_V=profile.film_potential_drop_V,
            migration_flux_fraction=migration_fraction,
            diffusion_limit_A_m2=self.diffusion_limit_A_m2,
            transport_limit_A_m2=self._cached_transport_limit(0.0),
            precipitation_active=supersat >= 1.0,
            feoh2_supersaturation=supersat,
            converged=converged,
            profile=profile,
        )

    def _migration_fraction(self, profile: FilmProfile, i_fe: float) -> float:
        """Share of the surface Fe2+ flux supplied by migration (signed)."""
        if i_fe <= 0.0:
            return 0.0
        x = profile.x_m
        c_fe = profile.fe_M * 1000.0
        phi = profile.potential_V
        # One-sided derivatives at the electrode surface.
        dphi = (phi[1] - phi[0]) / (x[1] - x[0])
        n_mig = -2.0 * self.diffusivity_fe_m2_s * self.f_RT * c_fe[0] * dphi
        n_total = -i_fe / (2.0 * FARADAY)
        return float(n_mig / n_total)

    # ─── Transport limit ──────────────────────────────────────────────
    def _cached_transport_limit(self, i_her_A_m2: float) -> float:
        """Transport limit on a coarse HER grid, cached to keep the loop cheap.

        The limit depends only weakly on the HER current (through the local
        composition), so rounding to two significant figures is ample and
        avoids re-running the bisection on every fixed-point iteration.
        """
        key = round(float(np.log10(max(i_her_A_m2, 1e-12))), 1)
        if key not in self._limit_cache:
            self._limit_cache[key] = self.transport_limit_A_m2(i_her_A_m2)
        return self._limit_cache[key]

    def transport_limit_A_m2(self, i_her_A_m2: float = 0.0) -> float:
        """Largest Fe deposition current the film can sustain (A/m^2).

        Found by bisection on the Fe2+ surface concentration; unlike the
        Levich value this includes the migration contribution and the effect
        of concurrent HER on the local composition.
        """
        target = 1e-4 * self.fe_conc_M * 1000.0  # near-zero surface Fe2+

        def residual(i_fe: float) -> float:
            profile = self.integrate(i_fe, i_her_A_m2)
            return float(profile.fe_M[0]) * 1000.0 - target

        lo = 1e-6
        hi = self.diffusion_limit_A_m2
        for _ in range(60):
            if residual(hi) < 0.0:
                break
            hi *= 1.4
        else:  # pragma: no cover - defensive
            return hi
        return float(brentq(residual, lo, hi, xtol=1e-6, rtol=1e-10))

    # ─── Galvanostatic operation ──────────────────────────────────────
    def solve(self, j_mA_cm2: float) -> NernstPlanckState:
        """Solve the cathode potential and film state at an applied current."""
        if j_mA_cm2 <= 0.0:
            raise ValueError("j_mA_cm2 must be positive")
        target = j_mA_cm2 * 10.0

        def residual(E: float) -> float:
            return self._state_at_potential(E).applied_current_A_m2 - target

        E = brentq(residual, -3.0, 0.2, xtol=1e-7)
        return self._state_at_potential(float(E))

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
            "E cathode (V vs SHE)": round(s.potential_V, 3),
            "Current efficiency (%)": round(s.current_efficiency * 100, 1),
            "Surface pH": round(s.surface_pH, 2),
            "ΔpH (surface − bulk)": round(s.local_pH_rise, 2),
            "Surface Fe²⁺ (M)": round(s.surface_fe_M, 4),
            "i_lim diffusion (A/m²)": round(s.diffusion_limit_A_m2, 1),
            "i_lim with migration (A/m²)": round(s.transport_limit_A_m2, 1),
            "Migration enhancement (×)": round(s.migration_enhancement, 3),
            "Migration share of Fe²⁺ flux": round(s.migration_flux_fraction, 3),
            "Film potential drop (mV)": round(s.film_potential_drop_V * 1000, 2),
            "t_Fe²⁺ (bulk)": round(self.fe_transference_number, 3),
            "Max Fe(OH)₂ supersaturation": float(f"{s.feoh2_supersaturation:.3g}"),
        }


def compare_support_levels(
    support_levels_M: Optional[Iterable[float]] = None,
    j_mA_cm2: float = 100.0,
    **film_kwargs,
) -> list[dict]:
    """Sweep supporting-electrolyte concentration at fixed current density.

    Demonstrates the collapse of migration as inert salt is added: the
    transport limit falls back toward the Levich value and the film potential
    drop shrinks.
    """
    if support_levels_M is None:
        support_levels_M = (0.0, 0.1, 0.5, 2.0)
    rows = []
    for c_s in support_levels_M:
        film = NernstPlanckFilm(support_conc_M=float(c_s), **film_kwargs)
        state = film.solve(j_mA_cm2)
        rows.append(
            {
                "support_M": float(c_s),
                "t_Fe": film.fe_transference_number,
                "diffusion_limit_A_m2": state.diffusion_limit_A_m2,
                "transport_limit_A_m2": state.transport_limit_A_m2,
                "migration_enhancement": state.migration_enhancement,
                "migration_flux_fraction": state.migration_flux_fraction,
                "film_potential_drop_mV": state.film_potential_drop_V * 1000.0,
                "surface_pH": state.surface_pH,
                "surface_fe_M": state.surface_fe_M,
                "current_efficiency": state.current_efficiency,
            }
        )
    return rows
