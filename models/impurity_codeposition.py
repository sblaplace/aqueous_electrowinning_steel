"""
Impurity co-deposition model — Cu/Ni/Zn/Pb/Sn uptake vs bath concentration.

Predicts impurity incorporation in the iron deposit as a function of bath
impurity concentration, current density, temperature, and electrolyte type.
Answers the engineering question: "Do I need to purify the feedstock before
the first experiment, or can I plate first and purify later?"

Physics
-------
Each impurity species M^{2+} + 2e- → M competes with the primary
Fe^{2+} + 2e- → Fe deposition reaction.  The partial current for each
species follows Butler-Volmer kinetics capped by a mass-transport limit:

    i_M = 1 / (1/i_kin + 1/i_lim)

where
    i_kin = i0 * 10^(eta / beta_c)       (cathodic Tafel, beta_c > 0)
    i_lim = z F D C_M / delta             (Levich diffusion limit)

The deposit composition is determined by the faradaic mass rates:

    molar_rate_M = i_M / (z_M * F)        [mol/(m²·s)]
    mass_rate_M  = molar_rate_M * M_M      [kg/(m²·s)]

    wt% M = mass_rate_M / sum(mass_rate_all) * 100

Cu and Ni are nobler than Fe — they deposit preferentially at low
overpotentials.  Zn is less noble — it should not co-deposit significantly
under typical iron electrowinning conditions.

Bath type
---------
Two electrolyte families are modeled:

* **Sulfate** (FeSO₄-based): lower conductivity, moderate kinetics.
* **Chloride** (FeCl₂-based): higher conductivity, faster kinetics for
  most couples, but different complexation equilibria shift the effective
  exchange current densities.

Parameters are drawn from standard electrochemistry references; users
should calibrate to their specific bath composition.

References
----------
* Schlesinger & Paunovic, *Modern Electroplating*, 5th ed. (2010).
* Bockris & Reddy, *Modern Electrochemistry*, 2nd ed.
* Brenner, *Electrodeposition of Alloys* (1963).
* Task spec: t_05cd6587 — Impurity co-deposition model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Literal, Optional

import numpy as np

from .electrochemistry import FARADAY, R_GAS, E0_FE, M_FE, Z_FE
from .kinetics import limiting_current_density

# -------------------------------------------------------------------
# Physical constants for impurity species
# -------------------------------------------------------------------

# Copper
M_CU = 63.546e-3        # kg/mol
Z_CU = 2                # electrons per Cu2+ → Cu
E0_CU = 0.337           # V vs. SHE (Cu2+/Cu)

# Nickel
M_NI = 58.693e-3        # kg/mol
Z_NI = 2                # electrons per Ni2+ → Ni
E0_NI = -0.257          # V vs. SHE (Ni2+/Ni) — task-specified value

# Zinc
M_ZN = 65.38e-3         # kg/mol
Z_ZN = 2                # electrons per Zn2+ → Zn
E0_ZN = -0.763          # V vs. SHE (Zn2+/Zn)

# Lead
M_PB = 207.2e-3         # kg/mol
Z_PB = 2                # electrons per Pb2+ → Pb
E0_PB = -0.126          # V vs. SHE (Pb2+/Pb)

# Tin
M_SN = 118.71e-3        # kg/mol
Z_SN = 2                # electrons per Sn2+ → Sn
E0_SN = -0.136          # V vs. SHE (Sn2+/Sn)

# Diffusivities in aqueous sulfate at 25 °C (m²/s)
D_CU_DEFAULT = 0.72e-9
D_NI_DEFAULT = 0.66e-9
D_ZN_DEFAULT = 0.72e-9
D_PB_DEFAULT = 0.95e-9  # Pb2+ is larger but less hydrated
D_SN_DEFAULT = 0.72e-9

# Hot shortness threshold for copper in steel
CU_HOT_SHORTNESS_WT = 0.1  # wt%


# -------------------------------------------------------------------
# Bath-type kinetic parameter sets
# -------------------------------------------------------------------

@dataclass(frozen=True)
class BathKinetics:
    """Kinetic parameters for a specific bath chemistry."""

    # Exchange current densities (A/m²) for each species
    fe_i0: float
    cu_i0: float
    ni_i0: float
    zn_i0: float
    pb_i0: float
    sn_i0: float

    # Cathodic Tafel slopes (V/decade, positive)
    fe_tafel: float
    cu_tafel: float
    ni_tafel: float
    zn_tafel: float
    pb_tafel: float
    sn_tafel: float

    # Conductivity factor (multiplier on diffusivity for migration enhancement)
    conductivity_factor: float = 1.0


# Sulfate bath: FeSO₄-based, moderate kinetics
SULFATE_KINETICS = BathKinetics(
    fe_i0=1.0e-2,
    cu_i0=5.0,        # Cu2+/Cu is fast
    ni_i0=2.0e-2,     # Ni2+/Ni is moderate
    zn_i0=3.0e-2,     # Zn2+/Zn is moderate
    pb_i0=1.0e-1,     # Pb2+/Pb is fast
    sn_i0=5.0e-2,     # Sn2+/Sn is moderate
    fe_tafel=0.120,
    cu_tafel=0.120,
    ni_tafel=0.100,
    zn_tafel=0.120,
    pb_tafel=0.120,
    sn_tafel=0.110,
    conductivity_factor=1.0,
)

# Chloride bath: FeCl₂-based, higher conductivity and faster kinetics
CHLORIDE_KINETICS = BathKinetics(
    fe_i0=2.0e-2,
    cu_i0=8.0,        # Cu deposits even faster in chloride
    ni_i0=4.0e-2,     # Ni slightly faster in chloride
    zn_i0=5.0e-2,     # Zn slightly faster in chloride
    pb_i0=1.5e-1,
    sn_i0=8.0e-2,
    fe_tafel=0.110,    # slightly lower slopes = faster kinetics
    cu_tafel=0.110,
    ni_tafel=0.090,
    zn_tafel=0.110,
    pb_tafel=0.110,
    sn_tafel=0.100,
    conductivity_factor=1.3,  # chloride is more conductive
)

BATH_KINETICS: Dict[str, BathKinetics] = {
    "sulfate": SULFATE_KINETICS,
    "chloride": CHLORIDE_KINETICS,
}


# -------------------------------------------------------------------
# Main model
# -------------------------------------------------------------------

@dataclass
class ImpurityCoDeposition:
    """
    Predict impurity uptake in an iron electrodeposit.

    Parameters
    ----------
    fe_conc_M : float
        Bulk Fe²⁺ concentration (mol/L).
    cu_conc_ppm, ni_conc_ppm, zn_conc_ppm, pb_conc_ppm, sn_conc_ppm : float
        Impurity concentrations in the bath (ppm by mass).
    pH : float
        Bulk electrolyte pH.
    temperature_C : float
        Bath temperature (°C).
    boundary_layer_m : float
        Nernst diffusion layer thickness (m).
    bath_type : {"sulfate", "chloride"}
        Electrolyte chemistry family.
    custom_kinetics : BathKinetics or None
        Override the built-in kinetic parameter set.
    """

    fe_conc_M: float = 1.0
    cu_conc_ppm: float = 100.0
    ni_conc_ppm: float = 50.0
    zn_conc_ppm: float = 50.0
    pb_conc_ppm: float = 10.0
    sn_conc_ppm: float = 5.0
    pH: float = 3.0
    temperature_C: float = 60.0
    boundary_layer_m: float = 5e-5
    bath_type: Literal["sulfate", "chloride"] = "sulfate"
    custom_kinetics: Optional[BathKinetics] = None

    @property
    def T_K(self) -> float:
        return self.temperature_C + 273.15

    @property
    def kinetics(self) -> BathKinetics:
        if self.custom_kinetics is not None:
            return self.custom_kinetics
        return BATH_KINETICS[self.bath_type]

    # --- Concentration conversions ---

    @staticmethod
    def ppm_to_mol_per_m3(ppm: float, molar_mass_kg: float) -> float:
        """Convert ppm (mg/L) to mol/m³."""
        # 1 ppm = 1 mg/L = 1 g/m³ = 1e-3 kg/m³
        kg_per_m3 = ppm * 1e-3
        return kg_per_m3 / molar_mass_kg

    @property
    def cu_conc_mol_m3(self) -> float:
        return self.ppm_to_mol_per_m3(self.cu_conc_ppm, M_CU)

    @property
    def ni_conc_mol_m3(self) -> float:
        return self.ppm_to_mol_per_m3(self.ni_conc_ppm, M_NI)

    @property
    def zn_conc_mol_m3(self) -> float:
        return self.ppm_to_mol_per_m3(self.zn_conc_ppm, M_ZN)

    @property
    def pb_conc_mol_m3(self) -> float:
        return self.ppm_to_mol_per_m3(self.pb_conc_ppm, M_PB)

    @property
    def sn_conc_mol_m3(self) -> float:
        return self.ppm_to_mol_per_m3(self.sn_conc_ppm, M_SN)

    @property
    def fe_conc_mol_m3(self) -> float:
        return self.fe_conc_M * 1000.0

    # --- Equilibrium potentials (Nernst, activity ≈ concentration) ---

    def _equilibrium_potential(self, E0: float, conc_mol_m3: float, z: int) -> float:
        """Nernst equilibrium: E = E° + (RT/zF) ln(a_ox / a_red).
        For M2+ + 2e- → M, a_red = 1 (pure solid), a_ox = C_M in mol/m³.
        Concentration normalized to standard state 1 M = 1000 mol/m³.
        """
        a_ox = max(conc_mol_m3 / 1000.0, 1e-30)
        return E0 + (R_GAS * self.T_K / (z * FARADAY)) * np.log(a_ox)

    # --- Limiting current densities ---

    def _limiting_current(
        self, conc_mol_m3: float, diffusivity_m2_s: float, z: int
    ) -> float:
        """Mass-transport-limited current density (A/m²)."""
        effective_delta = self.boundary_layer_m / self.kinetics.conductivity_factor
        return limiting_current_density(conc_mol_m3, diffusivity_m2_s, effective_delta, z)

    # --- Butler-Volmer partial currents ---

    def _partial_current(
        self,
        E_V: float,
        conc_mol_m3: float,
        E0: float,
        i0: float,
        tafel_V: float,
        diffusivity_m2_s: float,
        z: int,
        M_species: float,
    ) -> tuple[float, float]:
        """
        Partial current density and mass deposition rate for one species.

        Returns (i_species_A_m2, mass_rate_kg_m2_s).
        Uses cathodic Tafel kinetics with Koutecky-Levich transport limit.
        """
        if conc_mol_m3 <= 0.0 or i0 <= 0.0:
            return 0.0, 0.0

        E_eq = self._equilibrium_potential(E0, conc_mol_m3, z)
        eta = E_eq - E_V  # cathodic overpotential (positive = cathodic)

        if eta <= 0.0:
            return 0.0, 0.0

        i_kin = i0 * 10.0 ** (eta / tafel_V)
        i_lim = self._limiting_current(conc_mol_m3, diffusivity_m2_s, z)

        # Koutecky-Levich
        i_species = 1.0 / (1.0 / max(i_kin, 1e-30) + 1.0 / max(i_lim, 1e-30))

        # Mass deposition rate
        molar_rate = i_species / (z * FARADAY)  # mol/(m²·s)
        mass_rate = molar_rate * M_species       # kg/(m²·s)

        return float(i_species), float(mass_rate)

    # --- Deposit composition at a fixed current density ---

    def deposit_composition(self, j_mA_cm2: float) -> Dict[str, Any]:
        """
        Predict deposit composition at a given applied current density.

        Parameters
        ----------
        j_mA_cm2 : float
            Applied cathodic current density (mA/cm²).

        Returns
        -------
        dict with keys:
            "potential_V"        — estimated cathode potential (V vs. SHE)
            "fe_wt_percent"      — Fe in deposit (wt%)
            "cu_wt_percent"      — Cu in deposit (wt%)
            "ni_wt_percent"      — Ni in deposit (wt%)
            "zn_wt_percent"      — Zn in deposit (wt%)
            "pb_wt_percent"      — Pb in deposit (wt%)
            "sn_wt_percent"      — Sn in deposit (wt%)
            "cu_in_ppm"          — Cu in deposit (ppm)
            "ni_in_ppm"          — Ni in deposit (ppm)
            "zn_in_ppm"          — Zn in deposit (ppm)
            "pb_in_ppm"          — Pb in deposit (ppm)
            "sn_in_ppm"          — Sn in deposit (ppm)
            "total_impurity_wt"  — total impurity content (wt%)
            "fe_current_A_m2"    — Fe partial current (A/m²)
            "cu_current_A_m2"    — Cu partial current (A/m²)
            "ni_current_A_m2"    — Ni partial current (A/m²)
            "zn_current_A_m2"    — Zn partial current (A/m²)
            "cu_exceeds_hot_shortness" — bool, Cu > 0.1 wt%
            "bath_type"          — "sulfate" or "chloride"
            "current_density_mA_cm2" — echo of input
        """
        j_A_m2 = j_mA_cm2 * 10.0  # mA/cm² → A/m²
        kin = self.kinetics

        # Estimate cathode potential from iron Tafel at the applied current.
        # For screening, assume the applied current ≈ Fe current (impurities
        # are trace, so their contribution to total is negligible).
        E_eq_fe = self._equilibrium_potential(E0_FE, self.fe_conc_mol_m3, Z_FE)
        # Invert Tafel: E = E_eq - beta * log10(j/i0)
        if j_A_m2 > 0 and kin.fe_i0 > 0:
            E_V = E_eq_fe - kin.fe_tafel * np.log10(max(j_A_m2 / kin.fe_i0, 1e-30))
        else:
            E_V = E_eq_fe

        # Compute partial currents for each species
        i_fe, m_fe = self._partial_current(
            E_V, self.fe_conc_mol_m3, E0_FE, kin.fe_i0, kin.fe_tafel,
            D_CU_DEFAULT, Z_FE, M_FE,  # use Fe diffusivity ≈ Cu diffusivity
        )
        # Override with Fe-specific diffusivity
        i_fe, m_fe = self._partial_current(
            E_V, self.fe_conc_mol_m3, E0_FE, kin.fe_i0, kin.fe_tafel,
            7.2e-10, Z_FE, M_FE,
        )

        i_cu, m_cu = self._partial_current(
            E_V, self.cu_conc_mol_m3, E0_CU, kin.cu_i0, kin.cu_tafel,
            D_CU_DEFAULT, Z_CU, M_CU,
        )
        i_ni, m_ni = self._partial_current(
            E_V, self.ni_conc_mol_m3, E0_NI, kin.ni_i0, kin.ni_tafel,
            D_NI_DEFAULT, Z_NI, M_NI,
        )
        i_zn, m_zn = self._partial_current(
            E_V, self.zn_conc_mol_m3, E0_ZN, kin.zn_i0, kin.zn_tafel,
            D_ZN_DEFAULT, Z_ZN, M_ZN,
        )
        i_pb, m_pb = self._partial_current(
            E_V, self.pb_conc_mol_m3, E0_PB, kin.pb_i0, kin.pb_tafel,
            D_PB_DEFAULT, Z_PB, M_PB,
        )
        i_sn, m_sn = self._partial_current(
            E_V, self.sn_conc_mol_m3, E0_SN, kin.sn_i0, kin.sn_tafel,
            D_SN_DEFAULT, Z_SN, M_SN,
        )

        total_mass = m_fe + m_cu + m_ni + m_zn + m_pb + m_sn

        if total_mass <= 0.0:
            return {
                "potential_V": E_V,
                "fe_wt_percent": 100.0,
                "cu_wt_percent": 0.0,
                "ni_wt_percent": 0.0,
                "zn_wt_percent": 0.0,
                "pb_wt_percent": 0.0,
                "sn_wt_percent": 0.0,
                "cu_in_ppm": 0.0,
                "ni_in_ppm": 0.0,
                "zn_in_ppm": 0.0,
                "pb_in_ppm": 0.0,
                "sn_in_ppm": 0.0,
                "total_impurity_wt": 0.0,
                "fe_current_A_m2": 0.0,
                "cu_current_A_m2": 0.0,
                "ni_current_A_m2": 0.0,
                "zn_current_A_m2": 0.0,
                "cu_exceeds_hot_shortness": False,
                "bath_type": self.bath_type,
                "current_density_mA_cm2": j_mA_cm2,
            }

        fe_wt = 100.0 * m_fe / total_mass
        cu_wt = 100.0 * m_cu / total_mass
        ni_wt = 100.0 * m_ni / total_mass
        zn_wt = 100.0 * m_zn / total_mass
        pb_wt = 100.0 * m_pb / total_mass
        sn_wt = 100.0 * m_sn / total_mass

        return {
            "potential_V": E_V,
            "fe_wt_percent": fe_wt,
            "cu_wt_percent": cu_wt,
            "ni_wt_percent": ni_wt,
            "zn_wt_percent": zn_wt,
            "pb_wt_percent": pb_wt,
            "sn_wt_percent": sn_wt,
            "cu_in_ppm": cu_wt * 10000.0,  # wt% → ppm
            "ni_in_ppm": ni_wt * 10000.0,
            "zn_in_ppm": zn_wt * 10000.0,
            "pb_in_ppm": pb_wt * 10000.0,
            "sn_in_ppm": sn_wt * 10000.0,
            "total_impurity_wt": cu_wt + ni_wt + zn_wt + pb_wt + sn_wt,
            "fe_current_A_m2": i_fe,
            "cu_current_A_m2": i_cu,
            "ni_current_A_m2": i_ni,
            "zn_current_A_m2": i_zn,
            "cu_exceeds_hot_shortness": cu_wt > CU_HOT_SHORTNESS_WT,
            "bath_type": self.bath_type,
            "current_density_mA_cm2": j_mA_cm2,
        }

    # --- Sweep: wt% vs concentration at multiple current densities ---

    def cu_uptake_vs_concentration(
        self,
        cu_ppm_range: np.ndarray,
        j_values: list[float],
    ) -> Dict[str, np.ndarray]:
        """
        Sweep Cu²⁺ concentration and compute deposit Cu (wt%) at each j.

        Parameters
        ----------
        cu_ppm_range : array of float
            Bath Cu²⁺ concentrations (ppm).
        j_values : list of float
            Current densities (mA/cm²).

        Returns
        -------
        dict with "cu_ppm" and one key per j value: "wt_j{int(j)}_mA_cm2"
        """
        result: Dict[str, np.ndarray] = {"cu_ppm": np.asarray(cu_ppm_range)}
        for j in j_values:
            wt_arr = []
            for cu_ppm in cu_ppm_range:
                model = ImpurityCoDeposition(
                    fe_conc_M=self.fe_conc_M,
                    cu_conc_ppm=float(cu_ppm),
                    ni_conc_ppm=self.ni_conc_ppm,
                    zn_conc_ppm=self.zn_conc_ppm,
                    pb_conc_ppm=self.pb_conc_ppm,
                    sn_conc_ppm=self.sn_conc_ppm,
                    pH=self.pH,
                    temperature_C=self.temperature_C,
                    boundary_layer_m=self.boundary_layer_m,
                    bath_type=self.bath_type,
                    custom_kinetics=self.custom_kinetics,
                )
                res = model.deposit_composition(j)
                wt_arr.append(res["cu_wt_percent"])
            key = f"wt_j{int(j)}_mA_cm2"
            result[key] = np.array(wt_arr)
        return result

    def ni_uptake_vs_concentration(
        self,
        ni_ppm_range: np.ndarray,
        j_values: list[float],
    ) -> Dict[str, np.ndarray]:
        """Sweep Ni²⁺ concentration and compute deposit Ni (wt%) at each j."""
        result: Dict[str, np.ndarray] = {"ni_ppm": np.asarray(ni_ppm_range)}
        for j in j_values:
            wt_arr = []
            for ni_ppm in ni_ppm_range:
                model = ImpurityCoDeposition(
                    fe_conc_M=self.fe_conc_M,
                    cu_conc_ppm=self.cu_conc_ppm,
                    ni_conc_ppm=float(ni_ppm),
                    zn_conc_ppm=self.zn_conc_ppm,
                    pb_conc_ppm=self.pb_conc_ppm,
                    sn_conc_ppm=self.sn_conc_ppm,
                    pH=self.pH,
                    temperature_C=self.temperature_C,
                    boundary_layer_m=self.boundary_layer_m,
                    bath_type=self.bath_type,
                    custom_kinetics=self.custom_kinetics,
                )
                res = model.deposit_composition(j)
                wt_arr.append(res["ni_wt_percent"])
            key = f"wt_j{int(j)}_mA_cm2"
            result[key] = np.array(wt_arr)
        return result

    def zn_uptake_vs_concentration(
        self,
        zn_ppm_range: np.ndarray,
        j_values: list[float],
    ) -> Dict[str, np.ndarray]:
        """Sweep Zn²⁺ concentration and compute deposit Zn (wt%) at each j."""
        result: Dict[str, np.ndarray] = {"zn_ppm": np.asarray(zn_ppm_range)}
        for j in j_values:
            wt_arr = []
            for zn_ppm in zn_ppm_range:
                model = ImpurityCoDeposition(
                    fe_conc_M=self.fe_conc_M,
                    cu_conc_ppm=self.cu_conc_ppm,
                    ni_conc_ppm=self.ni_conc_ppm,
                    zn_conc_ppm=float(zn_ppm),
                    pb_conc_ppm=self.pb_conc_ppm,
                    sn_conc_ppm=self.sn_conc_ppm,
                    pH=self.pH,
                    temperature_C=self.temperature_C,
                    boundary_layer_m=self.boundary_layer_m,
                    bath_type=self.bath_type,
                    custom_kinetics=self.custom_kinetics,
                )
                res = model.deposit_composition(j)
                wt_arr.append(res["zn_wt_percent"])
            key = f"wt_j{int(j)}_mA_cm2"
            result[key] = np.array(wt_arr)
        return result

    # --- Purification threshold ---

    def cu_purification_threshold(
        self,
        j_mA_cm2: float,
        max_cu_wt: float = CU_HOT_SHORTNESS_WT,
    ) -> Dict[str, Any]:
        """
        Find the maximum bath [Cu²⁺] (ppm) that keeps deposit Cu below threshold.

        Uses bisection on Cu concentration.

        Parameters
        ----------
        j_mA_cm2 : float
            Operating current density (mA/cm²).
        max_cu_wt : float
            Maximum allowable Cu in deposit (wt%), default 0.1.

        Returns
        -------
        dict with:
            "threshold_cu_ppm" — max bath Cu (ppm)
            "deposit_cu_wt"   — deposit Cu at threshold (wt%)
            "max_cu_wt"       — the limit used
        """
        # Check if even 1 ppm gives too much Cu
        test_low = ImpurityCoDeposition(
            fe_conc_M=self.fe_conc_M, cu_conc_ppm=1.0,
            pH=self.pH, temperature_C=self.temperature_C,
            boundary_layer_m=self.boundary_layer_m, bath_type=self.bath_type,
            custom_kinetics=self.custom_kinetics,
        )
        res_low = test_low.deposit_composition(j_mA_cm2)
        if res_low["cu_wt_percent"] >= max_cu_wt:
            return {
                "threshold_cu_ppm": 1.0,
                "deposit_cu_wt": res_low["cu_wt_percent"],
                "max_cu_wt": max_cu_wt,
            }

        # Bisect between 1 and 10000 ppm
        lo, hi = 1.0, 10000.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            model = ImpurityCoDeposition(
                fe_conc_M=self.fe_conc_M, cu_conc_ppm=mid,
                pH=self.pH, temperature_C=self.temperature_C,
                boundary_layer_m=self.boundary_layer_m, bath_type=self.bath_type,
                custom_kinetics=self.custom_kinetics,
            )
            res = model.deposit_composition(j_mA_cm2)
            if res["cu_wt_percent"] > max_cu_wt:
                hi = mid
            else:
                lo = mid

        # Final result at lo (just under threshold)
        model = ImpurityCoDeposition(
            fe_conc_M=self.fe_conc_M, cu_conc_ppm=lo,
            pH=self.pH, temperature_C=self.temperature_C,
            boundary_layer_m=self.boundary_layer_m, bath_type=self.bath_type,
            custom_kinetics=self.custom_kinetics,
        )
        res = model.deposit_composition(j_mA_cm2)
        return {
            "threshold_cu_ppm": lo,
            "deposit_cu_wt": res["cu_wt_percent"],
            "max_cu_wt": max_cu_wt,
        }


# -------------------------------------------------------------------
# Convenience: compare sulfate vs chloride
# -------------------------------------------------------------------


def compare_bath_types(
    cu_conc_ppm: float = 100.0,
    ni_conc_ppm: float = 50.0,
    zn_conc_ppm: float = 50.0,
    j_mA_cm2: float = 100.0,
    fe_conc_M: float = 1.0,
    pH: float = 3.0,
    temperature_C: float = 60.0,
    boundary_layer_m: float = 5e-5,
) -> Dict[str, Dict[str, Any]]:
    """
    Compare impurity uptake in sulfate vs chloride baths.

    Returns dict with keys "sulfate" and "chloride", each containing
    the deposit_composition result for that bath type.
    """
    results = {}
    for bath in ("sulfate", "chloride"):
        model = ImpurityCoDeposition(
            fe_conc_M=fe_conc_M,
            cu_conc_ppm=cu_conc_ppm,
            ni_conc_ppm=ni_conc_ppm,
            zn_conc_ppm=zn_conc_ppm,
            pH=pH,
            temperature_C=temperature_C,
            boundary_layer_m=boundary_layer_m,
            bath_type=bath,
        )
        results[bath] = model.deposit_composition(j_mA_cm2)
    return results
