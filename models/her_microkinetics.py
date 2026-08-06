"""
DFT-anchored HER microkinetics on Fe — consistency check, not a replacement.

Why this module exists
----------------------
The empirical HER branch (``DepositionKinetics``: i0 + fixed 140 mV/dec
Tafel slope) is the model's default at all conditions.  This module builds
the same branch from a mean-field Volmer–Heyrovský picture anchored on the
DFT hydrogen-adsorption free energy ΔG_H*, and asks the only question a
screening model should ask: **does the DFT-anchored microkinetic branch
reproduce the empirical one within its own uncertainty?**  If yes — it
does, within ~20 % on the slope at reference conditions — the DFT picture
*supports* the screening choice and, more usefully, tells us which way the
empirical parameters should move with T and pH when no data are fitted.

Model
-----
Volmer quasi-equilibrium (computational hydrogen electrode):
    H⁺ + e⁻ + * ⇌ H*,      θ/(1−θ) = exp(−(ΔG_H* + eU)/kT),  U vs RHE
On Fe(110) ΔG_H* ≈ −0.40 eV, so θ_H is pinned at ≈1 over the whole
cathodic window of interest and the Tafel step is suppressed, leaving
    Heyrovský RDS:  H* + H⁺ + e⁻ → H₂ + *
    i = 2F · k_Hey(T) · θ_H · a_H+ · exp(+αFη/RT)
which at θ≈1 is exactly a Tafel law with b = 2.303RT/(αF) = 118 mV/dec
(25 °C, α=0.5), i.e. the slope α=0.5 charge transfer always gives at high
coverage — and the reason iron-group HER measures 110–140 mV/dec.  The
intrinsic rate coefficient k_Hey(T) carries an Arrhenius activation
barrier; at L0 it inherits the empirical branch's apparent Ea (60 kJ/mol).

ΔG_H* source: DFT on bcc Fe low-index facets — atomic-H adsorption on
Fe(110) (bridge/hollow sites) converts to ΔG_H* ≈ −0.3…−0.7 eV vs ½H₂;
Nørskov-family volcano compilations place Fe at ≈ −0.4 eV.  Screening
central value −0.40 eV, flagged range −0.30…−0.55 eV.  This is NOT fitted
to our bath and carries the usual CHE/RPA caveats — hence "unvalidated".

Anchor protocol: k_Hey is fixed by matching the empirical branch at ONE
reference state (kinetics_ref_K, pH_ref, η_ref).  Consistency is then
reported at other T/pH — where the only physics content of the module is
the θ-corrected α=0.5 form with explicit a_H+ and RT/F factors.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from .electrochemistry import FARADAY, R_GAS

# 1 eV per particle ≡ F J per mole (per-mole CHE bookkeeping).
EV_J_MOL = 96485.0

# DFT ΔG_H* on Fe(110), screening central value (J/mol) and flagged range.
DG_HSTAR_FE110_J = -0.40 * EV_J_MOL
DG_HSTAR_RANGE_J = (-0.55 * EV_J_MOL, -0.30 * EV_J_MOL)

ALPHA_HEY = 0.5            # charge-transfer symmetry of the Heyrovský step
SCREENING_FLAG = "unvalidated (L0)"


def hydrogen_coverage(U_vs_RHE_V: float, T_K: float,
                      dg_hstar_J: float = DG_HSTAR_FE110_J) -> float:
    """Volmer quasi-equilibrium H coverage θ_H (CHE)."""
    x = -(dg_hstar_J + FARADAY * U_vs_RHE_V) / (R_GAS * T_K)
    x = max(min(x, 60.0), -60.0)
    return 1.0 / (1.0 + math.exp(-x))


def microkinetic_tafel_slope_V(T_K: float, alpha: float = ALPHA_HEY) -> float:
    """High-coverage Heyrovský-RDS slope b = 2.303RT/(αF) (V/decade)."""
    return math.log(10.0) * R_GAS * T_K / (alpha * FARADAY)


@dataclass(frozen=True)
class HeyrovskyBranch:
    """θ-corrected Heyrovský HER current, anchored to an empirical i0.

    ``k_hey`` is solved at construction so that ``current`` at the anchor
    state equals ``anchor_current_A_m2`` — the branch never invents a
    prefactor; it only propagates the empirical one through the
    microkinetic T/pH/θ form.  The intrinsic rate coefficient carries the
    same apparent activation energy as the empirical branch, so T-offsets
    isolate the slope-form difference rather than being swamped by the
    (shared) Arrhenius barrier.
    """
    k_hey: float
    dg_hstar_J: float = DG_HSTAR_FE110_J
    alpha: float = ALPHA_HEY
    Ea_J_mol: float = 60_000.0      # inherits the empirical HER apparent Ea
    T_ref_K: float = 323.15         # anchor temperature

    def current(self, U_vs_RHE_V: float, a_h: float, T_K: float) -> float:
        """HER current density (A/m², cathodic positive)."""
        theta = hydrogen_coverage(U_vs_RHE_V, T_K, self.dg_hstar_J)
        eta = -U_vs_RHE_V  # RHE frame: cathodic overpotential = −U
        k_T = self.k_hey * math.exp(
            (self.Ea_J_mol / R_GAS) * (1.0 / self.T_ref_K - 1.0 / T_K)
        )
        return (2.0 * FARADAY) * k_T * theta * a_h * math.exp(
            self.alpha * FARADAY * eta / (R_GAS * T_K)
        )


def anchor_heyrovsky(anchor_current_A_m2: float, pH_ref: float,
                     eta_ref_V: float, T_ref_K: float,
                     dg_hstar_J: float = DG_HSTAR_FE110_J,
                     Ea_J_mol: float = 60_000.0) -> HeyrovskyBranch:
    """Fix k_Hey from one empirical current at (pH_ref, η_ref, T_ref)."""
    theta = hydrogen_coverage(-eta_ref_V, T_ref_K, dg_hstar_J)
    a_h = 10.0 ** (-pH_ref)
    denom = (2.0 * FARADAY) * theta * a_h * math.exp(
        ALPHA_HEY * FARADAY * eta_ref_V / (R_GAS * T_ref_K)
    )
    return HeyrovskyBranch(k_hey=anchor_current_A_m2 / denom,
                           dg_hstar_J=dg_hstar_J, Ea_J_mol=Ea_J_mol,
                           T_ref_K=T_ref_K)


def consistency_report(kinetics) -> dict:
    """Compare the DFT-anchored Heyrovský branch with the empirical branch.

    Anchors at the kinetics reference state (kinetics_ref_K, bulk pH,
    η_ref = 0.2 V vs RHE) and reports the slope similarity and the
    current ratio at two OFF-anchor temperatures.  Pure screening
    consistency — not gate evidence.
    """
    T_ref = kinetics.kinetics_ref_K
    pH = kinetics.pH
    eta_ref = 0.20
    # Empirical HER branch current at the anchor state:
    E_eq_ref = float(kinetics.her_branch.E_eq)  # vs SHE at (pH, T_ref)
    i_ref = kinetics.her_i0_T * 10.0 ** (eta_ref / kinetics.her_tafel_V)
    branch = anchor_heyrovsky(i_ref, pH, eta_ref, T_ref,
                              Ea_J_mol=kinetics.her_i0_Ea_J_mol)
    slope_micro = microkinetic_tafel_slope_V(T_ref)
    slope_emp = kinetics.her_tafel_V
    theta_operating = hydrogen_coverage(-0.3, T_ref)

    def ratio_at(T_C: float) -> float:
        T_K = T_C + 273.15
        kin = type(kinetics)(**{**kinetics.__dict__, "temperature_C": T_C})
        a_h = 10.0 ** (-pH)
        eta = 0.20
        i_emp = kin.her_i0_T * 10.0 ** (eta / kin.her_tafel_V)
        i_mic = branch.current(-eta, a_h, T_K)
        return float(i_mic / i_emp)

    return {
        "slope_microkinetic_V": slope_micro,
        "slope_empirical_V": slope_emp,
        "slope_ratio": slope_micro / slope_emp,
        "theta_H_operating": theta_operating,
        "i_ratio_25C": ratio_at(25.0),
        "i_ratio_70C": ratio_at(70.0),
        "dg_hstar_J_mol": DG_HSTAR_FE110_J,
        "dg_hstar_range_J_mol": DG_HSTAR_RANGE_J,
        "verdict": (
            "DFT-anchored Volmer-Heyrovský branch reproduces the empirical "
            "slope within ~20% at the reference state (θ_H≈1); use the "
            "empirical branch for operation, this module for T/pH direction "
            "and credibility. Anchor-state E_eq reference: " + f"{E_eq_ref:.3f} V."
        ),
        "flag": SCREENING_FLAG,
    }
