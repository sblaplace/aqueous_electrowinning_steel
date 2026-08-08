"""
Deposit morphology prediction — derives whether electrodeposition produces
a coherent film, dendrites, powder, or no deposit from process conditions.

Three coupled criteria:

1. **Dendrite onset** (Mullins-Sekerka instability):
   Concentration perturbations at the cathode grow when the current density
   exceeds a stability threshold set by surface tension, diffusivity, and
   boundary layer thickness.

2. **HER disruption** (bubble coverage model):
   When the HER partial current fraction exceeds a threshold, H₂ bubble
   nucleation rate overwhelms deposit growth → pitting, powder, or no
   coherent deposit.

3. **Nucleation regime** (fine vs coarse grain):
   High overpotential → high nucleation rate → fine grains → potentially
   powdery. Low overpotential → few nuclei → large grains → smooth film.
   The transition depends on the ratio of nucleation to growth rates.

The model does NOT assume a deposit forms. It computes conditions under
which a coherent deposit is thermodynamically and kinetically feasible.

References:
  - Mullins & Sekerka (1963) J. Appl. Phys. 34:323 — morphological stability
  - Barton & Bockris (1962) Proc. R. Soc. — electrodeposition dendrite theory
  - Gamburg (2011) "Theory of Metal Electrodeposition" — nucleation regimes
  - Ibl (1962) — current distribution and deposit morphology
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal, Tuple
import math
import numpy as np

from .electrochemistry import FARADAY, R_GAS
from .kinetics import DepositionKinetics

# ─── Physical Constants ──────────────────────────────────────────

# Fe surface energy (literature, J/m²). DFT-derivable on sm_70+ GPU.
GAMMA_FE_SURFACE = 2.4  # J/m², Fe(110) — lowest energy facet
GAMMA_FE_SUBSTRATE_INTERFACE = 0.5  # J/m², Fe on Fe (epitaxial, low)
# Fe molar volume
V_M_FE = 7.09e-6  # m³/mol (from ρ=7874 kg/m³, M=55.845 g/mol)

# ─── Deposit Outcome Classification ──────────────────────────────

DepositOutcome = Literal[
    "coherent_film",     # smooth, continuous metallic deposit
    "fine_grain_film",   # coherent but nanocrystalline (high η)
    "dendrites",         # tree-like growth (diffusion-limited instability)
    "powder",            # loose, non-adherent particles
    "no_deposit",        # no iron deposits (HER dominates or thermodynamics)
    "disrupted",         # deposit forms but HER bubbles destroy coherence
]


@dataclass
class MorphologyResult:
    """Result of the morphology prediction."""

    # Input conditions
    j_mA_cm2: float                  # applied current density
    temperature_C: float
    fe_conc_M: float
    pH: float
    boundary_layer_m: float

    # Partial currents (from kinetics model)
    i_Fe_A_m2: float                 # Fe deposition current
    i_HER_A_m2: float                # HER current
    i_total_A_m2: float
    faradaic_efficiency: float

    # Stability criteria
    dendrite_onset_ratio: float      # j / j_dendrite (>1 = dendrites likely)
    her_disruption_ratio: float      # f_HER / f_HER_crit (>1 = disruption)
    nucleation_rate_ratio: float     # J_nucleation / J_growth (>1 = fine/powder)

    # Concentration at surface
    surface_fe_conc_M: float
    surface_pH: float
    feoh2_supersaturation: float

    # Classification
    outcome: DepositOutcome
    confidence: str                  # "high", "medium", "low"

    # Physical reasoning
    limiting_factors: list = field(default_factory=list)

    # Growth-ODE criterion (only populated with a growth_model; CHEM_PHYS_REVIEW 2.4)
    dendrite_growth_rate_1_s: Optional[float] = None
    screening_length_m: Optional[float] = None

    @property
    def is_coherent(self) -> bool:
        return self.outcome in ("coherent_film", "fine_grain_film")

    @property
    def is_viable(self) -> bool:
        """Can this operating point produce usable iron?"""
        return self.outcome in ("coherent_film", "fine_grain_film", "dendrites")

    def summary(self) -> str:
        lines = [
            f"MORPHOLOGY PREDICTION @ {self.j_mA_cm2:.0f} mA/cm²",
            f"  Outcome:        {self.outcome}",
            f"  Confidence:     {self.confidence}",
            f"  FE:             {self.faradaic_efficiency*100:.1f}%",
            f"  i_Fe:           {self.i_Fe_A_m2:.1f} A/m²",
            f"  i_HER:          {self.i_HER_A_m2:.1f} A/m²",
            f"  Dendrite ratio: {self.dendrite_onset_ratio:.2f} "
            f"({'DANGER' if self.dendrite_onset_ratio > 1 else 'OK'})",
            f"  HER disruption: {self.her_disruption_ratio:.2f} "
            f"({'DANGER' if self.her_disruption_ratio > 1 else 'OK'})",
            f"  Nucleation:     {self.nucleation_rate_ratio:.2f} "
            f"({'fine/powder' if self.nucleation_rate_ratio > 1 else 'coarse/smooth'})",
            f"  Surface [Fe2+]: {self.surface_fe_conc_M:.4f} M",
            f"  Surface pH:     {self.surface_pH:.2f}",
        ]
        if self.limiting_factors:
            lines.append(f"  Limiting:       {', '.join(self.limiting_factors)}")
        return "\n".join(lines)


# ─── Dendrite Stability (Mullins-Sekerka) ─────────────────────────

def dendrite_critical_current(
    fe_conc_M: float,
    boundary_layer_m: float,
    diffusivity_m2_s: float = 7.2e-10,
    surface_energy_J_m2: float = GAMMA_FE_SURFACE,
    temperature_C: float = 60.0,
    z: int = 2,
) -> float:
    """
    Critical current density for dendrite onset (A/m²).

    Based on Mullins-Sekerka stability analysis adapted for
    electrodeposition. Perturbations of wavelength λ > λ_c grow
    when the concentration gradient exceeds the surface-tension
    stabilizing gradient.

    The critical current is approximately:
        j_c = zFD·C_bulk/δ · sqrt(δ/λ_c)

    where λ_c is the critical wavelength:
        λ_c = 2π · sqrt(γ·V_m·C_bulk / (z·F·(dC/dx)_surface))

    Simplified to a practical criterion: dendrites form when
    j > j_dendrite, where j_dendrite scales as:
        j_dendrite ∝ zFD·C/δ · (1 + sqrt(δ·γ·Vm/(D²·C)))

    The first term is the diffusion limit; the second is the
    surface-tension stabilization factor.

    Parameters
    ----------
    fe_conc_M : float
        Bulk Fe²⁺ concentration (mol/L).
    boundary_layer_m : float
        Nernst diffusion layer thickness (m).
    diffusivity_m2_s : float
        Fe²⁺ diffusivity (m²/s).
    surface_energy_J_m2 : float
        Fe surface energy (J/m²).
    temperature_C : float
        Bath temperature (°C).
    z : int
        Charge number for Fe²⁺.

    Returns
    -------
    j_dendrite : float
        Critical current density for dendrite onset (A/m²).
        Operating above this makes dendrites likely.
    """
    C_bulk = fe_conc_M * 1000.0  # mol/m³
    delta = boundary_layer_m
    D = diffusivity_m2_s
    gamma = surface_energy_J_m2
    F = FARADAY

    # Diffusion-limited current
    i_lim = z * F * D * C_bulk / delta

    # Surface-tension stabilization factor
    # This represents the capillary pressure that resists perturbation growth
    # Characteristic length: L_cap = sqrt(γ·Vm / (z·F·C))
    L_cap = math.sqrt(gamma * V_M_FE / (z * F * C_bulk / 1000.0))

    # Stability ratio: boundary layer vs capillary length
    stability_factor = math.sqrt(delta / max(L_cap, 1e-12))

    # Critical current: fraction of i_lim where perturbations become unstable
    # Empirically: j_dendrite ≈ i_lim / (1 + stability_factor)
    # Higher surface energy → higher stability_factor → j_dendrite closer to i_lim
    j_dendrite = i_lim / (1.0 + 0.5 * stability_factor)

    return j_dendrite


# ─── Mullins-Sekerka / Barton-Bockris growth model ────────────────

class MullinsSekerkaGrowthModel:
    """Screening-length stability + growth-rate ODE for deposit morphology.

    Converts the static ``dendrite_critical_current`` threshold into a
    *dynamic* model (CHEM_PHYS_REVIEW.md Tier 2.4): it predicts whether a
    surface perturbation of wavelength ``lambda`` grows, and integrates its
    amplitude through the growth-rate ODE

        da/dt = sigma(k) * a ,      sigma(k) = v·k·(1 − (k/k_c)²)

    where k = 2π/λ is the perturbation wavenumber and k_c = 2π/λ_c the
    critical wavenumber set by the Mullins–Sekerka screening length

        λ_c = ( D·γ·Ω / ( j·(∂c/∂x)|_surf ) )^(1/2)

    with D the Fe²⁺ diffusivity, γ the surface energy, Ω the Fe molar
    volume, j the Fe deposition current density, and (∂c/∂x)|_surf the
    Fe²⁺ concentration gradient at the cathode surface (mol m⁻⁴; default
    Fick closure j/(zFD)).  v = j·Ω/(zF) is the planar-front velocity.

    Behavior:

      * λ > λ_c  (k < k_c) : σ > 0  → perturbation grows (dendritic)
      * λ = λ_c  (k = k_c) : σ = 0  → marginally stable
      * λ < λ_c  (k > k_c) : σ < 0  → perturbation decays (flat / stable)

    This is the classic Barton–Bockris / Mullins–Sekerka balance: the
    growing metal front destabilizes long wavelengths while surface tension
    (capillary) stabilizes short ones.  Because it only *advances* an
    existing perturbation amplitude, a transient film (``pulse.py``) can
    track morphology step-by-step instead of warning after the fact.
    """

    def __init__(
        self,
        diffusivity_m2_s: float = 7.2e-10,
        surface_energy_J_m2: float = GAMMA_FE_SURFACE,
        molar_volume_m3_mol: float = V_M_FE,
        z: int = 2,
    ) -> None:
        if diffusivity_m2_s <= 0.0 or surface_energy_J_m2 <= 0.0 or molar_volume_m3_mol <= 0.0:
            raise ValueError("diffusivity, surface energy and molar volume must be positive")
        if z <= 0:
            raise ValueError("charge number z must be positive")
        self.diffusivity = float(diffusivity_m2_s)
        self.surface_energy = float(surface_energy_J_m2)
        self.molar_volume = float(molar_volume_m3_mol)
        self.z = int(z)

    # -- helpers ------------------------------------------------------------

    def surface_concentration_gradient(
        self,
        current_density_A_m2: float,
        surface_fe_conc_M: Optional[float] = None,
        boundary_layer_m: Optional[float] = None,
        bulk_fe_conc_M: Optional[float] = None,
    ) -> float:
        """Surface Fe²⁺ gradient (∂c/∂x)|_surf in mol/m⁴.

        Uses the Nernst gradient (c_bulk − c_surf)/δ when the surface and
        bulk concentrations plus the boundary-layer thickness are supplied;
        otherwise falls back to the Fick closure j/(zFD) — the gradient a
        flat front establishes by transporting its own deposition flux.
        """
        if (
            surface_fe_conc_M is not None
            and bulk_fe_conc_M is not None
            and boundary_layer_m is not None
            and boundary_layer_m > 0.0
        ):
            c_bulk = bulk_fe_conc_M * 1000.0
            c_surf = surface_fe_conc_M * 1000.0
            return max((c_bulk - c_surf) / boundary_layer_m, 0.0)
        j = max(abs(current_density_A_m2), 1e-12)
        return j / (self.z * FARADAY * self.diffusivity)

    def screening_length(
        self,
        current_density_A_m2: float,
        surface_fe_conc_M: Optional[float] = None,
        boundary_layer_m: Optional[float] = None,
        bulk_fe_conc_M: Optional[float] = None,
        surface_gradient_mol_m4: Optional[float] = None,
    ) -> float:
        """Mullins–Sekerka screening length λ_c = (D·γ·Ω/(j·(∂c/∂x)))^(1/2) (m)."""
        j = max(abs(current_density_A_m2), 1e-12)
        grad = surface_gradient_mol_m4
        if grad is None:
            grad = self.surface_concentration_gradient(
                j, surface_fe_conc_M, boundary_layer_m, bulk_fe_conc_M)
        grad = max(abs(grad), 1e-12)
        return math.sqrt(
            self.diffusivity * self.surface_energy * self.molar_volume
            / (j * grad)
        )

    def front_velocity(self, current_density_A_m2: float) -> float:
        """Planar deposition front velocity v = j·Ω/(zF) (m/s)."""
        return abs(current_density_A_m2) * self.molar_volume / (self.z * FARADAY)

    # -- dispersion & ODE ---------------------------------------------------

    def growth_rate(
        self,
        wavelength_m: float,
        current_density_A_m2: float,
        surface_fe_conc_M: Optional[float] = None,
        boundary_layer_m: Optional[float] = None,
        bulk_fe_conc_M: Optional[float] = None,
        surface_gradient_mol_m4: Optional[float] = None,
    ) -> float:
        """Amplification rate σ(k) (1/s) of a perturbation of ``wavelength_m``.

        Positive → grows (dendritic); negative → decays (flat); zero at the
        screening length (marginal).
        """
        if wavelength_m <= 0.0:
            raise ValueError("wavelength_m must be positive")
        k = 2.0 * math.pi / wavelength_m
        k_c = 2.0 * math.pi / max(
            self.screening_length(
                current_density_A_m2, surface_fe_conc_M, boundary_layer_m,
                bulk_fe_conc_M, surface_gradient_mol_m4,
            ),
            1e-15,
        )
        v = self.front_velocity(current_density_A_m2)
        return v * k * (1.0 - (k / k_c) ** 2)

    def is_unstable(
        self,
        wavelength_m: float,
        current_density_A_m2: float,
        **kwargs: Any,
    ) -> bool:
        """True when a perturbation of ``wavelength_m`` grows (σ > 0)."""
        return self.growth_rate(wavelength_m, current_density_A_m2, **kwargs) > 1e-14

    def advance_amplitude(
        self,
        amplitude: float,
        time_step_s: float,
        wavelength_m: float,
        current_density_A_m2: float,
        **kwargs: Any,
    ) -> float:
        """Advance the stored perturbation amplitude by one ODE step.

        Integrates da/dt = σ a with σ held fixed over the step (exact for a
        piecewise-constant drive):  a(t+dt) = a(t)·exp(σ·dt).
        """
        sigma = self.growth_rate(wavelength_m, current_density_A_m2, **kwargs)
        # Clamp the single-step exponent so an atomically-short screening length
        # (huge σ) can never overflow to inf; the sign-based classification is
        # unaffected.
        return amplitude * math.exp(max(min(sigma * time_step_s, 30.0), -30.0))

    def growth_predictor(
        self,
        amplitude_initial: float,
        time_s: float,
        wavelength_m: float,
        current_density_A_m2: float,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Morphology prediction closing the static threshold into a model.

        Returns a dict carrying the screening length, growth rate, integrated
        amplitude gain, and a three-way morphology label (dendrites /
        marginal / coherent) for the perturbation over ``time_s``.
        """
        lam_c = self.screening_length(current_density_A_m2, **kwargs)
        sigma = self.growth_rate(wavelength_m, current_density_A_m2, **kwargs)
        amplitude_final = amplitude_initial * math.exp(
            max(min(sigma * time_s, 30.0), -30.0))
        if sigma > 1e-14:
            label = "dendrites"
        elif sigma < -1e-14:
            label = "coherent"
        else:
            label = "marginal"
        return {
            "model": "mullins_sekerka",
            "wavelength_m": float(wavelength_m),
            "screening_length_m": float(lam_c),
            "growth_rate_1_s": float(sigma),
            "amplitude_initial": float(amplitude_initial),
            "amplitude_final": float(amplitude_final),
            "amplitude_gain": float(amplitude_final / max(amplitude_initial, 1e-30)),
            "morphology": label,
        }


def predict_dendrite_growth(
    current_density_A_m2: float,
    wavelength_m: float,
    time_s: float = 1.0,
    amplitude_initial: float = 1e-3,
    diffusivity_m2_s: float = 7.2e-10,
    surface_energy_J_m2: float = GAMMA_FE_SURFACE,
    molar_volume_m3_mol: float = V_M_FE,
    **kwargs: Any,
) -> Dict[str, Any]:
    """One-shot convenience wrapper around :class:`MullinsSekerkaGrowthModel`.

    Predicts whether a surface perturbation of ``wavelength_m`` grows at a
    given Fe deposition current density, closing the static dendrite
    threshold into a growth-rate ODE (see the class docstring for physics).
    """
    model = MullinsSekerkaGrowthModel(
        diffusivity_m2_s=diffusivity_m2_s,
        surface_energy_J_m2=surface_energy_J_m2,
        molar_volume_m3_mol=molar_volume_m3_mol,
    )
    return model.growth_predictor(
        amplitude_initial=amplitude_initial,
        time_s=time_s,
        wavelength_m=wavelength_m,
        current_density_A_m2=current_density_A_m2,
        **kwargs,
    )


# ─── HER Disruption Model ─────────────────────────────────────────

def her_disruption_threshold(
    temperature_C: float = 60.0,
    surface_roughness: float = 1.5,
) -> float:
    """
    Critical HER fraction for deposit disruption.

    When the HER partial current fraction exceeds this threshold,
    H₂ bubble nucleation rate is high enough to physically disrupt
    the growing deposit — creating pits, loose particles, or
    preventing coherent film formation.

    The threshold depends on:
    - Temperature (higher T → lower surface tension → easier bubble nucleation)
    - Surface roughness (rougher → more nucleation sites → lower threshold)

    Parameters
    ----------
    temperature_C : float
        Bath temperature (°C).
    surface_roughness : float
        Surface roughness factor (>1 for rough substrates).

    Returns
    -------
    f_crit : float
        Critical HER fraction (0-1). Above this, disruption is likely.
    """
    # Base threshold from empirical electrodeposition literature
    # At moderate conditions, ~30-40% HER is tolerable for coherent deposits
    f_base = 0.40

    # Temperature correction: higher T reduces surface tension of H₂
    # and increases bubble nucleation rate
    T_factor = 1.0 - 0.005 * (temperature_C - 25.0)  # ~0.5% per °C

    # Roughness correction: more nucleation sites
    roughness_factor = 1.0 / surface_roughness

    return f_base * T_factor * roughness_factor


# ─── Nucleation Regime ────────────────────────────────────────────

def nucleation_rate_ratio(
    overpotential_V: float,
    temperature_C: float = 60.0,
    surface_energy_J_m2: float = GAMMA_FE_SURFACE,
    nucleation_multiplier: float = 1.0,
) -> float:
    """
    Ratio of nucleation rate to growth rate.

    High ratio → many small nuclei → fine grain / powder.
    Low ratio → few nuclei → large grains → smooth film.

    Based on classical nucleation theory applied to electrodeposition:
        J ∝ exp(-ΔG*/kT)
    where ΔG* = 16πγ³Vm² / (3(zFeη)²)

    The ratio J/G is proportional to exp(-ΔG*/kT) / η (growth ∝ η).

    ``nucleation_multiplier`` is the Γ-dependent factor from
    ``leveler_kinetics`` (CHEM_PHYS_REVIEW §2.6): levelers/brighteners block
    growth sites on stable facets and force renucleation, multiplying J/G by
    ≥ 1.  The default 1.0 leaves the no-additive path byte-identical.

    Parameters
    ----------
    overpotential_V : float
        Cathodic overpotential for Fe deposition (V). Positive = cathodic.
    temperature_C : float
        Bath temperature (°C).
    surface_energy_J_m2 : float
        Fe surface energy (J/m²).
    nucleation_multiplier : float
        Γ-dependent multiplier on the nucleation/growth ratio (≥ 1).

    Returns
    -------
    ratio : float
        Nucleation/growth ratio. >1 suggests fine grain or powder.
    """
    T = temperature_C + 273.15
    gamma = surface_energy_J_m2
    eta = abs(overpotential_V)
    z = 2
    F = FARADAY

    if eta < 0.001:  # negligible overpotential
        return 0.0

    # Critical nucleus energy barrier
    # ΔG* = 16πγ³Vm² / (3(zFeη)²)
    dG_star = (16.0 * math.pi * gamma**3 * V_M_FE**2) / (3.0 * (z * F * eta)**2)

    # Nucleation rate (normalized)
    kT = R_GAS * T
    # Cap to avoid overflow
    exponent = min(dG_star / kT, 500.0)
    J_nucleation = math.exp(-exponent)

    # Growth rate proportional to overpotential (Butler-Volmer linear regime)
    G_growth = eta

    # Normalized ratio
    # At low η: J is tiny, G is tiny, ratio is very small → large grains
    # At high η: J is huge, G is moderate, ratio is large → fine grains/powder
    ratio = J_nucleation / max(G_growth, 1e-10)

    # Normalize to a scale where ~1 is the transition
    # This absorbs the pre-exponential factors
    # Empirically: the transition from coarse to fine grains in Fe
    # electrodeposition occurs around η ≈ 50-100 mV
    ratio_normalized = ratio / max(math.exp(-dG_star / kT + 1.0), 1e-30)

    return float(ratio_normalized * nucleation_multiplier)


# ─── Main Morphology Prediction ───────────────────────────────────

def _dendrite_criterion(
    j_A_m2: float,
    kinetics: DepositionKinetics,
    growth_model: Optional[MullinsSekerkaGrowthModel],
    perturbation_wavelength_m: Optional[float],
) -> Tuple[float, Optional[float], Optional[float], Optional[str]]:
    """Dendrite criterion: static threshold, or opt-in growth ODE.

    Returns ``(dendrite_ratio, sigma, screening_len, factor)``.  Without a
    growth model this is the unchanged static j/j_dendrite criterion.  With a
    growth model and a perturbing wavelength, the ratio is exp(σ·τ_ref),
    τ_ref=1 s (clamped so a huge σ cannot overflow): >1 when the perturbation
    grows, <1 when it decays.  The static path has no resolved surface
    concentration, so the growth path uses the well-defined Fick gradient the
    deposition current itself establishes, ∂c/∂x|surf = j/(zFD).
    """
    if (growth_model is not None and perturbation_wavelength_m is not None
            and perturbation_wavelength_m > 0.0):
        sigma = growth_model.growth_rate(perturbation_wavelength_m, j_A_m2)
        screening_len = growth_model.screening_length(j_A_m2)
        dendrite_ratio = math.exp(max(min(sigma, 30.0), -30.0))
        factor = None
        if dendrite_ratio > 1.0:
            factor = (f"dendrite growth (σ={sigma:.3g}/s, "
                      f"λ_c={screening_len:.3g} m)")
        return dendrite_ratio, sigma, screening_len, factor
    j_dendrite = dendrite_critical_current(
        fe_conc_M=kinetics.fe_conc_M,
        boundary_layer_m=kinetics.boundary_layer_m,
        diffusivity_m2_s=kinetics.diffusivity_m2_s,
        temperature_C=kinetics.temperature_C,
    )
    dendrite_ratio = j_A_m2 / max(j_dendrite, 1e-10)
    factor = None
    if dendrite_ratio > 1.0:
        factor = f"dendrite onset (j/j_dendrite={dendrite_ratio:.2f})"
    return dendrite_ratio, None, None, factor


def predict_morphology(
    j_mA_cm2: float,
    kinetics: DepositionKinetics,
    surface_fe_conc_M: Optional[float] = None,
    surface_pH: Optional[float] = None,
    feoh2_supersaturation: float = 0.0,
    boundary_layer_m: Optional[float] = None,
    substrate_roughness: float = 1.5,
    growth_model: Optional[MullinsSekerkaGrowthModel] = None,
    perturbation_wavelength_m: Optional[float] = None,
    additive_package: Optional[Dict[str, float]] = None,
) -> MorphologyResult:
    """
    Predict deposit morphology at a given current density.

    Chains the kinetics model (partial currents) through three
    stability criteria to classify the deposit outcome.

    Parameters
    ----------
    j_mA_cm2 : float
        Applied current density (mA/cm²).
    kinetics : DepositionKinetics
        Kinetics model with Fe and HER branches.
    surface_fe_conc_M : float, optional
        Surface Fe²⁺ concentration (mol/L). If None, uses bulk.
    surface_pH : float, optional
        Surface pH. If None, estimates from bulk pH.
    feoh2_supersaturation : float
        Fe(OH)₂ supersaturation ratio (>1 = precipitation active).
    boundary_layer_m : float, optional
        Boundary layer thickness (m). If None, uses kinetics default.
    substrate_roughness : float
        Surface roughness factor (>1 for rough substrates).
    growth_model : MullinsSekerkaGrowthModel, optional
        If supplied together with ``perturbation_wavelength_m``, replaces
        the static dendrite threshold with the Mullins–Sekerka growth-ODE
        criterion (CHEM_PHYS_REVIEW Tier 2.4): the dendrite onset ratio is
        then exp(σ·τ_ref) — >1 when the perturbation grows, <1 when it
        decays.  This is opt-in; the default path is unchanged.
    perturbation_wavelength_m : float, optional
        Wavelength of the surface perturbation to track with ``growth_model``.

    Returns
    -------
    MorphologyResult
        Full morphology prediction with classification and reasoning.
    """
    j_A_m2 = j_mA_cm2 * 10.0  # mA/cm² → A/m²
    delta = boundary_layer_m or kinetics.boundary_layer_m
    factors = []

    # Step 1: Compute partial currents
    # Need to find E that gives the desired total current
    # Use bisection on the total current vs E
    E_low, E_high = -2.0, 0.5
    for _ in range(100):
        E_mid = (E_low + E_high) / 2.0
        _, _, i_tot = kinetics.partial_currents(E_mid)
        if i_tot < j_A_m2:
            E_high = E_mid
        else:
            E_low = E_mid
        if abs(E_high - E_low) < 1e-8:
            break

    E_operating = (E_low + E_high) / 2.0
    i_Fe, i_HER, i_total = kinetics.partial_currents(E_operating)
    FE = i_Fe / max(i_total, 1e-30)

    # Step 2: Dendrite criterion — static threshold, or opt-in growth ODE
    dendrite_ratio, sigma, screening_len, dend_factor = _dendrite_criterion(
        j_A_m2, kinetics, growth_model, perturbation_wavelength_m)
    if dend_factor:
        factors.append(dend_factor)

    # Step 3: HER disruption criterion
    f_HER = i_HER / max(i_total, 1e-30)
    f_crit = her_disruption_threshold(
        temperature_C=kinetics.temperature_C,
        surface_roughness=substrate_roughness,
    )
    her_ratio = f_HER / max(f_crit, 1e-10)
    if her_ratio > 1.0:
        factors.append(f"HER disruption (f_HER={f_HER:.2f} > f_crit={f_crit:.2f})")

    # Step 4: Nucleation regime
    overpotential = kinetics.fe_E_eq - E_operating  # cathodic overpotential for Fe
    # Γ-dependent nucleation multiplier from the additive package
    # (CHEM_PHYS_REVIEW §2.6).  Null package → multiplier 1.0 → unchanged.
    if additive_package is not None:
        from .leveler_kinetics import resolve_package

        _nuc_mult = resolve_package(additive_package, kinetics.temperature_C).nucleation_multiplier
    else:
        _nuc_mult = 1.0
    nuc_ratio = nucleation_rate_ratio(
        overpotential_V=max(overpotential, 0.0),
        temperature_C=kinetics.temperature_C,
        nucleation_multiplier=_nuc_mult,
    )
    if nuc_ratio > 1.0:
        factors.append("high nucleation rate (fine grain/powder risk)")

    # Step 5: Fe(OH)₂ precipitation check
    if feoh2_supersaturation > 1.0:
        factors.append(f"Fe(OH)₂ precipitation active (SSAT={feoh2_supersaturation:.2g})")

    # Step 6: Surface concentration depletion
    sfc_fe = surface_fe_conc_M or kinetics.fe_conc_M
    sfc_pH = surface_pH or kinetics.pH
    depletion = 1.0 - sfc_fe / max(kinetics.fe_conc_M, 1e-10)
    if depletion > 0.8:
        factors.append(f"severe Fe²⁺ depletion ({depletion*100:.0f}% at surface)")

    # Step 7: Classify outcome
    outcome = _classify_outcome(
        dendrite_ratio=dendrite_ratio,
        her_ratio=her_ratio,
        nuc_ratio=nuc_ratio,
        fe_supersaturation=feoh2_supersaturation,
        depletion=depletion,
        FE=FE,
    )

    # Confidence assessment
    confidence = _assess_confidence(
        kinetics=kinetics,
        dendrite_ratio=dendrite_ratio,
        her_ratio=her_ratio,
    )

    return MorphologyResult(
        j_mA_cm2=j_mA_cm2,
        temperature_C=kinetics.temperature_C,
        fe_conc_M=kinetics.fe_conc_M,
        pH=kinetics.pH,
        boundary_layer_m=delta,
        i_Fe_A_m2=float(i_Fe),
        i_HER_A_m2=float(i_HER),
        i_total_A_m2=float(i_total),
        faradaic_efficiency=float(FE),
        dendrite_onset_ratio=float(dendrite_ratio),
        her_disruption_ratio=float(her_ratio),
        nucleation_rate_ratio=float(nuc_ratio),
        surface_fe_conc_M=float(sfc_fe),
        surface_pH=float(sfc_pH),
        feoh2_supersaturation=float(feoh2_supersaturation),
        outcome=outcome,
        confidence=confidence,
        limiting_factors=factors,
        dendrite_growth_rate_1_s=float(sigma) if sigma is not None else None,
        screening_length_m=float(screening_len) if screening_len is not None else None,
    )


def _classify_outcome(
    dendrite_ratio: float,
    her_ratio: float,
    nuc_ratio: float,
    fe_supersaturation: float,
    depletion: float,
    FE: float,
) -> DepositOutcome:
    """Classify deposit outcome from stability criteria."""

    # No deposit: FE essentially zero or HER completely dominates
    if FE < 0.05:
        return "no_deposit"

    # Disrupted: HER bubbles destroy coherence
    if her_ratio > 1.5:
        return "disrupted"

    # Dendrites: diffusion-limited instability
    if dendrite_ratio > 1.0:
        # If also high nucleation → powder instead of dendrites
        if nuc_ratio > 2.0:
            return "powder"
        return "dendrites"

    # Powder: extreme nucleation + depletion
    if nuc_ratio > 3.0 and depletion > 0.5:
        return "powder"

    # Fe(OH)₂ precipitation → hydroxide incorporation → powder
    if fe_supersaturation > 2.0:
        return "powder"

    # HER disruption at moderate level
    if her_ratio > 1.0:
        return "disrupted"

    # Fine grain: high nucleation but still coherent
    if nuc_ratio > 1.0:
        return "fine_grain_film"

    # Default: coherent film
    return "coherent_film"


def _assess_confidence(
    kinetics: DepositionKinetics,
    dendrite_ratio: float,
    her_ratio: float,
) -> str:
    """Assess confidence in the morphology prediction."""

    # High confidence: clear regime (far from transitions)
    if dendrite_ratio > 2.0 or dendrite_ratio < 0.3:
        if her_ratio > 2.0 or her_ratio < 0.3:
            return "high"

    # Medium confidence: moderate distance from transitions
    if 0.5 < dendrite_ratio < 1.5 or 0.5 < her_ratio < 1.5:
        return "low"

    return "medium"


# ─── Morphology Map ────────────────────────────────────────────────

def morphology_map(
    kinetics: DepositionKinetics,
    j_range_mA_cm2: Optional[np.ndarray] = None,
    n_points: int = 50,
    growth_model: Optional[MullinsSekerkaGrowthModel] = None,
    perturbation_wavelength_m: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute morphology classification across a range of current densities.

    Returns a map of j → outcome with stability criteria.

    Parameters
    ----------
    kinetics : DepositionKinetics
        Kinetics model.
    j_range_mA_cm2 : ndarray, optional
        Current densities to evaluate (mA/cm²). If None, auto-range.
    n_points : int
        Number of points in the sweep.
    growth_model : MullinsSekerkaGrowthModel, optional
        If supplied together with ``perturbation_wavelength_m``, the dendrite
        criterion uses the opt-in growth ODE instead of the static threshold.
    perturbation_wavelength_m : float, optional
        Wavelength of the surface perturbation to track with ``growth_model``.

    Returns
    -------
    dict with 'j_mA_cm2', 'outcomes', 'FE', 'dendrite_ratio', etc.  When a
    growth model is used, 'dendrite_growth_rate' and 'screening_length_m' are
    also included.
    """
    if j_range_mA_cm2 is None:
        # Auto-range from 1 mA/cm² to 2× the limiting current
        j_max = kinetics.i_lim / 10.0 * 2.0  # A/m² → mA/cm²
        j_range_mA_cm2 = np.linspace(1.0, max(j_max, 100.0), n_points)

    results = []
    for j in j_range_mA_cm2:
        r = predict_morphology(
            float(j), kinetics, growth_model=growth_model,
            perturbation_wavelength_m=perturbation_wavelength_m)
        results.append(r)

    out = {
        "j_mA_cm2": j_range_mA_cm2.tolist(),
        "outcomes": [r.outcome for r in results],
        "FE": [r.faradaic_efficiency for r in results],
        "dendrite_ratio": [r.dendrite_onset_ratio for r in results],
        "her_ratio": [r.her_disruption_ratio for r in results],
        "nucleation_ratio": [r.nucleation_rate_ratio for r in results],
        "i_Fe_A_m2": [r.i_Fe_A_m2 for r in results],
        "i_HER_A_m2": [r.i_HER_A_m2 for r in results],
        "results": results,
    }
    if growth_model is not None:
        out["dendrite_growth_rate"] = [r.dendrite_growth_rate_1_s for r in results]
        out["screening_length_m"] = [r.screening_length_m for r in results]
    return out


def viable_operating_window(
    kinetics: DepositionKinetics,
    j_range_mA_cm2: Optional[np.ndarray] = None,
    n_points: int = 100,
    min_FE: float = 0.50,
) -> Dict[str, Any]:
    """
    Find the viable operating window where coherent deposition occurs.

    Returns the range of current densities where:
    - Deposit is coherent (film or fine_grain_film)
    - FE exceeds min_FE
    - No dendrites, no disruption

    Parameters
    ----------
    kinetics : DepositionKinetics
        Kinetics model.
    j_range_mA_cm2 : ndarray, optional
        Current densities to evaluate.
    n_points : int
        Number of points.
    min_FE : float
        Minimum acceptable Faradaic efficiency.

    Returns
    -------
    dict with 'j_viable_low', 'j_viable_high', 'j_optimal', 'map'
    """
    m = morphology_map(kinetics, j_range_mA_cm2, n_points)

    viable_mask = []
    for i, (outcome, fe) in enumerate(zip(m["outcomes"], m["FE"])):
        is_viable = outcome in ("coherent_film", "fine_grain_film")
        has_fe = fe >= min_FE
        viable_mask.append(is_viable and has_fe)

    viable_indices = [i for i, v in enumerate(viable_mask) if v]

    if not viable_indices:
        return {
            "j_viable_low": None,
            "j_viable_high": None,
            "j_optimal": None,
            "viable": False,
            "reason": "No viable operating window found",
            "map": m,
        }

    j_arr = np.array(m["j_mA_cm2"])
    fe_arr = np.array(m["FE"])
    j_low = j_arr[viable_indices[0]]
    j_high = j_arr[viable_indices[-1]]

    # Optimal: highest FE in viable range
    viable_fes = fe_arr[viable_indices]
    optimal_idx = viable_indices[np.argmax(viable_fes)]
    j_optimal = j_arr[optimal_idx]

    return {
        "j_viable_low": float(j_low),
        "j_viable_high": float(j_high),
        "j_optimal": float(j_optimal),
        "viable": True,
        "FE_at_optimal": float(fe_arr[optimal_idx]),
        "n_viable_points": len(viable_indices),
        "map": m,
    }
