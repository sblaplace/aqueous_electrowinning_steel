"""Oxygen in the electrodeposited iron — a deposit-quality / density driver.

Why this module exists
----------------------
The mechanical model treats the deposit as solid-solution Fe alloyed with Ni
and C (``mechanical_properties.py``).  Real electrolytic iron always also
carries oxygen, which enters the growing deposit as co-deposited
Fe(OH)₂ / FeOOH inclusion particles rather than as a substitutional alloying
element.  Per CHEM_PHYS_REVIEW.md §1.2:

* O in the deposit sets the **upper-bound yield strength** (a 1 000 ppm O
  deposit is hard and brittle) and
* O controls whether the as-deposited foil can be **cold-rolled at all**
  (oxide network → edge cracking), independent of the Ni/C chemistry.

This module is the ``oxygen_in_iron.py`` analogue of the existing
``hydrogen_embrittlement.py``: it converts a precipitation flux into an
oxygen content, then feeds that content into (a) an upper-bound yield
strengthening term and (b) a cold-roll "can this as-deposited foil be
rolled" gate.

Coupling to the pulse waveform
------------------------------
The oxygen budget is driven by the Fe(OH)₂/FeOOH precipitation flux coming
out of the diffusion-layer precipitation / pH model.  Two entry points:

* ``oxygen_in_iron(precipitation_flux_mol_m2_s, ...)`` — the direct path.
  Feed it ``DiffusionLayer1D.solve(j).precipitation_flux_mol_m2_s`` (or
  equivalently ``CellPhysics``'s ``precipitation_flux_mol_m2_s``) and it
  returns the O budget and its consequences.

* ``OxygenInIronModel.predict(...)`` — the ``hydrogen_embrittlement``-style
  wrapper.  When no ``precipitation_flux`` is supplied it *estimates* one
  from the pulse waveform via ``co_deposition.surface_pH_from_pulse`` using
  the exact same closed-form trailing flux estimate as
  ``diffusion_layer_1d``, so a DC vs PE vs PRE waveform plugs in directly.

This is a screening module.  All numbers carry
``SCREENING_FLAG = "unvalidated (L1)"`` and are **not** gate evidence; O
content must be confirmed by inert-gas-fusion (LECO) analysis before the
cold-roll gate is treated as measured.

References (screening calibrations)
-----------------------------------
* O in electrolytic iron: 500–2 000 ppm O is typical of un-levelled aqueous
  Fe electrodeposits; > 1 000 ppm embrittles the foil for cold rolling
  (Brenner 1963; Schlesinger & Paunovic 2010).
* Fe(OH)₂ / FeOOH inclusion capture: a fraction of the precipitation flux
  (Fe(OH)₂, M = 89.86 g/mol; FeOOH, M = 88.85 g/mol) is mechanically
  enclosed during growth instead of being swept / redissolved.
* Oxide dispersion strengthens (upper-bound hardening): dispersion of hard
  nanometric oxides raises yield ~ 10² MPa per 1 000 ppm O
  (Orowan-type; screened central value).
* Density: iron oxides are less dense than Fe (FeOOH ≈ 4.2 g/cm³,
  Fe₃O₄ ≈ 5.2 g/cm³ vs Fe 7.87 g/cm³), so O lowers average deposit density.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Literal, Optional, Tuple
import math

from .electrochemistry import FARADAY, M_FE, RHO_FE
from .co_deposition import surface_pH_from_pulse
from .pourbaix import ksp_feoh2

# ── Honesty flag ─────────────────────────────────────────────────────────────
SCREENING_FLAG = "unvalidated (L1)"

# ── Physical constants ───────────────────────────────────────────────────────
M_O_KG = 16.0e-3              # kg/mol, atomic oxygen
N_O_PER_FE = 2.0              # O atoms per precipitated Fe (Fe(OH)₂ & FeOOH both carry 2 O)
M_FEOH2 = 89.86e-3            # kg/mol
M_FEOOH = 88.85e-3            # kg/mol
# O mass fraction within the as-captured oxide (FeOOH dominant; O/FeOOH = 32/88.85)
O_FRAC_IN_OXIDE = 0.36

RHO_FEOOH = 4200.0            # kg/m³ (goethite)
RHO_FE3O4 = 5200.0            # kg/m³ (magnetite)
RHO_OXIDE_DEFAULT = RHO_FEOOH

# Entrapment: fraction of the *precipitation flux* that becomes an enclosed
# inclusion in the deposit rather than evolving as sludge / being redissolved.
INC_CAPTURE_FRACTION = 0.10

# Upper-bound yield strengthening from oxide dispersion (MPa per 1 000 ppm O)
K_O_MPA_PER_KPPM = 120.0
O_STRENGTHENING_EXP = 0.8

# Cold-rollability gates (screening central values)
COLD_ROLL_FREE_O_PPM = 400.0    # below this: freely cold-rollable
COLD_ROLL_FORBIDDEN_O_PPM = 1000.0  # above this: not cold-rollable as-deposited

# Ceiling on the estimated precipitation flux from a pulse waveform: at most
# this fraction of the total Fe deposition flux precipitates into the film
# (a well-run bath deposits most Fe as metal).  Anchors the O budget at a
# physically plausible few-hundreds-to-low-thousands ppm level rather than the
# unbounded volumetric estimate.
MAX_PRECIP_FRAC_OF_FE_FLUX = 0.02


# ── Oxygen budget from a precipitation flux ─────────────────────────────────


def oxygen_mass_rate_kg_m2_s(
    precipitation_flux_mol_m2_s: float,
    capture_fraction: float = INC_CAPTURE_FRACTION,
) -> float:
    """Oxygen mass deposition rate (kg/m²/s) from a Fe(OH)₂/FeOOH flux.

    Each mole of precipitated Fe carries ``N_O_PER_FE`` oxygen atoms into the
    inclusion; only ``capture_fraction`` of the precipitation flux is
    mechanically enclosed in the growing deposit.
    """
    if precipitation_flux_mol_m2_s <= 0.0:
        return 0.0
    return float(
        max(precipitation_flux_mol_m2_s, 0.0)
        * max(capture_fraction, 0.0)
        * N_O_PER_FE * M_O_KG
    )


def iron_deposition_rate_kg_m2_s(
    j_mA_cm2: float,
    current_efficiency_percent: float = 100.0,
) -> float:
    """Metallic Fe deposition rate (kg/m²/s) by Faraday's law (Fe²⁺ → Fe)."""
    if j_mA_cm2 <= 0.0:
        return 0.0
    j_A_m2 = j_mA_cm2 * 10.0
    ce = max(min(current_efficiency_percent, 100.0), 0.0) / 100.0
    return float(j_A_m2 * ce * M_FE / (2.0 * FARADAY))


def oxygen_wt_percent(
    precipitation_flux_mol_m2_s: float,
    j_mA_cm2: float,
    current_efficiency_percent: float = 100.0,
    capture_fraction: float = INC_CAPTURE_FRACTION,
) -> float:
    """Oxygen content (wt%) of the deposit from a precipitation flux."""
    m_o = oxygen_mass_rate_kg_m2_s(precipitation_flux_mol_m2_s, capture_fraction)
    m_fe = iron_deposition_rate_kg_m2_s(j_mA_cm2, current_efficiency_percent)
    total = m_o + m_fe
    if total <= 0.0:
        return 0.0
    return float(100.0 * m_o / total)


def oxygen_ppm(
    precipitation_flux_mol_m2_s: float,
    j_mA_cm2: float,
    current_efficiency_percent: float = 100.0,
    capture_fraction: float = INC_CAPTURE_FRACTION,
) -> float:
    """Oxygen content (ppm by mass) of the deposit from a precipitation flux."""
    return oxygen_wt_percent(
        precipitation_flux_mol_m2_s, j_mA_cm2,
        current_efficiency_percent, capture_fraction,
    ) * 1e4


def deposit_density_kg_m3(
    o_ppm: float,
    oxide_density_kg_m3: float = RHO_OXIDE_DEFAULT,
) -> float:
    """Average deposit density (kg/m³) given O, via inclusion volume fraction.

    Oxygen rides in the oxide inclusion (FeOOH / Fe(OH)₂); back out the oxide
    mass fraction as ``O / O_FRAC_IN_OXIDE``, form its volume fraction by a
    rule of mixtures with metallic Fe, and take the volume-weighted density.
    """
    o_wt = max(o_ppm, 0.0) * 1e-6
    if o_wt <= 0.0:
        return float(RHO_FE)
    w_oxide = o_wt / O_FRAC_IN_OXIDE
    vol_oxide = w_oxide / oxide_density_kg_m3
    vol_fe = (1.0 - w_oxide) / RHO_FE
    total = vol_oxide + vol_fe
    if total <= 0.0:
        return float(RHO_FE)
    return float((vol_oxide * oxide_density_kg_m3 + vol_fe * RHO_FE) / total)


# ── Mechanical consequences ─────────────────────────────────────────────────


def oxygen_strengthening_MPa(o_ppm: float) -> float:
    """Upper-bound yield-strength contribution (MPa) from oxide dispersion.

    Screening Orowan-type hardening of a hard oxide dispersion in ferrite:
    roughly linear-ish in O content with mild sub-linear saturation.  This is
    the *upper bound* the deposit carries (it also embrittles it for rolling —
    see :func:`cold_rollability`).
    """
    o = max(o_ppm, 0.0)
    if o <= 0.0:
        return 0.0
    return float(K_O_MPA_PER_KPPM * (o / 1000.0) ** O_STRENGTHENING_EXP)


def cold_rollability(
    o_ppm: float,
    yield_MPa: Optional[float] = None,
    free_o_ppm: float = COLD_ROLL_FREE_O_PPM,
    forbidden_o_ppm: float = COLD_ROLL_FORBIDDEN_O_PPM,
) -> Dict[str, Any]:
    """Verdict on whether the as-deposited foil can be cold-rolled.

    The oxygen-based rolling ceiling: below ``free_o_ppm`` the foil is freely
    cold-rollable on O grounds; above ``forbidden_o_ppm`` it is not (oxide
    network → edge cracking).  In between it is marginal.  ``yield_MPa``, if
    given, is reported alongside for context but the O gate is the primary
    discriminator (a high-strength / hard deposit is brittle even if the O is
    modest).

    Returns
    -------
    dict with ``rollable`` (bool), ``status`` in {"free", "marginal",
    "forbidden"}, ``o_ppm`` and the thresholds used.
    """
    o = max(o_ppm, 0.0)
    if o < free_o_ppm:
        status, rollable = "free", True
    elif o > forbidden_o_ppm:
        status, rollable = "forbidden", False
    else:
        status, rollable = "marginal", False  # treat marginal as not free-rollable
    return {
        "rollable": rollable,
        "status": status,
        "o_ppm": o,
        "free_o_ppm": free_o_ppm,
        "forbidden_o_ppm": forbidden_o_ppm,
        "yield_MPa": yield_MPa,
        "flag": SCREENING_FLAG,
    }


# ── Pulse-waveform coupling ─────────────────────────────────────────────────


def precipitation_flux_from_pulse(
    j_avg_mA_cm2: float,
    j_peak_mA_cm2: Optional[float] = None,
    duty_cycle: float = 0.5,
    waveform: Literal["dc", "pe", "pre"] = "pe",
    bath_pH: float = 3.5,
    fe_surface_M: float = 0.9,
    temperature_C: float = 60.0,
    buffer_capacity_M: float = 0.05,
    boundary_layer_m: float = 5e-5,
) -> Tuple[float, float]:
    """Estimate the Fe(OH)₂ precipitation flux from a pulse waveform.

    Uses ``co_deposition.surface_pH_from_pulse`` for the surface pH, then the
    same closed-form supersaturation estimate as ``diffusion_layer_1d``:

      S        = [Fe²⁺][OH⁻]² / Ksp      (surface supersaturation)
      frac     = MAX_PRECIP_FRAC · (1 − e^{−(S−1)})
      flux     = (j/(2F)) · frac           (mol/m²/s)

    The precipitation flux is a saturating fraction of the Fe deposition
    flux, so it cannot outrun deposition (a well-run bath deposits most Fe
    as metal) while still preserving the waveform/pH dependence of the
    surface pH.  Returns ``(precipitation_flux_mol_m2_s, surface_pH)``.
    """
    if j_peak_mA_cm2 is None:
        # For a pulse the peak exceeds the average; recover the implied peak
        # from the duty cycle (j_avg ≈ j_peak · duty).  DC keeps peak = avg.
        if waveform == "dc" or duty_cycle <= 0.0:
            j_peak_mA_cm2 = j_avg_mA_cm2
        else:
            j_peak_mA_cm2 = j_avg_mA_cm2 / max(duty_cycle, 1e-6)

    pH_surf = surface_pH_from_pulse(
        j_avg_mA_cm2, j_peak_mA_cm2, duty_cycle, bath_pH,
        waveform, buffer_capacity_M, temperature_C, boundary_layer_m,
    )

    Ksp = ksp_feoh2(temperature_C + 273.15)
    oh_surf_M = 10.0 ** (pH_surf - 14.0)
    # Surface supersaturation ratio S = [Fe²⁺][OH⁻]² / Ksp
    supersat = fe_surface_M * oh_surf_M ** 2 / max(Ksp, 1e-30)

    if supersat <= 1.0:
        return 0.0, float(pH_surf)

    # Saturating precipitation ramp: at most ``MAX_PRECIP_FRAC_OF_FE_FLUX`` of
    # the Fe deposition flux precipitates, approached smoothly as the surface
    # supersaturation rises.  This keeps the O budget at a plausible
    # low-hundreds-to-thousands ppm level AND preserves the waveform/pH
    # dependence of ``surface_pH_from_pulse`` (the fully-coupled film model in
    # ``diffusion_layer_1d`` supersedes this when its flux is supplied).
    fe_flux_mol_m2_s = (j_avg_mA_cm2 * 10.0) / (2.0 * FARADAY)
    frac = MAX_PRECIP_FRAC_OF_FE_FLUX * (1.0 - math.exp(-(supersat - 1.0)))
    flux = fe_flux_mol_m2_s * max(frac, 0.0)
    return float(flux), float(pH_surf)


# ── Model class (hydrogen_embrittlement-style wrapper) ──────────────────────


@dataclass
class OxygenInIronParams:
    """Screening parameters for the oxygen-in-iron model.

    All values are ``unvalidated (L1)`` central estimates; replace with
    inert-gas-fusion (LECO) measured O once deposit samples exist.
    """

    capture_fraction: float = INC_CAPTURE_FRACTION
    oxide_density_kg_m3: float = RHO_OXIDE_DEFAULT
    k_o_MPa_per_kppm: float = K_O_MPA_PER_KPPM
    o_strengthening_exp: float = O_STRENGTHENING_EXP
    free_o_ppm: float = COLD_ROLL_FREE_O_PPM
    forbidden_o_ppm: float = COLD_ROLL_FORBIDDEN_O_PPM

    def __post_init__(self) -> None:
        if not 0.0 <= self.capture_fraction <= 1.0:
            raise ValueError("capture_fraction must be in [0, 1]")
        if self.oxide_density_kg_m3 <= 0:
            raise ValueError("oxide_density_kg_m3 must be positive")
        if not 0 < self.free_o_ppm <= self.forbidden_o_ppm:
            raise ValueError("must have 0 < free_o_ppm <= forbidden_o_ppm")


class OxygenInIronModel:
    """Oxygen budget and its deposit-quality consequences for a pulse waveform.

    ``hydrogen_embrittlement.py``-style wrapper: takes an operating point
    (j / waveform / bath) and returns the O budget, the density, the
    upper-bound yield strengthening and the cold-rollability verdict.

    Parameters
    ----------
    params : OxygenInIronParams
        Screening constants (defaults are ``unvalidated (L1)``).

    Example
    -------
    >>> model = OxygenInIronModel()
    >>> r = model.predict(j_avg_mA_cm2=100, waveform="pe", bath_pH=3.5)
    >>> r["o_ppm"], r["cold_rollable"], r["flag"]
    """

    def __init__(self, params: Optional[OxygenInIronParams] = None) -> None:
        self.params = params or OxygenInIronParams()

    def predict(
        self,
        j_avg_mA_cm2: float = 100.0,
        j_peak_mA_cm2: Optional[float] = None,
        duty_cycle: float = 0.5,
        waveform: Literal["dc", "pe", "pre"] = "pe",
        bath_pH: float = 3.5,
        temperature_C: float = 60.0,
        current_efficiency_percent: float = 90.0,
        buffer_capacity_M: float = 0.05,
        boundary_layer_m: float = 5e-5,
        fe_surface_M: float = 0.9,
        precipitation_flux_mol_m2_s: Optional[float] = None,
        yield_MPa: Optional[float] = None,
        include_edge_effect: bool = False,
        edge_params: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run the oxygen-in-iron pipeline at an operating point.

        If ``precipitation_flux_mol_m2_s`` is given (e.g. from
        ``DiffusionLayer1D.solve(j).precipitation_flux_mol_m2_s``) it is used
        directly; otherwise it is estimated from the waveform via
        :func:`precipitation_flux_from_pulse`.

        Returns a dict with ``o_ppm``, ``deposit_density_kg_m3``,
        ``delta_strength_MPa``, ``yield_upper_bound_MPa`` (if base given),
        ``cold_rollability`` verdict, and the ``flag``.
        """
        if precipitation_flux_mol_m2_s is None:
            precip, _ph = precipitation_flux_from_pulse(
                j_avg_mA_cm2, j_peak_mA_cm2, duty_cycle, waveform,
                bath_pH, fe_surface_M, temperature_C, buffer_capacity_M,
                boundary_layer_m,
            )
        else:
            precip = float(precipitation_flux_mol_m2_s)

        p = self.params
        o_ppm = oxygen_ppm(
            precip, j_avg_mA_cm2, current_efficiency_percent, p.capture_fraction
        )
        dens = deposit_density_kg_m3(o_ppm, p.oxide_density_kg_m3)
        d_sigma = oxygen_strengthening_MPa(o_ppm)

        # Round 5 (D2): edge/terminal current crowding concentrates O (and H) at
        # the foil edges, which is the binding constraint for cold rolling.
        edge = None
        roll_o_ppm = o_ppm
        if include_edge_effect:
            from .edge_effect import edge_oh_penalty

            eo = edge_oh_penalty(center_O_ppm=o_ppm, params=edge_params)
            edge = eo
            roll_o_ppm = eo["edge_O_ppm"]

        roll = cold_rollability(
            roll_o_ppm,
            yield_MPa=yield_MPa,
            free_o_ppm=p.free_o_ppm,
            forbidden_o_ppm=p.forbidden_o_ppm,
        )

        out = {
            "j_avg_mA_cm2": j_avg_mA_cm2,
            "waveform": waveform,
            "precipitation_flux_mol_m2_s": precip,
            "o_ppm": o_ppm,
            "o_wt_percent": o_ppm * 1e-4,
            "deposit_density_kg_m3": dens,
            "delta_oxygen_strength_MPa": d_sigma,
            "yield_upper_bound_MPa": (
                (yield_MPa + d_sigma) if yield_MPa is not None else None
            ),
            "cold_rollable": roll["rollable"],
            "cold_roll_status": roll["status"],
            "cold_rollability": roll,
            "flag": SCREENING_FLAG,
        }
        if edge is not None:
            out["edge_effect"] = {
                "edge_current_ratio": edge["edge_current_ratio"],
                "edge_o_ppm": edge["edge_O_ppm"],
                "edge_oh_penalty": edge["oh_penalty"],
                "roll_gate_o_ppm": roll_o_ppm,
            }
        return out
