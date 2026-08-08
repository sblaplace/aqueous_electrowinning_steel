"""Electro-osmotic water drag through the cation-exchange membrane.

Divided-cell iron electrowinning transports H⁺ (and cations) through the
Nafion-class membrane under the applied field.  Each proton that crosses the
membrane drags a shell of coordinated water with it — the *electro-osmotic
drag* effect.  Because the membrane allows water to pass but the cell is
nominally closed, this is a real leak in the catholyte mass balance: water
leaves the catholyte for the anolyte, concentrating the non-volatile
catholyte solutes (Fe²⁺, Fe³⁺, H⁺).

This module provides:

* :func:`n_w` — the electro-osmotic drag coefficient (mol H₂O transported per
  mol H⁺ across the membrane), as a function of current density, temperature
  and membrane age.
* :func:`water_volume_flux_m3_m2_s` / :func:`water_volume_flux_L_m2_hr` — the
  resulting volumetric water flux out of the catholyte.

Physics
-------
The proton flux through the membrane carries the ionic current::

    J_H+ = j / F            (mol H⁺ / m² / s)

The electro-osmotic water flux is that times the drag coefficient::

    J_H2O = n_w · j / F     (mol H₂O / m² / s)

and the volumetric flux uses the molar volume of water::

    Q_w   = n_w · (j / F) · (M_w / ρ_w)   (m³ / m² / s)

At 300 mA/cm² (3000 A/m²) through hydrated Nafion, n_w ≈ 2.5 gives roughly
0.5–1 mL/(cm²·hr) of water crossing cathode→anode (≈5–10 L/m²·hr), matching
the documented magnitude (the CHEM_PHYS_IMPROVEMENTS_V2 §1.3 figure of
"0.5–1 mL/(m²·hr)" reads as a cm² typo — the physical value at n_w 2–3 is
~5000× larger).  Over a 100 hr run this removes a meaningful fraction of the
catholyte volume and shifts the bulk concentrations by several percent.

Dependencies of the drag coefficient
------------------------------------
* **j** — n_w is roughly current-independent for a well-hydrated membrane
  (each proton carries the same water shell), so n_w is flat in j by default.
  j enters through the proton flux, not the coefficient.  Kept as an argument
  so a current-dependent (e.g. drying at very high j) parametrisation can be
  plugged in.
* **T** — mild linear rise of n_w with temperature (transport of the water
  shell is faster / the membrane is more hydrated hot).  Default ≈2% per 10 °C.
* **membrane_age** — the membrane slowly dehydrates and fouls with operating
  time, so the drag coefficient decays exponentially toward a floor below the
  fresh value.  Default half-life is long (≈1.4 yr of continuous operation).

Additive / opt-in by design: the module exports pure functions and ships no
state.  The callsites in :mod:`~models.membrane_transport` and
:mod:`~models.bath_dynamics` enable the term behind a flag (default off), so
the baseline twin is byte-identical when drag is not requested.
"""

from __future__ import annotations

from .electrochemistry import FARADAY

# ─── Physical constants (transported water) ─────────────────────────────
M_WATER_KG_MOL = 0.01801528      # kg/mol — molar mass of water
RHO_WATER_KG_M3 = 998.2          # kg/m³ — density of the transported water (~25 °C)

# ─── Default drag-coefficient parameters ────────────────────────────────
# Literature n_w for hydrated Nafion is ≈ 2–3 H₂O per H⁺ (Zawodzinski et al.,
# J. Electrochem. Soc. 1993); 2.5 is the midpoint used here.
N_W_REF = 2.5                    # dimensionless drag coefficient at reference
T_REF_C = 60.0                   # reference temperature (°C)
TEMP_DEPENDENCE_PER_C = 0.002    # relative n_w increase per °C above T_ref (~2%/10°C)
AGE_HALFLIFE_HR = 12000.0        # membrane-aging half-life (h) ≈ 1.4 yr continuous run
AGE_FLOOR_FRACTION = 0.6         # n_w decays toward 60% of its fresh value with age


def n_w(
    j_A_m2: float,
    temperature_C: float = T_REF_C,
    membrane_age_hr: float = 0.0,
    n_w_ref: float = N_W_REF,
    temperature_dependence_per_c: float = TEMP_DEPENDENCE_PER_C,
    age_halflife_hr: float = AGE_HALFLIFE_HR,
    age_floor_fraction: float = AGE_FLOOR_FRACTION,
) -> float:
    """Electro-osmotic drag coefficient n_w (mol H₂O per mol H⁺).

    Parameters
    ----------
    j_A_m2 : float
        Current density (A/m²).  Held for a current-dependent parametrisation;
        the default model keeps n_w flat in j (each proton carries the same
        water shell).
    temperature_C : float
        Operating temperature (°C).  n_w rises mildly with temperature.
    membrane_age_hr : float
        Cumulative membrane operating age (h).  n_w decays exponentially
        toward ``age_floor_fraction · n_w_ref`` as the membrane dehydrates
        / fouls.
    n_w_ref : float
        Fresh-membrane drag coefficient at T_ref.  Default 2.5 (midpoint of
        the 2–3 literature range).
    temperature_dependence_per_c : float
        Relative change in n_w per °C deviation from T_ref.
    age_halflife_hr : float
        Membrane-aging half-life (h).  n_w halves this gap toward the floor
        every half-life.
    age_floor_fraction : float
        Long-term retention fraction of the fresh n_w.

    Returns
    -------
    float
        Drag coefficient (mol H₂O per mol H⁺), >= 0.
    """
    t_var = max(0.0, temperature_C - T_REF_C)
    base = n_w_ref * (1.0 + temperature_dependence_per_c * t_var)

    # The long-time floor is a retention *fraction* of the fresh value, not an
    # absolute n_w (docstring contract).  The aged coefficient decays
    # exponentially from the operating-temperature value toward that floor.
    floor = age_floor_fraction * n_w_ref
    age_dep = 0.5 ** (max(0.0, membrane_age_hr) / max(age_halflife_hr, 1e-12))
    aged = floor + (base - floor) * age_dep

    return max(float(aged), 0.0)


def water_volume_flux_m3_m2_s(
    j_A_m2: float,
    temperature_C: float = T_REF_C,
    membrane_age_hr: float = 0.0,
    n_w_ref: float = N_W_REF,
    temperature_dependence_per_c: float = TEMP_DEPENDENCE_PER_C,
    age_halflife_hr: float = AGE_HALFLIFE_HR,
    age_floor_fraction: float = AGE_FLOOR_FRACTION,
) -> float:
    """Volumetric electro-osmotic water flux (m³/(m²·s)), positive out of the
    catholyte (cathode → anode).

    Q_w = n_w · (j / F) · (M_w / ρ_w)

    Parameters are the operating conditions plus the drag-coefficient knobs of
    :func:`n_w`.
    """
    d = n_w(
        j_A_m2,
        temperature_C=temperature_C,
        membrane_age_hr=membrane_age_hr,
        n_w_ref=n_w_ref,
        temperature_dependence_per_c=temperature_dependence_per_c,
        age_halflife_hr=age_halflife_hr,
        age_floor_fraction=age_floor_fraction,
    )
    proton_flux_mol_m2_s = j_A_m2 / FARADAY  # all cationic current carried by H⁺
    mol_flux = d * proton_flux_mol_m2_s      # mol H₂O / (m²·s)
    molar_volume_m3_mol = M_WATER_KG_MOL / RHO_WATER_KG_M3
    return float(max(mol_flux * molar_volume_m3_mol, 0.0))


def water_volume_flux_L_m2_hr(
    j_A_m2: float,
    temperature_C: float = T_REF_C,
    membrane_age_hr: float = 0.0,
    n_w_ref: float = N_W_REF,
    temperature_dependence_per_c: float = TEMP_DEPENDENCE_PER_C,
    age_halflife_hr: float = AGE_HALFLIFE_HR,
    age_floor_fraction: float = AGE_FLOOR_FRACTION,
) -> float:
    """Volumetric electro-osmotic water flux (L/(m²·hr)), the engineering unit
    used by the bath CSTR.  Positive = leaves the catholyte.
    """
    return water_volume_flux_m3_m2_s(
        j_A_m2,
        temperature_C=temperature_C,
        membrane_age_hr=membrane_age_hr,
        n_w_ref=n_w_ref,
        temperature_dependence_per_c=temperature_dependence_per_c,
        age_halflife_hr=age_halflife_hr,
        age_floor_fraction=age_floor_fraction,
    ) * 3600.0 * 1000.0