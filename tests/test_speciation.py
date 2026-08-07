"""Tests for multicomponent electrolyte speciation and activity model.

The default path is the Pitzer model (models/pitzer.py) — valid at the
multi-molal ionic strengths of ferrous-sulfate electrowinning baths.  The
legacy Davies path (valid only to I ≈ 0.5 mol/kg) is retained for A/B
comparison; its characteristic failure mode (≈97 % of iron forced into
FeSO4⁰ pairs) is pinned explicitly as regression archaeology.
"""

import math

import pytest

from models.speciation import (
    SolutionComposition,
    davies_A,
    davies_gamma,
    solve_speciation,
    speciation_temperature_sweep,
)

REF = SolutionComposition(c_FeSO4=1.0, c_Na2SO4=0.5, c_H2SO4=0.01, c_H3BO3=0.4, T_C=50.0)


# ─── Shared helper functions ─────────────────────────────────────────


def test_davies_A_temperature_scaling():
    """A parameter should increase slightly with temperature as dielectric constant drops."""
    A_25 = davies_A(25.0)
    A_60 = davies_A(60.0)
    assert 0.49 < A_25 < 0.52
    assert A_60 > A_25


def test_davies_gamma():
    """Activity coefficient gamma for z=2 should be lower than for z=1 at same I > 0."""
    gamma1 = davies_gamma(z=1, I=0.5, A=0.509)
    gamma2 = davies_gamma(z=2, I=0.5, A=0.509)
    assert 0.0 < gamma2 < gamma1 < 1.0


# ─── Pitzer path (default) ───────────────────────────────────────────


def test_speciation_baseline_pitzer():
    """Reference bath on the Pitzer model: honest small gammas, fully
    dissociated iron, modest secondary contact-pair estimate."""
    res = solve_speciation(REF)

    assert res["activity_model"] == "pitzer"
    assert res["ionic_strength_molal"] > 3.0             # concentrated bath
    assert 0.02 < res["gamma_Fe2"] < 0.20                # 2–2-salt behaviour
    assert res["c_Fe2_free_M"] == pytest.approx(1.0)     # dissociated convention
    assert 0.0 < res["c_FeSO4_pair_M"] < 0.4             # secondary estimate
    assert 0.02 < res["a_Fe2"] < 0.10
    assert -0.52 < res["E_rev_Fe_V_SHE"] < -0.46
    assert 5.0 < res["pH_precip_Fe_OH2"] < 6.5
    assert 5.0 < res["conductivity_S_m"] < 25.0
    assert 0.90 < res["water_activity"] < 1.00
    assert res["solution_density_kg_L"] > 1.1


def test_speciation_charge_balance_and_ka2_residual():
    """Sulfate/bisulfate equilibrium and electroneutrality close exactly."""
    res = solve_speciation(REF)
    kw = res["ionic_strength_M"] / res["ionic_strength_molal"]
    mS = res["c_SO4_free_M"] / kw
    mHSO4 = res["c_HSO4_free_M"] / kw
    mH = res["c_H_free_M"] / kw

    T_K = REF.T_C + 273.15
    from models.thermodynamic_constants import (
        KA_HSO4_25, DH_HSO4_J_MOL, vanthoff_constant
    )
    Ka2_T = vanthoff_constant(KA_HSO4_25, T_K, DH_HSO4_J_MOL)
    resid = (res["gamma_H"] * mH * res["gamma_SO4"] * mS
             - Ka2_T * res["gamma_HSO4"] * mHSO4)
    assert abs(resid) < 1e-10

    q = (2 * REF.c_FeSO4 + 2 * REF.c_Na2SO4 + res["c_H_free_M"]
         - res["c_HSO4_free_M"] - 2 * res["c_SO4_free_M"])
    assert abs(q) < 1e-9


def test_conductivity_pure_feso4_matches_measurement():
    """1 M FeSO4 at 25 °C: measured κ ≈ 5–6 S/m; estimate must be in ±40 %
    (screening tolerance) — the Davies path was ~2× low for the wrong reason."""
    pure = SolutionComposition(c_FeSO4=1.0, c_Na2SO4=0.0, c_H2SO4=0.0, c_H3BO3=0.0, T_C=25.0)
    res = solve_speciation(pure)
    assert 3.5 < res["conductivity_S_m"] < 8.0


def test_conductivity_pure_na2so4_matches_measurement():
    """1 M Na2SO4 at 25 °C: measured κ ≈ 7–8 S/m."""
    nas = SolutionComposition(c_FeSO4=0.0, c_Na2SO4=1.0, c_H2SO4=0.0, c_H3BO3=0.0, T_C=25.0)
    res = solve_speciation(nas)
    assert 5.0 < res["conductivity_S_m"] < 10.0


def test_no_acid_pH_from_boric_buffer():
    """Without sulfuric acid the pH falls back to the boric buffer (~5),
    not an unphysical pH-14 blow-up."""
    res = solve_speciation(
        SolutionComposition(c_FeSO4=1.0, c_Na2SO4=0.0, c_H2SO4=0.0, c_H3BO3=0.4, T_C=25.0)
    )
    assert 4.0 < res["pH_activity"] < 6.0


def test_speciation_temperature_sweep():
    """Speciation sweep over temperature range 20-80 °C."""
    comp = SolutionComposition(c_FeSO4=1.0, c_Na2SO4=0.5)
    sweep = speciation_temperature_sweep(comp, T_min=20.0, T_max=80.0, num=5)

    assert len(sweep["temperature_C"]) == 5
    # Electrical conductivity should increase with temperature
    assert sweep["conductivity_S_m"][-1] > sweep["conductivity_S_m"][0]


# ─── Legacy Davies path (superseded; pinned for archaeology) ─────────


def test_davies_path_documents_superseded_overpairing():
    """The Davies model at bath ionic strength forces ~97 % of the iron
    into FeSO4⁰ pairs — the failure mode the Pitzer path fixed.  Pinned
    here so the A/B comparison stays reproducible; do NOT use for process
    numbers."""
    res = solve_speciation(REF, model="davies")
    assert res["activity_model"] == "davies"
    assert res["fe2_pair_percentage"] > 90.0        # the artefact
    assert res["c_Fe2_free_M"] < 0.1                # phantom free-Fe deficit
    assert res["gamma_Fe2"] > 0.5                   # out-of-calibration γ₂


def test_pitzer_vs_davies_nernst_shift():
    """The activity-model change moves the Nernst potential ~10–20 mV less
    negative at the reference bath — a first-order correction to every
    voltage balance downstream."""
    pitzer = solve_speciation(REF, model="pitzer")
    davies = solve_speciation(REF, model="davies")
    dE = pitzer["E_rev_Fe_V_SHE"] - davies["E_rev_Fe_V_SHE"]
    assert 0.005 < dE < 0.030
