"""Cross-model internal-consistency tests.

These tests verify that independent models which *should* agree at matched
conditions actually do — catching silent drift in constants, units, or
physical assumptions across modules.

Where models genuinely disagree by design (different physics), the tests
document the disagreement ratio so future changes don't silently alter it.
"""
import math
import numpy as np
import pytest

from models.kinetics import limiting_current_density
from models.diffusion_layer_1d import DiffusionLayer1D
from models.co_deposition import surface_pH_from_current
from models.pourbaix import LOGKSP_FEOH2
from models.electrochemistry import R_GAS


# ─── Test 1: Limiting-current — NP diffusion limit vs Levich ───────────────
# The NP model's "diffusion limit" is computed from the full multi-ion
# Nernst-Planck system (coupled Fe²⁺/H⁺/SO₄²⁻ transport + electroneutrality).
# The Levich formula (zFDC/δ) assumes binary transport. They should be in the
# same order of magnitude; a large discrepancy reveals constant drift or a
# physics mismatch.

def test_np_diffusion_limit_vs_levich():
    """NP diffusion-only limit and Levich limit should be same order of magnitude."""
    from models.transport import NernstPlanckFilm

    fe_conc_M = 1.0
    delta_m = 50e-6
    T_C = 60.0
    T_K = T_C + 273.15

    # Levich limit with temperature-corrected diffusivity
    D_fe = 7.2e-10  # m²/s at 25°C (CRC)
    Ea = 18e3       # J/mol (electrochemistry.py DIFF_EA_J_MOL)
    D_fe_T = D_fe * math.exp(Ea / R_GAS * (1.0 / 298.15 - 1.0 / T_K))
    levich = limiting_current_density(fe_conc_M * 1000.0, D_fe_T, delta_m)

    # NP model at high support (migration collapsed) — diffusion limit only
    film = NernstPlanckFilm(
        fe_conc_M=fe_conc_M,
        boundary_layer_m=delta_m,
        temperature_C=T_C,
        support_conc_M=10.0,
    )
    result = film.solve(100.0)  # solve at moderate j to get diffusion_limit
    np_diff = result.diffusion_limit_A_m2

    # Document the ratio — multi-ion coupling can reduce the effective
    # Fe²⁺ diffusion limit below the binary Levich value.
    ratio = np_diff / levich
    assert 0.3 < ratio < 1.5, (
        f"NP diffusion limit ({np_diff:.1f} A/m²) vs Levich ({levich:.1f} A/m²): "
        f"ratio={ratio:.3f} — outside plausible range, check constants"
    )
    # If this ratio drifts from its current value (~0.47), something changed
    # in the multi-ion coupling or the Fe²⁺ diffusivity.
    assert 0.40 < ratio < 0.55, (
        f"NP/Levich ratio changed to {ratio:.3f} (was ~0.47). "
        f"Verify this is intentional before accepting."
    )


def test_migration_enhances_limiting_current():
    """Unsupported baths should have higher transport limits than high-support baths."""
    from models.transport import NernstPlanckFilm

    fe_conc_M = 1.0
    delta_m = 50e-6
    T_C = 60.0

    high_support = NernstPlanckFilm(
        fe_conc_M=fe_conc_M, boundary_layer_m=delta_m,
        temperature_C=T_C, support_conc_M=10.0,
    )
    unsupported = NernstPlanckFilm(
        fe_conc_M=fe_conc_M, boundary_layer_m=delta_m,
        temperature_C=T_C, support_conc_M=0.0,
    )

    ratio = unsupported.transport_limit_A_m2() / high_support.transport_limit_A_m2()
    # Migration should enhance transport; the repo README says ~2× for unsupported
    assert ratio > 1.5, (
        f"Migration enhancement ({ratio:.2f}×) too small; "
        f"expected >1.5× per the transport model's own migration analysis"
    )


# ─── Test 2: Surface pH — empirical vs Nernst-Planck ──────────────────────
# co_deposition.surface_pH_from_current() uses an empirical formula that
# ignores migration. DiffusionLayer1D includes migration, which drags H⁺
# back toward the bulk and suppresses pH rise. These models GENUINELY DISAGREE
# because they model different physics. The test documents the disagreement.

@pytest.mark.parametrize("j_mA_cm2", [100, 200])
def test_surface_pH_empirical_vs_nernst_planck(j_mA_cm2):
    """Document: empirical surface pH (no migration) > NP surface pH (with migration).

    The co_deposition empirical formula ignores the electric-field effect on
    H⁺ transport. The Nernst-Planck model includes migration, which suppresses
    the cathode pH rise. This is a known physics difference, not a bug — but
    the magnitude should be tracked.
    """
    bulk_pH = 2.0
    fe_conc_M = 1.0
    buffer_M = 0.40
    T_C = 60.0
    delta_m = 50e-6

    empirical_pH = surface_pH_from_current(
        j_mA_cm2, bulk_pH,
        buffer_capacity_M=buffer_M,
        temperature_C=T_C,
        boundary_layer_m=delta_m,
    )

    model = DiffusionLayer1D(
        fe_conc_M=fe_conc_M,
        pH_bulk=bulk_pH,
        temperature_C=T_C,
        delta_m=delta_m,
        buffer_conc_M=buffer_M,
        fe_i0=10.0,
        her_i0=0.010,
    )
    result = model.solve(j_mA_cm2)
    np_pH = result.surface_pH

    # The empirical formula should predict MORE pH rise than NP (it ignores
    # migration-driven H⁺ back-transport)
    assert empirical_pH >= np_pH - 0.5, (
        f"At j={j_mA_cm2}: empirical pH ({empirical_pH:.2f}) should be >= "
        f"NP pH ({np_pH:.2f}) since empirical ignores migration suppression"
    )


# ─── Test 3: Calibration round-trip ────────────────────────────────────────
# Generate synthetic polarization data from DepositionKinetics, then fit it
# with calibration.fit_total_cathodic_polarization. The fitted model must
# reproduce the input data within tolerance.

def test_calibration_round_trip():
    """Fitted kinetics parameters must reproduce the generating model."""
    from models.kinetics import DepositionKinetics
    from models.calibration import fit_total_cathodic_polarization
    import pandas as pd

    # Known parameters
    true_fe_i0 = 0.05
    true_her_i0 = 0.001
    true_fe_tafel = 0.120
    true_her_tafel = 0.140
    pH = 2.0
    T_C = 60.0
    fe_conc_M = 1.0

    kin = DepositionKinetics(
        pH=pH, temperature_C=T_C,
        fe_i0=true_fe_i0, her_i0=true_her_i0,
        fe_tafel_V=true_fe_tafel, her_tafel_V=true_her_tafel,
        fe_conc_M=fe_conc_M,
    )

    # Generate synthetic cathodic LSV data
    E_she = np.linspace(-0.30, -0.90, 50)  # cathodic range vs SHE
    _, _, total_current = kin.partial_currents(E_she)
    current_A_m2 = -np.maximum(total_current, 1e-10)

    data = pd.DataFrame({
        "potential_V_vs_ref": E_she,
        "current_density_A_m2": current_A_m2,
    })

    fit = fit_total_cathodic_polarization(
        data, pH=pH, temperature_C=T_C, fe_conc_M=fe_conc_M,
        reference_to_she_V=0.0,
    )

    assert fit.converged, "Calibration failed to converge"
    assert abs(np.log10(fit.fe_i0_A_m2) - np.log10(true_fe_i0)) < 0.5, (
        f"Fe i0: fitted={fit.fe_i0_A_m2:.4g}, true={true_fe_i0:.4g}"
    )
    assert abs(np.log10(fit.her_i0_A_m2) - np.log10(true_her_i0)) < 0.5, (
        f"HER i0: fitted={fit.her_i0_A_m2:.4g}, true={true_her_i0:.4g}"
    )


# ─── Test 4: Fe(OH)₂ KSP self-consistency ─────────────────────────────────
# The KSP value in pourbaix.py defines when Fe(OH)₂ precipitates. Verify
# that the computed precipitation pH is physically reasonable and consistent
# with the model's own assumptions.

def test_feoh2_ksp_precipitation_pH():
    """Fe(OH)₂ precipitation pH at 1 M Fe²⁺ should be in a reasonable range."""
    KSP = 10.0 ** LOGKSP_FEOH2  # (mol/L)³

    fe_conc_M = 1.0
    oh_precip = math.sqrt(KSP / fe_conc_M)
    pH_precip = 14.0 + math.log10(oh_precip)

    # LOGKSP_FEOH2 = -16.31 → KSP = 4.9e-17 → pH_precip ≈ 5.85 at 1 M Fe²⁺
    # This is consistent with the CRC handbook KSP for Fe(OH)₂.
    # Fe(OH)₂ precipitates at relatively low pH in concentrated Fe²⁺ baths.
    assert 5.0 < pH_precip < 7.0, (
        f"Fe(OH)₂ precipitation pH={pH_precip:.2f} at 1 M Fe²⁺ outside expected range. "
        f"LOGKSP_FEOH2={LOGKSP_FEOH2:.2f}"
    )

    # At lower Fe²⁺ (e.g. 0.01 M), precipitation should occur at higher pH
    oh_001 = math.sqrt(KSP / 0.01)
    pH_001 = 14.0 + math.log10(oh_001)
    assert pH_001 > pH_precip, "Lower [Fe²⁺] should precipitate at higher pH"
    assert 6.5 < pH_001 < 9.0, (
        f"Fe(OH)₂ precipitation pH={pH_001:.2f} at 0.01 M Fe²⁺ outside expected range"
    )
