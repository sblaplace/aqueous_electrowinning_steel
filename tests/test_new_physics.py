"""Tests for the four new physics/chemistry modules added in August 2026.

These are fast, deterministic unit tests (no external data, no slow solvers).
They follow the repository convention of screening-level assertions with
clear tolerance bands.
"""

import pytest

# The new modules are pure enough that we can import their core functions
# even if numpy is not present in the base interpreter (the real test env
# has it via the venv created by arena_setup.sh).

try:
    from models.mhd_convection import (
        compute_mhd_solution,
        effective_mass_transfer_with_mhd,
        lorentz_velocity_scale,
    )
    from models.sonoelectrochemistry import (
        compute_sonoelectro_result,
        ultrasonic_delta_reduction,
        acoustic_streaming_velocity,
    )
    from models.bath_rheology import (
        BathRheologyParams,
        herschel_bulkley_viscosity,
        effective_viscosity_for_gas_holdup,
    )
    from models.eqcm_metrology import (
        simulate_eqcm_run,
        sauerbrey_mass,
    )
    NEW_MODULES_AVAILABLE = True
except Exception:
    NEW_MODULES_AVAILABLE = False


@pytest.mark.skipif(not NEW_MODULES_AVAILABLE, reason="new physics modules not importable")
class TestMHDConvection:
    def test_lorentz_velocity_scale_positive(self):
        u = lorentz_velocity_scale(j_A_m2=3000, b_T=0.0002, char_length_m=0.003)
        assert u > 0.0
        assert u < 0.01   # screening order of magnitude

    def test_mhd_solution_basic(self):
        sol = compute_mhd_solution(j_mean_mA_cm2=300, b_field_T=0.00025)
        assert sol.b_field_T == 0.00025
        assert sol.effective_delta_reduction_factor > 0.1
        assert sol.lorentz_velocity_m_s >= 0.0   # can be extremely small at screening B-field

    def test_effective_mass_transfer_helper(self):
        delta = effective_mass_transfer_with_mhd(50e-6, j_mA_cm2=200, b_field_T=0.0003)
        assert 10e-6 < delta < 60e-6


@pytest.mark.skipif(not NEW_MODULES_AVAILABLE, reason="new physics modules not importable")
class TestSonoelectrochemistry:
    def test_streaming_velocity(self):
        u = acoustic_streaming_velocity(power_w=100, transducer_area_m2=0.001)
        assert u > 0.005   # realistic streaming velocity

    def test_sonoelectro_result(self):
        res = compute_sonoelectro_result()
        assert res.acoustic_streaming_velocity_m_s > 0
        # δ-reduction should sit in the module's claimed 2-5× band (0.2-0.5),
        # not collapse toward ~0 (which would overstate the benefit ~70×).
        assert 0.15 < res.effective_delta_reduction_factor < 0.5
        # Micro-jet velocity must stay subsonic (physical), not ~1.3 km/s.
        assert 0 < res.microjet_velocity_m_s < 600
        assert res.effective_delta_reduction_factor > 0.01
        assert res.degassing_factor > 1.0

    def test_ultrasonic_delta_reduction(self):
        red = ultrasonic_delta_reduction(0.02, 50e-6)
        assert red > 0.1   # meaningful reduction


@pytest.mark.skipif(not NEW_MODULES_AVAILABLE, reason="new physics modules not importable")
class TestBathRheology:
    def test_herschel_bulkley(self):
        params = BathRheologyParams(yield_stress_Pa=0.05, flow_index_n=0.85)
        eta = herschel_bulkley_viscosity(10.0, params)
        assert eta > 0.0001   # realistic order of magnitude

    def test_effective_viscosity_gas_holdup(self):
        params = BathRheologyParams(phi_particles=0.03)
        eta = effective_viscosity_for_gas_holdup(params, shear_rate_s=20.0)
        assert eta > 0.0004


@pytest.mark.skipif(not NEW_MODULES_AVAILABLE, reason="new physics modules not importable")
class TestEQCMMetrology:
    def test_sauerbrey(self):
        m = sauerbrey_mass(delta_f_Hz=-1000)
        assert 15 < m < 20   # ~17.7 µg/cm² expected

    def test_simulate_eqcm_run(self):
        eq = simulate_eqcm_run(charge_density_C_cm2=1800, fe_efficiency=0.90, trapped_h_ppm=150)
        assert eq.mass_gain_ug_cm2 > 400_000   # realistic ~468k µg/cm²
        assert eq.trapped_h_ppm > 100
        assert eq.frequency_shift_Hz < 0
        assert eq.frequency_shift_Hz < -1e6   # consistent with ~4.7e5 µg/cm² mass gain


@pytest.mark.skipif(not NEW_MODULES_AVAILABLE, reason="new physics modules not importable")
def test_new_physics_integration_smoke():
    """Smoke test that the four modules can be used together in a realistic chain."""
    # MHD + US both produce a thinned δ
    delta0 = 50e-6
    mhd_sol = compute_mhd_solution(j_mean_mA_cm2=250, b_field_T=0.0002)
    us_res = compute_sonoelectro_result()
    delta_mhd = delta0 * mhd_sol.effective_delta_reduction_factor
    delta_us = delta0 * us_res.effective_delta_reduction_factor
    assert delta_mhd < delta0 and delta_us < delta0

    # Rheology gives a viscosity that could be fed to gas_holdup
    rheo = BathRheologyParams()
    eta = effective_viscosity_for_gas_holdup(rheo)
    assert eta > 0

    # EQCM can consume a charge that would come from a diffusion_layer_1d run
    eq = simulate_eqcm_run(charge_density_C_cm2=3600, fe_efficiency=0.88)
    assert eq.mass_gain_ug_cm2 > 800_000   # realistic order of magnitude

    assert True  # if we reached here the chain is live
