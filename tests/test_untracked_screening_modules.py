"""Smoke tests for the untracked screening modules added in the chemistry/physics
board-gap PR: additive aging, micro-pH buffer, and substrate passivation.

These are L0 screening scaffolds, so the asserts are loose sanity bands that pin
the *physical monotonicity and magnitude* rather than exact values:
  - additive decay: half-life in the 10–500 h range, faster at higher T/j, and
    coverage decays toward a replenishment steady state over campaign time.
  - micro-pH: surface pH rises with current (bounded, not a constant clamp) and
    buffer capacity stays positive; no spurious Fe(OH)2 precipitation at the
    reference acidic operating point.
  - substrate passivation: parabolic oxide growth gives a nm-scale film at 1000 h,
    monotonically thinning the critical peel thickness and lowering fracture energy.
"""

from models.additive_aging import decay_rate_per_hour, effective_leveler_coverage
from models.microph_buffer import compute_surface_pH
from models.substrate_passivation import (
    critical_peel_thickness,
    interfacial_fracture_energy,
    oxide_growth_parabolic,
)


def test_additive_decay_half_life_in_screening_range():
    d = decay_rate_per_hour()
    assert 10.0 < d["half_life_h"] < 500.0
    # primary channel resolves and k_eff is positive
    assert d["k_eff_h"] > 0.0


def test_additive_decay_speeds_up_with_temperature():
    d_hot = decay_rate_per_hour(T_C=80.0)
    d_cold = decay_rate_per_hour(T_C=30.0)
    # Arrhenius: hotter -> faster decay -> shorter half-life
    assert d_hot["half_life_h"] < d_cold["half_life_h"]


def test_additive_coverage_tends_to_replenishment_steady_state():
    c_0 = effective_leveler_coverage(0.002, 0.01)
    c_200 = effective_leveler_coverage(0.002, 200.0)
    c_2000 = effective_leveler_coverage(0.002, 2000.0)
    # With continuous feed the coverage rises from C0 toward a fixed steady
    # state C_ss = replenishment/k_eff, monotonically and asymptotically —
    # it does not decay to zero (replenishment balances decay).
    assert c_0 <= c_200 <= c_2000
    # C_ss is bounded and finite (positive, not runaway).
    assert 0.0 < c_2000 < 1.0


def test_surface_pH_rises_monotonically_with_current():
    p_low = compute_surface_pH(2.0, 1.0, 0.5, 200.0)[0]
    p_high = compute_surface_pH(2.0, 1.0, 0.5, 5000.0)[0]
    # current-dependent, not a constant clamp
    assert p_high > p_low + 0.2
    # stays physically bounded (not driven to pH 14)
    assert p_high < 6.0


def test_surface_pH_reference_operating_point():
    # Reference: bulk pH 2, ~2000 A/m2 (0.2 A/cm2), 60 C acidic sulfate.
    p, beta, _, precip = compute_surface_pH(2.0, 1.0, 0.5, 2000.0)
    assert 2.0 <= p <= 3.5
    assert beta > 0.01
    # Fe(OH)2 does not precipitate at the acidic reference operating point.
    assert precip is False


def test_oxide_growth_parabolic_scale_at_1000h():
    delta_ox, k_p, rate = oxide_growth_parabolic(1000.0)
    nm = delta_ox * 1e9
    # ~636 nm at 1000 h (anodic exposure): nm-scale film, positive k_p.
    assert 100.0 < nm < 1500.0
    assert k_p > 0.0
    assert abs(rate / nm - 1.0) < 1e-9  # rate_1000h == thickness at 1000 h


def test_oxide_growth_cathodic_suppresses_anodic():
    delta_anodic = oxide_growth_parabolic(1000.0, exposure_mode="anodic")[0]
    delta_cathodic = oxide_growth_parabolic(1000.0, exposure_mode="cathodic")[0]
    assert delta_cathodic < delta_anodic


def test_passivation_thins_critical_peel_and_lowers_gc():
    delta_ox = oxide_growth_parabolic(1000.0)[0]
    h_c = critical_peel_thickness(delta_ox)
    g_c = interfacial_fracture_energy(delta_ox)
    # Clean-baseline critical thickness (15 nm) is reduced, and Gc drops
    # from its 12 J/m2 clean value toward the brittle 3.5 J/m2 floor.
    assert 0.0 <= h_c < 15.0
    assert 3.5 <= g_c <= 12.0
    # More oxide -> smaller h_c (monotonic)
    h_c_thin = critical_peel_thickness(oxide_growth_parabolic(100.0)[0])
    assert h_c <= h_c_thin
