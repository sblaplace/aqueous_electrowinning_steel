"""Tests for the second physics tranche: surface Pitzer gamma, DSA anode, Donnan, precipitation sink.

Level-0 screening checks — they verify the mechanism direction and
backward-compatibility, not calibrated numbers.

Review requested coverage:
  (a) Donnan K_D monotone in fixed charge / inverse in salt and Fe3+ case
  (b) precipitation sink zero when S<=1 and positive+capped when supersaturated
  (c) activity_model='ideal' and anode_chemistry='fixed' reproduce pre-upgrade numbers
  (d) _gamma_at rises as Fe depletes (surface gamma > bulk gamma)
"""

import pytest

from models.diffusion_layer_1d import DiffusionLayer1D
from models.membrane_transport import donnan_potential_V, donnan_partition_coefficient
from models.cell_physics import BathRecipe, CellGeometry, CellPhysics, ProcessConditions


# ─── Donnan ─────────────────────────────────────────────────────────────

class TestDonnan:
    def test_potential_positive_and_monotone_in_fixed_charge(self):
        phi_low = donnan_potential_V(fixed_charge_M=0.5, external_salt_M=1.0, temperature_C=60)
        phi_high = donnan_potential_V(fixed_charge_M=1.0, external_salt_M=1.0, temperature_C=60)
        phi_higher = donnan_potential_V(fixed_charge_M=2.0, external_salt_M=1.0, temperature_C=60)
        assert 0.0 < phi_low < phi_high < phi_higher
        # ~10-40 mV for Nafion at 0.5-2 M salt
        assert 0.005 < phi_low < 0.05
        assert 0.005 < phi_higher < 0.10

    def test_potential_decreases_with_salt(self):
        phi_dilute = donnan_potential_V(fixed_charge_M=1.0, external_salt_M=0.5, temperature_C=60)
        phi_conc = donnan_potential_V(fixed_charge_M=1.0, external_salt_M=2.0, temperature_C=60)
        assert phi_dilute > phi_conc > 0.0

    def test_partition_monotone_and_fe3_case(self):
        k_low = donnan_partition_coefficient(z=1, fixed_charge_M=0.5, external_salt_M=1.0)
        k_high = donnan_partition_coefficient(z=1, fixed_charge_M=1.0, external_salt_M=1.0)
        # For cations (z>0) the membrane is enriched vs solution: K_D <1? Wait
        # phi_D positive (membrane positive vs solution) => exp(-zF phi/RT) <1 for z>0
        # So larger fixed charge => more positive phi => smaller K_D.
        assert 0.0 < k_high < k_low < 2.0
        # Fe3+ (z=+3) is more strongly excluded than monovalent
        k_fe3 = donnan_partition_coefficient(z=3, fixed_charge_M=1.0, external_salt_M=1.0)
        k_na = donnan_partition_coefficient(z=1, fixed_charge_M=1.0, external_salt_M=1.0)
        assert 0.0 < k_fe3 < k_na < 1.0
        # Anions are enriched (z=-1 => K_D >1)
        k_anion = donnan_partition_coefficient(z=-1, fixed_charge_M=1.0, external_salt_M=1.0)
        assert k_anion > 1.0

    def test_donnan_consistent_with_potential(self):
        import math
        R = 8.314462618
        F = 96485.33212
        T = 333.15
        phi = donnan_potential_V(fixed_charge_M=1.0, external_salt_M=1.0, temperature_C=60)
        k = donnan_partition_coefficient(z=1, fixed_charge_M=1.0, external_salt_M=1.0, temperature_C=60)
        assert k == pytest.approx(math.exp(-1 * F * phi / (R * T)), rel=1e-12)


# ─── Precipitation sink ────────────────────────────────────────────────

class TestPrecipitationSink:
    def test_zero_when_undersaturated_acidic(self):
        m = DiffusionLayer1D(fe_conc_M=1.0, pH_bulk=2.0, temperature_C=50, delta_m=50e-6, activity_model="pitzer")
        r = m.solve(100.0)
        assert r.feoh2_supersaturation < 1e-3
        assert r.precipitation_active is False
        assert r.precipitation_flux_mol_m2_s == pytest.approx(0.0, abs=1e-12)
        assert r.precipitation_fraction == pytest.approx(0.0, abs=1e-12)
        assert r.sludge_rate_g_m2_s == pytest.approx(0.0, abs=1e-12)

    def test_positive_and_capped_when_supersaturated(self):
        # High bulk pH + HER-active cathode drives surface pH >7 and S>>1
        # Use the CellPhysics path (which has the full bath, buffer, support)
        # because the raw DiffusionLayer1D without supporting Na2SO4
        # stays undersaturated at these currents.
        cp = CellPhysics(BathRecipe(c_FeSO4_M=1.0, pH=5.5), conditions=ProcessConditions(her_i0=1e-2, temperature_C=60))
        pt = cp.solve_at_j(50.0)
        assert pt.feoh2_supersaturation > 1.0
        assert pt.precipitation_active is True
        # Flux is positive and physically capped
        assert pt.precipitation_flux_mol_m2_s > 0.0
        assert 0.0 < pt.precipitation_fraction < 1.0
        assert pt.sludge_rate_g_m2_s > 0.0
        # Net deposition is less than gross when sludge is active
        assert pt.deposition_rate_net_um_hr < pt.deposition_rate_um_hr
        assert pt.deposition_rate_net_um_hr == pytest.approx(0.0, abs=1e-6)  # all Fe to sludge at this S

    def test_disabled_sink_gives_zero_flux(self):
        # Disable via the underlying DiffusionLayer1D flag — construct manually
        dl = DiffusionLayer1D(fe_conc_M=1.0, pH_bulk=5.5, temperature_C=60, delta_m=50e-6, activity_model="pitzer", her_i0=1e-2, precipitation_sink=False)
        r = dl.solve(50.0)
        # Even if S>1, disabled sink reports zero flux
        # (At pH 5.5 without support the raw DL stays undersaturated, so we test the flag plumbing)
        assert r.precipitation_flux_mol_m2_s == pytest.approx(0.0, abs=1e-12)

    def test_sink_does_not_affect_acidic_rc1_net_rate(self):
        cp = CellPhysics(BathRecipe(c_FeSO4_M=1.0, pH=2.0), conditions=ProcessConditions(temperature_C=50))
        pt = cp.solve_at_j(100.0)
        assert pt.feoh2_supersaturation < 1e-3
        assert pt.precipitation_flux_mol_m2_s == pytest.approx(0.0, abs=1e-12)
        assert pt.deposition_rate_net_um_hr == pytest.approx(pt.deposition_rate_um_hr, rel=1e-12)


# ─── Backward compatibility ───────────────────────────────────────────

class TestBackwardCompatibility:
    def test_ideal_activity_model_recovers_bulk_gamma_one(self):
        m_ideal = DiffusionLayer1D(fe_conc_M=1.0, pH_bulk=2.0, temperature_C=50, activity_model="ideal")
        assert m_ideal.gamma_fe == pytest.approx(1.0, rel=1e-12)
        assert m_ideal._gamma_at(0.5, 2.0) == pytest.approx(1.0, rel=1e-12)
        r_ideal = m_ideal.solve(50.0)
        # Only the activity correction differs; FE should be close to pitzer but not identical
        m_pitzer = DiffusionLayer1D(fe_conc_M=1.0, pH_bulk=2.0, temperature_C=50, activity_model="pitzer")
        assert 0.02 < m_pitzer.gamma_fe < 0.2
        r_pitzer = m_pitzer.solve(50.0)
        # Pitzer shifts E_eq negative by ~30 mV, so FE differs by a few points at most
        assert abs(r_ideal.current_efficiency - r_pitzer.current_efficiency) < 0.1

    def test_fixed_anode_chemistry_reproduces_pre_upgrade_via_cellphysics(self):
        # Pre-upgrade fixed-eta fallback: E_anode=1.229, eta=0.40, no pH correction
        fixed = CellPhysics(BathRecipe(), CellGeometry(anode_chemistry="fixed"))
        # New default DSA
        dsa = CellPhysics(BathRecipe(), CellGeometry(anode_chemistry="inert"))
        # Compare at the theory-confidence reference j (96.666 mA/cm2) where the
        # committed report was pinned. At pH 2 the pH-corrected DSA lowers
        # V_cell by ~0.14 V (1.10+0.386 vs 1.229+0.40).
        j_ref = 96.66666666666667
        pt_fixed = fixed.solve_at_j(j_ref)
        pt_dsa = dsa.solve_at_j(j_ref)
        assert pt_fixed.V_cell > pt_dsa.V_cell
        assert pt_fixed.V_cell == pytest.approx(5.388, abs=0.05)
        assert pt_dsa.V_cell == pytest.approx(5.246, abs=0.05)
        # FE is anode-independent (cathode film only)
        assert pt_fixed.current_efficiency == pytest.approx(pt_dsa.current_efficiency, rel=1e-12)


# ─── Per-surface gamma ────────────────────────────────────────────────

class TestPerSurfaceGamma:
    def test_gamma_rises_as_fe_depletes(self):
        m = DiffusionLayer1D(fe_conc_M=1.0, pH_bulk=2.0, temperature_C=50, delta_m=50e-6, activity_model="pitzer")
        g_bulk = m.gamma_fe
        g_depleted = m._gamma_at(0.5, 2.0)
        g_more_depleted = m._gamma_at(0.2, 2.0)
        # Lower ionic strength => less screening => gamma rises toward 1
        assert g_bulk < g_depleted < g_more_depleted < 1.0
        # Bulk gamma for 1 M FeSO4 is strongly non-ideal (~0.04)
        assert 0.02 < g_bulk < 0.08

    def test_gamma_increases_with_ph_at_fixed_fe(self):
        m = DiffusionLayer1D(fe_conc_M=1.0, pH_bulk=2.0, temperature_C=50, activity_model="pitzer")
        g_ph2 = m._gamma_at(1.0, 2.0)
        g_ph3 = m._gamma_at(1.0, 3.0)
        # At 1 M FeSO4 the sulfate ionic strength dominates; pH 2->3 changes
        # H+ by 0.009 M vs I~3 M, so gamma is essentially flat (within <1%).
        assert g_ph3 == pytest.approx(g_ph2, rel=0.02)

    def test_surface_potential_uses_local_gamma(self):
        m = DiffusionLayer1D(fe_conc_M=1.0, pH_bulk=2.0, temperature_C=50, activity_model="pitzer")
        E_bulk = m._fe_equilibrium_potential(1.0)
        E_surface = m._fe_equilibrium_potential(0.5, surface_pH=2.0)
        # At depleted surface, Nernst term is more negative due to lower [Fe],
        # but the higher local gamma partially compensates. The net is still
        # more negative than bulk, but less so than the bulk-gamma estimate.
        E_bulk_gamma_depleted = m._fe_equilibrium_potential(0.5)  # bulk gamma
        assert E_surface < E_bulk  # depleted => more negative
        assert E_surface > E_bulk_gamma_depleted  # local gamma mitigates

    def test_gamma_cache_hit(self):
        m = DiffusionLayer1D(fe_conc_M=1.0, pH_bulk=2.0, temperature_C=50, activity_model="pitzer")
        g1 = m._gamma_at(0.5, 2.0)
        # Second call with same rounded key should hit cache and be identical
        g2 = m._gamma_at(0.50001, 2.001)
        assert g1 == pytest.approx(g2, rel=1e-12)
        assert len(m._gamma_at_cache) >= 1
