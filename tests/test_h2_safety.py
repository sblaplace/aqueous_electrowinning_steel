"""Tests for models.h2_safety — H₂ enclosure safety envelope.

Verifies the mass-balance physics (generation rate, accumulation,
steady-state, ventilation sizing) and the edge cases (sealed enclosure,
infinite ventilation, zero HER).
"""

import math
import pytest
import numpy as np

from models.h2_safety import (
    LEL_H2_VOLUME_PERCENT,
    ALARM_SETPOINT_PERCENT,
    VENTILATION_SAFETY_FACTOR,
    EnclosureSpec,
    H2GenerationRate,
    H2SafetyResult,
    assess_h2_safety,
    min_ventilation_for_fe_rate,
    bench_cell_worst_case,
    ventilation_sizing,
)
from models.electrochemistry import FARADAY


# ─── H2GenerationRate ──────────────────────────────────────────────

class TestH2GenerationRate:
    """Test H₂ generation from cell parameters."""

    def test_basic_generation(self):
        """At FE=80 %, 20 % of current goes to HER."""
        gen = H2GenerationRate(
            j_total_A_m2=3000.0,  # 300 mA/cm²
            fe_fraction=0.80,
            cathode_area_m2=0.01,
            n_cells=1,
        )
        assert gen.her_fraction == pytest.approx(0.20)
        assert gen.her_current_A == pytest.approx(3000 * 0.01 * 0.20)
        # H₂ mol/s = I_HER / (2F)
        expected_mol_s = 3000 * 0.01 * 0.20 / (2 * FARADAY)
        assert gen.h2_mol_per_s == pytest.approx(expected_mol_s)

    def test_perfect_fe_no_h2(self):
        """At FE=100 %, no H₂ is generated."""
        gen = H2GenerationRate(
            j_total_A_m2=1000.0,
            fe_fraction=1.0,
            cathode_area_m2=0.01,
        )
        assert gen.her_fraction == pytest.approx(0.0)
        assert gen.h2_mol_per_s == pytest.approx(0.0)
        assert gen.h2_L_per_hour == pytest.approx(0.0)

    def test_multi_cell(self):
        """Multiple cells scale linearly."""
        gen1 = H2GenerationRate(1000.0, 0.8, 0.01, n_cells=1)
        gen4 = H2GenerationRate(1000.0, 0.8, 0.01, n_cells=4)
        assert gen4.h2_mol_per_s == pytest.approx(4 * gen1.h2_mol_per_s)

    def test_l_per_hour_consistency(self):
        """L/hr = mol/s × Vm × 3600."""
        gen = H2GenerationRate(3000.0, 0.5, 0.005)
        from models.h2_safety import VMOL_25C_1ATM_L
        expected = gen.h2_mol_per_s * VMOL_25C_1ATM_L * 3600
        assert gen.h2_L_per_hour == pytest.approx(expected)

    def test_invalid_fe(self):
        with pytest.raises(ValueError):
            H2GenerationRate(1000.0, 0.0, 0.01)  # fe=0 invalid
        with pytest.raises(ValueError):
            H2GenerationRate(1000.0, 1.5, 0.01)  # fe>1 invalid

    def test_invalid_area(self):
        with pytest.raises(ValueError):
            H2GenerationRate(1000.0, 0.8, 0.0)

    def test_invalid_current(self):
        with pytest.raises(ValueError):
            H2GenerationRate(-100.0, 0.8, 0.01)


# ─── EnclosureSpec ─────────────────────────────────────────────────

class TestEnclosureSpec:
    def test_basic(self):
        enc = EnclosureSpec(volume_m3=1.0, ventilation_m3_s=0.01)
        assert enc.volume_m3 == 1.0
        assert enc.vmol_m3 > 0

    def test_temperature_correction(self):
        enc_25 = EnclosureSpec(1.0, temperature_C=25.0)
        enc_60 = EnclosureSpec(1.0, temperature_C=60.0)
        # Higher T → larger molar volume
        assert enc_60.vmol_m3 > enc_25.vmol_m3

    def test_invalid_volume(self):
        with pytest.raises(ValueError):
            EnclosureSpec(volume_m3=0)
        with pytest.raises(ValueError):
            EnclosureSpec(volume_m3=-1.0)

    def test_invalid_ventilation(self):
        with pytest.raises(ValueError):
            EnclosureSpec(volume_m3=1.0, ventilation_m3_s=-0.01)


# ─── assess_h2_safety ──────────────────────────────────────────────

class TestAssessH2Safety:
    """Test the full safety assessment."""

    def test_sealed_enclosure_accumulates(self):
        """In a sealed enclosure, H₂ accumulates linearly."""
        gen = H2GenerationRate(3000.0, 0.5, 0.01)  # 50% FE
        enc = EnclosureSpec(volume_m3=0.001, ventilation_m3_s=0.0)  # 1 L sealed
        result = assess_h2_safety(gen, enc)
        # Should reach alarm quickly
        assert result.h2_percent_at_1hr > ALARM_SETPOINT_PERCENT
        assert result.time_to_25pct_lel_s < 3600  # less than 1 hour
        assert result.time_to_25pct_lel_s > 0

    def test_ventilation_holds_below_alarm(self):
        """Sufficient ventilation keeps steady-state below alarm."""
        gen = H2GenerationRate(1000.0, 0.8, 0.001)  # small cell, high FE
        # First find min ventilation, then test with 2× that
        result = assess_h2_safety(gen, EnclosureSpec(0.1, 0.0))
        q_min = result.min_ventilation_m3_s
        # Use 2× minimum ventilation
        enc = EnclosureSpec(0.1, ventilation_m3_s=q_min)
        result2 = assess_h2_safety(gen, enc)
        # Steady state should be well below alarm
        assert result2.steady_state_percent < ALARM_SETPOINT_PERCENT
        assert not result2.alarm_triggers

    def test_insufficient_ventilation_triggers_alarm(self):
        """Too little ventilation → alarm."""
        gen = H2GenerationRate(3000.0, 0.5, 0.01)  # large H₂ rate
        enc = EnclosureSpec(0.5, ventilation_m3_s=1e-5)  # tiny ventilation
        result = assess_h2_safety(gen, enc)
        assert result.alarm_triggers
        assert result.steady_state_percent > ALARM_SETPOINT_PERCENT

    def test_time_to_lel_longer_than_25_lel(self):
        """Time to LEL should be longer than time to 25 % LEL."""
        gen = H2GenerationRate(3000.0, 0.5, 0.01)
        enc = EnclosureSpec(0.01, ventilation_m3_s=0.0)
        result = assess_h2_safety(gen, enc)
        if math.isfinite(result.time_to_25pct_lel_s) and math.isfinite(result.time_to_lel_s):
            assert result.time_to_lel_s > result.time_to_25pct_lel_s

    def test_zero_her_infinite_time(self):
        """At FE=100 % there is no H₂, so time to alarm is infinite."""
        gen = H2GenerationRate(1000.0, 1.0, 0.01)
        enc = EnclosureSpec(0.1, ventilation_m3_s=0.0)
        result = assess_h2_safety(gen, enc)
        assert result.time_to_25pct_lel_s == float('inf')
        assert result.h2_percent_at_1hr == pytest.approx(0.0)

    def test_min_ventilation_positive(self):
        """Min ventilation is positive when H₂ is generated."""
        gen = H2GenerationRate(3000.0, 0.5, 0.01)
        enc = EnclosureSpec(0.5, 0.0)
        result = assess_h2_safety(gen, enc)
        assert result.min_ventilation_m3_s > 0
        assert result.min_ventilation_ach > 0

    def test_summary_is_string(self):
        gen = H2GenerationRate(1000.0, 0.8, 0.01)
        enc = EnclosureSpec(0.5, 0.01)
        result = assess_h2_safety(gen, enc)
        s = result.summary()
        assert isinstance(s, str)
        assert "H₂" in s
        assert "Safety" in s


# ─── Convenience functions ─────────────────────────────────────────

class TestConvenienceFunctions:
    def test_min_ventilation_for_fe_rate(self):
        """1 kg/hr Fe at 80 % FE requires finite ventilation."""
        q = min_ventilation_for_fe_rate(1.0, fe_percent=80.0, enclosure_m3=1.0)
        assert q > 0

    def test_min_ventilation_scales_with_rate(self):
        """More Fe → more H₂ → more ventilation needed."""
        q1 = min_ventilation_for_fe_rate(1.0, 80.0)
        q2 = min_ventilation_for_fe_rate(2.0, 80.0)
        assert q2 > q1

    def test_bench_cell_worst_case(self):
        """Bench cell at 300 mA/cm², 80 % FE in 1 L sealed."""
        result = bench_cell_worst_case(300.0, 80.0, 25.0)
        assert result.h2_percent_at_1hr > 0
        assert result.time_to_25pct_lel_s > 0

    def test_ventilation_sizing(self):
        q, ach = ventilation_sizing(300.0, 80.0, 25.0, 1, 0.5)
        assert q > 0
        assert ach > 0


# ─── Physics sanity checks ────────────────────────────────────────

class TestPhysicsSanity:
    """Cross-check the mass balance physics."""

    def test_faradaic_balance(self):
        """H₂ generation rate = (1-FE) × I_total / (2F)."""
        j = 3000.0  # A/m²
        fe = 0.70
        area = 0.01  # m²
        gen = H2GenerationRate(j, fe, area)
        i_total = j * area
        i_her = i_total * (1 - fe)
        expected = i_her / (2 * FARADAY)
        assert gen.h2_mol_per_s == pytest.approx(expected, rel=1e-10)

    def test_sealed_accumulation_rate(self):
        """In a sealed enclosure, C(t) = G·t / V."""
        gen = H2GenerationRate(3000.0, 0.5, 0.01)
        V = 0.01  # 10 L
        enc = EnclosureSpec(V, 0.0)
        result = assess_h2_safety(gen, enc)
        # At time t_25_lel, C should equal alarm setpoint
        from models.h2_safety import VMOL_25C_1ATM_L
        vmol = VMOL_25C_1ATM_L * 1e-3
        t = result.time_to_25pct_lel_s
        c_mol_m3 = gen.h2_mol_per_s * t / V
        c_pct = c_mol_m3 * vmol * 100
        assert c_pct == pytest.approx(ALARM_SETPOINT_PERCENT, rel=1e-6)

    def test_steady_state_matches_analytic(self):
        """C_ss = G / Q for well-mixed enclosure."""
        gen = H2GenerationRate(2000.0, 0.6, 0.005)
        Q = 0.05  # m³/s
        V = 1.0
        enc = EnclosureSpec(V, Q)
        result = assess_h2_safety(gen, enc)
        from models.h2_safety import VMOL_25C_1ATM_L
        vmol = VMOL_25C_1ATM_L * 1e-3
        expected_c_pct = (gen.h2_mol_per_s / Q) * vmol * 100
        assert result.steady_state_percent == pytest.approx(expected_c_pct, rel=1e-10)

    def test_lel_constants(self):
        """LEL and alarm setpoint are self-consistent."""
        assert LEL_H2_VOLUME_PERCENT == 4.0
        assert ALARM_SETPOINT_PERCENT == pytest.approx(1.0)
        assert VENTILATION_SAFETY_FACTOR == 2.0
