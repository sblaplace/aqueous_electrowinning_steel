"""
Unit tests for the Round 3 advanced physical and chemical modeling runner.
"""

from models.run_physics_tranche3 import run_physics_tranche3


def test_run_physics_tranche3_output():
    """Verify that run_physics_tranche3 generates all 9 structured sections."""
    rep = run_physics_tranche3()

    assert "solubility_and_scaling" in rep
    assert "pulse_rc_filtering" in rep
    assert "bdd_microkinetics" in rep
    assert "stack_shunt_currents" in rep
    assert "hydrogen_trapping_and_bakeout" in rep
    assert "ore_leaching" in rep
    assert "chemical_osmosis" in rep
    assert "tempering_lsw_kinetics" in rep
    assert "solutal_mixed_convection" in rep

    assert rep["ore_leaching"]["reductive_recovery_4h"] >= 70.0
    assert rep["tempering_lsw_kinetics"]["yield_strength_MPa"] > 300.0
    assert rep["solutal_mixed_convection"]["is_downward_reversal_threat"]
