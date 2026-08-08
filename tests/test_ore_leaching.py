"""
Unit tests for shrinking core ore leaching and reductive leaching kinetics.
"""

from models.ore_leaching import (
    OreSpec,
    simulate_ore_leaching,
)


def test_reductive_leaching_accelerates_dissolution():
    """Reductive leaching (Fe0/SO2) drastically increases extraction kinetics over direct acid."""
    ore = OreSpec(mineral_type="hematite", particle_p80_um=75.0)

    # 4-hour direct acid leach vs reductive leach at 80 °C
    res_direct = simulate_ore_leaching(
        ore=ore,
        temperature_C=80.0,
        acid_concentration_M=2.0,
        residence_time_hours=4.0,
        use_reductant=False,
    )
    res_reductive = simulate_ore_leaching(
        ore=ore,
        temperature_C=80.0,
        acid_concentration_M=2.0,
        residence_time_hours=4.0,
        use_reductant=True,
    )

    # Absolute conversion asserts
    assert res_reductive.fe_recovery_fraction >= 0.70  # High recovery in 4h at 80 °C with reductant
    assert res_direct.fe_recovery_fraction < res_reductive.fe_recovery_fraction
    assert res_reductive.fe2_to_fe3_product_ratio >= 0.90
    assert res_direct.fe2_to_fe3_product_ratio <= 0.10


def test_temperature_dependence_of_leaching():
    """Higher temperature accelerates shrinking core dissolution."""
    ore = OreSpec(mineral_type="hematite")

    res_cold = simulate_ore_leaching(ore=ore, temperature_C=40.0, residence_time_hours=3.0)
    res_hot = simulate_ore_leaching(ore=ore, temperature_C=85.0, residence_time_hours=3.0)

    assert res_hot.fe_recovery_fraction > res_cold.fe_recovery_fraction
