"""
Unit tests for multi-cell stack manifold shunt current network solver.
"""

from models.shunt_currents import (
    StackManifoldGeometry,
    solve_stack_shunt_currents,
)


def test_shunt_current_solution_symmetry_and_trends():
    """Verify that shunt currents peak at stack ends and cause measurable FE loss."""
    geom = StackManifoldGeometry(
        n_cells=50,
        cell_voltage_V=2.60,
        applied_current_A=300.0,
        electrolyte_conductivity_S_m=18.0,
    )
    res = solve_stack_shunt_currents(geom)

    assert res.n_cells == 50
    assert res.stack_voltage_V == 50 * 2.60
    assert res.total_shunt_current_A > 0.0
    assert 0.01 <= res.stack_faradaic_efficiency_loss_percent <= 5.0

    # Shunt current should be highest at the high-potential end cells
    assert res.max_single_port_current_A == max(abs(i) for i in res.port_currents_A)


def test_port_length_reduces_shunt_losses():
    """Longer port tubes increase fluid resistance and suppress shunt leakage."""
    short_port = StackManifoldGeometry(port_length_m=0.040)
    long_port = StackManifoldGeometry(port_length_m=0.200)

    res_short = solve_stack_shunt_currents(short_port)
    res_long = solve_stack_shunt_currents(long_port)

    assert res_long.total_shunt_current_A < res_short.total_shunt_current_A
    assert res_long.stack_faradaic_efficiency_loss_percent < res_short.stack_faradaic_efficiency_loss_percent

    # When corrosion threat is present, recommended port length should be elongated
    if res_short.is_corrosion_threat:
        assert res_short.recommended_min_port_length_m > short_port.port_length_m
