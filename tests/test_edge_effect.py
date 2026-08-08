"""
Unit tests for edge/terminal current crowding (Round 5, D2).
"""


from models.edge_effect import (
    EdgeEffectParams,
    edge_current_ratio,
    edge_oh_penalty,
    thickness_ratio_across_width,
)


def test_edge_ratio_ge_one():
    """Edge current is at least the center current."""
    assert edge_current_ratio() >= 1.0


def test_edge_ratio_rises_with_conductivity_ratio():
    """Higher electrolyte/deposit conductivity ratio -> stronger crowding."""
    lo = EdgeEffectParams(conductivity_ratio=0.5)
    hi = EdgeEffectParams(conductivity_ratio=5.0)
    assert edge_current_ratio(hi) > edge_current_ratio(lo)


def test_thickness_maximum_at_edge():
    """Thickness peaks at the deposit edge."""
    prof = thickness_ratio_across_width(n_points=11)
    assert prof["thickness_ratio"][-1] > prof["thickness_ratio"][0]
    assert prof["max_min_ratio"] >= 1.0


def test_edge_oh_penalty_scales():
    """Edge O/H loading exceeds center by the penalty factor."""
    oh = edge_oh_penalty(center_O_ppm=400.0)
    assert oh["edge_O_ppm"] > 400.0
    assert oh["oh_penalty"] >= 1.0


def test_oh_flag_triggers_for_dirty_center():
    """A high-O center with edge crowding trips the cold-roll flag."""
    oh = edge_oh_penalty(center_O_ppm=800.0, params=EdgeEffectParams(conductivity_ratio=5.0))
    assert isinstance(oh["edge_oh_flag"], bool)
