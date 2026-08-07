"""Tests for the literature-anchor registry.

These are Tier-3.2 (cross-cutting) tests: they pin the screening
central values in models/surface_state and models/fe_chloride_speciation
to the published numbers in the anchor registry, so the screening
budget is auditable in a single pytest run.
"""
from __future__ import annotations

import pytest

from models.anchors import ANCHORS, Anchor, audit_anchors, get_anchor
from models.surface_state import (
    BORATE_AWARE, CL_NA_AWARE, HSO4_AWARE, SO4_AWARE,
    DG_HSTAR_FE100_J, DG_HSTAR_FE110_J, DG_HSTAR_FE211_J,
    TEMKIN_G_H_J_MOL,
)
from models.fe_chloride_speciation import (
    FECL2_PITZER,
    LOG10_K_FECL2_AQ_25, LOG10_K_FECL3_MINUS_25, LOG10_K_FECL_PLUS_25,
)


class TestAnchorRegistry:
    def test_anchors_dict_is_nonempty(self):
        assert len(ANCHORS) > 5

    def test_all_entries_are_anchor_instances(self):
        for v in ANCHORS.values():
            assert isinstance(v, Anchor)

    def test_get_anchor_known_key(self):
        a = get_anchor("DG_HSTAR_FE110")
        assert a.value < 0.0
        assert a.ref.startswith("Nørskov")

    def test_get_anchor_unknown_raises(self):
        with pytest.raises(KeyError):
            get_anchor("NOT_A_REAL_ANCHOR")


class TestSurfaceStateAnchors:
    def test_dg_hstar_fe110_within_tolerance(self):
        """The screening ΔG_H*(Fe110) should be within ±0.15 eV of
        the Nørskov CHE volcano central value."""
        a = get_anchor("DG_HSTAR_FE110")
        assert abs(DG_HSTAR_FE110_J - a.value) < 1.0  # exact match
        assert abs(DG_HSTAR_FE110_J - a.paper_value) <= a.uncertainty

    def test_dg_hstar_fe100_and_fe211_anchored(self):
        """Both surface_state facet anchors exist."""
        get_anchor("DG_HSTAR_FE100")
        get_anchor("DG_HSTAR_FE211")
        assert DG_HSTAR_FE100_J < DG_HSTAR_FE110_J
        assert DG_HSTAR_FE211_J > DG_HSTAR_FE110_J

    def test_temkin_g_within_jerkiewicz_range(self):
        """Temkin g should be in the literature 5-15 kJ/mol range on Fe."""
        a = get_anchor("TEMKIN_G_H_FE")
        assert abs(TEMKIN_G_H_J_MOL - a.value) < 1.0
        # And within the paper's range.
        assert abs(TEMKIN_G_H_J_MOL - a.paper_value) <= a.uncertainty

    def test_anion_dg_ads_within_tolerances(self):
        for anion, key in (
            (CL_NA_AWARE, "DG_ADS_CL_FE"),
            (SO4_AWARE, "DG_ADS_SO4_FE"),
            (HSO4_AWARE, "DG_ADS_HSO4_FE"),
            (BORATE_AWARE, "DG_ADS_BORATE_FE"),
        ):
            a = get_anchor(key)
            assert abs(anion.DG_ads_J_mol - a.value) < 1.0
            # And within the paper's reported range.
            assert abs(anion.DG_ads_J_mol - a.paper_value) <= a.uncertainty


class TestFeChlorideAnchors:
    def test_log10_K_fecL_plus_anchored(self):
        a = get_anchor("LOG10_K_FECL_PLUS")
        assert abs(LOG10_K_FECL_PLUS_25 - a.value) < 1e-6
        assert abs(LOG10_K_FECL_PLUS_25 - a.paper_value) <= a.uncertainty

    def test_log10_K_fecL2_anchored(self):
        a = get_anchor("LOG10_K_FECL2_AQ")
        assert abs(LOG10_K_FECL2_AQ_25 - a.value) < 1e-6
        assert abs(LOG10_K_FECL2_AQ_25 - a.paper_value) <= a.uncertainty

    def test_log10_K_fecL3_anchored(self):
        a = get_anchor("LOG10_K_FECL3_MINUS")
        assert abs(LOG10_K_FECL3_MINUS_25 - a.value) < 1e-6
        assert abs(LOG10_K_FECL3_MINUS_25 - a.paper_value) <= a.uncertainty

    def test_pitzer_betas_anchored(self):
        a0 = get_anchor("FECL2_BETA0")
        a1 = get_anchor("FECL2_BETA1")
        assert abs(FECL2_PITZER.beta0 - a0.value) < 1e-6
        assert abs(FECL2_PITZER.beta1 - a1.value) < 1e-6


class TestAuditFunction:
    def test_audit_returns_within_tolerance_dict(self):
        result = audit_anchors()
        # Every currently-anchored value is exact, so all True.
        assert all(result.values())

    def test_audit_covers_all_keys(self):
        result = audit_anchors()
        assert set(result.keys()) == set(ANCHORS.keys())


class TestAnchorStructure:
    """Schema tests — adding a new anchor should require all the
    right fields, and the schema should be stable."""

    def test_anchor_has_required_fields(self):
        for k, a in ANCHORS.items():
            assert a.key == k
            assert isinstance(a.value, float)
            assert isinstance(a.paper_value, float)
            assert a.uncertainty >= 0.0
            assert a.ref  # non-empty
            assert isinstance(a.notes, str)

    def test_within_tolerance_logic(self):
        """Anchor.within_tolerance should return True iff |value - paper| ≤ uncertainty."""
        # Make a synthetic anchor that is and is not within tolerance.
        a_tight = Anchor("a", 1.0, 1.05, 0.1, "x")          # |0.05| <= 0.1: OK
        a_outside = Anchor("b", 1.0, 2.0, 0.5, "y")          # |1.0| > 0.5: not OK
        assert a_tight.within_tolerance()
        assert not a_outside.within_tolerance()
