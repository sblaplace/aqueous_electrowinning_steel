"""Tests for the process candidate registry loader."""

import pytest

from models.process_registry import (
    load_registry,
    get_candidate,
    registry_summary,
    VALID_CANDIDATE_STATUSES,
    VALID_PRIORITIES,
    VALID_PRODUCT_FORMS,
    VALID_DEPLOYMENTS,
)


class TestRegistryLoads:
    def test_loads_candidates(self):
        candidates = load_registry()
        assert len(candidates) >= 3

    def test_candidate_ids_unique(self):
        candidates = load_registry()
        ids = [c.id for c in candidates]
        assert len(ids) == len(set(ids))

    def test_status_values_valid(self):
        for c in load_registry():
            assert c.status in VALID_CANDIDATE_STATUSES
            assert c.priority in VALID_PRIORITIES
            assert c.product_form in VALID_PRODUCT_FORMS
            assert c.deployment in VALID_DEPLOYMENTS

    def test_primary_candidate_present(self):
        c = get_candidate("divided_sulfate_dissolved_feed")
        assert c.priority == "primary"
        assert c.status == "active"

    def test_candidate_has_gates(self):
        for c in load_registry():
            assert len(c.gates) > 0

    def test_assumed_operations_flagged(self):
        c = get_candidate("divided_sulfate_dissolved_feed")
        assumed = c.assumed_operations()
        assert len(assumed) > 0
        assert all(op.assumed for op in assumed)

    def test_pending_gates(self):
        c = get_candidate("divided_sulfate_dissolved_feed")
        pending = c.pending_gates()
        assert len(pending) == len(c.gates)
        assert all(g.status == "pending" for g in pending)

    def test_summary_shape(self):
        summary = registry_summary()
        assert summary["n_candidates"] >= 3
        assert all("n_pending_gates" in c for c in summary["candidates"])


class TestRegistryValidation:
    def test_duplicate_id_rejected(self, tmp_path):
        import yaml
        bad = {
            "candidates": [
                {"id": "x", "name": "A", "status": "active", "priority": "primary",
                 "product_form": "flake", "deployment": "modular_proving_ground",
                 "summary": "s", "feedstocks": [], "unit_operations": [], "gates": []},
                {"id": "x", "name": "B", "status": "active", "priority": "primary",
                 "product_form": "flake", "deployment": "modular_proving_ground",
                 "summary": "s", "feedstocks": [], "unit_operations": [], "gates": []},
            ]
        }
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.dump(bad))
        with pytest.raises(ValueError, match="Duplicate candidate id"):
            load_registry(path)

    def test_invalid_status_rejected(self, tmp_path):
        import yaml
        bad = {
            "candidates": [
                {"id": "x", "name": "A", "status": "bogus", "priority": "primary",
                 "product_form": "flake", "deployment": "modular_proving_ground",
                 "summary": "s", "feedstocks": [], "unit_operations": [], "gates": []},
            ]
        }
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.dump(bad))
        with pytest.raises(ValueError, match="invalid status"):
            load_registry(path)
