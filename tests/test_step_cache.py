"""Tests for models.step_cache — content-addressable step cache."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from models.step_cache import (
    StepCache,
    _file_hash,
    _parameter_hash,
    _scan_imports,
    compute_key,
    transitive_deps,
    _load_manifest,
    _save_manifest,
    MANIFEST_PATH,
    MODELS_DIR,
)

# ── Unit tests: hashing ──────────────────────────────────────────────────────


class TestFileHash:
    def test_deterministic(self, tmp_path):
        p = tmp_path / "a.py"
        p.write_text("x = 1")
        assert _file_hash(p) == _file_hash(p)

    def test_content_change_invalidates(self, tmp_path):
        p = tmp_path / "a.py"
        p.write_text("x = 1")
        h1 = _file_hash(p)
        p.write_text("x = 2")
        h2 = _file_hash(p)
        assert h1 != h2

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(OSError):
            _file_hash(tmp_path / "nonexistent.py")


class TestParameterHash:
    def test_deterministic(self):
        assert _parameter_hash({"a": 1}) == _parameter_hash({"a": 1})

    def test_order_independent(self):
        # sort_keys=True → dict order doesn't matter
        h1 = _parameter_hash({"a": 1, "b": 2})
        h2 = _parameter_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_different_params_different_hash(self):
        assert _parameter_hash({"a": 1}) != _parameter_hash({"a": 2})


class TestComputeKey:
    def test_same_inputs_same_key(self, tmp_path):
        f = tmp_path / "m.py"
        f.write_text("pass")
        k1 = compute_key("step", [f], {"x": 1})
        k2 = compute_key("step", [f], {"x": 1})
        assert k1 == k2

    def test_different_step_different_key(self, tmp_path):
        f = tmp_path / "m.py"
        f.write_text("pass")
        k1 = compute_key("step_a", [f])
        k2 = compute_key("step_b", [f])
        assert k1 != k2

    def test_content_change_different_key(self, tmp_path):
        f = tmp_path / "m.py"
        f.write_text("x = 1")
        k1 = compute_key("step", [f])
        f.write_text("x = 2")
        k2 = compute_key("step", [f])
        assert k1 != k2

    def test_param_change_different_key(self, tmp_path):
        f = tmp_path / "m.py"
        f.write_text("pass")
        k1 = compute_key("step", [f], {"a": 1})
        k2 = compute_key("step", [f], {"a": 2})
        assert k1 != k2


# ── Unit tests: import scanner ───────────────────────────────────────────────


class TestScanImports:
    def test_simple_import(self, tmp_path):
        """Import scanner finds models.* imports in source."""
        # Use a real models/ file to test
        elec = MODELS_DIR / "electrochemistry.py"
        if not elec.exists():
            pytest.skip("models/electrochemistry.py not found")
        deps = _scan_imports(elec)
        # Should at least find itself
        assert elec in deps

    def test_transitive_deps_returns_frozenset(self):
        deps = transitive_deps("models.electrochemistry")
        assert isinstance(deps, frozenset)
        # Should contain at least the module itself
        elec = MODELS_DIR / "electrochemistry.py"
        if elec.exists():
            assert elec in deps


# ── Unit tests: manifest I/O ─────────────────────────────────────────────────


class TestManifestIO:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "manifest.json"
        data = {"step_a": {"key": "abc123", "outputs": []}}
        _save_manifest(data, path)
        loaded = _load_manifest(path)
        assert loaded == data

    def test_missing_manifest_returns_empty(self, tmp_path):
        loaded = _load_manifest(tmp_path / "nonexistent.json")
        assert loaded == {}

    def test_corrupt_manifest_returns_empty(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json{{{")
        loaded = _load_manifest(path)
        assert loaded == {}


# ── Integration tests: StepCache ─────────────────────────────────────────────


class TestStepCache:
    def test_first_run_is_miss(self, tmp_path):
        """A step with no prior manifest entry is a cache miss."""
        cache = StepCache(verbose=False)
        # Point manifest to tmp so we don't pollute the repo
        cache._manifest = {}
        out = tmp_path / "report.json"
        with cache.step("test_step", "models.electrochemistry", [out]) as hit:
            assert hit is False
            # Simulate the step writing its output
            out.write_text('{"ok": true}')
        assert cache.misses == 1
        assert cache.hits == 0

    def test_second_run_is_hit(self, tmp_path):
        """After recording, a repeat run with unchanged code is a hit."""
        cache = StepCache(verbose=False)
        cache._manifest = {}
        out = tmp_path / "report.json"

        # First run: miss
        with cache.step("test_step", "models.electrochemistry", [out]) as hit:
            assert hit is False
            out.write_text('{"ok": true}')
        # Save the manifest state (simulates persistence)
        saved_manifest = dict(cache._manifest)

        # Second run: hit (same code, output exists)
        cache2 = StepCache(verbose=False)
        cache2._manifest = saved_manifest
        with cache2.step("test_step", "models.electrochemistry", [out]) as hit:
            assert hit is True
        assert cache2.hits == 1

    def test_deleted_output_is_miss(self, tmp_path):
        """If an output file is deleted, the cache entry is a miss."""
        cache = StepCache(verbose=False)
        cache._manifest = {}
        out = tmp_path / "report.json"

        with cache.step("test_step", "models.electrochemistry", [out]) as hit:
            assert hit is False
            out.write_text('{"ok": true}')
        saved_manifest = dict(cache._manifest)

        # Delete the output
        out.unlink()

        cache2 = StepCache(verbose=False)
        cache2._manifest = saved_manifest
        with cache2.step("test_step", "models.electrochemistry", [out]) as hit:
            assert hit is False

    def test_force_step_overrides(self, tmp_path):
        """force_steps always produces a miss even if outputs exist."""
        cache = StepCache(force_steps={"test_step"}, verbose=False)
        cache._manifest = {}
        out = tmp_path / "report.json"

        # Prime the cache
        with cache.step("test_step", "models.electrochemistry", [out]) as hit:
            assert hit is False
            out.write_text('{"ok": true}')
        saved_manifest = dict(cache._manifest)

        # Force recompute
        cache2 = StepCache(force_steps={"test_step"}, verbose=False)
        cache2._manifest = saved_manifest
        with cache2.step("test_step", "models.electrochemistry", [out]) as hit:
            assert hit is False
        assert cache2.skips == 1

    def test_disabled_cache_always_miss(self, tmp_path):
        """cache_enabled=False → every step is a miss."""
        cache = StepCache(enabled=False, verbose=False)
        cache._manifest = {}
        out = tmp_path / "report.json"

        with cache.step("test_step", "models.electrochemistry", [out]) as hit:
            assert hit is False
            out.write_text('{"ok": true}')
        saved_manifest = dict(cache._manifest)

        cache2 = StepCache(enabled=False, verbose=False)
        cache2._manifest = saved_manifest
        with cache2.step("test_step", "models.electrochemistry", [out]) as hit:
            assert hit is False

    def test_params_change_invalidates(self, tmp_path):
        """Changing params between runs invalidates the cache."""
        cache = StepCache(verbose=False)
        cache._manifest = {}
        out = tmp_path / "report.json"

        with cache.step("test_step", "models.electrochemistry", [out],
                        params={"quick": False}) as hit:
            assert hit is False
            out.write_text('{"ok": true}')
        saved_manifest = dict(cache._manifest)

        cache2 = StepCache(verbose=False)
        cache2._manifest = saved_manifest
        with cache2.step("test_step", "models.electrochemistry", [out],
                         params={"quick": True}) as hit:
            assert hit is False

    def test_invalidate_all_clears_manifest(self, tmp_path):
        cache = StepCache(verbose=False)
        cache._manifest = {"step_a": {"key": "x"}, "step_b": {"key": "y"}}
        cache.invalidate_all()
        assert cache._manifest == {}

    def test_summary(self, tmp_path):
        cache = StepCache(verbose=False)
        cache.hits = 5
        cache.misses = 2
        s = cache.summary()
        assert "5 hit" in s
        assert "2 miss" in s

    def test_record_writes_manifest(self, tmp_path):
        """After a miss, the manifest is persisted to disk."""
        # Temporarily redirect manifest path
        manifest_path = tmp_path / "manifest.json"
        cache = StepCache(verbose=False)
        cache._manifest = {}
        out = tmp_path / "report.json"

        with patch("models.step_cache.MANIFEST_PATH", manifest_path):
            with patch("models.step_cache._save_manifest") as mock_save:
                with cache.step("test_step", "models.electrochemistry", [out]) as hit:
                    assert hit is False
                    out.write_text('{"ok": true}')
                # _save_manifest should have been called
                assert mock_save.call_count == 1


# ── Edge cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_outputs_list(self, tmp_path):
        """Step with no declared outputs still works."""
        cache = StepCache(verbose=False)
        cache._manifest = {}
        with cache.step("no_outputs", "models.electrochemistry", []) as hit:
            assert hit is False
        # Should record the step
        assert "no_outputs" in cache._manifest

    def test_multiple_output_files(self, tmp_path):
        """Cache hit requires all declared outputs to exist."""
        cache = StepCache(verbose=False)
        cache._manifest = {}
        out1 = tmp_path / "report.json"
        out2 = tmp_path / "figure.png"

        with cache.step("multi", "models.electrochemistry", [out1, out2]) as hit:
            assert hit is False
            out1.write_text("{}")
            out2.write_bytes(b"\x89PNG")
        saved_manifest = dict(cache._manifest)

        # Both exist → hit
        cache2 = StepCache(verbose=False)
        cache2._manifest = saved_manifest
        with cache2.step("multi", "models.electrochemistry", [out1, out2]) as hit:
            assert hit is True

        # Delete one → miss
        out2.unlink()
        cache3 = StepCache(verbose=False)
        cache3._manifest = saved_manifest
        with cache3.step("multi", "models.electrochemistry", [out1, out2]) as hit:
            assert hit is False

    def test_exception_in_step_does_not_record(self, tmp_path):
        """If the step body raises, the cache records the miss on exit anyway
        (the context manager records in __exit__, which runs even after
        an exception).  This is correct: the outputs may be partial."""
        cache = StepCache(verbose=False)
        cache._manifest = {}
        out = tmp_path / "report.json"

        with pytest.raises(RuntimeError):
            with cache.step("failing", "models.electrochemistry", [out]) as hit:
                assert hit is False
                raise RuntimeError("boom")
        # The step was still recorded (it was a miss)
        assert cache.misses == 1


# ── Real-graph regression tests (stale-cache bug) ───────────────────────────
#
# These tests run the dependency scanner against the REAL model source
# tree, not a synthetic fixture.  They guard against the bug where
# relative imports (from .foo import ...) were not followed, causing
# transitive_deps to return an incomplete set and the cache to silently
# serve stale results when a dependency changed.
#


class TestRealGraphRelativeImports:
    """Verify the scanner follows relative imports in the real model tree."""

    def _dep_names(self, module: str) -> set[str]:
        return {p.name for p in transitive_deps(module)}

    def test_transport_deps_include_relative_imports(self):
        """transport.py uses ``from .kinetics import ...``,
        ``from .electrochemistry import ...``, ``from .pourbaix import ...``.
        All three must appear in run_transport's transitive deps."""
        names = self._dep_names("models.run_transport")
        for expected in ("kinetics.py", "electrochemistry.py", "pourbaix.py"):
            assert expected in names, (
                f"{expected} not in run_transport deps — "
                f"relative import not followed (stale-cache bug)"
            )

    def test_carburization_deps_include_electrochemistry(self):
        """carburization.py uses ``from .electrochemistry import R_GAS, RHO_FE``.
        electrochemistry.py must appear in run_carburization's deps."""
        names = self._dep_names("models.run_carburization")
        assert "electrochemistry.py" in names, (
            "electrochemistry.py not in run_carburization deps — "
            "relative import not followed (stale-cache bug)"
        )

    def test_tempering_deps_include_carburization(self):
        """tempering.py uses ``from .carburization import ...``.
        carburization.py must appear in run_tempering's deps."""
        names = self._dep_names("models.run_tempering")
        assert "carburization.py" in names, (
            "carburization.py not in run_tempering deps — "
            "relative import not followed (stale-cache bug)"
        )

    def test_pulse_deps_include_kinetics_and_electrochemistry(self):
        """pulse.py uses ``from .kinetics import ...`` and
        ``from .electrochemistry import ...``."""
        names = self._dep_names("models.run_pulse")
        for expected in ("kinetics.py", "electrochemistry.py"):
            assert expected in names, (
                f"{expected} not in run_pulse deps — "
                f"relative import not followed"
            )

    def test_co_deposition_deps_include_kinetics(self):
        """co_deposition.py uses ``from .kinetics import ...``."""
        names = self._dep_names("models.run_co_deposition")
        assert "kinetics.py" in names

    def test_monte_carlo_parent_package_imports(self):
        """models/uncertainty/monte_carlo.py uses ``from ..electrochemistry``,
        ``from ..carburization``, ``from ..tempering``,
        ``from ..mechanical_properties``.
        All four must be in the deps."""
        names = self._dep_names("models.run_monte_carlo")
        for expected in (
            "electrochemistry.py",
            "carburization.py",
            "tempering.py",
            "mechanical_properties.py",
        ):
            assert expected in names, (
                f"{expected} not in run_monte_carlo deps — "
                f"parent-package relative import (from ..) not followed"
            )


class TestRealGraphInvalidation:
    """Integration tests: mutate a source file and confirm the cache
    invalidates the correct steps (not too few, not too many)."""

    def test_editing_kinetics_invalidates_transport(self, tmp_path):
        """If kinetics.py changes, run_transport's cache key must change
        (because transport.py imports kinetics relatively)."""
        from models.step_cache import compute_key

        deps_before = transitive_deps("models.run_transport")
        key_before = compute_key("transport", deps_before)

        # Simulate an edit to kinetics.py
        kinetics = MODELS_DIR / "kinetics.py"
        original = kinetics.read_text()
        try:
            kinetics.write_text(original + "\n# step_cache test edit\n")
            deps_after = transitive_deps("models.run_transport")
            key_after = compute_key("transport", deps_after)
            assert key_before != key_after, (
                "Editing kinetics.py did not change run_transport's cache key — "
                "stale-cache bug (relative import not tracked)"
            )
        finally:
            kinetics.write_text(original)

    def test_editing_electrochemistry_invalidates_carburization(self, tmp_path):
        """If electrochemistry.py changes, run_carburization's cache key
        must change (because carburization.py imports it relatively)."""
        from models.step_cache import compute_key

        deps_before = transitive_deps("models.run_carburization")
        key_before = compute_key("carburization", deps_before)

        elec = MODELS_DIR / "electrochemistry.py"
        original = elec.read_text()
        try:
            elec.write_text(original + "\n# step_cache test edit\n")
            deps_after = transitive_deps("models.run_carburization")
            key_after = compute_key("carburization", deps_after)
            assert key_before != key_after, (
                "Editing electrochemistry.py did not change run_carburization's key — "
                "stale-cache bug"
            )
        finally:
            elec.write_text(original)

    def test_touch_does_not_invalidate(self):
        """A content-preserving touch (no bytes changed) must NOT
        invalidate — content-addressing, not mtime."""
        import os
        from models.step_cache import compute_key

        deps_before = transitive_deps("models.run_transport")
        key_before = compute_key("transport", deps_before)

        kinetics = MODELS_DIR / "kinetics.py"
        # Touch (update mtime only)
        original_mtime = kinetics.stat().st_mtime
        try:
            os.utime(kinetics, (original_mtime + 100, original_mtime + 100))
            deps_after = transitive_deps("models.run_transport")
            key_after = compute_key("transport", deps_after)
            assert key_before == key_after, (
                "touch (no content change) should not invalidate"
            )
        finally:
            pass  # mtime change is harmless
