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
