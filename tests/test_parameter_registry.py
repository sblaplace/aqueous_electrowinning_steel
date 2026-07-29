"""Tests for the parameter registry and sampling utilities."""

from __future__ import annotations

import math
import numpy as np
import pytest

from models.uncertainty import (
    Parameter,
    REGISTRY,
    registry_summary,
    sample_parameters,
    parameter_matrix_to_kwargs,
    sobol_sequence,
)


# ── Test 1: Registry count and structure ─────────────────────────────────

def test_registry_covers_all_modules():
    """REGISTRY must have >= 40 parameters across all model modules."""
    summary = registry_summary()
    assert summary["total_parameters"] >= 40, (
        f"Need >= 40 params, got {summary['total_parameters']}"
    )
    # Must cover the key modules listed in the task
    expected_modules = {
        "mechanical_properties",
        "carburization",
        "carbon_potential",
        "tempering",
        "transport",
        "kinetics",
        "co_deposition",
        "pulse",
        "anode",
        "closed_loop",
    }
    present = set(summary["by_module"].keys())
    missing = expected_modules - present
    assert not missing, f"Missing modules in registry: {missing}"


def test_every_parameter_has_required_fields():
    """Each Parameter must have mean, std, bounds, distribution, source."""
    for name, p in REGISTRY.items():
        assert isinstance(p, Parameter), f"{name} is not a Parameter"
        assert p.name == name, f"Parameter name mismatch: {p.name} != {name}"
        assert isinstance(p.mean, (int, float)), f"{name}: mean not numeric"
        assert p.std >= 0, f"{name}: std must be non-negative"
        assert len(p.bounds) == 2, f"{name}: bounds must be a 2-tuple"
        lo, hi = p.bounds
        assert lo < hi, f"{name}: bounds[0] ({lo}) must be < bounds[1] ({hi})"
        assert p.distribution in ("normal", "uniform", "lognormal", "triangular"), (
            f"{name}: unknown distribution {p.distribution}"
        )
        assert p.source, f"{name}: source must be non-empty"


# ── Test 2: Sampling produces values within bounds ──────────────────────

def test_sample_parameters_within_bounds():
    """All sampled values must lie within their parameter bounds."""
    samples = sample_parameters(200, seed=42)
    assert len(samples) == 200

    for sample in samples:
        for name, value in sample.items():
            p = REGISTRY[name]
            lo, hi = p.bounds
            assert lo <= value <= hi, (
                f"{name} = {value} outside bounds [{lo}, {hi}]"
            )


def test_sample_parameters_reproducible():
    """Same seed must produce identical samples."""
    s1 = sample_parameters(50, seed=123)
    s2 = sample_parameters(50, seed=123)
    for d1, d2 in zip(s1, s2):
        for k in d1:
            assert d1[k] == pytest.approx(d2[k], abs=1e-15), (
                f"{k}: non-reproducible with same seed"
            )


# ── Test 3: Distribution-specific tests ─────────────────────────────────

def test_lognormal_params_stay_positive():
    """Lognormal parameters must always produce strictly positive samples."""
    lognormal_names = [
        name for name, p in REGISTRY.items()
        if p.distribution == "lognormal"
    ]
    assert len(lognormal_names) > 0, "No lognormal params in registry"
    samples = sample_parameters(200, seed=99)
    for sample in samples:
        for name in lognormal_names:
            assert sample[name] > 0, (
                f"Lognormal param {name} = {sample[name]} <= 0"
            )


def test_triangular_samples_centered_on_mean():
    """Triangular samples should have sample mean near the distribution mean."""
    tri_names = [
        name for name, p in REGISTRY.items()
        if p.distribution == "triangular"
    ]
    assert len(tri_names) > 0, "No triangular params in registry"
    samples = sample_parameters(500, seed=77)
    for name in tri_names:
        values = [s[name] for s in samples]
        sample_mean = float(np.mean(values))
        p = REGISTRY[name]
        lo, hi = p.bounds
        # For triangular(a, mode, b), distribution mean = (a + mode + b) / 3
        dist_mean = (lo + p.mean + hi) / 3.0
        # Sample mean should be within a reasonable tolerance of distribution mean
        half_width = (hi - lo) / 2.0
        assert abs(sample_mean - dist_mean) < half_width * 0.15, (
            f"Triangular {name}: sample mean {sample_mean} far from dist mean {dist_mean}"
        )


# ── Test 4: Sobol sequence fills the space ──────────────────────────────

def test_sobol_sequence_shape_and_range():
    """Sobol sequence must be (n, d) with values in [0, 1)."""
    seq = sobol_sequence(64, 5, seed=0)
    assert seq.shape == (64, 5)
    assert np.all(seq >= 0.0) and np.all(seq < 1.0)


def test_sobol_sequence_low_discrepancy():
    """Sobol fills the unit cube more uniformly than pseudorandom."""
    d = 3
    n = 256
    sob = sobol_sequence(n, d)
    rng = np.random.default_rng(42)
    mc = rng.random((n, d))

    # Check: divide each dimension into 4 bins; Sobol should hit more bins
    n_bins = 4
    sob_hits = set()
    mc_hits = set()
    for i in range(n):
        sob_bins = tuple(int(min(sob[i, j] * n_bins, n_bins - 1)) for j in range(d))
        mc_bins = tuple(int(min(mc[i, j] * n_bins, n_bins - 1)) for j in range(d))
        sob_hits.add(sob_bins)
        mc_hits.add(mc_bins)

    # Sobol should cover more cells
    assert len(sob_hits) >= len(mc_hits), (
        f"Sobol {len(sob_hits)} cells < MC {len(mc_hits)} cells — not more uniform"
    )


# ── Test 5: parameter_matrix_to_kwargs dispatch ─────────────────────────

def test_parameter_matrix_to_kwargs():
    """parameter_matrix_to_kwargs must map registry names to model kwargs."""
    sample = {name: p.mean for name, p in REGISTRY.items()}
    mech = parameter_matrix_to_kwargs(sample, "mechanical_properties")
    assert "sigma0_MPa" in mech, "Missing sigma0_MPa in mechanical kwargs"
    assert "k_hp_MPa_sqrt_m" in mech
    assert mech["sigma0_MPa"] == pytest.approx(100.0)

    kin = parameter_matrix_to_kwargs(sample, "kinetics")
    assert "fe_i0" in kin
    assert kin["fe_i0"] == pytest.approx(1.0e-2)


# ── Test 6: Sample with sobol method ────────────────────────────────────

def test_sample_sobol_method():
    """sample_parameters(method='sobol') must work and stay in bounds."""
    samples = sample_parameters(100, seed=0, method="sobol")
    assert len(samples) == 100
    for sample in samples:
        for name, value in sample.items():
            p = REGISTRY[name]
            lo, hi = p.bounds
            assert lo <= value <= hi, (
                f"Sobol sample {name}={value} outside [{lo}, {hi}]"
            )


# ── Test 7: Registry importable from expected path ──────────────────────

def test_import_path():
    """REGISTRY and sample_parameters must be importable from models.uncertainty."""
    import models.uncertainty as unc
    assert hasattr(unc, "REGISTRY")
    assert hasattr(unc, "sample_parameters")
    assert hasattr(unc, "Parameter")
    assert hasattr(unc, "parameter_matrix_to_kwargs")
    assert hasattr(unc, "sobol_sequence")
