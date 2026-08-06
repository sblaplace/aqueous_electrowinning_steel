# Testing tiers

The full test suite is large enough (~1,400 tests) that a single serial `pytest`
run takes 20+ minutes and can exceed local runtime caps. It is therefore split
into two tiers so the fast loop stays quick, while the full-fidelity tier
remains available on demand. This is a marker-based tiering (`pytest.mark.slow`),
configured in `pyproject.toml` `[tool.pytest.ini_options].markers`.

## Tiers

**Fast tier — `pytest -m "not slow"`** (~1,250 tests)
Run often (per commit, per card, during iteration). Must stay small enough to
fit comfortably under any runtime cap. This is the gate that CI enforces.

**Slow tier — `pytest -m slow`** (~125 tests)
Full-fidelity / computationally heavy: pulse frequency sweeps, Monte Carlo,
Sobol global sensitivity, Bayesian/EIS fitting, Pareto optimization, benchmarks,
scale-up/design-space exploration. Run on demand (`make test-slow`) or nightly —
not in the normal feedback loop.

A module that grows an expensive test should move that test behind the `slow`
mark so the fast tier stays fast.

## Parallel runs

`pytest -n auto` (xdist) runs the **whole** suite — including the slow tier — in
about 7 minutes here, comfortably inside any runtime cap. So the simplest way to
get full coverage locally is `make test-all` (or `pytest -n auto`). The slow
marker matters when you want a *serial* or *quick* green without the expensive
FE/sweep tests, and for CI gating. The `slow` files were chosen from measured
durations (the FE engine, hull-cell FE, bath dynamics, Monte Carlo, Sobol,
Bayesian/EIS, optimization and benchmark tests dominate serial wall-time).

## numba is load-bearing for wall-time (2026-08 profiling)

Every wall-time number above assumes **numba is importable**:
`models/transport.py` integrates the Nernst–Planck film through
`models/_transport_jit.py` when numba is present and silently falls back to
`scipy.solve_ivp` (LSODA) when it is not. Measured on the RC-1 reference bath
the fallback is **~17× slower per coupled solve** (10-point `CellPhysics`
sweep: 35.1 s → 2.1 s; the physics-derived economics suite: >60 s → ~3.4 s;
full fast tier, `pytest -q -n auto -m "not slow"`: never finished >
15 CPU-min in one worker → **~3.5 min** with numba; 1,398 tests).

numba is a hard dependency in `pyproject.toml` (`[project.dependencies]`), so
CI (`pip install -e ".[test]"`) has always been on the fast path. The drift
was the sandbox/dev path: `requirements.txt` (consumed by
`scripts/arena_setup.sh`) omitted numba and pyyaml even though
`models/` imports both at runtime. Both were added on 2026-08; **keep
`requirements.txt` mirroring `pyproject.toml [project.dependencies]`** —
a silently missing numba degrades every NP-coupled test without failing
anything, which is the worst kind of slow.

The JIT functions are compiled with `cache=True`, so the one-time compile
cost lands on first use per checkout, not per test process.

## Commands (make)

    make test         fast tier, parallel (pytest -n auto -m "not slow")
    make test-slow    full-fidelity tier (pytest -n auto -m slow)
    make test-all     whole suite, parallel (~7 min) — the complete gate
    make test-change  only tests for modules you changed (fastest targeted trigger)
    make test-fail    last-failed only (failure-focused cache; safe, built-in)
    make test-inc     incremental heavy tier via pytest-testmon (opt-in)

## Change-scoped selection ("trigger in part")

`scripts/test_changed.sh` (wrapped by `make test-change`) runs only the mirror
test file for each changed `models/*.py` module (plus any directly-changed
`tests/*.py`). Because tests mirror their module (`tests/test_<module>.py`),
this is a clean, dependency-free partial trigger:

    bash scripts/test_changed.sh            # unstaged + staged changes
    bash scripts/test_changed.sh origin/main # changes since origin/main

When CI runs on a PR, the same idea can be applied against the merge base so
only the tests touching the diff run locally.

## Caching unchanged results

Two flavors are supported, with very different risk:

1. **pytest's built-in last-failed cache (`--lf` / `--ff`)** — safe, no plugin.
   Re-runs only the tests that failed previously. Failure-focused, so it cannot
   mask a regression in a test that previously passed. Use `make test-fail`.

2. **pytest-testmon (opt-in, `make test-inc` / `install-testmon`)** — the real
   "only re-run tests whose covered lines changed" tool. It records per-test
   coverage and selects affected tests on change. This is incremental
   verification for the heavy tier. Caveat: testmon assumes deterministic tests
   and adds a coverage dependency; only rely on it for incrementality, and still
   run the full fast tier fresh (`make test`) before any release claim.

We intentionally do **not** cache "pass" verdicts with a custom hash store: a
cached pass is a claim that an input set is unchanged, and missing any
dependency (transitive imports, pytest version, RNG state) yields a silent
false-green. `--lf` (failure-only) and testmon (coverage-scoped) stay inside
that safety boundary.
