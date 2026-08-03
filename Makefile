# Test tiering for the aqueous-electrowinning-steel suite.
#
# The full suite is long (>20 min serial) and can exceed local runtime caps,
# so it is split into a fast feedback tier and a full-fidelity slow tier.
# See docs/TESTING.md for the rationale and the `slow` marker.

.PHONY: test test-slow test-all test-change test-fail test-inc install-testmon arena-setup

## Fast feedback tier — parallel; fits easily under runtime caps
test:
	pytest -n auto -m "not slow" -q

# Arena AI sandbox / fresh-clone one-liner: re-establishes the venv + deps +
# build tool after a workspace reset (OS packages don't persist; this does).
arena-setup:
	bash scripts/arena_setup.sh

## Full-fidelity tier — heavy tests (FE engine, hull-cell FE, bath dynamics, Monte Carlo, Sobol, Bayesian/EIS, optimization, benchmarks)
test-slow:
	pytest -n auto -m slow -q

## Whole suite in parallel (~7 min with xdist) — the complete gate
test-all:
	pytest -n auto -q

## Change-scoped selection — only tests for modules you changed (fastest targeted trigger)
test-change:
	bash scripts/test_changed.sh

## Last-failed only — re-run failures from the previous run (safe, failure-focused cache)
test-fail:
	pytest -n auto -m "not slow" --lf -q

## Incremental heavy tier via pytest-testmon (opt-in; see docs/TESTING.md)
test-inc:
	pytest --testmon -q

## Install pytest-testmon (change-driven test selection with coverage)
install-testmon:
	pip install pytest-testmon
