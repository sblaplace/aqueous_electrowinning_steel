# Test tiering for the aqueous-electrowinning-steel suite.
#
# The full suite is long (>20 min serial) and can exceed local runtime caps,
# so it is split into a fast feedback tier and a full-fidelity slow tier.
# See docs/TESTING.md for the rationale and the `slow` marker.
#
# NUMBA_CACHE_DIR is hoisted into every recipe so the JIT cache survives Arena
# sandbox resets (Arena wipes __pycache__/ but NOT .numba_cache/).

export NUMBA_CACHE_DIR := $(CURDIR)/.numba_cache
export PYTHONPATH := $(CURDIR):$(PYTHONPATH)
VENV_PY := $(CURDIR)/.venv/bin/python

.PHONY: test test-slow test-all test-change test-fail test-inc install-testmon arena-setup warmup compileall run-all run-all-fresh run-all-quick run-all-force cache-status

## Fast feedback tier — parallel; fits easily under runtime caps
test:
	pytest -n auto -m "not slow" -q

# Arena AI sandbox / fresh-clone one-liner: re-establishes the venv + deps +
# build tool after a workspace reset (OS packages don't persist; this does).
arena-setup:
	bash scripts/arena_setup.sh

## Pre-compile Python bytecode for the whole tree (saves ~30-60s of lazy
## compilation on first import after a sandbox reset).
compileall:
	python -m compileall -q models/ tests/

## Pre-compile Numba JIT functions so the first transport solve doesn't pay
## the LLVM compile cost.  Requires the venv to be set up first.
warmup: compileall
	@mkdir -p $(NUMBA_CACHE_DIR)
	@echo "==> warming up numba JIT (one-shot compile cost, cached to .numba_cache/)"
	python -c "\
from models._transport_jit import get_integrate_film_jit; \
fn = get_integrate_film_jit(); \
import numpy as np; \
y0 = np.array([100.0, 1e-4, 500.0, 250.0]); \
x_eval = np.linspace(0, 1e-4, 20); \
fn(y0, 2.0, -1.0, 0.0, 1e-4, x_eval, \
   7e-10, 9.3e-9, 5.3e-9, 1.3e-9, 1.1e-9, 38.7, 1e-20); \
print('   numba JIT warmup complete')\
"

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

## Full model suite (incremental — skips unchanged steps via content-addressed cache)
run-all:
	$(VENV_PY) -m models.run_all

## Full model suite — force recompute of everything (ignore step cache)
run-all-fresh:
	$(VENV_PY) -m models.run_all --no-cache

## Full model suite — quick mode (skip heavy pulse grids)
run-all-quick:
	$(VENV_PY) -m models.run_all --quick

## Force recompute of one step (e.g. make run-all-force STEP=electrochemistry)
run-all-force:
	$(VENV_PY) -m models.run_all --force-step $(STEP)

## Show step cache status
cache-status:
	@cat experiments/data/.step_cache/manifest.json 2>/dev/null | $(VENV_PY) -m json.tool || echo "No cache manifest found"
