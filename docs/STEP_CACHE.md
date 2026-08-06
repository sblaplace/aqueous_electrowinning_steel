# Step Cache — Incremental Compute for the Model Suite

The `run_all` orchestrator runs 22+ modeling steps that produce JSON reports
and PNG figures.  Without caching, every invocation recomputes everything from
scratch (~50–60 s even in `--quick` mode).  The **step cache** makes repeated
runs incremental: only steps whose source code, dependencies, or parameters
changed are recomputed.

## How it works

Each step is (approximately) a pure function of:

1. **Source code** — the step's own module and every `models/*.py` it
   transitively imports (the dependency graph within the package).
2. **Parameters** — runtime kwargs like `{"quick": True}` or
   `{"temperature": 900.0}`.

The cache key is a SHA-256 digest of all those inputs.  If the key matches
the recorded entry in the manifest and all output files still exist on disk,
the step is a **cache hit** and its computation is skipped.

The manifest lives at `experiments/data/.step_cache/manifest.json` (gitignored;
safe to delete — the next run rebuilds it from scratch).

## Performance

| Scenario | Time | Notes |
|---|---|---|
| Full recompute (`--no-cache`) | ~50–60 s | Every step runs |
| Incremental (nothing changed) | ~3–4 s | All 28 steps cached |
| One model file changed | ~5–8 s | Only affected steps rerun |
| `--force-step` one step | ~4–6 s | Targeted recompute |

## CLI flags

```bash
python -m models.run_all               # incremental (cache enabled by default)
python -m models.run_all --quick       # incremental + skip heavy grids
python -m models.run_all --no-cache    # force full recompute
python -m models.run_all --force-step electrochemistry   # recompute one step
python -m models.run_all --force-step electrochemistry --force-step transport
```

## Make targets

```bash
make run-all          # incremental
make run-all-fresh    # --no-cache
make run-all-quick    # --quick
make run-all-force STEP=electrochemistry
make cache-status     # print the manifest
```

## Safety

- **Content-addressed, not mtime-based.** `touch models/foo.py` does not
  invalidate anything — only actual content changes trigger recomputation.
- **All outputs verified.** A cache hit requires every declared output file to
  exist on disk.  Deleted or partial outputs always produce a miss.
- **Conservative import scanning.** The dependency scanner walks `import` /
  `from ... import` statements statically.  It may over-estimate dependencies
  (causing more invalidation than strictly necessary), but never under-estimates
  (which would silently serve stale results).
- **Escape hatches.** `--no-cache` disables caching entirely; `--force-step`
  forces recompute of specific steps.

## Dependency invalidation examples

| File changed | Steps invalidated | Why |
|---|---|---|
| `carburization.py` | carburization, tempering | tempering imports carburization |
| `kinetics.py` | electrochemistry, carburization, tempering | all import kinetics (transitively) |
| `transport.py` | transport, reference_cell_pipeline | pipeline imports transport |
| `process_flow.py` | process_flow only | no other step imports it |

## CI integration

The GitHub Actions `figures` job uses `actions/cache` to persist the step
cache between runs.  On a PR, the cache is seeded from the most recent `main`
run, so only steps affected by the PR diff are recomputed.

## Implementation

- `models/step_cache.py` — `StepCache` class, dependency scanner, manifest I/O
- `models/run_all.py` — each step wrapped in `cache.step()` context manager
- `tests/test_step_cache.py` — 27 tests covering hits, misses, invalidation,
  forced steps, multi-output verification, and edge cases
