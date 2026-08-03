# Repo Output Policy (shared)

Applies identically to **immortality-workspace** and **aqueous_electrowinning_steel**
(and any future model repo in this family). Both repos had the *same* defect,
so the policy is one contract, not per-repo exceptions.

## The problem this exists to stop

Model scripts emit artifacts (JSON reports, CSVs, figures, caches). Three bugs
recurred:

1. **Quick-vs-full drift** — a "screening / --quick / fast" variant of an
   artifact is committed, but the *default* (publication / full-coupling) run
   that CI and readers actually invoke produces different numbers (aq #37: the
   committed gas_holdup report was the `--quick` build with the coupling result
   missing; immortality #102: outputs committed from an earlier code state).
2. **Stale committed outputs** — an artifact is committed, then the code
   changes, and nothing forces a re-generation, so committed != fresh run.
3. **Treadmill** — "just regenerate everything every time in CI" is wasteful
   (~3 h runs) and was the reason the easy fix kept getting rejected.

## The rule (one sentence)

> **A committed artifact must equal what a fresh, full-grade run of the
> committed code produces — no weaker (quick/screening) variants committed as
> the publication artifact, no stale outputs allowed.**

## Tiering

| Tier | Examples | Committed? | Fresh-regen gate |
|---|---|---|---|
| **Publication/decision** (`pub`/`ci-pub`) | S7.3 power surface, gas_holdup full-coupling report, cost/observability ledgers, RC-1 deployment manifest | YES | MUST equal a fresh full-grade run (CI hash-check) |
| **Screening/quick** (`--quick`, `fast`, smoke MC) | any artifact with "DO NOT CITE" / screening-grade MC reps | NO — gitignored | never committed |
| **Heavy/reproducible cache** | e7 power cells, COMSOL outputs, `raw/` data | gitignored; seed-cache may be CI-committed | restored/assemble, verified at release |
| **Docs/registry** | claim-2 gaps, SIM_THEORY_CONFIDENCE, methodologies | YES | reviewed with the PR |

## Enforcement (cheap, not a treadmill)

- **Write provenance**: each generator stamps `{artifact, recipe, source_hashes,
  mode}` at write time.
- **CI = hash-check, not re-run**: for each tracked pub artifact, verify its
  stamp lists the *current* committed source hashes and the **full-grade mode**.
  Match → clean (no re-run). Mismatch → fail fast on the hash/mode diff.
- **Heavy targets** (e7-power, full sharded grids) are **exempt from per-PR
  re-run**; their provenance is verified at release / by the dedicated workflow
  (e.g. `s73-power`) instead, with a `STALE-regenerate-before-release` tag.
- **The one-liner** (`scripts/arena_setup.sh`) re-establishes the env after a
  workspace reset (committed project file, not a persistent OS package).

## The build-DAG (why we might adopt redo, not a treadmill)

Both repos' runners are **recipe-order graphs** (just/Make `.PHONY`): they know
*ordering* but not *file causality*, so they always re-run and can't detect
stale outputs. The heavy targets (e7-power) are **already decomposed into
content-addressed cells** — they just aren't *orchestrated*. A pure-Python
`redo` committed at `build/redo` adds the **file-causality DAG** (skip what's
fresh, parallelize, assemble) with zero install (sandbox-reset-proof because
it's committed source). Layered *under* the existing recipe graph — it
augments, doesn't replace.

## Per-repo status

- **immortality**: setup one-liner + just/make parity DONE (2026-08-02, main
  `566e4c7`). Next: add provenance stamps + CI hash-check; adopt redo for heavy
  targets (e7-power pilot).
- **aq**: needs the setup one-liner + Make `arena-setup`; adopt the same stamp
  + hash-check; regenerate `gas_holdup_report.json` in full mode at #37 merge.
