# Reference-Cell Theory-Confidence Simulation (Chain of Claims)

**Date:** 2026-08-01
**Status:** Level-0 *screening* — **NOT gate evidence**
**Module:** `models/theory_confidence.py` (+ runner `models/run_theory_confidence.py`, tests `tests/test_theory_confidence.py`)

## What this is

This is a single, honest, bottom-up simulation that checks the *theory* that
the divided-cell electrowinning route can plausibly "solve electrowinning
steel". It builds one explicit, immutable **reference divided-cell design** and,
from the merged physics modules, predicts:

1. a reference operating point (`FE`, `V_cell`, specific energy, deposition
   rate) and a per-target pass/fail verdict;
2. a thermal/energy balance driven by the simulated `V_cell`/current;
3. charge, iron, and energy ledger closure from a synthetic run;
4. a chain-of-claims truth table mapped to `docs/NEXT_STEPS.md` §"The standard
   we should use".

Everything here is a transparent, synthetic, bottom-up prediction. It is
**explicitly NOT gate evidence**: gates are measurement-only
(`models/process_gates.py`), and there is **no real lab data in this
repository**. Every predicted number carries the `unvalidated (L0)` flag until
run data arrives.

## Reference cell (immutable, in `reference_cell()`)

| Parameter | Value |
|---|---|
| Cell | divided (Nafion membrane), benchtop |
| Cathode active area | 200 cm² |
| Interelectrode gap | 2 cm |
| Membrane area resistance | 3.0e-4 Ω·m² |
| Anode bubble fraction | 0.10 |
| Bath | 1.0 M FeSO₄, 0.5 M Na₂SO₄, 0.4 M H₃BO₃, pH 2.0 |
| Temperature / agitation | 50 °C / moderate (50 µm boundary layer) |
| Electrolyte volume | 3.0 L |
| Jacket cooling | active, 15 °C coolant |

The reference current density is **not hardcoded**; it is chosen by
`CellPhysics.find_optimal_j(min_FE=0.70)` (max j with FE ≥ 0.70 and no
Fe(OH)₂ precipitation).

## Screening acceptance targets (module constants)

| Target | Value | Provenance |
|---|---|---|
| FE | ≥ 0.80 | screening floor (aqueous Fe EW 0.80–0.95 with suppressed HER) |
| V_cell | 2.5–6.0 V | practical divided-cell DC-stack window |
| Specific energy | ≤ 6000 kWh/t Fe | route threshold (E = 959.9·V/FE) |
| Transport margin | limit/j ≥ 1.2 | transport must not bind at reference |
| Deposition rate | 20–300 µm/hr | harvestable flake/deposit window |
| Thermal limit | ≤ 60 °C | electrolyte/membrane stability ceiling |
| Charge residual | ≤ 2 % of applied charge | ledger closure (L0) |
| Iron residual | ≤ 5 % of initial Fe | NEXT_STEPS iron balance closure ±5 % |

## Reference operating point (predicted, `unvalidated (L0)`)

| Quantity | Predicted | Acceptance | Verdict |
|---|---|---|---|
| Current density | 96.7 mA/cm² (19.3 A) | — | — |
| FE | 0.995 (HER ≈ 0.5 %) | ≥ 0.80 | **PASS** |
| V_cell | 4.44 V | 2.5–6.0 V | **PASS** |
| Specific energy | 4281 kWh/t Fe | ≤ 6000 kWh/t | **PASS** |
| Transport limit | 371 mA/cm² (margin 3.8×) | ≥ 1.2 | **PASS** |
| Deposition rate | 127 µm/hr | 20–300 µm/hr | **PASS** |
| Surface pH | 1.98; Fe(OH)₂ ss ≈ 2e-8 | no precipitation | PASS |

Note the high predicted FE (~0.995). This is an artifact of the default
suppressed-HER kinetics in the merged `cell_physics` module — it is exactly
what the screening model returns, reported here without adjustment, and it is
precisely the sort of claim a Level-1 calibration would have to confirm or
correct with run data.

## Thermal balance (predicted, L0)

| Quantity | Value |
|---|---|
| Heat generation | 61.1 W |
| Joule (IR) heat | 34.2 W (dominates) |
| Activation heat | 18.4 W |
| Steady-state T (cooled) | 21.2 °C |
| Steady-state T (uncooled) | 33.3 °C |
| Cooling duty to hold 50 °C | 0 W |
| Verdict | **PASS** — steady-state T ≤ 60 °C with active cooling |

At this reference scale the cell is thermally mild; passive heat rejection
already keeps it under the limit and the active jacket adds margin. This is
reported honestly (both cooled and uncooled steady states shown).

## Ledger closure (complete screening fixtures, L0)

| Ledger | Status | Residual | Tolerance | Verdict |
|---|---|---|---|---|
| Charge | `partial_with_fe_deposit` | 635 C (0.5 %) | ≤ 2 % | **PASS** |
| Iron | `closed` | 0.000 mol (0.0 %) | ≤ 5 % | **PASS** |
| Energy | `closed` | missing = [] | none missing | **PASS** |

The charge ledger is **Fe-specific** (independent deposit composition fixture),
not mass-only-apparent. All fixtures are explicit screening assumptions
documented in the module; the iron fixture is an idealized closing run (no
unmeasured crossover/precipitate). A negative test confirms the machinery is
honest: if a stream (energy log, or the post-run bath analysis) is omitted,
the corresponding ledger reports `partial` with a non-empty `missing` list —
it is never silently zero-filled.

## Chain of claims (from `docs/NEXT_STEPS.md` §standard)

| # | Claim | Substantiated by | Predicted (L0) | Acceptance | Verdict |
|---|---|---|---|---|---|
| 1 | feed/electrolyte are what we think they are | reference recipe + speciation | free [Fe²⁺]=0.018 M, σ=13.5 S/m, pH 2.0 @50 °C | recipe reproduces intended bath | **PASS** (L0; real feed identity is L1) |
| 2 | cell produces predicted fields | `CellPhysics` + thermal transient | V_cell=4.44 V, T_ss=21 °C, transport limit=371 mA/cm² | V in window, T ≤ 60 °C | **PASS** (L0; gas/flow partially modeled) |
| 3 | electrochemistry produces Fe/HER split & rate | `CellPhysics.solve_at_j` | FE=0.995, HER≈0.5 %, 127 µm/hr | FE≥0.80, rate in window | **PASS** (L0) |
| 4 | deposit harvestable, predicted composition/quality | deposit rate + pure-Fe composition fixture | 100 wt% Fe fixture, 127 µm/hr | composition + rate | **PARTIAL** (L0; harvestability deferred to peel-coupon branch) |
| 5 | quantities hold over time (membrane ageing, anode wear) | — (no run day-1+ data) | — | accelerated-life data | **NOT COVERED / deferred** (Level 3) |
| 6 | balance of plant closes on mass/charge/heat/energy | `compute_ledgers` | residual 0.5% / 0.0% / none missing | ≤2% / ≤5% / none | **PASS** (L0) |

Claim 5 is **explicitly NOT COVERED / deferred** — the repository has no
run day-1+ durability data, so membrane ageing, anode wear, and impurity-driven
drift cannot be substantiated at any level here.

## Robustness summary (bonus, Level-0)

A coarse T × [Fe] sweep (40/50/60 °C × 0.75/1.0/1.25 M FeSO₄, reference
geometry) predicts a **usable window of 9/9 combinations**: the reference
operating point lands inside all screening targets across that grid, with
`j*` 96.7–140 mA/cm², FE 0.993–0.996, V_cell 4.4–4.9 V, and specific energy
4265–4750 kWh/t. This is a Level-0, model-internal robustness check — it says
nothing about real feed variability or unmodeled impurity chemistry.

## What this does NOT claim

- This is **not a calibrated digital twin** and **not gate evidence**. Gates
  (`models/process_gates.py`) are measurement-only.
- No real wet-lab data is used; no uncertainty from measurement is carried
  (that is Level 1, out of scope).
- Claim 5 (durability/membrane-ageing/anode-wear) is deferred.
- Deposit harvestability/adhesion (claim 4) is deferred to the peel-coupon
  branch.
- Crate/site, operating-twin safe-state, DFT, phase-field, CFD, and PID tuning
  are explicitly out of scope (see `docs/NEXT_STEPS.md`).

The predicted high FE and clean closures are *plausibility* results from
transparent assumptions — they become engineering evidence only when
instrumented runs close the same three ledgers from measurements.

## Files

- `models/theory_confidence.py` — the simulation + `main()` report
- `models/run_theory_confidence.py` — CLI runner (`python -m models.run_theory_confidence`)
- `tests/test_theory_confidence.py` — locks the screening claims
- `docs/SIM_THEORY_CONFIDENCE.md` — this report
