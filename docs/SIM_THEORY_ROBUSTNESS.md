# Operating-Window Theory-Confidence Screen

**Date:** 2026-08-01  
**Status:** **unvalidated (L0)** screening — **NOT gate evidence**  
**Module:** `models/operating_window_confidence.py`  
**Run:** `python -m models.operating_window_confidence`

## Scope and evidence status

This is a transparent, synthetic, bottom-up extension of the single-reference
simulation in `models/theory_confidence.py`. It predicts whether the reference
divided-cell route remains inside the same Level-0 acceptance screen while bath
temperature, Fe(II) concentration, and applied current density change.

It uses no real laboratory data. It is **not gate evidence**: process gates are
measurement-only in `models/process_gates.py`. The screen does not calibrate a
digital twin, establish measurement uncertainty, demonstrate deposit quality or
harvestability, or address durability, membrane ageing, anode ageing, or site
operation.

## Swept screen

The bounded 3 × 3 × 5 grid contains 45 physics solves:

| Variable | Grid |
|---|---|
| Bath temperature | 40, 50, 60 °C |
| FeSO₄ / Fe(II) concentration | 0.75, 1.00, 1.25 M |
| Applied current density | 30.0, 96.7, 140, 180, 240 mA/cm² |

The 96.7 mA/cm² entry is the #32 `solve_reference()` operating current. At
each point, the module builds a fresh `CellPhysics` instance and applies the
existing `ScreeningTargets` bundle—there is no duplicate target definition:
FE ≥ 0.80; 2.5 ≤ `V_cell` ≤ 6.0 V; specific energy ≤ 6000 kWh/t; transport
limit / applied current ≥ 1.2; and 20 ≤ deposition rate ≤ 300 µm/hr.

At each of the four extreme temperature × Fe(II) bath corners (40/60 °C ×
0.75/1.25 M), the active-cooling thermal balance is also run on the sampled
current slice nearest the #32 reference current (96.7 mA/cm²). Its target is
cooled steady-state temperature ≤ 60 °C.

## Predicted operating-window result

**31 / 45 points are usable: 68.9%.** A usable point passes every applicable
screening target simultaneously. The remaining points are retained as failures,
not silently omitted. This clears the stated non-trivial screening expectation
of more than one third of the grid, but is still only a model-internal L0
prediction.

### Usable-region headroom

Margins are dimensionless. Positive values mean headroom; for two-sided
voltage and deposition-rate criteria, the smaller margin to either bound is
reported. “Closest” is the minimum margin among usable points.

| Target | Median margin | Closest margin | Interpretation |
|---|---:|---:|---|
| FE floor | +0.244 | +0.238 | above FE ≥ 0.80 |
| Cell-voltage window | +0.197 | +0.002 | inside 2.5–6.0 V |
| Specific-energy ceiling | +0.226 | +0.037 | below 6000 kWh/t |
| Transport margin | +1.845 | +0.239 | above limit/j ≥ 1.2 |
| Deposition-rate window | +0.575 | +0.210 | inside 20–300 µm/hr |
| Cooled thermal limit | +0.644 | +0.634 | below 60 °C at checked corners |

In particular, the maximum specific energy over usable grid points remains at
or below the **6000 kWh/t** route ceiling. All cooled corner predictions remain
at or below the **60 °C** thermal limit.

## Reference-point interior claim

The #32 reference point is predicted to be **strictly interior** to every
screening bound, not merely on a pass/fail edge. Its margins are:

| Target | Reference margin |
|---|---:|
| FE | +0.244 |
| Cell voltage | +0.260 |
| Specific energy | +0.286 |
| Transport limit | +2.197 |
| Deposition rate | +0.576 |
| Cooled thermal temperature | +0.646 |

This supports the narrow statement that, **within this L0 model and sampled
grid**, the reference point is not knife-edge. It does not prove robustness in
a physical cell.

## Sampled first-trip boundary

`window_boundary()` finds, for each target, the nearest sampled failing point
relative to the #32 reference and reports the axis that contributes the largest
normalized departure. This is a sampled boundary, not an interpolated process
control limit.

| Target | First sampled trip |
|---|---|
| FE | none in this grid |
| Cell voltage | `j_mA_cm2 = 180` |
| Specific energy | temperature axis, `T_C = 40` (at its closest failing combination) |
| Transport margin | `j_mA_cm2 = 240` |
| Deposition rate | `j_mA_cm2 = 240` |
| Thermal | none in checked corner samples |

The deliberately off-design 0.25 M Fe(II), 300 mA/cm² check fails at least one
numerical target. That negative check prevents interpreting the machinery as
an “all pass everywhere” classifier.

## Reproduction and tests

```bash
python -m models.operating_window_confidence
pytest tests/test_operating_window_confidence.py
```

The full surface test is marked `slow`, because it runs the Nernst–Planck
solver at every grid point. The test locks numerical claims against the shared
targets: usable fraction > 1/3, strict reference interior margins, the
specific-energy ceiling over usable points, the four bath-corner thermal limit,
and an off-design failure.

## Explicitly not claimed

- No real-data calibration or Level-1 measurement uncertainty.
- No measured deposit morphology, composition, harvestability, or durability.
- No membrane/anode ageing, impurity chemistry, CFD, PID, site/crate layer, or
  operating-twin safe-state claim.
- No replacement for measurement-only process gates.
