# Physics-derived economics (Level-0 screening)

> **Regeneration note (2026-08-06):** the numeric tables below predate the
> reactive cathode-film, shared thermodynamic constants, temperature-resolved
> transport, and single-temperature conductivity corrections. Re-run the
> physics-derived economics driver before treating them as the current screen.
> They remain a prior-report record and are not gate evidence.

## Status and boundary

This report is a transparent, bottom-up **Level-0** prediction. Every number
below is **unvalidated (L0)**: it is not a laboratory observation, calibration,
plant forecast, or gate result. In particular, this is **NOT gate evidence**;
gates are measurement-only in `models/process_gates.py`. No real laboratory
data is used.

`models/economics_from_physics.py` closes a previously visible wiring gap. It
calls `CellPhysics.solve_at_j()` for the Fe/HER kinetic-branch FE and coupled
cell voltage, then passes those values to `ElectrolyzerParams`, `CAPEXModel`,
`OPEXModel`, and `LevelizedCost`. It does not use the 0.90 FE assumption for
its derived case. The 0.90 case is retained only as an explicitly labelled
contrast with `run_technoeconomic.py`.

## Explicit reference configuration

The immutable named configuration is `RC-1-reference-divided-cell`, mirroring
the published RC-1 design basis (`docs/REFERENCE_CELL_DESIGN_BASIS.md`): a
**10 cm² cathode** and **3 mm electrode-to-membrane gap**, divided Nafion-like
membrane (`3.0e-4 ohm m²` area resistance), `5.0e-4 ohm m²` contact
resistance, and 0.10 anode bubble fraction. The bath is 1.0 M FeSO4, 0.5 M
Na2SO4, 0.01 M H2SO4, and 0.4 M boric acid at pH 2.0 and 50 °C, with a 50 µm
moderate-agitation boundary layer.

The target point is 300 mA/cm² (RC-1 / the program decision point). Screening
thresholds are copied from `PROGRAM_SUMMARY.md`: FE >= 70 %, net DC specific
energy <= 4,000 kWh/t Fe, and transport limit greater than applied current.
The energy identity is reused, not re-derived:
`E = specific_energy_kWh_per_t(V_cell, FE) = 959.9 * V_cell / FE` kWh/t Fe.

## Derived 300 mA/cm² result

| Quantity | Physics-derived value | Screening verdict |
|---|---:|---|
| FE | 98.51 % **unvalidated (L0)** | pass: >= 70 % |
| Cell voltage | 5.678 V **unvalidated (L0)** | diagnostic input to energy |
| Net DC specific energy | 5,532 kWh/t Fe **unvalidated (L0)** | **fail**: > 4,000 kWh/t Fe |
| Transport limit | 370.9 mA/cm² **unvalidated (L0)** | pass: > 300 mA/cm² |
| Deposition rate | 391.0 µm/h **unvalidated (L0)** | reported only |

This is intentionally not re-tuned to force a pass. The model predicts strong
FE and non-binding transport at the decision current, but a cell voltage large
enough that its own net-DC energy criterion fails. That is the useful
screening finding to validate before building; it is not a route kill based on
simulation alone.

## Economics derived from that operating point

RC-1's 10 cm² bench coupon tests the *physics*; it is not the production-cell
area the CAPEX stack is calibrated for. Feeding the coupon directly into the
plant TEA produces a meaningless $80k/t tiny-plant number, so the cost stack
uses production-scale parameters sized to the program's stated target. The
program frames the deployment question at **100 kt/yr Fe**
(`RESEARCH_PROGRAM.md` Q7, `FEEDSTOCK_SOURCING_MEMO.md`); 40 stacks × 100 cells
of 1 m² (300 mA/cm², FE 98.5%) reaches **~98.5 kt/yr**. At $0.04/kWh that gives
**LCOFe $374/t Fe unvalidated (L0)** and annual capacity
**98,526 t/yr unvalidated (L0)**. Holding the same physics-derived voltage but
substituting the old hardcoded FE=0.90 produces **$402/t Fe unvalidated (L0)**.

Thus derived minus hardcoded LCOFe is **-$28/t Fe unvalidated (L0)**: the
hardcoded 0.90 FE is pessimistic *for this model's FE branch*. This does not
make the economics favourable—the coupled voltage still drives very high
energy/cost (5,532 kWh/t at 300 mA/cm²)—and it must not be interpreted as a
measured advantage.

A note on the absolute number: this L0 economics uses the 5,532 kWh/t derived
energy, which alone implies a high operating cost. The result is a *screening*
signal that, at RC-1's real geometry, the route looks energy/cost-heavy before
any calibration—exactly the "make it known before building" finding, not a
configuration-tuned pass. There is also a strong production-scale message: the
cost-minimizing current density is well below the 300 mA/cm² decision point
(see sweep below), so the economics and the energy gate both argue against
running the aggressive 300 mA/cm² benchmark duty.

## Current-density profile

| j (mA/cm²) | FE | V_cell (V) | Energy (kWh/t Fe) | LCOFe ($/t Fe) |
|---:|---:|---:|---:|---:|
| 50 | 99.58 % | 3.106 | 2,994 | 409 |
| 100 | 99.54 % | 3.630 | 3,501 | 340 |
| 150 | 99.46 % | 4.142 | 3,997 | 333 |
| 200 | 99.32 % | 4.650 | 4,494 | 341 |
| 250 | 99.06 % | 5.160 | 4,999 | 355 |
| 300 | 98.51 % | 5.678 | 5,532 | 374 |
| 350 | 96.98 % | 6.219 | 6,155 | 400 |
| 400–500 | invalid **unvalidated (L0)** | — | — | not priced |

All displayed values are **unvalidated (L0)** and assume the ~98.5 kt/yr
production stack. The runner requests 50–500 mA/cm² through
`CellPhysics.sweep`; it preserves unsolved/out-of-transport points as `invalid`
instead of silently generating impossible economics. Energy increases across
the feasible sweep as voltage rises; transport is exceeded above ~371 mA/cm²,
so 400–500 are surfaced as `invalid`. Two things stand out:

- The energy criterion is crossed between 150 and 200 mA/cm²—well below the
  300 mA/cm² decision point—so the voltage problem is not confined to an
  aggressive operating condition.
- LCOFe is *minimized* near 150 mA/cm² ($333/t), which is also the band where
  energy just barely passes. So at production scale the cost- and energy-
  disciplined operating point is ~150 mA/cm², not 300; the 300 benchmark duty
  is both energy-infeasible and cost-suboptimal.

## FE/voltage uncertainty and the next measurement

A deterministic four-corner sensitivity applies the `NEXT_STEPS.md` +/-5
percentage-point FE acceptance span and +/-0.10 V cell-voltage span around the
300 mA/cm² prediction. It gives LCOFe **$356–$395/t Fe unvalidated (L0)**,
around a base of **$374/t Fe unvalidated (L0)**. In this local sensitivity,
**FE** moves LCOFe more than voltage; therefore the single first measurement to
buy is a weighed-deposit/charge-balance FE measurement at 300 mA/cm², while
logging decomposed cell voltage in the same run.

That ranking is conditional on this uncalibrated model and stated spans; it is
not measurement-derived uncertainty. The experimental program should use the
result to prioritize an instrumented divided-cell run, then upgrade the 1-D
diffusion-layer model only where those measurements show error—not jump to CFD
or phase-field modelling.

## Run

```bash
python -m models.economics_from_physics
```

The report flags output as `unvalidated (L0)`, prints the derived-versus-
hardcoded contrast, uncertainty range, and the measurement recommendation.
