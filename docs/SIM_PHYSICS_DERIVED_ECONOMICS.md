# Physics-derived economics (Level-0 screening)

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

The immutable named configuration is
`physics-economics-reference-divided-cell-v1`: a 200 cm² cathode, 2 cm
interelectrode gap, divided Nafion-like membrane (`3.0e-4 ohm m²` area
resistance), `5.0e-4 ohm m²` contact resistance, and 0.10 anode bubble
fraction. The bath is 1.0 M FeSO4, 0.5 M Na2SO4, 0.01 M H2SO4, and 0.4 M boric
acid at pH 2.0 and 50 °C, with a 50 µm moderate-agitation boundary layer.

The target point is 300 mA/cm² (RC-1 / the program decision point). Screening
thresholds are copied from `PROGRAM_SUMMARY.md`: FE >= 70 %, net DC specific
energy <= 4,000 kWh/t Fe, and transport limit greater than applied current.
The energy identity is reused, not re-derived:
`E = specific_energy_kWh_per_t(V_cell, FE) = 959.9 * V_cell / FE` kWh/t Fe.

## Derived 300 mA/cm² result

| Quantity | Physics-derived value | Screening verdict |
|---|---:|---|
| FE | 98.51 % **unvalidated (L0)** | pass: >= 70 % |
| Cell voltage | 8.298 V **unvalidated (L0)** | diagnostic input to energy |
| Net DC specific energy | 8,085 kWh/t Fe **unvalidated (L0)** | **fail**: > 4,000 kWh/t Fe |
| Transport limit | 370.9 mA/cm² **unvalidated (L0)** | pass: > 300 mA/cm² |
| Deposition rate | 391.0 µm/h **unvalidated (L0)** | reported only |

This is intentionally not re-tuned to force a pass. The model predicts strong
FE and non-binding transport at the decision current, but a cell voltage large
enough that its own net-DC energy criterion fails. That is the useful
screening finding to validate before building; it is not a route kill based on
simulation alone.

## Economics derived from that operating point

At $0.04/kWh and the existing ten-stack TEA defaults, the derived calculation
returns annual capacity **493 t/y unvalidated (L0)** and LCOFe **$4,531/t Fe
unvalidated (L0)**. Holding the same physics-derived voltage but substituting
the old hardcoded FE=0.90 produces **$4,956/t Fe unvalidated (L0)**.

Thus derived minus hardcoded LCOFe is **-$425/t Fe unvalidated (L0)**: the
hardcoded 0.90 FE is pessimistic *for this model's FE branch*. This does not
make the economics favourable—the coupled voltage still drives very high
energy/cost—and it must not be interpreted as a measured advantage.

## Current-density profile

| j (mA/cm²) | FE | V_cell (V) | Energy (kWh/t Fe) | LCOFe ($/t Fe) |
|---:|---:|---:|---:|---:|
| 50 | 99.58 % | 3.543 | 3,415 | 24,405 |
| 100 | 99.54 % | 4.504 | 4,343 | 12,368 |
| 150 | 99.46 % | 5.451 | 5,261 | 8,384 |
| 200 | 99.32 % | 6.396 | 6,182 | 6,432 |
| 250 | 99.06 % | 7.343 | 7,115 | 5,273 |
| 300 | 98.51 % | 8.298 | 8,085 | 4,531 |
| 350 | 96.98 % | 9.275 | 9,180 | 4,058 |
| 400–500 | invalid **unvalidated (L0)** | — | — | not priced |

All displayed values are **unvalidated (L0)**. The runner requests 50–500
mA/cm² through `CellPhysics.sweep`; it preserves unsolved/out-of-transport
points as `invalid` instead of silently generating impossible economics.
Energy increases across the feasible sweep as voltage rises.

## FE/voltage uncertainty and the next measurement

A deterministic four-corner sensitivity applies the `NEXT_STEPS.md` +/-5
percentage-point FE acceptance span and +/-0.10 V cell-voltage span around the
300 mA/cm² prediction. It gives LCOFe **$4,312–$4,774/t Fe unvalidated (L0)**,
around a base of **$4,531/t Fe unvalidated (L0)**. In this local sensitivity,
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
