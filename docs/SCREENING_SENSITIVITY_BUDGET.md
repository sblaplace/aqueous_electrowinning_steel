# Reference-cell screening-uncertainty / sensitivity budget

**Status: Level-0 screening, unvalidated, and NOT gate evidence.** This report
uses transparent synthetic predictions only. Gates are measurement-only in
`models/process_gates.py`; the repository contains no real laboratory data.
The purpose is narrower: identify which remaining model inputs can change the
reference-cell go/no-go and therefore what the first real run should measure.

This is the additive uncertainty-budget deliverable after the single-point
screen in `docs/SIM_THEORY_CONFIDENCE.md` (#32) and the operating-window screen
in `docs/SIM_THEORY_ROBUSTNESS.md` (#33). It reuses
`models/theory_confidence.solve_reference()` and
`models.operating_window_confidence.sweep_window()`/`margins()`; it does not
modify either module.

## Method and interpretation

`models/screening_sensitivity.py` performs a deterministic one-at-a-time (OAT)
screen. For each coefficient it:

1. takes the reference-cell dataclass value;
2. multiplies that one value by the low and high factor below;
3. rebuilds the cell with `dataclasses.replace` and calls `solve_reference()`;
4. records FE, `V_cell`, specific energy, target verdicts, and pass/fail; and
5. sends the more decision-threatening endpoint through #33's window solver.

The reported `±` values are the largest absolute displacement from the central
reference result across the two endpoints, not a standard deviation or a
confidence interval. `window min` is the smallest positive #33 dimensionless
margin among usable points on the bounded reference-bath current slice
`j = 0.75, 1.00, 1.25 × j_reference`; `0.000` means no usable row remained at
the selected endpoint. The slice is intentionally bounded to keep the
Nernst--Planck solve CI-tractable; it is not a claim that the full #33 surface
has been exhaustively recomputed for every coefficient.

The priority score used for deterministic ordering is explicitly:

```text
100 × flip
+ |ΔFE| / 0.80
+ |ΔV_cell| / 3.50 V
+ |Δspecific energy| / 6000 kWh/t
+ 1 / (1 + window min)
```

A verdict flip receives the dominant term because a parameter that crosses a
screening acceptance bound is a more urgent go/no-go uncertainty than one that
only moves a comfortably passing value. Remaining ties are broken by the
numeric influence terms and then parameter name. This is a prioritization
metric, not a probability of failure.

## Central reference result

The unperturbed #32 reference cell is the 200 cm², divided/Nafion, 3 L cell at
50 °C and 1.0 M FeSO₄. Its central operating point is:

| Quantity | Central prediction | Screening target | Verdict |
|---|---:|---:|---|
| Current density | 96.7 mA/cm² | selected by #32 `find_optimal_j` | — |
| FE | 0.9954 | ≥ 0.80 | **PASS** |
| `V_cell` | 4.440 V | 2.5–6.0 V | **PASS** |
| Specific energy | 4,281 kWh/t Fe | ≤ 6,000 kWh/t | **PASS** |
| Transport margin | 3.84× | ≥ 1.20× | **PASS** |
| Deposition rate | 127.3 µm/hr | 20–300 µm/hr | **PASS** |

These are model outputs, not measurements. In particular, the central pass does
not validate any coefficient or establish gate evidence.

## Coefficient ranges and provenance

`range` in the implementation is a multiplicative factor `(low, high)` around
the central `value`; actual endpoint values are shown here for readability.
The intervals are deliberately labeled **screening assumptions**. They are
anchored to defaults, parameter documentation, or existing synthetic test/grid
values in this repository—not to a claim that the repository has measured
uncertainty for any coefficient.

| Parameter | Maps into | Central value | Factor range | Actual endpoint range | Provenance used by the screen |
|---|---|---:|---:|---:|---|
| `fe_i0` | `ProcessConditions` → `NernstPlanckFilm.fe_i0` | 1.0e-2 A/m² | 0.10–10 | 1.0e-3–0.10 A/m² | `models/cell_physics.py:ProcessConditions.fe_i0` default |
| `her_i0` | `ProcessConditions` → `NernstPlanckFilm.her_i0` | 1.0e-6 A/m² | 0.10–10 | 1.0e-7–1.0e-5 A/m² | `models/cell_physics.py:ProcessConditions.her_i0` default |
| `fe_tafel_V` | Fe cathode Tafel input | 0.120 V/decade | 0.75–1.25 | 0.090–0.150 V/decade | `models/cell_physics.py:ProcessConditions.fe_tafel_V` default; no fit claimed |
| `her_tafel_V` | HER cathode Tafel input | 0.140 V/decade | 0.75–1.25 | 0.105–0.175 V/decade | `models/cell_physics.py:ProcessConditions.her_tafel_V` default; no fit claimed |
| `boundary_layer_m` | `NernstPlanckFilm.boundary_layer_m` | 50 µm | 0.40–2 | 20–100 µm | `models/cell_physics.py`; same order as repo still/stirred transport tests |
| `c_FeSO4_M` | Fe transport and speciation input | 1.0 M | 0.50–1.50 | 0.50–1.50 M | `models/theory_confidence.py:reference_cell`; #33 Fe grid is a synthetic screen |
| `c_Na2SO4_M` | `NernstPlanckFilm.support_conc_M` / migration | 0.50 M | 0.50–2 | 0.25–1.00 M | `models/cell_physics.py` and `models/transport.py` constructor defaults/field |
| `membrane_area_resistance_ohm_m2` | `MembraneModel.R_membrane_ohm_m2` | 3.0e-4 Ω m² | 0.50–2 | 1.5e-4–6.0e-4 Ω m² | `models/theory_confidence.py` reference geometry and `models/electrochemistry.py` membrane term |
| `interelectrode_gap_m` | electrolyte IR drop | 0.020 m | 0.50–2 | 0.010–0.040 m | `models/theory_confidence.py` reference geometry; `CellVoltageModel.IR_electrolyte` |
| `contact_resistance_ohm_m2` | contact/busbar IR drop | 5.0e-4 Ω m² | 0.50–2 | 2.5e-4–1.0e-3 Ω m² | `models/theory_confidence.py` reference geometry; `CellVoltageModel.IR_contacts` |
| `anode_bubble_fraction` | effective electrolyte conductivity | 0.10 | 0.50–2 | 0.05–0.20 | `models/theory_confidence.py` reference geometry; `CellVoltageModel.IR_electrolyte` |
| `temperature_C` | speciation, transport and `conductivity_S_m(T)` | 50 °C | 0.80–1.20 | 40–60 °C | `models/theory_confidence.py` reference; #33 40–60 °C grid; `models/electrochemistry.py` conductivity model |

The fixed electrochemical constants `FARADAY`, `M_FE_G`, and `Z_FE` from
`models/electrochemistry.py` are not perturbed.

## Influence results

The central pass margins are FE headroom `0.9954 - 0.80 = 0.1954` and upper
voltage headroom `6.0 - 4.440 = 1.560 V`. The following OAT results are from
`python -m models.screening_sensitivity`; `±` is defined above.

| Parameter | ±FE | ±`V_cell` (V) | ±specific energy (kWh/t) | Flips a reference verdict? | #33 window min |
|---|---:|---:|---:|:---:|---:|
| `fe_i0` | 0.0252 | 0.120 | 228 | no | +0.164 |
| `her_i0` | 0.0351 | 0.00266 | 154 | no | +0.183 |
| `fe_tafel_V` | 0.0427 | 0.153 | 342 | no | +0.149 |
| `her_tafel_V` | **0.2355** | **1.734** | 863 | **YES** | **0.000** |
| `boundary_layer_m` | 0.00269 | 0.0274 | 38.0 | no | +0.177 |
| `c_FeSO4_M` | 0.00492 | 0.0605 | 79.9 | no | +0.168 |
| `c_Na2SO4_M` | 0.00228 | 0.914 | 894 | no | +0.108 |
| `membrane_area_resistance_ohm_m2` | 0 | 0.290 | 280 | no | +0.123 |
| `interelectrode_gap_m` | 0 | 0.993 | 958 | no | +0.0945 |
| `contact_resistance_ohm_m2` | 0 | 0.483 | 466 | no | +0.0828 |
| `anode_bubble_fraction` | 0 | 0.124 | 120 | no | +0.158 |
| `temperature_C` | 0.000354 | 0.445 | 430 | no | +0.0913 |

This is intentionally not a claim that low- or high-side effects are
symmetric. For example, `her_tafel_V` at the low endpoint moves the selected
reference current to 10.0 mA/cm² and gives FE 0.7599, while the high endpoint
gives FE 0.9998. The low endpoint therefore fails the FE floor and the
20 µm/hr deposition-rate floor (10.1 µm/hr); `V_cell` remains within its
voltage window at 2.706 V. The result is a concrete non-vacuous uncertainty,
not an assertion that all parameters are insensitive.

## Dominant remaining unknown

**The HER Tafel slope (`her_tafel_V`) is the closest single coefficient to
flipping the reference verdict and is the dominant remaining L0 unknown in this
budget.** Its stated low screening factor, 0.75 (0.105 V/decade), is already a
failed reference result: FE falls to 0.7599 versus the 0.80 floor. It also
reduces the selected deposition rate to 10.1 µm/hr versus the 20 µm/hr floor.
The central result still passes, so this is a sensitivity finding—not a claim
that the real cell fails.

The same coefficient has no usable row on the bounded perturbed current slice
at that threatening endpoint (`window min = 0.000` in the implementation's
conservative convention). This makes it the first unknown to resolve before
interpreting the other voltage-only sensitivities as the deciding risk.

## Ranked "calibrate this first" list

The deterministic priority list from the current screen is:

1. **`her_tafel_V` — HER Tafel/polarization measurement on the actual cathode,
   at the reference bath and temperature.** Fit the local HER branch rather
   than importing the 0.140 V/decade default.
2. **`interelectrode_gap_m` — physical gap measurement plus a voltage
   breakdown/ohmic check.** The gap has the largest non-flipping `V_cell` and
   energy displacement in this range.
3. **`c_Na2SO4_M` — conductivity and limiting-current/migration experiment.**
   This is the most influential migration/support input for the tested screen.
4. **`contact_resistance_ohm_m2` — four-wire/contact resistance or an EIS/IR
   voltage breakdown under load.**
5. **`temperature_C` — in-cell temperature log paired with conductivity,
   rather than relying on the bath-setpoint value.**
6. **`fe_tafel_V` — Fe deposition polarization/Tafel measurement.**
7. **`membrane_area_resistance_ohm_m2` — membrane area-resistance measurement in
   the actual sulfate bath and temperature.**
8. **`fe_i0` — Fe exchange-current measurement on the target cathode surface.**
9. **`anode_bubble_fraction` — gas hold-up/void-fraction observation paired
   with IR drop.**
10. **`her_i0` — repeat HER exchange-current measurement after controlled
    cathode preparation.**
11. **`c_FeSO4_M` — start/end Fe(II) titration or ICP check.**
12. **`boundary_layer_m` — hydrodynamic/limiting-current experiment to infer
    the effective film thickness.**

This ordering says where one first run buys the most decision information under
these *stated* L0 ranges. It does not replace a DOE, calibration fit, or
measurement plan, and it does not establish deposit morphology, durability,
membrane ageing, or any other explicitly deferred claim.

## Limitations and handoff to a real run

- The OAT factor intervals are not posterior distributions and have no
  statistical confidence attached to them.
- The model is Level-0 and synthetic. It omits real surface-state drift,
  impurity effects, morphology, ageing and measurement error.
- The operating-window check is intentionally a small #33-compatible slice,
  not a full global uncertainty propagation.
- A real first run should record the HER and Fe polarization branches, current,
  voltage breakdown, temperature, Fe(II), conductivity/support salt, membrane
  resistance and hydrodynamic limiting-current surrogate before treating any
  central or perturbed result as evidence.
- Until those data exist, all values remain **unvalidated (L0)** and **NOT gate evidence**.
  Gates remain measurement-only in `models/process_gates.py`.
