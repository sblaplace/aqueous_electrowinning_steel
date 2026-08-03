# Coupled operating point: does the #40 reachability verdict survive the gas correction?

## Scope and status

This report is a transparent, bottom-up **Level-0 → Level-1 boundary
prediction** for the RC-1 reference divided cell. Every number below is
**`unvalidated (L0)`**. There is no laboratory data anywhere in this
calculation, and no value here is a measured result.

This is **NOT gate evidence.** The FE and energy gates are measurement-only and
are implemented in `models/process_gates.py`. Nothing in this document may be
cited as gate satisfaction.

**What this is instead:** a *fidelity-raising* step. It couples the two
reduced-order 1-D models the repository already has — `models/cell_physics.py`
(speciation → transport → voltage) and `models/gas_holdup.py` (Faradaic H₂,
axial drift-flux void fraction, Bruggeman conductivity, current
redistribution) — into a single voltage/energy solve, and re-derives #40's
energy-gate reachability verdict with the gas penalty included.

**What this deliberately is not:** CFD. `docs/SIM_THEORY_CONFIDENCE.md` states
that "a calibrated compartment/strip model is preferable to an unvalidated full
CFD model", and `docs/NEXT_STEPS.md` explicitly defers unvalidated full CFD,
FEM, phase-field and DFT. Coupling the existing two 1-D models is the
disciplined, calibration-friendly step those docs call for, and it is the only
thing done here.

## The gap this closes

#40 (`docs/SIM_OPTIMIZED_OPERATING_POINT.md`) reported:

> "Energy gate **IS reachable**: minimum energy 3,306 kWh/t at j = 150 mA/cm²,
> gap = 1.5 mm, contact = 1.0e-4 Ω·m²."

That verdict comes from the **uncoupled** `cell_physics` voltage solve, which
feeds a fixed `anode_bubble_fraction = 0.10` into `CellVoltageModel`
(`cell_physics.py:229`) and otherwise treats the electrolyte as gas-free and
uniform. It ignores the axial cathodic void profile ε(y), the Bruggeman penalty
κ_eff = κ(1 − ε)^1.5, and the current redistribution — all three of which
`gas_holdup.py` already models in detail but which never fed back into the cell
voltage. A reachability claim backed by a model that structurally omits the
term another module in the same repo was written to capture is a credibility
gap, and it is the sort of gap that should be closed in simulation *before* the
physical build.

## Method

`models/coupled_cell_physics.py` is additive: it imports and composes the
upstream modules and modifies none of them.

1. `coupled_reference_cell()` builds one cell seen by both physics — the #38
   `economics_from_physics.reference_cell()` (RC-1 geometry, acidic sulfate
   bath, 50 °C) plus a `gas_holdup.ChannelGeometry` built from the *same* RC-1
   channel dimensions (10 cm² electrode = 50 mm × 20 mm, 3 mm deep channel,
   `docs/REFERENCE_CELL_DESIGN_BASIS.md`). The interelectrode gap is a single
   shared number, so changing it moves the ohmic path in the voltage solve and
   the gassy resistance in the channel solve together.
2. `CellPhysics.solve_at_j(j)` gives the uncoupled `V_cell`, the **derived** FE,
   the electrolyte conductivity κ and the migration-enhanced transport limit.
3. `gas_holdup.solve_coupled(...)` — the repo's own fixed point over
   `holdup_profile` → `solve_current_distribution` → FE — is run on the same
   channel, with κ from step 2 and with its `fe_model` hook wired to
   `CellPhysics`. FE therefore stays a *derived output of the same engine* that
   produced the uncoupled baseline; the Faradaic gas fraction is `1 − FE`, never
   an injected value.
4. The gas ohmic burden `CoupledGasResult.ohmic_penalty_V` (the area-average of
   `j·L/κ_eff` on the redistributed current minus the gas-free value) is added
   to `V_cell`, and energy is re-evaluated with the program identity
   `E = 959.9 · V_cell / FE` (`electrochemistry.specific_energy_kWh_per_t`).
5. The #38/#40 transport-limit rule is enforced unchanged: points at or beyond
   the migration-enhanced limit are marked invalid and never priced.

**Two coupled energies are reported, and the conservative one is the headline.**
The headline `coupled_specific_energy_kWh_t` applies the gas *ohmic* penalty at
the **uncoupled** FE, so the coupling can only ever raise voltage and energy —
it is structurally incapable of flattering the gate. The full two-way number
`coupled_energy_with_FE_shift_kWh_t` additionally applies the model's own
`FE_shift` (bubble microconvection thins the diffusion layer, which in this
model helps FE slightly); it is reported for contrast and never used to declare
the gate reached.

Numerics are inherited from `run_gas_holdup.main()`'s call pattern (4 axial
segments, ≤ 6 fixed-point iterations) with `solve_coupled`'s own 2e-3
convergence tolerance on both the FE and current vectors, unchanged. Nothing
was loosened to make the coupled solve cheaper; the full joint scan takes ~3
minutes.

## Result — uncoupled vs coupled, over #40's joint space

Same levers as #40: j ∈ {150, 300} mA/cm², gap ∈ {1.5, 3.0} mm, membrane
3.0e-4 Ω·m², contact ∈ {1.0e-4, 5.0e-4} Ω·m². All values `unvalidated (L0)`.

| j (mA/cm²) | gap (mm) | contact (Ω·m²) | V_unc (V) | V_cpl (V) | gas ΔV (mV) | E_unc (kWh/t) | E_cpl (kWh/t) | ΔE (kWh/t) | E ≤ 4,000? |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--|
| 150 | 1.5 | 1.0e-4 | 3.4259 | 3.4260 | 0.025 | 3306.31 | **3306.34** | +0.024 | **Yes** |
| 150 | 1.5 | 5.0e-4 | 4.0259 | 4.0260 | 0.025 | 3885.36 | 3885.39 | +0.024 | Yes |
| 150 | 3.0 | 1.0e-4 | 3.5415 | 3.5416 | 0.050 | 3417.85 | 3417.90 | +0.048 | Yes |
| 150 | 3.0 | 5.0e-4 | 4.1415 | 4.1416 | 0.050 | 3996.90 | 3996.95 | +0.048 | Yes |
| 300 | 1.5 | 1.0e-4 | 4.2467 | 4.2470 | 0.245 | 4137.85 | 4138.09 | +0.238 | No |
| 300 | 1.5 | 5.0e-4 | 5.4467 | 5.4470 | 0.245 | 5307.08 | 5307.32 | +0.238 | No |
| 300 | 3.0 | 1.0e-4 | 4.4779 | 4.4784 | 0.489 | 4363.07 | 4363.55 | +0.477 | No |
| 300 | 3.0 | 5.0e-4 | 5.6779 | 5.6784 | 0.489 | 5532.30 | 5532.78 | +0.477 | No |

FE is 99.458 % (uncoupled) / 99.461 % (coupled) at 150 mA/cm² and 98.511 % /
98.683 % at 300 mA/cm² — all `unvalidated (L0)`, all ≥ the 70 % floor, all
derived from `CellPhysics`.

### Headline verdict

| Quantity | Value |
|---|---|
| `uncoupled_min_energy` | **3,306.31 kWh/t Fe** `unvalidated (L0)` |
| `coupled_min_energy` | **3,306.34 kWh/t Fe** `unvalidated (L0)` |
| `energy_delta` (coupled − uncoupled) | **+0.024 kWh/t (+0.0007 %)** |
| `reachable_uncoupled` (≤ 4,000 kWh/t) | **True** |
| `reachable_coupled` (≤ 4,000 kWh/t) | **True** |
| Coupled best combination | j = 150 mA/cm², gap = 1.5 mm, contact = 1.0e-4 Ω·m² |

**Plain language: yes — the #40 reachable operating point survives the coupled
gas correction, and by a very wide margin.** Coupling the axial void fraction,
the Bruggeman conductivity penalty and the current redistribution into the
voltage moves the minimum specific energy by **+0.024 kWh/t Fe out of
3,306 kWh/t — seven parts in a million** — leaving 694 kWh/t of headroom below
the 4,000 kWh/t gate. The gas term is not what decides this verdict. (This is a
Level-0 prediction, not gate evidence.)

The reverse also holds and is worth stating: coupling does **not** rescue the
300 mA/cm² kill-criterion duty. Those points fail the energy gate before the gas
correction (4,138–5,533 kWh/t) and fail it by marginally more after. The binding
term there remains contact and membrane ohmic resistance, exactly as #39 found.

## Gas impact, isolated (at 150 mA/cm², 1.5 mm gap, 1.0e-4 Ω·m²)

| Quantity | Value `unvalidated (L0)` |
|---|---|
| Mean void fraction | 0.0100 % |
| Outlet void fraction | 0.0176 % |
| Bruggeman conductivity penalty | 1.000151× (κ 13.545 → 13.543 S/m) |
| Current uniformity, min j / max j | 0.99998 |
| Bubble departure diameter | 152 µm |
| Gas ohmic penalty | **0.025 mV** (0.0007 % of V_cell) |
| FE shift from bubble coupling | **+0.0030 pp** (favourable in this model) |
| Energy contribution | **+0.024 kWh/t** (+0.0007 %) |
| Wet H₂ at the vent | 0.0046 L/h |

**Why the effect is so small: RC-1 is 50 mm tall and runs at 99.5 % FE.** Gas
hold-up scales with the gas generated below a height and with the channel
height itself, and the RC-1 channel produces almost no hydrogen — `1 − FE` is
about half a percent of the current. The physics is fully present in the
coupling; it simply has almost nothing to act on at bench scale. This
reproduces, from a different direction, `run_gas_holdup.py`'s own finding that
"gas hold-up does not threaten the RC-1 decision run".

### The same coupling at scale-up heights (150 mA/cm², 1.5 mm gap, 1e-4 Ω·m²)

| Electrode height | Mean ε | Outlet ε | κ penalty | Uniformity | Gas ΔV | Coupled E | ΔE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 mm (RC-1) | 0.010 % | 0.018 % | 1.00015× | 1.0000 | 0.03 mV | 3306.3 | +0.02 |
| 300 mm | 0.060 % | 0.105 % | 1.0009× | 0.9999 | 0.15 mV | 3306.5 | +0.14 |
| 1.0 m | 0.200 % | 0.350 % | 1.0030× | 0.9996 | 0.50 mV | 3306.8 | +0.48 |
| 2.0 m | 0.399 % | 0.697 % | 1.0060× | 0.9992 | 1.00 mV | 3307.3 | +0.97 |

At the 300 mA/cm² kill duty the same sweep reaches 1.9 % mean void, a 1.029×
Bruggeman penalty and 9.8 mV on a 2 m electrode — still under 10 kWh/t. Gas
hold-up is a **scale-up geometry question**, not an energy-gate question, in
this model. Its energy signature stays sub-percent even at plant-height
electrodes; where it does bite is current uniformity, which is a deposit-quality
and dendrite risk, already screened by `gas_holdup.height_scaling_screen`.

## Closure and faithfulness of the coupling

Two numeric checks, both asserted in `tests/test_coupled_cell_physics.py`:

- **Voltage ledger.** `|V_uncoupled + ohmic_penalty − V_coupled| = 0.0e+00 V`
  (tolerance 1e-12 V). The gas term is added once and nothing else is touched;
  the decomposition closes exactly.
- **Null-gas closure.** Driving the Faradaic gas fraction to zero (a
  perfect-FE stub) gives void fraction 0, conductivity penalty 1.000000,
  `ohmic_penalty_V` = 0 V and a coupled energy identical to `cell_physics`'s
  uncoupled value within 1e-2 kWh/t. The coupling layer is faithful, not
  double-counting an existing term.

The tests additionally assert, by identity check, that the coupling calls
`gas_holdup`'s own `bruggeman_conductivity`, `solve_current_distribution`,
`holdup_profile`, `drift_flux_void_fraction` and `solve_coupled`/
`CoupledGasResult` rather than reimplementing them — a drift guard, so the two
models cannot silently diverge.

## What this does and does not license

**Does:** the #40 reachability claim is no longer backed by a model that ignores
the gas term. It has been re-derived with the axial void profile, the Bruggeman
penalty and the current redistribution coupled in, and it holds with 694 kWh/t
of margin. The finding is derived, not tuned: no tolerance was relaxed, no
penalty was shrunk, and the headline coupled energy is computed in the direction
that can only hurt the gate.

**Does not:** this remains a Level-0 prediction with **zero** measurements
behind it. Two things dominate the remaining uncertainty, and coupling did not
touch either:

1. **The inputs that actually set the energy** — contact resistance (#39's
   largest ohmic term, and the difference between 3,306 and 3,885 kWh/t in the
   table above), anodic overpotential, and the FE the cell really delivers. The
   coupled result *strengthens* #40's measurement priority rather than
   displacing it: buy the 4-wire contact-resistance measurement
   (`models/contact_resistance_protocol.py`) first.
2. **The two-phase correlations themselves.** Bubble departure diameter enters
   the rise velocity quadratically, so a 2× sizing error moves void fraction
   ~4×. Even a 10× void error, however, leaves the RC-1 energy shift under
   0.3 kWh/t — which is precisely why gas measurement is *not* promoted to first
   priority by this result. It should be bought when the program moves to tall
   electrodes, where the same coupling predicts the uniformity, not the energy,
   is what degrades. The protocol is already written:
   `gas_holdup.measurement_protocol()`.

**On the 1-D assumption.** Coupling shows the axial term is present, correctly
signed and small at bench scale; it does not show the 1-D drift-flux closure is
*right*. If the transparent RC-1 cell shows behaviour this reduced-order model
cannot represent — channelling, dead zones, bubble curtains against the membrane
— the escalation rule in `docs/REFERENCE_CELL_DESIGN_BASIS.md` applies, and the
answer is a *calibrated* compartment/strip model on measured hold-up, not an
unvalidated CFD run. No CFD, FEM, phase-field or DFT was used or is proposed
here.

## Reproduce

From the repository root:

```bash
python -m models.coupled_cell_physics
python -m models.run_coupled_cell_physics --json   # + experiments/data/coupled_cell_physics_report.json
pytest tests/test_coupled_cell_physics.py -q       # add -m slow for the joint-space scan
```

Both runners print the coupled-vs-uncoupled contrast, the reachability verdict
under coupling and the gas impact, with the `unvalidated (L0)` flag and the
"not gate evidence" statement on every number.
