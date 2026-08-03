# Voltage decomposition: which lever buys the energy gate?

## Scope and status

This report is a transparent, bottom-up **Level-0 screening prediction** for the
RC-1 reference divided cell at **300 mA/cm²**. Every predicted value in this
report is **unvalidated (L0)**. There is no real laboratory data in the
calculation, and no value below is a measured result.

This is **NOT gate evidence**. The energy and FE gates are measurement-only and
are implemented in `models/process_gates.py`. The tables below are intended to
prioritize the next measurement, not to declare that the route passes or fails
an experimental gate.

The implementation is additive:

- `models/voltage_decomposition.py` reuses
  `economics_from_physics.reference_cell()` for the RC-1 configuration;
- `CellPhysics.solve_at_j()` supplies the coupled transport/FE point;
- `CellPhysics._build_voltage_model()` supplies the existing
  `CellVoltageModel` decomposition;
- `specific_energy_kWh_per_t(V, FE)` translates each voltage scenario into
  specific energy.

The module enforces closure of the unrounded components to `V_cell` within
`1e-6 V` **[unvalidated (L0) implementation tolerance]**. It does not stack
single-lever changes in the reported ranking.

## RC-1 reference point

The reused reference cell is the named
`RC-1-reference-divided-cell`: a **10 cm² cathode [unvalidated (L0)]**, a
**3.0 mm interelectrode gap [unvalidated (L0)]**, membrane area resistance of
**3.0 × 10⁻⁴ Ω m² [unvalidated (L0)]**, contact resistance of
**5.0 × 10⁻⁴ Ω m² [unvalidated (L0)]**, and anode bubble fraction of
**0.10 [unvalidated (L0)]**. The bath and conditions are the same 1.0 M FeSO₄,
0.5 M Na₂SO₄, acidic sulfate configuration used by
`docs/SIM_PHYSICS_DERIVED_ECONOMICS.md`.

The governing screening identity is:

\[
E_\mathrm{specific} = 959.9 \times V_\mathrm{cell}/FE
\quad \mathrm{kWh/t\ Fe},
\]

where the **959.9 coefficient is the program's rounded presentation of the
shared Faraday-law helper [unvalidated (L0) calculation]**. FE is a fraction,
not a percentage.

## Full decomposition at 300 mA/cm²

The exact unrounded values returned by `decompose_at(reference_cell(), 300.0)`
are shown to six significant decimal places. Each value is a model prediction,
not a measurement.

| Component | Value | Interpretation |
|---|---:|---|
| `E_thermodynamic` | **1.719606 V [unvalidated (L0)]** | Thermodynamic cell-voltage requirement |
| `eta_cathode` | **0.695993 V [unvalidated (L0)]** | Coupled transport/kinetic cathode overpotential |
| `eta_anode` | **0.400000 V [unvalidated (L0)]** | Fixed-OER fallback in the RC-1 voltage model |
| `IR_electrolyte` | **0.462288 V [unvalidated (L0)]** | Gap/conductivity/bubble-adjusted electrolyte drop |
| `IR_membrane` | **0.900000 V [unvalidated (L0)]** | Membrane area resistance drop |
| `IR_contacts` | **1.500000 V [unvalidated (L0)]** | Terminal/contact/current-collector drop |
| `IR_total` | **2.862288 V [unvalidated (L0)]** | Sum of the three ohmic components |
| `V_cell` | **5.677888 V [unvalidated (L0)]** | Closed total cell voltage |
| `FE` | **0.985112 [unvalidated (L0)]** | Fractional current efficiency |
| `specific_energy_kWh_t` | **5,532.3 kWh/t Fe [unvalidated (L0)]** | Shared voltage/FE energy calculation |

The closure check is:

```text
1.719605622 V [unvalidated (L0)] + 0.695993498 V [unvalidated (L0)]
+ 0.400000000 V [unvalidated (L0)] + 2.862288445 V [unvalidated (L0)]
= 5.677887565 V [unvalidated (L0)]
```

with residual below **1 × 10⁻⁶ V [unvalidated (L0) implementation
threshold]**. The ohmic term is larger than the combined thermodynamic and
kinetic terms:

- ohmic: **2.862288 V [unvalidated (L0)]**;
- thermodynamic plus both overpotentials: **2.815599 V [unvalidated (L0)]**.

Within the ohmic pool, the predicted ordering is contact resistance first at
**1.500000 V [unvalidated (L0)]**, membrane second at **0.900000 V
[unvalidated (L0)]**, and electrolyte third at **0.462288 V [unvalidated
(L0)]**. The coupled model's cathode overpotential is **0.695993 V
[unvalidated (L0)]**, rather than the rounded illustrative **0.30 V** value
sometimes used in the program narrative; this report retains the actual
`CellPhysics` output and does not tune it to force a gate result.

At the baseline FE, the **4,000 kWh/t Fe energy threshold [program gate
threshold]** is not met by the predicted **5,532.3 kWh/t Fe [unvalidated
(L0)]**. That statement is a screening result only and is not a gate decision.

## Single-lever sensitivity

Each row changes exactly one input and recomputes the voltage model. FE is held
fixed at the baseline **0.985112 [unvalidated (L0)]** in every row. `delta_V`
is the baseline voltage minus the scenario voltage, so larger positive values
are better. `energy_after` uses the shared `specific_energy_kWh_per_t` helper.

| Lever | Stated current value | Stated proposed value | `delta_V` | `V_after` | `energy_after` | `gate_pass_after` |
|---|---:|---:|---:|---:|---:|:---:|
| contact resistance | 5.0 × 10⁻⁴ Ω m² **[unvalidated (L0)]** | 1.0 × 10⁻⁴ Ω m² **[unvalidated (L0)]** | **1.200000 V [unvalidated (L0)]** | **4.477888 V [unvalidated (L0)]** | **4,363.1 kWh/t Fe [unvalidated (L0)]** | **False [unvalidated (L0)]** |
| membrane area resistance | 3.0 × 10⁻⁴ Ω m² **[unvalidated (L0)]** | 1.5 × 10⁻⁴ Ω m² **[unvalidated (L0)]** | **0.450000 V [unvalidated (L0)]** | **5.227888 V [unvalidated (L0)]** | **5,093.8 kWh/t Fe [unvalidated (L0)]** | **False [unvalidated (L0)]** |
| electrode gap | 3.0 mm **[unvalidated (L0)]** | 1.5 mm **[unvalidated (L0)]** | **0.231144 V [unvalidated (L0)]** | **5.446743 V [unvalidated (L0)]** | **5,307.1 kWh/t Fe [unvalidated (L0)]** | **False [unvalidated (L0)]** |
| anode overpotential | 0.400000 V **[unvalidated (L0)]** | 0.300000 V **[unvalidated (L0)]** | **0.100000 V [unvalidated (L0)]** | **5.577888 V [unvalidated (L0)]** | **5,434.9 kWh/t Fe [unvalidated (L0)]** | **False [unvalidated (L0)]** |
| anode bubble fraction | 0.10 **[unvalidated (L0)]** | 0.05 **[unvalidated (L0)]** | **0.024331 V [unvalidated (L0)]** | **5.653557 V [unvalidated (L0)]** | **5,508.6 kWh/t Fe [unvalidated (L0)]** | **False [unvalidated (L0)]** |

### Basis for the proposed changes

- **Contact resistance:** a bolt, busbar, and current-collector contact
  optimization scenario. It attacks the single largest predicted ohmic term.
- **Membrane area resistance:** a thinner or lower-resistance separator
  scenario, without removing the divided-cell architecture.
- **Electrode gap:** halve the electrolyte path from **3.0 mm to 1.5 mm
  [unvalidated (L0)]**, subject to the RC-1 geometry constraint and without
  claiming that the smaller gap is experimentally buildable.
- **Anode bubble fraction:** reduce the modeled fraction from **0.10 to 0.05
  [unvalidated (L0)]** through degassing or improved anolyte gas release.
- **Anode overpotential:** reduce the fixed-model OER overpotential from
  **0.40 V to 0.30 V [unvalidated (L0)]** as a preferred-catalyst scenario.
  The reference `CellPhysics` construction uses the `CellVoltageModel`
  fixed-anode fallback, so this is an explicit model-input sensitivity, not a
  claim that a catalyst has achieved that value.

No single stated improvement flips the **4,000 kWh/t Fe [program gate
threshold]** in this screen. The largest single improvement—contact resistance—
still leaves **4,363.1 kWh/t Fe [unvalidated (L0)]**. This is reported honestly;
no combined scenario is used to manufacture a passing headline. A combined
optimization, if desired, is a separate follow-up and must be labelled as such
and measured before it can inform a gate.

## Ranked answer: which lever to pull?

Sorted by `delta_V` (largest voltage saving first), the L0 ranking is:

1. **Contact resistance** — **1.200000 V [unvalidated (L0)]** saved in the
   stated scenario.
2. **Membrane area resistance** — **0.450000 V [unvalidated (L0)]** saved.
3. **Electrode gap** — **0.231144 V [unvalidated (L0)]** saved.
4. **Anode overpotential** — **0.100000 V [unvalidated (L0)]** saved.
5. **Anode bubble fraction** — **0.024331 V [unvalidated (L0)]** saved.

**Screening conclusion:** closing the contact resistance is the highest-value
single modeled voltage lever. This is conditional on the unvalidated RC-1
model, not a procurement or experimental claim.

## Buy-next-measurement recommendation

Buy a **measured terminal-to-electrode contact resistance** next, including the
current-collector, busbar, and relevant terminal interfaces. The reason is
specific: contact resistance is the largest predicted ohmic component at
**1.500000 V [unvalidated (L0)]**, ranks first in the single-lever screen, and
the proposed measurement directly tests the input responsible for the largest
predicted saving of **1.200000 V [unvalidated (L0)]**. A membrane area-resistance
measurement is the secondary inexpensive measurement because the membrane is
the second-largest pooled ohmic term at **0.900000 V [unvalidated (L0)]**.

This is a prioritization recommendation, not an asserted measured value. If the
contact measurement refutes the ranking, the decomposition should be rerun with
the measured value; the next buy should then follow the revised ranking. The
measurement and any future FE/voltage gate decision remain subject to the
measurement-only process gates in `models/process_gates.py`.

## Reproduce

From the repository root:

```bash
python -m models.voltage_decomposition
```

The runner prints the full decomposition, every single-lever result, the
volts-saved ranking, and the recommendation. It marks predicted values as
**unvalidated (L0)** and prints the explicit **not gate evidence** warning.
