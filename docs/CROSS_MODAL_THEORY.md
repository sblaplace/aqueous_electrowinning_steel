# Cross-Modal Bottom-Up Theory — One Shared Parameter Set

**Module:** `models/cross_modal_theory.py`
**Registry:** `models/uncertainty/parameter_registry.py` (extended with thermal + crate/env props)
**Tests:** `tests/test_cross_modal_theory.py`
**Run:** `python -m models.cross_modal_theory` (or import `run_all_modalities`)

## The principle

The repository is built bottom-up: diffusivities, exchange currents, thermal and
transport properties are *the same physics* whether they appear in the cell
chemistry, the electrolyte transport, the thermal transient, or the envelope.
That commitment has a sharp corollary — **one theory must reproduce many
independent observable domains at once**. `cross_modal_theory.py` is the
consistency harness that forces the existing seams
(`cell_physics`, `thermal_balance`, `crate`, `env_coupling`) through a single
`:class:`SharedScenario`` and reports per-modality PASS/FAIL with the
*controlling parameters*.

The point is not "does each model run" — each already runs on its own. The point
is whether they **stay consistent when fed the same numbers**. The seams already
disagree by design in places (documented in `test_cross_model_consistency.py`);
this harness adds the *thermal* and *envelope/environmental* axes and asks the
stronger question: is there a single parameter set that closes **all** of them?

## The shared parameter registry

Everything begins in `models/uncertainty/parameter_registry.py`. The `thermal`
and `crate` modules were added so the registry — previously metallurgy/
electrochem/transport only — now carries the heat-transfer props
(`volume_L`, `UA_amb_W_K`, `UA_jacket_W_K`, `electrode_area_m2`,
`thermoneutral_V`, `T_ambient_C`) and envelope/site props
(`crate_mass_kg`, `crate_height_m`, `drag_coefficient`, `design_gust_m_s`,
`soil_bearing_kPa`) alongside the existing exchange currents (`fe_i0`, `her_i0`,
Tafel slopes) and transport/thermal diffusivities.

`SharedScenario.from_registry()` builds the one parameterization *from* these
nominals, so the registry is the single source of truth that every modality
reads. This is what "used by at least the cell/thermal/env models" means
operationally: no model gets a private copy of a constant anymore.

## The harness

`run_all_modalities(scenario)` runs every modality with the one `SharedScenario`
and returns a `CrossModalReport` with five modality verdicts:

| Modality | Seam | What it checks | Controlling params (examples) |
|----------|------|----------------|-------------------------------|
| `electrochem` | `cell_physics` | FE ≥ floor, no Fe(OH)₂ precipitation, sane surface pH | `fe_i0`, `her_i0`, `j`, `T_operating`, `pH` |
| `transport` | `cell_physics` (Nernst–Planck) | applied `j` ≤ migration-enhanced transport limit | `j`, `transport_limit`, `boundary_layer`, `T` |
| `thermal` | `thermal_balance` | converged `T_ss` within tolerance of the assumed `T_operating` AND inside a safe band | `V_cell`, `I`, `UA_amb`, `volume`, `UA_jacket`, `thermoneutral_V` |
| `crate` | `crate` | envelope stably sits the wind (overturn/bearing/sliding) | `gust_m_s`, `crate_mass`, `crate_height`, `soil_bearing` |
| `environment` | `env_coupling` | thermal transient used the env-consistent ambient T and wind-driven `h_conv` | `gust_m_s`, `rain`, `T_ambient`, `h_conv` |

The thermal verdict is computed from the **converged** steady-state of the exact
heat balance (`Q_gen = I·(V_cell − E_therm)` vs ambient + evaporative + jacket
losses), not a fixed 4 h cut — a large bath never equilibrates in 4 h, so a
fixed-horizon endpoint would mask a real runaway.

## The headline conflict: thermal closure

The strongest cross-modal tension is thermal. The electrochem/transport modals
fix an operating temperature `T_operating` and evaluate every kinetic and
diffusive property there. But the **same** parameter set, run through the
thermal balance, produces the *actual* steady-state temperature from the very
`V_cell` and `I` the electrochem model returns:

```
Q_gen = I · (V_cell − E_therm)  =  UA_amb·(T − T_amb) + Q_evap(T) + UA_jacket·(T − T_cool)
```

If that equilibrates far from `T_operating`, **no single parameterization
exists** — the chemistry is valid at 60 °C, but the cell physically sits at
76 °C (or, on a windy site, at 26 °C). `test_default_parameterization_surfaces_thermal_conflict`
locks this in: with the registry nominals and no cooling, electrochem, transport,
crate and env all PASS while **thermal FAILs** (T_ss ≈ 76 °C vs 60 °C) and the
report's verdict reads `CONFLICT — no single parameter set satisfies every modality`.

This is the value of the harness: it does not "fix" the disagreement away; it
names the controlling parameters (`V_cell`, `current_A`, `UA_amb_W_K`,
`volume_L`, `UA_jacket_W_K`) so a designer sees exactly what must move.

## The one degree of freedom that re-closes it

`find_consistent_cooling` / `consistent_scenario` show that the loop re-closes
with a single free parameter — the cooling-jacket `UA` sized to the `V_cell·I`
heat load. `test_consistent_parameterization_closes_every_modality` asserts that
the **same** parameterization, with a jacket that removes the electrochem heat
button, satisfies every modality at once. That is the "one bottom-up
parameterization reproduces the cell chemistry AND the transport AND the thermal
transient AND the envelope" case.

## Secondary conflicts the harness surfaces

- **Transport discipline** — push `j` above the migration-enhanced transport
  limit and transport (and eventually chemistry, via precipitation) fails;
  FE drops and the bath precipitates.
- **Crate/envelope** — a light, tall crate at 60 m/s wind fails overturning /
  sliding; the same gust that endangers the envelope also **overcools the bath**
  through forced convection (`env_coupling`'s `h_conv` augments the thermal loss)
  — the wind the crate is sized for is the wind that keeps the bath from
  reaching its operating temperature.
- **Environment→thermal wiring** — with no wind/rain/ingress the disturbance is
  a strict no-op (`T_amb` and `h_conv` unchanged), preserving the "coupling off
  by default" guarantee from `env_coupling.py`.

## Interpreting the verdict

`CrossModalReport.consistent` is **True only if all five modalities pass with
the same parameter set**. A conflict is not a bug to be papered over; it is the
harness telling you which observable domain the current numbers contradict.
All numbers remain L0 screening like the source seams (`credibility vector:
process/crate/site all L0`) until real cell, load-test and site-survey data
validate them.
