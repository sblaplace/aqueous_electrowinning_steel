# First chemistry/physics consistency upgrade

**Date:** 2026-08-06  
**Status:** Level-0 screening correction; not gate evidence

This change is the first implementation tranche from the physics audit. It
prioritizes consistency of the existing RC-1 calculation over adding another
high-fidelity downstream model.

## What changed

### One default cathode-film path

`CellPhysics` now defaults to the chemically richer `DiffusionLayer1D` path.
It carries Fe²⁺, H⁺, sulfate/bisulfate, borate, migration, local pH, and the
Fe(OH)₂ supersaturation diagnostic. The Pitzer Fe²⁺ activity correction is
applied to the cathode equilibrium potential using the bulk bath state.

The former five-species dilute `NernstPlanckFilm` remains available with:

```python
ProcessConditions(transport_model="dilute_np")
```

This is an A/B comparison path, not the default reference-cell model. The
reactive path is still a reduced-order screening model: it does not yet solve
Pitzer activities at every film node, include multicomponent Maxwell–Stefan
cross-diffusion, or consume Fe into precipitated solids.

### Shared thermodynamic anchors

`models/thermodynamic_constants.py` is now the source for the shared Fe, OER,
Fe³⁺/Fe²⁺, sulfate, borate, water, Ksp, and diffusivity screening anchors.
The former −0.447 V versus −0.440 V Fe standard-potential split is removed.

`NernstPlanckFilm` now temperature-resolves its default diffusivities. An
explicit numeric diffusivity remains a calibration/measurement override.

### Cell-voltage bookkeeping

`CellVoltageModel` now:

- honors its supplied cathode standard-state potential;
- distinguishes a 25 °C reference conductivity from conductivity already
  evaluated at the operating temperature;
- avoids a second temperature correction when the speciation model supplies
  `electrolyte_conductivity_at_temperature=True`; and
- does not apply an OER bubble penalty to a soluble-iron anode.

This raises the RC-1 screening voltage relative to the previous double-corrected
conductivity path. The change is deliberately not hidden: energy results must
be regenerated and remain Level 0 until measured conductivity and voltage taps
replace the screening inputs.

### Explicit anode chemistry

`CellGeometry` now declares `anode_chemistry="inert"` or
`"soluble"`. The soluble branch builds an `AnodeKinetics` object and uses the
Fe²⁺/Fe equilibrium rather than silently reusing the OER equilibrium. The RC-1
YAML declares its current comparator explicitly as `inert`; a soluble-anode
campaign must opt into the other mode and provide its Fe²⁺ concentration and
calibrated dissolution kinetics.

Anode gas/acid/Fe inventory balances are still a follow-up. The inert branch
continues to use the fixed OER fallback until a material-specific anode is
selected and measured.

### Buffer capacity

`buffer_capacity_M_per_pH()` computes a screening acid-equivalent capacity
from sulfate/bisulfate, borate, and water equilibria. The RC-1 digital-twin
builder no longer equates 0.4 M boric acid with 0.4 M/pH buffer capacity. At
pH 2, boric acid is nearly fully undissociated; its increment to buffer
capacity is negligible compared with free proton and sulfate contributions.

## Current regenerated screens

After this upgrade, the regenerated machine-readable reports show the impact
of the consistency fixes (all still `unvalidated (L0)`):

- theory-confidence reference point: FE **0.9926**, V_cell **5.388 V**,
  specific energy **5211 kWh/t Fe**;
- coupled gas screen minimum: **3448.0 → 3448.1 kWh/t Fe** after the gas
  correction at the screened 150 mA/cm² / 1.5 mm / low-contact point;
- the reactive-film transport limit at that point is approximately
  **782 mA/cm²**, versus the 150 mA/cm² operating duty.

These are updated screens, not experimental validation. The previous report
numbers were not silently overwritten in prose; the prior-report documents now
carry regeneration notes and the JSON artifacts were regenerated.

## What remains next

1. Make the recipe pH and acid inventory consistent through an activity-based
   speciation solve rather than accepting both as independent inputs.
2. Add Fe³⁺/Fe(OH)₃ and Fe(OH)₂ precipitation source/sink terms to the film and
   iron/acid ledgers.
3. Replace the membrane's linear Fe³⁺/H⁺ flux with a Donnan/multicomponent
   transport model calibrated to in-situ resistance and crossover.
4. Add measured anode mode, membrane, conductivity, reference-electrode, and
   gas data to the RC-1 calibration pipeline.

All numerical outputs remain predictions, not experimental gate evidence.
