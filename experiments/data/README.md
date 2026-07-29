# Experimental data

Use `voltammetry_template.csv` as the canonical long-form schema for CV/LSV
exports. Keep raw instrument files unchanged and record the conversion in the
run notes. One row represents one measurement point.

## Required columns

- `timestamp_s` — elapsed time from the start of the run
- `potential_V_vs_ref` — applied/working-electrode potential, V vs the stated reference
- `current_A` — signed measured current (cathodic current is negative by convention)
- `working_electrode_area_cm2` — geometric area used for current-density conversion

## Recommended metadata columns

`cycle`, `segment`, `temperature_C`, `pH`, `fe2_concentration_M`,
`electrolyte_id`, `reference_electrode`, and `notes`.

Validate and derive current density with:

```python
from models.experimental_data import load_measurements, summarize_run
run = load_measurements("experiments/data/voltammetry_template.csv")
print(summarize_run(run))
```

For a real run, create a separate metadata record containing instrument model,
scan rate, potential limits, electrode materials, solution preparation, and
calibration date. Do not overwrite raw files.

## Tafel fit

For an LSV cathodic branch, select a region that is kinetic-controlled (before
mass-transport curvature) and fit it directly:

```python
from models.tafel import fit_tafel
fit = fit_tafel(run, potential_min_V=-0.80, potential_max_V=-0.55,
                equilibrium_potential_V=-0.44)
print(fit.slope_V_decade, fit.exchange_current_A, fit.r_squared)
```

Inspect the selected region and report its bounds and R²; do not interpret a
transport-limited region as a kinetic Tafel slope.
