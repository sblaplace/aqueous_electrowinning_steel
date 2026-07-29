# Experimental data

Keep raw instrument files unchanged.  Convert a copy into the canonical CSV
schemas below and record the conversion, instrument, calibration, and sample
identifiers in run notes.  One row represents one measurement point unless
otherwise noted.

## Phase I voltammetry

Use `voltammetry_template.csv` as the canonical long-form schema for CV/LSV
exports.

### Required columns

- `timestamp_s` — elapsed time from the start of the run
- `potential_V_vs_ref` — applied/working-electrode potential, V vs the stated reference
- `current_A` — signed measured current (cathodic current is negative by convention)
- `working_electrode_area_cm2` — geometric area used for current-density conversion

### Recommended metadata columns

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

## EIS spectra

Use `eis_template.csv` as the canonical schema for impedance exports: one
frequency point per row. Frequencies are swept high → low, and the imaginary
part `z_imag_ohm` is negative for the capacitive (faradaic) semicircle.

### Required columns

- `frequency_hz`
- `z_real_ohm`
- `z_imag_ohm`

### Recommended metadata columns

`z_magnitude_ohm`, `phase_deg`, `working_electrode_area_cm2`, `dc_bias_V_vs_ref`,
`temperature_C`, `pH`, `fe2_concentration_M`, `electrolyte_id`,
`reference_electrode`, and `notes`. Derived columns (magnitude, phase,
area-normalized Ω·cm²) are added automatically by the loader.

Validate, summarize, and fit a Randles equivalent circuit with:

```python
from models.eis import load_spectrum, summarize_spectrum, fit_randles_spectrum
run = load_spectrum("experiments/data/eis_template.csv")
print(summarize_spectrum(run))
freq = run["frequency_hz"].to_numpy()
z = run["z_real_ohm"].to_numpy() + 1j * run["z_imag_ohm"].to_numpy()
fit = fit_randles_spectrum(freq, z, include_warburg=True)
print(fit.rs_ohm, fit.rct_ohm, fit.cdl_F, fit.chi_squared)
```

Compare fits with and without the Warburg diffusion element (χ² ratio) before
reporting Rct; a low-frequency diffusion tail inflates Rct if modeled as a
plain semicircle. Convert Rct to an exchange current only for spectra measured
near an equilibrium potential.

## Phase II galvanostatic trace

Use `hull_cell_galvanostatic_template.csv` for the current history associated
with a Hull-cell or galvanostatic deposition run.  It is a separate file from
the weighing record so that unmodified time-series exports and auditable
pre/post measurements can be preserved.

### Required columns

- `timestamp_s` — elapsed time, seconds; chronological and non-decreasing
- `current_A` — signed cell/cathode current, A; **negative is cathodic** in the
  repository convention

### Recommended metadata columns

`cell_voltage_V`, `working_electrode_area_cm2`, `temperature_C`, `pH`,
`fe2_concentration_M`, `electrolyte_id`, `current_sign_convention`, and
`notes`.

`models.hull_cell.load_galvanostatic_trace` validates these fields and derives
signed current-density columns when `working_electrode_area_cm2` is supplied.
The gravimetric calculation clips the non-cathodic portion before integrating
charge, which allows a documented pulse/reverse trace to be analyzed without
counting anodic charge as iron-reduction charge.

## Phase II gravimetry

Use `hull_cell_gravimetry_template.csv` for **one row per weighed coupon/run**.
The record must identify the coupon that matches the trace and document a
consistent rinse/dry procedure.

### Required columns

- `mass_before_g` — dry coupon mass before deposition, g
- `mass_after_g` — dry coupon mass after deposition, g

### Recommended columns

- `run_id`, `coupon_id` — linkage to trace and physical coupon
- `blank_mass_change_g` — matched blank mass change, g, subtracted from coupon
  mass gain; use zero only when no justified blank correction is used
- `mass_uncertainty_g` — one-standard-deviation uncertainty of **each** coupon
  weighing, g
- `blank_mass_uncertainty_g` — uncertainty of the blank correction, g
- `electrode_area_cm2`, `drying_protocol`, `notes`

Calculate the result with:

```python
from models.hull_cell import (
    analyze_gravimetric_efficiency, load_galvanostatic_trace, load_gravimetry,
)
trace = load_galvanostatic_trace("experiments/data/hull_cell_galvanostatic_template.csv")
weighing = load_gravimetry("experiments/data/hull_cell_gravimetry_template.csv")
result = analyze_gravimetric_efficiency(trace, weighing, cathodic_sign="negative")
print(result.summary())
```

The calculation is

\[
\mathrm{FE}_{\mathrm{app}} =
\frac{m_{\mathrm{after}}-m_{\mathrm{before}}-m_{\mathrm{blank}}}
{Q_{\mathrm{cathodic}} M_{\mathrm{Fe}}/(2F)}.
\]

It is **apparent gravimetric Fe FE**, not a substitute for composition analysis.
Verify deposit identity and dryness (e.g., retain/inspect rinse residues and
pair with SEM/EDS when available).  Do not cap a result above 100%; it is a QA
signal that can indicate retained electrolyte, oxidation, codeposition, a
charge-sign problem, or weighing/drying error.

## Phase IV durability and closed-loop series

Use `phase4_durability_template.csv` for synchronized accelerated-life and
inventory measurements. Required columns are `time_hr`, `current_A`,
`anode_area_m2`, `cell_voltage_V`, `fe_M`, `ligand_M`, `impurity_M`, and
`anode_mass_loss_mg_m2`. Time must be strictly increasing; current is positive
anodic current. Recommended metadata include chloride, temperature, pH,
electrolyte ID, anode lot/coating loading, analytical methods, and notes.

Mass loss should be cumulative per geometric anode area and independently
verified where possible (e.g. dissolved-metal analysis versus coupon mass).
The included rows demonstrate the schema only and are not experimental claims.
Analyze a mapped copy while retaining raw instrument and ICP/OES files:

```bash
python experiments/notebooks/phase4_closed_loop.py \
  experiments/data/phase4_durability_template.csv
```
