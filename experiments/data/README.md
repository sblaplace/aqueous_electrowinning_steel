# Experimental data

Keep raw instrument files unchanged.  Convert a copy into the canonical CSV
schemas below and record the conversion, instrument, calibration, and sample
identifiers in run notes.  One row represents one measurement point unless
otherwise noted.

## Data lifecycle and evidence separation

The repository separates data by provenance so that models cannot silently
train on their own fixtures and so that process gates read only from
experimental records.

| Directory | Contents | Git tracked? | Role |
|-----------|----------|--------------|------|
| `raw/` | Immutable instrument exports, original file formats | No (`.gitignore`) | Source of truth for every run |
| `synthetic/` | Model-generated fixtures and templates with synthetic values | No (`.gitignore`) | Test fixtures only; never used for calibration or gates |
| `literature/` | Extracted observations from papers, patents, archaeology | No (`.gitignore`) | Prior art and external validation anchors |
| `processed/` | Mapped, reviewed CSVs derived from raw exports | No (`.gitignore`) | Calibration and analysis inputs |
| `calibrations/` | Fitted parameter sets from processed data | No (`.gitignore`) | Versioned calibration outputs |

**Contract:** Process gates and candidate evaluations read only from
`processed/` (experimental records linked by campaign manifests) or
`literature/` (external observations). `synthetic/` is for tests and
documentation only. A model must never be calibrated on its own synthetic
fixtures.

## Campaign manifest and run metadata

Before acquiring data, copy `campaign_manifest_template.csv` to a dated campaign
manifest and give each experiment a permanent, unique `run_id`. The manifest
links the immutable vendor export, mapped analysis CSV, metadata JSON, and any
characterization record. Keep vendor exports in `experiments/data/raw/` (ignored
by Git); keep mapped, reviewed data under `processed/` and record a SHA-256 hash
in metadata when practical. Start every sidecar from `metadata_template.json`.

A run marked `complete` must have all file links and the required instrument,
calibration, electrode, electrolyte, temperature, agitation, and preparation
metadata. It also requires a characterization link; for a Phase-I-only run this
may be a deliberately documented "not applicable" characterization record.
Validate the record and produce an auditable QA report with:

```bash
python -m models.campaign experiments/data/campaign_manifest.csv \
  --output experiments/data/campaign_qa_report.json
```

The validator does not alter raw files or infer scientific validity. Its
`ready_for_analysis` result confirms only that an explicitly complete record has
all required links and metadata.

## Phase I kinetic calibration

After an LSV/CV run has passed campaign QA, fit it through the manifest rather
than by pointing a model at an untracked local file:

```bash
python -m models.run_calibration experiments/data/campaign_manifest.csv P1-YYYYMMDD-001 \
  --pH 3 --temperature-C 60 --fe-conc-M 1.0 --reference-to-she-V 0.210 \
  --eis experiments/data/processed/P1-YYYYMMDD-001_eis.csv \
  --output experiments/data/calibrations/P1-YYYYMMDD-001_parameters.json
```

`reference_to_she_V` is added to the recorded potential and must be the
reference conversion appropriate to the electrolyte and temperature; document
its source in the run metadata. The fit reports a bounded Fe+HER total-current
screening model and approximate parameter uncertainty. **An LSV total current
cannot by itself distinguish Fe reduction from HER.** Treat the branch-specific
values as provisional until they are constrained by independent Faradaic
efficiency, hydrogen/gas, RDE, or deposit-composition data. The optional EIS
result is a near-equilibrium consistency check only: under cathodic bias its
Rct is the combined faradaic conductance, not a Fe-only exchange current.

Keep the generated parameter JSON versioned with its campaign run IDs and do
not overwrite an earlier calibration after new data are acquired.

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

## Deposit characterization: SEM/EDS, combustion, and XRD

Use `characterization_template.csv` as a long-form, measurement-level record
for characterization linked from the campaign manifest. One row is one analyte
or phase result; retain the original spectrum, image export, combustion export,
or diffraction scan at the `analysis_file` path. The file supports:

- `SEM_EDS` composition entries in `wt%`/`mass%` (record area/spot basis and
  measurement locations in notes);
- `COMBUSTION` bulk carbon/sulfur/etc. entries in `wt%`/`mass%`, including
  blank/certified-reference details; and
- `XRD` phase-identification or refinement metrics, with their stated units and
  refinement basis.

Do not merge EDS, combustion, and XRD results into an artificial composition
balance: they measure different volumes and quantities. `models.characterization`
validates the schema and reports EDS totals outside 95–105 wt% as visible QA
flags rather than renormalizing data:

```python
from models.characterization import load_characterization, summarize_characterization
records = load_characterization("experiments/data/characterization_template.csv")
print(summarize_characterization(records))
```

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
