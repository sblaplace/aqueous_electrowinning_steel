# Experimental data contract — version 1.0

This document defines the boundary between a physical experiment and the
modeling suite. Raw vendor exports remain immutable. A mapped copy is placed in
a run directory and is validated before any derived quantity or process-gate
evidence is produced.

## Three separate contracts

| Contract | Purpose | Required measurement columns |
|---|---|---|
| `voltammetry-timeseries` | Phase-I CV/LSV | `timestamp_s`, `potential_V_vs_ref`, `current_A`, `working_electrode_area_cm2` |
| `plating-timeseries` | Deposition, Hull-cell, and divided-cell runs | `timestamp_s`, `current_actual_A`, `voltage_V` |
| `campaign-manifest` | Cross-run index and traceability | `run_id`, `phase`, `technique`, `status`, `raw_file`, `processed_file`, `metadata_file` |

The Phase-II Hull-cell trace historically uses `current_A` and
`cell_voltage_V`. `models.run_record.load_plating_timeseries` maps those names
to `current_actual_A` and `voltage_V` when—and only when—the canonical names
are absent. A file containing both spellings is rejected instead of guessing
which channel is authoritative.

## Plating run directory

A complete run has this layout:

```text
run-20260801-001/
├── manifest.json          # required; schema_version = "1.0"
├── bath_batch.json        # required; linked by manifest.bath_batch
├── metadata.json          # required instrument/sample sidecar
├── timeseries.csv         # required plating trace
├── mass_log.csv           # optional until dry weighing is complete
├── characterization.csv   # optional SEM/EDS, combustion, or XRD records
├── video_index.csv        # optional camera/event index
└── energy_log.csv         # optional auxiliary loads, one row per component
```

The manifest may put files in subdirectories through relative paths under
`files`. Absolute paths are rejected. Default names are the names shown above,
so a minimal manifest can omit `files` while a real campaign should record all
paths explicitly.

`record_status` is one of `planned`, `in_progress`, `complete`, or `excluded`.
Missing required files are warnings for planned/in-progress records and errors
for complete records. A run is `ready_for_analysis` only when the manifest,
bath batch, metadata, and valid plating trace are present and its derived
quantities can be computed.

Validate a run without throwing on an incomplete record:

```bash
python -m models.run_record experiments/data/runs/beaker-20260801-001 \
  --output experiments/data/runs/beaker-20260801-001/qa_report.json
```

Use `--strict` in automation when an incomplete record should fail the command.
Programmatically:

```python
from models.run_record import build_qa_report, load_run_record

qa = build_qa_report("experiments/data/runs/beaker-20260801-001")
run = load_run_record("experiments/data/runs/beaker-20260801-001")  # fail-fast
```

## Sidecar rules

### `metadata.json`

The sidecar contains the instrument, calibration, electrode, electrolyte,
temperature, agitation, and preparation fields in
`experiments/data/metadata_template.json`. Optional `raw_export_sha256` records the
checksum of the immutable vendor file. A checksum is traceability, not a
scientific validation result.

### `bath_batch.json`

The `batch_id` must equal `manifest.bath_batch`. The initial bath inventory is
computed only when `composition.fe2_g_L` and `composition.volume_mL` are
available. If post-run `analysis.fe2_measured_g_L` is present, the report also
shows the post-run inventory. No precipitate or crossover loss is invented.

### `mass_log.csv`

A run-level mass log has one row with `mass_before_g` and `mass_after_g`. The
optional blank correction and weighing uncertainties are retained. Panel-level
Hull-cell records should continue to use
`models.hull_cell.load_gravimetry`, not be collapsed into one run-level row.

### `characterization.csv`

Use the existing long-form characterization contract. Fe composition in `wt%`
or `mass%` allows the charge and iron ledgers to distinguish Fe mass from total
dry mass. SEM/EDS, combustion, and XRD measurements remain separate; no
cross-technique normalization is performed.

### `energy_log.csv`

One row per auxiliary component:

```text
component,energy_Wh,uncertainty_Wh,measurement_method,notes
pumps,0.42,0.03,inline power meter,
heating,1.10,0.08,plug meter,
```

Allowed components are `pumps`, `heating`, `cooling`, `gas_handling`,
`drying`, and `other_auxiliary`. Stack energy is always integrated from the
plating trace as `V × |I|`; it must not be entered again in this file.

## Ledger semantics

The JSON QA report contains three independent ledgers:

- **Charge:** integrated cathodic charge and apparent FE from total dry mass.
  An Fe-specific deposition charge is emitted only when an independent Fe
  composition is available. Hydrogen/other-product charge remains unresolved
  unless it is measured separately.
- **Iron:** initial bath inventory, measured deposit Fe, post-run bath
  inventory, and any explicitly recorded solids/other Fe streams. The status
  remains `partial` until all streams are recorded.
- **Energy:** measured stack electrical energy plus logged auxiliary loads.
  The status is `partial` when one or more auxiliary components are missing.

A mass-only FE above 100% is retained as a QA signal. It is never clipped and
never relabeled as verified iron FE.

### Predicted (not measured) idle-corrosion terms

If the run included an **unpowered soak** (idle, shutdown, weekend hold),
declare it in the manifest and the QA pipeline attaches a *predicted* term
from `models/deposit_corrosion.py` (L1 screening) to the charge and iron
ledgers:

```json
"setup": {
  "idle": {
    "hours": 8.0,
    "pH": 2.0,
    "T_C": 40.0,
    "a_fe3_M": 1.0e-4,
    "o2_fraction_of_sat": 0.05,
    "theta_additive": 0.5,
    "mixing": "stagnant",
    "deposit_thickness_um": 50.0
  },
  "cathode": {"area_cm2": 100.0}
}
```

- `hours` is required; every other field is an optional override of the
  anchored defaults (bath Fe³⁺ defaults to the live `fe3_shuttle` steady
  state at the run's pH).
- The prediction **annotates residuals, it never rewrites measured or
  derived values**: the charge ledger gains
  `predicted_idle_redissolution_charge_C` and
  `unresolved_charge_after_predicted_idle_C`; the iron ledger gains
  `predicted_idle_transfer_to_bath_fe_mol`. Idle redissolution moves Fe
  deposit→bath, so it biases gravimetric FE low while **conserving** the
  mol-scale iron closure — do not "close" the closure by hand with it.
- The prediction is advisory: a missing or malformed block changes nothing
  and is never a QA error.

## Gate evidence

A QA-ready run is not automatically a process-gate pass. To create measured
evidence records, declare mappings in `manifest.json`:

```json
"gate_evidence": [
  {
    "candidate_id": "divided_sulfate_dissolved_feed",
    "gate_id": "gate_2_gravimetric_fe",
    "metric": "faradaic_efficiency",
    "value_from": "apparent_faradaic_efficiency",
    "unit": "fraction",
    "notes": "Apparent until deposit composition is independently verified"
  }
]
```

The pipeline resolves `value_from` only against measured or measurement-derived
metrics and forces `source: "experimental"`. It does not evaluate thresholds;
pass/fail remains the responsibility of `models.process_gates` after the run
has been reviewed.

## Campaign index

Use `experiments/data/campaign_manifest_template.csv` for the cross-run index.
`models.run_record.load_campaign_manifest` validates its structure, while
`models.campaign.validate_manifest` checks linked files and required metadata.
The two levels are intentional:

1. the per-run manifest describes the physical setup and sidecars;
2. the campaign index supports multi-run traceability and analysis selection.

Do not make a model-generated report look like a processed experimental record.
Keep synthetic fixtures under `experiments/data/synthetic/` and keep raw and
processed data in the ignored directories described in
`experiments/data/README.md`.
