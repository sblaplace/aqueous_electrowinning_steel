# Reference-cell configuration spec — RC-1 (immutable), v1.0

**Status:** D1 engineering deliverable — frozen reference configuration for the
reference divided cell. This document is the single authoritative, immutable
statement of what the reference configuration **is**. It consolidates, at one
version, the inputs that were previously spread across
`docs/REFERENCE_CELL_DESIGN_BASIS.md`, `docs/BATH_SPEC.md`,
`docs/DEPOSIT_METROLOGY.md`, and `processes/reference_cell_rc1.yaml`.

**Spec version:** `1.0`  ·  **Configuration id:** `RC-1`
**Machine-readable form:** `processes/reference_cell_spec.v1.json` (content-pinned by SHA-256)
**Effective / frozen:** content-pinned at the D1 engineering freeze (2026-08-05, sha256 `44f69f3dc5…`); the formal design-review sign-off recorded in §10 is the authoritative approval to run a campaign under `RC-1`.

> A configuration is **immutable** in the sense that matters here: once this
> spec version is signed, no value in it may be changed in place. Any intended
> change — geometry, membrane, batch recipe, operating setpoint, measurement
> requirement — requires a **new spec version** (`1.1`, `2.0`, …), never an edit
> to `1.0`. Every experimental run pins the spec version + content hash it was
> executed under (`reference_cell.json` → `reference_cell_spec.spec_sha256`).
> Results may be compared across runs **only** when they pin the same spec
> version; a changed geometry/membrane/batch is a new configuration, not a
> repeat. This is the mechanism that makes `run_record` comparisons meaningful.

This document is a **design/basis** deliverable for Nanshe `Next-steps §1`. It
is **not** experimental evidence and does not claim that predicted FE, voltage,
deposit quality, or durability will be achieved. It is the definition of the
apparatus against which raw data are collected and ledgers are closed.

---

## 0. How to read this spec

Every value below carries one of three tags:

| Tag | Meaning |
|---|---|
| **FROZEN** | A hard target that a conforming run must meet (within the stated tolerance) or explicitly record as an as-built deviation. |
| **BAND** | An acceptance interval. A batch/flow/temperature measured outside the band is **off-spec**: it must be flagged in the run record and reason about, never silently averaged in. |
| **MEASURE** | A quantity that must be *measured and recorded per run* because it cannot be set from geometry/recipe alone (e.g. per-batch conductivity). Its reference value is a target, its recorded value is the run's truth. |

The versioned raw-linked run record is defined in §9 and consumed by
`models.run_record`. The immutable machine spec is `processes/reference_cell_spec.v1.json`.

---

## 1. Divided cell, membrane, and active area

### 1.1 Cell stack (geometry) — FROZEN

| Item | Specification | Notes |
|---|---|---|
| Cell type | **Divided**, membrane-separated, recirculating | Independent catholyte/anolyte loops. |
| Cathode active area | **10.0 cm²** (50 mm × 20 mm) FROZEN | Masked 316L coupon. Reaches 300 mA/cm² at 3.0 A. |
| Cathode material | **316L**, 1.0 mm thick FROZEN | Removable, weighable commissioning substrate; not a harvest-architecture choice. |
| Anode exposed area | **≥ 10.0 cm²** FROZEN (actual recorded) | Area ratio ≥ 1:1. Anode identity/area are controlled variables. |
| Anode material | Specified OER-capable / insoluble anode, controlled per campaign FROZEN | Material may not silently change between runs; see §2.4 of the design basis. |
| Electrode-to-membrane gap | **3.0 mm per side** FROZEN (as-built recorded) | A geometry input, not an assumed 2 cm gap. |
| Membrane | **Nafion N117**, nominal wetted 50 mm × 50 mm FROZEN | Cation-exchange, removable cassette. First comparator, not proven optimum. |
| Cathode channel | 50 mm L × 20 mm W × 3 mm D FROZEN | Defined flow cross-section. |
| Anode channel | 50 mm L × 20 mm W × 3 mm D FROZEN | Symmetric for first interpretation. |
| Orientation / flow path | Vertical channel, bottom inlet → top outlet FROZEN | Buoyant gas escape; required H₂ path. |
| Wetted materials allowed | Borosilicate, PP, PVDF, PTFE, FEP; EPDM/Viton only after compatibility review FROZEN | No unqualified metal/adhesive/3-D-print resin in hot acidic electrolyte. |

### 1.2 Membrane cassette and area bookkeeping — FROZEN RULE

The membrane cassette defines the **exposed membrane area independently** of the
coupon mask. Record **both** the wetted membrane area and the coupon active
area in every run. No performance result may be compared across runs if either
changes without a new configuration version.

---

## 2. Cathode and anode: material, spacing, roughness, exposed area

| Quantity | Reference | Tag |
|---|---|---|
| Cathode material | 316L, 1.0 mm | FROZEN |
| Cathode active area | 10.0 cm² (50 × 20 mm mask) | FROZEN |
| Cathode surface preparation | Controlled and recorded per run; **Ra target ≤ 0.8 µm** (ground/polished), measured Ra recorded | FROZEN target / MEASURE |
| Cathode roughness band | Ra ≤ 2.0 µm accepted; above → off-spec flag | BAND |
| Anode material | OER-capable / insoluble, fixed per campaign | FROZEN |
| Anode exposed area | ≥ 10.0 cm², actual measured | FROZEN min / MEASURE |
| Anode roughness | Recorded; no first-campaign requirement | MEASURE |
| Electrode-to-membrane gap | 3.0 mm/side (as-built recorded after compression) | FROZEN / MEASURE |

Roughness and exposed area are the two electrode-surface quantities the task
calls out explicitly; both are therefore **recorded per run** and compared to
the frozen target. A roughness or area change that would affect current-density
or surface-state interpretation requires a config note / new version before the
result is comparable across runs.

---

## 3. Electrolyte batch: recipe, impurities, physical properties, temperature

### 3.1 Batch recipe — FROZEN targets (from `docs/BATH_SPEC.md`)

| Component | Target | Unit | Tolerance |
|---|---|---|---|
| FeSO₄·7H₂O | 278 (1.0 M) | g/L | ±5 g/L (recipe grading) |
| H₃BO₃ (boric acid) | 24.8 (0.40 M) | g/L | ±0.5 g/L |
| Ascorbic acid | 1.0 | g/L | ±0.2 g/L |
| pH | 2.0 | — | **band 1.8–2.2** |
| Supporting Na₂SO₄ | 0 (optional) | mol/L | not required |

### 3.2 Impurity ceilings — FROZEN (batch acceptance)

| Impurity | Bath ceiling (ppm) | Rationale |
|---|---:|---|
| Cu | **< 15** | hot-shortness (< 0.1 wt% deposit at j ≥ 200 mA/cm²) |
| Ni | **< 50** | co-deposition limit |
| Zn | **< 50** | co-deposition limit |
| Pb | **< 10** | toxicant, minimised |
| Sn | **< 5** | low concern |

A batch whose **certificate of analysis / independent assay** exceeds a ceiling
is off-spec; the run must flag it (§9) and the affected quantity is not cleanly
comparable across runs.

### 3.3 Physical properties — MEASURE, target + acceptance band (L0)

These are per-batch *measured* values. The reference targets are engineering
estimates from the models; the recorded batch value is the run's truth.

| Property | Reference target | Acceptance band | Basis |
|---|---|---|---|
| Density | 1.050 g/mL | 1.00–1.20 g/mL | `reference_cell_design.RHO_ELECTROLYTE_KG_M3 = 1050 kg/m³` |
| Conductivity κ | 10 S/m | 5–15 S/m | cell_physics/anode default κ=10; gas_holdup κ=13.5 |
| Viscosity η | 0.47 mPa·s @ 60 °C | 0.30–1.00 mPa·s | Andrade water proxy; **replace with measured bath value** |
| Temperature | 60 °C | **50–70 °C** qualified | program range |

Conductivity, viscosity and density must be **measured on the actual batch**
(ctx: the L0 numbers are proxies). A measured value outside the band → off-spec
flag, reason recorded.

---

## 4. Flow, mixing, gas, and thermal boundaries

### 4.1 Flow and mixing — FROZEN setpoints, MEASURE actuals

| Quantity | Reference | Tag |
|---|---|---|
| Recirculation loops | Independent catholyte & anolyte | FROZEN |
| Nominal flow per loop | **0.5 L/min** | FROZEN setpoint / MEASURE |
| Installed capability band | 0.10–1.00 L/min | BAND |
| Superficial velocity (20×3 mm) | ≈ 0.14 m/s at 0.5 L/min | derived |
| Mixing mechanism | forced recirculation (no magnetic stirrer in cell) | FROZEN |
| Channel ΔP | measured per loop; fouling observable | MEASURE |

### 4.2 Gas handling — FROZEN safety boundaries

- Separate cathode and anode headspace vents to site-approved safe exhaust;
  **no sealed gas path**.
- Optional measured cathode-gas takeoff (no meaningful backpressure).
- Hydrogen design bound: **1.37 L/h** at 3.0 A all-HER (25 °C, 1 atm) — electrical
  upper bound, not a predicted HER rate. External ventilation verified by site EHS.
- H₂ monitor (GT-101) alarm before LEL; hardwired independent rectifier disable.

### 4.3 Thermal boundary — FROZEN range, MEASURE per run

| Item | Reference |
|---|---|
| Qualified temperature range | 50–70 °C |
| Target operating temperature | 60 °C |
| Required temperature probes | catholyte reservoir, anolyte reservoir, cathode outlet, anode outlet |
| Heat generation | measured as `I × V` (never inferred from an assumed voltage) |
| Auxiliary energy | logged per component (energy ledger) |

---

## 5. Rectifier voltage/current with synchronized timestamps

| Quantity | Reference | Tag |
|---|---|---|
| Supply | 30 V / 10 A CC/CV bench supply | FROZEN |
| Normal current ceiling | **3.0 A** (300 mA/cm²) | FROZEN |
| Voltage hard ceiling (screening) | **8.0 V** (steady-state target ~6 V) | FROZEN |
| Independent measurements | DMM cell voltage (VT-201), series coulomb counter (CT-201) | FROZEN |
| Current sign convention | cathodic negative (repo convention) | FROZEN |
| **Synchronized timestamps** | all channels on **one time base (DAQ-101)**; current, voltage, temperature, flow, pressure, pH, event flags share a clock | FROZEN |
| Trace file | `timeseries.csv` (`timestamp_s`, `current_actual_A`, `voltage_V`) | MEASURE |

The rectifier `V/I` trace is the charge and energy ledger source. It must be on
the same synchronized time base as temperature/flow/pressure/pH so the run can
be replayed as one coherent record (Nanshe observability requirement).

---

## 6. Sampling: before, during, and after the run

| Timing | Required samples | Purpose |
|---|---|---|
| **Before** | catholyte & anolyte: Fe(II), pH, conductivity, density, viscosity, impurity assay | initial batch state → iron/charge ledger initial inventory |
| **During** | periodic catholyte & anolyte grabs at stated `timestamp_s` (labelled SP-101 etc.) | drift, crossover, Fe consumption rate |
| **After** | catholyte & anolyte: Fe(II) measured, pH, volume `analysis.fe2_measured_g_L` | iron ledger closure + post-run inventory |

Every sample records `sample_id`, `timestamp_s`, `loop` (catholyte/anolyte),
`sample_point` (SP-101 …), and links to its raw `file`. Sample *before* and
*after* are **required**; *during* is required for anything claimed about drift
or crossover (§9).

---

## 7. Dry deposit measurement set (metrology)

The deposit is measured after rinsing/drying to constant mass on **all seven**
axes the task calls out. Required-to-declare are `mass`, `thickness_map`,
`composition`; the others (`morphology`, `porosity`, `adhesion`,
`hydrogen_content`) are required for a **complete** reference deposit record.

| Measurement | Reference / method | Tag |
|---|---|---|
| Dry deposit mass | `mass_before_g` / `mass_after_g` (run-level), uncertainty recorded | MEASURE |
| Thickness map | dual optical (opt-101) + ultrasonic (thk-101) co-live pair; 0–500 µm | MEASURE |
| Composition | SEM/EDS (wt%/mass%), Fe; plus combustion (C/S), XRD phase — kept separate | MEASURE |
| Morphology | imaging / micrograph record | MEASURE |
| Porosity | from image/metallographic analysis | MEASURE |
| Adhesion | peel/strip force (branch decision; not a materials project) | MEASURE |
| Hydrogen content | independent H analysis (bake-out / combustion) | MEASURE |

Composition is needed to distinguish Fe mass from total dry mass so the charge
& iron ledgers can close (mass-only apparent FE is not verified iron FE).

---

## 8. Closure ledgers (unchanged contract)

Every reference run must close **three ledgers independently**; the measured
inputs above feed them, and `models.run_record` computes them with no
model-generated gate evidence:

- **Charge:** applied charge = Fe + HER + other measured products.
- **Iron:** Fe in = dissolved Fe out + deposit Fe + solids/precipitate + analytical uncertainty.
- **Energy:** DC stack energy + pumps + heating + cooling + gas handling + drying.

A ledger stays `partial` until every stream is measured; no precipitate or
crossover loss is invented (`DATA_CONTRACT.md`).

---

## 9. Versioned raw-linked data record (what `run_record` consumes)

Every reference run is a directory whose `manifest.json` declares an optional
`reference_cell.json` raw-linked sidecar. That sidecar **pins the immutable spec
and enumerates the raw-linked measurement files**. `models.run_record` validates
it (schema `aqueous-electrowinning.run-record`, `schema_version: "1.0"`).

```text
run-<refcell>-YYYYMMDD-NNN/
├── manifest.json              # experiment_type: "divided_cell"; files.reference_cell_json
├── reference_cell.json        # pins spec version + sha256 + enumerates sidecars (§9.1)
├── bath_batch.json            # per-batch measured recipe & physical properties (§3)
├── metadata.json              # instrument/sample sidecar
├── timeseries.csv             # rectifier V/I, synchronized timestamps (§5)
├── sample_log.csv             # before/during/after samples (§6)
├── mass_log.csv               # dry deposit mass (§7)
├── deposit/
│   ├── thickness_map.csv      # thickness map (§7)
│   ├── composition.csv        # SEM/EDS / combustion / XRD (§7)
│   ├── morphology.csv         # morphology record (§7)
│   ├── porosity.csv           # porosity record (§7)
│   ├── adhesion.csv           # adhesion/peel force (§7)
│   └── hydrogen.csv           # H content (§7)
├── characterization.csv       # long-form analyte/phase record
└── energy_log.csv             # auxiliary loads (energy ledger)
```

### 9.1 `reference_cell.json` schema (raw-linked)

```jsonc
{
  "schema_version": "1.0",
  "run_id": "refcell-20260801-001",

  "reference_cell_spec": {
    "spec_version": "1.0",
    "configuration_id": "RC-1",
    "spec_sha256": "<64-hex of processes/reference_cell_spec.v1.json>",
    "spec_file": "processes/reference_cell_spec.v1.json"
  },

  "as_built_deviations": [
    {"path": "cell_stack.electrode_to_membrane_gap_mm",
     "as_built": 3.2, "authorized_by": "RC1-ops", "date": "2026-08-02"}
  ],

  "rectifier": {
    "applied_mode": "constant_current",
    "current_setpoint_A": 3.0,
    "current_density_mA_cm2_bound": [100, 300],
    "sync_timestamp_source": "DAQ-101",
    "trace_file": "timeseries.csv"
  },

  "samples": {
    "before": [
      {"sample_id": "RC1-001-BC", "timestamp_s": 0.0, "loop": "catholyte",
       "sample_point": "SP-101", "file": "sample_log.csv"}
    ],
    "during": [{"sample_id": "RC1-001-D1", "timestamp_s": 1800.0, "loop": "catholyte", "...": ""}],
    "after":  [{"sample_id": "RC1-001-AC", "timestamp_s": 7200.0, "loop": "catholyte", "...": ""}]
  },

  "deposit_metrology": {
    "mass":            {"file": "mass_log.csv",            "required": true},
    "thickness_map":   {"file": "deposit/thickness_map.csv", "required": true},
    "composition":     {"file": "deposit/composition.csv",   "required": true},
    "morphology":      {"file": "deposit/morphology.csv",    "required": false},
    "porosity":        {"file": "deposit/porosity.csv",      "required": false},
    "adhesion":        {"file": "deposit/adhesion.csv",      "required": false},
    "hydrogen_content":{"file": "deposit/hydrogen.csv",      "required": false}
  }
}
```

### 9.2 Validation semantics

- `spec_sha256` must be a 64-hex digest and, when the canonical spec file is
  reachable, must **match the content hash** of that file (tamper-evident pin).
- `samples.before` and `samples.after` are **required** (at least one entry each,
  error if absent in a `complete` record); `samples.during` is required when a
  run claims drift/crossover.
- `deposit_metrology.mass/thickness_map/composition` are required; the remaining
  groups are required for a complete reference deposit record (warning otherwise).
- `as_built_deviations` is optional; when present each item must give a path, an
  `as_built` value, and authorization.
- A run that does not use the reference configuration simply omits
  `reference_cell.json`; the contract stays backward compatible with
  `beaker_galvanostatic` / `hull_cell` runs.

---

## 10. Change control and sign-off

- **Ban on in-place edits.** No value in spec v1.0 may be edited. A change is a
  new version.
- **Who signs:** named run authority on the board approves a spec version before
  it becomes the reference `configuration_id` an experimental campaign pins.
- **What a new version requires:** a new section in the version table below, a
  regenerated sha256, and a re-pin by every downstream campaign. Old spec files
  are retained (never overwritten); they remain the definition of historical runs.
- **Sign-off of v1.0 is gated** on the pre-procurement package in
  `docs/REFERENCE_CELL_DESIGN_BASIS.md` §5 (drawings, controlled BOM, measurement
  map, independent shutdown, vent/H₂ review).

### Version history

| Version | Date | Change | Status |
|---|---|---|---|
| 1.0 | 2026-08-05 | Initial frozen reference configuration (D1) | **FROZEN** — content-pinned (sha256 `44f69f3dc5…`); formal board sign-off per §10 recorded separately |

---

## Traceability

| Quantity | Canonical source |
|---|---|
| Geometry / stack / safety | `docs/REFERENCE_CELL_DESIGN_BASIS.md`, `processes/reference_cell_rc1.yaml` |
| Batch recipe & impurities | `docs/BATH_SPEC.md` |
| Deposit metrology | `docs/DEPOSIT_METROLOGY.md`, `docs/DATA_CONTRACT.md` |
| Data record & ledgers | `docs/DATA_CONTRACT.md`, `models/run_record.py` |
| Physical-property targets | `models/reference_cell_design.py`, `models/electrochemistry.py`, `models/gas_holdup.py` |
