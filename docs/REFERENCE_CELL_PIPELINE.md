# RC-1 reference-cell pipeline

`models/reference_cell_pipeline.py` is the integration boundary for the
reference-cell program. It connects existing modules without turning the
repository into one giant simulator.

## Data flow

```text
reference_cell_rc1.yaml + runtime inputs
    │
    ├─ CellPhysics
    │    speciation → Nernst–Planck transport → FE, surface chemistry, V_cell
    │
    ├─ gas_holdup.solve_coupled
    │    HER fraction → H₂ → void fraction → conductivity/current distribution
    │                 ↘ bubble microconvection → effective boundary layer → FE
    │
    ├─ thermal_balance.simulate_thermal_transient
    │    coupled V × I → heat generation → temperature/cooling duty
    │
    ├─ charge / iron / energy screening ledgers
    │
    └─ OperatingTwin (advisory replay / safety request only)
```

The canonical state is the typed `ReferenceCellState`. Its branches are
intentionally separate:

- `predicted`: model outputs, always labelled screening/L0;
- `observed`: validated `run_record` metrics, ledgers, and sensor snapshots;
- `safety`: replay through `OperatingTwin`, advisory only;
- `gates`: `process_gates` verdicts from explicitly declared experimental
evidence only;
- `calibration`: residuals and the next calibration action, never an implicit
parameter update.

## Commands

Run the screening pipeline from the RC-1 design basis:

```bash
python -m models.reference_cell_pipeline
```

Useful overrides:

```bash
python -m models.reference_cell_pipeline \
  --j 100 --flow 0.25 --temperature 60 \
  --out experiments/data/reference_cell_state.json
```

Ingest and replay a measured run directory:

```bash
python -m models.reference_cell_pipeline \
  --run-dir path/to/run-directory \
  --out path/to/reference_cell_state.json
```

After installation, the equivalent entry point is:

```bash
aq-steel-reference-cell-pipeline --run-dir path/to/run-directory
```

## Measurement boundary

A complete run is loaded through `models.run_record`. Missing or invalid data
is reported as pending; model values are not substituted for missing
measurements. For a complete run, the pipeline:

1. maps the measured current, temperature, Fe(II), and pH into a prediction
   condition when those fields exist;
2. runs the same coupled model chain at the measured condition;
3. reports residuals for apparent FE, cell voltage, and apparent product energy
   where the required measurements exist;
4. preserves the run-record's charge, iron, and auxiliary-energy ledger status;
5. replays synchronized trace points through the safety twin; and
6. evaluates declared gate evidence through `models.process_gates`.

Residuals are calibration inputs, not gate evidence. No parameter is fitted or
written automatically. The next calibration step must use replicated runs and
held-out conditions, with the fitted parameter set versioned separately.

## Safety boundary

The pipeline never arms actuation. `OperatingTwin` can return an advisory
command or a `ShutdownRequest`, but the request explicitly says
`request_only; independent_channel_executes`. Rectifier disable, gas isolation,
and emergency shutdown remain responsibilities of the independent hardwired
safety channel described in `docs/INDEPENDENT_SHUTDOWN.md`.

## What is deliberately not coupled

Carburization, tempering, structural-grade screening, site economics, and
whole-system crate/site assessment remain downstream branches. They should
consume a qualified deposit/process result rather than feed back into the
first RC-1 electrochemical solve. This keeps the first build/no-build gate
about measured iron deposition, FE, voltage decomposition, balance closure,
and component stability.

All integrated outputs are L0 screening results until the RC-1 acceptance
criteria in `docs/NEXT_STEPS.md` are met.
