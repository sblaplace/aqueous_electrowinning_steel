# Experiments

Lab notebooks, raw data, and experimental protocols for the four-phase research program.

## Structure

- `notebooks/` — Jupyter notebooks / executable analysis scripts
- `data/` — Raw and processed experimental data (CSV, HDF5)

## Experimental Phases

| Phase | Focus | Key Methods |
|-------|-------|-------------|
| I | Electrolyte Formulation & Voltammetry | CV, LSV, EIS, Tafel on RDE |
| II | Hull Cell & Galvanostatic Deposition | Hull cell, long-duration plating, dry-mass balance |
| III | Carbon & Alloy Co-Deposition | Composite plating, carburization |
| IV | Anode Durability & Closed-Loop | ALT, CSTR integration |

## Protocols

Experimental protocols will be documented in notebooks with embedded metadata
(electrolyte composition, temperature, current density, electrode geometry, etc.).

The first executable Phase I analysis is `notebooks/phase1_voltammetry.py`. It
loads the canonical CSV schema, estimates scan rate, reports anodic/cathodic
extrema, applies baseline correction, and writes a polarization plot:

```bash
python experiments/notebooks/phase1_voltammetry.py path/to/run.csv
```

## Phase II: Hull-Cell Current Screen and Gravimetric FE

`notebooks/phase2_hull_cell.py` joins a galvanostatic time/current export to a
pre/post-weighing record, calculates blank-corrected apparent gravimetric Fe
Faradaic efficiency, and writes the matching variable-gap Hull-panel current
screen:

```bash
python experiments/notebooks/phase2_hull_cell.py \
  --trace experiments/data/hull_cell_galvanostatic_template.csv \
  --gravimetry experiments/data/hull_cell_gravimetry_template.csv
```

For a real run, preserve the original instrument export and map a copy to the
canonical trace schema.  The panel map treats a straight angled cathode and
planar anode as locally parallel ohmic paths ($j\propto1/g$), normalized to the
applied current.  It is suitable for choosing or documenting coupon positions,
not for replacing a calibration of the physical cell: edge effects, shields,
anode geometry, kinetics, transport, bubbles, and conductivity gradients are
not represented.

### Minimum gravimetric QA procedure

1. Mask and record the exposed cathode area; record actual panel length, width,
   near gap, and far gap.
2. Clean, rinse, dry, cool in a desiccator if appropriate, and weigh the coupon
   before deposition. Use a stable, documented balance and record its
   uncertainty/readability.
3. Log time and current through the entire deposition. The canonical sign
   convention is **negative cathodic current**; explicitly select the other
   convention only if the instrument documents it.
4. Use a consistent post-plating rinse/dry-to-constant-mass procedure. Record
   the post-run coupon mass and, where used, a matched blank mass change.
5. Treat the result as **apparent gravimetric Fe FE** until composition and
   dry-deposit integrity are verified. Retained salts, oxides, codeposits, or
   moisture can inflate mass. Do not clip an FE above 100%; investigate it.
6. Pair the screening result with real microscopy/compositional data (SEM/EDS)
   when available; no synthetic SEM/EDS data are supplied by this repository.

See `data/README.md` for exact headers and units, and run the synthetic example
with `python -m models.run_hull_cell` to verify the complete toolchain.
