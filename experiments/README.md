# Experiments

Lab notebooks, raw data, and experimental protocols for the four-phase research program.

## Structure

- `notebooks/` — Jupyter notebooks for data analysis and visualization
- `data/` — Raw and processed experimental data (CSV, HDF5)

## Experimental Phases

| Phase | Focus | Key Methods |
|-------|-------|-------------|
| I | Electrolyte Formulation & Voltammetry | CV, LSV, EIS, Tafel on RDE |
| II | Hull Cell & Galvanostatic Deposition | Hull cell, long-duration plating |
| III | Carbon & Alloy Co-Deposition | Composite plating, carburization |
| IV | Anode Durability & Closed-Loop | ALT, CSTR integration |

## Protocols

Experimental protocols will be documented in Jupyter notebooks with embedded metadata
(electrolyte composition, temperature, current density, electrode geometry, etc.).

The first executable Phase I analysis is `notebooks/phase1_voltammetry.py`. It
loads the canonical CSV schema, estimates scan rate, reports anodic/cathodic
extrema, applies baseline correction, and writes a polarization plot:

```bash
python experiments/notebooks/phase1_voltammetry.py path/to/run.csv
```
