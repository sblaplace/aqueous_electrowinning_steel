# Models

Electrochemical and process simulation code for aqueous electrowinning of iron/steel.

## Implemented Modules

| Module | Contents |
|--------|----------|
| `electrochemistry.py` | Constants, Faraday's law, cell-voltage decomposition, specific energy |
| `pourbaix.py` | Fe–H₂O potential–pH equilibria, hydrolysis boundaries, HER/OER water window, HER thermodynamic margin |
| `kinetics.py` | Butler–Volmer / Tafel partial currents for Fe deposition vs. HER, Koutecký–Levich mass-transport limit, galvanostatic current efficiency |
| `boundary_layer.py` | Steady cathode film: local pH, Fe²⁺ depletion, Fe(OH)₂ precipitation, concentration profiles |
| `technoeconomic.py` | CAPEX/OPEX, levelized cost of iron, sensitivity analysis, route benchmarking |
| `scenarios.py` | Four literature-anchored operating scenarios |

## Drivers

```bash
python -m models.run_electrochemistry   # Pourbaix + kinetics figures & report
python -m models.run_technoeconomic     # Base-case techno-economics
python -m models.run_scenarios          # Scenario comparison
```

## Still Planned

- **Nernst-Planck transport** — replace the current steady-film approximation with full 1-D concentration profiles and migration
- **Pulse-reverse electrodeposition** — transient deposition modeling

## Dependencies

See `requirements.txt` in the repository root.
