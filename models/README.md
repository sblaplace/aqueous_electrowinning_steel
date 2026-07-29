# Models

Electrochemical and process simulation code for aqueous electrowinning of iron/steel.

## Implemented Modules

| Module | Contents |
|--------|----------|
| `electrochemistry.py` | Constants, Faraday's law, cell-voltage decomposition, specific energy |
| `pourbaix.py` | Fe–H₂O potential–pH equilibria, hydrolysis boundaries, HER/OER water window, HER thermodynamic margin |
| `kinetics.py` | Butler–Volmer / Tafel partial currents for Fe deposition vs. HER, Koutecký–Levich mass-transport limit, galvanostatic current efficiency |
| `boundary_layer.py` | Steady cathode film: local pH, Fe²⁺ depletion, Fe(OH)₂ precipitation, concentration profiles |
| `transport.py` | Steady 1-D Nernst–Planck film: diffusion **+ migration**, electroneutral multi-ion profiles, migration-corrected limiting current, diffusion potential |
| `technoeconomic.py` | CAPEX/OPEX, levelized cost of iron, sensitivity analysis, route benchmarking |
| `scenarios.py` | Four literature-anchored operating scenarios |

## Drivers

```bash
python -m models.run_electrochemistry   # Pourbaix + kinetics figures & report
python -m models.run_technoeconomic     # Base-case techno-economics
python -m models.run_scenarios          # Scenario comparison
python -m models.run_transport          # Nernst-Planck migration analysis
```

## Still Planned

- **Pulse-reverse electrodeposition** — transient deposition modeling
- **Experimental data tooling** — voltammetry parsers and Tafel/i₀ extraction from lab data

## Transport Model Notes

`transport.py` supersedes the linear stagnant-film closure in `boundary_layer.py`
for local-composition questions. It tracks Fe²⁺, H⁺, OH⁻, Na⁺ and SO₄²⁻ with

    N_i = -D_i ∇C_i - z_i D_i (F/RT) C_i ∇φ

closed by pointwise electroneutrality (differentiated to give ∇φ explicitly) and
fast water autoprotolysis, so the conserved proton variable is S = C_H⁺ − C_OH⁻
with N_S = −i_HER/F. Validation: an unsupported binary FeSO₄ bath reproduces the
exact analytical enhancement i_lim = 2·i_Levich for a symmetric 2:2 salt, and
heavy supporting electrolyte recovers the pure-diffusion Levich limit.

The two models disagree sharply on surface pH in acid, and the Nernst–Planck
answer is the physical one: the film model has no mechanism to resupply protons,
so it predicts a cathode surface at pH ≈ 11.5 in a pH-2 bath, whereas including
H⁺ diffusion and migration keeps the surface near pH 2.7 at 100 mA/cm².

## Dependencies

See `requirements.txt` in the repository root.
- `tafel.py` | Tafel-region fitting with exchange-current and R² estimates
