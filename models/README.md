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
| `pulse.py` | Transient 1-D diffusion-kinetics model for **pulsed (PE) and pulse-reverse (PRE)** electrodeposition |
| `tafel.py` | Tafel-region fitting with exchange-current and R² estimates |
| `voltammetry.py` | Phase I CV/LSV analysis, scan rate estimation, baseline correction, polarization curves |
| `eis.py` | Phase I EIS: Randles/CPE/Warburg equivalent circuits, complex NLLS spectrum fitting, Rct→exchange-current conversion |
| `hull_cell.py` | Phase II angled-panel primary current-density screen, galvanostatic trace loading, and gravimetric apparent Fe Faradaic efficiency |
| `experimental_data.py` | Long-form experimental measurement loading, validation, and run summaries |
| `technoeconomic.py` | CAPEX/OPEX, levelized cost of iron, sensitivity analysis, route benchmarking |
| `scenarios.py` | Four literature-anchored operating scenarios |

## Drivers

```bash
python -m models.run_electrochemistry   # Pourbaix + kinetics figures & report
python -m models.run_technoeconomic     # Base-case techno-economics
python -m models.run_scenarios          # Scenario comparison
python -m models.run_transport          # Nernst-Planck migration analysis
python -m models.run_pulse              # Pulse-reverse transient dynamics & comparison
python -m models.run_voltammetry        # Synthetic voltammetry sweep, Phase I analysis & Tafel fitting
python -m models.run_eis                # Synthetic EIS spectrum & Randles equivalent-circuit fitting
python -m models.run_hull_cell          # Phase II angled-panel current screen + gravimetric FE example
```

## Transport Model Notes

`transport.py` supersedes the linear stagnant-film closure in `boundary_layer.py`
for local-composition questions. It tracks Fe²⁺, H⁺, OH⁻, Na⁺ and SO₄²⁻ with

    N_i = -D_i ∇C_i - z_i D_i (F/RT) C_i ∇φ

closed by pointwise electroneutrality (differentiated to give ∇φ explicitly) and
fast water autoprotolysis.

## Pulse-Reverse Transient Model Notes

`pulse.py` models transient 1D diffusion-reaction dynamics under pulsed (PE) and
pulse-reverse (PRE) waveforms ($j_\text{cathodic}$, $t_\text{cathodic}$, $j_\text{anodic}$, $t_\text{anodic}$, $t_\text{off}$).
During pulse-off and reverse-pulse intervals, surface Fe²⁺ depletion relaxes and local
surface pH spikes are mitigated, preventing Fe(OH)₂ precipitation and allowing higher peak
plating current densities than steady DC electrodeposition.

## Phase II Hull-Cell and Gravimetric FE Notes

`hull_cell.py` makes a **primary-current, variable-gap** map for a straight
angled cathode opposite a planar anode.  For panel coordinate $s$, it assumes
narrow strips are parallel ohmic paths through the local gap $g(s)$:

\[
j(s) \propto \frac{1}{g(s)}, \qquad \int_A j(s)\,dA = I_{\mathrm{applied}}.
\]

The second condition is enforced exactly across the returned strips.  It is a
transparent screening map for locating coupon/current-density windows, **not**
a calibrated prediction of a particular commercial Hull cell.  It omits edge
and shielding effects, anode shape, activation/secondary distribution, mass
transfer, bubbles, and solution-conductivity gradients.  Record the actual
cell dimensions and, where local current is consequential, calibrate against
a known bath or segmented-coupon mass gain.

The same module integrates only the cathodic portion of a galvanostatic trace
(the repository convention is negative cathodic current) and calculates:

\[
\mathrm{FE}_{\mathrm{app}} =
\frac{m_{\mathrm{after}} - m_{\mathrm{before}} - m_{\mathrm{blank}}}
{Q_{\mathrm{cathodic}} M_{\mathrm{Fe}}/(2F)}.
\]

This is **apparent gravimetric Fe FE** until the dry deposit composition has
been verified.  Retained electrolyte, oxides, codeposits, and incomplete drying
can give an apparent value above 100%; the API intentionally leaves that value
visible as a quality-control signal.  Optional balance and charge uncertainties
are propagated to the reported FE uncertainty.

## Dependencies

See `requirements.txt` in the repository root.
