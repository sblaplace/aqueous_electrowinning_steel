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
| `co_deposition.py` | Phase III anomalous Fe–Ni kinetics and Guglielmi carbon incorporation screen + pulse-coupled pH recovery & transport enhancement (`run_at_current_pulsed`) |
| `mechanical_properties.py` | Phase III → structural: Hall-Petch grain-size, Ni solid-solution, Guglielmi C dispersion → YS/UTS/HV/elongation, grade mapping |
| `carburization.py` | Post-deposition gaseous/plasma carburization: Fickian finite-slab diffusion, case depth, Maynier HV, Hollomon-Jaffe tempering, energy & composite strength |
| `process_flow.py` | Process block-flow diagram generator (BFD: ore→leach→cell→wash→carburize→product + recycle/purge) + detailed variant |
| `anode.py` | OER/CER anode kinetics, bubble resistance, and full-cell voltage coupling |
| `closed_loop.py` | Phase IV charge-throughput anode wear, CSTR electrolyte balances, process costs and QA flags |
| `experimental_data.py` | Long-form experimental measurement loading, validation, and run summaries |
| `campaign.py` | Experimental run-manifest validation, traceability links, and QA report |
| `calibration.py` | QA-gated Phase-I LSV total-current calibration and EIS consistency fit, with traceable parameter reports |
| `characterization.py` | SEM/EDS, combustion, and XRD long-form record validation and composition QA summaries |
| `technoeconomic.py` | CAPEX/OPEX, levelized cost of iron, sensitivity analysis, route benchmarking |
| `scenarios.py` | Four literature-anchored operating scenarios |

## Drivers

```bash
python -m models.run_electrochemistry       # Pourbaix + kinetics figures & report
python -m models.run_technoeconomic         # Base-case techno-economics
python -m models.run_scenarios              # Scenario comparison
python -m models.run_transport              # Nernst-Planck migration analysis
python -m models.run_pulse                  # Pulse-reverse transient dynamics & comparison
python -m models.run_voltammetry            # Synthetic voltammetry sweep, Phase I analysis & Tafel fitting
python -m models.run_eis                    # Synthetic EIS spectrum & Randles equivalent-circuit fitting
python -m models.run_calibration             # QA-gated real-data LSV calibration (requires manifest/options)
python -m models.run_hull_cell              # Phase II angled-panel current screen + gravimetric FE example
python -m models.run_co_deposition          # Phase III Fe–Ni/carbon incorporation screen (DC + pulsed)
python -m models.run_mechanical_properties  # Phase III → mechanical: YS/UTS/HV/grade + process-flow diagrams
python -m models.run_carburization          # Post-deposition carburization: case depth, HV profile, energy
python -m models.run_closed_loop            # Phase IV durability + closed-loop CSTR example
python -m models.run_all                    # Full suite (12 steps) + master_report.json + dashboard
python -m models.run_all --quick            # Same but skips heavy pulse frequency sweep
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

## Phase IV Durability and Closed-Loop Notes

`closed_loop.py` couples an empirical coating-wear law to the existing anode
kinetics and a constant-volume ideal-CSTR balance. It tracks Fe consumption,
ligand decay/makeup, chloride, impurity accumulation, precipitation, coating
remaining, voltage drift, energy, purge, and variable costs. Operating-limit
violations remain visible as quality flags.

The default coefficients are synthetic screening assumptions. Anode wear must
be calibrated from charge-normalized accelerated-life measurements; the ideal
CSTR does not replace activity-coefficient/speciation, solids, or residence-time
distribution models. Use `phase4_durability_template.csv` and
`experiments/notebooks/phase4_closed_loop.py` to summarize real measurements.

## Experimental campaign traceability

`campaign.py` validates a CSV manifest that links every experimental `run_id`
to immutable raw export(s), mapped analysis data, a metadata JSON sidecar, and
characterization records. This protects the distinction between synthetic
screening output and real measurements, while providing a machine-readable QA
check before data are used to calibrate model parameters.

```bash
python -m models.campaign experiments/data/campaign_manifest.csv \
  --output experiments/data/campaign_qa_report.json
```

See `experiments/data/campaign_manifest_template.csv`,
`experiments/data/metadata_template.json`, and `experiments/data/README.md` for
the required fields and storage convention.

## Dependencies

See `requirements.txt` in the repository root. Create an isolated environment
when the system Python is externally managed:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
```
