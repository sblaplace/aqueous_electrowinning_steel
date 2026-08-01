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
| `hull_cell_inverse.py` | Inverse Hull-cell analysis: measured deposit-thickness profile → strip-wise local Fe FE, binned FE(j) calibration table, mass-closure check vs gravimetry, and an FE-space sigmoid fit `FE(j) = sigmoid(a + b·ln j)` |
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

### Transport, speciation and cell physics

| Module | Contents |
|--------|----------|
| `diffusion_layer_1d.py` | **The FE prediction engine.** Full Nernst-Planck film over Fe²⁺/H⁺/OH⁻/HSO₄⁻/SO₄²⁻/borate with fast homogeneous equilibria, Arrhenius diffusivities, computed surface pH and Fe(OH)₂ precipitation criterion. Outputs FE(j, T, C, δ, pH, buffer) and V(j) |
| `speciation.py` | Davies activity coefficients, HSO₄⁻/FeSO₄ ion pairing, conductivity |
| `cell_physics.py` | Unified solver chaining speciation → transport → cell voltage into one self-consistent operating point and window sweep |
| `membrane_transport.py` | Divided-cell membrane: crossover, ohmic drop, transport numbers, acid balance |
| `membrane_fouling.py` | Hermia fouling laws, flux decline, cleaning cycles, membrane replacement cost |
| `operating_window.py` | Feasible (j, T, C, pH) region from combined constraints |

### Cell, scale-up and architecture

| Module | Contents |
|--------|----------|
| `cell_architecture.py` | **Reactor-type screen.** Plate-and-frame, rotating cylinder, drum-and-strip, moving belt, fluidized bed compared on literature Sherwood correlations, practical/footprint current ceilings, harvest duty cycle, areal productivity, $/m² → $/annual tonne, and the kill-criterion-#3 affordability threshold |
| `adhesion_peel.py` | **Deposit release mechanics.** Hoffman intrinsic + hydrogen-effusion + thermal-mismatch residual stress, Hutchinson-Suo energy release rate, Dupré work of adhesion with thickness-confined plastic amplification and Rice-Wang hydrogen knockdown, Kendall peel force, web-tear and cohesive-failure criteria → peel window, substrate ranking, drum-and-strip branch verdict, and the coupon test that replaces the estimate |
| `internal_stress.py` | **Deposit internal stress.** Forward/inverse Stoney cantilever deflection, exact two-layer laminate finite-thickness correction, GUM uncertainty budget, mechanism decomposition (intrinsic Hoffman, hydrogen effusion, thermal mismatch), additive relief (saccharin/chloride), stress evolution σ(h) (local vs Stoney average), and the bent-strip coupon curvature protocol |
| `rde_levich.py` | **RDE kinetics/transport separation.** Levich limiting current i_lim = 0.62 z F D^(2/3) ω^(1/2) ν^(-1/6) C, diffusivity from a Levich slope, Nernst film thickness δ on the rotating disk, Koutecký–Levich kinetic-current correction, Fe + HER Tafel extraction from one RDE polarization set, synthetic simulator, and the measurement matrix/gate rules that calibrate `diffusion_layer_1d`'s boundary layer |
| `scale_up.py` | Primary/secondary current distribution, Wagner number, boundary-layer growth, thermal management, geometry optimization |
| `thermal_balance.py` | Joule heating vs cooling duty; steady-state electrolyte temperature |
| `pid.py` | Pilot P&ID generation (overview and detailed) |
| `dark_mill.py` | Site-level digital twin: physics-driven sizing, mass/energy balance, go/no-go assessment across site scenarios |

### Feed, impurities and product

| Module | Contents |
|--------|----------|
| `purification.py` | Cementation, hydrolysis, selective electrowinning and ion exchange for Cu/Ni/Zn removal; enforces the Cu < 0.1% hot-shortness spec |
| `impurity_codeposition.py` | Co-deposition of nobler impurities and their effect on deposit purity |
| `deposit_morphology.py` | Mullins-Sekerka dendrite onset, HER bubble disruption, nucleation regime → coherent film / dendrite / powder classification |
| `steel_grade.py` | Composition → AISI grade routing (1008–8620) |
| `tempering.py` | Andrews Ms, Koistinen-Marburger retained austenite, Hollomon-Jaffe tempering |
| `carbon_potential.py` | Gas carburizing atmosphere: CO/CO₂ Boudouard, CH₄/H₂, dew point, O₂ probe, Acm solubility |
| `hydrogen_embrittlement.py` | H uptake, diffusivity, bake-out kinetics, embrittlement index |
| `bath_startup.py` | Bath make-up, conditioning and startup sequence |

### Economics, uncertainty and program tooling

| Module | Contents |
|--------|----------|
| `pilot_costing.py` | Pilot-scale CAPEX/OPEX buildup |
| `supply_chain.py` | Centralized vs on-site deployment economics per feedstock and haul distance |
| `experimental_matrix.py` | Factorial DOE matrix generation |
| `pulse_optimization.py` | Pareto search over pulse waveform parameters |
| `calibration_pipeline.py` | End-to-end calibration from raw records to fitted parameters |
| `foil_calibration.py` | Foil thickness and O₂-probe calibration helpers |
| `plating_data.py` | Plating-run data structures and validation |
| `process_registry.py` | Loader/validator for `processes/candidates.yaml` — the flowsheet hypothesis registry |
| `process_gates.py` | Measurement-only gate engine: literature evidence never passes a gate |
| `transport_sensitivity.py` | **Saltelli-Sobol global sensitivity of the 1D diffusion-layer FE engine** over 10 experimental levers → ranked "which experiment to do next" (fixes the prior "sensitivity analysis of a fiction" flagged in `docs/RESEARCH_PROGRAM.md`) |
| `uncertainty/` | Parameter registry, Monte Carlo, Sobol sensitivity, Bayesian calibration |

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
python -m models.run_cell_architecture      # Reactor-type screen: productivity, $/m², kill criterion #3
python -m models.run_purification           # Cu/Ni/Zn removal train
python -m models.run_speciation             # Activity coefficients, ion pairing, conductivity
python -m models.run_thermal_balance        # Joule heating vs cooling duty
python -m models.run_operating_window       # Feasible (j, T, C, pH) region
python -m models.run_scale_up               # Current distribution, transport, thermal, geometry
python -m models.run_membrane_fouling       # Hermia fouling and cleaning cycles
python -m models.run_hydrogen_embrittlement # H uptake, diffusivity, bake-out
python -m models.run_pilot_costing          # Pilot CAPEX/OPEX
python -m models.run_monte_carlo            # Uncertainty propagation and sensitivity
python -m models.run_transport_sensitivity  # Sobol GSA of the FE engine -> which experiment next
python -m models.run_dark_mill              # Site-level sizing and go/no-go
python -m models.run_all                    # Full suite (17 steps) + master_report.json + dashboard
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

## Inverse Hull-Cell Notes (thickness profile → FE(j) calibration)

`hull_cell_inverse.py` turns the measured deposit-thickness profile of a
Hull panel into a **local FE-vs-j calibration** in one step.  Each strip's
thickness is converted to the charge that must have produced it (at 100 % Fe)
and divided by the charge the primary map assigns to that strip:

\[
\mathrm{FE}(s) = \frac{h(s)\,\rho_\mathrm{Fe}\,nF/M}{t_\mathrm{run}\,
j_\mathrm{primary}(s)}.
\]

Three checks keep the result honest:

1. **Mass closure.**  The profile-integrated iron mass must agree with the
   weighed mass gain of the same panel (default tolerance ±15 %; profilometry
   on a rough deposit reads high/low easily).  A mismatch is resolved before
   the FE map is trusted.
2. **Gravimetric identity.**  The current-weighted mean of the local FE
   profile *equals* the whole-panel gravimetric FE — the profile and the
   weighing describe the same panel or they do not.
3. **Tafel-consistent shape.**  With Fe and HER both in the Tafel region,
   `FE(j) = sigmoid(a + b·ln j)` with `b = 1 - α_H/α_Fe < 0`.  The fit is
   performed by nonlinear least squares **in FE space**: least squares on the
   logit transform is biased under measurement noise because logit is convex
   for FE > 0.5, and noisy high-FE (thin-deposit) strips dominate the mean.
   Monte-Carlo round trips show a 10-strip point-micrometer profile (σ_h =
   2 µm) recovers FE at a reference current density to ~±2 % but only the
   slope to ~±0.2; profilometry-grade noise (σ_h = 0.5 µm) brings the slope
   to ~±0.05.  In protocol terms: one coarse panel pins FE(j) points, not the
   slope.

FE above 100 % is retained as a `above_100` QA flag (retained salts/oxides),
never clipped; `zero_deposit` strips are flagged and excluded from the fit.
`run_hull_cell_inverse.py` demonstrates the full pipeline on a synthetic
Day-1-style panel (2 A, 10 × 5 cm, 1.5 → 9 cm gap) and writes a JSON report
plus a recovery figure.

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
