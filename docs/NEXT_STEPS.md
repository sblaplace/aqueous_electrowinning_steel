# Next steps: from screening suite to a predictive physical twin

**Date:** 2026-08-01  
**Purpose:** turn the existing model collection into a model that can support a build/no-build decision and then operate the first physical cell safely.

## The standard we should use

"If the twin works, the one we build will work" is not one claim. It is a chain of claims:

1. the feed and electrolyte are what we think they are;
2. the cell produces the predicted local current, voltage, temperature, gas and flow fields;
3. the electrochemistry produces the predicted Fe/HER split and deposit rate;
4. the deposit can be harvested and has the predicted composition and quality;
5. those quantities remain true over time, through impurities, membrane ageing and anode wear;
6. the balance of plant closes on mass, charge, heat, gas, waste and energy.

The repository is strong on screening hypotheses and test plumbing. It is not yet a calibrated digital twin: there is no real wet-lab dataset in the repository, and many downstream material, durability and cost coefficients are explicitly screening assumptions.

The next milestone should therefore be **one instrumented, divided, recirculating reference cell with a fully specified geometry and a versioned data record**, not another broad model.

## Recommended product decision

Keep **iron flake/powder as the primary first product**. It removes carbon control, hydrogen embrittlement, thick crack-free deposition and near-net-shape release from the first go/no-go. Keep foil/drum work as a cheap parallel branch only: the iron-on-titanium peel coupon can close or preserve that branch at very low cost.

The first process target should be a qualified iron feedstock for a melt shop, not structural steel made directly in the bath.

## Work in priority order

### 1. Build the reference cell and create the first real dataset

Use the existing Day-1 packet, but make the reference configuration explicit and immutable:

- divided cell, membrane type and active area;
- cathode/anode materials, spacing, roughness and exposed area;
- electrolyte batch recipe, impurities, density, conductivity, viscosity and temperature;
- flow rate, mixing, gas handling and thermal boundary conditions;
- rectifier voltage/current and synchronized timestamps;
- catholyte/anolyte samples before, during and after the run;
- dry deposit mass, thickness map, composition, morphology, porosity, adhesion and hydrogen content.

Every run must close **three ledgers independently**:

- charge ledger: applied charge = Fe + HER + other measured products;
- iron ledger: Fe in = dissolved Fe out + deposit Fe + solids/precipitate + analytical uncertainty;
- energy ledger: DC stack energy + pumps + heating + gas handling + drying.

Do not use a model-generated value as gate evidence. The existing `campaign.py` and `process_gates.py` are the right foundation; the next step is populating them with raw-linked runs.

### 2. Close the highest-value physical unknowns

Run these before optimizing waveforms or building a pilot:

1. **Hull-cell morphology map:** temperature × Fe concentration × pH/buffer × current density. This finds the usable deposit window cheaply.
2. **Divided-cell polarization and FE:** replicate FE(j), V(j), and local pH at three temperatures and three Fe concentrations. Measure hydrogen volumetrically and verify deposit Fe by digestion/ICP or equivalent.
3. **RDE/Levich:** identify the transport coefficient and separate Fe kinetics from HER on the actual bath and cathode surface.
4. **Membrane/anode test:** crossover, area resistance, ferric generation, gas composition, voltage drift and post-run inspection.
5. **Adhesion/release coupon:** plate iron on candidate titanium release surfaces and measure peel/strip force. This is a branch decision, not a material-properties project.
6. **Stress and hydrogen:** bent-strip curvature plus hydrogen uptake/bake-out on representative deposits.

The first three calibrate the cathode model. The fourth determines whether the proposed cell architecture is chemically viable. The last two protect against building a machine that makes unharvestable or unsafe material.

### 3. Upgrade the model only where measurements show model error

#### 3.1 Cathode electrochemistry and electrolyte

Replace default constants with fitted, uncertainty-bearing functions for:

- Fe and HER exchange currents and Tafel slopes versus temperature, pH, Fe activity and surface state;
- activity coefficients and complexation for the selected sulfate/chloride/ligand bath;
- conductivity and viscosity versus composition and temperature;
- Fe(II)/Fe(III), hydrolysis, precipitation and impurity speciation;
- gas evolution and hydrogen absorption partition.

The current 1-D diffusion-layer model should remain the gating model. It becomes predictive when its boundary conditions are fitted to divided-cell and RDE data. Do not jump to phase-field or 3-D CFD before this calibration fails systematically.

#### 3.2 Cell voltage and current distribution

Make `V_cell` a measured, spatially resolved result rather than a single assumed number. Add a geometry-specific electrical model for the reference cell:

- electrolyte, membrane, contacts and busbar resistances;
- primary/secondary current distribution from actual electrode geometry;
- contact resistance and temperature-dependent conductivity;
- anode OER/CER/ferric competition and gas void fraction;
- membrane area resistance and crossover as functions of current, temperature and ageing.

Validate against segmented cathodes, reference electrodes, voltage taps and gas analysis. Only then extrapolate to drum, belt or rotating-cylinder geometry.

#### 3.3 Hydrodynamics, bubbles and thermal balance

**Update (2026-08-02):** the reduced-order two-phase model this section asked
for now exists — `models/gas_holdup.py` (`aq-steel-gas-holdup`). It closes the
last unmodelled field in the cell: cathodic H₂ generation from `1 − FE`, a
drift-flux void-fraction profile up the vertical channel, Bruggeman effective
conductivity, equipotential current redistribution over electrode height,
Stephan-Vogt bubble microconvection feeding a thinner `δ_eff` back into the
1-D diffusion-layer FE engine, and a headspace LFL/dilution calculation.

Three screening results, all L0:

1. **Hold-up is not an RC-1 problem.** At the 300 mA/cm² kill criterion the
   50 mm channel reaches ~1.2 % outlet void fraction, <1 % ohmic penalty and
   ~1 % axial current spread.
2. **It is a scale-up problem.** Holding current density fixed and growing the
   electrode, axial uniformity crosses the 0.90 floor near 0.5 m of height and
   reaches 0.81 at 1 m. RC-1 cannot observe this; the geometry-transfer
   experiment in §4 must.
3. **Bubbles are net-favourable for FE and net-unfavourable for voltage.**
   Self-stirring thins the diffusion layer faster than blanketing and
   redistribution remove FE, so at bench scale the coupled solve returns
   *higher* FE than the bubble-free baseline (+1.4 pp at 300 mA/cm²). Ignoring
   bubbles is therefore conservative for FE but **not** conservative for
   `V_cell` — which is the term the energy number is most sensitive to.

The dominant uncertainty is bubble departure diameter (it enters rise velocity
quadratically) and the surface-coverage ceiling. Both are cheap to measure;
`gas_holdup.measurement_protocol()` specifies the ~$450, 3-day experiment.

For the reference cell, measure flow and gas hold-up and calibrate the model for:

- local boundary-layer thickness;
- bubble coverage and detachment;
- mixing and dead zones;
- heat generation, evaporation and cooling;
- transient startup and shutdown.

Higher-fidelity simulation is a pursued, resourced track — not something to
avoid. The reduced-order two-phase model above is the fast, screening layer and
stays valuable; but CFD / FEM for the flow field, two-phase gas-liquid
transport, and current/thermal distribution in the target geometry are things
we intend to build out, dedicating GPU/accelerator time to them. The physics
thesis should be backed by the best simulation we can afford *before* the
physical build, not after. Use the reduced-order model to scope where
fidelity matters most, then resolve those regions with resolved simulation;
validate any resolved-simulation claim against the physical cell once it
produces data, exactly as the reduced-order models are validated.

#### 3.4 Deposit and harvesting

Connect electrochemical outputs to product acceptance:

- thickness and mass uniformity;
- Fe, C, Ni, Cr, Cu, Zn, S, O and H content;
- porosity, inclusions, roughness and phase/texture;
- residual stress and adhesion/release force;
- flake size distribution and downstream melt-shop behavior.

For the feedstock path, optimize clean, recoverable iron rather than tensile strength. Keep the existing mechanical, carburization and tempering modules as a later product-path branch, not as evidence for the first route.

#### 3.5 Dynamic plant loop

After a stable single-pass cell is measured, add the real recirculation loop:

- reservoir volume and residence-time distribution;
- Fe consumption and dissolution/makeup;
- acid/base and ligand balance;
- impurity accumulation and purge;
- membrane crossover and anolyte/catholyte separation;
- filter solids, wash water and drying;
- anode coating wear and voltage drift.

The current CSTR/closed-loop models are useful scaffolds, but their synthetic wear and impurity laws must be replaced by run data.

## Model credibility ladder

| Level | What the model may claim | Evidence required |
|---|---|---|
| 0 — screening | trends, bounds, experiment ranking | transparent assumptions and unit tests |
| 1 — calibrated coupon | FE, V, deposit rate for one bath/cell | replicated runs, charge/mass closure, held-out conditions |
| 2 — reference-cell twin | spatial current, temperature, gas, composition and quality over a run | synchronized sensors, segmented cathodes, independent analyses |
| 3 — durability twin | drift and recovery over many charge-throughput cycles | accelerated life, membrane/anode inspection, impurity challenge |
| 4 — pilot design model | scale-up predictions within a stated uncertainty envelope | geometry-transfer experiment and pilot acceptance test |
| 5 — operating twin | constrained online estimation/control | validated sensors, fault injection, independent shutdown |

The current repository is mostly Level 0, with a good path to Level 1. We should not label the whole suite a digital twin until the reference cell reaches Level 2.

## Acceptance criteria for the first predictive twin

Before using the model to freeze pilot hardware, require:

- at least three independent runs at each calibration condition;
- held-out-condition prediction of FE, cell voltage, deposit rate and temperature within predeclared tolerances;
- charge, iron and energy balances reported with uncertainty and no unexplained closure failure;
- no systematic residual versus current density, temperature, Fe concentration, time or batch;
- membrane crossover and anode gas composition measured, not inferred;
- deposit composition and morphology predicted as distributions, not just a mean;
- a failure-mode table with detection signal, abort limit and recovery action;
- an uncertainty budget that identifies what still controls the go/no-go decision.

Suggested initial tolerances, to be tightened after metrology is qualified:

- FE: ±5 percentage points;
- cell voltage: ±0.10 V;
- deposit mass/rate: ±5%;
- bulk temperature: ±2 °C;
- iron balance closure: ±5%;
- membrane crossover: ±20% of measured value.

These are acceptance targets for a model against a defined apparatus, not universal claims about the chemistry.

## Explicitly defer

Do not spend the next cycle on DFT, plant-wide digital-twin infrastructure,
broad supply-chain modeling, or autonomous PID optimization. They can be
valuable later, but none resolves the present dominant uncertainty: whether a
divided physical cell can sustain the required FE, voltage, deposit quality
and balance closure with real feed and real gas/membrane behavior.

Resolved flow / transport simulation (CFD, FEM) is **not** deferred — it is a
pursued track (§3.3): we intend to build it out and dedicate GPU/accelerator
time to it, using the reduced-order models to scope where fidelity matters and
validating resolved-simulation claims against the physical cell once it
produces data. What *is* deferred is jumping straight to plant-wide or
autonomous-control infrastructure before a measured cell exists to support it.

## Decision gates

1. **Chemistry gate:** coherent/recoverable Fe deposit and reproducible FE at the reference condition.
2. **Energy gate:** measured AC/DC specific energy meets the route threshold at the target current density.
3. **Architecture gate:** continuous harvest or an explicitly accepted batch duty cycle; peel/release branch decided.
4. **Loop gate:** membrane, anode, impurity and recycle balances close over a multi-day campaign.
5. **Scale gate:** geometry-transfer test reproduces the calibrated twin within its uncertainty envelope.

If Gate 2 fails, redesign chemistry or switch comparator route. If Gate 3 fails, stay with flake/feedstock or change architecture. If Gate 4 fails, do not call the process continuous. If Gate 5 fails, the twin is not transferable and the pilot should not be frozen.

## Immediate next action

Order/assemble the reference divided cell and run the Day-1 campaign; in parallel, run the titanium adhesion/release coupon. The next repository changes should be driven by the first raw run files: schema adapters, calibration fixtures, uncertainty propagation from measurement data, residual dashboards and gate evidence—not more synthetic output.
