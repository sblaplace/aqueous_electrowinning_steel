# Aqueous Electrowinning Steel — Research Program

## The Question

Can we make structural steel by electrodeposition — bypassing the blast furnace entirely?

The electrochemistry is known. The thermodynamics are known. The heat treatment is centuries old. What's unknown is whether you can put them together into a process that produces dense, ductile, carbon-bearing steel at a rate and cost that matters.

This plan defines the hard problems, what we can learn from computation, and what ultimately needs a beaker.

---

## The Real Competitor

BOF steel ($300–500/t) is the price benchmark. But the real competitor is **DRI-H2** — hydrogen-based direct reduced iron. HYBRIT is at pilot scale now, uses existing downstream metallurgy, and the iron-making step is already solved at temperature with well-understood kinetics.

The question isn't "can we beat a blast furnace?" It's "can we beat DRI-H2 at the margin?" The advantages of aqueous electrowinning are:

- **Modularity.** Add cells, not blast furnaces. Scale by replication.
- **Intermittent power.** Electrowinning tolerates power cycling in ways that high-temperature processes don't. Pair with wind/solar directly.
- **Lower capital.** No reduction shaft, no gas handling, no pelletizing.
- **Scrap-compatible feedstock.** FeSO₄ from pickling waste, TiO₂ byproduct.

These are real but need quantification. The sensitivity model must answer: **at what electricity price and current density does aqueous electrowinning beat DRI-H2?** If the answer is "never within plausible parameters," the rest is academic.

---

## The Hard Problems

### 1. Dense Deposits at Production Current Density (GATING)

At 100 mA/cm² we get beautiful iron — 3.5 µm grains, 367 MPa yield strength, 84% Faradaic efficiency. At 500 mA/cm² we get 52% FE and porous garbage. The gap between "works in a beaker" and "works at production rate" is entirely about pushing that curve to the right.

**Unknown:** Can pulse or pulse-reverse plating recover deposit density at >300 mA/cm²? Our model says yes (pH recovery, enhanced nucleation). Nobody has demonstrated it experimentally for iron.

**Computational approach:** Phase-field microstructure evolution during deposition. Simulate grain nucleation, growth, and competition under DC vs PE vs PRE waveforms. Map the j–duty–frequency space to find the boundary between dense and porous regimes.

**Kill criterion:** If phase-field shows FE < 75% at j > 400 mA/cm² for any pulse waveform, the pathway to cost parity with DRI-H2 closes. Pivot or kill.

### 2. Carbon via Carburization (NOT on critical path)

Structural steel needs 0.1–0.8 wt% carbon. The original plan focused on electrolytic co-deposition of carbon particles (Guglielmi mechanism). But **pack carburization of electrodeposited pure iron is the lower-risk path.** A 1 mm deposit carburized at 925°C reaches 0.2 wt% C at the center in roughly 30 minutes (D ≈ 2×10⁻¹¹ m²/s). Diffusion distances in thin deposits are short.

This eliminates carbon co-deposition from the critical path entirely. Co-deposition work (A4, B4) remains scientifically interesting and is a process intensification option, but the primary carbon pathway is post-deposition carburization.

**Unknown:** Does the ultrafine-grained structure survive carburizing temperatures? (See Problem 5.)

### 3. Hydrogen Embrittlement (MANAGEABLE with bake-out)

Every ampere that doesn't go to iron deposition produces hydrogen. At 84% FE, 16% of current is HER. Some hydrogen absorbs into the deposit. For deposits above ~500 MPa yield strength, even a few ppm of diffusible hydrogen causes catastrophic brittle fracture.

However, hydrogen bake-out is standard practice — 4 hours at 200°C is well-proven for hard chromium and other electrodeposits. The question is whether bake-out is sufficient for our deposits, or whether the ultrafine grain boundary density creates traps that resist desorption at 200°C.

**Unknown:** What is the critical diffusible hydrogen concentration for ultrafine-grained electrodeposited iron? What bake-out conditions are needed?

**Computational approach:** Ab initio hydrogen trap binding energies at grain boundaries, dislocations, and carbide interfaces. Diffusion–trapping FEM to predict H profiles and bake-out kinetics.

**Key coupling (from feedback):** Carbon particles and carbides are H-trap sites. High carbon loading may simultaneously solve grain stability (Zener pinning) and worsen hydrogen retention (more interfaces). The DFT scope for H trapping must include Fe/C interfaces explicitly, not just GBs and dislocations.

### 4. Thick, Stress-Free Deposits (GATING)

This is the critical path. If you can't make a dense, stress-free deposit above ~1 mm at production rates, nothing else matters:

- Carbon can be added by carburization (Problem 2 — solved)
- Hydrogen can be baked out (Problem 3 — likely solved)
- Grain size can be managed by controlled heat treatment (Problem 5)
- But thick, crack-free deposits at high current density is the gate

Electrodeposited iron has high internal stress from grain growth constraints, hydrogen incorporation, and lattice mismatch. At 10 µm it's manageable. At 1 mm (structural thickness), it cracks or delaminates.

**Unknown:** What is the stress evolution during deposition as a function of thickness, current density, and pulse parameters? Can we predict and prevent cracking?

**Computational approach:** FEM of stress evolution during deposition — couple the grain-scale microstructure (from phase-field) to continuum mechanics. Include hydrogen-induced swelling.

**Kill criterion:** If stress modeling shows no path to 1 mm without cracking at any j > 100 mA/cm², the process cannot achieve competitive throughput. Pivot to thin-film applications or kill.

### 5. Grain Stability During Heat Treatment

Electrodeposited iron with 1–5 µm grains is **ultrafine-grained**, not nanocrystalline. This distinction matters:

- **Ultrafine-grained (1–5 µm):** Moderate Hall-Petch strengthening. Grain growth at 900°C is slow enough that carburizing times may be tolerable without Zener pinning.
- **Nanocrystalline (<100 nm):** Extreme strengthening but catastrophic grain growth at 900°C — coarsens in minutes. Zener pinning becomes essential. May also hit inverse Hall-Petch.

If our deposits are 3.5 µm (as the model predicts at 100 mA/cm²), grain stability is a concern but not a crisis. If pulse plating pushes grains below 500 nm, Zener pinning becomes critical.

**Unknown:** Does carbon pin grain boundaries during carburizing? How fast do grains grow in the 850–950°C range?

**Computational approach:** Phase-field grain growth with carbon diffusion coupling. Monte Carlo Potts for Zener pinning. Map the T–t–C space.

### 6. Cost Parity with DRI-H2

DRI-H2 steel: $600–1000/t (depending on H2 price). Our model: $800–2000/t. The gap is smaller than vs BOF, and the competitive analysis must use DRI-H2 as the benchmark:

**Kill criteria:**
- If electricity > $0.05/kWh AND current density < 300 mA/cm² at >80% FE → no path to parity
- If deposit thickness < 0.5 mm without cracking → insufficient for structural applications
- If hydrogen bake-out does not reduce diffusible H below critical threshold → HE risk unacceptable

Each kill criterion maps to a specific computational task. If Phase A (microstructure) shows the j–FE curve cannot be pushed right, stop before investing in Phases B–E.

---

## Computational Research Program

### Tiering

Not all tasks are equal. The computational program is triaged by criticality:

**Tier 1 — Gating. Do these first. They determine whether the rest is worth doing.**

| Task | Method | Answers |
|------|--------|---------|
| A1. Grain nucleation/growth | Phase-field (JAX/GPU) | Can pulse plating push the j–FE curve? |
| A3. Stress evolution | FEM coupled to microstructure | Can we make 1 mm deposits without cracking? |
| D2. H diffusion–trapping | FEM (FEniCS/JAX) | Is bake-out sufficient? |

Run A1 and B2 (HER mechanism) simultaneously — they share parameter space (j, pH, surface coverage). If B2 shows H coverage dominates nucleation suppression, A1's enhanced nucleation mechanism may need revision.

**Tier 2 — Do these if Tier 1 is encouraging.**

| Task | Method | Answers |
|------|--------|---------|
| C1. Grain growth during carburizing | Phase-field + diffusion | Do grains survive 900°C? |
| B1. Cell CFD for scale-up | OpenFOAM / analytical | Does the boundary layer thicken at pilot scale? |
| E3. Techno-economic with calibrated params | Monte Carlo | What's the real cost distribution? |
| A2. Deposit porosity | Phase-field + bubble model | Porosity vs j, FE |

**Tier 3 — Scientifically interesting but not on critical path.**

| Task | Method | Answers |
|------|--------|---------|
| A4. Carbon particle–matrix interface | DFT | Co-deposition mechanism (not needed if carburization is primary path) |
| B2. HER mechanism | DFT + microkinetic | Tafel slopes, H coverage (run in parallel with A1) |
| B3. Fe–Ni anomalous co-deposition | kMC | Ni content control |
| B4. Carbon particle transport | CFD + Lagrangian | Co-deposition transport (not on critical path) |
| C2. Zener pinning | MC Potts | Only needed if grains < 500 nm |
| C3. Quench cracking | Thermo-mech FEM | Process design limit |
| C4. Retained austenite | pycalphad | CALPHAD on laptop, cheap to run |
| D1. H trap binding energies | DFT | Fe/C interface + GB + dislocation traps |
| D3. HE threshold | Cohesive zone FEM | Critical H for fracture |
| D4. Anode degradation | DFT + kinetics | DSA dissolution |
| E1. Multi-scale process model | Coupled simulator | Full sensitivity |
| E2. Bayesian optimization | BoTorch | Optimal operating point |
| E4. Pilot cell CFD | OpenFOAM | 1 m² current distribution |

### Anode–particle interaction (from feedback)

Carbon particles in suspension will abrade or foul anodes differently than clear electrolyte. This must be included in the FMEA as a failure mode for the closed-loop model. If co-deposition is deprioritized (carburization is primary), this risk diminishes but doesn't vanish — any particulate in the electrolyte (sludge, additive decomposition products) can cause the same issue.

### Hardware strategy

Consumer GPU + cloud burst. No HPC allocation needed for Tier 1.

| Tool | What it solves | Hardware |
|------|---------------|----------|
| JAX phase-field (custom 2D) | A1, A2, A3, C1 | Consumer GPU (RTX 4090) |
| pycalphad | C4 | Laptop |
| FEniCS / JAX FEM | D2, D3 | Consumer GPU |
| scikit-optimize / BoTorch | E2 | Laptop |
| MACE / M3GNet (ML potentials) | A4, B2, D1 | Consumer GPU (after DFT training on cloud) |
| GPAW / Quantum ESPRESSO | DFT reference calcs | Cloud GPU burst ($0.20–0.50/hr) |

Total: ~$1,600 GPU + $50–200/month cloud. No HPC queue wait times, no allocation proposals, no scheduler.

---

## What Computation Can and Cannot Tell Us

**Can tell us (reduce uncertainty before touching a beaker):**
- Whether the j–FE curve can be pushed right by pulse plating (phase-field)
- Whether stress evolution permits 1 mm deposits (FEM)
- Whether bake-out removes sufficient hydrogen (diffusion–trapping FEM)
- The cost-optimal operating point given physics constraints
- Which experiments are most informative (Bayesian optimal design)

**Cannot tell us (needs a beaker):**
- Whether the deposit actually cracks at the predicted stress threshold
- Whether the real electrolyte behaves like the model (impurities, additives, aging)
- Whether a 1 mm-thick deposit can be peeled, handled, and heat-treated without failure
- The actual hardness and tensile strength after heat treatment

**Important caveat:** The 393 tests in this repository test internal model consistency — unit tests, conservation laws, boundary conditions. They tell you the code is correct. They tell you nothing about whether the models are *right*. The most important number isn't test count — it's how many independent experimental data points the models have been validated against. Currently that number is approximately zero. The models are hypothesis generators, not design tools, until calibrated against reality.

---

## Experimental Validation

Computation and experiment run concurrently, not serially. The garage lab starts as soon as Tier 1 computation identifies the first candidate operating points.

### Round 1: Garage Lab (concurrent with Tier 1 computation)

Purpose: calibrate the three most important models with minimal data.

- 3 current densities × 3 pulse conditions × 3 replicates = 27 deposits
- Characterization: weight (FE), thickness (rate), visual (density), hardness
- **Hydrogen bake-out test:** hardness before and after 200°C × 4h on 10 deposits. If hardness changes significantly, hydrogen is a real problem.
- **Stress/deposit test:** deposit 200 µm, 500 µm, and 1000 µm at 100 mA/cm². Visual inspection for cracking, curling, delamination. This is the stress threshold experiment.
- Feed measurements into calibration pipeline → calibrated models

These three measurements — grain size vs j, hydrogen bake-out response, stress vs thickness — calibrate the three gating models and tell you whether to continue.

### Gate: Mechanical Peel and Bend Test (after Round 1, before Round 2)

Before committing to carburization and full characterization, verify that the deposit survives basic handling:

- Peel deposit from cathode substrate
- 180° bend test (does it crack?)
- If it cracks at < 200 µm, the stress problem is real — stop and fix before proceeding
- If it survives to 1 mm, proceed to Round 2

### Round 2: Targeted Validation

- Pack carburize the best deposits from Round 1
- Cross-section metallography (optical microscope)
- Hardness traverse (Vickers)
- Feed into Bayesian calibration → posterior parameter distributions
- Run confidence report → P(meet ASTM A36)

### Round 3: Pilot Demonstration

- 10× scale-up from garage rig
- Full characterization: SEM, EBSD, tensile, ICP-OES
- Run full qualification pipeline → PASS/CONDITIONAL PASS/FAIL verdict

---

## Repository Structure

```
models/
  uncertainty/           ← qualification framework (done)
    parameter_registry   ← 76 parameters with distributions
    monte_carlo          ← full-chain propagation
    sensitivity          ← Sobol indices + tornado
    bayesian_calibration ← EnKF + MCMC
    specification        ← A36/1010/1020/carburized specs
    design_space         ← robust operating region finder
    fmea                 ← 20+ failure modes
    validation_planner   ← DOE for maximum info gain
    confidence_report    ← end-to-end qualification verdict
  digital_twin           ← real-time model updating (in progress)
  process_control        ← PID loops for P&ID (in progress)
  transient              ← startup/shutdown/upsets (in progress)
  lca                    ← carbon footprint vs BOF/EAF/DRI-H2 (in progress)
  supply_chain           ← raw materials + siting (in progress)

docs/
  garage_lab/            ← shopping list + setup guide (in progress)
  protocols/             ← experimental SOPs (done)
  RESEARCH_PROGRAM.md    ← this document
```

---

## Status

- 23 kanban cards on `aqueous-steel` board
- 14 done, 2 running, 7 queued
- 393 tests passing (consistency, not validation — see caveat above)
- 214 model symbols exported
- Full pipeline: `aq-steel --quick`

**Next action:** Start Tier 1 computation (A1 phase-field + B2 HER, simultaneously). Start garage lab procurement in parallel — the first 27 deposits should be running before Tier 1 computation finishes, not after.
