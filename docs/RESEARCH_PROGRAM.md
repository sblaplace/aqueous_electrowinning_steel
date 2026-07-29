# Aqueous Electrowinning Steel — Research Program

## The Question

Can we make structural steel by electrodeposition — bypassing the blast furnace entirely?

The electrochemistry is known. The thermodynamics are known. The heat treatment is centuries old. What's unknown is whether you can put them together into a process that produces dense, ductile, carbon-bearing steel at a rate and cost that matters.

This plan defines the hard problems, what we can learn from computation, and what ultimately needs a beaker.

---

## The Hard Problems

### 1. Dense Deposits at Production Current Density

At 100 mA/cm² we get beautiful iron — 3.5 µm grains, 367 MPa yield strength, 84% Faradaic efficiency. At 500 mA/cm² we get 52% FE and porous garbage. The gap between "works in a beaker" and "works at production rate" is entirely about pushing that curve to the right.

**Unknown:** Can pulse or pulse-reverse plating recover deposit density at >300 mA/cm²? Our model says yes (pH recovery, enhanced nucleation). Nobody has demonstrated it experimentally for iron.

**Computational approach:** Phase-field microstructure evolution during deposition. Simulate grain nucleation, growth, and competition under DC vs PE vs PRE waveforms. Map the j–duty–frequency space to find the boundary between dense and porous regimes.

### 2. Carbon Co-Deposition at Structural Levels

Structural steel needs 0.1–0.8 wt% carbon. Electrolytic co-deposition of carbon particles (Guglielmi mechanism) has been demonstrated at trace levels (<0.1 wt%). Whether you can reliably incorporate enough carbon, uniformly dispersed at nanoscale, to produce steel-grade composition is unknown.

**Unknown:** Does carbon incorporate as dispersed nanoparticles (good — Orowan strengthening) or as agglomerated inclusions (bad — crack nucleation sites)? What controls the transition?

**Computational approach:** Molecular dynamics / DFT for the Fe–C particle interface energy. Kinetic Monte Carlo for particle adsorption and incorporation during deposition. CALPHAD for the Fe–C phase diagram at nanocrystalline grain sizes (Gibbs–Thomson effect shifts solubility).

### 3. Hydrogen Embrittlement at Production Rates

Every ampere that doesn't go to iron deposition produces hydrogen. At 84% FE, 16% of current is HER. Some hydrogen absorbs into the deposit. For deposits above ~500 MPa yield strength, even a few ppm of diffusible hydrogen causes catastrophic brittle fracture.

**Unknown:** What is the critical diffusible hydrogen concentration for electrodeposited nanocrystalline iron? Conventional HE thresholds are calibrated for wrought steel — nanocrystalline structures have different trap site densities.

**Computational approach:** Ab initio hydrogen trap binding energies at grain boundaries, dislocations, and carbon particle interfaces in nanocrystalline Fe. Diffusion–trapping finite element model to predict H concentration profiles under realistic deposition and bake-out conditions.

### 4. Thick, Stress-Free Deposits

Electrodeposited iron has high internal stress from grain growth constraints, hydrogen incorporation, and lattice mismatch. At 10 µm it's manageable. At 1 mm (structural thickness), it cracks or delaminates.

**Unknown:** What is the stress evolution during deposition as a function of thickness, current density, and pulse parameters? Can we predict and prevent cracking?

**Computational approach:** Finite element model of stress evolution during deposition — couple the grain-scale microstructure (from phase-field) to continuum mechanics. Include hydrogen-induced swelling and lattice contraction from carbon incorporation.

### 5. Grain Stability During Heat Treatment

Nanocrystalline electrodeposited iron (1–5 µm grains) is thermodynamically unstable. Carburizing at 900°C will drive grain growth. If grains coarsen to >10 µm, the Hall-Petch strengthening benefit is lost.

**Unknown:** Does carbon pin grain boundaries during carburizing (Zener pinning), preserving the nanocrystalline structure? Or does grain growth outrun carbon diffusion?

**Computational approach:** Phase-field grain growth simulation with carbon diffusion coupling. Monte Carlo Potts model for Zener pinning by carbide particles. Map the T–t–C space to find conditions that preserve <5 µm grains.

### 6. Cost Parity with Blast Furnace

BOF steel: $300–500/t. Our model: $800–2000/t. The gap is electricity cost, chemical cost, and throughput. The pathway to parity requires:
- Cheap electricity (<$0.02/kWh — available from curtailed renewables)
- High current density (>300 mA/cm² at >80% FE) — Problem 1
- Low electrolyte makeup cost (recycling, minimal purge)
- Modular scale-up (add cells, not build new plants)

**Unknown:** At what electricity price and current density does electrowinning steel become cheaper than DRI-H2? Our sensitivity model can answer this — but needs calibrated inputs.

---

## Computational Research Program

Each item below can be done on a computer, without a wet lab. They reduce uncertainty in the screening models and identify the most promising experimental targets.

### Phase A: Microstructure Simulation

| Task | Method | Output | Reduces Uncertainty In |
|------|--------|--------|----------------------|
| A1. Grain nucleation/growth during deposition | Phase-field (MOOSE, OpenPhase, or custom) | Grain size vs j, duty, waveform | `mechanical_properties.py` k_HP, grain size model |
| A2. Deposit porosity prediction | Phase-field + gas bubble inclusion | Porosity vs j, FE | `mechanical_properties.py` porosity penalty |
| A3. Stress evolution during deposition | FEM coupled to microstructure | Stress vs thickness, cracking threshold | Deposit design limits |
| A4. Carbon particle–matrix interface | DFT (VASP, Quantum ESPRESSO) | Interface energy, cohesion | `co_deposition.py` Guglielmi parameters |

### Phase B: Transport and Kinetics

| Task | Method | Output | Reduces Uncertainty In |
|------|--------|--------|----------------------|
| B1. 3D CFD of electrowinning cell | OpenFOAM / COMSOL | Flow field, boundary layer map | `transport.py`, `scale_up.py` |
| B2. HER competition mechanism | DFT + microkinetic modeling | Tafel slopes, H coverage | `kinetics.py` exchange current density |
| B3. Fe–Ni anomalous co-deposition | Phase-field + kinetic Monte Carlo | Ni content vs j, pH | `co_deposition.py` anomalous model |
| B4. Carbon particle transport in boundary layer | CFD + Lagrangian particle tracking | Deposition rate vs particle size/loading | `co_deposition.py` Guglielmi σ |

### Phase C: Heat Treatment

| Task | Method | Output | Reduces Uncertainty In |
|------|--------|--------|----------------------|
| C1. Grain growth during carburizing | Phase-field + carbon diffusion | Grain size vs T, t, C | `carburization.py`, `mechanical_properties.py` |
| C2. Zener pinning by carbides | Monte Carlo Potts | Critical carbide density for grain pinning | Heat treatment design |
| C3. Quench cracking risk | Thermo-mechanical FEM | Stress vs cooling rate, geometry | Process design limits |
| C4. Retained austenite stability | CALPHAD (Thermo-Calc, OpenCALPHAD) | RA fraction vs composition, T | `tempering.py` Ms, RA model |

### Phase D: Hydrogen and Degradation

| Task | Method | Output | Reduces Uncertainty In |
|------|--------|--------|----------------------|
| D1. H trap binding energies | DFT | Binding energy at GB, dislocation, carbide | `hydrogen_embrittlement.py` trap model |
| D2. H diffusion–trapping FEM | MOOSE / FEniCS | H profile vs deposition conditions | `hydrogen_embrittlement.py` effective diffusivity |
| D3. HE threshold for nanocrystalline Fe | Cohesive zone FEM + H concentration | Critical H for fracture | Design limits |
| D4. Anode degradation kinetics | DFT + electrochemical kinetics | DSA coating dissolution rate | `anode.py`, `closed_loop.py` |

### Phase E: Process Optimization

| Task | Method | Output | Reduces Uncertainty In |
|------|--------|--------|----------------------|
| E1. Multi-scale process model | Couple A–D outputs into unified simulator | Full process sensitivity | All models |
| E2. Bayesian optimization of operating point | BO on coupled model | Optimal j, duty, T, pH, Ni, C | Design space |
| E3. Techno-economic with calibrated parameters | Monte Carlo on technoeconomic | Cost distribution with confidence intervals | Business case |
| E4. Scale-up CFD for pilot cell | OpenFOAM at 1 m² scale | Current distribution, thermal map | `scale_up.py` |

---

## What Computation Can and Cannot Tell Us

**Can tell us (reduce uncertainty before touching a beaker):**
- Which parameters dominate deposit quality (sensitivity analysis — done)
- Whether the j–FE curve can be pushed right by pulse plating (phase-field)
- Whether carbon particles will incorporate as dispersed or agglomerated (DFT + MD)
- Whether grain boundaries survive carburizing (phase-field + Zener)
- Where hydrogen accumulates and at what concentration (FEM)
- The cost-optimal operating point given physics constraints (BO on coupled model)

**Cannot tell us (needs a beaker):**
- Whether the real Fe–C co-deposition mechanism matches our DFT predictions
- Whether the deposit actually cracks at the predicted stress threshold
- Whether the real electrolyte behaves like the model (impurities, additives, aging)
- Whether a 1mm-thick deposit can be peeled, handled, and heat-treated without failure
- The actual hardness and tensile strength after heat treatment (validation)

**The strategy:** Use computation to narrow the experimental search space from "try everything" to "try these three conditions." Each computational phase produces a deliverable (model, parameter, operating window) that feeds the next. By Phase E, we should know exactly what to build and what to test.

---

## Experimental Validation (After Computation)

Once the computational program identifies the most promising conditions:

### Round 1: Garage Lab
- 3 current densities × 3 pulse conditions × 3 replicates = 27 deposits
- Characterization: weight (FE), thickness (rate), visual (density), hardness (if tester available)
- Feed measurements into calibration pipeline → calibrated models
- See `docs/garage_lab/` for shopping list and protocols

### Round 2: Targeted Validation
- Pack carburize the best deposits from Round 1
- Cross-section metallography (optical microscope)
- Hardness traverse (Vickers)
- Feed into Bayesian calibration → posterior parameter distributions
- Run confidence report → P(meet ASTM A36)

### Round 3: Pilot Demonstration
- 10× scale-up from garage rig
- Continuous operation for 1 week
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
  lca                    ← carbon footprint vs BOF/EAF (in progress)
  supply_chain           ← raw materials + siting (in progress)

docs/
  garage_lab/            ← shopping list + setup guide (in progress)
  protocols/             ← experimental SOPs (done)
  CI_WORKFLOW.md         ← CI config (ready to add via GitHub UI)
```

---

## Status

- 23 kanban cards on `aqueous-steel` board
- 14 done, 2 running, 7 queued
- 393 tests passing
- 214 model symbols exported
- 50+ figures generated
- Full pipeline: `aq-steel --quick`

**Next action:** Start Phase A computational work. The phase-field microstructure simulation (A1) is the highest-value item — it directly answers "can pulse plating save the j–FE curve?" without touching a beaker.
