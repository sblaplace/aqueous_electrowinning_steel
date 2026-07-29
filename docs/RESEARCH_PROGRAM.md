# Dark Mill — Autonomous Steel Production via Aqueous Electrowinning

## The Vision

A steel mill that runs itself. No blast furnace, no continuous casting, no human operators on the floor. A modular cell stack that takes in electricity, water, and iron sulfate — and outputs steel sheet. Each cell is a dark mill: self-monitoring, self-calibrating, self-correcting. Scale by replication, not construction.

The core process is aqueous electrowinning with post-deposition carburization and heat treatment. The enabling technology is a digital twin that knows the deposit quality before the cathode is pulled, and control loops that maintain spec without human intervention.

---

## The Real Competitors

Not the blast furnace. The competitors are:

**DRI-H2** — HYBRIT is at pilot scale. Uses existing downstream metallurgy. Iron-making step already solved at temperature. Our advantages: modularity, intermittent power tolerance, lower capital, scrap-compatible feedstock. These need quantification.

**Boston Metal** — molten oxide electrolysis. High temperature, high current density, but high capital and no intermittency tolerance.

**Electra** — aqueous low-T iron. Raised money on approximately this thesis. Their public disclosures are our best free calibration point. If they're at 500 mA/cm² and 90% FE, our model needs to explain how. If they're at 100 mA/cm² and 80%, our model matches and we're derisked on "is this possible at all."

**The question:** At what electricity price and current density does a dark mill beat DRI-H2 at the margin? Run this sensitivity NOW with current uncorrelated priors. If the winning corner requires j > 800 mA/cm² at 90% FE with $0.02/kWh electricity, we're chasing a physically implausible scenario. If it's satisfied at 250 mA/cm² and 78% FE, we're on plausible ground.

---

## Tier 0: Does the Corner Exist?

Before any phase-field, before any DFT, before the garage lab:

1. **Techno-economic sensitivity NOW.** Run Monte Carlo on the cost model with current screening priors. Find the Pareto front of (electricity price, current density, FE) that hits cost parity with DRI-H2. If the front is empty, stop. If it exists, define it precisely. This is a day of compute.

2. **Read the electroforming literature.** Iron electroforming is a done industry at small scale (bellows, precision parts). Di Bari's chapter in *Modern Electroplating* covers internal stress, hydrogen, additives (saccharin as stress reliever, boric acid buffering, chloride vs sulfate baths). Cohen and Fedotev's work from the 60s–80s on thick iron electroforming got to millimeters with hot electrolytes (>90°C), specific chloride chemistries, and hours-long deposition. Some of our "unknowns" are actually known and just not in our models.

3. **Check Electra and Boston Metal public disclosures.** Current density, FE, electrolyte chemistry, operating temperature. Free calibration.

4. **Buy commercial electroformed iron foil** (Toyo Kohan or similar). Do bake-out kinetics with a hydrogen analyzer (LECO RH-402 or service lab, ~$50/sample). Real diffusible-H numbers for electrodeposited iron in two weeks. That calibrates D2 without waiting for our own deposits. Not *our* iron, but bounds the problem.

Cost of Tier 0: days, not weeks. Value: might reshuffle the entire tier list.

---

## The Hard Problems

### 1. Dense Deposits at Production Current Density

At 100 mA/cm² we get 3.5 µm grains, 367 MPa YS, 84% FE. At 500 mA/cm² we get 52% FE and porous deposit. The j–FE curve collapse is the economic gate.

Pulse plating precedent exists for nickel and copper. Extrapolation to iron is plausible but unvalidated.

**Kill criterion:** If phase-field shows FE < 75% at j > 400 mA/cm² for any pulse waveform, the pathway to cost parity closes.

### 2. Carbon via Carburization (Derisked, Not Solved)

Pack carburization of electrodeposited iron is the primary carbon pathway. A 1 mm deposit at 925°C reaches 0.2 wt% C at center in ~30 minutes. This eliminates co-deposition from the critical path.

**But "derisked" is not "solved."** Unresolved risks:
- Sulfur from FeSO₄ residue segregating to grain boundaries → hot-shortness at carburizing temperature
- Residual stress relief during heating → warping in thin plate
- Recrystallization textures → anisotropic mechanical properties
- Grain growth at 925°C (see Problem 5)

**Cheap early test:** Buy commercial electrolytic iron foil, carburize it, see what happens. $200 experiment that eliminates a whole class of downstream surprises before we deposit anything.

### 3. Hydrogen Embrittlement (Likely Manageable)

Standard bake-out (4h at 200°C) is well-proven for hard chromium and other electrodeposits. Our deposits are ultrafine-grained (1–5 µm), which means more grain boundary area and more trap sites — but still far from nanocrystalline (<100 nm).

**Buy the answer this week:** Commercial electroformed iron foil + LECO hydrogen analysis. Real numbers in two weeks.

**Key coupling:** Carbon particles and carbides are H-trap sites. High carbon loading may simultaneously help grain stability (Zener pinning) and worsen H retention. DFT scope must include Fe/C interfaces.

### 4. Thick, Stress-Free Deposits (PROBABLY THE REAL GATE)

This is likely harder than the FE problem. The FE literature at least has pulse-plating precedent. The 1 mm iron deposit literature is thin because **people have tried and failed.** Cohen and Fedotev got to millimeters but with hot electrolytes (>90°C), chloride chemistries, and long deposition times. Whether that's compatible with our economics is a separate question.

**Stress is the gate, not FE.** If we can't make a dense deposit above ~1 mm, carbon can be added (carburization), hydrogen can be removed (bake-out), grains can be managed (heat treatment) — but nothing else matters.

**Run A3 (stress FEM) before or in parallel with A1 (grain nucleation).** If stress kills us, the pulse waveform optimization is moot.

**Kill criterion:** If stress modeling shows no path to 1 mm without cracking at any j > 100 mA/cm², pivot to thin-film applications (battery foil, electrical steel laminations) or kill.

### 5. Grain Stability During Heat Treatment

1–5 µm grains are **ultrafine-grained**, not nanocrystalline. This matters:
- At 3.5 µm: grain growth at 900°C is slow, carburizing times may be tolerable
- Below 500 nm: Zener pinning becomes essential
- Below 100 nm: catastrophic coarsening, inverse Hall-Petch risk

If our deposits are 3.5 µm (model prediction at 100 mA/cm²), grain stability is a concern but not a crisis.

### 6. Anode Cost (Potentially Tier 1 in Disguise)

DSA lifetime under Fe²⁺-containing sulfate at high current density is not a solved problem. Carbon particles in suspension will abrade or foul anodes. If anode replacement cost dominates OPEX, this is a Tier 1 problem. Check early: what's the anode cost per kg of iron produced in existing electroforming operations?

---

## Computational Program

### Tiering

**Tier 0 — Do this week. No hardware needed.**

| Task | Method | Output |
|------|--------|--------|
| Techno-economic sensitivity | Monte Carlo on current model | Pareto front: (j, FE, electricity) vs DRI-H2 |
| Literature review | Di Bari, Cohen, Fedotev, Electra patents | Known unknowns → known knowns |
| Buy electroformed iron foil | Toyo Kohan or similar | Substrate for H analysis and carburization test |
| Competitive analysis | Electra, Boston Metal public disclosures | Calibration data |

**Tier 1 — Gating. Determines whether the rest is worth doing.**

| Task | Method | Answers |
|------|--------|---------|
| A1. Grain nucleation/growth | 2D phase-field (JAX/GPU) | Can pulse plating push the j–FE curve? |
| A3. Stress evolution | FEM coupled to microstructure | Can we make 1 mm deposits without cracking? |
| D2. H diffusion–trapping | FEM | Is bake-out sufficient? |
| E2. Bayesian DOE | BoTorch on coupled model | Which 27 experiments are most informative? |

Run A1 and A3 in parallel. Run B2 (HER mechanism) alongside A1 — they share parameter space. Run E2 *before* Round 1, not after.

**Tier 2 — Do these if Tier 1 is encouraging.**

| Task | Method | Answers |
|------|--------|---------|
| C1. Grain growth during carburizing | Phase-field + diffusion | Do grains survive 925°C? |
| B1. Cell CFD for scale-up | OpenFOAM / analytical | Boundary layer at pilot scale |
| E3. Calibrated techno-economic | Monte Carlo with fitted params | Real cost distribution |
| A2. Deposit porosity | Phase-field + bubble model | Porosity vs j |
| Substrate selection study | Experiment | Ti vs SS vs Cu vs graphite nucleation |

**Tier 3 — Scientifically interesting, not on critical path.**

A4, B3, B4, C2, C3, C4, D1, D3, D4, E1, E4.

### Dimensionality honesty

2D phase-field on a 4090 is fine for grain nucleation and j–FE mapping. 3D phase-field with coupled species transport, hydrogen, and pulse-timescale resolution is not fine on a 4090. Grain growth and porosity are notoriously 3D phenomena (percolation of voids, GB curvature). Be explicit about where 2D→3D might change the answer, and use cloud GPU burst for the 3D runs.

### Hardware

| Tool | Hardware |
|------|----------|
| JAX phase-field (2D) | Consumer GPU |
| JAX phase-field (3D) | Cloud GPU burst ($0.20–0.50/hr) |
| pycalphad | Laptop |
| FEniCS / JAX FEM | Consumer GPU |
| BoTorch | Consumer GPU |
| MACE / M3GNet | Consumer GPU (after DFT training on cloud) |
| GPAW | Cloud GPU burst |

---

## Dark Mill Architecture

A dark mill cell is the unit of replication. Each cell contains:

```
┌─────────────────────────────────────────────────────┐
│                    DARK MILL CELL                    │
│                                                     │
│  Electrolyte tank + cathode/anode stack             │
│  Recirculation loop (pump, filter, heat exchanger)  │
│  Gas handling (O₂ vent, H₂ safety)                 │
│  Electrolyte makeup (auto-dosing from tank sensors) │
│  Carburizing retort (pack, sealed box)              │
│  Quench + temper station                            │
│                                                     │
│  INSTRUMENTS                                        │
│    TT, pHAT, FT, AT, VT, AIT (from P&ID)           │
│    O₂ probe (carburizing atmosphere)                │
│    Inline hardness (Vickers indent robot)           │
│    Weight sensor (deposit mass tracking)            │
│                                                     │
│  CONTROL                                            │
│    PID loops (8, from process_control)              │
│    Digital twin (real-time state estimation)        │
│    Confidence report (per-batch quality cert)       │
│    FMEA watchdog (anomaly detection + shutdown)     │
│                                                     │
│  AUTOMATION                                         │
│    Cathode insertion/extraction robot               │
│    Carburizing cycle (auto load/unload/heat/quench) │
│    Product handling (stack, label, ship)             │
│    Electrolyte purge + makeup (concentration ctrl)  │
└─────────────────────────────────────────────────────┘
```

Scale by replication: one cell makes ~1 kg/day. Ten cells make 10 kg/day. A hundred cells make 100 kg/day. No scale-up physics needed — just more cells.

**The dark mill doesn't scale up. It copies.**

This eliminates Problem 4's scale-up dimension. The stress problem is per-cell, not per-plant. If one cell works, a hundred cells work.

---

## Experimental Program

### Pre-Round 0: Buy Answers (this week)

- Commercial electroformed iron foil → LECO hydrogen analysis → calibrate H diffusion model
- Carburize commercial foil → metallographic cross-section → calibrate grain growth model
- Read Di Bari, Cohen, Fedotev → convert unknowns to knowns
- Check Electra/Boston Metal disclosures → calibrate j–FE expectations

### Round 1: Garage Lab (concurrent with Tier 1)

27 deposits, but informed by Bayesian DOE (E2) rather than orthogonal grid. Substrate chosen deliberately (not a hidden variable).

Three gates:
1. **FE vs j:** Does the model match reality within 10%?
2. **Bake-out test:** Hardness before/after 200°C × 4h on 10 deposits.
3. **Stress test:** Deposit 200 µm, 500 µm, 1000 µm. Peel, bend 180°, inspect for cracking.

If deposits crack at < 200 µm → stress is the real problem, fix before proceeding.
If deposits survive to 1 mm → proceed to carburization.

### Round 2: Targeted Validation

Pack carburize best deposits. Hardness traverse. Metallographic cross-section. Feed into Bayesian calibration. Run confidence report.

### Gate: Does a Single Cell Work as a Dark Mill?

Before scaling to multiple cells, prove one cell runs unattended for 72 hours:
- Auto-dosing maintains pH and concentration
- Digital twin detects and flags injected faults
- Deposit quality stays within spec across 3 consecutive batches
- No unplanned shutdowns

### Round 3: Pilot

10 cells, continuous operation. Full characterization.

---

## Revenue Path Before Cost Parity

The cost model needs to beat DRI-H2 eventually. It doesn't need to beat DRI-H2 at product #1.

Higher price ceiling products that the same cell can produce:
- **Electrical steel laminations** — thin, high-silicon, premium price ($3000–8000/t)
- **Battery current collector foil** — ultra-thin iron foil, high purity
- **Precision electroforms** — bellows, molds, satellite components ($10,000+/t)
- **Corrosion-resistant coatings** — electrodeposited Fe-Ni-Cr alloys

A dark mill producing precision electroforms at $10,000/t funds the R&D to get to structural steel at $500/t. The cell is the same; the product spec changes.

---

## Questions to Answer Before Investing Further

1. **What's Electra actually achieving?** Public disclosures on current density and FE. One hour of research.
2. **What does the electroforming industry know?** Di Bari's chapter. One day of reading.
3. **What's the anode cost per kg iron?** If DSA replacement dominates OPEX, everything else is secondary.
4. **What's the actual experimentalist bandwidth?** If it's one person nights and weekends, Round 1 is months, not weeks, and the concurrent story changes.
5. **Is there someone at Electra or ex-Boston Metal to talk to?** A one-hour conversation could save six months of computation.

---

## Status

- 23 kanban cards on `aqueous-steel` board
- 393 tests passing (consistency, not validation)
- 214 model symbols exported
- Full pipeline: `aq-steel --quick`
- Qualification framework: Monte Carlo, sensitivity, specs, FMEA, Bayesian calibration, design space, validation planner, confidence report — all built

**Next action:** Tier 0. Run the techno-economic sensitivity. Read the electroforming literature. Buy commercial iron foil. Check Electra's numbers. All of this is days, not weeks, and might reshuffle everything that follows.
