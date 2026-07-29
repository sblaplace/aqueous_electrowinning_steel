# Dark Mill — Autonomous Iron Production via Aqueous Electrowinning

## The Vision

A steel mill that runs itself. Modular cell stacks that take in electricity, water, and iron feedstock — and output iron. Each unit is self-monitoring, self-calibrating, self-correcting. Scale by replication, not construction. Deploy next to a wind farm, a mine, or a pickle liquor source.

---

## Page 1 Decision: Product or Feedstock?

**This is the highest-leverage decision in the program.** Everything downstream depends on it.

### Option A: Feedstock for a melt shop

The electrowon iron is delivered as powder, flake, or thin foil to an EAF or induction furnace. The melt shop adds carbon, manages hydrogen (boils off at 1600°C), and produces finished steel using existing metallurgy.

**What this deletes:**
- Problem 2 (carbon): added in the melt. Deleted.
- Problem 3 (hydrogen): boils off at 1600°C. Deleted.
- Problem 4 (thick crack-free deposits): you *want* friable, easily-strippable deposit. Inverted to design goal.
- Problem 5 (grain stability): irrelevant. Deleted.

**What remains:** FE, cell voltage, current density, electrolyte loop closure, feedstock cost, impurity management. This is a hydrometallurgy program — maybe 20% of the work scoped in earlier versions.

### Option B: Near-net-shape structural product

The electrowon plate is the final product — steel sheet, foil, or plate. All six hard problems apply. The physical metallurgy program is required.

### The answer is probably: feedstock first, product later.

A dark mill that produces iron units at competitive cost proves the economics. Once the cell works, extending to near-net-shape product is an engineering problem on a proven platform. The reverse order — trying to solve physical metallurgy and economics simultaneously — is how programs die.

**This document assumes Option A (feedstock) as the primary path, with Option B (product) as a later extension.**

---

## The Real Competitors

Not the blast furnace. The competitors are:

**DRI-H2** — HYBRIT at pilot scale. Uses existing downstream metallurgy. ~3,300–3,500 kWh/t (including H₂ electrolysis + melting). Our energy: 0.96 × V_cell / FE kWh/kg. At V = 2.5, FE = 0.85 → 2,820 kWh/t. **We may be more energy-efficient than DRI-H2** — transferring electrons to iron directly instead of laundering them through hydrogen. That's the headline.

**Electra** — aqueous low-T electrowinning of iron from ore. Colorado. Well-funded. Read their patents — enablement forces disclosure of operating window.

**Boston Metal** — molten oxide electrolysis. High temperature, high capex.

**ΣIDERWIN / ArcelorMittal** — alkaline suspension electrolysis of iron oxide at ~110°C. Pilot-scale, public deliverables.

**Allanore & Sadoway (MIT)** — molten salt electrolysis for iron.

**Action:** Read Electra's patent family, ΣIDERWIN deliverables, and Allanore/Sadoway publications before doing any modeling. If the alkaline suspension route dominates on capex, switch.

---

## The Headline Number

Energy per tonne of iron = 0.96 × V_cell / FE × 1000 kWh/t

| V_cell | FE   | Energy (kWh/t) | Cost @ $0.04/kWh |
|--------|------|-----------------|-------------------|
| 2.0    | 0.90 | 2,133           | $85/t             |
| 2.5    | 0.85 | 2,824           | $113/t            |
| 3.0    | 0.80 | 3,600           | $144/t            |
| 3.5    | 0.70 | 4,800           | $192/t            |

DRI-H2 needs ~3,300–3,500 kWh/t. **Electricity is not the binding constraint.** CAPEX per m² of installed cell is.

At 100 mA/cm² and 85% FE, areal productivity is ~7.8 t/(m²·yr). A zinc tankhouse runs ~500 A/m² and costs ~$1,000–1,500 per annual tonne of capacity — for a product worth $2,500–3,000/t. We'd be building the same machine to make something worth $400–600/t.

**That's the entire problem in one sentence: you need roughly 5× a zinc tankhouse's areal productivity, or roughly 5× cheaper cells, or some product of the two.**

---

## The Cell Architecture Question

Cell engineering for high current density is industrially mature — it's just never been asked to eject a solid product:

- **Chlor-alkali:** 400–600 mA/cm² through a membrane, filter-press stacks, few hundred $/m²
- **PEM electrolysis:** 2,000 mA/cm²
- **ED copper foil drums:** 400–1,000 mA/cm², continuous

The gating question is architectural: **is there a cell that combines filter-press current densities with continuous solid harvesting?**

Candidates worth a week of paper-study each:
- Rotating cylinder electrode with scraper (Eco-Cell is commercial)
- Fluidized/particulate bed cathodes
- Moving-belt cathodes
- Drum-and-strip (copper foil technology adapted for iron)

All of these produce powder, flake, or thin foil — they only work if the product decision (§1) is feedstock. Which is another argument for Option A.

---

## The Anode Problem (Tier 0, Not Tier 3)

In an undivided sulfate cell, the dominant anodic reaction is not OER — it's Fe²⁺ → Fe³⁺ (E° = +0.77 V, fast, no overpotential). The Fe³⁺ diffuses to the cathode and is reduced back to Fe²⁺. This is a perfect redox shuttle that eats an enormous fraction of current and depresses FE far below anything our kinetics model predicts.

**You need a divided cell.** Membrane cost, membrane ohmic drop at 300–500 mA/cm², crossover, anolyte composition, and acid balance are all first-order to both CAPEX and energy.

Conversely — if you *deliberately* run Fe²⁺/Fe³⁺ as the anode reaction and pipe the Fe³⁺ to an oxidative leach circuit, you drop anode potential by ~1 V and cut energy ~35%. Worth an afternoon on a whiteboard.

**Action:** Add membrane and divided-cell parameters to the model. This changes everything about V_cell and FE.

---

## Missing Physics (Roughly Ordered by Impact)

1. **Cell voltage.** Not in the 76-parameter registry. For an electrowinning process this is THE number. Energy = 0.96 × V / FE. Add V_cell decomposition: E_cathode, E_anode, η_cathode, η_anode, IR_electrolyte, IR_membrane, IR_contacts.

2. **Temperature.** Not in the registry. Sets D, conductivity, viscosity, solubility, FE, and internal stress. Chinese electrolytic iron practice runs 50–60°C. Its absence is diagnostic of where the parameters came from.

3. **Divided cell / membrane.** Missing entirely. Determines anode reaction, V_cell, acid balance, Fe³⁺ crossover.

4. **Purification circuit.** Copperas from TiO₂ route carries Mn, Mg, Al, Ti, V, Cr. Pickle liquor carries Cu, Ni, Zn, Pb, Sn. Everything nobler than Fe co-deposits preferentially. Cu > 0.1% → hot shortness. Zinc tankhouses spend roughly as much on purification as on the tankhouse itself. This is a missing unit operation.

5. **Mass-transport limit.** Pulse plating cannot beat it. At 400 mA/cm² average, 30% duty → 1,330 mA/cm² peak. With 2 M Fe²⁺ and δ = 20 µm, i_L ≈ 1,350 mA/cm². You're at the wall. Pulse redistributes flux in time; it does not create it. The route to high j is [Fe²⁺] → 2 M+, temperature → 50–70°C, δ → <30 µm via forced convection. Pulse is a microstructure and pH-recovery tool, not the rate lever.

6. **Feedstock sourcing.** Global spent pickle liquor + copperas: ~3–6 Mt Fe/yr vs 1,900 Mt steel/yr. Beachhead, not thesis. Chase negative-cost feedstocks: acid mine drainage, red mud, pickle liquor you're paid to take. A −$50/t feedstock is worth more to the model than 10 points of FE.

---

## Tier 0: Cheapest Awareness Available

Ranked by bits per dollar. Do all of these before any modeling.

### 1. Two weeks of archaeology — $0

- **US Bureau of Mines RI-series reports** on iron electrowinning (1960s–70s). FE vs j vs T tables that we are currently planning to regenerate with a GPU. Read them first.
- **ΣIDERWIN / ArcelorMittal EU project deliverables** — public, alkaline suspension, pilot-scale.
- **Allanore & Sadoway (MIT)** — molten salt electrolysis.
- **Electra patents** — enablement forces disclosure of operating window. Best free source of competitor's actual operating conditions.
- **Di Bari's chapter in Modern Electroplating** — internal stress, hydrogen, additives (saccharin as stress reliever, boric acid buffering, chloride vs sulfate baths).
- **Cohen and Fedotev** — thick iron electroforming from 60s–80s. Hot electrolytes (>90°C), chloride chemistries, hours-long deposition. They got to millimeters.
- **Fe and Fe-Ni pulse plating in MEMS/LIGA** — "nobody has demonstrated pulse plating for iron" is very likely false. Check.

### 2. Talk to someone who has operated a zinc or cobalt tankhouse — $0, 2 hours

They will name ten failure modes — manganese sludge, anode passivation, stripping-machine jams, short-circuit detection, acid mist and lead/health limits, electrolyte bleed, CE drift over a cathode cycle — that no first-principles FMEA will ever generate.

### 3. Hull cell — ~$300, one afternoon

One 10-minute deposit maps deposit appearance across ~2 decades of current density. Map (T, C, pH, additive) space in a week. Highest information-per-dollar experiment in the history of electroplating.

### 4. Polarization + FE curves in a divided beaker cell — ~$1,000, two weeks

V(j) and FE(j) at 3 temperatures × 3 concentrations × 2 pH. Measure FE by hydrogen volumetry (inverted burette over cathode) — continuous, real-time, more sensitive than weighing. This is the complete input to the techno-economic model.

### 5. RDE + Levich — ~$2–5k

Separates kinetics from transport. Gives Tafel slopes for Fe deposition AND HER on the same surface. Directly calibrates the 1D transport model. Single measurement that makes models predictive rather than decorative.

### 6. Bent-strip / Stoney stress measurement — ~$200

Deposit on one face of a thin shim, measure curvature, get internal stress in real time as a function of thickness and waveform. Answers Problem 4 in hours. The associated FEM answers it in months, less reliably.

**Total: ~$4–8k and six weeks. These six activities subsume most of what was Tier 1 and Tier 2.**

---

## The Right Tier 1 Model

Phase-field doesn't compute Faradaic efficiency. FE is set by competing charge-transfer kinetics (Butler–Volmer for Fe²⁺/Fe vs HER on an evolving Fe surface) coupled to transport and local pH. Phase-field takes FE as an *input*.

**The right model is boring and cheap:** a 1D diffusion-layer model with:
- Two Faradaic reactions (Fe²⁺/Fe and HER)
- Migration + diffusion for Fe²⁺/H⁺/HSO₄⁻/SO₄²⁻
- Homogeneous acid–base equilibria (boric acid, Fe(II) hydrolysis)
- Computed surface pH with Fe(OH)₂ precipitation criterion
- Outputs: FE(j, T, C, δ, pH, buffer) and V(j)

This is a few hundred lines. It calibrates against a week of beaker work. It has well-established literature analogues in Ni and Zn electrowinning. Phase-field becomes a Tier 3 curiosity by comparison.

---

## Kill Criteria — Against Measurements, Not Model Outputs

You cannot kill a program on the output of an uncalibrated model. Replace with things a multimeter can adjudicate:

- Measured V_cell × 0.96 / FE > 4,000 kWh/t at j ≥ 300 mA/cm², 60°C, divided cell, after optimizing C/T/flow → **kill.**
- Measured FE < 70% at 300 mA/cm² under the same conditions → **kill.**
- No coherent 100 g iron plate (or powder/flake if feedstock path) produced by month 6 → **stop and reassess.**
- Cost per m² of a cell that can be stripped continuously > threshold → **pivot to cell architecture work.**
- If Electra or ΣIDERWIN is already at our target economics → **pivot to complementary niche or license.**

---

## What to Freeze

These modules are modeling a plant whose unit operation hasn't been demonstrated. They're two years early:

- `digital_twin` — real-time model updating of a process we haven't run
- `process_control` — PID loops for a cell we haven't built
- `transient` — startup/shutdown of a plant that doesn't exist
- `supply_chain` — logistics for a product we haven't made
- `lca` — one page of arithmetic once you know kWh/t and grid carbon intensity; does not need a module
- `uncertainty/sensitivity` — Sobol indices over 76 invented priors are sensitivity analysis of a fiction. Run them on the 10-parameter transport model instead, where they'll actually tell you which experiment to do next.

**The ratio of epistemic apparatus to input data is the most alarming feature of the program.** A photograph of a 10 × 10 cm iron plate next to a ruler is worth more than the entire uncertainty-quantification suite — for our own beliefs and for anyone we need to convince.

Keep: `technoeconomic`, `transport` (1D diffusion-layer), `hull_cell`, `voltammetry`, `eis`, `pourbaix`, `kinetics`. These are the models that matter before the first deposit exists.

---

## Revised Computational Tiers

### Tier 0 — This week. $0.

- [ ] Read Bureau of Mines RI reports on iron electrowinning
- [ ] Read Electra patents, ΣIDERWIN deliverables, Allanore/Sadoway
- [ ] Read Di Bari on electroforming, Cohen/Fedotev on thick deposits
- [ ] Check MEMS/LIGA pulse plating literature for iron
- [ ] Add cell voltage decomposition to the model (E_cathode, E_anode, η, IR_membrane)
- [ ] Add temperature as a parameter (was missing from 76-param registry)
- [ ] Add divided cell / membrane model
- [ ] Run techno-economic sensitivity: is there a winning corner vs DRI-H2?
- [ ] Decision: product or feedstock? (Page 1)

### Tier 1 — After archaeology informs the question.

- [ ] 1D diffusion-layer model (replaces phase-field as gating model)
- [ ] Hull cell experiments (300 mA/cm² maps in one afternoon)
- [ ] Polarization + FE curves in divided beaker cell
- [ ] Cell architecture paper study (drum, belt, rotating cylinder, filter-press)
- [ ] RDE + Levich for kinetics/transport separation
- [ ] Stoney stress measurement on thin shim
- [ ] Purification circuit design (cementation, hydrolysis for Cu/Ni/Zn removal)

### Tier 2 — After Tier 1 shows a viable operating window.

- [ ] Phase-field for microstructure (now informed by real FE data)
- [ ] Pilot cell design (continuous harvesting architecture)
- [ ] Carburization trials (if product path)
- [ ] Hydrogen bake-out characterization (if product path)
- [ ] Calibrated techno-economic model

### Tier 3 — Science enrichment.

- [ ] DFT for interface energies
- [ ] 3D CFD
- [ ] Grain growth / Zener pinning
- [ ] Digital twin, process control, transient, supply chain, LCA

---

## Questions to Answer Before Anything Else

1. **Product or feedstock?** (Determines 60% of the program.)
2. **What's V_cell, and where does the number come from?** (Currently hard-coded at 2.5V with no decomposition.)
3. **Divided or undivided?** What is the anode reaction, what is the anolyte, what happens to the Fe³⁺?
4. **Where does the sulfate go?** Is the acid loop closed, and if so, against what dissolution chemistry?
5. **What's the feedstock at 100 kt/yr, and what does it cost per tonne of contained Fe?**
6. **Why is temperature not a parameter?** (It should be — it sets everything.)
7. **What's the assumed $/m² of installed cell, and does the program survive that number being 2× worse?**
8. **Have we read the ΣIDERWIN deliverables and Electra's patent family?** If the alkaline suspension route dominates on capex, would we switch?

---

## Revenue Path

The cost model needs to beat DRI-H2 eventually. It doesn't need to beat DRI-H2 at product #1.

**Feedstock beachhead:** negative-cost feedstocks (acid mine drainage, red mud, pickle liquor). You're paid to take the feedstock and sell the iron. The economics work before you compete with BOF.

**Product extensions (same cell, higher price ceiling):**
- Electrical steel laminations — thin, high-silicon, premium ($3,000–8,000/t)
- Battery current collector foil — ultra-thin, high purity
- Precision electroforms — bellows, molds, satellite components ($10,000+/t)

A dark mill producing precision electroforms funds the R&D to get to commodity iron at $400/t.

---

## Status

- 23 kanban cards on `aqueous-steel` board
- 393 tests passing (consistency, not validation)
- 214 model symbols exported
- Full pipeline: `aq-steel --quick`

**Immediate next action:** Tier 0 archaeology. Read the Bureau of Mines reports, Electra patents, and ΣIDERWIN deliverables. Add cell voltage decomposition and temperature to the model. Run the techno-economic sensitivity. Make the product/feedstock decision. All of this is days, and might reshape everything that follows.

**Then:** Hull cell. $300, one afternoon. Start plating.
