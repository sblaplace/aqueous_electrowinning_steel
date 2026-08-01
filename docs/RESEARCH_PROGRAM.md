# Dark Mill — Autonomous Iron Production via Aqueous Electrowinning

## The Vision

A steel mill that runs itself. Modular cell stacks that take in electricity, water, and iron feedstock — and output iron. Each unit is self-monitoring, self-calibrating, self-correcting. Scale by replication, not construction. Deploy next to a wind farm, a mine, or a pickle liquor source.

### Platform requirement: reconfigure in place, redeploy when the feed moves

The program is not selecting one frozen electrolyte and flowsheet. It is selecting a
**reconfigurable production platform** whose process can be changed at the cheapest safe
layer as feedstock, power, product demand, and evidence change.

The platform has three modification layers:

1. **Runtime recipe** — change current or pulse waveform, flow, temperature, pH setpoint,
   reagent dosing, purge/recycle fraction, harvesting cadence, and active stack count
   without opening process equipment.
2. **Replaceable wet-end modules** — swap membrane cassettes, electrodes, cell frames,
   cathode/harvester modules, polishing media, filters, and gas-handling cartridges while
   retaining the enclosure, rectifier, controls, thermal system, pumps, and instrumentation.
3. **Site redeployment** — drain and preserve the chemistry, decontaminate and isolate the
   wet end, transport containerized modules, reconnect standardized power/water/feed/product
   interfaces, assay the new feed, and commission a new validated recipe.

Runtime modification is bounded by the installed materials-of-construction, membrane,
gas-handling equipment, thermal envelope, and safety interlocks. A sulfate-to-chloride
change, for example, is not merely a software setting if it changes corrosion or chlorine
hazards.

### Fixed proving ground, deployable article

The field unit is also the experimental article. At the home site it operates inside a fixed,
instrumented testing zone built to contain leaks, gases, over-temperature events, electrical
faults, precipitation, fouling, and failed harvesting. There, supervised campaigns may cross
the current validated envelope deliberately to find failure boundaries, recover the unit,
modify the weak module or interlock, and repeat. This is how the platform learns its limits;
the laboratory is not a separate benchtop surrogate for the deployable machine.

Configurations move through an evidence lifecycle:

1. **Experimental** — boundary crossing is permitted only in the proving ground under an
   explicit test plan, containment envelope, abort conditions, and independent shutdown.
2. **Qualified** — the configuration has a measured operating envelope, known failure modes,
   recovery procedure, and inspection interval; autonomous optimization may explore inside
   that envelope at the proving ground.
3. **Field-approved** — a conservative subset of the qualified envelope is signed for a named
   hardware bill, feed envelope, site utility envelope, and gas/waste controls. Field operation
   cannot silently widen it.

Every boundary-crossing run must produce useful evidence even when hardware fails: synchronized
raw data, material and energy balances up to the abort, failure classification, affected-module
identity, post-run inspection, recovery actions, and the resulting design or envelope change.
The fixed zone carries expensive containment, analytical, and recovery infrastructure that
should not burden every redeployed unit.

Therefore process candidates are judged on two axes:

- **Performance:** FE, voltage, current density, balance closure, product quality, durability,
  and full-process cost.
- **Option value:** runtime operating range, number and cost of physical swaps, retained
  balance-of-plant fraction, changeover losses and waste, commissioning burden, transport
  envelope, and the feedstocks/products reachable from the same installed platform.

The winner is the smallest common hardware and control substrate that supports the largest
useful validated process envelope. A slightly less efficient chemistry can be the better
platform if it survives feed variability and can be retuned or re-cartridged instead of
stranded.

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

`models/cell_architecture.py` now computes both sides of that sentence. Running
the zinc benchmark's 500 A/m² through the iron Faraday arithmetic gives
**3.9 t/(m²·yr)**, so the target is **~19.5 t/(m²·yr)**. Of the architectures
screened, only the continuously scraped rotating cylinder clears it.

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

**Status: screened.** `models/cell_architecture.py` now evaluates all four
against the plate-and-frame baseline, using literature Sherwood correlations
(Eisenberg–Tobias–Wilke for rotating geometries, turbulent-duct for planar,
Ranz–Marshall for beds), an explicit practical-current ceiling per
architecture, and a harvest duty cycle in which batch downtime grows as
plating rate rises. Run `aq-steel-architecture`.

The screen's answer to the gating question is **yes, provisionally**: the
rotating cylinder reaches ~39 t/(m²·yr) — about 10× the zinc-tankhouse
benchmark and comfortably past the ~5× target — at ~$5/t Fe of cell capital
charge, because continuous scraping avoids the duty-cycle penalty entirely.
The batch plate-and-frame baseline assumed in `technoeconomic.py` manages
only ~0.66× the zinc benchmark once harvest downtime is counted.

Three caveats keep this provisional:

1. The rotating cylinder produces **powder only**. It is a feedstock-path
   answer (Option A), not a product-path one.
2. Drum-and-strip is the only architecture that yields coherent foil
   continuously, and it turns on an unverified assumption: **that iron peels
   from the drum.** Copper foil production depends on a passive TiO₂ release
   layer; iron adhesion on titanium is uncharacterised here. This is the
   single highest-value cheap experiment the screen identifies.
3. Every number is a screening estimate from correlations measured in other
   chemistries, with costs that are engineering estimates rather than quotes.

---

## The Anode Problem (Tier 0, Not Tier 3)

In an undivided sulfate cell, the dominant anodic reaction is not OER — it's Fe²⁺ → Fe³⁺ (E° = +0.77 V, fast, no overpotential). The Fe³⁺ diffuses to the cathode and is reduced back to Fe²⁺. This is a perfect redox shuttle that eats an enormous fraction of current and depresses FE far below anything our kinetics model predicts.

**You need a divided cell.** Membrane cost, membrane ohmic drop at 300–500 mA/cm², crossover, anolyte composition, and acid balance are all first-order to both CAPEX and energy.

Conversely — if you *deliberately* run Fe²⁺/Fe³⁺ as the anode reaction and pipe the Fe³⁺ to an oxidative leach circuit, you drop anode potential by ~1 V and cut energy ~35%. Worth an afternoon on a whiteboard.

**Action:** Add membrane and divided-cell parameters to the model. This changes everything about V_cell and FE.

---

## Missing Physics (Roughly Ordered by Impact)

*Items 1–4 were the original diagnosis and have since been addressed; they are
retained with their resolution so the record shows what changed and where.*

1. ~~**Cell voltage.**~~ **Addressed** — `models/electrochemistry.py`
   (`V_decomposition`, `CellVoltageModel`): E_cathode, E_anode, η_cathode,
   η_anode, IR_electrolyte, IR_membrane, IR_contacts. Energy = 0.96 × V / FE
   remains THE number.

2. ~~**Temperature.**~~ **Addressed** — carried as a first-class parameter
   through kinetics, transport, speciation, thermal balance and TEA. Sets D,
   conductivity, viscosity, solubility, FE, and internal stress.

3. ~~**Divided cell / membrane.**~~ **Addressed** — `models/membrane_transport.py`
   and `models/membrane_fouling.py`: crossover, ohmic drop, acid balance.

4. ~~**Purification circuit.**~~ **Addressed** — `models/purification.py`:
   cementation, hydrolysis, selective electrowinning, ion exchange, with the
   Cu < 0.1% hot-shortness spec enforced. Zinc tankhouses spend roughly as much
   on purification as on the tankhouse itself, so this stays cost-relevant.

5. **Mass-transport limit.** Pulse plating cannot beat it. At 400 mA/cm² average, 30% duty → 1,330 mA/cm² peak. With 2 M Fe²⁺ and δ = 20 µm, i_L ≈ 1,350 mA/cm². You're at the wall. Pulse redistributes flux in time; it does not create it. The route to high j is [Fe²⁺] → 2 M+, temperature → 50–70°C, δ → <30 µm via forced convection. Pulse is a microstructure and pH-recovery tool, not the rate lever.

6. **Feedstock sourcing.** Global spent pickle liquor + copperas: ~3–6 Mt Fe/yr vs 1,900 Mt steel/yr. Beachhead, not thesis. Chase negative-cost feedstocks: acid mine drainage, red mud, pickle liquor you're paid to take. A −$50/t feedstock is worth more to the model than 10 points of FE.

7. ~~**Deposit internal stress.** Still missing. No Stoney/bent-strip model
   exists; `run_mechanical_properties.py` explicitly disclaims texture and
   residual stress.~~ **Addressed** in `models/internal_stress.py` (forward/inverse Stoney, exact two-layer finite-thickness correction, GUM uncertainty budget, mechanism decomposition from plating conditions, additive relief, and the Tier 0 coupon curvature protocol).

8. **Deposit adhesion / release.** Newly surfaced by the architecture screen.
   Continuous harvesting requires the deposit to come off the cathode on
   purpose — and only on purpose. Adhesion is the hinge for drum-and-strip
   and the doctor-blade routes, and nothing in the model set predicts it.

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

- V(j) and FE(j) at 3 temperatures × 3 concentrations × 2 pH. Measure FE by hydrogen volumetry (inverted burette over cathode) and independently close a charge, dry-deposit-mass, and electrolyte iron balance. This establishes both HER loss and iron recovery; it is the core input to the techno-economic model.

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

- At j ≥ 300 mA/cm², replicated divided-cell runs cannot sustain both FE ≥ 70% and net DC specific energy ≤ 4,000 kWh/t Fe after optimizing concentration, temperature, and flow → **kill or redesign the route.** Specific energy is \(959.9 \times V_{cell}/FE\) kWh/t Fe for Fe²⁺ + 2e⁻ → Fe; 4,000 kWh/t at FE = 70% corresponds to \(V_{cell}\) ≈ 2.92 V.
- Report pumps, heating, concentration, filtration, drying, and rectifier losses separately, then establish an AC plant-energy threshold when the flowsheet is defined.
- No weighed and characterized coherent 100 g iron plate (or qualified powder/flake for a feedstock path), together with a closed charge/mass/electrolyte balance → **stop and reassess.** A photograph is a milestone, not sufficient process evidence.
- Cost per m² of a cell that can be stripped continuously > threshold → **pivot to cell architecture work.** The threshold is now computable rather than rhetorical: `cell_architecture.max_affordable_cost_per_m2(productivity, budget)` returns `budget × productivity / CRF`. At a $60/t Fe capital-charge budget (~10–15% of a $400–600/t product), 8% WACC and 25 years, a cell delivering 39 t/(m²·yr) may cost up to ~$25,000/m², while one delivering 2.6 t/(m²·yr) may cost only ~$1,600/m². **Productivity, not cell price, is what the architecture decision buys.**
- If Electra or ΣIDERWIN is already at our target economics → **pivot to complementary niche or license.**

---

## What to Freeze

These modules are modeling a plant whose unit operation hasn't been demonstrated. They're two years early:

- `digital_twin` — real-time model updating of a process we haven't run
- `process_control` — PID loops for a cell we haven't built
- `transient` — startup/shutdown of a plant that doesn't exist
- `supply_chain` — logistics for a product we haven't made
- `lca` — one page of arithmetic once you know kWh/t and grid carbon intensity; does not need a module
- `uncertainty/sensitivity` — Sobol indices over 76 invented priors are sensitivity analysis of a fiction. **Done** — the fix is `models/transport_sensitivity.py` + `models/run_transport_sensitivity.py`: a proper Saltelli-Sobol decomposition of the 1D diffusion-layer FE engine over 10 experimental levers (no invented priors — only controllable/measurable bath, operating, and kinetic parameters), returning ranked "which experiment to do next" guidance for FE, V_cell and surface pH. Run with `python -m models.run_transport_sensitivity`.

**The ratio of epistemic apparatus to input data is the most alarming feature of the program.** A photograph of a 10 × 10 cm iron plate next to a ruler is worth more than the entire uncertainty-quantification suite — for our own beliefs and for anyone we need to convince.

Keep: `technoeconomic`, `transport` (1D diffusion-layer), `hull_cell`, `voltammetry`, `eis`, `pourbaix`, `kinetics`. These are the models that matter before the first deposit exists.

---

## Revised Computational Tiers

### Tier 0 — This week. $0.

- [x] Read Bureau of Mines RI reports on iron electrowinning — substituted: no iron-EW RI located in the open index; Pyror 1947–57 pilot record, the abandoned FerWIN (Cardarelli) application, and modern AEM re-runs documented instead; RI archive on library-retrieval list (`TIER0_ARCHAEOLOGY.md` §1, §7)
- [x] Read Electra patents, ΣIDERWIN deliverables, Allanore/Sadoway — `electra_patent_family.md`, `FTO_PRELIMINARY_ASSESSMENT.md`, `CLAIM_CHARTS_PRELIMINARY.md`, `TIER0_ARCHAEOLOGY.md` §2
- [x] Read Di Bari on electroforming, Cohen/Fedotev on thick deposits — desk-level coverage via archaeology card (TIER0_ARCHAEOLOGY.md §7); library-retrieval list maintained for deep read when lab begins
- [x] Check MEMS/LIGA pulse plating literature for iron — answered: Fe/Fe-alloy pulse plating is extensively demonstrated; thick structural Fe pulse plating remains the open white space (`TIER0_ARCHAEOLOGY.md` §6)
- [x] Add cell voltage decomposition to the model (E_cathode, E_anode, η, IR_membrane) — `models/electrochemistry.py` (`V_decomposition`)
- [x] Add temperature as a parameter (was missing from 76-param registry) — °C parameters carried through kinetics/transport/TEA
- [x] Add divided cell / membrane model — `models/membrane_transport.py`
- [x] Run techno-economic sensitivity: is there a winning corner vs DRI-H2? — done: Monte Carlo on cost model with Pareto front of (electricity price, current density, FE) vs DRI-H2 (card t_f8ec57cd); dark mill site assessments in experiments/data/
- [x] Decision: product or feedstock? (Page 1) — working assumption ratified in this document: Option A, feedstock first

### Tier 1 — After archaeology informs the question.

Desk/model items are marked done when the model exists and is tested; wet-lab
items remain open until there is measured data. The distinction matters: a
model is a hypothesis generator, not evidence.

- [x] 1D diffusion-layer model (replaces phase-field as gating model) — `models/diffusion_layer_1d.py`: Nernst-Planck over Fe²⁺/H⁺/OH⁻/HSO₄⁻/SO₄²⁻/borate, fast homogeneous equilibria, computed surface pH, Fe(OH)₂ criterion; 311 test lines
- [ ] Hull cell experiments for morphology and gross plating-behavior screening — **wet lab**; tooling ready (`models/hull_cell.py`, `docs/FIRST_LAB_DAY.md`)
- [ ] Instrumented divided-beaker cell: polarization, FE, voltage decomposition, crossover, and iron-speciation curves — **wet lab**; this is the gate-2/gate-3 dataset
- [x] Cell architecture paper study (drum, belt, rotating cylinder, filter-press) — `models/cell_architecture.py` + `run_cell_architecture.py`: literature Sherwood correlations per architecture, harvest-continuity duty cycle, $/m² → $/annual tonne, and a computable kill-criterion-#3 threshold
- [ ] RDE + Levich for kinetics/transport separation — **not started**; the measurement that makes `diffusion_layer_1d` calibratable. Highest-value remaining Tier 1 model+experiment pair
- [x] Stoney stress measurement on thin shim — **protocol specified** (`models/internal_stress.py`, `coupon_curvature_protocol`: bent-strip cantilever deflection, finite-thickness correction, GUM budget, and decision rules)
- [x] Purification circuit design (cementation, hydrolysis for Cu/Ni/Zn removal) — `models/purification.py` + `run_purification.py`: four-stage screen (cementation, hydrolysis, selective EW, ion exchange) with Cu spec check and stage costs

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

- 44 kanban cards on `aqueous-steel` board (36 done, 3 todo, 2 blocked, 3 new)
- 859 tests passing, 5 skipped (CadQuery unavailable), fresh run 2026-07-31
- Full pipeline: `aq-steel --quick` (17 steps)
- Tier 0 archaeology done at desk level: `TIER0_ARCHAEOLOGY.md` (calibration anchors §8, prior corrections §9)
- Program gate 1 done at desk level: `CLAIM_CHARTS_PRELIMINARY.md` (feeds counsel; design-around rules adopted in lab packet)
- Lab-ready packet: `FIRST_LAB_DAY.md` + `experiments/data/day1_run_sheet.csv`, `bath_batch_template.json`, `run_manifest_template.json`
- Dark mill digital twin: physics-driven site sizing, parametric 3D CAD, steel grade routing (AISI 1008–8620), 3 site scenarios assessed
- Tier 0 checklist complete at desk level (all checkboxes marked)
- Platform requirement added: reconfigurable/redeployable platform + fixed proving ground (this document, §The Vision)
- `models/deposit_morphology.py` + tests and `scripts/dft_h_adsorption_fe.py` landed in `cfed4d0` — no longer untracked
- Tier 1 desk items closed: 1D diffusion-layer model, purification circuit, cell architecture screen, and Stoney internal stress model. Remaining Tier 1 is wet-lab (Hull cell, divided beaker cell) plus one unbuilt model: **RDE/Levich**
- Test coverage reconciled: `technoeconomic`, `kinetics`, `pourbaix`, `scenarios` and `cell_physics` now have direct tests (they previously had none despite carrying every kill criterion). Only `process_flow` (figure generation) and `supply_chain` (frozen per §What to Freeze) remain untested
- `models/README.md` completed: it documented 30 of 57 modules, now all of them

**Immediate next action:** patent counsel review of the claim charts + order the Tier A/B equipment (`EQUIPMENT_LIST.md`, now split into deployable article vs fixed proving-ground zone).

**Then:** Hull cell per the Day-1 packet. ~$650–920 all-in for the deployable article; fixed-zone items are listed separately. Start plating.

**New, cheap, and high-value from the architecture screen:** an iron-on-titanium
**peel/adhesion coupon test**. The drum-and-strip route is the only screened
architecture that produces coherent foil continuously, and its entire viability
rests on an assumption nobody in this program has tested. Plate iron on a
titanium coupon under Day-1 bath conditions and try to peel it. Cost is
approximately zero on top of the Hull-cell order; the information decides
whether a whole architecture branch stays open.
