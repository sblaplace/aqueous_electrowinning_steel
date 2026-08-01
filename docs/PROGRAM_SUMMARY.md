# Program Summary — Aqueous Iron Electrowinning

## Decision statement

The models identify a plausible operating and deployment window: aqueous iron electrowinning from dissolved or readily soluble waste streams could become competitive with hydrogen direct-reduced iron (H2-DRI), particularly where feedstock transport and disposal liabilities favor deploying at the feedstock source. This is a testable hypothesis, not an established cost or energy advantage.

The target product is a reconfigurable, redeployable production platform rather than a
single frozen flowsheet. Process selection must preserve a common balance of plant and expose
validated runtime recipes where possible, with membrane, electrode, purification, harvesting,
and gas-handling functions implemented as replaceable wet-end modules where chemistry or
safety prevents runtime switching. Candidate ranking must report reconfiguration and
redeployment burden alongside electrochemical and economic performance.

The same deployable unit is qualified inside a fixed, instrumented proving ground at the home
site. That zone deliberately supports controlled excursions beyond the known operating envelope
to expose failure boundaries and drive module, interlock, and recovery-design changes. Only a
conservative, versioned subset of the resulting qualified envelope is field-approved; the
proving ground retains the heavy containment, analytical, and recovery equipment.

The next program gate is a controlled experimental dataset: iron deposition, Faradaic efficiency (FE), cell-voltage decomposition, iron quality, electrolyte balance, and component stability in the intended divided sulfate cell at useful current density.

## What the scenario model says

`experiments/data/scenario_comparison_report.json` contains scenario outputs, not calibrated plant forecasts:

| Scenario | Assumed j | Assumed FE | Assumed V_cell | Model energy | Model LCOFe |
|---|---:|---:|---:|---:|---:|
| Optimized alkaline | 200 mA/cm² | 93% | 1.418 V | 1,464 kWh/t Fe | $281/t Fe |
| AWARE acidic comparator | 500 mA/cm² | 99% | 2.485 V | 2,410 kWh/t Fe | $253/t Fe |
| Future target | 400 mA/cm² | 97% | 2.441 V | 2,415 kWh/t Fe | $214/t Fe |

These values are conditional on assumed voltage, FE, utilization, component cost, electricity price, feedstock cost, and operating costs. They are not a demonstration that the proposed sulfate route beats H2-DRI or that it produces steel at those costs. The model reports iron cost; comparisons with H2-DRI + EAF must make product scope and finishing energy explicit.

For Fe²⁺ + 2e⁻ → Fe, DC electrolysis energy is:

\[
E = 959.9 \times V_{cell}/FE \quad \mathrm{kWh/t\ Fe}.
\]

Electricity, feedstock/logistics, capital utilization, separations, reagent recovery, labor, component lifetime, and product finishing are material cost drivers. They must be measured or treated as explicit sensitivities rather than rounding error.

## Cell architecture and the capital constraint

For an electrowinning process the binding constraint is installed cell cost per
m², not electricity. `models/cell_architecture.py` screens five reactor types —
plate-and-frame, rotating cylinder, drum-and-strip, moving belt, fluidized bed —
against literature Sherwood correlations, explicit practical current ceilings,
and a harvest duty cycle in which batch downtime grows as plating rate rises.

Running the zinc-tankhouse benchmark (500 A/m²) through iron's Faraday
arithmetic gives 3.9 t/(m²·yr), so the program's stated "~5×" requirement is
19.5 t/(m²·yr). Of the screened architectures only the continuously scraped
rotating cylinder clears it, at ~39 t/(m²·yr) and ~$5/t Fe of cell capital
charge. The batch plate-and-frame baseline assumed in `technoeconomic.py`
reaches ~0.66× the benchmark once harvest downtime is counted.

The screen also makes kill criterion #3 computable rather than rhetorical:
`max_affordable_cost_per_m2` returns `budget × productivity / CRF`. At a $60/t
Fe capital-charge budget, a cell at 39 t/(m²·yr) may cost ~$25,000/m² while one
at 2.6 t/(m²·yr) may cost only ~$1,600/m². **Productivity is the lever, not
cell price.**

Three qualifications keep this from being a decision:

1. The rotating cylinder yields powder only — a feedstock-path (Option A)
   answer, not a product-path one.
2. Drum-and-strip is the only screened route to continuous coherent foil, and
   it depends on an assumption that had no experimental support: that **iron
   peels from a titanium drum**. `models/adhesion_peel.py` now screens it (see
   below); a peel-coupon test remains nearly free alongside the Day-1 Hull cell
   order and should be added to it.
3. All correlations are transferred from other chemistries and all costs are
   engineering estimates. This is screening evidence, not measurement.

## Deposit adhesion and internal stress (the continuous-foil branch)

`models/adhesion_peel.py` and `models/internal_stress.py` compute what the architecture screen declined to:
whether the deposit releases and what internal stresses drive that release. `adhesion_peel.py` treats peeling as an energy balance — stored
elastic energy `G = (1−ν)σ²h/E` against interfacial toughness
`Γ = W_adh × φ_plastic × roughness × f_H` — with two failure modes that adhesion alone cannot express: the web tearing
under its own peel force, and the crack abandoning the interface to run
through the deposit when the interface is tougher than the film.

`internal_stress.py` provides the deposit stress model (Stoney / bent-strip) that feeds this energy balance, decomposing residual stress into Hoffman grain coalescence, hydrogen effusion, and thermal mismatch, and computing exact two-layer laminate finite-thickness corrections alongside a full GUM standard-uncertainty budget.

At the drum's 25 µm target on a low-hydrogen deposit, the reference passive
TiO₂ surface returns a controlled peel at ~5 N/m, the metallic negative
controls (copper, depassivated titanium) come back bonded at 400–550 N/m, and
a PTFE release coating is rejected for being electrically insulating rather
than for its release behaviour. Critical self-delamination thickness on the
reference surface is ~187 µm, which bounds foil thickness from above
independently of any winder.

The operationally important result is not the substrate ranking. Propagating a
real operating point through the existing models — 100 mA/cm², 85% FE, 15 min,
giving 28 µm carrying ~240 ppm diffusible hydrogen — flips the verdict to
spontaneous delamination, with hydrogen contributing 373 of 414 MPa of
residual stress. **Hydrogen management, not drum surface selection, is the
lever on peelability**, which ties this branch back to the same HER problem
that governs Faradaic efficiency. `internal_stress.py` demonstrates that Pulse Reverse Electrowinning (PRE) and saccharin additive relief can mitigate this stress.

The branch verdict is `proceed_with_coupon_test`. It is not `proceed` because
the outcome moves within the plausible range of the plastic amplification
factor — measured peel work over thermodynamic work of adhesion — which spans
an order of magnitude in the literature and cannot be estimated from first
principles. The suite therefore specifies the replacement measurements: a
$1,750 peel test (`adhesion_peel.coupon_test_protocol`) paired with a $200 Stoney bent-strip coupon-curvature protocol (`internal_stress.coupon_curvature_protocol`) with explicit kill, confirm, and redirect-to-flake decision rules. This is screening fracture mechanics and mechanics; no
iron peel or internal-stress data exists in this repository.

## Deployment hypothesis

The supply-chain model favors on-site modular deployment for 8 of the 9 illustrative feedstocks at its stated default assumptions. At 500 km feedstock distance, it returns $145.67/t Fe centralized versus −$96.42/t Fe decentralized for pickle liquor, and $540.53/t versus $423.45/t for 30% Fe low-grade ore.

Those are scenario outputs from `models/supply_chain.py`, not site quotes. A beachhead decision requires a named site’s assay, annual volume and variability, disposal contract, electricity supply, water and reagent use, permitting, product offtake, and transport route.

## Competitive and IP position

SIDERWIN demonstrates that low-temperature alkaline iron electrowinning has reached a substantial pilot scale; AWARE is a high-FE acidic chloride comparator. Neither validates the proposed sulfate waste-feed process window.

The reviewed Electra disclosures are architecturally distinct from the proposed waste-feed, divided-cell route, particularly US12054837B2’s thermal-reduction and acid-dissolution path. This is preliminary architectural differentiation, not FTO clearance. `FTO_PRELIMINARY_ASSESSMENT.md` identifies the required claim charts, prior-art work, and counsel review.

The ferric/ferrous anode-shuttle concept is an invention hypothesis only. Any patent effort should claim a demonstrated integrated process, not a familiar redox couple in isolation.

## Program gates and dependency order

1. Build claim charts for active Electra family members and search relevant AWARE and sulfate divided-cell filings.
2. Screen the actual sulfate feed surrogate in a Hull cell for deposition window, morphology, and gross plating behavior. **Add the iron-on-substrate peel/adhesion coupon set to this run** (`adhesion_peel.coupon_test_protocol`) — it is nearly free, it gates the entire continuous-foil architecture branch, and it replaces the model's least-constrained parameter with a measurement.
3. Run an instrumented divided-cell matrix to measure FE, V_cell decomposition, crossover, iron speciation, deposit morphology, and impurity behavior versus current density.
4. Demonstrate a continuous run long enough to expose bath drift, membrane fouling, crossover, component degradation, and reagent loss.
5. Calibrate the transport and TEA models only to the measured data; include auxiliary loads and separations.
6. Qualify a specific waste-stream beachhead site and then obtain a jurisdiction-specific FTO opinion before commercialization.

## Decision-grade kill criterion

At j ≥ 300 mA/cm², kill or redesign the route if replicated divided-cell runs cannot sustain FE ≥ 70% and net DC specific energy ≤ 4,000 kWh/t Fe after concentration, temperature, and flow optimization. At 70% FE, the energy threshold corresponds to V_cell ≈ 2.92 V.

Report auxiliary loads separately until a complete process flowsheet supports an AC plant-energy criterion. The primary experimental artifact is a weighed and characterized iron deposit accompanied by a closed charge, mass, and electrolyte balance—not only a photograph of a plate.