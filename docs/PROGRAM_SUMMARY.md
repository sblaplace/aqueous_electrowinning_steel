# Program Summary — Aqueous Iron Electrowinning

## Decision statement

The models identify a plausible operating and deployment window: aqueous iron electrowinning from dissolved or readily soluble waste streams could become competitive with hydrogen direct-reduced iron (H2-DRI), particularly where feedstock transport and disposal liabilities favor deploying at the feedstock source. This is a testable hypothesis, not an established cost or energy advantage.

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

## Deployment hypothesis

The supply-chain model favors on-site modular deployment for 8 of the 9 illustrative feedstocks at its stated default assumptions. At 500 km feedstock distance, it returns $145.67/t Fe centralized versus −$96.42/t Fe decentralized for pickle liquor, and $540.53/t versus $423.45/t for 30% Fe low-grade ore.

Those are scenario outputs from `models/supply_chain.py`, not site quotes. A beachhead decision requires a named site’s assay, annual volume and variability, disposal contract, electricity supply, water and reagent use, permitting, product offtake, and transport route.

## Competitive and IP position

SIDERWIN demonstrates that low-temperature alkaline iron electrowinning has reached a substantial pilot scale; AWARE is a high-FE acidic chloride comparator. Neither validates the proposed sulfate waste-feed process window.

The reviewed Electra disclosures are architecturally distinct from the proposed waste-feed, divided-cell route, particularly US12054837B2’s thermal-reduction and acid-dissolution path. This is preliminary architectural differentiation, not FTO clearance. `FTO_PRELIMINARY_ASSESSMENT.md` identifies the required claim charts, prior-art work, and counsel review.

The ferric/ferrous anode-shuttle concept is an invention hypothesis only. Any patent effort should claim a demonstrated integrated process, not a familiar redox couple in isolation.

## Program gates and dependency order

1. Build claim charts for active Electra family members and search relevant AWARE and sulfate divided-cell filings.
2. Screen the actual sulfate feed surrogate in a Hull cell for deposition window, morphology, and gross plating behavior.
3. Run an instrumented divided-cell matrix to measure FE, V_cell decomposition, crossover, iron speciation, deposit morphology, and impurity behavior versus current density.
4. Demonstrate a continuous run long enough to expose bath drift, membrane fouling, crossover, component degradation, and reagent loss.
5. Calibrate the transport and TEA models only to the measured data; include auxiliary loads and separations.
6. Qualify a specific waste-stream beachhead site and then obtain a jurisdiction-specific FTO opinion before commercialization.

## Decision-grade kill criterion

At j ≥ 300 mA/cm², kill or redesign the route if replicated divided-cell runs cannot sustain FE ≥ 70% and net DC specific energy ≤ 4,000 kWh/t Fe after concentration, temperature, and flow optimization. At 70% FE, the energy threshold corresponds to V_cell ≈ 2.92 V.

Report auxiliary loads separately until a complete process flowsheet supports an AC plant-energy criterion. The primary experimental artifact is a weighed and characterized iron deposit accompanied by a closed charge, mass, and electrolyte balance—not only a photograph of a plate.