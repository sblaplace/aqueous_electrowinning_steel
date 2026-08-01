# Aqueous Electrowinning for Sustainable Steel Production

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Research](https://img.shields.io/badge/status-research--in--progress-blue)](https://github.com/sblaplace/aqueous_electrowinning_steel)
[![Sustainability](https://img.shields.io/badge/focus-decarbonization-green)](https://github.com/sblaplace/aqueous_electrowinning_steel)

> **Low-temperature aqueous electrodeposition of iron and steel from renewable electricity**

This project is developed in the open so that anyone can benefit from both the method and the lessons learned along the way. See [Open Development](#open-development) below.

A research repository exploring **aqueous electrowinning** as a transformative pathway for decarbonizing primary steel production. This approach operates at near-ambient temperatures (25–90 °C), leverages renewable electricity directly, and avoids the extreme thermal and materials challenges of high-temperature routes such as hydrogen-DRI or molten oxide electrolysis.

---

## Overview

Traditional steelmaking via the blast furnace-basic oxygen furnace (BF-BOF) route accounts for ~7–9% of global greenhouse gas emissions. Emerging low-carbon alternatives (green hydrogen DRI, molten oxide electrolysis) still face high-temperature challenges.

**Aqueous electrowinning** offers a fundamentally different paradigm: room-to-moderate temperature electrodeposition of iron from aqueous electrolytes. Success requires overcoming key challenges including hydrogen evolution reaction (HER) competition, electrolyte stability, carbon/alloy incorporation, and achieving structural-grade material properties.

This repository hosts the technical exposition, literature synthesis, and proposed experimental roadmap for advancing the technology from bench-scale electroplating toward scalable, green steel manufacturing.

**Key Highlights**
- Thermodynamic fundamentals and Pourbaix analysis
- Comprehensive mitigation strategies for HER, morphology control, and alloying
- Four-phase experimental protocol (voltammetry → Hull cell → co-deposition → long-run durability)
- Integration of recent literature (2024–2025 advances including AWARE acidic route and techno-economic analyses)

---

## Quickstart

The repository contains a working Python modeling suite (thermodynamics, kinetics, techno-economics) alongside the technical report. No wet-lab data yet.

```bash
# Clone the repository
git clone https://github.com/sblaplace/aqueous_electrowinning_steel.git
cd aqueous_electrowinning_steel

# Install Python dependencies
pip install -r requirements.txt

# Run the models (each writes figures to docs/figures/ and a JSON report to experiments/data/)
python -m models.run_electrochemistry       # Pourbaix diagram + HER-competition kinetics
python -m models.run_technoeconomic         # Base-case CAPEX/OPEX/LCOFe
python -m models.run_scenarios              # Four-scenario comparison
python -m models.run_transport              # Nernst-Planck transport: migration effects
python -m models.run_pulse                  # Transient pulse & pulse-reverse dynamics
python -m models.run_voltammetry            # Synthetic voltammetry sweep, Phase I analysis & Tafel fitting
python -m models.run_eis                    # Synthetic EIS spectrum & Randles fitting
python -m models.run_hull_cell              # Phase II angled-panel current screen + gravimetric FE example
python -m models.run_co_deposition          # Phase III Fe–Ni/carbon co-deposition screen + pulse-coupled
python -m models.run_mechanical_properties  # Phase III → mechanical: YS/UTS/HV/grade
python -m models.run_carburization          # Post-deposition carburization: case depth, HV profile, energy
python -m models.run_carbon_potential       # Gas carburizing atmosphere: a_C from CO/CO2, CH4/H2, dew point, Acm
python -m models.run_tempering              # Tempering + retained austenite: Ms, KM RA, Hollomon-Jaffe
python -m models.run_closed_loop            # Phase IV anode durability + closed-loop CSTR screen
python -m models.run_cell_architecture      # Cell architecture screen: productivity, $/m², kill criterion #3
python -m models.run_transport_sensitivity  # Sobol GSA of the FE engine -> ranked "which experiment to do next"
python -m models.run_adhesion_peel          # Does iron peel from a drum? Peel window + coupon test
python -m models.run_internal_stress        # Residual stress (Stoney / bent-strip) + coupon curvature protocol
python -m models.run_all                    # Full suite (19 steps) + master_report.json + dashboard
python -m models.run_all --quick            # Same but skips heavy pulse frequency sweep

# Or use CLI entry points after pip install -e .
aq-steel --quick                               # full suite
aq-steel-carburization                         # carburization only
aq-steel-carbon-potential                      # carbon potential only
aq-steel-tempering                             # tempering only
aq-steel-architecture                          # cell architecture screen only
aq-steel-sensitivity                           # Sobol GSA of the FE engine (which experiment next)
aq-steel-adhesion                              # Adhesion/peel screen (the drum-and-strip gating unknown)
aq-steel-stress                                # Internal stress and coupon-curvature protocol

# Run the test suite
pytest tests -q

# Jupyter notebooks
jupyter lab experiments/notebooks/full_workflow.ipynb   # 24-cell end-to-end workflow
# also: phase1_voltammetry, phase2_hull_cell, phase3_co_deposition,
#       phase4_closed_loop, phase5_mechanical, phase6_carburization

# View the detailed technical report
open RESEARCH_REPORT.md   # or cat RESEARCH_REPORT.md
```

### Modeling Suite

| Module | Purpose |
|--------|---------|
| `models/electrochemistry.py` | Faraday's law, cell-voltage decomposition, specific energy |
| `models/pourbaix.py` | Fe–H₂O potential–pH equilibria, hydrolysis boundaries, HER thermodynamic margin |
| `models/kinetics.py` | Butler–Volmer Fe/HER partial currents, mass-transport limits, current efficiency |
| `models/boundary_layer.py` | Local cathode pH, Fe²⁺ depletion, Fe(OH)₂ precipitation, concentration profiles |
| `models/transport.py` | Steady 1-D Nernst–Planck film: diffusion **+ migration**, multi-ion profiles, migration-corrected limiting current |
| `models/pulse.py` | Transient 1-D diffusion-kinetics model for **pulsed (PE) and pulse-reverse (PRE)** electrodeposition |
| `models/voltammetry.py` | Phase I CV/LSV analysis, scan rate estimation, baseline correction, polarization curves |
| `models/tafel.py` | Tafel-region fitting with exchange-current and $R^2$ estimates |
| `models/eis.py` | Equivalent-circuit EIS: Randles/CPE/Warburg models, complex NLLS fitting, $R_{ct}$→$i_0$ conversion |
| `models/hull_cell.py` | Phase II variable-gap angled-panel primary-current screen and gravimetric apparent Fe Faradaic efficiency |
| `models/co_deposition.py` | Phase III anomalous Fe–Ni kinetics, Guglielmi C, **pulse-coupled pH recovery & δ thinning** (`run_at_current_pulsed`) |
| `models/mechanical_properties.py` | Phase III → structural: Hall-Petch grain-size, Ni SS, C dispersion → YS/UTS/HV/elongation + grade mapping |
| `models/carburization.py` | Post-deposition gaseous carburization: Fickian finite-slab, case depth, Maynier HV, tempering flag, energy & composite strength |
| `models/carbon_potential.py` | Gas atmosphere: CO/CO2 Boudouard, CH4/H2, dew-point via WGS, O2 probe, Acm solubility, a_C ↔ C wt% |
| `models/tempering.py` | Tempering + RA: Andrews Ms, Koistinen-Marburger RA, Hollomon-Jaffe P, tempered HV/YS, case tempering, recommended T |
| `models/process_flow.py` | Process block-flow diagrams: ore→leach→cell→wash→carburize→product + recycle/purge, detailed variant |
| `models/anode.py` | OER/CER kinetics, bubble resistance, and anode/full-cell voltage coupling |
| `models/closed_loop.py` | Phase IV charge-throughput anode wear and closed-loop electrolyte CSTR balances |
| `models/experimental_data.py` | Long-form measurement loading, validation, and run summaries |
| `models/campaign.py` | Experimental run-manifest validation, traceability links, and QA report |
| `models/calibration.py` | QA-gated Phase-I LSV kinetic calibration plus optional EIS consistency fitting |
| `models/characterization.py` | Validated SEM/EDS, combustion, and XRD characterization records |
| `models/cell_architecture.py` | Reactor-type screen: plate-and-frame vs rotating cylinder vs drum vs belt vs fluidized bed — Sherwood transport, harvest duty cycle, areal productivity, $/m² and kill criterion #3 |
| `models/diffusion_layer_1d.py` | 1-D Nernst-Planck diffusion layer with borate buffering, surface pH and Fe(OH)₂ criterion — the FE prediction engine |
| `models/purification.py` | Cu/Ni/Zn removal train: cementation, hydrolysis, selective EW, ion exchange |
| `models/technoeconomic.py` | CAPEX, OPEX, levelized cost of iron, sensitivity analysis |
| `models/scenarios.py` | Literature-anchored operating scenarios |
| `models/adhesion_peel.py` | Deposit release mechanics: residual stress (Hoffman + hydrogen effusion + thermal mismatch), energy release rate, work of adhesion with thickness-confined plastic amplification, peel force, web-tear and cohesive-failure criteria → the continuous-foil peel window and its coupon test |
| `models/internal_stress.py` | Deposit internal stress (Stoney / bent-strip): forward/inverse cantilever deflection, exact two-layer laminate finite-thickness correction, GUM uncertainty budget, mechanism decomposition (intrinsic, H, thermal), additive relief (saccharin/chloride), stress evolution σ(h), and the coupon-curvature protocol |
| `models/transport_sensitivity.py` | Saltelli-Sobol global sensitivity of the 1D diffusion-layer FE engine over 10 experimental levers → ranked "which experiment to do next" (first-order S1 + total-order ST for FE/V_cell/surface-pH) |

### Selected Model Results

Fe–H₂O thermodynamics (a\_Fe = 1 M, 60 °C) show iron deposition lies **below the HER
line at every pH** — the HER penalty narrows from ~440 mV in strong acid to ~47 mV in
alkali, which is precisely why alkaline routes are attractive despite Fe(OH)₂ formation.

Galvanostatic kinetics at 100 mA/cm² illustrate that HER suppression is the dominant lever:

| Case | Current efficiency | Deposition rate | Specific energy @2.6 V |
|------|-------------------|-----------------|------------------------|
| Acidic, active cathode (i₀,H = 10⁻² A/m²) | 1.8% | 2 µm/hr | ~138,000 kWh/t |
| Acidic + HER inhibitor (i₀,H = 10⁻⁵ A/m²) | 95.8% | 127 µm/hr | ~2,600 kWh/t |
| Mildly acidic, complexed, 150 mA/cm² | 99.8% | 198 µm/hr | ~2,500 kWh/t |
| Transport-limited, stagnant (0.1 M Fe²⁺) | 6.9% | 9 µm/hr | ~35,900 kWh/t |
| Same bath, agitated (δ 200→20 µm) | 68.6% | 91 µm/hr | ~3,600 kWh/t |

### Transport: Migration Matters in Weakly Supported Baths

`models/transport.py` replaces the linear stagnant-film closure with a steady 1-D
Nernst–Planck solve (diffusion **+ migration**) over Fe²⁺, H⁺, OH⁻, Na⁺ and SO₄²⁻,
closed by pointwise electroneutrality and fast water autoprotolysis.

Iron baths are often run with little inert salt, and there the electric field does
real work — it drags Fe²⁺ inward and lifts the transport limit well above the
Levich value:

| Supporting Na₂SO₄ | t(Fe²⁺) | i_lim / i_Levich |
|---|---|---|
| 0 (unsupported) | 0.40 | **2.00** |
| 0.5 M | 0.27 | 1.34 |
| 2 M | 0.14 | 1.13 |
| 10 M | 0.04 | 1.03 |

The unsupported value of exactly 2.00× is the analytical result for a symmetric
2:2 binary salt, and the heavily supported end recovers pure diffusion — the model
is pinned at both limits.

Adding migration also **overturns the film model's local-pH prediction**. With no
mechanism to resupply protons, the diffusion-only film puts a pH-2 bath's cathode
surface at pH ≈ 11.5; carrying H⁺ transport properly keeps it near pH 2.7 at
100 mA/cm², which is far more consistent with acidic baths plating iron at all:

| j (mA/cm²) | Surface pH (Nernst–Planck) | Surface pH (film) |
|---|---|---|
| 5 | 2.04 | 9.54 |
| 100 | 2.69 | 11.51 |
| 200 | 3.32 | 11.69 |

### Pulse-Reverse Electrodeposition Dynamics

`models/pulse.py` simulates transient 1-D concentration profiles during high-peak-current
pulsed electrodeposition. During the pulse-off and reverse-pulse periods ($t_\text{anodic}$ / $t_\text{off}$),
surface Fe²⁺ concentration recovers and local pH spikes relax, enabling higher peak current densities
without Fe(OH)₂ precipitation or hydrogen embrittlement.

### Phase II Hull-Cell Screen & Gravimetric Efficiency

`models/hull_cell.py` provides the first executable Phase II workflow:

- A variable-gap **primary current** map across a 10 × 5 cm angled panel.  It
  normalizes local strip conductance ($j\propto1/g$) to the applied current,
  so positions can be assigned to current-density windows.
- A canonical galvanostatic trace plus pre/post-weighing schema and
  blank-corrected gravimetric calculation,
  $\mathrm{FE}_{app}=m_{net}/[Q_{cathodic}M_{Fe}/(2F)]$.
- A synthetic, reproducible example with a JSON report and figures (no wet-lab data yet).

The current map is a screening aid rather than a calibrated Hull-cell solver:
it omits edge/shielding effects, electrode kinetics, mass transfer, bubbles,
and conductivity gradients.  Gravimetric output is **apparent** Fe FE until
deposit composition and dry mass are verified; an FE above 100% is retained as
a QA flag rather than hidden.  See `models/README.md` and
`experiments/data/README.md` for the model scope and procedure.

### Cell Architecture: Where the CAPEX Actually Goes

For an electrowinning process, electricity is not the binding constraint —
**installed cell cost per m² is.** At 100 mA/cm² and 85% FE a cell produces
roughly 7.8 t of iron per m² per year, and iron sells for $400–600/t. A zinc
tankhouse, the closest mature analogue, costs ~$1,000–1,500 per annual tonne
of capacity to make a product worth $2,500–3,000/t.

`models/cell_architecture.py` screens five reactor types against that problem,
each with its own literature Sherwood correlation, an explicit engineering
current ceiling, and a harvest duty cycle:

| Architecture | Harvest | Evidence | j (mA/cm²) | t/(m²·yr) | $/m² | $/t Fe |
|---|---|---|---:|---:|---:|---:|
| Rotating cylinder + scraper | continuous | commercial | 548 | 39.1 | 2,288 | 5.48 |
| Drum-and-strip (Cu-foil type) | continuous | commercial | 338 | 12.1 | 3,718 | 28.85 |
| Plate-and-frame (filter press) | batch | commercial | 53 | 2.6 | 858 | 31.27 |
| Fluidized / particulate bed | semi-continuous | pilot | 0.17\* | 7.1 | 2,678 | 35.28 |
| Moving belt + doctor blade | continuous | concept | 74 | 3.7 | 2,288 | 58.22 |

\* per unit of *particle* area; the bed is limited by potential distribution
through its depth, not by film transport.

Two results matter. First, **continuous harvesting wins despite costing more
per m²**: a batch cell's duty cycle falls as its plating rate rises, because
it must stop more often to be stripped. Second, the zinc benchmark run through
iron's Faraday arithmetic is 3.9 t/(m²·yr), so the program's "~5×" target is
19.5 — and only the rotating cylinder clears it.

The screen also makes kill criterion #3 computable. `max_affordable_cost_per_m2`
inverts the question: at a $60/t Fe capital-charge budget, a cell delivering
39 t/(m²·yr) may cost up to ~$25,000/m², while one delivering 2.6 t/(m²·yr)
may cost only ~$1,600/m². **Productivity, not cell price, is the lever.**

Caveats are structural, not cosmetic: the rotating cylinder makes powder only
(a feedstock-path answer), and drum-and-strip — the one route to continuous
coherent foil — rests on the assumption that **iron peels from a titanium
drum**. Copper foil relies on a passive TiO₂ release layer. That assumption is
taken up directly in the next section. All correlations are transferred from
other chemistries and all costs are engineering estimates.

### Does Iron Peel From a Drum? The Foil Route's Gating Unknown

The architecture screen named its own blocker and declined to compute it.
`models/adhesion_peel.py` computes it. Peeling is an energy competition: a
residually stressed film stores `G = (1−ν)σ²h/E` per unit area, and the
interface resists with a toughness `Γ = W_adh × φ_plastic × roughness × f_H`.
Three residual-stress mechanisms are carried separately — Hoffman grain
coalescence (`σ ∝ 1/d`), hydrogen left behind when codeposited H effuses, and
thermal mismatch on cool-down.

At the drum's 25 µm target thickness, on a low-hydrogen deposit:

| Surface | Evidence | σ_res (MPa) | G (J/m²) | Γ (J/m²) | Peel (N/m) | Outcome |
|---|---|---:|---:|---:|---:|---|
| Ti drum, passive TiO₂ | commercial (for Cu) | 96 | 0.77 | 5.8 | 5 | clean peel |
| 316L stainless, passive | commercial | 19 | 0.03 | 43.5 | 43 | clean peel |
| Hard-chromium mandrel | commercial (for Ni) | 121 | 1.23 | 4.0 | 3 | marginal peel |
| Copper (negative control) | lab | 14 | 0.02 | 426 | 426 | bonded, no release |
| Ti, etched/depassivated | lab | 96 | 0.77 | 550 | 549 | bonded, no release |
| PTFE release coating | concept | −855 | 61.5 | 0.05 | 0 | self-releases — **but insulating** |

Four results carry the argument. First, the screen **discriminates**: the
deliberate metallic controls come back bonded, and the polymer release coating
is rejected on physics — it releases perfectly and cannot pass current, so it
cannot be a cathode. Second, adhesion is not the only way to fail. A crack
prefers the cheaper path, so when the interface is tougher than the deposit's
own tearing energy the crack leaves the interface and runs through the foil;
work of adhesion alone cannot predict that, which is why copper-foil experience
does not transfer to iron by analogy. Third, the foil window is **bounded from
both sides** — the deposit must release hard enough to strip and hold on well
enough to be wound — and above a critical thickness of ~187 µm the reference
surface self-delaminates regardless of the winder.

Fourth, and most consequential: propagating a *real* operating point through
the existing models (100 mA/cm², 85% FE, 15 min → 28 µm carrying ~240 ppm
diffusible H via `hydrogen_embrittlement.py`) flips the verdict to spontaneous
delamination, with hydrogen contributing 373 of 414 MPa. **The substrate is not
the main variable — the hydrogen is.** The same HER the program fights for
Faradaic reasons also decides whether the deposit stays on the drum.

The branch verdict is therefore `proceed_with_coupon_test`, not `proceed`: the
outcome moves within the plausible range of the plastic amplification factor,
the one parameter that cannot be estimated from first principles. So the module
ends by specifying the experiment that replaces it — a $1,750, 3-day peel and
coupon-curvature set with explicit kill, confirm, and redirect-to-flake rules,
run alongside the Day-1 Hull cell. Every number above is screening fracture
mechanics; **no iron peel data exists in this repository.**

---

## Repository Structure

```
.
├── README.md                          # This file
├── RESEARCH_REPORT.md                 # Comprehensive technical report
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── docs/
│   └── figures/                       # Diagrams, Pourbaix plots, SEM & pulse dynamics images
├── experiments/
│   ├── data/                          # Raw and processed experimental & simulation data
│   └── notebooks/                     # Jupyter notebooks / script analysis
├── models/                            # Electrochemical & process simulations
│   ├── electrochemistry.py            # Faraday's law, cell voltage, specific energy
│   ├── pourbaix.py                    # Fe-H₂O potential-pH equilibria
│   ├── kinetics.py                    # Butler-Volmer Fe vs. HER competition
│   ├── boundary_layer.py              # Cathode film composition & Fe(OH)₂ solubility
│   ├── transport.py                   # Steady 1-D Nernst-Planck film with migration
│   ├── pulse.py                       # Transient pulse & pulse-reverse electrodeposition
│   ├── voltammetry.py                 # CV/LSV analysis helpers
│   ├── tafel.py                       # Tafel fitting and exchange current estimation
│   ├── eis.py                         # EIS equivalent circuits and spectrum fitting
│   ├── hull_cell.py                   # Phase II Hull-current map and gravimetric FE
│   ├── co_deposition.py               # Phase III alloy/carbon co-deposition screen
│   ├── anode.py                       # OER/CER anode kinetics
│   ├── closed_loop.py                 # Phase IV durability and electrolyte recycle
│   ├── experimental_data.py           # Long-form experimental data loader
│   ├── technoeconomic.py              # CAPEX / OPEX / LCOFe
│   └── scenarios.py                   # Operating scenario definitions
├── tests/                             # Pytest suite for the modeling code
└── references/                        # Key papers & bibliography
```

---

## Program Summary and Detailed Technical Report

For the decision-grade program position, model scope, experimental gates, and preliminary IP framing, see **[docs/PROGRAM_SUMMARY.md](docs/PROGRAM_SUMMARY.md)**.

Pre-lab desk work (July 2026): **[docs/TIER0_ARCHAEOLOGY.md](docs/TIER0_ARCHAEOLOGY.md)** (prior-art anchors and prior corrections), **[docs/CLAIM_CHARTS_PRELIMINARY.md](docs/CLAIM_CHARTS_PRELIMINARY.md)** (Electra claim charts, desk level), and **[docs/FIRST_LAB_DAY.md](docs/FIRST_LAB_DAY.md)** (bath recipe + Hull/FE run protocol).

The full technical exposition is available in **[RESEARCH_REPORT.md](RESEARCH_REPORT.md)**.

It includes:
- Executive summary & context
- Thermodynamic & electrochemical fundamentals
- Key technical challenges & solutions
- Proposed experimental matrix (4 phases)
- Literature review with recent advances (AWARE, alkaline AHE, techno-economics)
- Conclusion & research roadmap

---

## Literature Snapshot (Selected Recent Works)

- **Yuan et al. (2009)** – Foundational alkaline aqueous electrowinning of hematite suspensions (current efficiency >90%, ~3 kWh/kg Fe).
- **Humbert et al. (2024)** – Techno-economic analysis comparing Aqueous Hydroxide Electrolysis (AHE), MOE, and H₂-DRI (*Journal of Sustainable Metallurgy*).
- **AWARE process (2024–2025)** – Acidic electro-Winning in Anion-Rich Electrolytes achieving >99% Coulombic efficiency at high current densities (ChemRxiv / follow-up publications).
- **Kempler et al. (2025)** – Mechanistic studies on nanoporous Fe₂O₃ and Fe(II) intermediates in alkaline NaOH(aq) electrodeposition (*ACS Nano*).

Full references and expanded discussion are in the [RESEARCH_REPORT.md](RESEARCH_REPORT.md).

---

## Open Development

This project is developed in the open so that anyone can benefit — not just from the method, but from the lessons learned in developing it. The thermodynamic models, process architectures, experimental protocols, and design decisions are all here for anyone to use, learn from, and build on.

If you're working on decarbonization, electrochemistry, or autonomous manufacturing — take what's useful. If you find something wrong, better, or missing, pull requests are welcome.

## Contributing

We welcome contributions from electrochemists, metallurgists, materials scientists, and sustainability researchers.

1. Open an issue to discuss new ideas or experimental gaps.
2. Submit pull requests for literature additions, modeling code, or experimental protocols.
3. All contributions should follow the proposed experimental matrix where possible.

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).

The Apache 2.0 license includes an **express patent grant**: any Contributor who submits code also grants a royalty-free patent license for patents necessarily infringed by their contribution. This protects the project's freedom to operate — no one can contribute code and later assert patent rights over it.

> **Inconsistency to resolve:** `pyproject.toml` currently declares
> `license = {text = "MIT"}`, and there is no `LICENSE` file in the
> repository. Until an owner decides, treat the licence as unsettled — the
> metadata claim and this section contradict each other, and neither is
> authoritative without a `LICENSE` file.

---

## Acknowledgments

This research direction builds on foundational work in electrochemical metallurgy and is motivated by the urgent need to decarbonize the steel sector. Feedback and collaboration are highly encouraged.

---

*Last updated: July 2026*