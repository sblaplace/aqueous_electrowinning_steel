# Aqueous Electrowinning for Sustainable Steel Production

[![Research](https://img.shields.io/badge/status-research--in--progress-blue)](https://github.com/)
[![Sustainability](https://img.shields.io/badge/focus-decarbonization-green)](https://github.com/)

> **Low-temperature aqueous electrodeposition of iron and steel from renewable electricity**

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
git clone https://github.com/your-org/aq-steel-electrowinning.git
cd aq-steel-electrowinning

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
python -m models.run_all                    # Full suite (14 steps) + master_report.json + dashboard
python -m models.run_all --quick            # Same but skips heavy pulse frequency sweep

# Or use CLI entry points after pip install -e .
aq-steel --quick                               # full suite
aq-steel-carburization                         # carburization only
aq-steel-carbon-potential                      # carbon potential only
aq-steel-tempering                             # tempering only

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
| `models/technoeconomic.py` | CAPEX, OPEX, levelized cost of iron, sensitivity analysis |
| `models/scenarios.py` | Literature-anchored operating scenarios |

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
- A synthetic, reproducible example with a JSON report and figures.  It does
  **not** claim wet-lab performance or fabricate microscopy data.

The current map is a screening aid rather than a calibrated Hull-cell solver:
it omits edge/shielding effects, electrode kinetics, mass transfer, bubbles,
and conductivity gradients.  Gravimetric output is **apparent** Fe FE until
deposit composition and dry mass are verified; an FE above 100% is retained as
a QA flag rather than hidden.  See `models/README.md` and
`experiments/data/README.md` for the model scope and procedure.

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

## Contributing

We welcome contributions from electrochemists, metallurgists, materials scientists, and sustainability researchers.

1. Open an issue to discuss new ideas or experimental gaps.
2. Submit pull requests for literature additions, modeling code, or experimental protocols.
3. All contributions should follow the proposed experimental matrix where possible.

---

## License

*License TBD.*

---

## Acknowledgments

This research direction builds on foundational work in electrochemical metallurgy and is motivated by the urgent need to decarbonize the steel sector. Feedback and collaboration are highly encouraged.

---

*Last updated: July 2026*