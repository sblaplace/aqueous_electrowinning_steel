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
python -m models.run_electrochemistry   # Pourbaix diagram + HER-competition kinetics
python -m models.run_technoeconomic     # Base-case CAPEX/OPEX/LCOFe
python -m models.run_scenarios          # Four-scenario comparison

# Run the test suite
pytest tests -q

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

### Next Steps (Planned)
- Nernst-Planck transport and migration (the current release includes a steady film approximation)
- Pulse-reverse electrodeposition (transient) modeling
- Experimental data logging templates and voltammetry parsers

---

## Repository Structure

```
.
├── README.md                          # This file
├── RESEARCH_REPORT.md                 # Comprehensive technical report
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── docs/
│   └── figures/                       # Diagrams, Pourbaix plots, SEM images
├── experiments/
│   ├── data/                          # Raw and processed experimental data
│   └── notebooks/                     # Jupyter notebooks for analysis
├── models/                            # Electrochemical & process simulations
│   ├── electrochemistry.py            # Faraday's law, cell voltage, specific energy
│   ├── pourbaix.py                    # Fe-H₂O potential-pH equilibria
│   ├── kinetics.py                    # Butler-Volmer Fe vs. HER competition
│   ├── technoeconomic.py              # CAPEX / OPEX / LCOFe
│   └── scenarios.py                   # Operating scenario definitions
├── tests/                             # Pytest suite for the modeling code
└── references/                        # Key papers & bibliography
```

---

## Detailed Technical Report

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