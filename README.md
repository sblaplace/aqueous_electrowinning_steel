# Aqueous Electrowinning for Sustainable Steel Production

[![Research](https://img.shields.io/badge/status-research--in--progress-blue)](https://github.com/)
[![Sustainability](https://img.shields.io/badge/focus-decarbonization-green)](https://github.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

This is currently a **research proposal / conceptual repository** (no experimental code yet).

```bash
# Clone the repository
git clone https://github.com/your-org/aq-steel-electrowinning.git
cd aq-steel-electrowinning

# View the detailed technical report
open RESEARCH_REPORT.md   # or cat RESEARCH_REPORT.md
```

### Next Steps (Planned)
- Electrolyte formulation & voltammetry scripts
- COMSOL / Python electrochemical modeling
- Experimental data logging templates
- Techno-economic model (Python / Excel)

---

## Repository Structure

```
.
├── README.md                          # This file
├── RESEARCH_REPORT.md                 # Comprehensive technical report
├── docs/
│   └── figures/                       # (Future) diagrams, Pourbaix plots, SEM images
├── experiments/                       # (Future) lab notebooks, raw data
├── models/                            # (Future) electrochemical simulations
├── references/                        # Key papers & bibliography
└── LICENSE
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

This work is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

This research direction builds on foundational work in electrochemical metallurgy and is motivated by the urgent need to decarbonize the steel sector. Feedback and collaboration are highly encouraged.

---

*Last updated: July 2026*