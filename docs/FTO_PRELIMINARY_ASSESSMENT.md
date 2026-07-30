# Freedom-to-Operate (FTO) Preliminary Assessment

**Date:** 2026-07-29
**Status:** Preliminary architectural comparison — NOT a freedom-to-operate opinion or patent attorney opinion.
**Prepared by:** Automated literature and patent search; no legal review performed.

> **DISCLAIMER:** This document is a preliminary internal assessment based on publicly available patent abstracts, claims text, and technical literature. It does not constitute legal advice and has not been reviewed by a patent attorney. The conclusions herein are subject to revision upon professional legal review. Before making investment, production, or commercialization decisions based on this assessment, consult a registered patent attorney.

---

## 1. Competitor IP Landscape

### 1.1 Electra (Elektra Steel, Inc.) — Boulder, CO

**Company profile:** Well-funded startup (~$100M+ raised), stealth-mode, developing aqueous low-temperature electrowinning of iron from ore. Founded ~2018. Key inventor: Ai Quoc Pham.

**Patent families identified (5):**

| Family | Title | Key Publication | Status | Priority |
|--------|-------|-----------------|--------|----------|
| 1 | Impurity removal in iron conversion systems | US12054837B2 (granted) | Active | 2021-03-24 |
| 1 (cont.) | Ore dissolution and iron conversion system | US20240158939A1, US20250146155A1 | Active | 2021-03-24 |
| 2 | Electrochemical metallurgical slag recovery | CN120187870A, KR20250072624A | Active | 2022-09-19 |
| 3 | Stabilized lead dioxide anode | CN120265830A | Active | 2022-09-26 |
| 4 | Iron feedstock conversion with improved efficiency | EP4594541A2 | Active | (div. of Family 1) |
| 5 | High efficiency iron electrowinning | WO2025199035A1 | Pending | 2024-03-20 |

**Family 1 (US12054837B2) — Granted, active to 2042:**
- Covers: thermally reducing non-magnetite iron oxides to magnetite, then dissolving in acid using electrochemically generated protons.
- Scope: Feedstock preparation (ore → dissolved Fe²⁺). Two-cell architecture (dissolution cell + plating cell) with pH-swing impurity precipitation between them.

**Family 5 (WO2025199035A1) — Pending PCT:**
- Covers: "Ferric scrubber" — a method for decreasing Fe³⁺ concentration in the plating catholyte by chemically converting Fe³⁺ to Fe²⁺ using an iron-based metal (and optionally a reduction-catalyst).
- Claim 1 (independent): scrubbing Fe³⁺ → Fe²⁺ in catholyte using iron-based metal, providing scrubbed solution to catholyte, electroplating.
- Key dependent claims specify: rate of Fe³⁺ reduction > H₂ generation rate; handles spontaneous Fe²⁺ oxidation by O₂; rate range 0.05 mM/hr – 5 M/hr.

### 1.2 ΣIDERWIN (ArcelorMittal, EU H2020)

- Grant 768788, €6.8M, 2017–2023
- Alkaline suspension electrolysis of iron oxide in NaOH at ~100–130°C
- Produced 1.25 m² intact iron plate (TRL 6)
- Current efficiency: ~70% at 130°C
- 87% CO₂ reduction vs BF-BOF
- Approach is fundamentally different (alkaline vs acidic sulfate)
- No claims-level comparison against a ΣIDERWIN patent family has been completed; its different alkaline suspension route is an architectural distinction, not an FTO conclusion.

### 1.3 AWARE Process

- Concentrated LiCl electrolyte, acidic pH < 2
- >99% coulombic efficiency at high current density
- Published ChemRxiv (2024), Electrochimica Acta (2025)
- Patents not found in public search (may have filed pre-publication)
- Different electrolyte chemistry (LiCl vs our sulfate)

### 1.4 Allanore & Sadoway (MIT)

- Molten oxide electrolysis (>1600°C) — completely different process
- No FTO relevance to aqueous electrowinning

---

## 2. Our Technical Approach vs. Electra's Patents

| Architectural axis | Electra disclosure / claim focus | Proposed approach | Review implication |
|---------|------------------------------|--------------|------------|
| Feedstock | Ore thermal reduction and acid dissolution | Dissolved or readily soluble waste streams (pickle liquor, copperas, AMD, red mud) | Favorable distinction; confirm against independent claims and continuations. |
| Architecture | Dissolution and plating subsystems | Divided plating cell after separate feed conditioning | Favorable distinction; do not treat this as non-infringement without a claim chart. |
| Fe³⁺ management | Ferric scrubber in catholyte (WO2025199035A1) | Membrane/crossover management and proposed ferric/ferrous mediation | Requires claims-level comparison, including whether any process element is shared. |
| Anode/redox chemistry | PbO₂/OER disclosed in a separate family | Proposed Fe²⁺/Fe³⁺ anode shuttle | Architectural distinction only; prior-art and claim review remain required. |
| Purification | pH-swing precipitation | Proposed cementation and hydrolysis | Confirm actual unit operations and compare with active claims. |

---

## 3. Assessment

**Preliminary conclusion: architectural differentiation appears favorable; FTO is unestablished.**

The proposed aqueous sulfate process—electrowinning from dissolved feedstock in a divided cell—appears architecturally differentiated from the reviewed Electra disclosures, especially the ore thermal-reduction/acid-dissolution path in US12054837B2. This review does not establish non-infringement: active continuations, divisionals, national-phase filings, unreviewed independent claims, and other patent families can alter the conclusion. In particular, this document must not be used as a clearance finding for investment, production, or commercialization.

**Key differentiators:**
1. Waste-stream feedstock (not ore)
2. Divided cell with membrane (no ferric scrubber needed)
3. Fe²⁺/Fe³⁺ anode shuttle (not PbO₂/OER)
4. Cementation purification (not pH-swing)

**Potential IP hypothesis:** A potentially protectable system may be the integrated combination of sulfate waste-feed conditioning, divided-cell architecture, membrane/crossover management, ferric/ferrous anode mediation, and demonstrated operating conditions. The ferric/ferrous couple alone is established electrochemical subject matter; novelty, inventive step, and freedom to practice require a dedicated prior-art search and patent counsel review.

---

## 4. Outstanding Items for Patent Attorney Review

1. Create claim charts for each independent claim in US12054837B2, US20240158939A1, US20250146155A1, and WO2025199035A1; identify every proposed process element and any unresolved element.
2. Confirm whether "iron-based metal" in WO2025199035A1 Claim 1 could reach any planned reductant, catholyte treatment, or iron-mediated ferric-management step.
3. Search for AWARE process patent filings and for sulfuric/sulfate divided-cell iron-electrowinning families.
4. Conduct novelty/prior-art searching for the integrated divided-cell, sulfate waste-feed, and ferric/ferrous-mediation concept before making a patentability claim.
5. Obtain a jurisdiction-specific FTO opinion from registered patent counsel before commercialization, investment, or public claims of clearance.

---

## References

- Patent family table: `../electra_patent_family.md`
- ΣIDERWIN CORDIS: https://cordis.europa.eu/project/id/768788
- Humbert 2024 (TEA): DOI 10.1007/s40831-024-00878-3 — PDF in `references/`
- Kempler/Shekhar 2025 (ACS Nano): DOI 10.1021/acsnano.5c10559
- AWARE 2025 (Electrochimica Acta): DOI 10.1016/j.electacta.2025.147367
