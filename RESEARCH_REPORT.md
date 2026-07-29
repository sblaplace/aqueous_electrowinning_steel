# Aqueous Electrowinning for Sustainable Steel Production: A Comprehensive Technical Exposition

**Version 1.1**  
**Date:** July 2026  
**Status:** Research Proposal & Technical Roadmap

---

## Abstract

Traditional steelmaking via pyrometallurgical routes (BF-BOF and EAF) contributes ~7–9% of global greenhouse gas emissions. While hydrogen-based direct reduced iron (H₂-DRI) and molten oxide electrolysis (MOE) offer promising decarbonization pathways, both operate at extreme temperatures (>1000 °C) with significant materials and thermal management challenges.

This report presents **aqueous electrowinning** as a radically different, low-temperature (25–90 °C) paradigm for electrodepositing iron and steel directly from renewable electricity. By leveraging aqueous electrolytes, the approach bypasses high-temperature refractories and enables modular, decentralized production. Key scientific and engineering challenges—including hydrogen evolution reaction (HER) competition, electrolyte stability, carbon/alloy incorporation, and morphology control—are systematically analyzed. A four-phase experimental roadmap is proposed to bridge the gap between mature electroplating technology and structural-grade steel production. Recent literature (including 2024–2025 advances in acidic anion-rich electrolytes and techno-economic modeling) is integrated to provide an up-to-date research foundation.

---

## 1. Executive Summary & Context

### 1.1 Global Steel Decarbonization Imperative
The iron and steel industry is responsible for approximately 7–9% of global CO₂ emissions, with the majority originating from primary production via the blast furnace–basic oxygen furnace (BF-BOF) route. Electric arc furnace (EAF) recycling of scrap is lower-emission but limited by scrap availability. Emerging alternatives such as green-hydrogen direct reduced iron (H₂-DRI) combined with electric smelting furnaces and molten oxide electrolysis (MOE) represent important progress, yet all involve high operating temperatures that introduce severe engineering constraints.

### 1.2 The Aqueous Electrowinning Opportunity
**Aqueous electrowinning** of iron offers a fundamentally different approach: electrodeposition of metallic iron from aqueous electrolytes at near-ambient to moderate temperatures. This route:
- Directly utilizes renewable electricity without intermediate hydrogen production or high-temperature thermal processes.
- Avoids refractory degradation and complex high-temperature materials requirements.
- Enables potential for decentralized, modular production facilities.
- Can potentially utilize industrial waste streams (spent pickle liquor, steel mill dust, low-grade ore leachates) as feedstock.

However, transitioning from electroplated iron coatings (a mature industrial process) to bulk structural steel requires overcoming thermodynamic and kinetic obstacles, principally:
- Competing hydrogen evolution reaction (HER)
- Electrolyte stability and iron solubility
- Controlled incorporation of carbon and alloying elements
- Achieving dense, low-stress deposits with appropriate mechanical properties

---

## 2. Thermodynamic & Electrochemical Fundamentals

### 2.1 Pourbaix Diagram and Iron Speciation
The standard reduction potential for ferrous iron is:

$$
\text{Fe}^{2+} + 2e^- \rightleftharpoons \text{Fe}_{(s)} \quad E^\circ = -0.440\,\text{V vs. SHE}
$$

Iron speciation in aqueous media is strongly pH-dependent:

- **Acidic media (pH < 4):** Fe²⁺ is stable, but HER (\(E^\circ = 0.00\) V) is strongly favored, leading to low current efficiencies.
- **Neutral to alkaline media (pH 7–14):** Fe(OH)₂ precipitation occurs readily (\(K_{sp} \approx 4.87 \times 10^{-17}\)). Complexing agents (citrate, tartrate, gluconate, glycine, triethanolamine) are essential to maintain soluble iron complexes.

### 2.2 Hydrogen Evolution Reaction (HER) Competition
HER is the dominant parasitic reaction across nearly all aqueous pH ranges because the iron deposition potential lies negative of the reversible hydrogen potential. Consequences include:
- Reduced Faradaic efficiency
- Local pH increase at the cathode surface causing hydroxide precipitation
- Hydrogen embrittlement and porous/dendritic morphology

---

## 3. Key Technical Challenges & Solutions

### 3.1 Electrolyte Stability and Iron Solubility
**Challenge:** High concentrations of free Fe²⁺ are required for industrially relevant current densities, yet uncomplexed salts hydrolyze above pH 6.

**Mitigation Strategies:**
- Chelating ligands (citrate, gluconate) for pH 6–9 operation
- Chloride-based or mixed sulfate-chloride baths for improved conductivity
- Recent acidic routes using concentrated anion-rich electrolytes (e.g., LiCl-based) to stabilize Fe³⁺ and suppress HER

### 3.2 Suppressing Hydrogen Evolution & Controlling Morphology
**Challenge:** Uncontrolled HER leads to embrittlement, sponge-like deposits, and oxide inclusions.

**Mitigation Strategies:**
- High-overpotential cathode materials and pulse-reverse electrodeposition (PRE)
- Organic additives (saccharin, thiourea, polyacrylamide) as levelers and brighteners
- Elevated temperature operation (70–90 °C) to improve mass transport and kinetics
- Agitation strategies (rotating electrodes, jet impingement, ultrasonics)

### 3.3 Achieving Alloy and Composite Steel Properties
**Challenge:** Structural steels require 0.05–1.0 wt% carbon and alloying elements (Ni, Cr, Mn, Mo).

**Mitigation Strategies:**
- Electro-codeposition of carbon nanoparticles or graphene oxide
- Co-deposition of Fe-Ni or Fe-Cr alloys
- Post-deposition gaseous carburization of pure iron deposits

### 3.4 Corrosion-Resistant Anodes
Dimensionally stable anodes (DSAs) based on IrO₂–Ta₂O₅ on titanium, or nickel-cobalt spinel oxides for alkaline systems, are recommended for long-term OER stability.

### 3.5 Energy Efficiency & Techno-Economic Viability
**Target Metrics:**
- Current efficiency: >95%
- Specific energy consumption: <1,500 kWh/t Fe (competitive with EAF)
- Utilization of waste-derived iron sources

Recent techno-economic analyses (Humbert et al., 2024) indicate that **aqueous hydroxide electrolysis (AHE)** offers the best near-term balance between technological readiness and economic performance among electrolytic routes.

---

## 4. Recent Advances & Literature Review (2024–2025)

Significant progress has been made since foundational studies:

### 4.1 Alkaline Aqueous Routes
- **Yuan et al. (2009)** demonstrated hematite particle reduction in concentrated NaOH at 70–114 °C with current efficiencies >90% and energy consumption ~3 kWh/kg Fe. Deposits exhibited unique tetrahedron-shaped twin crystal morphology.
- **Kempler et al. (2025)** provided mechanistic insight into the role of nanoporous Fe₂O₃ and soluble Fe(II) intermediates in accelerating electrodeposition in NaOH(aq) (*ACS Nano*).

### 4.2 Acidic Aqueous Routes (AWARE)
- **AWARE process (2024–2025)**: Acidic electro-Winning in Anion-Rich Electrolytes using concentrated LiCl-based systems achieved Coulombic efficiencies up to 99.8% at current densities up to 1000 mA/cm² and temperatures of 25–80 °C. High impurity tolerance and zero-waste operation represent major breakthroughs (ChemRxiv 2024; follow-up publications 2025).

### 4.3 Techno-Economic Assessments
- **Humbert et al. (2024)** performed a comprehensive comparison of H₂-DRI, AHE, molten salt electrolysis, and MOE. AHE emerged as the most deployment-ready electrolytic technology, while MOE offers advantages in ore flexibility and potential cement by-product valorization (*Journal of Sustainable Metallurgy*).

These advances collectively demonstrate that aqueous electrowinning is transitioning from a laboratory curiosity to a viable industrial pathway.

---

## 5. Proposed Experimental Matrix & Research Protocol

A structured four-phase program is proposed to systematically address the challenges:

### Phase I: Electrolyte Formulation & Voltammetry (Screening)
- Variables: pH (3–12), temperature (25/50/75 °C), iron source (FeSO₄/FeCl₂), complexants (citrate/glycine)
- Methods: Cyclic & linear sweep voltammetry on RDE, EIS, Tafel analysis
- Output: Electrochemical windows, exchange current densities, HER suppression metrics

### Phase II: Hull Cell & Galvanostatic Deposition
- Variables: Current density (10–100 mA/cm²), additives, agitation
- Methods: Hull cell tests, long-duration galvanostatic runs
- Output: Faradaic efficiency, SEM-EDS morphology, grain size analysis

### Phase III: Carbon and Alloy Co-Deposition Trials
- Variables: Carbon particle loading (0.1–5 g/L), surfactants, hydrodynamics
- Methods: Composite plating, post-carburization trials
- Output: Vickers hardness, tensile properties, XRD phase analysis

### Phase IV: Anode Durability & Closed-Loop Integration
- Variables: DSA lifetime, ore leaching integration, impurity build-up
- Methods: Accelerated life testing, CSTR closed-loop trials
- Output: Anode degradation rates, purification protocols, techno-economic validation

---

## 6. Comparison with Alternative Decarbonization Routes

| Route                  | Operating Temp. | Energy Intensity | Technology Readiness | Key Challenges                     | Aqueous Electrowinning Advantage          |
|------------------------|-----------------|------------------|----------------------|------------------------------------|-------------------------------------------|
| BF-BOF                 | 1500–2000 °C   | High             | Mature               | High CO₂ emissions                 | —                                         |
| H₂-DRI + EAF           | 800–1000 °C    | Moderate         | Pilot/Commercial     | H₂ infrastructure, cost            | Lower temperature, direct electricity use |
| Molten Oxide Electrolysis | 1400–1600 °C | Moderate         | Lab/Pilot            | Refractories, high temp.           | Ambient operation                         |
| **Aqueous Electrowinning** | **25–90 °C** | **Low–Moderate** | **Lab**              | HER, morphology, alloying          | **Lowest temperature, modular**           |

---

## 7. Conclusion & Research Roadmap

Aqueous electrowinning represents a promising low-temperature pathway for sustainable primary iron and steel production. Recent breakthroughs in both alkaline and acidic electrolytes, combined with favorable techno-economic positioning, justify accelerated research investment.

**Immediate Priorities (2026–2027):**
1. Reproduce and extend high-efficiency acidic and alkaline protocols
2. Demonstrate controlled carbon incorporation and mechanical properties
3. Develop detailed process modeling and techno-economic models
4. Identify industrial partners for pilot-scale validation

By systematically executing the proposed experimental roadmap, the research community can advance aqueous electrowinning from conceptual promise toward industrial reality.

---

## 8. References

1. Yuan, B., & Haarberg, G. M. (2009). Electrowinning of iron in aqueous alkaline solution using a rotating cathode. *Metallurgical Research & Technology*.
2. Humbert, M. S., et al. (2024). Economics of electrowinning iron from ore for green steel production. *Journal of Sustainable Metallurgy*, 10, 1679–1701.
3. Shekhar, R., et al. (2025). Nanoporous Fe₂O₃ and soluble Fe(II) intermediates accelerate the electrodeposition of Fe in NaOH(aq). *ACS Nano*, 19(37), 33449–33459.
4. AWARE process authors (2024). Sustainable and highly efficient production of high-purity iron from oxide ores by acidic electrowinning in anion-rich electrolytes. *ChemRxiv*.
5. Additional foundational and review literature available in the `references/` directory.

---

*This document serves as the primary technical reference for the research repository. Contributions and updates are welcome.*