# References

Key papers, reviews, and supplementary literature for aqueous electrowinning of iron and steel.

## Key References

### Techno-Economic Analysis

1. **Humbert, M. S., Brooks, G. A., Duffy, A. R., Hargrave, C., & Rhamdhani, M. A. (2024).** Economics of Electrowinning Iron from Ore for Green Steel Production. *Journal of Sustainable Metallurgy*, 10, 1679–1701. DOI: `10.1007/s40831-024-00878-3` — **CC BY, PDF in repo.**
   - Compares H2-DRI, AHE (aqueous hydroxide), MSE (molten salt), MOE (molten oxide)
   - Key finding: **AHE is best balance of deployment-ready + economics**
   - CAPEX: AHE $632/t CS vs H2-DRI $750/t CS (this work)
   - Specific energy: AHE 2.78 kWh/kg, H2-DRI 3.59 kWh/kg (low estimates)
   - AHE electrolyzer cost: ~$131/t (open stainless tank)
   - H2-DRI OPEX dominated by H₂ price; at $2–3/kg H₂ → OPEX > $400/t (near LME price)
   - Electricity: ~$100/t at $33/MWh and ~3 kWh/kg
   - **Conclusion: CAPEX of electrolyzer is dominant cost. OPEX driven by electricity + labor.**

### Kinetics & Mechanism

2. **Shekhar, R., Mukhopadhyay, S., Sanchez, F., Konovalova, A., Boettcher, S. W., Devaraj, A., & Kempler, P. (2025).** Nanoporous Fe₂O₃ and Soluble Fe(II) Intermediates Accelerate the Electrodeposition of Fe in NaOH(aq). *ACS Nano*, 19(37), 33449–33459. DOI: `10.1021/acsnano.5c10559`
   - Nanoporous hematite → dissolution-redeposition pathway (fast)
   - Dense hematite → reactive fracture (slow)
   - **Nanoscale porosity controls reactivity at <100°C**
   - Implications for feedstock selection: porous oxides are dramatically faster

3. **AWARE (2024/2025).** Sustainable and highly efficient production of high-purity iron from oxide ores by acidic electrowinning in anion-rich electrolytes. *ChemRxiv* preprint (2024): `10.26434/chemrxiv-2024-stwdn`; published in *Electrochimica Acta* (2025): `10.1016/j.electacta.2025.147367`
   - Concentrated LiCl electrolyte, acidic pH < 2
   - >99% coulombic efficiency at high current density
   - HER suppressed thermodynamically by high chloride activity
   - **This is the highest demonstrated FE at practical j — direct competitor to our model's best case.**

4. **J. Phys. Chem. C (2024).** Electrowinning for Room-Temperature Ironmaking: Mapping the Electrochemical Aqueous Iron Interface. DOI: `10.1021/acs.jpcc.4c01867`
   - First-principles Pourbaix diagram for Fe(110) aqueous interface
   - Iron surface always drives toward adsorbate coverage
   - Theoretical overpotentials for terrace and step sites
   - **Validates our Pourbaix model assumptions + provides DFT-calibrated parameters**

### ΣIDERWIN (ArcelorMittal / EU H2020)

5. **ΣIDERWIN Project (2017–2023).** Development of new methodologies for industrial CO₂-free steel production by electrowinning. EU Grant 768788, €6.8M. CORDIS: `https://cordis.europa.eu/project/id/768788`
   - Alkaline suspension electrolysis of iron oxide at ~110–130°C
   - **Produced 1.25 m² of intact iron plate** (TRL 6 pilot, 3m cell)
   - Current efficiency: **70% at 130°C** (from bauxite residue)
   - 87% direct CO₂ reduction vs BF-BOF
   - 31% direct energy reduction
   - Closed-loop electrolyte circulation
   - Economic viability projected ~2030 at earliest
   - 82% carbon reduction when coupled with induction furnace + decarbonized grid
   - Deliverable D3.4 (pilot validation) and D4.1 (LCA) available on CORDIS (EU login required)
   - Publication: Koutsoupa, Koutalidi, Balomenos (2020) — "Production of metallic iron with alkaline electrolysis under low temperatures" — BR2020 Conference

### Alkaline Electrowinning

6. **Yuan, B., & Haarberg, G. M. (2009).** Electrowinning of iron in aqueous alkaline solution using a rotating cathode.
   - >90% current efficiency demonstrated
   - ~3 kWh/kg Fe specific energy
   - Rotating cathode for mass-transport enhancement

### Historical / Bureau of Mines

7. **US Bureau of Mines RI-series reports** on iron electrowinning (1960s–70s)
   - Key reports: RI-3938 (1946, Fe from sulfate), RI-5371 (1957, electrolytic Fe production), RI-7477 (1971, Fe from chloride), RI-8019 (1975, Fe from sulfate)
   - **Not digitized in accessible repositories.** Physical copies may be at USGS Library (Denver) or NTIS. HathiTrust has some Bureau of Mines RIs but not these specific ones.
   - **Action: request via interlibrary loan or visit USGS library.**

### Electroforming & Pulse Plating

8. **Guglielmi, N. (1972).** Kinetics of the deposition of inert particles from electrolytic baths. *J. Electrochem. Soc.*, 119(5), 681.
   - Foundational co-deposition kinetics model
   - PDF in repo: `references/guglielmi_1972.txt`

### Electra (Competitor)

9. **Electra (Boulder, CO)** — Aqueous low-temperature electrowinning of iron from ore
   - Public family map: `../electra_patent_family.md`; preliminary architectural comparison: `../docs/FTO_PRELIMINARY_ASSESSMENT.md`.
   - Key reviewed family member: US12054837B2, *Ore dissolution and iron conversion system* (granted); active continuations and other families remain subject to claims-level review.
   - Company website: `https://electra.earth/` (formerly ElectraSteel). The public documents are useful technical sources but do not establish freedom to operate.

## Contributing

Add BibTeX entries or PDFs here. Update the main `RESEARCH_REPORT.md` references section when adding new literature.
