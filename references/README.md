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

### Product-price & downstream benchmarks (product_ladder.py anchors)

10. **USGS Mineral Commodity Summaries 2025** — *Iron and Steel* (ore-based metallics, scrap, and DRI/HBI price context).
11. **Public trade-press price bands (CRU / Fastmarkets class), 2024–26** — rebar/merchant bar, hot-rolled coil, non-oriented electrical steel. Screening bands for `FLAKE_FEED_PRICE_T`, `REBAR_PRICE_T`, `HRC_STRUCTURAL_PRICE_T`, `MAGNETIC_FOIL_PRICE_T`; replace with quoted offtakes before decisions.
12. **Höganäs AB**, *Iron and Steel Powders for Sintered Components* (handbook series) + **MPIF** industry reviews — PM iron-powder pricing, sizing, and finishing practice (`PM_POWDER_PRICE_T`, `PM_FINISH_*`).
13. **Historical electrolytic iron** — USBM RI-series (see §7: RI-5371/RI-7477/RI-8019) and surviving niche suppliers (electrolytic iron powder/flake list prices, Goodfellow-class vendors) — the niche where aqueous iron EW has actually persisted (`PM_POWDER_PRICE_T`, `LOWC_FOIL_PRICE_T`).
14. **Foundry/induction-melting handbooks** (coreless induction furnace energy, 0.50–0.65 MWh/t Fe) — `INDUCTION_MELT_KWH_T`.
15. **Weiss, V. (ed.),** *Steel Rolling Technology* / rolling-mill operating cost literature — `CAST_ROLL_*`, `SKINPASS_*`.
16. **Form Energy** public materials on iron-air battery chemistry — context for the speculative `BATTERY_IRON_PRICE_T` parity band.
17. **Turkdogan, E. T.,** *Fundamentals of Steelmaking*; IISI EAF mass-balance practice — `melt_balance.py` oxide carbo-reduction, `EAF_OXIDE_RECOVERY_FRAC`, `LIME_PER_S_KG`, `BASE_GANGUE_SLAG_KG_T`.
18. **EAF dust composition surveys & DRI/HBI product specifications** (industry handbooks; metallization 94–96 %) — `DUST_FE_FRACTION`, `DUST_CAPTURE_FINES_*`, `DRI_O_WT_PCT`.
19. **Stern, M. (1955),** *J. Electrochem. Soc.* 102 — mixed-potential corrosion of iron in acid media (Tafel slopes, passivity); **Kelly, E. J. (1965),** *J. Electrochem. Soc.* 112 — the active iron electrode — `deposit_corrosion.py` anchors `FE_ACID_JCORR_REF_UA_CM2`, `FE_ACID_ANODIC_TAFEL_MV_DEC`, `FE_ACID_HER_TAFEL_MV_DEC`, `FE_CORR_EA_KJ_MOL`; plus standard aqueous-diffusivity tables for `O2_DIFFUSIVITY_25C_M2_S` (companion to the Weiss (1970) solubility anchor in `bath_startup.py`).
20. **USBM RI-series chloride iron-EW flowsheets** (see §7 above: RI-7477 et al.) — ferric etch (`2Fe³⁺ + Fe → 3Fe²⁺`) as the classical current-efficiency killer — `deposit_corrosion.py` anchors `FE3_ETCH_*` (SPECULATIVE screening; cross-checked against FeCl₃-class iron etching practice).
21. **Landau, L. D., & Levich, V. G. (1942)** — dragging of a liquid by a moving plate (withdrawal film law `h_f = 0.94·l_c·Ca^{2/3}`) — `rinse_carryover.py`; with tankhouse Cu/Ni counter-current rinse-ratio & cascade practice and film-coating web-drainage practice for the `RINSE_*`, `DRAIN_RETENTION_FRAC`, `POWDER_CAKE_LIQUOR_FRAC`, `BATH_CONDUCTIVITY_MS_CM` and `RINSE_ENDPOINT_US_CM` screening anchors.

## Contributing

Add BibTeX entries or PDFs here. Update the main `RESEARCH_REPORT.md` references section when adding new literature.
