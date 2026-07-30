# Tier 0 Archaeology Digest — Iron Electrowinning Prior Art

**Date:** 2026-07-30
**Status:** Complete first pass. Open-web sources only; abstracts and publisher pages, no paywalled full text unless open. Every quantitative claim carries a source in the References section. Items that could **not** be located are explicitly listed in §7 rather than glossed over.

This document executes the "Tier 0 archaeology — $0" work item in `RESEARCH_PROGRAM.md`: read the prior art **before** trusting the models. Its function is to (1) correct stale priors, (2) supply calibration anchors with real measured FE/energy numbers, and (3) decide what still needs library retrieval.

---

## 1. The sulfate divided-cell route has a direct ancestor: the Pyror process

The single most important find. A sulfate iron-electrowinning pilot **already ran for a decade**.

- **Pyror process (Orkla Grube-Aktiebolag, Thamshavn, Norway, 1947–1957).** Developed to valorize copper-bearing pyrite from the Lökken mine. The crucial step was **electrowinning iron from sulphuric-acid solution using iron starting sheets and lead anodes behind diaphragms**. Reported results: **current yield 85%, power consumption 4.25 kWh/kg Fe** [1].
- The process is still being cited as the reference point: a 2024 ACS paper on iron electrowinning from pyrite sulfate solutions describes exactly the 1950s pilot (iron starting sheets, lead anodes, diaphragms) and its failure mode — **Fe³⁺ crossover through the leaking diaphragm caused rough deposits and depressed faradaic efficiency** [2].
- A related modern process embodiment (patent application US20110089045A1) reports iron electrolysis at **70–80 °C, 250 A/m² (~25 mA/cm²), 3.75 V cell, 85% cathodic CE** [3].
- **Modern re-run with membranes:** a 2019 study benchmarked anion-exchange-membrane (AEM) cells against the Pyror porous-diaphragm design. Best AEM: **95% CE at 3.53 kWh/kg Fe**; commercial AEM also beat Pyror by 0.22 kWh/kg. Efficiency collapses at low iron tenor (CE 88% at 10 g/L Fe; side reactions below 5 g/L) and **40 g/L Fe gave the best efficiency/SEC trade-off**; a depletion run stripped 99.99% of the iron in 13 h at 93% CE and 3.71 kWh/kg [4].

**What this means for the program:**

1. Our "novel" sulfate divided-cell route is a **re-invention with better separators** — the flotation line for novelty is membranes + waste-feed conditioning + operating point, not the concept. This matches the FTO assessment's caution and sharpens where claims must live.
2. The Pyror **4.25 kWh/kg = 4,250 kWh/t is above our own kill criterion (≤4,000 kWh/t)** — and it ran at ~85% CE, *above* the 70% kill floor, but at ~25 mA/cm², an **order of magnitude below the required ≥300 mA/cm²**. History says: the CE target is attainable; the current-density target is the actual program risk. Nobody has yet shown sulfate iron EW at 300+ mA/cm² with ≥70% FE. *That* is the white space.
3. Fe³⁺ crossover destroying deposit quality is not our theoretical worry — it is the documented killer. The membrane/divided-cell decision in the program docs is validated by 1950s pilot experience.
4. **40 g/L Fe floor:** plan bath monitoring so the DOE never quietly drifts below ~20–40 g/L Fe²⁺ during long runs, or CE data will confound depletion with kinetics.

---

## 2. ΣIDERWIN pilot anchors (public, verified)

From the project's official results page and public presentations (ArcelorMittal-led, EU H2020 grant 768788, €6.8M, 2017–2023) [5][6][7]:

| Parameter | Value |
|---|---|
| Electrolyte | Suspended iron oxide particles in concentrated NaOH, ~110 °C nominal |
| Scale demonstrated | Cathodes up to **1.25 m²**, intact iron plates (TRL 6) |
| Energy (cell, pilot-confirmed) | **2.7 MWh/t Fe reachable in optimized conditions** |
| Overall process energy | ~3.6 MWh/t (design target); −31% direct energy, −87% direct CO₂ vs BF-BOF |
| Cell architecture | **No membrane/diaphragm between electrodes; 1 cm electrode gap** |
| Current density | ~110 mA/cm² optimized for hematite (CE ~70% for bauxite-residue feed at 138 A/m²) [7] |
| Lab-scale antecedent | CE >90%, ~3 kWh/kg in rotating-disk lab cell [7] |

Mechanism literature now converges on dissolution/redeposition at particle contact points (nanoporous Fe₂O₃ and soluble Fe(II) intermediates accelerate deposition — cf. Kempler/Shekhar 2025, already in `FTO_PRELIMINARY_ASSESSMENT.md`) [8].

**Implications:**

- Alkaline suspension electrolysis is **rate-limited near ~100 mA/cm²** (particle contact area, slow electron transport in oxide particles, bubble trapping in viscous slurry, impurity intolerance — MgO/Al₂O₃/TiO₂ block conversion; SiO₂ gels the electrolyte) [8]. The same source that confirms SIDERWIN's 2.7 kWh/kg pilot number also states this rate ceiling. So: SIDERWIN validates the *energy* thesis of aqueous electrowinning and simultaneously documents *why* a dissolved-feed acidic/neutral route can win on current density — which is precisely our value proposition.
- Their **separator-free, 1 cm-gap cell** is a CAPEX benchmark: any membrane we add must be justified in $/m² against simply engineering around crossover the way they did in alkaline chemistry.

---

## 3. AWARE acidic route anchors

Electrochimica Acta (2025) results [8]:

- >99% Fe deposit purity in **acidic** solution, **batch and continuous** modes
- **FE up to 99%**, energy **as low as 2.7 kWh/kg Fe**, operation **up to 1,000 mA/cm²**, high impurity tolerance
- **5-hour continuous runs** using two different iron ores and their real leach solutions

**Implications:**

- AWARE is now the **front-runner on the exact cell-performance axes our kill criterion measures** (it passes FE ≥70% and energy ≤4,000 kWh/t at ≥300 mA/cm² on published numbers). The competitor question in `RESEARCH_PROGRAM.md` ("if Electra or ΣIDERWIN is at our target economics → pivot") should now read "AWARE": our differentiation must come from **waste-feed economics + sulfate chemistry + continuous mode beyond 5 h**, because the acidic high-j window itself is occupied in the literature (note: occupied in *publications* — the patent position is open; see `CLAIM_CHARTS_PRELIMINARY.md` §5).
- 5 h is the published continuous-run bar to beat for a sulfate route's long-run durability claim. Phase IV planning should treat **>5 h continuous with closed balances** as the publishable differentiator.

---

## 4. Industrial electrolytic iron practice (corrects a prior)

`RESEARCH_PROGRAM.md` notes "Chinese electrolytic iron practice runs 50–60 °C." The published practice is **chloride, hotter**:

- FeCl₂ electrolyte, **75–100 °C, pH ~2.5, 0.1–0.2 A/cm²**, soluble anodes (pure iron / low-carbon steel / scrap), **cathodes of titanium or Ti alloy**, product **99.8% Fe foils 20–100 µm thick**; another documented industrial set: 2 M NaCl, **95 °C, pH 3–4, 1–2 A/dm², pure-iron anodes, Ti cathodes, 99.983% Fe** [9].
- Commercial Chinese electrolytic iron flake is a **commodity product at 99.9–99.99% purity, 0.05–2.0 mm**, sold into alloy smelting, magnetic materials, powder metallurgy, and chemical synthesis [10][11].

**Implications:**

1. **Correct the temperature prior:** industrial electrolytic iron practice runs **75–100 °C in chloride**, not 50–60 °C. Hot is how industry buys conductivity, transport rate, and low stress. Our DOE upper level of 65 °C (see `experiments/data/factorial_doe_matrix.csv`) is conservative relative to practice — fine for Phase I/II stability, but the 75–90 °C chloride-comparative window should be on the Tier 1/2 list.
2. **Titanium cathodes are the industrial standard for stripable iron** — the equipment list's 316 SS Hull panels are fine for screening, but a Ti coupon pair should be on the Tier C/D list, and the "product = easily-stripped flake" feedstock thesis is directly validated by an existing commodity market.
3. There is **already a merchant market for electrolytic iron flake at high purity** — the "feedstock beachhead" in `RESEARCH_PROGRAM.md` has a named product category and price discovery is possible by RFQ, not modeling.

---

## 5. Anomalous codeposition canon (grounds `models/co_deposition.py`)

The hydroxide-suppression mechanism variant in `co_deposition.py` is not hypothetical; it is the founding model of the field, with quantified successors:

| Work | Anchor finding |
|---|---|
| Brenner, *Electrodeposition of Alloys* (1963) | Coined/compiled anomalous codeposition of the iron group [12] |
| **Dahms & Croll, JES 112 (1965) 771** | RDE evidence that Ni discharge is suppressed only when **surface pH rises enough to form iron hydroxide**; derived surface-pH equation vs bulk pH, H₂ rate, buffer, transport [13] |
| Hessami & Tobias, JES 136 (1989) 3611 | Mathematical RDE model for anomalous Ni–Fe codeposition [14] |
| Matlosz, JES 140 (1993) | Competitive **adsorbed-intermediate** model — anomaly possible **without** a surface-pH rise [14][15] |
| Gangasingh & Talbot, JES 138 (1991) 3605 | **Boric acid does not act as a buffer** during NiFe codeposition and does not eliminate the anomaly; borate complex seen at −1.0 V vs SCE [16] |
| Deligianni & Romankiw, IBM J. Res. Dev. 37 (1993) 85 | **In-situ surface pH measurement** during plating (rotating pH electrode) [14] |
| Yin, Wei, Fu, Popov, Popova & White, J. Appl. Electrochem. 25 (1995) 543 | Mass-transport effects with additives [14] |

**Implications for the models and the lab:**

- The repo's three `co_deposition.py` variants (hydroxide suppression / intermediate adsorption / mixed-metal intermediate) **map one-to-one onto this canon** — Dahms-Croll ↔ hydroxide suppression, Matlosz ↔ intermediate adsorption. Model output should be discussed against named literature mechanisms, not generic "anomaly."
- **Gangasingh–Talbot's boric-acid result directly tempers the equipment-list chemistry**: keep boric acid for its real roles (surface-buffering lore aside: complexation/pit suppression), and do **not** rely on it to control surface pH in the DOE. Surface pH must be *computed* (our transport model) and, Tier 1, *measured* RDE-style (Deligianni–Romankiw method is the template).
- Prediction worth carrying into Phase III: on Dahms–Croll physics, **raising forced convection and current density should move sulfate Fe–Ni deposits toward normal codeposition**; our screen should log which mechanism regime each run sits in.

---

## 6. Pulse plating of iron: the MEMS/magnetics review closes the open question

`RESEARCH_PROGRAM.md` asked to verify the suspicion that *"nobody has demonstrated pulse plating for iron" is very likely false.* **It is false — extensively.** The knowledge lives in the magnetic-MEMS literature:

- **Pulsed electrodeposition of Fe-Ni-Co alloys**, Electrochimica Acta (1994): PC/PR at 20–200 Hz; PR shifts alloy composition (lower Fe, higher Ni at higher peak j and temperature); smoother, brighter deposits at 55 °C and high rotation [17].
- **Lakatos-Varsanyi et al.** — nanostructured **pulsed-current Fe and Fe–Ni** coatings from stabilized sulfate/chloride electrolytes; optimized **t_on = 1 ms, t_off = 100 ms, J_peak = 800 mA/cm²** [18].
- **Smistrup, Tang & Møller (ECS, 2007)** — pulse-reversal permalloy for MEMS: **saccharin-free, low-stress** electrolyte; **5-sulfosalicylic acid as photometric Fe³⁺ assay** — a directly stealable bath-QA technique [19].
- **Bernasconi et al. (ECS MA 2023)** — reverse-pulse plating of **thick, crack-free** magnetic layers for MEMS [20].
- **Zoia, Cesaro, Bernasconi & Magagnin (2024)** — two-part **review of reverse pulse plating of magnetic alloys**: RPP reduces internal stress, modifies microstructure/morphology, enables thicker crack-free layers [21].
- Plus ~40 years of DC permalloy practice (IBM Romankiw school) and current work on thick nanocrystalline permalloy foils.

**The honest nuance:** this canon is **thin-film** (commonly <10 µm; LIGA-class thick work reaches tens-to-hundreds of µm), magnetics-driven, and mostly Ni-rich. **Thick (>100 µm), structural-grade, high-FE pulse plating of *pure iron* as an electrowinning practice is genuinely under-demonstrated** — that specific white space in our Phase III pulse work survives the archaeology. What transfers immediately: waveform parameter ranges, stress-management rationale, saccharin-free low-stress bath design, and the sulfosalicylic-acid Fe³⁺ assay.

---

## 7. Not located — library/retrieval list

These were **not** found in open search; listed so nobody mistakes omission for verification:

1. **USBM RI-series iron electrowinning tables.** The RI stacks are searchable at `https://stacks.cdc.gov/` (NIOSH-maintained USBM archive). Open-web search surfaced adjacent copper EW RIs (e.g., RI 8076, RI 9348) but **no publicly indexed iron-EW RI**. The canonical pre-war pilot record for sulfate iron EW in the open literature is the Pyror account [1] — treat it as the substitute anchor unless a USBM RI is retrieved.
2. **Fedot'ev (N.P.) and Soviet electroforming literature** on thick iron deposits; **B. Cohen** internal-stress papers. Monograph/journal archive material; needs library retrieval. One adjacent find on thick iron: *Electroforming of iron foil* (1978) — CE and foil quality vs j/T/pH, Fe electrodeposition demonstrated up to **120 A/dm² (cracked)** [22].
3. **Di Bari chapter specifics.** Di Bari's *Modern Electroplating* chapter is the nickel treatment; the iron-group content (saccharin stress relief, boric acid lore, chloride vs sulfate) is corroborated independently by the permalloy literature [19] and §5, but the chapter itself should be read before citing details. Verify attribution on retrieval.
4. **SIDERWIN D-series public deliverables.** Several project outputs are marked ArcelorMittal-confidential (per presentation footers [6]); the public results page [5], the MDPI proceedings paper [7], and follow-on papers/theses from project partners are the accessible track.

---

## 8. Calibration anchors for the models

Measured literature numbers the TEA/kinetics outputs should be **sane-checked against** (FE = faradaic efficiencies; SEC = specific energy consumption):

| Source | Chemistry | j | FE/CE | SEC | Notes |
|---|---|---:|---:|---:|---|
| Pyror pilot 1947–57 [1] | Sulfate, diaphragm, soluble ... lead anodes | ~25 mA/cm² [3] | 85% | 4.25 kWh/kg | Fe³⁺ crossover killed deposit quality |
| US20110089045A1 [3] | Sulfate-class, divided | 250 A/m² | 85% | (3.75 V cell) | 70–80 °C |
| AEM vs Pyror 2019 [4] | Sulfate + AEM | — | **95%** | **3.53 kWh/kg** | CE falls <20 g/L Fe; 40 g/L optimal |
| ΣIDERWIN pilot [5] | Alkaline suspension, 110 °C | ~110 mA/cm² | ~70% (BR feed) | **2.7 kWh/kg** cell | 1.25 m² plates; no separator |
| AWARE 2025 [8] | Acidic anion-rich | up to **1,000 mA/cm²** | **≈99%** | **2.7 kWh/kg** | 5 h continuous, real leach solutions |
| Yuan et al. 2009 (README) | Alkaline suspension | — | >90% | ~3 kWh/kg | Lab rotating-disk antecedent to SIDERWIN |

Program relevance check: our kill criterion (**≤4.0 kWh/kg at FE ≥70%, j ≥300 mA/cm²**) sits *between* the historical sulfate record (4.25 @ ~25 mA/cm² — fails on both energy and j) and the AWARE publication (2.7 at 99%/1000 mA/cm² — passes, but not sulfate, not waste-leach-continuous). The measured-prior distribution says the criterion is **ambitious but not ahistorical**; nobody has hit it in sulfate at 300 mA/cm². That is the experiment to run.

---

## 9. Prior updates triggered by this archaeology

| Program prior (location) | Archaeology finding | Action |
|---|---|---|
| "Chinese practice 50–60 °C" (`RESEARCH_PROGRAM.md`) | Industrial chloride practice 75–100 °C, Ti cathodes [9] | Correct the doc; add hot-chloride comparative window to Tier 1/2 |
| "Nobody has pulse plated Fe — probably false" (`RESEARCH_PROGRAM.md`) | Confirmed false for Fe/Fe-alloy films (§6); thick structural pulse plating remains open | Keep Phase III pulse scope; cite MEMS canon |
| Bath relies on boric acid buffering (equipment/shopping lists) | Boric acid is not an effective codeposition buffer [16] | Retain for other roles; do not credit it with surface-pH control in DOE interpretation |
| Sulfate divided-cell concept is novel (`PROGRAM_SUMMARY.md` IP framing) | Pyror 1947–57 sulfate diaphragm pilot; modern AEM re-runs [1][4] | Novelty must live in waste-feed conditioning + membrane operation + demonstrated window; feeds claim-chart work |
| CE ≥70% is the risk (`PROGRAM_SUMMARY.md`) | CE ≥70% is historically normal; **j ≥300 mA/cm² in sulfate** is untested | Shift Phase II attention to current-density scaling and transport |
| Electrolytic iron product is hypothetical (`RESEARCH_PROGRAM.md` revenue path) | Merchant 99.9–99.99% flake is a commodity [10][11] | Feedstock beachhead can do price discovery by RFQ |

---

## References

1. E. Mostad, S. Rolseth, J. Thonstad, "Electrowinning of iron from sulphate solutions," *Hydrometallurgy* 90 (2008) 213–220. https://www.researchgate.net/publication/222566309_Electrowinning_of_iron_from_sulphate_solutions
2. "A Novel Electrochemical Process for Recovery of Rare Earth Elements…" *ACS Sustainable Resource Management* (2024). https://pubs.acs.org/doi/10.1021/acssusresmgt.4c00026
3. US20110089045A1, "Electrochemical process for the recovery of iron." https://patents.google.com/patent/US20110089045A1/en
4. "Electrowinning of Iron from Spent Leaching Solutions Using Novel [AEMs]," *Membranes* 9(11) 137 (2019). https://www.mdpi.com/2077-0375/9/11/137
5. ΣIDERWIN official results page. https://www.siderwin-spire.eu/content/results
6. ΣIDERWIN project presentation (ETIPWind): 3.6 MWh/t, −31% direct energy, −87% direct CO₂, no separator, 1 cm gap. https://etipwind.eu/wp-content/uploads/Siderwin.pdf
7. S. Koutsoupa, S. Koutalidi, E. Balomenos, D. Panias, "ΣIDERWIN—A New Route for Iron Production," *Mater. Proc.* 5(1) 58 (2021). https://www.mdpi.com/2673-4605/5/1/58
8. "Sustainable and highly efficient production of high-purity iron [AWARE]," *Electrochimica Acta* (2025). https://www.sciencedirect.com/science/article/abs/pii/S0013468625017244
9. "The Effect of Electrolytic Temperature on the Purity of Electrolytic Pure Iron," *Metals* 15(9) 1055 (2025). https://www.mdpi.com/2075-4701/15/9/1055
10. Stanford Advanced Materials, electrolytic iron flake (commodity listing). https://www.samaterials.com/electrolytic-iron-flake.html
11. BL Pure Iron (China), electrolytic iron flakes 99.9–99.99% (commodity listing). https://www.blpureiron.com/electrical-pure-iron/electrolytic-iron-flakes-manufacturer-china.html
12. A. Brenner, *Electrodeposition of Alloys* (Academic Press, 1963) — cited in [19].
13. H. Dahms, I. M. Croll, "The Anomalous Codeposition of Iron–Nickel Alloys," *J. Electrochem. Soc.* 112(8) 771 (1965). https://iopscience.iop.org/article/10.1149/1.2423692
14. Reference list corroboration in "Effects of Rotating Speed of Rotating Cylinder Electrode…," *Trans. Indian Inst. Met.* (2025). https://link.springer.com/article/10.1007/s13391-025-00571-x
15. Anomalous-codeposition mechanism review context (Matlosz model summary). https://www.sciencedirect.com/science/article/abs/pii/S0925838817315499
16. D. Gangasingh, J. B. Talbot, "Anomalous Electrodeposition of Nickel–Iron," *J. Electrochem. Soc.* 138(12) 3605 (1991). https://iopscience.iop.org/article/10.1149/1.2085466
17. "Pulsed electrodeposition of Fe-Ni-Co alloys," *Electrochimica Acta* (1994) — via pulse-plating review. https://www.sciencedirect.com/science/article/abs/pii/0376458382900553
18. M. Lakatos-Varsanyi et al., "Nanostructured pulsed current metal coatings of Fe and Fe–Ni for microelectronic applications," *CIRP Annals / Trans.* (2017) — via [19] citation list. https://www.researchgate.net/publication/266869714_Pulse_Reversal_Permalloy_Plating_Process_for_MEMS_Applications
19. K. Smistrup, P. T. Tang, P. Møller, "Pulse Reversal Permalloy Plating Process for MEMS Applications," *ECS* (2007). https://www.researchgate.net/publication/266869714_Pulse_Reversal_Permalloy_Plating_Process_for_MEMS_Applications
20. R. Bernasconi et al., "Reverse Pulse Plating of Thick and Crack-Free Magnetic Layers for MEMS Manufacturing," *ECS Meet. Abstr.* (2023). https://ui.adsabs.harvard.edu/abs/2023ECSMA2023.1402B/abstract
21. F. Zoia, R. Cesaro, R. Bernasconi, L. Magagnin, "A review of reverse pulse plating techniques in the electrodeposition of magnetic alloys – Part 1," *Trans. IMF* (2024). https://www.tandfonline.com/doi/abs/10.1080/00202967.2024.2413270
22. "Electroforming of iron foil," *Surface Technology* (1978). https://www.sciencedirect.com/science/article/abs/pii/0378380478900281
23. USBM Report of Investigations archive (for item 7.1 retrieval). https://stacks.cdc.gov/

*Search date: 2026-07-30. All numbers above are as reported by the cited sources; none are measurements of this program.*
