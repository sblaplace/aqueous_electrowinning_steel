# Preliminary Claim Charts — Active Electra Family Members vs. Proposed Process

**Date:** 2026-07-30
**Status:** Preliminary desk work product — executes `PROGRAM_SUMMARY.md` program gate 1 (desk portion) and `FTO_PRELIMINARY_ASSESSMENT.md` §4 items 1–3. **NOT a freedom-to-operate opinion, NOT legal advice, NOT reviewed by patent counsel.**

> **DISCLAIMER:** These charts map claim elements read from public patent documents against the process described in this repository's program documents. Element mapping is judgment-laden: claim construction, proviso/transition-phrase effects, continuation practice, national-phase amendments, and prosecution history can all change scope. Every "absent"/"present" assessment herein is a working hypothesis for counsel to verify — none may be relied upon as a non-infringement or patentability conclusion. See §6 for the action list this feeds.

**Proposed process being charted (per `PROGRAM_SUMMARY.md` and `FTO_PRELIMINARY_ASSESSMENT.md`):** electrowinning of iron from *pre-dissolved* waste streams (copperas from TiO₂ sulfate process, spent pickle liquor, AMD, red-mud leachate) in a **divided sulfate cell** at ~pH 2–3, 35–65 °C, targeting ≥300 mA/cm²; purification by cementation/hydrolysis (not pH-swing of an ore leachate); catholyte Fe³⁺ managed by membrane/crossover control; proposed (unbuilt) concepts include Fe²⁺/Fe³⁺ anode mediation and ascorbate-stabilized bath chemistry (`models/bath_startup.py`).

---

## 1. US12054837B2 (granted 2024-08-06; Family 1; priority 2021-03-24) [1]

**Claim 1 as granted — a *system* claim. Element breakdown:**

| # | Claim element (paraphrased from grant text) | Present in proposed process? | Notes |
|---|---|---|---|
| 1a | A **first dissolution tank** for dissolving a first iron-containing ore using a first acid | **Absent** | Feed is pre-dissolved waste sulfate/liquor; no ore dissolution tank. |
| 1b | Dissolution of the ore in the acid forms an acidic iron-salt solution comprising dissolved **Fe³⁺** ions | **Absent** | No ore dissolution step exists to form such a solution. |
| 1c | An electrochemical cell **fluidically connected to the dissolution tank**, with cathode chamber, anode chamber, and **separator** | **Partially present structurally** (divided plating cell) | But ours is not fluidically connected to an ore-dissolution tank. |
| 1d | A **first circulation subsystem** circulating acidic iron-salt solution from the tank to the cathode chamber **and** catholyte back to the tank | **Absent** | No dissolution-tank ↔ plating-cell circulation loop exists or is planned. |
| 1e | Fe³⁺ from the acidic solution **electrochemically reduced to Fe²⁺ at the cathode**, consuming Fe³⁺ | **Absent as claimed purpose** | Our cathode reduces Fe²⁺→Fe⁰; Fe³⁺→Fe²⁺ reduction is not the cell's operating duty. |

**Assessment:** The granted independent claim is a **dissolution-side** system (electrochemically-assisted ore digestion with acid regeneration; thermal reduction appears only in dependent claim 7, and electroplating only in dependent claim 10 "in a separate electrochemical cell"). A sulfate waste-feed, dissolved-feed divided plating cell, as proposed, **does not practice elements 1a/1b/1d/1e** (preliminary). The FTO assessment's acid-dissolution characterization matches *dependent* claims 2–8 (proton generation across separator, thermal reduction at 200–600 °C), not claim 1.

**Watch items for counsel:**
1. Dependent claim 10 pulls "electroplating iron from the first acidic iron-salt solution in a separate electrochemical cell" into a combination — any future Electra continuation could pursue plating-cell scope standalone; **continuation watch recommended** (family includes active US20240158939A1 and US20250146155A1).
2. Family disclosure (D-series aspects in the specification [1]) teaches plating-cell operational envelopes: iron concentration **1–4 M**; termination at **60–70 % iron depletion**; **split feed 25–45 vol% to cathode / 55–75 % to anode**; Fe²⁺ oxidation to Fe³⁺ in the plating-cell anolyte; anolyte+catholyte mixing and recycle to dissolution. These envelopes blanket normal operating windows; they matter for §102/§103 review of any claims *we* draft, not directly for infringement of the granted claim.
3. Prosecution history (Global Dossier) not reviewed — claim construction risk not yet assessed.

---

## 2. US20240158939A1 (pending; "Impurity removal in an iron conversion system") [2]

**Claim 1 as published — a *method* claim. Element breakdown:**

| # | Claim element (paraphrased) | Present in proposed process? | Notes |
|---|---|---|---|
| 1a | Providing a feedstock having an **iron-containing ore** and impurities | **Absent** | Pre-dissolved waste salts/liquors, not ore. Copperas is an industrial byproduct crystalline salt, not an ore — construction question for counsel. |
| 1b | Dissolution subsystem with **first divided electrochemical cell** (anolyte/catholyte/separator) | Structurally similar, **functionally absent** | We have no dissolution subsystem; there is nothing to dissolve. |
| 1c | Dissolving the ore with acid to form acidic iron-salt solution with dissolved **Fe³⁺** | **Absent** | — |
| 1d | Electrochemically reducing Fe³⁺→Fe²⁺ in the first catholyte | **Absent** | — |
| 1e | Treating the iron-rich solution to remove impurities by **raising pH from an initial pH to an adjusted pH, precipitating impurities** | **Present as a generic unit op** | Our conditioning could include partial neutralization/hydrolysis. **This element is process-chemistry-generic; the claim's novelty lives in the combination** with 1a–1d. |
| 1f | Delivering treated solution to a **second electrochemical (iron-plating) cell** | **Present** (divided plating cell) | — |
| 1g | Reducing Fe²⁺→Fe metal at the second cathode; removing Fe metal | **Present** | — |

**Assessment:** The claim requires the **ore-dissolution + electrochemical Fe³⁺ reduction + pH-swing purification + plating chain**. Our waste-dissolved-feed route skips the entire front half. Preliminary read: not implicated by the proposed process as currently scoped. **However**, dependent claims disclose envelope boundaries that overlap our DOE directly:

- Adjusted pH 3–7 / 4–<7 (claims 9–10); initial pH 0.5–1.5 (claim 28); Fe³⁺/Fe²⁺ ≤ 0.1 (claim 30); **plating pH 2–6 and pH decreasing during plating** (claims 31–32); treated liquor Al <1 mM, P <1 mM (claim 18); product Fe with Al <0.1 wt%, P <0.01 wt% (claim 38); pH raised by adding **metallic iron and/or magnetite** (claims 12–17, incl. consuming H⁺, claim 12); colloidal-silica flocculation with polyethylene oxide (claims 23–27).
- If *our* purification ever evolves to "raise pH of an iron sulfate solution to precipitate Al/Ti/P impurities" (a very standard hydromet move), we would hold elements 1e–1g; only the ore-dissolution front half would separate us from a pending claim that may yet be amended toward standalone purification+plating scope. **Flag to counsel; continuation watch.**

---

## 3. US20250146155A1 (pending; "Separation of electrolytic iron from iron-containing feedstock") [3]

**Claim 1 as published — an apparatus *system* claim:**

| # | Claim element (paraphrased) | Present in proposed process? | Notes |
|---|---|---|---|
| 1a | Electrochemical cell: anode, cathode opposite | **Present** | Generic. |
| 1b | An **electrolyte stream comprising an iron-containing feedstock of particles** contacting the cathode, within a channel | **Absent** | Feed is fully dissolved; no solid-particle slurry. (This is ore-particle electrolysis — architecturally closer to ΣIDERWIN's suspension concept than to our dissolved-salt cell.) |
| 1c | A **magnetic field source** providing ≥1 gauss (pref. 0.03–0.3 T) at the cathode surface | **Absent** | No magnetics in the design. |
| 1d | Cell reduces **feedstock particles** to **iron particles** (50–1000 µm) at the cathode *in the magnetic field* | **Absent** | Product is a coherent deposit/flake, not flow-harvested particles. |

**Assessment:** Distinct architecture (particle slurry + magnet-assisted harvesting). Preliminary read: not implicated. **Notable for strategy, not infringement:** dependent claim 35 claims an *iron metal powder per se* defined by **embedded-emissions metrics** (CBAM, ISO 14404, IPCC 2006, World Steel LCI, WRI GHG Protocol, EU Reg. 2018/2066; <0.8 t CO₂/t Fe down to "<0"). Product-by-process claiming of low-carbon iron signals where Electra's continuation strategy is heading: **carbon-attribute claims on the metal product itself**. Relevant both as a blocking-claim risk on powder products and as a pattern our own IP position should reckon with (e.g., defensive publication of our measured carbon-intensity arithmetic per `RESEARCH_PROGRAM.md` LCA note).

Also noted: method dependent claims cover **50–140 °C** operation — the same hot window flagged in `TIER0_ARCHAEOLOGY.md` §4.

---

## 4. WO2025199035A1 (pending PCT; "High efficiency iron electrowinning" — the closest call) [4]

**Claim 1 as published — a *method* claim:**

| # | Claim element (paraphrased) | Present in proposed process? | Notes |
|---|---|---|---|
| 1a | **Scrubbing** a first aqueous acidic solution containing ferric and ferrous ions to *decrease ferric concentration* | **Present in purpose** | Our bath chemistry expressly manages Fe³⁺ (bath_startup.py; ascorbate stabilization; equipment-list ascorbic acid). |
| 1b | Wherein scrubbing comprises **chemically converting Fe³⁺→Fe²⁺** | **Present** | Ascorbate reduction of Fe³⁺ is chemical conversion. |
| 1c | And **contacting the solution with an iron-based metal** | **Absent as designed** | Our ferric management is reductant-chemical (ascorbic acid) and electrochemical (proposed shuttle), neither requires iron-metal contact with the catholyte. |
| 1d | Providing scrubbed solution to a plating catholyte of an iron electroplating cell | **Present** | — |
| 1e | Electroplating iron from the catholyte | **Present** | — |

**Assessment:** Literal claim 1 requires **1b AND 1c**: chemical ferric reduction *accompanied by iron-metal contact*. As currently designed (ascorbate-based), the proposed process reads on 1a–1b, 1d–1e but **not 1c**. This is the family member that most constrains routine lab practice:

**Design-around discipline to adopt NOW (and record in lab notebooks as contemporaneous design-around documentation):**
1. **Do not** store, condition, or circulate the plating catholyte over iron metal packing, iron wool, scrap, or immersed soluble iron coupons for the purpose of Fe³⁺ control. (An undivided cell with soluble iron anodes contacting the shared electrolyte would also be challenged under 1c — one more reason the equipment list's divided configuration and anode bags matter.)
2. Keep ferric management **reductant-based** (ascorbic) and/or **membrane/electrochemical**. Document reagent identity in every manifest entry (`campaign_manifest_template.csv` has the hooks).
3. Note that dependent claims blanket our operating window: scrub rate 0.05 mM/hr–5 M/hr (13); scrubbed Fe³⁺ ≤3 mM (20) / ≤5 mM (121); total Fe **0.3–2.0 M** (22); initial pH **1.5–2.2** → scrubbed pH **2.2–3.5** (24–25); **20–85 °C** (109); batch or continuous (111–112); in-line scrubbing or catholyte circulation loop through a scrubber cell (118–120); rate of Fe³⁺ conversion ≥ H₂-generation rate, H₂ ≤ 50 mM/hr (6–9) — any future iron-metal-contact implementation would sit squarely inside these.
4. Claim 122 family covers an **electrochemical ferric-scrubber cell** (membrane-separated, Fe³⁺ reduced at its cathode) feeding the plating cell. Our proposed Fe²⁺/Fe³⁺ *anode* mediation (oxidation direction, in the plating cell anolyte) is a different function, but an "electrochemical scrubber" as a separate catholyte-conditioning unit **would** approach claim 122 scope. Flag to counsel before building any standalone electrochemical Fe³⁺-reduction conditioning unit.
5. The FTO assessment's outstanding item 2 is answered provisionally: **"iron-based metal" reaches any iron-mediated contact step**; ascorbate-only conditioning avoids 1c. Counsel should confirm the read of "contacting the first aqueous acidic solution with an iron-based metal" and check for prosecution-history estoppel and national-phase amendments.

**Prior-art note for future patentability/validity work:** chemical reduction of Fe³⁺ by metallic iron in iron sulfate liquors is classical practice (and appears in Cardarelli's abandoned FerWIN teaching — "[0128] removal of trace ferric cations" from copperas liquors [6]). Directionally useful for §103 analysis of claim 1's combination; needs a real searcher.

---

## 5. Search results for sulfate divided-cell iron-EW and AWARE filings (FTO §4 item 3)

**Found — the most proximate prior art to the entire program:**

- **WO2009/124393 / US20110089045A1 / CN102084034A — Cardarelli "FerWIN" process** [6][7]: "Electrochemical process for the recovery of metallic iron and sulfuric acid values from iron-rich sulfate wastes, mining residues and pickling liquors." **Three-compartment divided cell; copperas (TiO₂ byproduct FeSO₄·7H₂O) and spent pickle liquor feeds; ferric removal during liquor preparation; acid regeneration at the anode; catholyte conditioning and "bath dummying"; reported operation pH 1.4, 50 °C, 250 A/m², 3.75 V, 85 % CE.** Legal status: US application **ABANDONED 2014-11-04** (failure to respond) [5]. Vendor claims a ~1/50-scale pilot line [7].
  - **Implication 1 (FTO):** abandoned/foreclosed applications cannot issue there; a 2009-priority disclosure is prior art available against everyone, including us and Electra. Practicing the abandoned teaching is *directionally* lower-risk (confirm no surviving family members in target jurisdictions — counsel task).
  - **Implication 2 (patentability):** FerWIN is square §102/§103 prior art against any broad claim of ours to "electrowinning iron from copperas/pickle liquor in a divided cell with acid regeneration." Our inventive space is what archaeology already said (`TIER0_ARCHAEOLOGY.md` §1): **≥300 mA/cm² operation, membrane selection/stack design for Fe²⁺ systems, ascorbate ferric management data, demonstrated integrated continuous operation** — the parts nobody has published at those operating points.
- Older art surfaced: US3853724 (1974, copper EW with dissolved iron) — background only; Pyror-era sulfate diaphragm operation (1947–57) per `TIER0_ARCHAEOLOGY.md` [1] — the grandfather art.
- **AWARE process:** no patent filings located by open-web search as of 2026-07-30. Absence here is **not** evidence of absence — 18-month publication lag means filings from 2024–2025 may not yet be public. Action remains with counsel's fee-based search (FTO §4 item 3).

---

## 6. Actions generated (feeds `PROGRAM_SUMMARY.md` gate 1 → gate 6)

| # | Action | Owner |
|---|---|---|
| 6.1 | Counsel: verify charts 1–4 against Global Dossier file histories; add CN/EP/JP/KR national-phase family members to watch; confirm abandoned status of FerWIN family in all target jurisdictions | Patent counsel |
| 6.2 | Counsel: check each active Electra family's **continuation/divisional pipeline** quarterly — granted claim 1 of Family 1 is dissolution-side, but the specification's D-aspects support plating-cell claims | Patent counsel / program |
| 6.3 | Lab: adopt the iron-metal-contact avoidance discipline of §4 and record it in the campaign manifest metadata | Lab |
| 6.4 | Program: before any IP filing, run a real prior-art search on ascorbate Fe³⁺ management for Fe²⁺ EW baths (our currently-clean differentiator) | Counsel/searcher |
| 6.5 | Program: consider defensive publication of carbon-intensity attributions of electrowon iron once measured (counters product-by-process emissions-claim pattern in US20250146155A1 claim 35) | Program |
| 6.6 | Program: update `FTO_PRELIMINARY_ASSESSMENT.md` §1.1 wording — granted US12054837B2 claim 1 is a dissolution-tank circulation system (thermal reduction in dependent claim 7); done in this round via cross-reference | Docs (this PR) |

---

## References

1. US12054837B2, "Ore dissolution and iron conversion system" (claims 1–11; D-series specification aspects). https://patents.google.com/patent/US12054837B2/en
2. US20240158939A1, "Impurity removal in an iron conversion system" (claims 1–40). https://patents.google.com/patent/US20240158939A1/en
3. US20250146155A1, "System and methods for separation of electrolytic iron from iron-containing feedstock" (claims 1, 29–35; aspects). https://patents.google.com/patent/US20250146155A1/en
4. WO2025199035A1, "High efficiency iron electrowinning" (claims 1–126 incl. scrubber-cell system claims 122+). https://patents.google.com/patent/WO2025199035A1/en
5. US20110089045A1 legal events (abandoned 2014-11-04). https://patents.google.com/patent/US20110089045A1/en
6. CN102084034A (FerWIN family member) — process description incl. copperas source, three-compartment cell, pH 1.4 / 50 °C operation. https://eureka.patsnap.com/patent-CN102084034A
7. Electrochem Technologies, "Electrowinning iron — the FerWIN® process" (pilot-scale claims). https://www.electrochem-technologies.com/English/010_Iron_Electrowinning.html

Related internal docs: `FTO_PRELIMINARY_ASSESSMENT.md` (architectural comparison), `electra_patent_family.md` (family inventory), `TIER0_ARCHAEOLOGY.md` (non-patent prior art), `PROGRAM_SUMMARY.md` (gate list).
