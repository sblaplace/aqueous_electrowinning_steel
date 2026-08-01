# Bath Specification — First Iron Electrowinning Experiments

**Status:** Model-derived, pre-experimental  
**Generated:** 2026-07-29 from models v7705312  
**Traceability:** Every number below links to a model and its inputs.

---

## 1. Bath Composition

| Component | Target | Unit | Source |
|---|---|---|---|
| FeSO₄·7H₂O | **278** | g/L | FE(j) optimisation — see §1.1 |
| Fe²⁺ (as molarity) | **1.0** | mol/L | ditto |
| H₃BO₃ (boric acid) | **24.8** | g/L (0.40 M) | Surface-pH model — see §1.2 |
| Ascorbic acid (C₆H₈O₆) | **1.0** | g/L | Startup kinetics — see §1.3 |
| pH target | **2.0** | — | Precipitation criterion — see §1.4 |
| Na₂SO₄ (supporting electrolyte) | 0 (optional) | mol/L | Adds conductivity; not required for first runs |

### 1.1 FeSO₄·7H₂O — from FE(j) optimisation

**Model:** `models/diffusion_layer_1d.py` — `faradaic_efficiency(j, T, C, δ, pH, buffer)`

The diffusion-layer model predicts FE as a function of Fe²⁺ concentration at
j = 200 mA/cm², T = 60 °C, δ = 50 µm, pH 2.0, 0.40 M boric acid:

| [Fe²⁺] (mol/L) | FeSO₄·7H₂O (g/L) | FE (%) |
|---|---|---|
| 0.5 | 139 | 84.4 |
| 0.8 | 222 | 89.1 |
| **1.0** | **278** | **90.5** |
| 1.5 | 417 | 92.2 |
| 2.0 | 556 | 93.0 |

**Choice: 1.0 M (278 g/L).** This delivers FE ≥ 90% at j = 200 mA/cm² while
staying well within the room-temperature solubility limit of FeSO₄·7H₂O
(~480 g/L at 20 °C).  Going to 1.5 M (417 g/L) risks crystallisation if the
bath cools overnight; it is available as a "high-performance" variant once the
first runs validate the model.

Molar mass: FeSO₄·7H₂O = 278.02 g/mol.  1 mol/L × 278.02 g/mol = 278 g/L.

### 1.2 H₃BO₃ — buffering capacity

**Model:** `models/diffusion_layer_1d.py` — `buffer_conc_M` parameter

Boric acid (pKa = 9.24 at 25 °C) participates in the proton-flux and
electroneutrality balance across the diffusion layer.  At pH 2.0 it is
>99.99% undissociated, but it still provides a finite reservoir that damps
surface-pH excursions under high current density.

The default of **0.40 M (24.8 g/L)** is the standard for iron plating baths
(Schlesinger & Paunovic, *Modern Electroplating* 5th ed.).  The model shows
surface pH = 2.00 (unchanged from bulk) at j = 200 mA/cm² with this buffer
loading — Fe(OH)₂ supersaturation is 2 × 10⁻⁸, safely below unity.

| Buffer (M) | Surface pH at j = 200 | Supersaturation |
|---|---|---|
| 0.00 | 2.11 (rises) | 4.1 × 10⁻⁷ |
| **0.40** | **2.00** | **2.0 × 10⁻⁸** |

### 1.3 Ascorbic acid — from startup kinetics

**Model:** `models/bath_startup.py` — `recommend_ascorbic_loading(pH, T_C, fe2_0, target_hr)`

The Fe²⁺ autoxidation rate is proportional to [OH⁻]².  At pH 2.0 the
autoxidation is so slow that even without ascorbic acid, [Fe³⁺]/[Fe²⁺] stays
below 5% for >24 hours:

| pH | T (°C) | Ascorbic acid for 24 h stability (g/L) |
|---|---|---|
| 2.0 | 25 | 0.00 (not needed) |
| 2.0 | 60 | 0.00 (not needed) |
| 2.5 | 60 | 4.14 |
| 3.0 | 60 | 29.3 |

**Recommendation: 1.0 g/L ascorbic acid** (5.68 mM) as cheap insurance
against unplanned oxidation (splash aeration, stirring vortex, warm
shutdown).  At pH 2.0 the model predicts consumption of only 0.003 g/L/day
— the 1 g/L initial charge lasts months.

**Maintenance:** Top up to 1.0 g/L weekly, or after any event that exposes
the bath surface to air (splash, drain, refill).  No continuous feed needed.

### 1.4 pH target and adjustment

**Model:** `models/diffusion_layer_1d.py` — surface pH and Fe(OH)₂ precipitation criterion

The model sweeps pH at j = 200 mA/cm², T = 60 °C, [Fe²⁺] = 1.0 M, δ = 30 µm:

| Bulk pH | Surface pH | FE (%) | Fe(OH)₂ supersaturation | Precipitation? |
|---|---|---|---|---|
| 1.5 | 1.50 | 86.6 | 2.0 × 10⁻⁹ | No |
| **2.0** | **2.00** | **91.6** | **2.0 × 10⁻⁸** | **No** |
| 2.5 | 2.49 | 94.9 | 2.0 × 10⁻⁷ | No |
| 3.0 | 2.99 | 97.0 | 2.0 × 10⁻⁵ | No |

**Window: pH 1.8 – 2.2.**  Below 1.8, FE drops below 90% and H₂ evolution
increases.  Above 2.5, Fe(OH)₂ supersaturation approaches 10⁻⁵ and the risk
of colloidal precipitation rises (especially with trace impurities).

**pH adjustment:**
- **To lower pH:** Add dilute H₂SO₄ (battery acid, 35 wt%) dropwise with stirring.
  For 1 L of bath at pH 2.5→2.0: ~0.3 mL of 35% H₂SO₄.
- **To raise pH:** Add Na₂CO₃ (sodium carbonate) in small portions.
  For 1 L at pH 1.5→2.0: ~0.1 g Na₂CO₃.
- **Always measure pH after 5 min equilibration** — the boric acid system
  is slow to equilibrate.

---

## 2. Operating Conditions

| Parameter | Value | Range | Source |
|---|---|---|---|
| Temperature | **60 °C** | 50 – 70 °C | FE(T) model — §2.1 |
| Current density | **200 mA/cm²** | 100 – 300 mA/cm² | FE(j) model — §2.2 |
| Stirring | moderate | δ ≤ 30 µm | Transport model — §2.3 |
| Anode | soluble Fe, bagged | low-carbon steel | §2.4 |
| Membrane (divided cell) | Nafion N117 | | §2.5 |

### 2.1 Temperature — from FE(T)

**Model:** `models/diffusion_layer_1d.py` — Arrhenius diffusivity correction

| T (°C) | FE at j = 200 mA/cm² (%) |
|---|---|
| 40 | 87.3 |
| 50 | 89.1 |
| **60** | **90.5** |
| 70 | 91.5 |
| 80 | 92.3 |

**Choice: 60 °C.** This is the standard operating temperature for iron sulfate
electrowinning.  Higher temperatures improve FE and conductivity but increase
evaporation and ascorbic acid consumption.  60 °C is achievable with a simple
hot plate or water bath.

### 2.2 Current density — Hull cell range

**Model:** `models/diffusion_layer_1d.py` — galvanostatic solver

| j (mA/cm²) | FE (%) | V_cell (V) |
|---|---|---|
| 50 | 91.0 | ~2.3 |
| 100 | 91.2 | ~2.4 |
| 150 | 90.9 | ~2.5 |
| **200** | **90.5** | **2.56** |
| 250 | 89.9 | ~2.6 |
| 300 | 89.1 | ~2.7 |

FE stays above 89% from 50 – 300 mA/cm².  The Hull cell should be run over
the full 100 – 300 mA/cm² range to map deposit quality vs. current density.

**Target: 200 mA/cm²** for the primary operating point.  This gives
FE = 90.5% at the reference conditions, with V_cell ≈ 2.56 V
(cathode + anode overpotential + electrolyte + membrane IR).

**Precipitation check at 200 mA/cm²:** Fe(OH)₂ supersaturation = 2.0 × 10⁻⁸
— no precipitation risk.

### 2.3 Stirring / diffusion-layer thickness

**Model:** `models/diffusion_layer_1d.py` — `delta_m` parameter

| δ (µm) | FE at j = 200 (%) | Interpretation |
|---|---|---|
| 20 | 92.2 | vigorous stirring |
| **30** | **91.6** | moderate stirring |
| 50 | 90.5 | gentle/no stirring |
| 100 | 86.2 | stagnant |

**Target: δ ≤ 30 µm**, corresponding to moderate stirring (magnetic stirrer
at 200–400 rpm, or gentle air sparge).  Tighter films improve FE by reducing
the diffusion overpotential for Fe²⁺ transport to the cathode.

For the Hull cell (no forced convection by default), δ is set by natural
convection — typically 50–100 µm.  FE will still be 86–90%.  Stirring is
recommended for the bench-scale cell.

### 2.4 Anode configuration

**Soluble iron anode** (low-carbon steel, AISI 1008/1010):
- Dissolves as Fe²⁺ → Fe²⁺ + 2e⁻ (E° = −0.440 V vs. SHE)
- Replenishes Fe²⁺ consumed at the cathode
- Bag in polypropylene or Dacron anode bag to contain any sludge
- Anode-to-cathode area ratio: 1:1 minimum

**Why not insoluble (DSA/Pb)?** Soluble Fe anode avoids Pb contamination and
the Fe²⁺ replenishment keeps the bath composition stable.  Anode
overpotential is low (~0.4 V) and is already included in the V_cell model.

### 2.5 Membrane (divided cell)

**Model:** `models/membrane_transport.py` — Nernst-Planck transport through Nafion N117

| Property | Nafion N117 | Fumasep FKE-50 |
|---|---|---|
| Fe³⁺ crossover flux | 5.73 × 10⁻⁴ mol/(m²·s) | 9.21 × 10⁻⁴ mol/(m²·s) |
| Fe³⁺ crossover current equiv. | 5.5 mA/cm² | ~8.9 mA/cm² |
| Membrane IR drop | 0.045 V | 0.010 V |
| H⁺ transport number | 0.917 | — |

**Nafion N117** is recommended for first experiments.  Its Fe³⁺ crossover is
lower (less parasitic shuttle), and the extra 35 mV of IR drop is negligible
compared to the 2.56 V cell voltage.  The crossover equivalent of 5.5 mA/cm²
is ~2.8% of the applied 200 mA/cm² — a manageable parasitic loss.

**Fe³⁺ purge schedule:** When anolyte Fe³⁺ fraction exceeds 50% of total Fe,
purge 20% of anolyte volume and replace with fresh FeSO₄ solution.  At
j = 200 mA/cm² and 100 cm² electrode, this occurs roughly every 4–8 hours
of operation (monitor with a simple KSCN spot test — red = Fe³⁺ present).

---

## 3. Bath Preparation SOP

### 3.1 Materials

| Chemical | Grade | Amount per litre | CAS |
|---|---|---|---|
| FeSO₄·7H₂O | ACS or reagent | 278 g | 7782-63-0 |
| H₃BO₃ (boric acid) | ACS or reagent | 24.8 g | 10043-35-3 |
| C₆H₈O₆ (ascorbic acid) | USP/FCC | 1.0 g | 50-81-7 |
| H₂SO₄ (battery acid, 35%) | reagent | ~1 mL | 7664-93-9 |
| Na₂CO₃ (sodium carbonate) | reagent | ~0.3 g | 497-19-8 |
| Deionised water | 18 MΩ·cm | to 1 L | 7732-18-5 |

### 3.2 Mixing order

1. **Heat** 800 mL deionised water to 50–60 °C in a glass beaker on a hot
   plate with magnetic stirring (300 rpm).

2. **FeSO₄·7H₂O first.** Add 278 g slowly while stirring.  Dissolution is
   endothermic; keep above 40 °C.  Allow 10–15 min for complete dissolution.
   The solution will be pale green.

3. **H₃BO₃ second.** Add 24.8 g while stirring.  Boric acid dissolves slowly
   in cold water but readily above 50 °C.  Allow 10 min.

4. **pH adjustment.** Measure pH with a calibrated meter.
   - If pH > 2.2: add dilute H₂SO₄ (35%) dropwise until pH = 2.0.
   - If pH < 1.8: add Na₂CO₃ in ~0.05 g increments until pH = 2.0.
   - Wait 5 min between adjustments (boric acid equilibration).

5. **Ascorbic acid last.** Add 1.0 g while stirring.  Dissolves in <1 min.
   Do NOT add before pH adjustment — ascorbic acid is unstable above pH 4.

6. **Top up** to 1.0 L with deionised water.  Re-check pH.  Record.

### 3.3 Storage

- **Cover the bath** with a lid or Parafilm to reduce dust and slow
  evaporative concentration.
- **Store in the dark** (amber bottle or wrapped in foil).  Fe²⁺ solutions
  photosensitise the oxidation to Fe³⁺.
- **Room temperature** is fine for storage (< 40 °C).
- **Shelf life before first use:** 7 days.  The ascorbic acid charge of
  1 g/L provides >24 h of protection at pH 2.0 (model: consumption rate
  0.003 g/L/day).  After 7 days, check Fe³⁺ content with KSCN test.

### 3.4 Pre-run checklist

1. Heat bath to 60 °C on hot plate / water bath.
2. Check pH — adjust to 2.0 ± 0.2 if needed.
3. Check Fe³⁺ — add 0.5 g/L ascorbic acid if KSCN test shows pink/red.
4. Verify anode bag integrity (no tears, no exposed bare iron).
5. Verify membrane is hydrated (soak Nafion N117 in DI water ≥ 2 h before use).

---

## 4. Impurity Limits

**Model:** `models/impurity_codeposition.py` — Butler-Volmer + Koutecky-Levich screening model

The co-deposition model predicts impurity incorporation in the iron deposit as
a function of bath concentration, current density, and temperature.

### 4.1 Deposit purity at recommended bath

Bath: Cu = 15 ppm, Ni = 10 ppm, Zn = 10 ppm, Pb = 5 ppm, Sn = 2 ppm
(achievable with reagent-grade FeSO₄·7H₂O, typical assay < 0.005% metals).

| j (mA/cm²) | Fe (wt%) | Cu (ppm) | Ni (ppm) | Zn (ppm) | Total impurity (wt%) |
|---|---|---|---|---|---|
| 100 | 99.63 | 1507 | 921 | 436 | 0.373 |
| **200** | **99.77** | **889** | **543** | **358** | **0.230** |
| 300 | 99.82 | 682 | 417 | 317 | 0.181 |

Higher current density = lower impurity uptake (the impurity partial current
is roughly constant while the Fe partial current scales with j).

### 4.2 Copper — hot shortness limit

Cu content in the deposit must stay below **0.1 wt% (1000 ppm)** to avoid
hot shortness in downstream steel processing.

Maximum bath Cu for < 0.1 wt% deposit:

| j (mA/cm²) | Max bath Cu (ppm) |
|---|---|
| 100 | 10 |
| 200 | 17 |
| 300 | 22 |

**Specification: Bath Cu < 15 ppm.**  This guarantees < 0.1 wt% in the
deposit at j ≥ 200 mA/cm² with margin.

### 4.3 Recommended impurity ceilings

| Impurity | Bath limit (ppm) | Rationale |
|---|---|---|
| Cu | < 15 | Hot shortness (0.1 wt% deposit) |
| Ni | < 50 | Moderate co-deposition; < 1000 ppm deposit at j ≥ 200 |
| Zn | < 50 | Low co-deposition (less noble than Fe); < 500 ppm deposit |
| Pb | < 10 | Toxic; minimised by reagent-grade feedstock |
| Sn | < 5 | Low concern; minimised by reagent-grade feedstock |

**Practical note:** Reagent-grade FeSO₄·7H₂O (ACS, ≥99.0%) typically
contains Cu < 5 ppm, Ni < 5 ppm, Zn < 10 ppm.  No additional purification
is needed for first experiments.

---

## 5. Safety

### 5.1 H₂ evolution and ventilation

**Model:** Faradaic calculation from FE(j) model

At j = 200 mA/cm² and FE = 90.5%, 9.5% of current goes to hydrogen evolution:

| Parameter | Value |
|---|---|
| H₂ generation rate | 0.79 L/h per 100 cm² electrode |
| H₂ LEL (lower explosive limit) | 4% v/v in air |
| Minimum air turnover for 100 cm² | 20 L/h |

**Requirements:**
- **Work under a fume hood** or in a well-ventilated area (>6 air changes/hour).
- At the Hull-cell scale (100 cm², 0.8 L H₂/h), a standard fume hood
  provides more than adequate ventilation.
- No ignition sources near the cell (no open flames, no sparking motors).
- H₂ is colourless and odourless — rely on engineering controls, not senses.

### 5.2 Chemical hazards

| Chemical | Hazard | PPE |
|---|---|---|
| FeSO₄·7H₂O | Irritant, environmental hazard | Gloves, goggles |
| H₃BO₃ | Reproductive toxicant (Cat 1B) | Gloves, goggles; avoid skin contact |
| Ascorbic acid | Low hazard | Gloves |
| H₂SO₄ (35%) | Corrosive | Acid-resistant gloves, goggles, face shield |
| Electrolyte (hot, pH 2) | Irritant, thermal burn risk | Gloves, goggles, lab coat |

### 5.3 Spill procedure

1. **Small spill (< 100 mL):** Absorb with paper towels.  Dispose as
   heavy-metal waste (iron, trace impurities).
2. **Large spill:** Contain with absorbent pads.  Do not wash into drains —
   iron sulfate is an environmental pollutant (aquatic toxicity).
3. **H₂SO₄ spill:** Neutralise with Na₂CO₃ or absorb with dry vermiculite.
   Do not mix with bleach or organics.
4. **Hot electrolyte burn:** Flush with copious cold water for 15 min.

### 5.4 Electrical safety

- Cell voltage: ~2.6 V DC at 200 mA/cm² — low voltage, low risk of shock.
- Power supply: use a current-regulated DC supply with GFCI protection.
- Never connect or disconnect electrodes while the supply is on.
- Keep all connections dry — hot acidic electrolyte is conductive.

---

## 6. Acceptance Criteria Traceability

| Criterion | Required | Achieved | Source |
|---|---|---|---|
| FE ≥ 80% at j ≥ 200 mA/cm² | ✓ | FE = 90.5% | `faradaic_efficiency(200, T=60, C=1.0, d=50e-6, pH=2.0, buf=0.40)` |
| Ascorbic acid for 24 h stability | ✓ | 1.0 g/L (consumes 0.003 g/L/day) | `recommend_ascorbic_loading(pH=2.0, T_C=60)` → 0.00 g/L; 1.0 g/L as margin |
| pH window avoids Fe(OH)₂ | ✓ | pH 2.0, supersaturation = 2 × 10⁻⁸ | `DiffusionLayer1D.solve(200).feoh2_supersaturation` |
| Stirring δ ≤ 30 µm | ✓ | FE = 91.6% at δ = 30 µm | `faradaic_efficiency(200, delta_m=30e-6)` |
| All numbers traced | ✓ | Model refs in each section | |

---

## Appendix: Model Version and Inputs

- **Diffusion-layer model:** `models/diffusion_layer_1d.py` (731 lines), commit c6f28b3
- **Bath startup model:** `models/bath_startup.py` (424 lines)
- **Impurity co-deposition model:** `models/impurity_codeposition.py` (655 lines)
- **Membrane transport model:** `models/membrane_transport.py` (741 lines)

All models run under Python 3.14 + NumPy + SciPy.  Test suites:
`tests/test_diffusion_layer_1d.py` (26 tests), `tests/test_bath_startup.py`,
`tests/test_impurity_codeposition.py`, `tests/test_membrane_transport.py`.
