# Dark Mill — Shopping List (v2)

Revised with technical feedback. Key changes: power supply downsized, anode chemistry corrected, FE measurement approach fixed, missing items added.

**Before ordering:** Paste each ASIN into Amazon and verify the specific variant selected. Amazon parent listings often resolve to different child ASINs (e.g., 30V/5A instead of 30V/50A). Prices on niche lab gear swing — recheck at order time.

---

## ORDER 1: Start Plating (~$350-450)

### Power Supply (DOWNSIZED)
A 30V/50A supply running at 2% of rated output has poor regulation and coarse current resolution at 0.5-3 A where you'll live. Get a 30V/10A instead.

| Product | ASIN | Price | Notes |
|---------|------|-------|-------|
| **DC Power Supply 30V/10A CC/CV** | Search "30V 10A adjustable" | ~$70-90 | 0.01 A resolution, good regulation at low current |
| Used BK Precision / Kepco / Sorensen | eBay | ~$50-150 | Lab-grade, excellent regulation. Best value if you can find one. |

**Recommendation:** 30V/10A new (~$80) or used lab supply on eBay (~$100). Save $300+ vs the 50A unit.

### Hull Cell
| Product | Source | Price | Notes |
|---------|--------|-------|-------|
| **Caswell 267ml Hull Cell** | caswellplating.com | ~$80-120 | Confirm they still stock it and it includes cathode panels. |
| **Hull Cell Test Panels (stack of 50+)** | caswellplating.com | ~$30-50 | Consumable. Brass or steel. |

### Anodes (CORRECTED)
Graphite sheds particulates and is an insoluble anode that will oxidize Fe(II)→Fe(III), killing your bath. Use soluble iron anodes as primary.

| Item | Source | Price | Notes |
|------|--------|-------|-------|
| **Pure iron or low-carbon steel bar/plate** | OnlineMetals.com or Amazon | ~$15-30 | Soluble anode — replenishes Fe(II) as it dissolves. Standard practice. |
| **Polypropylene anode bags** | Amazon, search "anode bag polypropylene" | ~$10-15 | Keeps sludge out of bath. |
| **Graphite rods** (throwaway for first tests) | B07YFGPMHZ | ~$10 | OK for screening, expect particulates. |
| MMO/platinized titanium mesh (upgrade) | Specialty supplier | ~$50-100 | For insoluble anode experiments later. |

### Cathode Substrates
| Item | Source | Price | Notes |
|------|--------|-------|-------|
| **316 SS sheet 1mm** (not 0.5mm — thinner warps) | OnlineMetals.com | ~$20-30 | Cut to Hull cell size with tin snips. |

### Chemicals
| Product | ASIN/Source | Price | Notes |
|---------|-------------|-------|-------|
| **Ferrous Sulfate Heptahydrate 5 lb** | B007ODUNJ4 | $14.99 | 4.6★, 430 reviews. Verify variant is FeSO₄·7H₂O. |
| **Boric Acid 5 lb** (NOT 100g) | Duda Diesel or Amazon "boric acid 5 lb" | ~$20-25 | You need 30-45 g/L. 5 lb covers ~50L of bath. |
| **Battery acid 35%** (pH down) | Auto parts store | ~$10/gal | Cheaper and safer than lab H₂SO₄. |
| **Sodium carbonate (washing soda)** (pH up) | Grocery store | ~$5 | For pH adjustment upward. |
| **Ascorbic acid (vitamin C)** | Amazon or grocery | ~$10 | Antioxidant — prevents Fe(II)→Fe(III) oxidation in bath. |
| **Citric acid** | B00EY3JCMO or grocery | ~$10 | Also helps chelate and prevent oxidation. |
| **Distilled water, several gallons** | Grocery store | ~$5-10 | NEVER use tap water. |

### Glassware
| Product | ASIN | Price | Notes |
|---------|------|-------|-------|
| **United Scientific 1000ml Beaker** | B0CZMC5L9Q | $12.36 | 4.6★, 780 reviews. |

### Safety & Ventilation
| Product | ASIN | Price | Notes |
|---------|------|-------|-------|
| **Nitrile gloves** | B0CYZT25WR | $40 | 4.6★, 203 reviews. |
| **Safety goggles** | B07T6H8G5V | $8 | 4.5★. |
| **Sodium bicarbonate (baking soda)** | Grocery store | ~$3 | Spill neutralizer. Keep nearby. |
| **Ventilation** | — | — | REQUIRED. You are generating hydrogen. Do NOT run in unventilated closet. |

### Electrical
| Item | Source | Price | Notes |
|------|--------|-------|-------|
| **Alligator clip leads** (10A+ rated) | Amazon | ~$10 | For connecting supply to electrodes. |
| **Multimeter** | Amazon, search "digital multimeter" | ~$15-25 | Measure actual cell voltage independently of supply display. |
| **Coulomb counter / DC Ah meter** | Amazon, search "dc energy meter ah" | ~$15-25 | CRITICAL. FE requires charge passed, not just current setpoint. |

### Substrate Prep
| Item | Source | Price | Notes |
|------|--------|-------|-------|
| **Acetone or IPA** | Hardware store | ~$8 | Degrease cathode before plating. |
| **Alconox or Simple Green** | Amazon | ~$10 | Lab detergent for cleaning. |
| **Abrasive pads (Scotch-Brite)** | Grocery/hardware | ~$5 | Surface prep for cathode. |

---

## ORDER 2: FE Measurement (~$50-100)

### Primary method: Mass gain (not H₂ volumetry)
The sub-$150 "0.001g" scales on Amazon effectively resolve ~0.01g in practice. For FE by mass gain, run longer deposits to make the mass change large relative to scale noise.

| Product | ASIN | Price | Notes |
|---------|------|-------|-------|
| **Precision scale 0.001g** | B07Q5JHKBL | ~$80-120 | Draft shield. Verify it actually repeatably reads 1mg. |
| **Used Mettler/Sartorius analytical balance** | eBay | ~$200-400 | Best option if FE accuracy matters. Real 1mg readability. |
| **Magnetic stirrer + hot plate** | B0BYJWFZS5 | ~$49 | Verify it actually heats (some cheap ones don't). Check if PTFE stir bar included. |
| **PTFE stir bars** (if not included) | Amazon | ~$8 | Magnetic stir bar, various sizes. |

### Cross-check: H₂ volumetry (optional, later)
| Item | Source | Price | Notes |
|------|--------|-------|-------|
| Inverted burette or eudiometer | Lab supply | ~$30-50 | NOT a graduated cylinder. Gas-tight, calibrated. |
| Gas-tight tubing + stopper | Amazon | ~$10 | |
| Temperature/pressure corrections | — | — | Water vapor pressure, ambient T/P. |

---

## ORDER 3: Divided Cell + Calibration (~$100-150)

### Divided Cell (CORRECTED — membrane alone is useless)
| Item | Source | Price | Notes |
|------|--------|-------|-------|
| **H-cell body** | Amazon, search "H-cell electrochemistry" | ~$30-50 | Two compartments with frit/membrane mount. |
| **DIY: PVC pipe flanges + gasket + clamp** | Hardware store | ~$25 | Cheaper alternative. Works fine for screening. |
| **Nafion N117 5×5cm** or **Fumasep FKE-50** | Amazon or Laborxing.com | ~$30-66 | Cation exchange membrane. FKE is cheaper. |

### pH Measurement & Buffers
| Product | ASIN/Source | Price | Notes |
|---------|-------------|-------|-------|
| **Apera AI209 pH Tester** | B01J1DNJQW | ~$50-60 | ±0.01 pH, 4.5K+ reviews. |
| **pH 4.00 buffer solution** | Amazon | ~$8-10 | For calibration. REQUIRED — probe is useless without it. |
| **pH 7.00 buffer solution** | Amazon | ~$8-10 | For calibration. |
| **KCl storage solution** | Amazon | ~$8-10 | For probe storage. Probes die if stored dry. |

### Optional Upgrade
| Item | Source | Price | Notes |
|------|--------|-------|-------|
| **Ag/AgCl reference electrode** | Amazon | ~$50 | Three-electrode measurement. Worth it if you care about deposition mechanism, not just "did metal appear." |

---

## REVISED BUDGET

| Category | Old | New | Change |
|----------|-----|-----|--------|
| Power supply | $150-480 | ~$80 | Downsized to 30V/10A |
| Anodes (iron + bag) | $10 | ~$40 | Soluble iron + anode bag |
| Coulomb counter | — | ~$20 | NEW — critical for FE |
| Multimeter | — | ~$20 | NEW — independent V measurement |
| Boric acid (5 lb) | $14 (100g) | ~$22 | Right quantity |
| pH buffers + KCl | — | ~$25 | NEW — required for pH meter |
| Divided cell body | — | ~$30 | NEW — membrane needs a home |
| Substrate prep | — | ~$25 | NEW — acetone, detergent, pads |
| Hull cell panels (50) | — | ~$40 | Consumable |
| Distilled water | — | ~$10 | |
| Ascobic/citric acid | — | ~$20 | Antioxidant for bath |
| Na₂CO₃ (pH up) | — | ~$5 | |
| Ventilation | — | — | REQUIRED, not purchasable |

**Grand total: ~$550-700** (roughly same as before, but correct items)

---

## ITEMS REMOVED (from v1)

- ~~30V/50A power supply~~ → 30V/10A (better regulation at low current)
- ~~100g boric acid~~ → 5 lb (right quantity)
- ~~1M H₂SO₄ $18~~ → battery acid $10/gal (better value)
- ~~Graduated cylinder for H₂ volumetry~~ → mass gain is primary FE method; inverted burette optional later

## SAFETY REMINDERS

- **Hydrogen gas.** Electrolysis produces H₂. Ventilate. Never collect in sealed vessel near power supply.
- **Acid.** Battery acid is corrosive. Gloves + goggles always. Baking soda nearby for spills.
- **Iron sulfate.** Mild irritant. Not acutely toxic but avoid ingestion.
- **Electrical.** DC at 10A can cause burns. Don't touch electrodes while powered.
- **Ferric contamination.** Keep bath covered when not in use. Add ascorbic acid to slow Fe(II)→Fe(III). Filter if graphite anodes shed.
