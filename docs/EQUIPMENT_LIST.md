# Dark Mill — Equipment List

Everything below gets you from zero to first iron deposit with FE measurement. The list is split by where the equipment lives, per the proving-ground architecture in `RESEARCH_PROGRAM.md`:

- **Deployable article** — equipment that goes inside the production unit (the Hull cell, electrodes, power supply, instrumentation). This is what eventually gets containerized and redeployed.
- **Fixed proving-ground zone** — containment, safety, recovery, and analytical infrastructure that stays at the home site. Boundary-crossing campaigns run only in this zone.

**Before ordering:** Paste each ASIN into Amazon and verify the specific variant. Amazon parent listings often resolve to different child ASINs. Prices on niche lab gear swing — recheck at order time.

---

## Deployable Article — Tier A: Start Plating (~$350-450)

### Power Supply — 30V/10A CC/CV (~$80)

NOT 30V/50A. A 50A supply at 2% load has poor regulation and coarse current resolution at 0.5-3A where you'll live. A 30V/10A with 0.01A steps is more accurate and costs less.

- New: search Amazon for "30V 10A adjustable DC power supply CC/CV" — ~$70-90
- Used: BK Precision, Kepco, or Sorensen on eBay — ~$50-150, excellent regulation

### Hull Cell (~$100-150)

Amazon does NOT sell complete 267ml Hull cell kits. Order from Caswell Plating (caswellplating.com):
- Hull cell body — ~$80-120
- Hull cell test panels, stack of 50+ — ~$30-50 (consumable, brass or steel)

### Anodes — Soluble Iron, NOT Graphite (~$30-45)

Graphite sheds particulates and is an insoluble anode that oxidizes Fe(II)→Fe(III), killing the bath in days. Use soluble iron anodes as primary.

- **Pure iron or low-carbon steel bar/plate** — OnlineMetals.com, ~$15-30. Dissolves to replenish Fe(II).
- **Polypropylene anode bags** — Amazon "anode bag polypropylene", ~$10-15. Keeps sludge out.
- **Graphite rods** (B07YFGPMHZ, ~$10) — throwaway for first tests only. Expect particulates.

### Cathode Substrates (~$20-30)

- **316 SS sheet 1mm** (NOT 0.5mm — thinner warps) — OnlineMetals.com, cut to Hull cell size with tin snips.

### Chemicals (~$65-80)

| Item | Where | ~$ | Notes |
|------|-------|-----|-------|
| FeSO₄·7H₂O 5 lb | B007ODUNJ4, $14.99, 4.6★/430 reviews | $15 | Verify variant is heptahydrate |
| Boric acid 5 lb | Duda Diesel or Amazon | $20-25 | Need 30-45 g/L. 5 lb covers ~50L bath. NOT 100g. |
| Battery acid 35% | Auto parts store | $10/gal | pH down. Cheaper/safer than lab H₂SO₄. |
| Sodium carbonate (washing soda) | Grocery store | $5 | pH up. |
| Ascorbic acid (vitamin C) | Amazon or grocery | $10 | Antioxidant — prevents Fe(II)→Fe(III) oxidation. |
| Citric acid | Amazon or grocery | $10 | Also helps chelate. |
| Distilled water, several gallons | Grocery store | $5-10 | NEVER use tap water. |

### Electrical (~$45-55)

| Item | Where | ~$ | Notes |
|------|-------|-----|-------|
| Coulomb counter / DC Ah meter | Amazon "dc energy meter ah" | $15-25 | CRITICAL. FE = charge passed, not current setpoint. |
| Multimeter | Amazon "digital multimeter" | $15-25 | Measure actual cell voltage independently. |
| Alligator clip leads, 10A+ rated | Amazon | $10 | Connect supply to electrodes. |

### Glassware (~$12)

- **Borosilicate beaker 1000ml** — B0CZMC5L9Q, $12.36, 4.6★/780 reviews.

### Substrate Prep (~$25)

| Item | Where | ~$ | Notes |
|------|-------|-----|-------|
| Acetone or IPA | Hardware store | $8 | Degrease cathode before plating. |
| Alconox or Simple Green | Amazon | $10 | Lab detergent. |
| Abrasive pads (Scotch-Brite) | Grocery/hardware | $5 | Surface activation for cathode. |

### Safety (~$50)

| Item | Where | ~$ | Notes |
|------|-------|-----|-------|
| Nitrile gloves chemical resistant | B0CYZT25WR, $40, 4.6★ | $40 | 20 pairs, 15 mil heavy duty. |
| Safety goggles splash-proof | B07T6H8G5V, $8, 4.5★ | $8 | |
| Sodium bicarbonate (baking soda) | Grocery | $3 | Spill neutralizer. Keep nearby. |
| Ventilation | — | — | REQUIRED. You are generating H₂. NOT in an unventilated closet. |
| Thermometer | B01IHHLB7W | $8-12 | -50 to 300°C. |

### Data Recording (~$145-200)

Continuous video of every run. No "oh let me just adjust the stirring" that goes unlogged.

| Item | Where | ~$ | Notes |
|------|-------|-----|-------|
| USB webcam x4 | Amazon "Logitech C270" or similar | ~$100 | 720p sufficient. Overhead, panel, instruments, wide. |
| Raspberry Pi 4 or used laptop | — | $0-50 | Runs MotionEye or OBS. You probably have one. |
| 128GB microSD card | Amazon | ~$15 | Continuous recording storage. |
| USB extension cables x4 | Amazon | ~$10 | Camera placement flexibility. |
| Gooseneck/clamp mounts x4 | Amazon | ~$20 | Position cameras without tape. |

---

## Deployable Article — Tier B: FE Measurement (~$50-120)

**Primary method: mass gain** on the precision scale. Run longer deposits to make mass change large relative to scale noise.

| Item | Where | ~$ | Notes |
|------|-------|-----|-------|
| Precision scale 0.001g | B07Q5JHKBL | $80-120 | Draft shield. Reality: sub-$150 "0.001g" scales resolve ~0.01g. Used Mettler/Sartorius on eBay ($200-400) is better if FE accuracy matters. |
| Magnetic stirrer + hot plate | B0BYJWFZS5 | $49 | Verify it actually heats. Check if PTFE stir bar included. |
| PTFE stir bars | Amazon | $8 | If not included with stirrer. |

**Cross-check method: H₂ volumetry (optional, later)**
- Inverted burette or eudiometer — NOT a graduated cylinder
- Gas-tight tubing + stopper
- Temperature/pressure/water-vapor corrections

---

## Deployable Article — Tier C: Divided Cell + Calibration (~$100-150)

The membrane is the easy part. You need a cell body to mount it in.

| Item | Where | ~$ | Notes |
|------|-------|-----|-------|
| H-cell body | Amazon "H-cell electrochemistry" | $30-50 | Two compartments with frit/membrane mount. |
| DIY PVC divided cell | Hardware store | $25 | PVC pipe flanges + gasket + clamp. Works fine. |
| Nafion N117 5×5cm | Amazon | $30-40 | Or Fumasep FKE-50 from Laborxing.com (~$66/10×10cm). |
| pH meter | B01J1DNJQW, Apera AI209 | $50-60 | ±0.01 pH, 4.5K+ reviews. |
| pH 4.00 buffer solution | Amazon | $8-10 | REQUIRED for calibration. Meter is useless without it. |
| pH 7.00 buffer solution | Amazon | $8-10 | REQUIRED. |
| KCl storage solution | Amazon | $8-10 | Probe dies if stored dry. |

---

## Deployable Article — Tier D: Upgrades (When Ready)

| Item | Where | ~$ | Notes |
|------|-------|-----|-------|
| Ag/AgCl reference electrode | Amazon | $50 | Three-electrode measurement. Worth it for mechanism studies. |
| Muffle furnace (used) | eBay | $500-1500 | For carburization and tempering. |
| Pack carburizing supplies | — | $50-100 | Charcoal + BaCO₃ + sealed steel box. |
| Vickers hardness tester (used) | eBay | $500-2000 | Mechanical characterization. |
| Optical microscope (used) | eBay | $100-400 | Grain structure, cross-sections. |
| MMO/platinized titanium mesh | Specialty | $50-100 | Clean insoluble anode for later experiments. |

---

## Fixed Proving-Ground Zone (~$260-450)

The deployable article runs **inside** this zone. Boundary-crossing campaigns (per `RESEARCH_PROGRAM.md` §Fixed proving ground) are permitted only here. These items stay at the home site when the article is redeployed.

### Containment & Spill Recovery

|| Item | Where | ~$ | Notes |
||------|-------|-----|-------|
|| Secondary containment tray, 10 gal+ | Amazon / hardware | $20-40 | Polypropylene or HDPE, sized for the beaker/Hull cell with margin |
|| Spill kit (acid neutralizer, absorbent, disposal bags) | Amazon / safety supply | $30-50 | Baking soda is in the deployable list; this adds neutralizing absorbent and PPE for larger spills |
|| Holding/recovery tank, 5 gal, sealed lid | Hardware | $15-25 | For failed electrolyte, contaminated rinse water, and post-abort bath preservation |
|| Spare wet-end modules (extra beaker, extra Hull cell panels, extra stir bar) | existing suppliers | $25-50 | Recovery means swapping the fouled or damaged module, not scrubbing in place |

### Gas Safety

|| Item | Where | ~$ | Notes |
||------|-------|-----|-------|
|| H₂ gas monitor / combustible gas detector | Amazon, search "hydrogen gas detector" | $50-150 | Must alarm before LEL (4% H₂ in air). Wall-mount near the cell, not on the cell |
|| Ventilation fan + duct (if workspace lacks active exhaust) | Hardware | $50-100 | Moves air across the cell toward an exterior vent. The article's own small fan is not sufficient for boundary crossing |

### Independent Shutdown

|| Item | Where | ~$ | Notes |
||------|-------|-----|-------|
|| Emergency stop / contactor on DC supply | Amazon / electrical supply | $30-80 | Hardwired, latching, physically interrupts output. The supply's own panel switch is not an independent shutdown |
|| GFCI / RCD outlet or adapter | Hardware | $15-25 | For the power supply and stir plate |

### Analytical / Recovery Support

|| Item | Where | ~$ | Notes |
||------|-------|-----|-------|
|| Sample vials, labels, pipettes (kit) | Amazon | $20-30 | For post-abort and boundary-crossing samples |
|| Log book / label printer | Amazon | $15-30 | Physical record alongside manifests; required when a run aborts and the instrument state is ambiguous |

---

## Budget Summary

|| Tier | What You Get | Cost |
||------|-------------|------|
|| A | First deposits, Hull cell screening, proper bath chemistry, data recording | $500-650 |
|| B | Quantified FE by mass gain + stirring | $50-120 |
|| C | Divided cell, pH control, three-electrode option | $100-150 |
|| **Deployable subtotal A+B+C** | **Full experimental capability inside the zone** | **$650-920** |
|| Zone | Containment, gas safety, shutdown, recovery | $260-450 |
|| **Total** | **Deployable article + fixed proving-ground zone** | **$910-1,370** |

The zone items are not optional if boundary-crossing campaigns are planned. They are the infrastructure that lets the deployable unit learn its limits without becoming the limit itself.

---

## What Changed from v1

| Old (v1) | New (v2) | Why |
|----------|----------|-----|
| 30V/50A supply ($150-480) | 30V/10A (~$80) | Better regulation at 0.5-3A where you live |
| Graphite anode | Soluble iron + anode bag | Graphite sheds, oxidizes Fe(II)→Fe(III) |
| 100g boric acid ($14) | 5 lb (~$22) | 100g covers 2L. Need 30-45 g/L. |
| 1M H₂SO₄ ($18) | Battery acid 35% ($10/gal) | Better value, same function |
| Graduated cylinder for H₂ | Mass gain primary, burette optional | Graduated cylinder isn't a eudiometer |
| No coulomb counter | DC Ah meter ($20) | FE = charge passed, not current setpoint |
| No pH buffers | pH 4.00/7.00 + KCl ($25) | pH meter useless without calibration |
| No divided cell body | H-cell or DIY PVC ($25-30) | Membrane needs a home |
| No substrate prep | Acetone, Alconox, pads ($25) | Bad prep = peeling deposits = wasted weeks |
| No multimeter | Digital multimeter ($20) | Independent cell voltage measurement |
| 0.5mm SS cathode | 1mm SS cathode | Thinner warps |
| No data recording | 4 USB cameras + Pi (~$145-200) | Continuous video of every run. No unlogged adjustments. |

---

## Safety (Read This)

- **Hydrogen gas.** Electrolysis produces H₂. Ventilate your workspace. Never collect gas in a sealed vessel near the power supply.
- **Acid.** Battery acid is corrosive. Gloves + goggles always. Baking soda nearby for spills.
- **Iron sulfate.** Mild irritant. Not acutely toxic but avoid ingestion.
- **Electrical.** DC at 10A can cause burns. Don't touch electrodes while powered.
- **Bath management.** Keep bath covered when not in use. Add ascorbic acid to slow Fe(II)→Fe(III). Filter if anodes shed.
