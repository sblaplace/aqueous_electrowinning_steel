# Dark Mill — Equipment Shopping List

Everything below gets you from zero to first iron deposit with FE measurement.
Sorted by priority: buy the first section this week, the rest when ready.

## Tier A: Start plating this week (~$500–800)

| Item | What | Why | Where | ~$ |
|------|------|-----|-------|-----|
| DC power supply | 30V/50A, constant-current + constant-voltage | Runs the cell | Amazon/McMaster | 150–300 |
| Hull cell | 267 mL standard Hull cell kit | Maps 2 decades of j in one 10-minute deposit | Caswell, Amazon | 100–200 |
| FeSO₄·7H₂O | Ferrous sulfate heptahydrate, 2–5 kg | Electrolyte source of Fe²⁺ | Amazon, chemical supplier | 20–40 |
| H₂SO₄ (dilute) | 10% sulfuric acid, 1 L | pH adjustment | Amazon, hardware store | 10–20 |
| H₃BO₃ | Boric acid, 500 g | Buffer for surface pH control | Amazon | 10 |
| Stainless steel cathodes | 316 SS sheet, 0.5 mm, cut to Hull cell size | Cathode substrate, peels easily | McMaster, OnlineMetals | 20–40 |
| Graphite anode | Graphite plate or rod | Cheap, works for initial tests | Amazon | 10–20 |
| Alligator clip leads | Heavy-duty, 10A+ rated | Connect power supply to electrodes | Amazon | 10 |
| Beakers | 1 L borosilicate, ×3 | Electrolyte containers | Amazon | 15 |
| pH strips | Wide-range pH 0–14 | Quick pH checks | Amazon | 8 |
| Digital pH meter | Basic pen-style | Accurate pH for experiments | Amazon | 15–30 |
| Thermometer | Digital, 0–100°C | Bath temperature | Amazon | 8 |
| Magnetic stirrer + hot plate | Basic model, 1000 RPM | Agitation + temperature control | Amazon | 40–80 |
| Precision scale | 0.001 g resolution | Weigh deposits for FE calculation | Amazon | 40–80 |
| Safety gear | Nitrile goggles, acid-resistant gloves, lab coat | H₂SO₄ and metal salts | Amazon | 30 |

**Total: ~$480–880**

## Tier B: FE measurement with hydrogen volumetry (~$100–200 additional)

| Item | What | Why | Where | ~$ |
|------|------|-----|-------|-----|
| Inverted graduated cylinder | 100 mL, glass | Collect H₂ gas over cathode | Amazon, lab supply | 15 |
| Rubber stopper + tubing | Fits cylinder, connects to collection | Gas collection setup | Amazon | 10 |
| Mineral oil or electrolyte | For water displacement in collection | Seal the gas column | Amazon | 5 |
| Stopwatch | Any | Time-resolved FE measurement | Phone works | 0 |
| Graduated cylinder (small) | 10 mL, for calibration | Volume calibration | Amazon | 5 |
| Rubber tubing | Silicone, 6mm ID | Connect cylinder to collection vessel | Amazon | 5 |

**Total: ~$40–40**

H₂ volumetry is better than weighing for FE because:
- Continuous, real-time (not endpoint)
- More sensitive at high FE (small H₂ signal)
- No need to dry and weigh deposit mid-experiment

## Tier C: Stress measurement (~$50–100 additional)

| Item | What | Why | Where | ~$ |
|------|------|-----|-------|-----|
| Thin SS shims | 301 SS, 0.1 mm, 100×15 mm strips ×10 | Stoney substrate — deposit on one face, measure curvature | McMaster | 20 |
| Dial gauge | 0.01 mm resolution, magnetic base | Measure shim curvature | Amazon, McMaster | 30–50 |
| Flat reference surface | Granite surface plate (small) or glass plate | Reference for curvature measurement | Amazon | 15–30 |

**Total: ~$65–100**

Deposit on one face of the shim. Internal stress causes curvature. Measure deflection at center vs edges. Stoney equation gives stress in MPa as a function of thickness and curvature. Real-time, in-situ, answers Problem 4 in hours.

## Tier D: Divided cell (~$100–300 additional)

| Item | What | Why | Where | ~$ |
|------|------|-----|-------|-----|
| Cation exchange membrane | Nafion 117 or cheaper alternative (Fumasep FKE) | Separates anolyte from catholyte, blocks Fe³⁺ crossover | Fuel Cell Store, Amazon | 50–150 |
| Two-compartment cell | H-cell or split beaker with membrane clamp | Divided cell geometry | Lab supply, or DIY with acrylic | 30–80 |
| Graphite or DSA anode | For anolyte compartment | Anode reaction control | Amazon, specialty | 20–50 |
| NaOH or Na₂SO₄ | For anolyte | Anolyte electrolyte | Amazon | 10 |

**Total: ~$110–290**

The divided cell is critical: in undivided mode, Fe²⁺→Fe³⁺ at the anode creates a redox shuttle that kills FE. The membrane blocks this.

## Tier E: Later (when you have deposits worth characterizing)

| Item | What | Why | ~$ |
|------|------|-----|-----|
| Muffle furnace | 1100°C, used | Carburization, tempering | 500–1500 |
| Pack carburizing supplies | Charcoal + BaCO₃ + sealed steel box | Carbon addition | 50–100 |
| Vickers hardness tester | Used | Mechanical characterization | 500–2000 |
| Optical microscope | 100–400×, used | Grain structure, cross-sections | 100–400 |
| Vernier calipers | Digital, 0.01 mm | Deposit thickness | 20–30 |

**Total: ~$1,170–4,030**

## Grand total: $700–1,300 to start plating with FE measurement

That gets you:
- Hull cell screening (one afternoon, maps full j range)
- FE measurement by H₂ volumetry (continuous, real-time)
- Stress measurement by Stoney method (answers Problem 4 in hours)
- Divided cell operation (suppresses Fe³⁺ shuttle)
- First 27 deposits with quantified FE, stress, and deposit quality

## What you DON'T need yet

- SEM/EDS (university access when you have deposits worth imaging)
- RDE (buy later when the 1D model needs Tafel slope calibration)
- Potentiostat (nice but the DC power supply + shunt resistor gets you 80% of the way)
- Gas carburizing setup (pack carburizing is safer and cheaper for garage)
- Tensile testing frame (hardness correlates with YS, use Vickers for now)
