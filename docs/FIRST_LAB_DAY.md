# First Lab Day Packet — Bath B0, Hull Screen, Gravimetric FE

**Date:** 2026-07-30
**Scope:** Executes program **gate 2** (`PROGRAM_SUMMARY.md`): screen the sulfate feed surrogate in a Hull cell and take the first gravimetric FE measurement — the "weighed deposit with a closed balance," not just a photograph. Day 1 is and **and** Tier A+B equipment days. It is *not* the divided-cell matrix (gate 3) and *not* Phase III (co-deposition).

All quantities below were computed **with this repository's own models** so the SOP and the toolchain stay consistent; rerun the snippets if anything changes. Equipment and suppliers are in `EQUIPMENT_LIST.md` / `SHOPPING_LIST.md`; do not improvise substitutions for the safety items.

---

## 0. Pre-flight checklist (must be all-YES before mixing)

- [ ] Ventilation: workspace moves air; H₂ is generated in every run. No sealed vessels near the powered cell (`EQUIPMENT_LIST.md` Safety).
- [ ] Nitrile gloves + splash goggles on; baking soda within reach; battery acid decanted into a labeled working bottle.
- [ ] pH meter **calibrated with 4.00 and 7.00 buffers** the same morning; probe stored in KCl since last use.
- [ ] Scale leveled on a stable bench, zero/tare drift checked over 10 min. Budget 0.001 g-nominal scales resolve ~0.01 g — assume 0.01 g for QA math.
- [ ] Coulomb counter (Ah meter) wired in series and zeroed; independent multimeter available across the cell.
- [ ] Cameras rolling before power (overhead, panel, instruments, wide — per equipment list).
- [ ] `experiments/data/day1_run_sheet.csv` printed or open; `campaign_manifest_template.csv` copied and renamed with today's date; **per run:** `run_manifest_template.json` copied (validated by `models/run_manifest.py` — requires `equipment`/`setup`/`video` blocks; a `recording_status: none` needs a written justification) and **`bath_batch_template.json` copied for batch B0** (validated by `models.run_manifest.load_bath_batch`; both templates above are pre-filled for this SOP and pass the validators as shipped).

---

## 1. Bath B0 — master sulfate bath (1.5 L, 1 M Fe²⁺, pH 2 baseline)

Computed for the shopping-list reagents. FeSO₄·7H₂O *M* = 278.01 g/mol; 1 M Fe²⁺ ⇒ **278.0 g/L**. Matches the DOE matrix's assumed bath (`models/experimental_matrix.py`: `c_FeSO4_M = 1.0`).

| Component | 1.5 L master batch | 267 mL Hull fill (from master) | Notes |
|---|---:|---:|---|
| FeSO₄·7H₂O | **417 g** | 74.2 g equivalent | 5 lb bag = 2268 g ⇒ ~8.2 L of 1 M bath ≈ 5 batches |
| Boric acid (buffer/complex — see note) | **52.5 g** (35 g/L) | 9.3 g equivalent | 5 lb covers ~50 L. Do **not** credit it with surface-pH control (`TIER0_ARCHAEOLOGY.md` §5, Gangasingh & Talbot 1991) |
| Ascorbic acid | 0 g at pH 2 | 0 g | See §2 — model says optional at pH 2 day-of |
| Distilled water | to 1.5 L | draw 267 mL | Never tap water |
| Battery acid (35%) | ~mL quantities, to pH 2.0 ± 0.05 | — | Add slowly, stirring, after salts dissolve |
| Na₂CO₃ slurry | for pH 3 aliquots only | — | See §2 and §5 |

**Make-up procedure**
1. ~1 L DI water in the 2 L beaker (or jug), stir bar in, stir plate on. Warm to ~40 °C helps boric acid dissolve.
2. Boric acid first — it is the slow one. Fully clear before continuing.
3. FeSO₄·7H₂O in portions; solution goes pale sea-green. Fully dissolved = completely clear, no crystals on the bottom.
4. Cool toward working temperature, then titrate pH down with battery acid to **pH 2.00 ± 0.05 at working temperature** (pH is temperature-dependent; measure where you will run).
5. Top to 1.5 L, verify pH again, log actual masses/volumes in the manifest notes — *actuals*, not recipe.

---

## 2. Fe³⁺ discipline (model-driven; also a patent-design-around boundary)

Bath-aging table from `models/bath_startup.py` (`simulate_bath`, literature rate constants, uncalibrated — order-of-magnitude truth; 5% Fe³⁺/Fe²⁺ degradation threshold):

| Bath | Ascorbic acid | Air-exposed (SA/V ≈ 2 cm⁻¹) | Covered |
|---|---|---|---|
| pH 2.0, 25 °C | 0–2 g/L | **>72 h to threshold** | longer |
| pH 3.0, 25 °C | 0 g/L | **≈5.9 h** | longer |
| pH 3.0, 25 °C | 1 g/L | ≈7.2 h | longer |
| pH 3.0 (held 24 h, open beaker) | model needs ≈37 g/L — impractical | | |

Rules that follow:
1. **pH 2 bath needs no stabilizer for a campaign week; make it once, keep it covered.**
2. **pH 3 aliquots are made from the pH 2 master day-of, dosed with 1 g/L ascorbic acid, and used within ~4 h of air exposure.** Sequence pH 3 runs back-to-back.
3. Log cumulative amp-hours against each bath batch; 1 M = 55.8 g/L Fe. Archaeology anchor: CE degrades badly below ~20–40 g/L Fe (`TIER0_ARCHAEOLOGY.md` §1, [4]) — a depleted bath will fake a FE problem with real data.
4. If a Fe³⁺ check is wanted mid-day, the MEMS literature's 5-sulfosalicylic-acid photometric assay is the field method (`TIER0_ARCHAEOLOGY.md` §6, [19]).
5. **Patent design-around boundary (`CLAIM_CHARTS_PRELIMINARY.md` §4):** ferric control stays chemical (ascorbate) or later electrochemical/membrane — **do not** park iron metal (coupons, wool, packing) in the plating bath for Fe³⁺ control. This note goes in the metadata of every run (`metadata_file` field) as contemporaneous documentation.

---

## 3. Runs R1–R2: Hull cell screening (Caswell 267 mL)

**Anode decision for Day 1:** use the **throwaway graphite rod** from the equipment list for these first undivided Hull runs (accept particulates; screen is qualitative). Rationale beyond bath life: an **undivided cell with a soluble iron anode is iron metal in contact with the plating solution**, which sits closer to WO2025199035A1 claim-1 element (c) than we want a default workflow to be (`CLAIM_CHARTS_PRELIMINARY.md` §4). Soluble iron anodes debut later, in the divided cell's anode compartment (gate 3). Escalate to counsel if anyone wants to change this default.

**Setup:** 267 mL from the master bath; panel (316 SS, 1 mm) scuffed with Scotch-Brite, acetone-degreased, DI-rinsed, dried, masked if you want a defined track; panel against the angled wall; graphite anode on the opposite wall; CC mode.

| Run | Program | Purpose |
|---|---|---|
| R1 | **2.0 A, 5 min** (600 C; theor. 0.174 g @100% FE) | Baseline appearance map across ~1 decade of j |
| R2 | **2.0 A, 10 min** (1200 C; theor. 0.347 g @100% FE) | Longer-run map; early morphology/stress signs (cracking, peeling at high-j edge) |

Repo Hull model (`models/hull_cell.py`, default 10×5 cm panel, 1.5→9.0 cm gaps, 48.6°) at 2 A — **measure your real cell and re-run `python -m models.run_hull_cell` with actuals**; the map is a primary-current screening aid (no kinetics/transport/edge effects):

| Strip center (cm from near edge) | 0.5 | 1.5 | 2.5 | 3.5 | 4.5 | 5.5 | 6.5 | 7.5 | 8.5 | 9.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| j (mA/cm²) | 90.5 | 64.2 | 49.8 | 40.7 | 34.4 | 29.8 | 26.3 | 23.5 | 21.3 | 19.4 |

**Record:** bath T before/after; cell V from the multimeter at 1-min marks; video; panel photo under consistent lighting, near edge oriented; annotate appearance per strip (burned/black powder / gray coherent / thin-none). Black powder at the near (high-j) edge is expected HER/transport behavior, not a failed run.

**Exit criteria to proceed to R3:** a coherent gray deposit exists over at least the mid-j strips on R1 or R2, and bath is still clear pale-green at end of R2. If everything is powder/no deposit at every strip: stop; recheck pH, temperature, Fe³⁺ discipline, and connections before burning more panels.

---

## 4. Runs R3–R4: Gravimetric FE coupons (the quantified artifact)

Conditions chosen to land on the model DOE's **predicted-PASS** mid-corner (`experiments/data/factorial_doe_matrix.csv`, RUN-005: 100 mA/cm², pH 2.0, 50 °C — model FE ≈ 99.6%, V_cell ≈ 3.45 V; *predictions are uncalibrated — we are here to replace them*).

| Run | Program | Theoretical deposit |
|---|---|---|
| R3 | **1.00 A CC, 2 h** (7200 C), 10 cm² masked 316 coupon ⇒ 100 mA/cm², 50 °C, mid-stir | 2.084 g @100% FE (1.667 g @80%) ≈ 260 µm coherent if smooth |
| R4 | Replicate of R3 | QA: agreement within combined scale/charge uncertainty |

Execution and QA follow `experiments/README.md` §Minimum gravimetric QA exactly: masked documented area; clean/rinse/dry/weigh before; **negative cathodic current** convention in the trace CSV; rinse/dry-to-constant-mass after; blank-corrected FE_app = Δm / (Q·M/2F). With ~0.01 g scale resolution and 2.08 g theoretical mass, single-run FE resolution is ~±0.5% — good enough to be meaningful on day one. **An FE_app >100% is not clipped; it is a QA flag** (retained salts/oxides/moisture) — inspect, re-dry, re-weigh, and note.

Files per run (schemas in `experiments/data/README.md`):
- trace → `hull_cell_galvanostatic_template.csv` format (timestamp_s, current_A)
- gravimetry → `hull_cell_gravimetry_template.csv` format
- analysis → `python experiments/notebooks/phase2_hull_cell.py --trace <run>.csv --gravimetry <run>.csv`

**R3/R4 outcome routing:**
- FE_app ≳ 70% and coherent coupon: gate-2 box checked; proceed to gate 3 divided-cell matrix planning; keep the coupon — it is the program's first *physical* milestone.
- FE_app 40–70%: repeat R3 at pH 3 (fresh aliquot per §2) and at 35 °C to map sensitivity before concluding anything.
- FE_app <40% *or* deposit won't stay on the coupon: stop and debug bath/prep (Fe³⁺, prep, connections), not the theory — archaeology says 85% CE in sulfate is a 70-year-old result (`TIER0_ARCHAEOLOGY.md` §1).

---

## 5. Optional R5: pH 3 point (if the day is young)

One gravimetric coupon (R3 recipe, but pH 3.0 fresh aliquot +1 g/L ascorbate, ≤4 h air exposure) brackets the DOE's pH axis on day one and exercises the §2 discipline. pH-adjust with Na₂CO₃ slurry slowly — local Fe(OH)₂ precipitation from overshoot clouds the bath and the data.

## 6. Day structure (~7 h)

| Clock | Block |
|---|---|
| 0:00–1:30 | Pre-flight + bath B0 make-up + pH titration |
| 1:30–2:15 | R1 + R2 Hull panels (5 min + 10 min + strip-down/photography each) |
| 2:15–4:30 | R3 gravimetric coupon (2 h) + dry/weigh |
| 4:30–6:45 | R4 replicate (2 h) + dry/weigh |
| parallel | R5 pH 3 coupon in the second beaker during R3/R4 if confident |
| 6:45–7:00 | Manifest closeout: run sheet filled, raw exports copied untouched, metadata JSONs written, bath batch log updated, coupon photos archived |

## 7. What Day 1 produces (and what it kills)

1. A Hull appearance map of the actual sulfate surrogate across j — gate 2 evidence.
2. Two (three) weighed FE numbers at a DOE corner, with full traceability. Validate before any calibration touches them: campaign CSV via `models/campaign.py`; per-run manifests and the bath batch via a one-liner —
   ```bash
   python -c "from models.run_manifest import load_experiment_manifest, load_bath_batch; \
   load_experiment_manifest('metadata/P2-YYYYMMDD-R3.json'); load_bath_batch('B0-YYYYMMDD.json'); print('traceability OK')"
   ```
3. First measured V(j) points to feed `models/calibration.py` (QA-gated LSV trace) and to confront the TEA's assumed 3.4–6.6 V band.
4. A checked-out workflow: equipment, cameras, manifests, analysis scripts — everything gate 3's divided-cell matrix reuses.
5. Explicit go/no-go for gate 3 spend (H-cell, membranes, AE) against the routing rules in §3–4 — and early data against the program kill criterion (`PROGRAM_SUMMARY.md`: ≥300 mA/cm², FE ≥70%, ≤4.0 kWh/kg; the archaeology context: Pyror 4.25 kWh/kg at ~25 mA/cm² is the number to beat, `TIER0_ARCHAEOLOGY.md` §8).

---

*Model and data provenance: recipe constants from `models/experimental_matrix.py` (1 M Fe²⁺) and first-principles MW/Faraday arithmetic; Hull strip table from `models/hull_cell.py` defaults (re-run with measured geometry); bath-aging rules from `models/bath_startup.py` with literature rate constants (Sung & Morgan 1980; Stumm & Lee 1961; Khan & Martell 1967) — uncalibrated, order-of-magnitude; DOE predictions from the uncalibrated screening models, to be replaced by Day-1+ measurements.*
