# Product Value Ladder — Feedstock vs. Steel, Computed Live

> **Generated artifact — do not hand-edit.** Rebuild with `python -m models.run_product_ladder` (or `aq-steel-product-ladder`) after any change to the models it derives from (cell_architecture, electrochemistry, technoeconomic, thermomechanical, anchors, this module). Numbers re-derive on every run; the comparative structure is what is decision-grade, not the decimals.

<!-- provenance:
```json
{
  "artifact": "product_ladder",
  "recipe": "python -m models.run_product_ladder  (full-grade; screening constants are L1 by declaration, not by mode)",
  "mode": "full-grade",
  "source_hashes": {
    "models/product_ladder.py": "f06a23dab69534ce",
    "models/cell_architecture.py": "77a08ed3afac2da7",
    "models/electrochemistry.py": "935e526a6598d30c",
    "models/technoeconomic.py": "db224bcf4b144584",
    "models/anchors.py": "cc6dde9d09b5ccac",
    "models/thermomechanical.py": "03d99ca6f2b215c2"
  }
}
```
-->

**Screening flag:** unvalidated (L1). Product prices are anchored screening bands (see Appendix B); everything physical re-derives from the model suite at the moment of generation.

---

## 1. Why this document exists

`docs/RESEARCH_PROGRAM.md` poses the page-1 decision — Option A (melt-shop feedstock) vs. Option B (direct steel) — as text. This ladder makes the same decision a **recomputed number**, and adds the rung the split missed: **Option A.5 (own-melt + cast to bar)**. Any model change that moves productivity, capital charge, DC energy, anneal energy, or the default electricity price moves every verdict here automatically — that is the entire design.

Contribution-margin basis (uniform across rungs; BOP/labour/feedstock live in `technoeconomic.py`, which is the full-plant model):

```
margin $/t  =  price_mid − ( (DC kWh/t + aux kWh/t)×$/kWh + aux cash $/t + installed-cell capital charge $/t )
margin $/(m²·yr)  =  margin $/t × areal productivity t/(m²·yr)
required ×Zn     =  installed $/m² × CRF / (budget_frac × price) / zinc-benchmark productivity
```

Context at generation: V_cell = 2.5 V, FE = 0.85, electricity = $0.04/kWh, zinc benchmark = 3.68 t/(m²·yr) at this FE, capital budget fraction = 0.1 of product price.

## 2. The ladder

| Rung | Option | Architecture | Price band $/t | Product |
|---|---|---|---|---|
| `flake_feed` | A | rotating_cylinder | 300–600 (mid 450) | flake/powder → passivated briquette |
| `own_melt_bar` | A.5 | rotating_cylinder | 500–1,000 (mid 750) | flake → induction melt → billet → bar |
| `annealed_foil` | B-lite | drum_and_strip | 1,200–2,800 (mid 2,000) | drum-harvested foil, annealed, temper-rolled |
| `structural_sheet` | B | drum_and_strip | 600–1,100 (mid 850) | foil + in-cell or carburized C + anneal |
| `pm_powder` | side | rotating_cylinder | 1,500–3,500 (mid 2,500) | powder → passivate/size/classify |
| `battery_iron` | side | rotating_cylinder | 1,500–4,500 (mid 3,000) | powder/foam → porosity-spec finish |
| `magnetic_foil` | side | drum_and_strip | 2,000–6,000 (mid 4,000) | drum foil → anneal → insulation coat → laminate |

## 3. Screening economics at mid price

| Rung | Productivity t/(m²·yr) | DC kWh/t | Aux kWh/t | Capital $/t | Capital share | Cost $/t | Margin $/t | Margin $/(m²·yr) | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| flake_feed | 39.1 | 2,823 | 105 | 5.5 | 1.2% | 143 | 307 | 12,028 | clears |
| own_melt_bar | 39.1 | 2,823 | 805 | 5.5 | 0.7% | 276 | 474 | 18,562 | clears |
| annealed_foil | 12.1 | 2,823 | 170 | 28.9 | 1.4% | 214 | 1,786 | 21,564 | clears |
| structural_sheet | 12.1 | 2,823 | 290 | 28.9 | 3.4% | 253 | 597 | 7,202 | clears |
| pm_powder | 39.1 | 2,823 | 120 | 5.5 | 0.2% | 191 | 2,309 | 90,337 | clears |
| battery_iron | 39.1 | 2,823 | 130 | 5.5 | 0.2% | 212 | 2,788 | 109,102 | clears |
| magnetic_foil | 12.1 | 2,823 | 170 | 28.9 | 0.7% | 274 | 3,726 | 44,983 | clears |

`min_price_for_budget` — the product price at which the cell's capital charge is exactly 10% of price — per rung:

| Rung | Capital $/t | Min price for budget $/t | Price band clears? |
|---|---:|---:|:---:|
| flake_feed | 5.5 | 55 | ✓ |
| own_melt_bar | 5.5 | 55 | ✓ |
| annealed_foil | 28.9 | 289 | ✓ |
| structural_sheet | 28.9 | 289 | ✓ |
| pm_powder | 5.5 | 55 | ✓ |
| battery_iron | 5.5 | 55 | ✓ |
| magnetic_foil | 28.9 | 289 | ✓ |

Price-band robustness (margin $/t at band edges). **These contribution margins are upper bounds** — at screening electricity prices and with non-cell costs held uniform, most things clear; the discriminators between rungs are the *size* of the margin per m²·yr (which buys down plant-wide risks) and the gate load (§5), not the sign:

| Rung | Margin @ low price | Margin @ mid | Margin @ high |
|---|---:|---:|---:|
| flake_feed | 157 | 307 | 457 |
| own_melt_bar | 224 | 474 | 724 |
| annealed_foil | 986 | 1,786 | 2,586 |
| structural_sheet | 347 | 597 | 847 |
| pm_powder | 1,309 | 2,309 | 3,309 |
| battery_iron | 1,288 | 2,788 | 4,288 |
| magnetic_foil | 1,726 | 3,726 | 5,726 |

## 4. The '5× imperative' is flake-economics, not physics

Two parameter-free facts the suite now recomputes:

1. **The README's '~5×' is a cross-architecture capital-share ratio at commodity price.** At $450/t, the drum's installed-cell capital charge is $28.85/t (6.4% of price) vs. the rotating cylinder's $5.48/t (1.2%) — a 5.3× ratio. Both technically clear a 10% capital-share budget even at flake price; the README's imperative corresponds to demanding ~4–6% capital share, which is what commodity-iron contribution margins can actually carry after non-cell costs. So the drum's real blocker at commodity price is not its $/m² — it is that thin margins leave no room for it.
2. **The same drum's capital share collapses going up the ladder.**

| Rung (drum architecture) | Price $/t | Capital $/t | Capital share | Min price to fit budget |
|---|---:|---:|---:|---:|
| annealed_foil | 2,000 | 28.85 | 1.4% | 289 |
| structural_sheet | 850 | 28.85 | 3.4% | 289 |
| magnetic_foil | 4,000 | 28.85 | 0.7% | 289 |

Productivity and product price are the *same lever* in kill criterion #3: required zinc-benchmark multiples across architectures and prices (budget = 10% of price):

| Architecture | Price $/t | Required t/(m²·yr) | Required ×Zn | Delivers ×Zn | Clears |
|---|---:|---:|---:|---:|:---:|
| rotating_cylinder | 450 | 4.8 | 1.29 | 10.62 | ✓ |
| rotating_cylinder | 750 | 2.9 | 0.78 | 10.62 | ✓ |
| rotating_cylinder | 850 | 2.5 | 0.68 | 10.62 | ✓ |
| rotating_cylinder | 2,000 | 1.1 | 0.29 | 10.62 | ✓ |
| rotating_cylinder | 2,500 | 0.9 | 0.23 | 10.62 | ✓ |
| rotating_cylinder | 3,000 | 0.7 | 0.19 | 10.62 | ✓ |
| rotating_cylinder | 4,000 | 0.5 | 0.15 | 10.62 | ✓ |
| drum_and_strip | 450 | 7.7 | 2.1 | 3.28 | ✓ |
| drum_and_strip | 750 | 4.6 | 1.26 | 3.28 | ✓ |
| drum_and_strip | 850 | 4.1 | 1.11 | 3.28 | ✓ |
| drum_and_strip | 2,000 | 1.7 | 0.47 | 3.28 | ✓ |
| drum_and_strip | 2,500 | 1.4 | 0.38 | 3.28 | ✓ |
| drum_and_strip | 3,000 | 1.2 | 0.32 | 3.28 | ✓ |
| drum_and_strip | 4,000 | 0.9 | 0.24 | 3.28 | ✓ |
| plate_and_frame | 450 | 1.8 | 0.48 | 0.7 | ✓ |
| plate_and_frame | 750 | 1.1 | 0.29 | 0.7 | ✓ |
| plate_and_frame | 850 | 0.9 | 0.26 | 0.7 | ✓ |
| plate_and_frame | 2,000 | 0.4 | 0.11 | 0.7 | ✓ |
| plate_and_frame | 2,500 | 0.3 | 0.09 | 0.7 | ✓ |
| plate_and_frame | 3,000 | 0.3 | 0.07 | 0.7 | ✓ |
| plate_and_frame | 4,000 | 0.2 | 0.05 | 0.7 | ✓ |

**Read:** with a conventional capital budget, hardware affordability is *not* the discriminator between rungs — the drum is affordable everywhere. What changes across the ladder is (a) margin per m²·yr (§3), which buys down the electricity-price and productivity risks of the whole plant, and (b) the **gate load** (§5): the higher rungs hold the physics the program has spent its modelling effort on (peel, stress, grade) or has yet to build (drum life, melt balance). The drum's real cost at the high rungs is *science risk*, not capital.

## 5. Gate status matrix (resolve live from the model tree)

Legend: ✓ modelled, validated beyond L1 · ◐ modelled, unvalidated L1 · ◑ modelled, screening flag unstated · ✗ **unmodelled** (proposed in `docs/CHEM_PHYS_IMPROVEMENTS_V6.md` or here) · — n/a. The ✗ cells are the *science agenda implied by each product rung*; when a named module lands in `models/`, its ✗ flips to ◐ on the next regeneration. Note: ◑ cells mark mature screening modules that pre-date the SCREENING_FLAG convention — 'modelled', not 'validated'.

| Gate | flake_feed | own_melt_bar | annealed_foil | structural_sheet | pm_powder | battery_iron | magnetic_foil |
|---|---|---|---|---|---|---|---|
| FE ≥ 70 % at j ≥ 300 mA/cm² (kill #1) | ◑ | ◑ | ◑ | ◑ | ◑ | ◑ | ◑ |
| net DC ≤ 4,000 kWh/t Fe (kill #2) | ◑ | ◑ | ◑ | ◑ | ◑ | ◑ | ◑ |
| continuous-harvest cell ≥ target productivity (kill #3) | ◑ | ◑ | ◑ | ◑ | ◑ | ◑ | ◑ |
| closed electrolyte loop & purge policy | ◑ | ◑ | ◑ | ◑ | ◑ | ◑ | ◑ |
| coherent foil peels from the drum | — | — | ◑ | ◑ | — | — | ◑ |
| deposit stress & hydrogen control in-window | — | — | ◑ | ◑ | — | — | ◑ |
| deposit → anneal → steel grade routing | — | ◐ | ◐ | ◐ | — | — | — |
| N pickup / Lüders-band forming gate | — | — | **✗** | **✗** | — | — | — |
| RT aging between harvest & metrology | — | — | **✗** | **✗** | — | — | **✗** |
| drum campaign life (oxide + hydriding) | — | — | **✗** | **✗** | — | — | **✗** |
| idle corrosion / ferric etch of deposit | **✗** | **✗** | — | — | **✗** | **✗** | — |
| post-harvest oxidation & passivation spec | **✗** | **✗** | — | — | **✗** | **✗** | — |
| rinse carryover → charge sulfur | **✗** | **✗** | — | — | — | — | — |
| densification / briquetting product-form gate | **✗** | **✗** | — | — | — | — | — |
| melt-shop remelt verdict (yield/boil/slag) | ◐ | — | — | — | — | — | — |
| in-house melt + solidification/casting block | — | **✗** | — | — | — | — | — |
| bath impurity / purification chain | ◑ | ◑ | ◑ | ◑ | ◑ | ◑ | ◑ |
| porosity / surface spec for anode feed | — | — | — | — | — | ◐ | — |
| magnetic property model (coercivity/loss) | — | — | — | — | — | — | **✗** |
| PM powder sizing & fines spec | — | — | — | — | **✗** | — | — |
| tramp-element hot-shortness ceiling | — | ◐ | — | ◐ | — | — | — |

Gate notes: **g_fe_gate** (diffusion_layer_1d): FE engine; calibration pending (Q3/RDE campaign).; **g_energy_gate** (cell_physics): Voltage decomposition; needs divided-cell measurement.; **g_arch_gate** (cell_architecture): Screen only; hardware evidence absent.; **g_loop** (closed_loop): CSTR screen; long-run drift unmeasured.; **g_peel** (adhesion_peel): Coupon test specified, not yet run.; **g_stress_h** (internal_stress): σ(h) screen + bent-strip protocol pending.; **g_grade** (as_deposited_grade): Coupling seam built (V5/A2); L1.; **g_strain_aging** (strain_aging): V6 §7.1 proposal — UNMODELLED.; **g_deposit_aging** (deposit_aging): V6 §5.2 proposal — UNMODELLED.; **g_drum_life** (ti_hydriding): V6 §4.1 proposal — UNMODELLED (oxide side in substrate_passivation).; **g_oc_corrosion** (deposit_corrosion): V6 §1.1 proposal — UNMODELLED.; **g_product_ox** (product_oxidation): V6 §1.2 proposal — UNMODELLED.; **g_rinse** (rinse_carryover): V6 §1.3 proposal — UNMODELLED.; **g_briquet** (briquetting): V6 §1.4 proposal — UNMODELLED.; **g_melt_balance** (melt_balance): V6 §1.5 proposal — UNMODELLED.; **g_casting** (strip_casting): Option A.5 gap beyond V6 — UNMODELLED (liquidus, segregation, cast-ability, incl. flotation).; **g_purity** (purification): Train screen exists; feed-fingerprint L1.; **g_form_factor** (bubble_engulfment): V5/B3 porosity model — L1.; **g_magnetic** (magnetic_properties): V6 short-list proposal — UNMODELLED.; **g_pm_finish** (pm_powder_finish): V6 short-list proposal — UNMODELLED.; **g_hot_short** (hot_shortness): V5/E1 ceiling model — L1.

## 6. What the uniform economics hide (read before acting)

- **flake_feed** — Buyer lab trials (yield/boil verification); months, not years. Price risks scrap/DRI substitution.
- **own_melt_bar** — Rebar is the lowest-certification steel SKU (ASTM A615); merchant quality achievable in 2–5 yr. Adds a metallurgical workforce the pure hydromet path avoids.
- **annealed_foil** — Non-structural niches (battery substrates, shielding, brazing foil): customer sampling, 1–2 yr. The drum coupon is the branch-defining experiment (already specified).
- **structural_sheet** — Structural certification (AISI/ASTM structural grades, welding, Charpy): multi-year spec culture. The full-academic-Option-B endpoint.
- **pm_powder** — PM-spec vendor qualification + small market (kt-scale, historically electrolytic-iron-powered).
- **battery_iron** — Customer-owned spec (iron-air developers); price anchor is the most speculative on the ladder — treat as parity target, not market quote.
- **magnetic_foil** — Core-loss certification (Epstein/SST); buyer = motor/transformer niche, not commodity steel.

## 7. Method, limitations, and how to rederive

- Contribution margin, not plant TEA: feedstock, labour, BOP, maintenance, working capital and market risk are held constant across rungs via `technoeconomic.py` conventions; `technoeconomic.py` remains the decision-grade plant model for a chosen rung.
- Live derivations on every run: architecture productivity & capital (`cell_architecture.evaluate_architecture`), DC energy (`electrochemistry.specific_energy_kWh_per_t`), zinc benchmark (`cell_architecture.zinc_tankhouse_productivity`), anneal energy (`thermomechanical.anneal_energy_kWh_per_kg` — anchor fallback only on API change), default electricity price (`technoeconomic.OPEXModel`), gate states (module tree probe).
- New constants on this rung only: product price bands and post-cell unit-operation energies/cash — all in `models/anchors.py` with refs.
- To rederive: `aq-steel-product-ladder` rewrites this file and `experiments/data/product_ladder_report.json` with a fresh provenance stamp. CI hash-checks the stamp (docs/REPO_OUTPUT_POLICY.md).

## Appendix A — rung descriptions

### Option A — passivated briquetted flake (melt-shop virgin units)

The RESEARCH_PROGRAM Option A path: friable powder is a feature, H/C/deposit-stress problems deleted by the buyer's furnace. Competes on iron-unit price against scrap/HBI — the lowest-margin rung.

*Post-cell ops:* counter-current rinse + dry + passivation (80 kWh/t + $8/t), briquetting press (die wear, binder-less) (25 kWh/t + $12/t).

### Option A.5 — own-melt + cast/roll to rebar/merchant bar

The missing middle from the product debate: keep the simple hydromet cell (powder fine, H fine), own the melt step in a commodity induction furnace, ship *steel* and capture the flake→bar margin delta yourself.

*Post-cell ops:* counter-current rinse + dry + passivation (80 kWh/t + $8/t), briquetting press (die wear, binder-less) (25 kWh/t + $12/t), induction melting 25→1,600 °C (550 kWh/t + $45/t), continuous cast + hot roll to bar (150 kWh/t + $60/t).

### Option B-lite — annealed ferritic foil / non-structural sheet

Near-net foil straight off the drum, annealed to ferritic low-C sheet. Keeps the six hard problems Option A deletes — but prices the product at 4–6× commodity iron.

*Post-cell ops:* recrystallization anneal (batch/intercover) (140 kWh/t + $25/t), temper mill / skin-pass (30 kWh/t + $40/t).

### Option B — structural low-C sheet (in-cell C or carburized)

The original program name. Note the economics screen prices it at HRC parity — *lower* than B-lite foil — so within pure economics B ranks below B-lite until volume/qualification learnings dominate. Kept as the identity rung.

*Post-cell ops:* gas carburize case (Option-B C route) (120 kWh/t + $35/t), recrystallization anneal (batch/intercover) (140 kWh/t + $25/t), temper mill / skin-pass (30 kWh/t + $40/t).

### Side — electrolytic PM iron powder

The one niche where aqueous iron EW has survived commercially (99.9 % purity electrolytic powder). Small market, proven route, real premiums over atomized powder.

*Post-cell ops:* counter-current rinse + dry + passivation (80 kWh/t + $8/t), PM-powder sizing/classification (inert gas) (40 kWh/t + $60/t).

### Side — iron-air battery anode feed

Iron-air storage is an electrochemistry-adjacent buyer for clean, porous iron; demand could be large if storage deploys at grid scale. High option value, thin evidence.

*Post-cell ops:* counter-current rinse + dry + passivation (80 kWh/t + $8/t), battery-anode finish (porosity/size spec) (50 kWh/t + $80/t).

### Side — soft-magnetic laminate foil

The drum's 25–50 µm form factor is *already* the eddy-current-optimal lamination thickness. Needs a magnetic property model (coercivity vs grain/inclusions — unmodelled).

*Post-cell ops:* recrystallization anneal (batch/intercover) (140 kWh/t + $25/t), magnetic lamination QA + insulation coat (30 kWh/t + $100/t).

## Appendix B — price & unit-op anchors

| Anchor | Value | ± | Ref | Notes |
|---|---:|---:|---|---|
| FLAKE_FEED_PRICE_T | 450 | 150 | USGS Mineral Commodity Summaries 2025 (iron & steel); Fastmarkets ore-based metallics (HBI/DRI) band 2024–26 | HBI-parity for residual-free virgin iron units; scrap substitution puts the floor ~$300, premium virgin units for flat-products EAF dilution the ceiling ~$600. |
| REBAR_PRICE_T | 750 | 250 | Public trade press rebar/merchant-bar band (CRU, US Midwest), 2024–26 | Lowest-certification finished-steel SKU; the Option A.5 (own-melt) endpoint product. |
| LOWC_FOIL_PRICE_T | 2000 | 800 | Small-volume pure-iron foil vendor list prices (Goodfellow-class suppliers), 2025 | Non-structural ferritic foil niche (battery substrates, shielding, brazing); volume is thin — the band reflects list-price, not deep-market, evidence. |
| HRC_STRUCTURAL_PRICE_T | 850 | 250 | CRU hot-rolled coil band, 2024–26 | Structural sheet at HRC/CRC parity; certification premium only after years of spec work. |
| PM_POWDER_PRICE_T | 2500 | 1000 | PM iron-powder industry price literature (Höganäs handbook; MPIF reviews); historical electrolytic-powder premiums | 99.9 %-purity electrolytic powder niche where aqueous iron EW has survived commercially; premium over atomized powder. |
| BATTERY_IRON_PRICE_T | 3000 | 1500 | Form Energy iron-air public materials + anode-material cost-parity estimate | SPECULATIVE — a parity target for iron-air anode feed, not a market quote.  Widest band on the ladder by design. |
| MAGNETIC_FOIL_PRICE_T | 4000 | 2000 | Non-oriented electrical-steel price band (public trade press, 2024–26); pure-Fe laminate premium assumed ~2× NOES | The drum's 25–50 µm form factor is the eddy-current-optimal lamination thickness; price hinges on certified core loss. |
| DRY_PASSIVATE_KWH_T | 80 | 30 | Industrial dryer energy practice (evaporation of ~10 % w/w residual film: 0.63 kWh/kg water) + inert-gas handling | V6 §1.2/§1.3 — rinse + dry + controlled-O₂ passivation of freshly harvested flake/powder. |
| RINSE_DRY_CASH_T | 8 | 4 | Tankhouse wash-water/reagent handling conventions | Water, N₂ bleed, minor reagents per tonne; screening. |
| BRIQUETTE_KWH_T | 25 | 10 | Roller-press briquetting practice (DRI/HBI industry, 15–35 kWh/t) | V6 §1.4 — press energy only; binder-less against Heckel screen. |
| BRIQUETTE_CASH_T | 12 | 6 | Die/roll wear conventions, DRI briquetting cost reviews | Wear parts + maintenance per tonne; screening. |
| INDUCTION_MELT_KWH_T | 550 | 100 | Coreless induction furnace handbooks (0.50–0.65 MWh/t Fe to 1,600 °C) | Option A.5 core energy term; theoretical minimum ~0.34 MWh/t, practical 0.5–0.65. |
| MELT_CASH_T | 45 | 20 | EAF/induction melt-shop consumables conventions (refractories, slag formers, melt loss) | Excludes the electrowon feed itself; screening. |
| CAST_ROLL_KWH_T | 150 | 75 | Thin/slab cast + hot-bar-mill energy literature | Reheat + rolling energy; yield loss carried in CAST_ROLL_CASH_T. |
| CAST_ROLL_CASH_T | 60 | 30 | Rolling-mill operating cost conventions (rolls, descale, yield loss ~3–5 %) | Screening; buyer-side cost, uniform across products. |
| SKINPASS_KWH_T | 30 | 15 | Temper-mill energy literature (light reductions) | Lüders-band suppression + gauge finish (V6 §7.1 lever). |
| SKINPASS_CASH_T | 40 | 20 | Temper-mill operating cost conventions | Rolls, coolant, yield trim; screening. |
| CARBURIZE_KWH_T | 120 | 60 | models/carburization.py screening furnace energy; ASM gas-carburizing practice | Option-B carbon route; in-cell alternative is V5/A1 carbon_electrodeposition.py. |
| CARBURIZE_CASH_T | 35 | 20 | Gas-carburizing atmosphere/handler cost conventions | Endothermic gas + quench oil + handling; screening. |
| PM_FINISH_KWH_T | 40 | 20 | PM powder sizing/classification practice (screens, inert blanket) | Pyrophoric-safe handling (V6 §1.2); screening. |
| PM_FINISH_CASH_T | 60 | 30 | PM powder finishing/blending cost conventions | QA + fines management; screening. |
| BATTERY_FINISH_KWH_T | 50 | 25 | Specialty porous-metal finishing estimates | Porosity-spec sizing + QA; customer-owned spec (L1 guess). |
| BATTERY_FINISH_CASH_T | 80 | 40 | Specialty anode-material QA estimates | Highly speculative, mirrors BATTERY_IRON_PRICE_T band. |
| MAGNETIC_QA_KWH_T | 30 | 15 | Lamination coating/curing practice (NOES industry) | Interlaminar insulation coat per tonne of foil. |
| MAGNETIC_QA_CASH_T | 100 | 50 | Epstein/SST core-loss certification + coating cost conventions | Certification is the price-gate for this rung. |
| ANNEAL_KWH_T | 140 | 60 | models/thermomechanical.py anneal_energy_kWh_per_kg at 700 °C, furnace efficiency 0.7 — the ladder calls this LIVE | Fallback anchor only; product_ladder derives the working value from the thermomechanical model at call time. |
| ANNEAL_CASH_T | 25 | 15 | Batch/box-anneal operating cost conventions (atmosphere, handling) | Screening; H₂/N₂ cover gas dominant. |

---

*Generated by models/product_ladder.py — full-grade (screening L1 constants)*.
