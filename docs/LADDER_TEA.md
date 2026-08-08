# Ladder × TEA — Does the Product Ranking Survive Full Plant Costing?

> **Generated artifact — do not hand-edit.** Rebuild with `python -m models.run_ladder_tea` (or `aq-steel-ladder-tea`) after any change to the models it derives from (product_ladder, cell_architecture, electrochemistry, technoeconomic, thermomechanical, anchors). Numbers re-derive on every run; the comparative structure is what is decision-grade, not the decimals.

<!-- provenance:
```json
{
  "artifact": "ladder_tea",
  "recipe": "python -m models.run_ladder_tea  (full-grade; screening constants are L1 by declaration, not by mode)",
  "mode": "full-grade",
  "source_hashes": {
    "models/ladder_tea.py": "3c029ccd4d1c8ddc",
    "models/product_ladder.py": "ceb7cc868dd87b72",
    "models/cell_architecture.py": "77a08ed3afac2da7",
    "models/electrochemistry.py": "935e526a6598d30c",
    "models/technoeconomic.py": "db224bcf4b144584",
    "models/thermomechanical.py": "03d99ca6f2b215c2",
    "models/anchors.py": "bf1c037f800c88d8"
  }
}
```
-->

**Screening flag:** unvalidated (L1). Companion to [PRODUCT_VALUE_LADDER.md](PRODUCT_VALUE_LADDER.md): the ladder screens rungs on contribution margin; this document prices **the whole plant** against the same live state.

---

## 1. The question

`product_ladder.py` ranks rungs on contribution margin — price minus (DC electricity + post-cell unit ops + installed cell capital) — and deliberately holds feedstock, electrolyte make-up, anode wear, labour, maintenance, insurance, overhead and the ore-side plant constant ("they live in the TEA"). That is a valid screen **only if the excluded lines are rung-independent**. This module runs every rung through the full plant cost stack at a common nameplate and checks the assumption directly.

## 2. Method and deliberate accounting choices

* Nameplate: **100,000 t product/yr** (scenario knob, `--capacity`; per-t variable costs are capacity-free, the capacity-sensitive lines — labour, ore-side plant — are disclosed per tonne).
* Cell block: area = capacity ÷ live architecture productivity; CAPEX = area × live installed $/m². The installed factor is declared by `OperatingConditions` as the round-trip equivalent of `CAPEXModel`'s indirect stack **including rectifier/electrolyte BOP**, so CAPEXModel BOP lines are *not* re-added (no double counting). Only the ore-side plant (leaching + grinding $/tpy) is added.
* OPEX lines from the live `OPEXModel` defaults: electricity (DC + post-op + grinding kWh), ore, electrolyte make-up, water, anode wear (∝ architecture area), post-op cash (rung unit ops), overhead (10 % of variable), maintenance + insurance (∝ CAPEX), labour.
* Money: WACC 8%, 25 yr → CRF 0.0937; NPV = −CAPEX + net × annuity (construction period ignored, L1).
* Conditions: V_cell 2.5 V, FE 0.85, electricity $0.04/kWh — identical to the ladder's live conditions so the gap is attributable to cost structure, not to physics drift.

## 3. Full-TEA results vs the contribution screen

```
rung        opt         $/t mid     full cost   TEA margin  share       ladder m.   gap         NPV M$      IRR         verdict   
----------------------------------------------------------------------------------------------------------------------------
flake_feed      A         450        255         195    43%        307    112      195  146%  clears
own_melt_bar    A.5       750        401         349    46%        474    126      359  261%  clears
annealed_foil   B-lite   2,000        343       1,657    83%      1,786    129    1,731  433%  clears
structural_sheetB         850        386         464    55%        597    133      457  121%  clears
pm_powder       side    2,500        308       2,192    88%      2,309    117    2,326 1000%  clears
battery_iron    side    3,000        331       2,669    89%      2,788    119    2,836 1000%  clears
magnetic_foil   side    4,000        409       3,591    90%      3,726    135    3,795  938%  clears
```

## 4. The screening gap, itemised

Gap = ladder contribution margin − full-TEA margin = exactly the cost lines the ladder excludes. Where it is uniform across rungs it cannot change the decision; where it varies, it names the cost that could.

```
rung            gap $/t   top excluded lines (the ladder's blind spot)
------------------------------------------------------------------------------------------------
flake_feed           112   ore feedstock 40, labour 20, overhead (10% of variable) 20, electrolyte make-up 15
own_melt_bar         126   ore feedstock 40, overhead (10% of variable) 33, labour 20, electrolyte make-up 15
annealed_foil        129   ore feedstock 40, overhead (10% of variable) 25, labour 20, electrolyte make-up 15
structural_sheet     133   ore feedstock 40, overhead (10% of variable) 29, labour 20, electrolyte make-up 15
pm_powder            117   ore feedstock 40, overhead (10% of variable) 25, labour 20, electrolyte make-up 15
battery_iron         119   ore feedstock 40, overhead (10% of variable) 27, labour 20, electrolyte make-up 15
magnetic_foil        135   ore feedstock 40, overhead (10% of variable) 31, labour 20, electrolyte make-up 15
```

## 5. Ranking verdict

* Ladder order (margin $/t): `magnetic_foil > battery_iron > pm_powder > annealed_foil > structural_sheet > own_melt_bar > flake_feed`
* Full-TEA order (margin $/t): `magnetic_foil > battery_iron > pm_powder > annealed_foil > structural_sheet > own_melt_bar > flake_feed`
* Pairwise order flips: **0**

**The ranking is preserved under full costing.** The ladder's contribution screen is decision-grade for *which product to make*; full-TEA margins sit lower in absolute dollars by the itemised gap above (rung-uniform to within $23/t at this capacity). The decision risk is therefore not cost structure but the anchored price bands and the L1 physics flags, in that order.

## 6. Caveats (read before citing)

* All cost lines are `technoeconomic.py` screening defaults; none are quoted projects. Product prices are anchored bands, battery-iron especially speculative.
* Construction period and working capital are ignored; IRR/NPV are ranking aids, not investment-committee numbers (IRR is capped at 1,000 % by the solver, so "1000 %" reads as "above cap").
* The ladder's `clears/stalls` verdicts and this table's may differ: the ladder asks "margin within the screen", this asks "margin within the plant". Both are shown, neither is hidden.
