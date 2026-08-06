# Strategic Positioning — Scale, Feedstock, and the "Cell as the Mill"

Status: forward-looking positioning, not scope. These are the program's defensible
positions on *who the customer is, how the technology scales, and what it should
produce where*. They were pressure-tested in discussion (2026-08-06); this doc records
where the thinking landed so it doesn't have to be re-derived. It does NOT mandate any
specific build or experiment — the physical build/no-build decision gates in
`PROGRAM_SUMMARY.md` still govern what gets built.

---

## 1. Scale-by-replication is the moat, not single-plant scale

Almost every iron-making route scales in **big lumpy plants** — blast furnaces, and
even DRI, carry strong economies of scale and a hard minimum efficient size with
multi-billion-dollar capex. **Aqueous electrowinning is one of the few iron routes
that scales additively**: the architecture is identical, redeployable cells
(cell → crate → site), and volume comes from stacking units, not building a bigger
single unit.

Consequence for how we talk about scale: **100 kt/yr is a reference-unit basis and a
beachhead denominator, not a ceiling.** A site running many crates is a legitimate
target; "we're too small to matter" is the wrong objection to a *modular*, additive
process. This is the strongest and most durable strategic asset — it changes siting,
financing, and deployment risk compared to a lumpy-plant route.

The corollary that keeps it honest: replication scales the *cells*, not the
*feedstock*. Adding crates does not add an ore-dissolution step. The additive-scale
claim is powerful only where a concentrated, soluble Fe²⁺ stream is already available
at that site — which is exactly the waste-feed co-location case below.

## 2. The beachhead is waste-Fe²⁺ co-location, not ore-mine freight

The phrase **"transport iron, not oxygen"** (refine to metal at the mine, ship metal
not ore) is real and is being executed at scale by the majors (Vale/Rio/BHP green-iron
at mine via DRI). But at this program's honest scale and with this technology's actual
feedstock, it points at the **wrong customer**:

- At ~100 kt/yr (or even ×100 crates) the freight arbitrage is a few percent of steel
  value and is dwarfed by the mine-side conversion capex; it is a rounding error on a
  mining major's logistics. The majors pitch mine-side iron on decarbonization, not
  freight.
- Concretely binding: aqueous electrowinning needs **Fe²⁺ in solution**, and oxide
  iron ore is not that. A hematite mine has nothing for the cell to eat without an
  added ore-dissolution/reduction step — the least-proven, most-capex part, and one
  that is **not modeled in this repo**.

**The inversed, defensible pitch** (this is the beachhead): the customer is the
generator who is *already paying to dispose* of a soluble iron stream — spent pickle
liquor (steel mills), copperas (sulfate-route TiO₂ plants), red mud (alumina). You take
the liability, get the iron at negative cost, and solve their disposal problem. You
sell the **disposal-solver**, not the **ore-refiner**. Feedstock-first economics, as
quantified in `FEEDSTOCK_SOURCING_MEMO.md`: copperas ~ −$111 to −$278/t Fe; SPL
−$300 to −$600/t Fe (see also `spl-economics-fact-sheet.md`).

## 3. The "micromill" reframe: the cell is the forming step

A blast furnace makes *metal*; you ship it to a mill to make *product*. Electrowinning
is different in a way that collapses that division of labor: it produces metal **at or
near net shape** (foil off a drum, deposit on a mandrel, powder in a cylinder), **and**
it does so **additively**. So the forming step and the production step can be the same
electrochemical act — a "cell is the mill."

This is the version of the distributed-steelmaking vision that actually leverages the
technology. It reframes the thesis from *"transport iron, not oxygen"* to something
stronger:

> Wherever there is power + a soluble Fe²⁺ stream, make the **product** (not just iron),
> on site — so nothing is transported but the finished good.

Honest constraints that bound this position:
- **Electrowon Fe is not steel yet.** The deposit is high-purity iron (or powder/foil).
  Steel = controlled carbon/alloy + thermal-mechanical history. Carbon incorporation /
  co-deposition and structural-grade properties are **Phase III, L0 — no wet-lab data**
  exists in the repo.
- **Net-shape electroforming is flat and thin.** Drum-and-strip targets ~25 µm foil,
  self-delamination caps ~187 µm; productivity is the capital lever. Best for sheet,
  foil, and near-net thin parts — not structural beams. The "micromill" version is
  **on-cell forming + on-line carburizing/co-deposition** (a steel-like surface/laminate),
  not melting-and-rolling.
- **Don't re-melt.** Re-melting your own electro-metal at ~1,600 °C double-pays energy
  and just becomes a mini-mill fed by electro-iron, cancelling the ambient-temperature
  advantage.

## The three positions in one line

**Modular, additive-scale** (a moat that BF/DRI can't match) × **waste-Fe²⁺ co-location**
(the customer who pays you) × **net-shape forming = a distributeable steel-product
factory** (the "cell is the mill"). Together they say the distributed vision is not
*mines making iron* — it is *wherever there's power + soluble Fe, making product*.

---

## Honest caveats (where the evidence stands)

All strategy above is **positioning on top of L0 screening and design.** Per
`PROGRAM_SUMMARY.md`: no wet-lab data exists in the repository; the program is at the
build/no-build decision, and the physics argues for building. The strategic claims do
not add credibility to the physics — the decision-grade kill criterion
(j ≥ 300 mA/cm², FE ≥ 70%, ≤ 4,000 kWh/t Fe) still gates everything. The "cell as the
mill" arrow in particular fires from the same unproven longbow as every other branch
here: carbon/alloy incorporation in an ambient aqueous deposit, structural-grade
properties, and continuous-foil peelability are all unmeasured Phase III L0 claims.
