# System Twin: from a cell twin to a whole-system picture

**Date:** 2026-08-01
**Status:** vision / scoping
**Purpose:** define the digital twin as the *complete system*, not just the
electrochemical cell — so that questions like "What does the crate do in the
wind? In the rain? How does it run on a twin of a mining site? How does it need
to mount to the ground for stability?" can be answered from one coherent model
stack.

## 1. The problem the current twin does not answer

The repository's twin machinery to date is **process-centric**:

- `models/digital_twin.py` — extended-Kalman state estimation of the *cell*
  (temperatures, pH, voltage, current) against a physics surrogate.
- `models/operating_twin.py` — the *safety/supervisory shell* for one cell
  (trips, arming, bounded actuation).
- `models/twin_physics.py` — the physics measurement model for the cell
  (FE, V_cell, surface pH, deposit rate).
- `models/dark_mill.py` — a *site* twin, but only as **process + economics**:
  it sizes the modular mill and asks "does this site make sense?" Its
  `ClimateSpec` (temperature, humidity, altitude, freeze) drives thermal and
  winterization sizing only.
- `models/supply_chain.py` — decentralized-vs-centralized deployment economics.

What none of these model is the **crate as a physical object sitting on real
ground under real weather**. `dark_mill.SiteDefinition` has no wind speed, no
rainfall, no soil bearing capacity, no seismic zone, no crate mass/dimensions,
no foundation or mounting. "Will the container blow over, leak, or sink?"
is currently unanswerable by the twin — and that is exactly the deployed,
autonomous, containerized future the program is aiming at
(`docs/RESEARCH_PROGRAM.md` § Site redeployment).

This document closes that gap in scope: **the twin is the whole artifact —
enclosure, structure, environment, and site — with the cell as one of its
layers.**

## 2. The three-layer stack

| Layer | What it is | Modeled today | Missing |
|---|---|---|---|
| **L1 · Process** | the cell: thermodynamics, kinetics, transport, FE, voltage, deposit | `cell_physics`, `digital_twin`, `twin_physics`, `operating_twin`, `diffusion_layer_1d` | calibration to real data |
| **L2 · Unit / crate** | the containerized autonomous unit: enclosure, HVAC/thermal, power/water/gas handling, **structure** | partial `dark_mill` (process sizing only) | **wind, rain/ingress, snow, thermal, mounting/stability** |
| **L3 · Site** | where the crate sits: ground/foundation, drainage/flood, climate regime, terrain, layout, feedstock/power/water/product access | `dark_mill` (economics), `supply_chain` | **ground bearing, foundation/ballast, flood, terrain wind exposure** |

Layers are **tightly nested**: the cell (L1) lives inside the crate (L2),
which sits on the site (L3). Loads and constraints flow down (site climate →
crate surface load → cell environment), and production/weight flow up
(cell output → crate throughput → site offtake). A whole-system twin must
resolve the loop, not just the cell.

## 3. The structural/environmental layer (the new ground)

This is a new engineering domain for the repo — structural/environmental/civil
mechanics — and it maps the open questions directly:

- **Crate in the wind.** Reference wind speed from the site (terrain category
  + local climate) → dynamic pressure on the crate's projected area →
  wind force and **overturning moment** vs. restoring moment (self-weight +
  ballast) → factor of safety → anchorage / tie-down requirement.
- **Crate in the rain.** Rainfall intensity (site climate) → roof/side drainage
  and runoff → ingress/sealing assessment → indoor humidity budget (electrical
  safety, corrosion of power/controls) → flood risk at grade for the siting.
- **On a twin of a mining/waste site.** Ground bearing capacity vs. the crate's
  footprint pressure → pad / cribbing / grout requirement; drainage away from
  the unit; proximity to feedstock, power, water and product offtake.
- **Mounting for stability.** Net bearing pressure, **overturning**, **sliding**,
  and **uplift** checks across wind + seismic + snow load combos → the minimum
  ballast mass and tie-down configuration needed for a given site.

Target outputs (screening-grade, consistent with everything else in the repo):

- overturning factor of safety `FS_overturn` (≥ ~1.5–2.0 for screening);
- net ground bearing pressure vs. allowable (`p_net ≤ p_allow`);
- sliding `FS_slide` with/without anchors;
- minimum ballast mass and footprint pad spec;
- ingress/rain risk flag and required drainage/sealing level.

Like every downstream model, these are **screening** figures until a real
load test / site survey validates them — see the ladder below.

## 4. Credibility ladder — generalized per layer

The cell-centric L0→L5 ladder in `docs/NEXT_STEPS.md` applies *per layer*.
Structure and environment are just as credibly gated:

| Level | What a layer may claim |
|---|---|
| 0 — screening | trends, bounds, experiment/site ranking under transparent assumptions |
| 1 — calibrated | one named unit/cell predicted from measured inputs |
| 2 — reference | spatial fields (loads, flow, temp, composition) over a run |
| 3 — durability | drift under wear, ageing, corrosion, repeated load cycles |
| 4 — design transfer | prediction across a new geometry/site within an uncertainty envelope |
| 5 — operating | constrained online estimation/control with validated sensors + independent shutdown |

A whole-system twin is therefore a **vector of layer-credibilities**, not a
single label. We should never say "the system twin is Level 5" — we should say
"process L0/L1, crate L0, site L0" and drive each independently. A handsome
structural model is still Level 0 until a wind/load test or site survey checks
it.

## 5. Architecture & composition

New module (proposed), placed beside the cell twin:

- `models/crate.py` (or `system_twin`) — the **unit/crate structural +
  environmental model**:

      crate geometry (L×W×H, mass, centroid, footprint)
      + environmental loads (wind regime, rainfall, seismic zone, snow)
      + ground/foundation spec (bearing capacity, drainage, pad/ballast)
      → stability verdicts: FS_overturn, p_net/p_allow, FS_slide,
        min_ballast_kg, ingress_risk, mounting_spec

- It **consumes** `dark_mill.SiteDefinition` / an extended `ClimateSpec` +
  `SiteSpec` (new structural/environmental fields) at L3.
- It is **consumed by** the site assessment (`dark_mill` go/no-go) and by the
  operating twin (environmental limits that bound permissible operation — e.g.
  a high-wind storm sets a safe-state condition).

The data flow stays one-directional and acyclic where possible:
`site → crate → cell`, with the site assessment aggregating all three layers
into the final go/no-go.

## 6. Dependency order

1. **Capture this vision** (this document).
2. **Extend the site definition** with structural/environmental inputs
   (wind regime, rainfall, soil, seismic, terrain) — small, additive.
3. **Build the crate structural model** (wind overturning, rain/ingress,
   foundation/ballast, stability checks) — screening grade, tested.
4. **Wire it into `dark_mill`** so the site go/no-go consumes a physical/
   environmental verdict, not just economics + process.
5. **Later:** couple to `digital_twin`/`operating_twin` so environmental
   limits drive safe-state conditions; add per-layer calibration as data
   (wind/load tests, site surveys) arrives.

## 7. Explicitly defer

- Full finite-element / CFD structural simulation — the reduced-order envelope
  and stability-factor checks are the screening step; move to FE only when a
  named site/geometry demands it.
- Seismic/civil code compliance (IBC/ASCE) — replace simplified factors with
  jurisdiction codes only at the named-beachhead stage.
- Adding structural modeling earlier than the process Layer-1 calibration —
  the cell's chemistry is still the dominant uncertainty; the crate model is
  built so it is ready when a site is named, not as a substitute for chemistry.

## 8. Immediate next action

Write the crate structural/environmental module as a screening twin with its
own tests, extend the site definition with the needed load/ground fields, and
compose it into the `dark_mill` site assessment. All screening numbers are
flagged as unvalidated until a real load/site check.
