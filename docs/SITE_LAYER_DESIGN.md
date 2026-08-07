# Site Layer (L3) Design — will the crate sink, flood, or get wind-exposed?

**D5 design deliverable · SYSTEM_TWIN layer L3 (Site)**
**Pairs with D4** — the crate structural/environmental model (`models/crate.py`,
`docs/SYSTEM_TWIN.md`).
**Status:** screening design (credibility **site L0**) — every number here is an
unvalidated estimate until a real soil test, hydraulic survey, and wind
campaign at the named-beachhead site replace it.
**Machine-readable result:** `aq-steel-site-layer` / `python -m models.run_site_layer`
→ `experiments/data/site_layer_report.json`, `docs/figures/site_layer_design.png`.

---

## 0. What this layer answers

SYSTEM_TWIN splits the twin into three nested layers.  D4 (`models/crate.py`)
answers **"is the container stable as a rigid body?"** — it computes
overturning / bearing / sliding factors of safety, an ingress risk band, and a
required ballast mass for a **named site**.  This D5 layer takes the *next*
step: it turns those crate-level checks into a **site design** — the ground the
crate sits on, the water that drains around it, the wind regime it is exposed
to, and the trucks / power / water / product it must physically connect to.

In short:

> **D4 (crate):** "Does the unit stand up under this wind/rain/soil?"
> **D5 (site):** "What do we build on the ground so it does, for a real site?"

The three deployment questions this design closes:

1. **Will the container sink?** — pad sizing + bearing pressure + ballast to
   meet the overturn target, frost-depth footer. (`FoundationDesign`)
2. **Will the container flood?** — freeboard vs. plausible flood depth, grade
   away from the unit, roof-runoff handling. (`DrainageDesign`)
3. **Will the container get wind-exposed?** — terrain exposure class scaling
   the design gust the crate check must survive. (`WindExposureDesign`)

Plus sitelayout / offtake access (feedstock delivery, grid capacity, water
supply, product haul) — the logistics that decide whether a site can actually
operate even if it is structurally sound. (`LayoutDesign`)

---

## 1. Design blocks

### 1.1 Foundation & ballast — "will it sink?"

Screening inputs from `dark_mill.SiteDefinition` + `dark_mill.ClimateSpec`
(reused, **not modified** — D4 ARENA brief parity):

| Input | Source field | Default |
|---|---|---|
| Allowable pad bearing | `soil_bearing_kPa` | 100 kPa |
| Frost line | `climate.freeze_depth_m` | 0 (none) |
| Crate footprint | 40-ft `CrateSpec` (12.19 × 2.44 m) | — |
| Snow load | `climate.snow_load_kPa` | 0 |

Design operations (`_design_foundation`):

- **Pad geometry** — extends the crate footprint by `pad_margin_m` (default
  0.6 m) on each side → a load-spreading pad for a nominal 40-ft unit.
- **Net pad bearing** — `p_pad = (m_crate + m_ballast + m_snow)·g / A_pad`.
  The check `bearing_ok` requires `p_pad ≤ allow_pad_bearing_kPa`.
- **Ballast** — carries the D4 crate task's `min_ballast_kg` through; where no
  crate verdict is supplied it recomputes the ballast to hit `design_fs_overturn`
  (default 1.5) using `Crate()._required_ballast` at the site's geotechnical wind.
- **Frost** — footer placed at `0.5 m + freeze_depth_m`, so a 0.9 m frost line
  (e.g. US Midwest pickle-liquor site) demands a ~1.4 m footer.
- Outputs a `foundation_type`: "compacted gravel pad + concrete footing" when
  margin is modest, "concrete slab" when the pad is lightly loaded.

**Reading:** a 40-ft crate fully ballasted is a *light* structure — a typical
compacted pad carries 1–15 kPa against a 100 kPa allowable, so pure bearing
almost never sinks a screening design.  The binding constraints are ballast mass
(overturn, D4) and frost depth on cold co-location sites.

### 1.2 Flood & drainage — "will it flood?"

| Input | Source field | Default |
|---|---|---|
| Freeboard | `SiteLayerSpec.freeboard_m` | 0.30 m |
| Plausible flood depth | `site.flood_depth_m` | 0 |
| Min grade | `SiteLayerSpec.min_grade_pct` | 2% |
| Design storm | `SiteLayerSpec.design_storm_mm_hr` | 50 mm/hr |

Design operations (`_design_drainage`):

- **Flood clearance** — `clearance = freeboard − flood_depth`.  Negative ⇒
  **FLOOD**: the design mandates the crate be elevated (`voussoirs/piers`) by
  `|clearance| + 0.15 m`.  `< 0.15 m` freeboard ⇒ "medium" (storm-drain +
  monitor).  This is the direct answer to "will the container flood".
- **Roof runoff** — rational-method roof discharge at the design storm:
  `Q = C·A·I` with `C_roof = 0.95`, off the crate roof footprint, vs a screening
  roof-capacity threshold.
- **Grade** — minimum fall away from the unit (default 2%) + perimeter swale.

**Beachhead reading:** the copperas TiO₂ site (`flood_depth_m = 0`) and the
Appalachia AMD site (`0.10 m`) are clear; the red-mud alumina site (`0.20 m`)
is *marginal* at 0.10 m clearance — the design flags "monitor + storm drain",
and a 1.0 m flash flood (any location) hard-floods and forces an elevated plinth.

### 1.3 Terrain wind exposure — "will it be exposed?"

The crate model already applies a coarse terrain multiplier
(open 1.00 / suburban 0.88 / urban 0.78).  This layer **refines** it into an
exposure class with a roughness-length (`z0`) height correction so a high
anemometer gust is not applied unchanged at crate mid-height, and so drag is
not under-rated on an open site.

| Input | Source field | Default |
|---|---|---|
| Reported gust | `climate.wind_gust_m_s` | 40 m/s |
| Terrain | `climate.wind_terrain` | open |
| Exposure / reference height | `SiteLayerSpec` | 2.6 / 10 m |
| Altitude | `climate.altitude_m` | 0 |

Design operations (`_design_wind`):

- **PBL log-profile height factor** — `h_f = ln(h/z0)/ln(h_ref/z0)` maps the
  reporting-anemometer gust down to crate mid-height.  Screening-only power-law
  substitute; real ASCE exposure coefficients replace it at beachhead stage.
- **Terrain multiplier** — reuses D4's class table (open/suburban/urban).
- **Design gust & pressure** — `v_design = v_reported · h_f · mult`;
  `q = ½·ρ·v_design²`, with `ρ` corrected for altitude.

**Reading:** the wind-farm ore site (55 m/s, open) designs to ~42 m/s at crate
mid-height — still the highest pressure in the library (~1.1 kPa), so it is the
release that D4's overturn check must bind on.  The copperas beachhead (38 m/s,
suburban) designs to ~20 m/s — benign.

### 1.4 Site layout & access — feedstock / power / water / product

This block is the **logistics-of-all** layer: a structurally sound crate is
useless if feedstock cannot arrive, the grid cannot carry it, water is scarce,
or product has no exit.

| Access | Gate (screening) | Source |
|---|---|---|
| Feedstock | distance ≤ 250 km **and** mode ∈ {onsite, pipeline, rail, truck} | `feedstock_distance_km`, `feedstock.transport_mode` |
| Power | `power_required ≤ power_available` | `target_capacity_t_Fe_yr` → MW vs `grid.max_power_MW` |
| Water | supply ≠ "scarce" | `climate.water_availability` |
| Product | road access and market ≤ 1000 km | `product_road_access`, `product_market_km` |
| Fit | designed site footprint ≤ available area | `available_area_m2` |

Power demand is computed from capacity at a screening ~2.6 MWh/t Fe with 85%
load factor; make-up water at ~6 m³/t Fe for H₂ evolution + dragout.  The
designed footprint scales with the crate and feedstock turnaround.

**Beachhead reading:** every example site passes feed/power/product/fit —
including the **copperas TiO₂ beachhead** (all four green).  The wind-farm ore
site is the one structural miss: **scarce water ⇒ layout no-go** until water is
trucked in ("secure water supply / truck-in make-up").

---

## 2. Design verdicts on the real candidate sites

| Site | Sink | Flood | Wind-exposed | Binding L3 driver |
|---|---|---|---|---|
| **Copperas TiO₂ (recommended beachhead)** | no | no | no | — clear on all L3 gates |
| Pickle liquor, US Midwest | no | no | no | frost → 1.4 m footer; suburban exposure |
| Red-mud alumina refinery | no | **marginal** | no | 0.20 m flood → storm drain + monitor |
| Acid-mine drainage, Appalachia | no | no | no | 0.10 m flood clear; tight 200 m² site |
| Wind-farm ore | no | no | no (42 m/s design) | **scarce water** ⇒ layout no-go |

The copperas beachhead (the D5 anchor from `docs/FEEDSTOCK_SOURCING_MEMO.md`)
comes out green on foundation, flood, wind, **and** layout — it is not just the
economic winner, it is the *easiest site* to physically deploy on.  That is the
strongest single D5 conclusion.

---

## 3. What is deliberately deferred

- **FE / CFD structural analysis** — the reduced-order rigid-body + envelope
  checks (D4) and the design here are the screening step; FE/CFD is a
  named-geometry, beachhead-stage activity (SYSTEM_TWIN § 7).
- **Jurisdictional codes (IBC/ASCE/local BFE)** — the simplified exposure
  factors and freeboard are placeholders for the authority's seismic / wind /
  flood code at the named site.
- **Geotechnical / hydraulic surveys** — `soil_bearing_kPa`, `flood_depth_m`
  and `wind_terrain` are desktop assumptions; a pad has no real bearing until a
  soil test says so.  Everything is flagged **site L0**.

---

## 4. Wire-in and verification

- **Entry point:** `aq-steel-site-layer` (CLI), `python -m models.run_site_layer`.
- **API:** `models.SiteLayer().evaluate(site, crate_verdict=cv)` → `SiteLayerVerdict`
  with `.flat()` (JSON) and `.summary()`.  Reuses `dark_mill.SiteDefinition` and
  the D4 `CrateVerdict` — nothing existing was modified (pure additive).
- **Tests:** `tests/test_site_layer.py` (18 cases) — foundation, drainage,
  wind, layout/access, water, and crate-verdict composition.
- **Credibility:** `site L0` — on the ladder in SYSTEM_TWIN § 4, this design is
  Level 0 until a real site survey / load test elevates it.  It must never be
  quoted as an authoritative civil-engineering clearance.