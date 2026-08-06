# Fe³⁺ Shuttle + O₂-Driven Bath Aging — Screening Note

**Date:** 2026-08-06
**Status:** Level-0 *screening* — new module, **not gate evidence**
**PR:** #43
**Modules:** `models/fe3_shuttle.py` (new; reuses
`models/bath_startup.py` autoxidation chemistry)
**Tests:** `tests/test_fe3_shuttle.py` (11 tests)
**Registry:** `fe2_autoxidation_k_ref` (lognormal around the Sung & Morgan
screening value)

## What this computes

`bath_startup.py` covers homogeneous Fe²⁺ autoxidation chemistry in a
still bath.  During electrolysis the loop closes:

```
production ──► [Fe³⁺] ──► shuttle sink (cathode: Fe³⁺ + e⁻ → Fe²⁺)
   │                            │
   └──► Fe(OH)₃ cap ◄──────────┘  excess precipitates (iron-inventory loss)
```

* **Production** — the inherited Singer-&-Stumm-family rate law
  `rate = k_eff(T)·[Fe²⁺]·[O₂]·([OH⁻]²-relative)` with dissolved O₂
  pinned at a scenario fraction of Weiss air saturation (sealed ≈ 0.5 %,
  open headspace ≈ 100 %), plus an optional anolyte crossover fault — a
  fraction of the anode's `j/(4F)` O₂ generation reaching the catholyte.
* **Shuttle sink** — mass-transfer-limited cathodic reduction,
  `flux = k_m·[Fe³⁺]`, `k_m = D_Fe3/δ` → parasitic current
  `i_sh = F·k_m·[Fe³⁺]` that steals current efficiency.
* **Hydrolysis cap** — `[Fe³⁺] ≤ Ksp/[OH⁻]³` (Fe(OH)₃, log Ksp ≈ −38.7);
  production beyond the shuttle sink at the cap precipitates as sludge,
  which is the *iron inventory* loss channel.

The steady state has the clean closed form

```
[Fe³⁺]_ss  =  min(cap, r_ox / (k_m·A/V))
i_shuttle,ss = F·(V/A)·r_prod            (while below the cap)
```

i.e. the parasitic shuttle fraction is **independent of k_m** — the
classic divided-cell CE leak, set by oxygen ingress and cell V/A, not by
diffusion-layer assumptions.

## Numbers at RC-1 (j = 300 mA/cm², 50 °C, pH 2, V = 0.5 L)

| scenario | dissolved O₂ | [Fe³⁺]_ss | shuttle CE loss | sludge |
|---|---|---|---|---|
| sealed divided cell | 0.5 % sat | 9.4e-05 M | **0.003 %** | no |
| open headspace | 100 % sat | 1.8e-04 M (cap) | **0.006 %** | **yes** |
| 1 % anolyte crossover fault | 100 % sat | 1.8e-04 M (cap) | **0.006 %** | **yes** |

## What the screen actually says

* The **CE penalty is a non-issue even wide open** (< 0.2 pp): the shuttle
  saturates at the Fe(OH)₃ solubility cap, below which the leak is tiny at
  hardness-relevant current densities.
* The real damage of O₂ ingress is the **iron-inventory bleed to Fe(OH)₃
  sludge** (reported both as mol/m²/s and g/L/day), plus eventually pH
  drift — not direct current efficiency.
* Dominant uncertainties are the homogeneous rate constant (now a
  registry prior, `fe2_autoxidation_k_ref`) and the O₂ ingress fractions;
  nothing here is gate evidence.

## Verification

`pytest tests/test_fe3_shuttle.py` — 11 tests: the steady-state identity
(and its k_m-independence), monotonicity in O₂ level, cap behavior and
sludge activation, crossover-fault scaling, ascorbic-acid suppression
hooks, and the sealed/open/crossover scenario numbers above.
`python -m models.fe3_shuttle` prints the scenario table shown here.
