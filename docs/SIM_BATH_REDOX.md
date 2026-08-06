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

`pytest tests/test_fe3_shuttle.py` — 13 tests: the steady-state identity
(and its k_m-independence), monotonicity in O₂ level, cap behavior and
sludge activation, crossover-fault scaling, ascorbic-acid suppression
hooks, litre/m³ unit pins (erratum below), and the sealed/open/crossover
scenario numbers above.
`python -m models.fe3_shuttle` prints the scenario table shown here.

## Erratum (2026-08-06): two ×1000 m³↔L slips, found while wiring the CSTR

Wiring this chemistry into `bath_dynamics` surfaced two independent litre↔m³
unit slips in the original module.  Both are fixed; unit pins now guard them.

1. **Crossover production overstated ×1000.**  The crossover term divided by
   the volume *in m³* (`flux·A/V_m3`, i.e. mol/m³/s) and was added to the
   homogeneous rate in mol/L/s.  Correct form: `flux·A/V_L` gives mol/L/s
   directly.  Effect on the shipped table: the production *rate* behind the
   `anolyte_crossover_fault` row drops 6.22e-04 → 1.03e-06 M/s.  **The
   headline conclusions are unchanged** — the fault row remains pinned at the
   Fe(OH)₃ cap with its 0.006 % CE loss — but the fault-row sludge estimate
   drops from ≈3000 to ≈5.0 g/L/day.  Physically: at RC-1's tiny A/V
   (1e-3 m² per 0.5 L) a 1 % anode leak is a *modest adder*, not a dominant
   source; it dominates only at pilot-scale A/V (new test
   `test_crossover_dominates_at_large_area_to_volume`).
2. **`iron_sludge_loss_mol_m2_s` understated ×1000.**  A mol/L/s volumetric
   rate was divided by A/V in 1/m; converting M → mol/m³ needs the extra
   ×1000.  (`iron_sludge_loss_g_L_day` was and is correct.)  Curiously the
   two slips partially cancelled in the old mol/m²/s field (3.1e-4 old vs
   4.9e-4 correct for the fault row) — which is exactly the kind of accident
   that lets both survive review.  Both fixed rows are pinned in
   `tests/test_bath_fe3_cstr.py::TestIronLedger`.

## Follow-up shipped (2026-08-06): the shuttle is now a CSTR term in `bath_dynamics`

The "remaining" item from this note and from `SIM_PITZER_ACTIVITY.md`
step 3 is **done**: `models/bath_dynamics.py` integrates the
production → shuttle | sludge triangle in time, behind the
`fe3_shuttle_enabled` design-point flag (default **off** → existing twin
runs are byte-identical; enable via `bath_dynamics.apply_fe3_scenario`).

* **States** (all auxiliary, outside the 7-state EKF vector, so nothing in
  the EKF/Jacobian path changes): `fe3_catholyte_M`, `fe3_reservoir_M`, and
  `fe3_sludge_cumulative_mol` on `BathAux` — the sludge ledger is what makes
  the total-iron ledger close once precipitation runs.
* **Integration**: production (homogeneous autoxidation + crossover fault,
  state-dependent in T/pH/Fe²⁺) and recirculation exchange enter an
  exact-exponential CSTR step — unconditionally stable at any recirc
  stiffness, the same treatment as the cell-voltage relaxation — followed by
  an operator-split instant Fe(OH)₃ cap whose excess joins the sludge ledger.
  The same cap applies in the reservoir at its own pH (a pH ≈ 3.5 balance
  tank holds almost no Fe³⁺).
* **Back-couplings**: autoxidation drains the Fe²⁺ balance while the
  cathodic shuttle returns it (net inventory loss only via sludge); the
  shuttle slip `i_sh = F·k_m·[Fe³⁺]` is subtracted galvanostatically from
  the current available to the Fe/HER pair (deposit growth and HER/OH⁻
  split); and the proton stoichiometry loads the pH balance (−1 H⁺ per Fe²⁺
  oxidised, +3 H⁺ per Fe precipitated — net **+2 H⁺ per mol of sludge**,
  so an ingress bath acidifies until the rising cap throttles precipitation).
* **Cross-validation**: the dynamic bath, held at fixed (T, pH, Fe²⁺) with
  precipitation inactive everywhere, relaxes to the static module's
  closed-form `[Fe³⁺]_ss = r_prod/(k_m·A/V)` (the recirculation terms cancel
  identically at mutual steady state) — pinned at 5 % after ~7 slow-mode time
  constants, plus instantaneous-terms and stiffness pins.

**Tests:** `tests/test_bath_fe3_cstr.py` (11) + the repinned
`tests/test_fe3_shuttle.py`.  All outputs remain L0 screening; the dominant
uncertainties are unchanged (`fe2_autoxidation_k_ref` prior, O₂ ingress
fractions) and none of this is gate evidence.

**Known limitation:** reservoir autoxidation is not modelled — the scenario
O₂ pinning describes the catholyte; a rained-out/aerated balance tank is a
documented extension (and the reservoir cap at least bounds its dissolved
Fe³⁺ consistently).
