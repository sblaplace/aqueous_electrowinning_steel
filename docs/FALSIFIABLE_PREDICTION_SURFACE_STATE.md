# Falsifiable Prediction — RC-1 Surface-State HER Suppression

**Branch:** `arena/019fdad2-aqueous-electrowinning-steel`  
**Module:** `models/surface_state.py` (`SurfaceStateKinetics`)  
**Claim type:** Mechanism (not scenario knob)  
**Date:** 2026-08-07

---

## What is being claimed

The chloride-rich bath (AWARE, 10 M LiCl, pH 2, 60 °C) suppresses HER
more than the sulfate bath (1 M FeSO₄ + 0.5 M Na₂SO₄ + 0.4 M H₃BO₃,
pH 2, 60 °C) at η ≈ 0.2 V, **not because of a tuned `her_i0` parameter**
but because the surface-state mechanism predicts it:

1. **Site-blocking (dominant, robust, ~14×)** — Cl⁻ specifically
   adsorbs at the IHP (θ_block ≈ 1 at 10 M) while SO₄²⁻ does not
   (θ_block ≈ 0.22 at 1.5 M). A Cl-covered site cannot form H*.
2. **Temkin coverage (robust)** — The mixed-facet ensemble
   (55 % (110) / 30 % (100) / 15 % (211) for sulfate; 50 % / 35 % / 15 %
   for chloride) carries ΔG_H* ≈ −0.40 eV (weighted), so θ_H is pinned
   near 1. The empty-site factor θ_H(1 − θ_H) is already small; blockage
   multiplies it down.
3. **Facet ensemble (robust, ~5 % shift)** — The chloride bath's
   slightly finer grain texture shifts ΔG_H* by ~5 %, a measurable but
   secondary contribution.

These three terms together give the **robust mechanism prediction**
(`use_frumkin=False`, the adapter default). The suppression is
independent of the calibration-fragile `eta_screening` amplifier.

---

## The Frumkin ψ₁ amplifier (opt-in, flagged OFF by default)

The `SurfaceStateKinetics` adapter carries `use_frumkin=False` by
default. When `True`, the Frumkin correction (`ψ₁ < 0` → effective
η larger → HER suppressed further) applies. This is shown **only**
as a sensitivity band, not as part of the core claim, because it
swings the suppression ratio from 44× (`eta_screening` = 0.01) to
10⁶× (`eta_screening` = 0.20) — the classic calibration-fragile
amplifier that hides mechanism behind a single tuning knob.

---

## Concrete RC-1 predictions

### Reference conditions (screening, 60 °C, pH 2, η = 0.2 V, 1 µm grain)

| Bath | θ_block | θ_H | θ_H(1−θ_H) | i₀,H_eff / i₀,H_intrinsic (robust, Frumkin OFF) | Frumkin factor (`eta_screening`=0.05) | i₀,H_eff / i₀,H_intrinsic (Frumkin ON) |
|---|---|---|---|---|---|---|
| **Sulfate** (screening) | 0.22 | 0.88 | 0.106 | **0.084** | 0.85 | **0.071** |
| **AWARE / chloride** (screening) | 0.99 | 0.85 | 0.128 | **0.0092** | 0.0011 | **1.0×10⁻⁵** |

### Robust mechanism claim (site-blocking + coverage + facets only)

```
Suppression ratio (chloride vs. sulfate) = 0.084 / 0.0092 ≈ 9.1×
```

But because the reference bath recipes include competitive multi-anion
adsorption (sulfate bath = SO₄²⁻ + HSO₄⁻ + B(OH)₄⁻; aware bath = 10 M Cl⁻
+ residual 0.5 M SO₄²⁻ mock), the full competitive Langmuir model
(used by the adapter) produces:

```
Site-blocking-only ratio (sulfate / chloride) ≈ 14×  (at η = 0.2 V)
```
This is the headline number from `test_site_blocking_only_ratio_is_robust`.
The **core claim** is:

> **At η = 0.2 V, 60 °C, pH 2, the chloride bath suppresses the HER
> exchange current density by ~14× relative to the sulfate bath, driven
> almost entirely by anion site-blocking and hydrogen-coverage effects.
> The suppression ratio does NOT vary with `eta_screening` when
> `use_frumkin=False`.**

### Frumkin sensitivity band

With `use_frumkin=True` (opt-in), the total suppression ratio grows
with `eta_screening` as required by the physics:

| `eta_screening` | Total suppression ratio (sulfate / chloride) | What it means |
|---|---|---|
| 0.00 | 14.0 | Site-blocking + coverage only (robust core) |
| 0.01 | 44 | Lower end of cited experimental range |
| 0.02 | ~70 | Within Bockris & Jeng 1990 range |
| 0.05 (screening central) | 238 | Screening central value |
| 0.10 | ~10⁴ | High screening, unverified |
| 0.20 | ~10⁶ | Upper extreme, calibration-unstable |

**The user-facing claim must quote the robust core (~14×), not the
headline (~238×).** The headline is a sensitivity, not a prediction.

---

## How RC-1 confirms or refutes

### Confirm
Run two galvanostatic batches at 100 mA/cm², 60 °C, pH 2:

- **Batch A (sulfate):** 1 M FeSO₄ + 0.5 M Na₂SO₄ + 0.4 M H₃BO₃
- **Batch B (chloride / AWARE):** 1 M FeCl₂ + 10 M LiCl (no borate)

Measure current efficiency (CE = Fe-deposited charge / total charge)
at the same applied current. The robust prediction is:

```
CE(sulfate) ≈ 60–75 %
CE(chloride) ≈ 85–95 %  (i.e. ~1.3–1.6× improvement, not 14× directly,
                      because CE = i_Fe / (i_Fe + i_HER) is non-linear)
```

More precisely: the HER partial current should drop by ~14× at the
same overpotential. If `i_HER(sulfate) ≈ 3.0 A/m²` at η ≈ 0.2 V
and `i_HER(chloride) ≈ 0.2 A/m²`, the mechanism claim is confirmed.
The Frumkin band predicts `i_HER(chloride)` could be 10⁻³ to 10⁻⁴
A/m² if `eta_screening` is at the upper end — but the **robust**
prediction requires only the ~14× suppression, not the extreme values.

### Refute
If both baths give the same CE (within ±5 %) at the same operating
point, or if the chloride bath gives *worse* CE, the mechanism claim
is falsified. The Frumkin amplifier alone (`use_frumkin=True`) does
not rescue the claim — if the core site-blocking + coverage model
fails, the adapter's `use_frumkin` flag does not provide an alternative
explanation.

---

## Code references

- `models/surface_state.py`: `SurfaceStateKinetics`, `SurfaceCoverage`,
  `FacetDistribution`, `AnionCoverage`, `chloride_aware_default()`
- `tests/test_surface_state.py`: `TestUseFrumkinOptIn`,
  `TestEtaScreeningPropagation`, `TestHeadlineRatioIsNotInvariant`,
  `TestFrumkinSensitivitySweep`
- Adapter interface: `kinetics.py` (additive, behind `her_i0_T` seam)
- Propagation fix: `surface_state()` passes `self.anion_coverages`
  directly, avoiding the silent default-rebuild that previously
  flattened the `eta_screening` dependency.

---

## Model-backed claim update (post-PR #51)

The adapter is now wired for direct comparison: `SurfaceStateKinetics`
can be called from any FE evaluation path. The mechanism predicts the
chloride/sulfate gap directly; the claim in this document is backed by
`models/surface_state.py`, not only by external scenario parameters.
FE wiring into `pulse.py` / `coupled_cell_physics` remains the named
next step (see PR description).
