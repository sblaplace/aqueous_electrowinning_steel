# Full Butler–Volmer Branches + DFT-Anchored HER Consistency Check

**Date:** 2026-08-06
**Status:** Level-0 *screening* — model change, **not gate evidence**
**PR:** #43
**Modules:** `models/kinetics.py` (`ButlerVolmerBranch`, `use_butler_volmer`),
`models/transport.py`, `models/diffusion_layer_1d.py` (solver wiring),
`models/her_microkinetics.py` (new, DFT consistency check)
**Tests:** `tests/test_butler_volmer.py`, `tests/test_her_microkinetics.py`,
`tests/test_kinetics.py` (cathodic-regime CE mask)

## What changed and why

### 1. Kinetics branches: full Butler–Volmer replaces Tafel-only

The Fe-deposition and HER branches were pure Tafel laws,
`i = i0·10^((Eeq−E)/b_c)`.  Two artefacts followed:

* `i(E_eq) = i0 ≠ 0` — thermodynamically wrong: at the reversible
  potential the net partial current must vanish exactly;
* nothing anodic of `E_eq` — no Fe dissolution branch, so mixed-potential
  corrosion and honest pulse-reverse (PRE) plating had no representation.

The branches are now full BV with decadic slopes,

```
i = i0 · ( 10^((Eeq−E)/b_c) − 10^((E−Eeq)/b_a) )
```

cathodic positive, signed, exactly zero at `E = E_eq`, negative
(dissolution) anodic of it.  Anodic slopes are not new free parameters:
they are derived from the 25 °C cathodic screening slopes through
`αa·n = n − αc·n`,

| branch | b_c (screening) | αc·n | αa·n = n − αc·n | b_a = 0.05916/αa·n |
|---|---|---|---|---|
| Fe (n=2) | 0.120 V/dec | 0.493 | 1.507 | **0.0393 V/dec** |
| HER (n=1) | 0.140 V/dec | 0.423 | 0.577 | **0.1025 V/dec** |

Design details (see `models/kinetics.py` docstrings):

* The Koutecký–Levich transport cap blends **only the cathodic arm**;
  the dissolution branch returns the kinetic current unchanged (a
  transport-limited dissolution law is outside the screening envelope).
* Galvanostatic solvers (`transport.py`, `diffusion_layer_1d.py`) keep
  their log-current bisection: a BV net current ≤ 0 is treated as
  branch-off there, since dissolution is not part of the cathodic
  operating window these solvers target.
* Current efficiency remains a *galvanostatic* concept, defined where
  both partial currents are cathodic.  `polarization_curve` sweeps that
  run anodic of `E_eq(Fe)` now show `i_fe < 0`; the CE bound in
  `tests/test_kinetics.py` is masked to the cathodic regime with a dated
  comment.
* `use_butler_volmer=False` restores the Tafel-only branches for A/B
  checks.

**Numeric footprint at operating points is nil by construction** — the
reverse term is 10⁻³–10⁻⁸ of the forward term at |η| ≥ 150 mV.  Parity
verified (PR #43): j = 100/200/300 mA/cm² → FE/V identical to the
pre-BV values at printed precision (e.g. j=300: FE 0.985112, V 5.770117).

### 2. DFT-anchored HER microkinetics — consistency check, not a replacement

`models/her_microkinetics.py` builds the HER branch from a mean-field
Volmer–Heyrovský picture anchored on the DFT hydrogen-adsorption free
energy ΔG_H* and checks it against the empirical branch we actually run:

* Fe(110): ΔG_H* ≈ −0.40 eV (screening central value, flagged range
  −0.30…−0.55 eV; Nørskov-family volcano position of Fe).
* Volmer quasi-equilibrium → θ_H pinned at ≈ 1 through the cathodic
  window, so Heyrovský RDS gives a Tafel law with
  b = 2.303 RT/(αF) = 118 mV/dec (25 °C, α = 0.5) — mechanistically why
  iron-group HER lands at 110–140 mV/dec.
* `k_Hey` is anchored by matching the empirical branch at **one**
  reference state; the intrinsic rate inherits the empirical apparent
  Ea (60 kJ/mol) so off-anchor T-ratios isolate the slope form.

Consistency at the reference state (this is the recorded, tested result):
**microkinetic slope 128 mV/dec vs empirical 140 mV/dec (ratio 0.916,
within ~20 %), θ_H ≈ 1.**  Verdict: the DFT picture *supports* the
empirical screening choice and provides the physically pointed direction
for T/pH response; the empirical branch remains the operational default.

## Limitations / honest flags

* Anodic slopes inherit the symmetric-barrier bookkeeping pinned to
  25 °C screening slopes; Fe dissolution at PRE-relevant overpotentials
  remains unvalidated (the `pulse.py` heuristic split is untouched).
* The BV reverse arm changes nothing at cathodic operating points by
  design; its value is correctness at/near equilibrium and the anodic
  regime representation, not better FE numbers.
* ΔG_H* carries the usual CHE/RPA caveats and is not fitted to this
  bath; `her_microkinetics` is a credibility/direction instrument, all
  outputs flagged `unvalidated (L0)`.

## Verification

`pytest tests/test_butler_volmer.py tests/test_her_microkinetics.py
tests/test_kinetics.py tests/test_transport.py` — green; BV pins
`i(E_eq)=0`, Tafel recovery ≥150 mV, anodic growth per b_a, KL cathodic
arm only, transport<->Tafel parity at j = 25–250 mA/cm²; the HER module
pins the 0.916 slope ratio, θ≈1 and the anchoring identities.
