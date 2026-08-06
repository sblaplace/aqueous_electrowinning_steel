# Honest Butler–Volmer Rewrite of the Pulse / Pulse-Reverse Current Split

**Date:** 2026-08-06
**Status:** Level-0 *screening* — model change, **not gate evidence**
**Modules:** `models/pulse.py` (split rewrite), `models/kinetics.py`
(`surface_bv_branches`, `ButlerVolmerBranch.current_scaled`)
**Tests:** `tests/test_pulse.py` (rewritten, 30 tests)
**Artifacts regenerated:** `docs/figures/dc_vs_pulse_comparison.png`,
`docs/figures/pulse_reverse_transient.png`,
`experiments/data/pulse_reverse_report.json` (`python -m models.run_pulse`)

## Why the old split had to go

The pre-2026-08 `_kinetic_split` divided applied current by a concentration-
weighted exchange-current ratio, clipped to [0.01, 0.995], with three
structural dishonesties:

1. **The clip set the answer.**  At the module's own defaults the ratio
   saturated its 0.995 ceiling at every operating point, so predicted FE was
   a parameter of the clip, not of the physics.  (Verified post-hoc:
   swapping the exchange currents by ×100 leaves the legacy FE unchanged.)
2. **Reverse pulses were kinetics-free.**  An anodic segment dissolved
   *exactly* the applied reverse charge with zero HER — by construction.
   In reality the reverse segment sits at a mixed (corrosion) potential.
3. **Transport was double-counted.**  A one-cell-gradient i_lim cap rode on
   top of the Crank–Nicolson film that already owns transport.

## What the BV split does

Per time step, solve for the **one cathode potential** that delivers the
applied current from the two signed Butler–Volmer surface branches
(`kinetics.surface_bv_branches`, transport-free: the resolved film owns
transport, and the pre-existing one-cell cap is gone):

```
i_Fe_BV(E; c_Fe,surf, T) + i_HER_BV(E; pH_surf, T) = j_app(t)
```

with a first-order **surface-activity closure** on both forward arms
(`ButlerVolmerBranch.current_scaled`, scale = c_surf/c_bulk, anodic arms
unscaled — their reactants are the solid / adsorbed H).  Exchange currents
are the gating engine's (`diffusion_layer_1d`) screening family anchored at
50 °C: fe_i0 = 10 A/m², her_i0 = 0.010 A/m²; diffusivities Arrhenius-scaled
from their 25 °C anchors.  E is solved by brentq on the strictly-monotone
total-current map (bracket −3.0 V to E_eq(Fe)+0.6 V vs SHE).

Physics the old form could not represent, now explicit:

* **Off periods corrode.**  At j = 0 the solve lands on the open-circuit
  mixed potential: i_Fe < 0 with equal-and-opposite i_HER > 0.  At the
  canonical screening params: i_corr ≈ −0.13 mA/cm² (pH 2, 50 °C) — about a
  0.1 % net loss per 40 ms rest phase at RC-1 waveforms, but real and
  pH/waveform-dependent.
* **Reverse pulses dissolve more iron than their charge suggests.**
  Applied anodic charge is carried by Fe dissolution PLUS the residual
  corrosion HER (E stays well below E_eq(HER)): measured ≈ +0.53 A/m²
  median excess dissolution on the −20 mA/cm² reverse segment — the
  honest count of what pulse-reverse leveling costs.
* **Deep depletion stays mass-consistent.**  Each forward flux dies with
  its reactant's surface concentration; nothing pins phantom current.

## Before / after (PRE: 100 mA/cm² peak, 10 Hz, 50 % duty, −20 mA/cm² reverse)

| model | cycle FE | plating rate | min c_s/c_b | max pH |
|---|---|---|---|---|
| heuristic, legacy constants (pre-change) | 95.4 % | 63.3 µm/hr | 0.878 | 2.01 |
| heuristic, canonical constants | 95.4 % | 63.3 µm/hr | 0.908 | 2.01 |
| **BV (new default)** | **89.4 %** | **59.3 µm/hr** | 0.913 | 2.12 |

The heuristic row is unchanged by a ×100 exchange-current swap — the clip
artifact — while the BV number moves with the kinetics, as it should.

## Verification

`tests/test_pulse.py`:

* off-period split is an exact corrosion couple (i_Fe < 0, i_HER > 0,
  sum zero; E between the two equilibrium potentials);
* reverse segment: i_Fe ≤ j_reverse with concurrent corrosion HER and
  charge closure i_Fe + i_HER = j exactly;
* both forward fluxes vanish as their surface reactant starves;
* **light-load DC anchor**: a long 5 mA/cm² DC run converges to
  `DepositionKinetics.efficiency_at_current` at matched params within 2 %
  (0.884 vs 0.878 — the Koutecký–Levich blend there is inert at this load);
* **DC late-time anchor**: a 30 s, 100 mA/cm² run converges to the
  independently-solved fixed point of its own steady film equations
  (linear-film identities + BV branches) within 2 % (FE 0.9388, c_s 0.615 M,
  pH 2.21 — agreement < 0.1 % in practice);
* monotone transport response in j_peak (depletion deepens, surface pH
  rises); FE-vs-j direction is deliberately NOT pinned (regime-dependent —
  see envelope below);
* the legacy split is preserved verbatim behind `kinetics="heuristic"`.

## Cross-model context vs the gating engine (`diffusion_layer_1d`, fast_mode)

Matched conditions (δ = 100 µm, 50 °C, pH 2, 1 M Fe²⁺, DC):

| j | pulse BV | DL1D | reading |
|---|---|---|---|
| 5 mA/cm² | 0.884 | 0.878 | converge at light load (Δ0.6 pt) |
| 100 mA/cm² | 0.939 | 0.882 | DL1D's HSO₄⁻/migration proton supply keeps HER alive (its surface pH 2.01 vs our 2.21) |
| 300 mA/cm² | starved (FE→0, flagged) | 0.724 | pulse BV is **outside its envelope** here (see below) |

The honest interpretation: in its valid envelope the pulse model agrees
with the gating engine; the gap grows exactly where the reduced two-species
film's missing proton sources (HSO₄⁻ buffer, water) start to matter.

## Validity envelope and the proton-limited flag

HER past the film's proton supply (~150 A/m² at pH 2, δ = 100 µm, 50 °C)
has no physical carrier in a two-species film.  Previously this failed
silently through the `max(c, 0)` clamp mass leak; now every forward-H⁺
current is concentration-choked, and each step where HER exceeds 2× the
film's steady supply while the surface is starved is counted into
`PulseResult.proton_limited_steps_fraction` (surfaced in `summary()`).
Verified discriminating: 0.000 at the reference waveforms and 100 mA/cm²
DC; 0.382 in the starved 300 mA/cm² steady DC.  **A nonzero fraction means
the result is outside the model's validity envelope** — instantaneous
samples there are unreliable.  RC-1 waveform studies sit at ~0; treat any
positive value as a stop sign, not a number.

## Honest limitations (all L0)

* E_eq(Fe) fixed at −0.440 V with the explicit first-order c-scale
  (`DepositionKinetics` convention); `diffusion_layer_1d` instead
  Nernst-shifts E_eq with fixed i0.  The two forms agree at mild depletion
  and deviate deep in the starved regime — calibrating one against real
  polarization data will settle it; until then both are screening forms.
* The HSO₄⁻/water-coupled HER regime is unmodelled by design (envelope flag
  above), as is Fe(OH)₂ precipitation at high surface pH.
* Off-period corrosion current is a screening estimate (homogeneous BV,
  no passivation, no oxide state); it is right to count it in charge
  balances, wrong to treat the magnitude as validated.
* `pulse_optimization.py`'s frequency CE/carbon factors remain independent
  screening factors on top of the co-deposition model and are untouched.

## Cross-references

* `docs/SIM_BUTLER_VOLMER.md` — the signed BV branches this consumes
  (its "pulse.py heuristic split is untouched" limitation is now closed).
* `docs/SIM_PITZER_ACTIVITY.md` — sibling physics-upgrade note.
