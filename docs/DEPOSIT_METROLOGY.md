# Deposit metrology — dual optical + ultrasonic thickness sensing: instrument build/selection spec

**Status:** sanctioned L0 spec (kanban task t_95e7b196). Replaces the design-note
status of the earlier draft; this document is the working instrument spec, and
it includes a runnable dry-run demonstration.
**Tier:** instrumentation plan feeding the L1 sensor set (`docs/TWIN_OBSERVABILITY.md` §4).
**Feeds:** Q5 harvestability and D6 closed-loop observability of the operating twin.
**Not gate evidence.** Everything here is L0 screening scaffold — the error
models are *assumptions to be characterized on the reference cell*, not
calibrated hardware.

The full design-note argument (why a co-live pair at all, even though one direct
observation already restores rank 7) is preserved in the git history of this
file and condensed in §3. This spec assumes that decision: **the two thickness
sensors run permanently co-live** as a cross-validation pair, because an
autonomous, unmanned crate has no on-site human to periodically re-validate a
single survivor sensor.

---

## 1. Purpose and scope

Close the observability gap found in `docs/TWIN_OBSERVABILITY.md`: with the base
5-sensor suite, `deposit_thickness` (state 5) is a decoupled **pure integrator**
with `H[:,5] ≡ 0` — structurally unobservable, non-detectable, and its estimate
diverges under open-loop integration. A direct thickness observation restores
full rank 7. The spec specifies **two** direct thickness observations with
disjoint error models whose agreement/discrepancy feeds the operating twin's
safe-state/trip machinery.

In scope:
- down-selected sensor choice (optical laser-line + pulse-echo ultrasound),
- mounting / line-of-sight in the divided gassed cell,
- calibration procedure,
- per-sensor error models (quantified, L0 assumptions),
- agreement/discrepancy logic → operating-twin safe-state implementation,
- a deterministic dry-run demonstration (simulated agree / disagree).

Out of scope: buying hardware, EKF retuning, real-data calibration, crate/site
integration beyond the existing OFF-by-default hooks.

---

## 2. Sensor selection — the co-live pair

### 2.1 The two technologies

| | **opt-101 (optical laser-line)** | **thk-101 (pulse-echo ultrasound)** |
|---|---|---|
| Physical principle | surface height vs a reference plane (laser-line triangulation / profile) | through-thickness echo time-of-flight; `t = 2·d / v_sound` |
| Line of sight | **required** (sight-glass into the divided cell) | **not required** (contact / wetted transducer) |
| Sensitivity profile | surface topology; reads roughness + refraction + bubble scatter, not a clean buried layer | aggregates over the transducer footprint (smooths local roughness); weak on iron-on-iron |
| Dominant failure | loss-of-line-of-sight / optical fouling → reads high or drops out | weak deposit/substrate echo as deposit thickens/roughens or bubbles muddle → under-reads or high variance |
| Measuring range (spec) | 0–500 µm | 0–500 µm |
| In-loop noise (L0 budget) | σ ≈ 1.5 µm | σ ≈ 4 µm |
| Systematic drift budget (L0) | ≈ 4 µm | ≈ 3 µm |

These are the **disjoint-error** pair: the same nominal quantity reached through
unrelated physics with unrelated biases.

### 2.2 Down-selection criteria and decision

Select by weighted criterion (weight given), with reference cell at 60 °C,
FeSO₄ bath, H₂ evolution, 150 mA/cm² nominal.

| Criterion | weight | optical laser-line | pulse-echo ultrasound |
|---|--:|:--:|:--:|
| Resolves the *divergent* state (deposit thickness) | 30% | ✓ | ✓ |
| Works without line-of-sight (deep-cell autonomy) | 20% | ✗ | ✓ |
| Immune to weak iron-on-iron interface echo | 20% | ✓ | ✗ |
| Unaffected by gas-bubble scatter | 15% | ✗ (partial — de-bubbled sight path mitigates) | partially (aperture/guard) |
| Rugged / non-contact (no coating wear in wetted path) | 15% | ✓ | ✗ (contact wear) |

**Decision: run both.** Neither technology blanket-satisfies the pair of hard
constraints (line-of-sight *and* weak-interface immunity). They are
complementary on exactly the axes that matter, which is the definition of a
cross-validation pair. The lower-noise optical is the *primary* thickness
measurement where it has line-of-sight; the ultrasound is the *persistence*
channel that keeps thickness observable when line-of-sight is lost or fouled.

### 2.3 Candidate instrument classes (build-or-buy, not yet purchased)

- **opt-101:** compact laser-line / point triangulation profilometer (e.g.
  405 nm line scanner) with an IP-rated wetted-window housing; target spec
  repeatability ≤ 2 µm over 0–500 µm. Candidate range: industrial confocal /
  chromatic-confocal or laser-triangulation displacement sensors with an
  immersion-capable optical probe. Buy a probe + industrial controller; the
  integration (sight-glass mount, purge) is the build part.
- **thk-101:** single or dual-element 5–10 MHz pulse-echo transducer + pulser
  receiver with > 1 GS/s digitizer; through-transmission and echo modes. The
  iron-on-iron contrast problem (`docs/DEPOSIT_METROLOGY.md` §3) means the
  instrument must expose *raw A-scan amplitude/echo-location* data, not just a
  canned thickness, so echo-quality (SNR) can be used as a live health signal.

Hardware should be selected against the quantified error budgets in §4 and the
calibration procedure in §6 before any order. No ASINs are committed here: the
point of a *selection spec* is the acceptance criteria, and prices/stock on
niche optics swing (see the shopping-list convention in `docs/SHOPPING_LIST.md`).

---

## 3. What the pair is for — the honest rationale (condensed from the design note)

Two direct observations of the same state **do not** add observability — one
(THK-101 alone) already restores full rank 7. The pair's value is:

1. **Disjoint error characterization** — each technology's bias is an
   independent random variable; agreement ⇒ real confidence, divergence ⇒ a
   systematic error on one channel that is now *detectable*.
2. **Autonomy substitutes for maintenance** — in a lab you run both to
   cross-validate, then drop to one because a human re-checks. In an unmanned
   crate there is no human: the pair is the *continuous* substitute for periodic
   manual validation.
3. **Fault tolerance / graceful failover** — if one channel degrades, the other
   keeps `deposit_thickness` observable (the EKF never returns to the divergent
   open-loop integrator state).
4. **Safe-state input** — a tracked discrepancy feeds the operating twin's trip
   machinery (see §7).

The cost is twice the calibration surface and twice the failure modes — carried
knowingly under the autonomy thesis (§10).

---

## 4. Per-sensor error models (L0 budgets)

These are the numbers the dry-run and any future EKF work use. Each channel's
total standard uncertainty is the RSS of a random in-loop term and a
slowly-varying systematic drift envelope:

```
u_opt = sqrt(sigma_opt_noise^2 + bias_opt^2) = sqrt(1.5^2 + 4.0^2) ≈ 4.27 µm
u_us  = sqrt(sigma_us_noise^2  + bias_us^2)  = sqrt(4.0^2 + 3.0^2) ≈ 5.00 µm
```

- **opt-101:** σ_noise = 1.5 µm (surface scatter); bias budget = 4 µm.
  Failure mode (line-of-sight fouling): large positive bias (modelled as
  ≈ +60 µm ramping in) and/or dropout.
- **thk-101:** σ_noise = 4 µm (footprint averaging); bias budget = 3 µm.
  Failure mode (interface-echo degradation): negative bias / under-read
  (modelled as ≈ −50 µm ramping in) and/or increased variance as the iron-on-iron
  echo weakens.

These deliberately exceed the sim's optimistic single-sensor `σ = 0.5 µm`
(`docs/TWIN_OBSERVABILITY.md` §4 uses σ=0.5 for the EKF assignment). **Neither
technology blanket-meets 0.5 µm**; the plan's whole point is to characterize the
real in-cell numbers, and the agreement logic (§7) uses the honest budgets above,
not the optimistic EKF spec.

> The EKF continues to use the `SensorSpec` noise floors as-is. The agreement
> logic is upstream of the EKF and uses the realistic budgets here, so a single
> optimistic number does not silently suppress a real discrepancy.

---

## 5. Mounting and line-of-sight in the divided gassed cell

The cell is a divided, membrane-enclosed, gassed (H₂-evolving) cell. Mounting
requirements:

- **opt-101 line-of-sight:** a **sight-glass port** in the catholyte compartment
  giving a clear optical path to the deposit surface, roughly normal to the
  cathode. Because the cell is gassed, the sight path must be **de-bubbled**:
  a transparent wetted window (~quartz or a chemically compatible glass) set
  flush into the cell wall, with a slow electrolyte purge or an upward-sloped
  path so H₂ bubbles do not accumulate on the window. The profilometer reads
  the surface height through this window; no part of the optics contacts the
  bath.
- **thk-101:** a **wetted transducer** pocket or clamp-on bracket on the cathode's
  back face (through-stainless) or a directly-wetted transducer on the deposit
  face. Back-face mounting keeps the transducer out of the gas and the deposit
  path, at the cost of added substrate transit time that calibration must
  remove. A guard/ring and a matched acoustic couplant are required for scratch
  contact. Echo-SNR is exposed live as the channel health signal.
- **Co-alignment:** both must sample the **same footprint** of the cathode (or
  overlapping footprints known to the twin) so the pair observes the same
  thickness, not two different regions. Specify a common measurement patch (e.g.
  a 5–10 mm circle) and record the patch coordinate in the data contract.
- **Gas:** all wetted seals and windows must be rated for H₂ and acidic FeSO₄
  bath; any purge gas must not upset the bath chemistry.

The divided-cell compartments also localize where each sensor can even be
placed — the optics require the catholyte side, the ultrasound can go on the
cathode back face. This asymmetry is *why* the pair is complementary (§2.2).

---

## 6. Calibration procedure

Purpose: turn raw optical height and ultrasonic transit time into a common
`deposit_thickness (µm)` with quantified uncertainty, and fix the constants the
error models (§4) assume.

1. **Reference artefacts:** a set of machined Fe-deposit-on-316 coupons of known
   thickness (0, 25, 50, 100, 200, 500 µm — the spec range). Thickness ground
   truth from **weighed mass gain** + density (the Tier B scale from
   `docs/SHOPPING_LIST.md`), cross-checked by a single cross-section micrograph.
2. **opt-101:** fit the surface-height → thickness map; establish the zero
   (bare cathode) plane and the scale factor (µm/pixel or µm/step). Quantify
   refraction-induced bias through the sight-glass for the exact window geometry
   and bath temperature.
3. **thk-101:** measure bulk sound speed *in the working bath* (temperature
   compensated) and the substrate transit-time offset (e.g. back-face through
   316 SS) so only the deposit layer's transit time remains. Characterize the
   minimum resolvable echo above the iron-on-iron interface.
4. **Error-model estimation:** from repeated measurements of the artefacts, fit
   the σ_noise and bias budget of each channel as a function of thickness and
   bath temperature; refine the §4 numbers. Record each fit in the calibration
   record (see `docs/DATA_CONTRACT.md` conventions).
5. **Co-live sanity:** run both on the coupon set, confirm the pair agrees
   within the §7 gates, and record the realized `u_opt`, `u_us` for the gates.

**Verification gate:** calibration passes when the realized per-channel budget
is within a stated factor (e.g. ≤ 2×) of the §4 assumptions *and* the pair
agrees on all reference artefacts at each temperature point. If a channel
exceeds budget, characterize and update the assumption — do not silently widen
the gates.

---

## 7. Discrepancy / agreement logic → operating-twin safe-state

### 7.1 The agreement test

For each synchronized pair `(opt, us)`:

```
d  = opt - us                       # raw discrepancy
u_c = hypot(u_opt, u_us)            # combined standard uncertainty of the pair
z  = |d| / u_c

z <= DEGRADE_Z (3)  -> AGREE      (both channels ok)
DEGRADE_Z < z <= FAULT_Z (6) -> DEGRADED  (one channel suspect; advisory hold)
z >  FAULT_Z (6)      -> FAULTED  (channels discordant; trip)
```

The fused thickness the twin consumes upstream of the EKF is the
**inverse-variance weighted mean** of the two channels (weights
`1/σ_noise²`), which weights the lower-noise optical more heavily when both are
healthy.

### 7.2 Attribution honesty (important)

With **only two** co-live channels, a discordant pair **cannot determine which**
channel is wrong: a biased channel pulls an inverse-variance fused mean toward
itself, so the innocent channel then looks off too. The safe and honest
behaviour is therefore:

- on **DEGRADED / FAULTED**, mark **both** channel qualities non-OK and let the
  operating twin trip/hold;
- **definitive attribution requires a third, independent reference** — a
  periodic weighed-mass coupon (offline), or a coulometric thickness channel
  (Faraday-based), run at a slower cadence. When a third reference is present it
  arbitrates which member of the pair failed, enabling maintenance dispatch.

In-flight there is no human and no third channel, so the twin treats any
sustained discordance as a **sensor-fault safe-state** (conservative and correct).

### 7.3 Feeding the existing trip machinery (no new trip code)

The verdict maps directly onto the operating twin's already-wired
`SensorSnapshot.sensor_quality`:

- AGREE → `{"opt-101": "ok", "thk-101": "ok"}` → normal operation;
- DEGRADED/FAULTED → `{"opt-101": degraded|faulted, "thk-101": ...}` → the twin's
  `_safety_reasons` produces `bad_sensor_quality:...` → latched TRIPPED mode,
  `ShutdownRequest(action="sensor_fault_hold")`, zero-current command.

This reuses the existing safe-state machinery verbatim — no new trip path, no
change to `operating_twin.py`. The co-live pair is a pure *data-source* upgrade
upstream of the twin, consistent with the repo's OFF-by-default convention.

---

## 8. Wiring into the twin — and the dry-run demonstration

### 8.1 OFF-by-default, additive

The pair is implemented as a **validator capability upstream of the EKF**, in
`models/deposit_metrology.py`. It does **not** modify `digital_twin.py` or
`operating_twin.py`; the optic never enters the EKF fusion (the EKF keeps
consuming THK-101). `digital_twin.py`/`L1_SENSOR_OBS_MAP` are byte-identical
with the capability disabled — the same contract as the env-coupling adapter.

### 8.2 The dry run

`models/deposit_metrology.py` (CLI: `python -m models.run_deposit_metrology`)
simulates three scenarios against a growing reference deposit (120 µm/hr,
20×0.1 hr steps, seeded for determinism):

| scenario | fault | expected verdict | operating-twin outcome |
|---|---|---|---|
| `agree` | none | agree | `actuation`, command > 0 A, EKF deposit σ < 3 µm (bounded) |
| `optical_foul` | +60 µm optical bias | faulted (both suspect) | TRIPPED, `sensor_fault_hold`, 0 A |
| `ultrasound_interface` | −50 µm ultrasonic bias | faulted (both suspect) | TRIPPED, `sensor_fault_hold`, 0 A |

It emits `experiments/data/deposit_metrology_report.json` and
`docs/figures/deposit_metrology_dryrun.png`. `tests/test_deposit_metrology.py`
(+ entry point `aq-steel-deposit-metrology`) lock the behaviours: disjoint
uncertainty, agree-under-noise, fault escalation, and the agree→normal /
disagree→trip safe-state transitions.

**Run:**
```bash
.venv/bin/python -m models.run_deposit_metrology
.venv/bin/python -m pytest tests/test_deposit_metrology.py -q
```

---

## 9. Cost / counterweight (stated plainly)

Two live error models to characterize, calibrate, and fail over in an autonomous
system — twice the calibration surface and twice the failure modes. The clean
`σ = 0.5 µm` sim spec is an optimistic single-sensor floor neither technology
blanket-meets. This cost is acceptable under the autonomy thesis but must be
carried knowingly and flagged as an assumption — this is exactly what the
two-sensor plan provides (real error characterization). The two-sensor *hardware*
adds an optics probe + controller and an ultrasonic transducer/pulser to the
instrument budget; the dominant cost is the calibration campaign (§6), not the
parts.

---

## 10. Staging

- **First milestone** (build/no-build + Level-1 calibration + ledger closure): the
  drivers are *periodic/offline* deposit metrology (thickness map + weighed mass
  gain). The live dual pair is **not** required yet — this is where the §6
  calibration explicitly happens.
- **Live dual in-loop pair** is load-bearing at the **operating-twin /
  closed-loop-control stage (L5)**, feeding Q5 harvestability and D6
  closed-loop observability. The dry run (§8.2) makes the pair's safe-state
  behaviour demonstrable in-silico *before* it is bought or mounted.

---

## 11. Decisions and open items

**Decided:**
- Co-live optical + ultrasonic pair is sanctioned (this spec).
- Pair is a validator capability; fused thickness (inverse-variance) is what the
  EKF consumes; optic is not fused into the EKF.
- Discordant pair → both channels flagged → existing `bad_sensor_quality` trip.
  No new trip code.

**Open / pending:**
- Whether a **periodic weighed-mass / coupon third reference** is warranted at
  the L5 stage to arbitrate *which* of the pair failed (enables targeted
  maintenance dispatch) — recommended, costed in §9.
- Final ASIN / vendor selection against the §2 down-selection criteria and §4
  budgets (buy later, after the reference cell's first runs).
- Whether primary + periodic-validator suffices for THIS crate before
  closed-loop control is reached (the design note's original "pending" question
  from §8) — resolved *toward* co-live only at the autonomy/closed-loop stage.

---

## 12. Acceptance criteria

This spec is complete when:
1. the sensor choice is down-selected against stated criteria (§2),
2. mounting/line-of-sight in the divided gassed cell is specified (§5),
3. a repeatable calibration procedure exists (§6),
4. per-sensor error models are quantified (L0) (§4),
5. the agreement/discrepancy logic is implemented OFF-by-default and feeds the
   existing operating-twin trip (§7, §8), demonstrated by the dry run (§8.2),
6. tests lock the agree→normal / disagree→trip behaviour.
