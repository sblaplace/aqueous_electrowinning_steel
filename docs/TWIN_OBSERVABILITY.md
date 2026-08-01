# Twin observability & sensor-placement analysis

**Tier:** L0 (screening). **Data:** none — analysis only. **Model:** PR #27 EKF
over 7 physical states with the `twin_physics` surrogate measurement model.

This document answers, before the reference cell's first run, whether the
7-state EKF state vector is **observable** from the current 5-sensor suite, how
well each state is observed, and what sensor additions make the weakly- or
un-observed states directly measurable. It feeds the L1 instrument spec.

Run it deterministically:

```bash
python -m models.run_observability
# or, after installing the project entry points:
aq-steel-observability
```

The analysis module is `models/observability.py`; its tests are
`tests/test_observability.py`. It reuses the twin's existing numerical
Jacobians (`_F_jacobian`, `H_jacobian`) and does **not** modify the EKF state
vector or `h_obs`.

---

## 1. Setup and method

**State vector** (`STATE_KEYS`, indices):

| idx | state | unit | direct sensor (current suite) |
|----:|-------|------|-------------------------------|
| 0 | catholyte_temperature | °C | TT-101 |
| 1 | anolyte_temperature | °C | TT-201 |
| 2 | bulk_fe2 | M | *(none — via VT-201 coupling)* |
| 3 | bulk_pH | pH | pHAT-101 |
| 4 | current_density | mA/cm² | CT-201 |
| 5 | deposit_thickness | µm | *(none)* |
| 6 | cell_voltage | V | VT-201 (see note §4) |

**Method.** At each representative operating point we build the linearized pair
`(F, H)` (numerical Jacobians of the state transition and measurement model),
form the finite-horizon observability matrix `O_N = [H; HF; …; HF^{N-1}]` and
Gramian `W_N = O_Nᵀ O_N` over a 12 h horizon (a few dynamics time constants).
`rank(W_N) == 7` ⇒ the full state is reconstructible. A state is *structurally
unobservable* when its column of `O_N` is identically zero. Whether an
unobserved state's error stays bounded is read off a finite-horizon Kalman
covariance recursion (detectability): a pure-integrator mode diverges, a
contractive mode does not.

**Operating points** (physics-surrogate grid centre + corners):

| point | j (mA/cm²) | T (°C) | Fe²⁺ (M) |
|------|-----------:|-------:|---------:|
| nominal | 150 | 60 | 1.00 |
| lo_j_lo_T_lo_fe2 | 50 | 40 | 0.50 |
| hi_j_hi_T_hi_fe2 | 250 | 80 | 1.50 |
| lo_j_hi_T_hi_fe2 | 50 | 80 | 1.50 |
| hi_j_lo_T_lo_fe2 | 250 | 40 | 0.50 |

---

## 2. Observability of the current 5-sensor suite

**The current suite is rank-deficient: `rank(W_N) = 6` at every tested point.**
6 of the 7 states are reconstructible; only `deposit_thickness` is not.

| operating_point | rank | smallest nonzero σ | condition |
|-----------------|-----:|-------------------:|----------:|
| nominal | 6 | 5.89e-04 | 5.15e+05 |
| lo_j_lo_T_lo_fe2 | 6 | 6.55e-05 | 4.63e+06 |
| hi_j_hi_T_hi_fe2 | 6 | 1.64e-03 | 1.86e+05 |
| lo_j_hi_T_hi_fe2 | 6 | 6.55e-05 | 4.63e+06 |
| hi_j_lo_T_lo_fe2 | 6 | 1.64e-03 | 1.86e+05 |

Per-state observability at the nominal point:

| state | flag | obs. score | rel. energy | est. σ |
|-------|------|-----------:|------------:|-------:|
| catholyte_temperature | observable | 1.000 | 0.057 | 0.239 °C |
| anolyte_temperature | observable | 1.000 | 0.070 | 0.291 °C |
| bulk_fe2 | observable | 1.000 | 0.004 | 0.032 M |
| bulk_pH | observable | 1.000 | 0.061 | 0.045 pH |
| current_density | observable | 1.000 | 1.000 | 0.386 mA/cm² |
| deposit_thickness | **unobservable_divergent** | 0.000 | 0.000 | 5.07 µm (growing) |
| cell_voltage | **weak** | 1.000 | 0.002 | 0.076 V |

### 2.1 deposit_thickness (state 5) — unobservable and divergent (confirmed)

`H[:, 5] ≡ 0` across all five tags (deposit thickness appears in no output
equation of `h_obs`), and `F[:, 5] = e₅` — it is a decoupled **pure integrator**
that accumulates `deposit_rate_um_hr`. Its column of `O_N` is therefore
identically zero (score 0). Because the mode is neutral (eigenvalue 1), it is
**not detectable**: the estimation error is pure open-loop integration and any
model error in `deposit_rate` appears as unbounded accumulated bias. Over a
24 h run the deposit covariance grows monotonically from `P₀ = 1.0` to ≈ 26.3
(σ ≈ 5.1 µm) and would keep growing without bound. This is the load-bearing
sensor gap in the current suite.

### 2.2 bulk_fe2 (state 2) — observable through the v_cell coupling

`bulk_fe2` reaches the output through the VT-201 row, because the physics
surrogate `v_cell(j, T, Fe²⁺)` depends on Fe²⁺ (`H[VT-201, 2] ≈ −0.07 M⁻¹`). It
is *observable* (score 1.000, flag observable), with end-of-run σ ≈ 0.032 M.
Its Gramian energy is low (~0.4 % of the best-observed state), so it is thinly
conditioned — recovering a tight bulk-Fe²⁺ estimate from voltage alone relies
on an accurate, calibrated `v_cell` surrogate. An inline probe further tightens
it (see §3).

### 2.3 cell_voltage (state 6) — weakly observable through the coupled dynamics

VT-201 observes the physics-predicted `v_cell(j, T, Fe²⁺)` (`h_obs` returns
`pred.v_cell_V`), not the `cell_voltage` state `x[6]`, so no measurement row
directly observes `x[6]`. However, under the PR #29 coupled-bath dynamics `x[6]`
relaxes toward the observed `v_cell` with a fast, physically-based electrical
time constant, so `x[6]` becomes **weakly observable** through that propagation
(score 1.000, flag `weak`) with a small, bounded end-of-run error (σ ≈ 0.08 V).
This is a measurement-model alignment gap, not an operational blind spot — the
physical VT-201 exists; the twin simply never reconciles `x[6]` against it
directly, and a direct observation (CVT-201) removes the residual weakness.

---

## 3. Sensor-placement ranking

Candidate sensors were evaluated by marginal reduction in their target state's
end-of-run estimation-error variance, with observability-restoring sensors
ranked first (and divergent modes ahead of bounded ones).

| rank | tag | target state | var before → after | reduction | rel. | resolves? |
|-----:|-----|--------------|-------------------:|----------:|-----:|-----------|
| 1 | **THK-101** | deposit_thickness | 25.75 → 0.117 | 25.63 | 1.00 | ✓ (divergent) |
| 2 | **CVT-201** | cell_voltage | 0.0058 → 0.0001 | 0.0057 | 0.98 | conditioning (weak) |
| 3 | **FE2P-101** | bulk_fe2 | 0.0010 → 0.0003 | 0.0008 | 0.72 | conditioning |
| 4 | pHAT-102 | bulk_pH | 0.0020 → 0.0011 | 0.0009 | 0.45 | redundancy |
| 5 | TT-102 | catholyte_temperature | 0.0571 → 0.0465 | 0.0106 | 0.19 | redundancy |

**Physical rationale.**

- **THK-101 deposit-thickness sensor (ultrasound / 2-beam profilometer /
  coulometric).** Resolves the only *divergent* state. Deposit thickness is the
  primary product state and its current estimate is pure open-loop integration
  with unbounded bias — unacceptable for any thickness-based quality gate. A
  direct observation collapses the deposit covariance to the sensor floor
  (σ ≈ 0.01 µm) and removes the divergence. **Highest priority.**
- **CVT-201 cell-voltage reconciliation.** State 6 is now *weakly observable*
  through the coupled dynamics (its error is already small and bounded σ ≈ 0.08 V),
  so a direct observation does **not** change the Gramian rank — it removes the
  residual weakness and the reliance on the dynamics-propagation alone.
  Physically this is not a new instrument — VT-201 already exists — but the
  measurement model should reconcile `x[6]` against it (see §4).
- **FE2P-101 inline Fe²⁺ probe.** Does not change the rank (bulk_fe2 is already
  observable), but removes the reliance on the `v_cell` coupling and tightens its
  conditioning towards a directly-observed floor. Useful for bath-management and
  the Faraday-depletion model, not for observability.
- **pHAT-102 surface-pH probe, TT-102 temperature redundancy.** Redundancies on
  already directly-observed states (3 and 0). Low marginal information gain;
  they improve fault tolerance / cross-checks but are **not** needed for
  observability.

---

## 4. Recommended minimum sensor set for the L1 data record

To make **all 7 states observable**, add:

1. **THK-101 — deposit-thickness sensor** (ultrasound / 2-beam profilometer /
   coulometric), direct observation of state 5.
2. **CVT-201 — cell-voltage reconciliation**, direct observation of state 6
   (align the existing VT-201 reading to the `cell_voltage` state).
3. **FE2P-101 — inline Fe²⁺ probe**, direct observation of state 2, to bring
   the weakly-conditioned bulk-Fe²⁺ estimate in line with the other states.

With this set the Gramian is **full rank (7) at every tested operating point**
(smallest singular value ≈ 1.0):

| operating_point | rank | smallest σ | full observable |
|-----------------|-----:|-----------:|:---------------:|
| nominal | 7 | 1.00e+00 | ✓ |
| lo_j_lo_T_lo_fe2 | 7 | 1.00e+00 | ✓ |
| hi_j_hi_T_hi_fe2 | 7 | 1.00e+00 | ✓ |
| lo_j_hi_T_hi_fe2 | 7 | 1.00e+00 | ✓ |
| hi_j_lo_T_lo_fe2 | 7 | 1.01e+00 | ✓ |

> **Strict minimum for observability.** Under the PR #29 coupled dynamics,
> `cell_voltage` is already weakly observable, so **rank 7 is achieved with a
> single additional sensor, `{THK-101}`** (the only remaining divergent state is
> `deposit_thickness`). The recommended set adds CVT-201 and FE2P-101 not to
> change the rank but to fix residual conditioning — cell_voltage (weak via
> dynamics propagation) and bulk_fe2 (thin via the `v_cell` coupling). We
> therefore recommend all three for the L1 data record.

**Measurement-model note (flagged explicitly).** The brief permits a minimal
`h_obs` change if a structural unobservability finding requires it. Under the
PR #29 dynamics no state is *structurally* unobservable except `deposit_thickness`
(no output equation references it — a genuine new-instrument gap). `cell_voltage`
is weakly observable via the coupled dynamics rather than structurally blind, and
we have **not** changed `h_obs` in this analysis (doing so would alter the
`v_cell` coupling used for bulk-Fe²⁺ inference and change existing twin
behaviour). The CVT-201 recommendation encodes the minimal forward change for
L1: reconcile the existing cell-voltage sensor against the `cell_voltage` state
while retaining the physics coupling used for bulk-Fe²⁺ inference.

---

## 5. Out of scope / deferred

- No real-data L1 calibration, no EKF retuning, no storm/shutdown behaviour, no
  crate/site layers, no new physics.
- `operating_twin` / `crate` are untouched and remain green.
- Remains L0: this is epistemic due-diligence before the first reference-cell
  run and feeds the L1 sensor spec.
