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

**The current suite is rank-deficient: `rank(W_N) = 5` at every tested point.**
Only 5 of the 7 states are reconstructible.

| operating_point | rank | smallest nonzero σ | condition |
|-----------------|-----:|-------------------:|----------:|
| nominal | 5 | 6.13e-01 | 4.95e+02 |
| lo_j_lo_T_lo_fe2 | 5 | 5.35e-01 | 5.67e+02 |
| hi_j_hi_T_hi_fe2 | 5 | 1.64e+00 | 1.86e+02 |
| lo_j_hi_T_hi_fe2 | 5 | 7.36e-02 | 4.12e+03 |
| hi_j_lo_T_lo_fe2 | 5 | 1.05e+01 | 2.90e+01 |

Per-state observability at the nominal point:

| state | flag | obs. score | rel. energy | est. σ |
|-------|------|-----------:|------------:|-------:|
| catholyte_temperature | observable | 1.000 | 0.186 | 0.292 °C |
| anolyte_temperature | observable | 1.000 | 0.186 | 0.334 °C |
| bulk_fe2 | **weak** | 1.000 | 0.045 | 0.084 M |
| bulk_pH | observable | 1.000 | 0.260 | 0.046 pH |
| current_density | observable | 1.000 | 1.000 | 0.40 mA/cm² |
| deposit_thickness | **unobservable_divergent** | 0.000 | 0.000 | 5.13 µm (growing) |
| cell_voltage | **unobservable_bounded** | 0.000 | 0.000 | 0.166 V |

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

### 2.2 bulk_fe2 (state 2) — weakly observable through the v_cell coupling only

`bulk_fe2` reaches the output only through the VT-201 row, because the physics
surrogate `v_cell(j, T, Fe²⁺)` depends on Fe²⁺ (`H[VT-201, 2] ≈ −0.07 M⁻¹`).
It is therefore *observable* (score 1.000) but **weakly conditioned**: its
Gramian energy is only ~4.5 % of the best-observed state, and its end-of-run
error σ ≈ 0.084 M is ~4× the ~0.02 M an inline probe would give. Recovering a
tight bulk-Fe²⁺ estimate from voltage alone relies on an accurate, calibrated
`v_cell` surrogate.

### 2.3 cell_voltage (state 6) — unobservable but bounded (finding beyond the brief)

A second, previously-unanticipated structural gap: `H[:, 6] ≡ 0` as well.
VT-201 observes the *physics-predicted* `v_cell(j, T, Fe²⁺)` (`h_obs` returns
`pred.v_cell_V`), not the `cell_voltage` state `x[6]`, so no measurement
corrects `x[6]`. Unlike deposit, `x[6]` mean-reverts to the observed
`pred.v_cell` (`F[6,6] ≈ 0.90 < 1`), so the mode is **stable/detectable**: its
error stays bounded (σ ≈ 0.17 V) and it is **not** a divergence risk. This is a
measurement-model alignment gap, not an operational blind spot — the physical
VT-201 exists; the twin simply never reconciles `x[6]` against it.

---

## 3. Sensor-placement ranking

Candidate sensors were evaluated by marginal reduction in their target state's
end-of-run estimation-error variance, with observability-restoring sensors
ranked first (and divergent modes ahead of bounded ones).

| rank | tag | target state | var before → after | reduction | rel. | resolves? |
|-----:|-----|--------------|-------------------:|----------:|-----:|-----------|
| 1 | **THK-101** | deposit_thickness | 26.29 → 0.117 | 26.18 | 1.00 | ✓ (divergent) |
| 2 | **CVT-201** | cell_voltage | 0.0276 → 0.0001 | 0.0275 | 1.00 | ✓ (bounded) |
| 3 | **FE2P-101** | bulk_fe2 | 0.0071 → 0.0003 | 0.0068 | 0.96 | ✓ conditioning |
| 4 | pHAT-102 | bulk_pH | 0.0021 → 0.0011 | 0.0009 | 0.46 | redundancy |
| 5 | TT-102 | catholyte_temperature | 0.0853 → 0.059 | 0.026 | 0.31 | redundancy |

**Physical rationale.**

- **THK-101 deposit-thickness sensor (ultrasound / 2-beam profilometer /
  coulometric).** Resolves the only *divergent* state. Deposit thickness is the
  primary product state and its current estimate is pure open-loop integration
  with unbounded bias — unacceptable for any thickness-based quality gate. A
  direct observation collapses the deposit covariance to the sensor floor
  (σ ≈ 0.01 µm) and removes the divergence. **Highest priority.**
- **CVT-201 cell-voltage reconciliation.** Restores observability of state 6
  (rank 6 → contributes to rank 7). Physically this is not a new instrument —
  VT-201 already exists — but the measurement model must reconcile `x[6]`
  against it (see §4). Its error is bounded either way, so it is the second
  observability gap but not an urgency driver.
- **FE2P-101 inline Fe²⁺ probe.** Does not change the rank (bulk_fe2 is already
  observable), but removes the reliance on the weak `v_cell` coupling: the
  Fe²⁺ σ drops ~5× (0.084 → 0.017 M) and its conditioning becomes comparable to
  a directly-observed state. Important for bath-management and the 
  Faraday-depletion model.
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
(smallest singular value ≈ 5.5):

| operating_point | rank | smallest σ | full observable |
|-----------------|-----:|-----------:|:---------------:|
| nominal | 7 | 5.52e+00 | ✓ |
| lo_j_lo_T_lo_fe2 | 7 | 5.52e+00 | ✓ |
| hi_j_hi_T_hi_fe2 | 7 | 5.51e+00 | ✓ |
| lo_j_hi_T_hi_fe2 | 7 | 5.52e+00 | ✓ |
| hi_j_lo_T_lo_fe2 | 7 | 5.50e+00 | ✓ |

> **Strict minimum for observability.** Rank 7 is already achieved with only
> `{THK-101, CVT-201}` (two sensors); the Fe²⁺ probe does not change the rank
> because bulk_fe2 is already observable — it is recommended to fix the weak
> **conditioning** of bulk_fe2, which the voltage-only coupling leaves ~5× worse
> than a directly-observed state. We therefore recommend all three.

**Measurement-model note (flagged explicitly).** The brief permits a minimal
`h_obs` change if a structural unobservability finding requires it. We found
`cell_voltage` structurally unobservable because `h_obs` reports the
physics-predicted `v_cell` rather than state `x[6]`. We have **not** changed
`h_obs` in this analysis-only PR (doing so would sever the `v_cell` coupling
that makes bulk_fe2 observable, and would alter existing twin behaviour). The
CVT-201 recommendation encodes the minimal forward change for L1: reconcile the
existing cell-voltage sensor against the `cell_voltage` state while retaining
the physics coupling used for bulk-Fe²⁺ inference.

---

## 5. Out of scope / deferred

- No real-data L1 calibration, no EKF retuning, no storm/shutdown behaviour, no
  crate/site layers, no new physics.
- `operating_twin` / `crate` are untouched and remain green.
- Remains L0: this is epistemic due-diligence before the first reference-cell
  run and feeds the L1 sensor spec.
