# D6 — Wire the operating twin closed-loop to a calibrated + observed cell

**D6 autonomy-design deliverable · L5 live-control target of the sled**
**Pairs with** `models/operating_twin.py` (safety/supervisory), `models/kinetics.py`
(calibration), `models/g0_co_location.py` (observability), `models/digital_twin.py`.
**Machine-readable result:** `aq-steel-closed-loop-twin` / `python -m models.run_closed_loop_twin`
→ `outputs/closed_loop_live_control_target.json`, `docs/figures/closed_loop_live_control_target.png`.

---

## 0. What this deliverable answers

The sled's L5 ladder (see `docs/NEXT_STEPS.md` → *Model credibility ladder*)
requires that, before any live-control claim, the cell be **both**:

* **calibrated** — the kinetics (Fe vs HER exchange-currents, boundary layer)
  were fit from *real* RDE/LSV + gas/deposit data (`kinetics_fit_pipeline`,
  `calibration`), not screening defaults; and
* **observed** — the sensor suite provably reconstructs the full 7-state EKF
  vector — the **G0 co-location coverage contract** HOLDS (full-rank Gramian +
  bounded covariance at every operating point).

D6 defines the *connection* between the safety/supervisory operating twin
(`models/operating_twin.py`) and such a cell.  The deliverable is a
**fail-closed qualification gate** plus an **L5 live-control target**
(`ClosedLoopLiveControlTwin.live_control_target()`).

> **This is an autonomy-design / wiring contract — it does NOT claim live
> control has been achieved.** Every result here is an in-silico dry run.
> Real Q1–Q5 evidence and the D1/D2/D3 calibration-observability basis are
> still **pending** on the board and are recorded as such in the target.

---

## 1. The fail-closed qualification gate

A cell qualifies for L5-armed actuation only when **all three** conditions
hold simultaneously:

1. **Calibration pillar** — we hold *fitted* kinetics (`CalibrationEvidence`,
   `source != "none"`).
2. **Observability pillar** — the G0 co-location contract fully holds
   (`ObservabilityEvidence.qualified`: full-rank Gramian == N_STATES == 7 AND
   bounded covariance AND no violations).
3. **Bounded-actuation envelope** — the calibrated kinetics sustain a current
   density at/above the current-efficiency floor (default `0.80`) somewhere
   inside the operating window, so a **non-zero bounded command exists**
   (`bounded_viable_current_density_mA_cm2 > 0`).

`CellQualification.reasons()` is the **single source of truth**:

```
qualified == (len(reasons()) == 0)
```

Removing any pillar — or supplying a calibration whose kinetics cannot hold
the CE floor — keeps the twin **ADVISORY** and rejects arming.  This is
deliberately fail-closed: a "qualified" cell that can never emit a real
bounded command would be a dishonest claim, so the CE-floor feasibility is
part of the gate, not a side note.

### Failure modes exercised in tests

| Case | calibration | observability | envelope | gate |
|---|---|---|---|---|
| No calibration | ✗ | — | — | blocked (`calibration_missing_or_pending`) |
| No observability | ✓ | ✗ | — | blocked (`observability_contract_not_holding`) |
| Poor cell (HER dominates) | ✓ | ✓ | ✗ | blocked (`no_viable_bounded_actuation...`) |
| Fully qualified | ✓ | ✓ | ✓ | **armed** |

---

## 2. The closed loop

`ClosedLoopLiveControlTwin.step()` closes the loop in-silico:

1. an observed sensor snapshot (`SensorSnapshot`) → `OperatingTwin.update()` —
   safety evaluation (trips, stale/bad sensors, limits);
2. the calibrated kinetics → the FE / deposit rate that bound actuation;
3. a **bounded command**, emitted ONLY when the gate is qualified.

The commanded current is the operating twin's bounded/ramped request, capped
by `CellQualification.bounded_target_current_A(area)` — the largest current
that keeps calibrated FE at/above the floor.  Arming runs through both the
fail-closed gate **and** the operating twin's own cell-id token guard
(`arm_actuation(token)`), so even a qualified twin cannot actuate without the
exact configured cell identity.  A hard trip (e.g. over-current) latches and
zeroes the command regardless of qualification.

---

## 3. The L5 live-control target (machine-readable)

`ClosedLoopLiveControlTwin.live_control_target()` emits:

```json
{
  "deliverable": "D6",
  "level": "L5",
  "title": "Wire operating_twin closed-loop to a calibrated + observed cell",
  "cell_id": "RC-1",
  "fail_closed_gate": "require calibration AND observability before arming",
  "qualification": { "qualified": true, "observability_rank": 7, ... },
  "observed_suite": ["TT-101","TT-201","pHAT-101","CT-201","VT-201","THK-101","CVT-201","FE2P-101"],
  "evidence_gates": {
    "q1_mass_charge_energy_ledgers": "pending",
    "q3_rde_volumetric_h2": "pending",
    "q4_membrane_anode_viability": "pending",
    "q5_adhesion_release_stress_h2": "pending",
    "d1_immutable_reference_spec": "pending",
    "d2_d3_calibration_and_observability_basis": "pending"
  }
}
```

The `evidence_gates` block is the honest ledger of what must land before
hardware actuation is armed — every real evidence gate is transparently
`"pending"`.

---

## 4. Dry-run result

`python -m models.run_closed_loop_twin` (uses a synthetic high-CE kinetics
stand-in, `source="real_data"`, until real data lands):

| Quantity | Value |
|---|---|
| Fail-closed without calibration | ✓ |
| Fail-closed without observability | ✓ |
| Qualified with both pillars + envelope | ✓ |
| Observability rank (G0) | 7 / 7 |
| Closed-loop final command current | 1.1 A (ramped, below 3.0 A bound) |
| Calibrated FE at operating density | 0.87 |
| Bounded target current (10 cm²) | 3.0 A |
| Trip on over-current, even when qualified | ✓ |

---

## 5. Files

* `models/closed_loop_twin.py` — the gate + closed-loop twin (additive, L0).
* `models/run_closed_loop_twin.py` — driver → `outputs/...json` + figure.
* `tests/test_closed_loop_twin.py` — 14 tests covering every gate path.
* `pyproject.toml` — `aq-steel-closed-loop-twin` console script.
