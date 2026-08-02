# G0 co-location coverage contract — in-silico observability proof

**Tier:** L0 / in-silico. **NOT gate evidence.**

This dry-run proves, before any L1 hardware is bought, that the **full co-located suite** — the base 5-sensor suite plus the wired L1 sensors (THK-101, CVT-201, FE2P-101) — reconstructs all 7 EKF states at every representative operating point, and that every state's estimation-error covariance stays bounded over a 24 h run.

- **Tags evaluated (real wired measurement model, `h_obs`/`H_jacobian`):** TT-101, TT-201, pHAT-101, CT-201, VT-201, THK-101, CVT-201, FE2P-101
- **Operating points evaluated:** 5 (nominal, lo_j_lo_T_lo_fe2, hi_j_hi_T_hi_fe2, lo_j_hi_T_hi_fe2, hi_j_lo_T_lo_fe2)

## 1. Rank comparison — base suite vs full co-located suite

| operating point | base rank | full rank | sv_min (full) | cond (full) |
|-----------------|----------:|----------:|--------------:|------------:|
| nominal | 6 | **7** | 1.003e+00 | 3.863e+02 |
| lo_j_lo_T_lo_fe2 | 6 | **7** | 1.003e+00 | 3.856e+02 |
| hi_j_hi_T_hi_fe2 | 6 | **7** | 1.004e+00 | 3.910e+02 |
| lo_j_hi_T_hi_fe2 | 6 | **7** | 1.003e+00 | 3.874e+02 |
| hi_j_lo_T_lo_fe2 | 6 | **7** | 1.009e+00 | 8.862e+04 |

The base 5-sensor suite is rank-6 (deposit_thickness unobservable + divergent). Adding the L1 sensors raises the Gramian to **full rank 7** at every point.

## 2. Covariance stability at each operating point (full suite)

Per-state end-of-run estimation-error 1-sigma and a stability flag (covariance bounded / non-divergent over a 24 h Riccati recursion).

| operating point | catholyte_temperature | anolyte_temperature | bulk_fe2 | bulk_pH | current_density | deposit_thickness | cell_voltage | cov stable |
|---|---|---|---|---|---|---|---|---|
| nominal | 0.238 / 0.291 / 0.017 / 0.045 / 0.384 / 0.342 / 0.010 | YES |
| lo_j_lo_T_lo_fe2 | 0.255 / 0.291 / 0.017 / 0.045 / 0.322 / 0.341 / 0.010 | YES |
| hi_j_hi_T_hi_fe2 | 0.200 / 0.291 / 0.017 / 0.045 / 0.420 / 0.342 / 0.010 | YES |
| lo_j_hi_T_hi_fe2 | 0.263 / 0.291 / 0.017 / 0.045 / 0.390 / 0.342 / 0.010 | YES |
| hi_j_lo_T_lo_fe2 | 0.225 / 0.291 / 0.011 / 0.045 / 0.436 / 0.365 / 0.010 | YES |

Stability icon: `sigma` is the final 1-sigma in state units.

## 3. Contract verdict

- **Full rank (7) at every operating point:** PASS
- **Covariance stable at every operating point:** PASS

**Verdict: the G0 co-location coverage contract HOLDS in-silico.** The full suite, co-located at the cell, covers all 7 states at every tested operating point with bounded estimation covariance.

## 4. Method & reuse
- Reuses `models/digital_twin.py` numerical Jacobians (`_F_jacobian`, `H_jacobian`) and the `models/observability.py` Gramian / Riccati machinery.
- L1 tags are the **wired** observations (`digital_twin.L1_SENSOR_OBS_MAP`), driven through `h_obs` with the non-negativity clamps and the VT-201 physics coupling — the EKF consumption path, not abstract unit rows.

*L0/in-silico only: this proves capability (full observability is possible), not real instrument performance. No hardware purchase decision is gated on this analysis alone.*