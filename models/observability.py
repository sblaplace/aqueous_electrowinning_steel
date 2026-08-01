"""
Observability & sensor-placement analysis for the digital-twin EKF.

This is a *screening* (L0) analysis-only module.  Before the twin consumes the
reference cell's first real run, we must prove the proposed sensor suite is
capable of reconstructing the full 7-state EKF state vector.  It builds the
linearized pair (``F``, ``H``) from the existing numerical Jacobians in
``digital_twin`` and answers three questions deterministically:

1. Is the current 5-sensor suite observable?  (Gramian rank + conditioning.)
2. Which states are observable / weakly observable / unobservable?
3. Which *candidate* sensors give the largest marginal information gain, and
   what is the minimum set that restores full observability of all 7 states?

No real data are used and nothing is calibrated — the twin's EKF dynamics and
measurement model (``h_obs``) are left untouched.  The analysis is exercised at
representative operating points swept across the physics surrogate grid
(``CellProcessModel``) so the conclusion holds at every tested point.

Theory notes
------------
For the linear time-invariant system ``x_{k+1} = F x_k``, ``y_k = H x_k`` the
finite-horizon observability matrix is ``O_N = [H; H F; ...; H F^{N-1}]`` and
the observability Gramian is ``W_N = O_N^T O_N``.  ``rank(W_N) == 7`` means the
full state is reconstructible from the outputs over the horizon.  A state is
*structurally unobservable* when its column of ``O_N`` is identically zero (it
never reaches any output).  Whether its error is *bounded* depends on the
stability of the unobservable mode (detectability), which we read off the
finite-horizon Kalman covariance recursion: a divergent (pure-integrator)
mode has unbounded estimation error, a stable (contractive) mode does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .digital_twin import (
    N_STATES,
    STATE_KEYS,
    OBSERVABLE_TAGS,
    _F_jacobian,
    H_jacobian,
    _R_observation,
    _DEFAULT_DP,
    get_default_process_model,
)
from .twin_physics import CellProcessModel

# ---------------------------------------------------------------------------
# Defaults (deterministic; match the EKF in digital_twin)
# ---------------------------------------------------------------------------

# EKF default process-noise *variances* (already squared) used by ExtendedKalmanFilter.
DEFAULT_Q_VAR = np.array([0.1, 0.1, 0.001, 0.01, 1.0, 0.1, 0.005])
# EKF default initial covariance (DigitalTwin.P0).
DEFAULT_P0 = np.diag([1.0, 1.0, 0.01, 0.1, 10.0, 1.0, 0.05])

# Time step used by the linearization / covariance recursion (hours).
DT_HR = 0.1
# Observability horizon spans a few dynamics time constants of the slowest mode
# (~4 h pH loop, ~2 h temperature, ~1 h voltage); 12 h @ 0.1 h = 120 steps.
HORIZON_HR = 12.0
# Run length for the finite-horizon estimation-error covariance (24 h).
RUN_LENGTH_HR = 24.0

# Numerical tolerances / thresholds.
RANK_TOL = 1e-8
# A structurally-observable state is "weak" when its estimation-error sigma is
# more than this many times the direct-measurement floor for that state.
WEAK_SIGMA_RATIO = 3.0
# Relative growth (over the last quarter of the run) above which an unobserved
# state's covariance is classified as divergent / unbounded.
_DIVERGENCE_SLOPE_TOL = 1e-8


# ---------------------------------------------------------------------------
# Candidate (additional) sensors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateSensor:
    """A candidate additional observation for sensor-placement ranking.

    Each candidate is modelled as a *direct* observation of a single target
    state, so its measurement Jacobian row is the canonical unit vector
    ``e_{target_state}``.  The measurement model for the current suite stays
    exactly as ``h_obs`` in ``digital_twin``.
    """

    tag: str
    quantity: str
    unit: str
    target_state: int
    noise_std: float
    physical_min: float
    physical_max: float
    rationale: str = ""

    def row(self) -> np.ndarray:
        r = np.zeros(N_STATES)
        r[self.target_state] = 1.0
        return r

    def measurement_noise_variance(self) -> float:
        return float(self.noise_std**2)


# Direct-observation floor used to judge "weak" vs "strong" conditioning, in
# the units of each state.  These are the 1-sigma measurement noises of the
# direct sensor that would observe that state (existing sensors for 0,1,3,4,6;
# candidate probes for 2 and 5).
_DIRECT_OBS_NOISE: Dict[int, float] = {
    0: 0.5,  # TT-101 catholyte temperature (C)
    1: 0.5,  # TT-201 anolyte temperature (C)
    2: 0.02,  # inline Fe2+ probe (M) — candidate
    3: 0.05,  # pHAT-101 bulk pH (pH)
    4: 0.5,  # CT-201 current density, 5 A / (area*10) = 0.5 mA/cm2
    5: 0.5,  # THK-101 deposit thickness (um) — candidate
    6: 0.01,  # VT-201 cell voltage (V)
}

FE2_PROBE = CandidateSensor(
    "FE2P-101",
    "bulk_fe2",
    "M",
    2,
    0.02,
    0.0,
    2.0,
    "Inline Fe2+ probe (direct observation of bulk_fe2)",
)
THICKNESS_SENSOR = CandidateSensor(
    "THK-101",
    "deposit_thickness",
    "um",
    5,
    0.5,
    0.0,
    500.0,
    "Ultrasound / 2-beam profilometer / coulometric deposit-thickness sensor "
    "(direct observation of deposit_thickness)",
)
CELL_VOLTAGE_DIRECT = CandidateSensor(
    "CVT-201",
    "cell_voltage",
    "V",
    6,
    0.01,
    0.0,
    10.0,
    "Reconcile the existing VT-201 cell-voltage reading to the cell_voltage "
    "state (adds the direct e_6 observation the current h_obs omits)",
)
SURFACE_PH_PROBE = CandidateSensor(
    "pHAT-102",
    "bulk_pH",
    "pH",
    3,
    0.05,
    0.0,
    14.0,
    "Surface-pH probe (redundancy / tighter bound on bulk_pH)",
)
CATHOLYTE_TEMP_REDUNDANCY = CandidateSensor(
    "TT-102",
    "catholyte_temperature",
    "C",
    0,
    0.5,
    20.0,
    95.0,
    "Catholyte-loop temperature redundancy (index 0)",
)

CANDIDATE_SENSORS: List[CandidateSensor] = [
    FE2_PROBE,
    THICKNESS_SENSOR,
    CELL_VOLTAGE_DIRECT,
    SURFACE_PH_PROBE,
    CATHOLYTE_TEMP_REDUNDANCY,
]

# Minimum set (by count) that restores full observability of all 7 states.
MINIMUM_SET_FOR_FULL_OBSERVABILITY: Tuple[CandidateSensor, ...] = (
    THICKNESS_SENSOR,
    CELL_VOLTAGE_DIRECT,
)

# Recommended set: the minimum set plus the inline Fe2+ probe, which does not
# change the rank but materially improves the (weak) conditioning of bulk_fe2.
RECOMMENDED_MINIMUM_SENSORS: Tuple[CandidateSensor, ...] = (
    FE2_PROBE,
    THICKNESS_SENSOR,
    CELL_VOLTAGE_DIRECT,
)


# ---------------------------------------------------------------------------
# Linear system construction (reusing the twin's numerical Jacobians)
# ---------------------------------------------------------------------------


def build_linearized_system(
    x: np.ndarray,
    dt_hr: float = DT_HR,
    obs_tags: Optional[Sequence[str]] = None,
    model: Optional[CellProcessModel] = None,
    design_point: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the linearized pair (F, H) from the twin's numerical Jacobians."""
    if obs_tags is None:
        obs_tags = OBSERVABLE_TAGS
    if model is None:
        model = get_default_process_model()
    F = _F_jacobian(x, dt_hr, model, design_point)
    H = H_jacobian(x, list(obs_tags), model, design_point)
    return F, H


def state_vector_from_operating_point(
    operating_point: Tuple[str, float, float, float],
    model: CellProcessModel,
    deposit_um: float = 50.0,
    bulk_pH: float = 3.5,
) -> np.ndarray:
    """Build a physically-consistent 7-state vector from an operating point.

    ``operating_point`` is ``(name, j_mA_cm2, temperature_C, fe2_M)``.
    """
    _name, j, T, fe2 = operating_point
    v_cell = model.predict(j, T, fe2).v_cell_V
    return np.array([T, T + 1.5, fe2, bulk_pH, j, deposit_um, v_cell])


def operating_points_from_model(
    model: CellProcessModel,
) -> List[Tuple[str, float, float, float]]:
    """Representative operating points: nominal + corners of the physics grid."""
    jg, Tg, fg = model.j_grid, model.T_grid, model.fe2_grid
    nom = model.nominal
    pts = [
        ("nominal", nom["j_avg_mA_cm2"], nom["temperature_C"], nom["fe2_M"]),
        ("lo_j_lo_T_lo_fe2", float(min(jg)), float(min(Tg)), float(min(fg))),
        ("hi_j_hi_T_hi_fe2", float(max(jg)), float(max(Tg)), float(max(fg))),
        ("lo_j_hi_T_hi_fe2", float(min(jg)), float(max(Tg)), float(max(fg))),
        ("hi_j_lo_T_lo_fe2", float(max(jg)), float(min(Tg)), float(min(fg))),
    ]
    return pts


# ---------------------------------------------------------------------------
# Observability matrices and Gramian
# ---------------------------------------------------------------------------


def observability_matrix(F: np.ndarray, H: np.ndarray, horizon_steps: int) -> np.ndarray:
    """Finite-horizon observability matrix O_N = [H; H F; ...; H F^{N-1}]."""
    blocks = [H]
    for _ in range(1, horizon_steps):
        blocks.append(blocks[-1] @ F)
    return np.vstack(blocks)


def observability_gramian(F: np.ndarray, H: np.ndarray, horizon_steps: int) -> np.ndarray:
    """Finite-horizon observability Gramian W_N = O_N^T O_N."""
    Om = observability_matrix(F, H, horizon_steps)
    return Om.T @ Om


# ---------------------------------------------------------------------------
# Estimation-error covariance recursion (finite-horizon Kalman covariance)
# ---------------------------------------------------------------------------


def estimation_covariance_trajectory(
    F: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    steps: int,
    Q_var: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Run the linear Kalman covariance recursion for ``steps`` updates.

    Returns an ``(steps + 1, N_STATES)`` array of the per-step diagonal of the
    estimation-error covariance P.  Because we only need the covariance (not a
    specific measurement sequence), this is the standard Riccati recursion:

        P_pred  = F P F^T + Q
        K       = P_pred H^T (H P_pred H^T + R)^{-1}
        P_new   = (I - K H) P_pred

    For structurally-unobservable modes the variance either stays bounded
    (detectable / stable mode) or grows without bound (pure integrator).
    """
    n = F.shape[0]
    Q_var = DEFAULT_Q_VAR if Q_var is None else np.asarray(Q_var, dtype=float)
    P0 = DEFAULT_P0 if P0 is None else np.asarray(P0, dtype=float)
    Q = np.diag(Q_var)
    P = P0.copy()
    diags = [P.diagonal().copy()]
    for _ in range(steps):
        Pp = F @ P @ F.T + Q
        S = H @ Pp @ H.T + R
        K = Pp @ H.T @ np.linalg.inv(S)
        P = (np.eye(n) - K @ H) @ Pp
        diags.append(P.diagonal().copy())
    return np.asarray(diags)


def _is_divergent_tail(var_traj: np.ndarray, frac: float = 0.25) -> bool:
    """True when the state's variance is still growing at the end of the run."""
    n = len(var_traj)
    k = max(1, int(n * frac))
    if var_traj[-1] < _DIVERGENCE_SLOPE_TOL:
        return False
    return bool(var_traj[-1] - var_traj[-k - 1] > _DIVERGENCE_SLOPE_TOL * var_traj[-1])


# ---------------------------------------------------------------------------
# Per-state observability analysis
# ---------------------------------------------------------------------------


@dataclass
class ObservabilityResult:
    """Results of the observability analysis at one operating point / sensor set."""

    operating_point: str
    tags: List[str]
    x: np.ndarray
    F: np.ndarray
    H: np.ndarray
    horizon_steps: int
    run_length_steps: int
    observability_matrix: np.ndarray
    gramian: np.ndarray
    singular_values: np.ndarray
    rank: int
    smallest_singular_value: float
    smallest_nonzero_singular_value: float
    condition_number: float
    per_state_score: np.ndarray  # structural observability score in [0,1]
    per_state_rel_energy: np.ndarray  # normalized Gramian column energy
    per_state_sigma: np.ndarray  # end-of-run estimation-error 1-sigma
    per_state_flag: Dict[str, str]  # state key -> flag string

    def summary_table(self) -> str:
        rows = []
        for i, key in enumerate(STATE_KEYS):
            rows.append(
                f"{key:22s} {self.per_state_flag[key]:24s} "
                f"score={self.per_state_score[i]:.3f} "
                f"rel_energy={self.per_state_rel_energy[i]:.4f} "
                f"sigma={self.per_state_sigma[i]:.4f}"
            )
        return "\n".join(rows)


def analyze_observability(
    x: np.ndarray,
    dt_hr: float = DT_HR,
    obs_tags: Optional[Sequence[str]] = None,
    model: Optional[CellProcessModel] = None,
    design_point: Optional[Dict[str, float]] = None,
    horizon_steps: Optional[int] = None,
    run_length_steps: Optional[int] = None,
    Q_var: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
    operating_point: str = "unnamed",
) -> ObservabilityResult:
    """Full observability + per-state conditioning analysis of a sensor set."""
    if obs_tags is None:
        obs_tags = list(OBSERVABLE_TAGS)
    if model is None:
        model = get_default_process_model()
    if design_point is None:
        design_point = dict(_DEFAULT_DP)
    horizon_steps = horizon_steps or max(1, int(round(HORIZON_HR / dt_hr)))
    run_length_steps = run_length_steps or max(1, int(round(RUN_LENGTH_HR / dt_hr)))

    F, H = build_linearized_system(x, dt_hr, list(obs_tags), model, design_point)
    Om = observability_matrix(F, H, horizon_steps)
    W = Om.T @ Om
    s = np.linalg.svd(W, compute_uv=False)
    rank = int(np.linalg.matrix_rank(W, tol=RANK_TOL))
    nz = np.array([v for v in s if v > RANK_TOL])
    s_min_nz = float(nz[-1]) if len(nz) else 0.0
    cond = float(s[0] / s_min_nz) if s_min_nz > 0 else float("inf")

    # Per-state structural observability score: observable-component fraction.
    P_null = np.eye(N_STATES) - np.linalg.pinv(Om) @ Om
    per_state_score = 1.0 - np.array([np.linalg.norm(P_null[:, i]) for i in range(N_STATES)])

    # Normalized Gramian column energy (relative "how well" each state is seen).
    col_energy = np.array([np.linalg.norm(Om[:, i]) for i in range(N_STATES)])
    cmax = float(col_energy.max()) if col_energy.max() > 0 else 1.0
    per_state_rel_energy = col_energy / cmax

    # Estimation-error covariance (for conditioning / divergence detection).
    R = _R_observation(list(obs_tags))
    traj = estimation_covariance_trajectory(F, H, R, run_length_steps, Q_var=Q_var, P0=P0)
    final_var = traj[-1]
    per_state_sigma = np.sqrt(np.maximum(final_var, 0.0))

    # Per-state flags.
    flags: Dict[str, str] = {}
    for i, key in enumerate(STATE_KEYS):
        if col_energy[i] <= RANK_TOL:
            # Structurally unobservable. Divergence depends on detectability.
            if _is_divergent_tail(traj[:, i]):
                flags[key] = "unobservable_divergent"
            else:
                flags[key] = "unobservable_bounded"
        else:
            noise = _DIRECT_OBS_NOISE.get(i, 1.0)
            ratio = float(per_state_sigma[i] / noise) if noise > 0 else 1.0
            if ratio > WEAK_SIGMA_RATIO:
                flags[key] = "weak"
            else:
                flags[key] = "observable"

    return ObservabilityResult(
        operating_point=operating_point,
        tags=list(obs_tags),
        x=np.asarray(x, dtype=float),
        F=F,
        H=H,
        horizon_steps=horizon_steps,
        run_length_steps=run_length_steps,
        observability_matrix=Om,
        gramian=W,
        singular_values=s,
        rank=rank,
        smallest_singular_value=float(s[-1]),
        smallest_nonzero_singular_value=s_min_nz,
        condition_number=cond,
        per_state_score=per_state_score,
        per_state_rel_energy=per_state_rel_energy,
        per_state_sigma=per_state_sigma,
        per_state_flag=flags,
    )


# ---------------------------------------------------------------------------
# Augmented (current + candidate) sensor sets
# ---------------------------------------------------------------------------


def augmented_H_and_R(
    F: np.ndarray,
    H: np.ndarray,
    obs_tags: Sequence[str],
    candidates: Sequence[CandidateSensor],
) -> Tuple[np.ndarray, np.ndarray]:
    """Append candidate direct-observation rows to (H, R).

    The current H rows (from ``h_obs``) are kept verbatim; each candidate adds a
    canonical unit row ``e_{target}`` and its measurement-noise variance.
    """
    rows = [H]
    rdiag = list(np.diag(_R_observation(list(obs_tags))))
    for c in candidates:
        rows.append(c.row())
        rdiag.append(c.measurement_noise_variance())
    return np.vstack(rows), np.diag(rdiag)


def analyze_sensor_set(
    x: np.ndarray,
    obs_tags: Sequence[str],
    candidates: Sequence[CandidateSensor],
    dt_hr: float = DT_HR,
    model: Optional[CellProcessModel] = None,
    design_point: Optional[Dict[str, float]] = None,
    horizon_steps: Optional[int] = None,
    run_length_steps: Optional[int] = None,
    Q_var: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
    operating_point: str = "unnamed",
) -> ObservabilityResult:
    """Observability of the current suite plus a set of candidate sensors."""
    if model is None:
        model = get_default_process_model()
    if design_point is None:
        design_point = dict(_DEFAULT_DP)
    horizon_steps = horizon_steps or max(1, int(round(HORIZON_HR / dt_hr)))
    run_length_steps = run_length_steps or max(1, int(round(RUN_LENGTH_HR / dt_hr)))
    F, H = build_linearized_system(x, dt_hr, list(obs_tags), model, design_point)
    H_aug, R_aug = augmented_H_and_R(F, H, list(obs_tags), candidates)

    # Reuse analyze_observability by injecting the augmented (F, H) via a custom
    # path: build the result manually so H reflects the augmented rows.
    Om = observability_matrix(F, H_aug, horizon_steps)
    W = Om.T @ Om
    s = np.linalg.svd(W, compute_uv=False)
    rank = int(np.linalg.matrix_rank(W, tol=RANK_TOL))
    nz = np.array([v for v in s if v > RANK_TOL])
    s_min_nz = float(nz[-1]) if len(nz) else 0.0
    cond = float(s[0] / s_min_nz) if s_min_nz > 0 else float("inf")

    P_null = np.eye(N_STATES) - np.linalg.pinv(Om) @ Om
    per_state_score = 1.0 - np.array([np.linalg.norm(P_null[:, i]) for i in range(N_STATES)])
    col_energy = np.array([np.linalg.norm(Om[:, i]) for i in range(N_STATES)])
    cmax = float(col_energy.max()) if col_energy.max() > 0 else 1.0
    rel = col_energy / cmax

    traj = estimation_covariance_trajectory(F, H_aug, R_aug, run_length_steps, Q_var=Q_var, P0=P0)
    final_var = traj[-1]
    per_state_sigma = np.sqrt(np.maximum(final_var, 0.0))

    flags: Dict[str, str] = {}
    for i, key in enumerate(STATE_KEYS):
        if col_energy[i] <= RANK_TOL:
            flags[key] = (
                "unobservable_divergent"
                if _is_divergent_tail(traj[:, i])
                else "unobservable_bounded"
            )
        else:
            noise = _DIRECT_OBS_NOISE.get(i, 1.0)
            ratio = float(per_state_sigma[i] / noise) if noise > 0 else 1.0
            flags[key] = "weak" if ratio > WEAK_SIGMA_RATIO else "observable"

    tags = list(obs_tags) + [c.tag for c in candidates]
    return ObservabilityResult(
        operating_point=operating_point,
        tags=tags,
        x=np.asarray(x, dtype=float),
        F=F,
        H=H_aug,
        horizon_steps=horizon_steps,
        run_length_steps=run_length_steps,
        observability_matrix=Om,
        gramian=W,
        singular_values=s,
        rank=rank,
        smallest_singular_value=float(s[-1]),
        smallest_nonzero_singular_value=s_min_nz,
        condition_number=cond,
        per_state_score=per_state_score,
        per_state_rel_energy=rel,
        per_state_sigma=per_state_sigma,
        per_state_flag=flags,
    )


# ---------------------------------------------------------------------------
# Sensor-placement ranking
# ---------------------------------------------------------------------------


@dataclass
class SensorRanking:
    """Marginal information gain of one candidate sensor on its target state."""

    sensor: CandidateSensor
    target_state: int
    target_key: str
    variance_before: float
    variance_after: float
    variance_reduction: float
    relative_reduction: float
    resolves_unobservability: bool
    divergent_before: bool  # target state's error was unbounded (divergent)
    rank_before: int
    rank_after: int


def rank_candidate_sensors(
    x: np.ndarray,
    obs_tags: Optional[Sequence[str]] = None,
    candidates: Optional[Sequence[CandidateSensor]] = None,
    dt_hr: float = DT_HR,
    model: Optional[CellProcessModel] = None,
    design_point: Optional[Dict[str, float]] = None,
    run_length_steps: Optional[int] = None,
    Q_var: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
    operating_point: str = "unnamed",
) -> List[SensorRanking]:
    """Rank candidate sensors by marginal reduction in target-state covariance.

    For each candidate, the target state's end-of-run estimation-error variance
    is computed with and without the candidate; the larger the reduction the
    more information the sensor adds.  Sensors that resolve a structurally
    unobservable state (deposit / cell_voltage) naturally rank highest.
    """
    if obs_tags is None:
        obs_tags = list(OBSERVABLE_TAGS)
    if candidates is None:
        candidates = CANDIDATE_SENSORS
    if model is None:
        model = get_default_process_model()
    if design_point is None:
        design_point = dict(_DEFAULT_DP)
    run_length_steps = run_length_steps or max(1, int(round(RUN_LENGTH_HR / dt_hr)))

    F, H = build_linearized_system(x, dt_hr, list(obs_tags), model, design_point)
    base = analyze_sensor_set(
        x,
        list(obs_tags),
        [],
        dt_hr=dt_hr,
        model=model,
        design_point=design_point,
        run_length_steps=run_length_steps,
        Q_var=Q_var,
        P0=P0,
        operating_point=operating_point,
    )
    # End-of-run estimation-error variances for the current suite.
    traj = estimation_covariance_trajectory(
        F,
        H,
        _R_observation(list(obs_tags)),
        run_length_steps,
        Q_var=Q_var,
        P0=P0,
    )
    base_var = traj[-1]

    results: List[SensorRanking] = []
    for c in candidates:
        aug = analyze_sensor_set(
            x,
            list(obs_tags),
            [c],
            dt_hr=dt_hr,
            model=model,
            design_point=design_point,
            run_length_steps=run_length_steps,
            Q_var=Q_var,
            P0=P0,
            operating_point=operating_point,
        )
        i = c.target_state
        var_before = float(base_var[i])
        var_after = float(aug.per_state_sigma[i] ** 2)
        reduction = var_before - var_after
        rel = reduction / max(var_before, 1e-12)
        base_flag = base.per_state_flag[STATE_KEYS[i]]
        resolves = bool(
            base_flag.startswith("unobservable")
            and aug.per_state_flag[STATE_KEYS[i]].startswith("observable")
        )
        results.append(
            SensorRanking(
                sensor=c,
                target_state=i,
                target_key=STATE_KEYS[i],
                variance_before=var_before,
                variance_after=var_after,
                variance_reduction=reduction,
                relative_reduction=rel,
                resolves_unobservability=resolves,
                divergent_before=(base_flag == "unobservable_divergent"),
                rank_before=base.rank,
                rank_after=aug.rank,
            )
        )

    # Rank sensors that restore a structurally-unobservable state first; among
    # those, resolve divergent (unbounded) modes before bounded ones; then by
    # the fraction of the target state's uncertainty the sensor removes.  This
    # keeps the ordering meaningful across states with different physical units
    # (M, um, V, C, pH).
    results.sort(
        key=lambda r: (
            r.resolves_unobservability,
            r.divergent_before,
            r.relative_reduction,
        ),
        reverse=True,
    )
    return results


# ---------------------------------------------------------------------------
# High-level convenience entry points
# ---------------------------------------------------------------------------


def characterize_current_suite(
    model: Optional[CellProcessModel] = None,
    dt_hr: float = DT_HR,
    horizon_steps: Optional[int] = None,
    run_length_steps: Optional[int] = None,
) -> Dict[str, ObservabilityResult]:
    """Run the full analysis at every representative operating point."""
    if model is None:
        model = get_default_process_model()
    results: Dict[str, ObservabilityResult] = {}
    for name, j, T, fe2 in operating_points_from_model(model):
        x = state_vector_from_operating_point((name, j, T, fe2), model)
        results[name] = analyze_observability(
            x,
            dt_hr=dt_hr,
            model=model,
            horizon_steps=horizon_steps,
            run_length_steps=run_length_steps,
            operating_point=name,
        )
    return results


def evaluate_sensor_set_over_grid(
    sensor_set: Sequence[CandidateSensor],
    model: Optional[CellProcessModel] = None,
    dt_hr: float = DT_HR,
    horizon_steps: Optional[int] = None,
    run_length_steps: Optional[int] = None,
) -> Dict[str, ObservabilityResult]:
    """Evaluate a candidate sensor set at every representative operating point."""
    if model is None:
        model = get_default_process_model()
    results: Dict[str, ObservabilityResult] = {}
    for name, j, T, fe2 in operating_points_from_model(model):
        x = state_vector_from_operating_point((name, j, T, fe2), model)
        results[name] = analyze_sensor_set(
            x,
            list(OBSERVABLE_TAGS),
            list(sensor_set),
            dt_hr=dt_hr,
            model=model,
            horizon_steps=horizon_steps,
            run_length_steps=run_length_steps,
            operating_point=name,
        )
    return results
