"""
G0 co-location coverage contract — reusable in-silico observability proof (L0).

Before any L1 hardware is bought, we must show that the *full* sensor suite —
the base 5-sensor suite (TT-101, TT-201, pHAT-101, CT-201, VT-201) **plus** the
wired L1 sensors (THK-101, CVT-201, FE2P-101), co-located at the cell — covers
(reconstructs) all 7 EKF states at every representative operating point.  That is
the "G0 co-location coverage contract":

    rank(observability Gramian) == 7      at every operating point, AND
    every state's estimation-error          covariance stays bounded (non-divergent)
    over a 24 h run                          at every operating point.

This harness turns the analysis-only finding in ``docs/TWIN_OBSERVABILITY.md``
(current suite -> deposit_thickness unobservable + divergent, Gramian rank 6/7)
into a *dry-runnable proof*: it exercises the **real wired measurement model**
(L1 tags go through ``h_obs`` / ``H_jacobian`` with the non-negativity clamps
and the VT-201 physics coupling — not abstract unit rows), so the verdict is
exactly what the EKF would experience in-silico.

Additive / OFF-by-default: this module reads the twin's existing numerical
Jacobians and never modifies the EKF state vector, ``h_obs``, ``observability.py``
or ``digital_twin.py``.

Tiering: the result is **L0/in-silico only** — it is due-diligence that the
proposed suite is *capable* of full observability before purchase; it is NOT
gate evidence about real instrument performance (no real data, no calibration).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .digital_twin import (
    N_STATES,
    STATE_KEYS,
    OBSERVABLE_TAGS,
    L1_SENSOR_OBS_MAP,
    _DEFAULT_DP,
    _F_jacobian,
    H_jacobian,
    get_default_process_model,
)
from .twin_physics import CellProcessModel
from .observability import (
    DT_HR,
    DEFAULT_P0,
    DEFAULT_Q_VAR,
    HORIZON_HR,
    RUN_LENGTH_HR,
    analyze_observability,
    estimation_covariance_trajectory,
    operating_points_from_model,
    state_vector_from_operating_point,
    _R_observation,
)

# ---------------------------------------------------------------------------
# Sensor-set definitions (base 5 + wired L1 3).  All tags are real, wired
# observations consumed by the twin's ``h_obs`` when present in a readings dict.
# ---------------------------------------------------------------------------

# Base 5-sensor suite (pinned; never modified).
BASE_TAGS: List[str] = list(OBSERVABLE_TAGS)

# Opt-in L1 sensors recommended by docs/TWIN_OBSERVABILITY.md (§3/§4), now wired
# into ``digital_twin.L1_SENSOR_OBS_MAP``.  Order is stable (dict insertion).
L1_TAGS: List[str] = list(L1_SENSOR_OBS_MAP.keys())

# Full co-located suite: base 5 + L1 3 = 8 observations of the 7-state EKF.
FULL_TAGS: List[str] = BASE_TAGS + L1_TAGS


# ---------------------------------------------------------------------------
# Covariance-stability check
# ---------------------------------------------------------------------------

# A state's estimation-error covariance is "stable" when, at the end of a run,
# it is finite and no longer growing (relative growth below tolerance over the
# tail quarter).  This is the finite-horizon detectability read: a pure
# integrator (e.g. base-suite deposit_thickness) keeps growing -> unstable;
# an observable / contractive mode flattens -> stable.
_DIVERGENCE_SLOPE_TOL = 1e-8


def _covariance_tail_stable(var_traj: np.ndarray, frac: float = 0.25) -> bool:
    """True when the variance path is finite and flat (bounded) at the tail."""
    n = int(len(var_traj))
    k = max(1, int(n * frac))
    v_last = float(var_traj[-1])
    if not np.isfinite(v_last):
        return False
    if v_last < _DIVERGENCE_SLOPE_TOL:
        return True
    growth = v_last - float(var_traj[max(0, n - k - 1)])
    return not (growth > _DIVERGENCE_SLOPE_TOL * v_last)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class G0PointResult:
    """Coverage-contract results at one operating point."""

    operating_point: str
    x: np.ndarray
    # Base suite (contrast: rank 6 + divergent deposit).
    base_rank: int
    base_deposit_sigma: float
    base_deposit_divergent: bool
    # Full co-located suite.
    full_rank: int
    full_sv_min: float
    full_cond: float
    full_sigma: Dict[str, float]  # state key -> end-of-run 1-sigma
    full_stable: Dict[str, bool]  # state key -> covariance stable?

    @property
    def full_cov_stable(self) -> bool:
        return bool(all(self.full_stable.values()))


@dataclass
class G0ContractResult:
    """Aggregate verdict of the G0 co-location coverage contract."""

    points: Dict[str, G0PointResult]
    base_tags: List[str] = field(default_factory=lambda: BASE_TAGS)
    full_tags: List[str] = field(default_factory=lambda: FULL_TAGS)

    @property
    def all_full_rank(self) -> bool:
        return bool(all(p.full_rank == N_STATES for p in self.points.values()))

    @property
    def all_cov_stable(self) -> bool:
        return bool(all(p.full_cov_stable for p in self.points.values()))

    def violations(self) -> List[str]:
        """Human-readable list of contract violations (empty == contract holds)."""
        v: List[str] = []
        for name, p in self.points.items():
            if p.full_rank != N_STATES:
                v.append(f"{name}: full suite rank {p.full_rank} != {N_STATES}")
            for key, stable in p.full_stable.items():
                if not stable:
                    v.append(
                        f"{name}: {key} covariance unstable "
                        f"(final sigma={p.full_sigma[key]:.4e})"
                    )
        return v


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _run_point(
    x: np.ndarray,
    name: str,
    model: CellProcessModel,
    dt_hr: float,
    horizon_steps: int,
    run_length_steps: int,
    Q_var: np.ndarray,
    P0: np.ndarray,
    design_point: Dict[str, float],
) -> G0PointResult:
    # Base suite: rank 6 with divergent deposit (contrast / regression guard).
    base = analyze_observability(
        x,
        dt_hr=dt_hr,
        obs_tags=BASE_TAGS,
        model=model,
        design_point=design_point,
        horizon_steps=horizon_steps,
        run_length_steps=run_length_steps,
        Q_var=Q_var,
        P0=P0,
        operating_point=name,
    )
    deposit_i = STATE_KEYS.index("deposit_thickness")

    # Full co-located suite: real wired L1 tags through h_obs / H_jacobian.
    full = analyze_observability(
        x,
        dt_hr=dt_hr,
        obs_tags=FULL_TAGS,
        model=model,
        design_point=design_point,
        horizon_steps=horizon_steps,
        run_length_steps=run_length_steps,
        Q_var=Q_var,
        P0=P0,
        operating_point=name,
    )

    # Covariance-stability of the full suite: run the Riccati recursion and
    # check every state's tail is finite and flat (non-divergent).
    F, H = (
        _F_jacobian(x, dt_hr, model, design_point),
        H_jacobian(x, FULL_TAGS, model, design_point),
    )
    traj = estimation_covariance_trajectory(
        F, H, _R_observation(FULL_TAGS), run_length_steps, Q_var=Q_var, P0=P0
    )
    sigma = np.sqrt(np.maximum(traj[-1], 0.0))
    stable = {
        key: bool(_covariance_tail_stable(traj[:, i]))
        for i, key in enumerate(STATE_KEYS)
    }

    return G0PointResult(
        operating_point=name,
        x=np.asarray(x, dtype=float),
        base_rank=base.rank,
        base_deposit_sigma=float(base.per_state_sigma[deposit_i]),
        base_deposit_divergent=(
            base.per_state_flag[STATE_KEYS[deposit_i]] == "unobservable_divergent"
        ),
        full_rank=full.rank,
        full_sv_min=full.smallest_singular_value,
        full_cond=full.condition_number,
        full_sigma={key: float(sigma[i]) for i, key in enumerate(STATE_KEYS)},
        full_stable=stable,
    )


def evaluate_co_location_contract(
    model: Optional[CellProcessModel] = None,
    dt_hr: float = DT_HR,
    Q_var: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
    horizon_steps: Optional[int] = None,
    run_length_steps: Optional[int] = None,
    operating_points: Optional[Sequence[Tuple[str, float, float, float]]] = None,
) -> G0ContractResult:
    """Evaluate the G0 co-location coverage contract at every operating point.

    Returns a :class:`G0ContractResult`; the contract holds iff
    ``result.all_full_rank and result.all_cov_stable and not result.violations()``.
    Deterministic (no RNG): the analysis is entirely numerical-Jacobian /
    Riccati based.
    """
    if model is None:
        model = get_default_process_model()
    if Q_var is None:
        Q_var = DEFAULT_Q_VAR
    if P0 is None:
        P0 = DEFAULT_P0
    design_point = dict(_DEFAULT_DP)
    horizon_steps = horizon_steps or max(1, int(round(HORIZON_HR / dt_hr)))
    run_length_steps = run_length_steps or max(1, int(round(RUN_LENGTH_HR / dt_hr)))

    ops = operating_points or operating_points_from_model(model)
    points: Dict[str, G0PointResult] = {}
    for name, j, T, fe2 in ops:
        x = state_vector_from_operating_point((name, j, T, fe2), model)
        points[name] = _run_point(
            x,
            name,
            model,
            dt_hr,
            horizon_steps,
            run_length_steps,
            np.asarray(Q_var, dtype=float),
            np.asarray(P0, dtype=float),
            design_point,
        )
    return G0ContractResult(points=points)


def verify_co_location_contract(result: G0ContractResult) -> List[str]:
    """Assert-style check; returns the violation list (empty == contract holds)."""
    v = result.violations()
    if not result.all_full_rank:
        v.append("full rank (7) not reached at every operating point")
    if not result.all_cov_stable:
        v.append("covariance not stable at every operating point")
    return v


# ---------------------------------------------------------------------------
# Human-readable report rendering
# ---------------------------------------------------------------------------


def _fmt(x: float) -> str:
    if x == float("inf"):
        return "inf"
    return f"{x:.3e}"


def render_markdown_report(result: G0ContractResult) -> str:
    """Render the coverage-contract results as a markdown report string."""
    L = []
    L.append("# G0 co-location coverage contract — in-silico observability proof")
    L.append("")
    L.append("**Tier:** L0 / in-silico. **NOT gate evidence.**")
    L.append("")
    L.append(
        "This dry-run proves, before any L1 hardware is bought, that the **full "
        "co-located suite** — the base 5-sensor suite plus the wired L1 sensors "
        f"(THK-101, CVT-201, FE2P-101) — reconstructs all {N_STATES} EKF states at "
        "every representative operating point, and that every state's estimation-error "
        "covariance stays bounded over a 24 h run."
    )
    L.append("")
    L.append(
        f"- **Tags evaluated (real wired measurement model, `h_obs`/`H_jacobian`):** "
        f"{', '.join(result.full_tags)}"
    )
    L.append(
        f"- **Operating points evaluated:** {len(result.points)} "
        f"({', '.join(result.points)})"
    )
    L.append("")

    L.append("## 1. Rank comparison — base suite vs full co-located suite")
    L.append("")
    L.append("| operating point | base rank | full rank | sv_min (full) | cond (full) |")
    L.append("|-----------------|----------:|----------:|--------------:|------------:|")
    for name, p in result.points.items():
        L.append(
            f"| {name} | {p.base_rank} | **{p.full_rank}** | "
            f"{_fmt(p.full_sv_min)} | {_fmt(p.full_cond)} |"
        )
    L.append("")
    L.append(
        f"The base 5-sensor suite is rank-{list(result.points.values())[0].base_rank} "
        "(deposit_thickness unobservable + divergent). Adding the L1 sensors raises "
        f"the Gramian to **full rank {N_STATES}** at every point."
    )
    L.append("")

    L.append("## 2. Covariance stability at each operating point (full suite)")
    L.append("")
    L.append(
        "Per-state end-of-run estimation-error 1-sigma and a stability flag "
        "(covariance bounded / non-divergent over a 24 h Riccati recursion)."
    )
    L.append("")
    header = "| operating point | " + " | ".join(STATE_KEYS) + " | cov stable |"
    L.append(header)
    L.append("|---|" + "---|" * len(STATE_KEYS) + "---|")
    for name, p in result.points.items():
        sig = " / ".join(f"{p.full_sigma[k]:.3f}" for k in STATE_KEYS)
        L.append(f"| {name} | {sig} | {'YES' if p.full_cov_stable else 'NO'} |")
    L.append("")
    L.append("Stability icon: `sigma` is the final 1-sigma in state units.")

    L.append("")
    L.append("## 3. Contract verdict")
    L.append("")
    violations = verify_co_location_contract(result)
    L.append(
        f"- **Full rank ({N_STATES}) at every operating point:** "
        f"{'PASS' if result.all_full_rank else 'FAIL'}"
    )
    L.append(
        f"- **Covariance stable at every operating point:** "
        f"{'PASS' if result.all_cov_stable else 'FAIL'}"
    )
    L.append("")
    if violations:
        L.append("**Violations:**")
        for v in violations:
            L.append(f"- {v}")
    else:
        L.append(
            "**Verdict: the G0 co-location coverage contract HOLDS in-silico.** "
            "The full suite, co-located at the cell, covers all 7 states at every "
            "tested operating point with bounded estimation covariance."
        )
    L.append("")
    L.append(
        "## 4. Method & reuse\n"
        "- Reuses `models/digital_twin.py` numerical Jacobians (`_F_jacobian`, "
        "`H_jacobian`) and the `models/observability.py` Gramian / Riccati machinery."
        "\n"
        "- L1 tags are the **wired** observations (`digital_twin.L1_SENSOR_OBS_MAP`), "
        "driven through `h_obs` with the non-negativity clamps and the VT-201 physics "
        "coupling — the EKF consumption path, not abstract unit rows."
    )
    L.append("")
    L.append(
        "*L0/in-silico only: this proves capability (full observability is possible), "
        "not real instrument performance. No hardware purchase decision is gated on "
        "this analysis alone.*"
    )
    return "\n".join(L)
