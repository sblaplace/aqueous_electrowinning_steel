"""D6 — L5 live-control target: wire the operating twin to a *calibrated + observed* cell.

This is an **autonomy-design / wiring-contract** module (D-series deliverable D6).
It does NOT claim live control has been achieved.  The L5 ladder in
``docs/NEXT_STEPS.md`` (§ Model credibility ladder) requires, before a model may
claim Level-5 "constrained online estimation/control", that the cell be **both**:

* **calibrated** — the kinetics (Fe vs HER Tafel exchange-currents, boundary
  layer) were fit from *real* RDE/LSV + gas/deposit data
  (``kinetics_fit_pipeline`` / ``calibration``), not screening defaults; and
* **observed** — the sensor suite provably reconstructs the full 7-state EKF
  vector, i.e. the **G0 co-location coverage contract** HOLDS
  (``g0_co_location`` → full-rank Gramian + bounded covariance at every point).

D6 defines the *connection* between ``models/operating_twin.py`` (safety /
supervisory: trips, arming, bounded actuation) and such a cell.  The central
artifact is a **fail-closed qualification gate**: the operating twin may be
**armed for actuation only when BOTH pillars qualify AND a bounded actuation
envelope exists** (the calibrated kinetics must sustain a current density at
or above the current-efficiency floor somewhere inside the operating window,
so a non-zero bounded command exists).  Removing either pillar (or leaving
the real Q1-Q5 + D1/D2/D3 evidence pending) — or a calibration whose kinetics
cannot hold the CE floor — keeps the twin in ADVISORY and refuses arming.
This is the L5 live-control target of the sled.

``CellQualification.reasons()`` is the single source of truth:
``qualified == not reasons()``.

The module is purely additive and screening-grade (L0): it imports and reuses
the existing twin, calibration, and observability machinery and does **not**
modify any existing ``models/*.py``.  It is exercised in-silico (a synthetic
calibration fit and the in-silico G0 proof) to demonstrate the wiring contract
and the gate's fail-closed behaviour before any hardware evidence lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .operating_twin import (
    OperatingTwin,
    SensorSnapshot,
    TwinMode,
)
from .kinetics import DepositionKinetics
from .g0_co_location import (
    G0ContractResult,
    evaluate_co_location_contract,
)

# Default minimum current efficiency (fraction) that calibrated kinetics must
# deliver at the bounded actuation target to remain L5-eligible (screening).
DEFAULT_MIN_FE_FLOOR = 0.80


# ---------------------------------------------------------------------------
# Calibration pillar
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationEvidence:
    """Evidence that a named cell's kinetics were fit from data (or explicitly
    in-silico/synthetic for the dry run)."""

    cell_id: str
    kinetics: DepositionKinetics
    source: str = "synthetic"        # "real_data" | "synthetic" | "none"
    n_points: int = 0
    fit_rmse_log10: Optional[float] = None
    notes: Tuple[str, ...] = ()

    @property
    def qualified(self) -> bool:
        """Calibration qualifies iff we hold fitted kinetics, not screening defaults."""
        return self.kinetics is not None and self.source != "none"

    def efficiency_at(self, j_mA_cm2: float) -> float:
        return float(self.kinetics.efficiency_at_current(j_mA_cm2))

    def deposit_rate_um_hr(self, j_mA_cm2: float) -> float:
        return float(self.kinetics.deposition_rate_um_hr(j_mA_cm2))


# ---------------------------------------------------------------------------
# Observability pillar
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObservabilityEvidence:
    """Evidence that the full sensor suite reconstructs all EKF states.

    Wraps the G0 co-location contract result.  The observability pillar
    qualifies only when that contract HOLDS: full-rank Gramian (== N_STATES)
    at every operating point AND bounded (non-divergent) covariance.
    """

    cell_id: str
    contract: G0ContractResult
    full_tags: Tuple[str, ...] = ()

    @property
    def qualified(self) -> bool:
        return bool(
            self.contract is not None
            and self.contract.all_full_rank
            and self.contract.all_cov_stable
            and not self.contract.violations()
        )

    @property
    def rank(self) -> int:
        ranks = {p.full_rank for p in self.contract.points.values()}
        return max(ranks) if ranks else 0


# ---------------------------------------------------------------------------
# Fail-closed qualification gate (the L5 precondition)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellQualification:
    """A cell is L5-live-control-eligible ONLY when calibrated AND observed
    AND a bounded-actuation envelope exists.

    This is the fail-closed gate: ``qualified`` is True iff BOTH pillars
    qualify *and* the calibrated kinetics deliver a current density at/above
    the current-efficiency floor somewhere within the operating envelope, so
    that a non-zero bounded command exists.  If calibration is missing, or
    observability is missing (G0 does not hold), or no bounded actuation is
    feasible, the twin must remain advisory and reject arming.

    ``reasons()`` is the single source of truth: ``qualified == not reasons()``.
    """

    cell_id: str
    calibration: CalibrationEvidence
    observability: ObservabilityEvidence
    min_fe_floor: float = DEFAULT_MIN_FE_FLOOR
    operating_current_density_mA_cm2: float = 300.0

    @property
    def calibration_ok(self) -> bool:
        return self.calibration.qualified

    @property
    def observability_ok(self) -> bool:
        return self.observability.qualified

    @property
    def bounded_viable_current_density_mA_cm2(self) -> float:
        """Largest current density (mA/cm^2) that keeps calibrated FE at/above
        the floor, walking down from the operating density.  Returns 0.0 if
        NO operating point satisfies the floor (no bounded actuation exists)."""
        j = self.operating_current_density_mA_cm2
        while j > 0.0 and self.calibration.efficiency_at(j) < self.min_fe_floor:
            j -= 10.0
        return max(0.0, j)

    @property
    def qualified(self) -> bool:
        return len(self.reasons()) == 0

    def reasons(self) -> List[str]:
        """Human-readable list of what is missing / blocking live control.

        This is exhaustive: a non-empty list means NOT qualified for arming.
        """
        r: List[str] = []
        if not self.calibration_ok:
            r.append("calibration_missing_or_pending")
        if not self.observability_ok:
            r.append("observability_contract_not_holding")
        # Only assess the bounded-actuation envelope once both pillars are in.
        if self.calibration_ok and self.observability_ok:
            if self.bounded_viable_current_density_mA_cm2 <= 0.0:
                r.append(
                    f"no_viable_bounded_actuation_fe_floor_{self.min_fe_floor:.3f}"
                )
        return r

    def bounded_target_current_A(self, area_cm2: float) -> float:
        """Largest bounded current (A) that keeps calibrated FE at/above the
        floor within the operating envelope (0.0 if none feasible)."""
        j = self.bounded_viable_current_density_mA_cm2
        return j * area_cm2 / 1000.0

    def to_dict(self) -> Dict:
        return {
            "cell_id": self.cell_id,
            "qualified": self.qualified,
            "calibration_ok": self.calibration_ok,
            "observability_ok": self.observability_ok,
            "observability_rank": (
                self.observability.rank if self.observability_ok else 0
            ),
            "bounded_viable_current_density_mA_cm2": (
                self.bounded_viable_current_density_mA_cm2
            ),
            "reasons": self.reasons(),
            "min_fe_floor": self.min_fe_floor,
            "operating_current_density_mA_cm2": (
                self.operating_current_density_mA_cm2
            ),
        }


# ---------------------------------------------------------------------------
# The closed-loop live-control twin
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LiveControlStep:
    """A single closed-loop step: observed state -> safety -> bounded command."""

    timestamp_s: float
    qualification: "CellQualification"
    observed_deposit_thickness_um: float
    calibrated_fe: float
    command_current_A: float
    mode: TwinMode
    reasons: Tuple[str, ...] = ()

    @property
    def qualified(self) -> bool:
        return self.qualification.qualified

    def to_dict(self) -> Dict:
        return {
            "timestamp_s": self.timestamp_s,
            "qualified": self.qualification.qualified,
            "observed_deposit_thickness_um": self.observed_deposit_thickness_um,
            "calibrated_fe": self.calibrated_fe,
            "command_current_A": self.command_current_A,
            "mode": self.mode.value,
            "reasons": list(self.reasons),
        }


class ClosedLoopLiveControlTwin:
    """Wire the safety/supervisory operating twin to a calibrated + observed cell.

    The twin owns an :class:`OperatingTwin` (trips, arming, bounded actuation)
    and a fail-closed :class:`CellQualification`.  It demonstrates the L5
    closed loop in-silico:

        1. an observed sensor snapshot (the full suite) -> safety evaluation;
        2. the calibrated kinetics -> the FE/deposit that bound actuation;
        3. a bounded command, emitted ONLY when both pillars qualify.

    ``arm_actuation`` reuses the operating twin's token guard, so even a
    qualified twin does not actuate without the explicit cell-id token.
    """

    def __init__(
        self,
        qualification: CellQualification,
        operating_twin: OperatingTwin,
        cathode_area_cm2: float = 10.0,
    ) -> None:
        self.qualification = qualification
        self.twin = operating_twin
        self.area_cm2 = cathode_area_cm2
        self._steps: List[LiveControlStep] = []
        self._steps.append(
            LiveControlStep(
                timestamp_s=0.0,
                qualification=qualification,
                observed_deposit_thickness_um=0.0,
                calibrated_fe=0.0,
                command_current_A=0.0,
                mode=self.twin.mode,
                reasons=tuple(self.qualification.reasons()),
            )
        )

    @property
    def qualified(self) -> bool:
        return self.qualification.qualified

    @property
    def mode(self) -> TwinMode:
        return self.twin.mode

    def arm_actuation(self, token: str) -> None:
        """Arm actuation ONLY through the fail-closed gate + token guard.

        Raises if the cell is not qualified (both pillars) or the token does
        not match the configured cell id.
        """
        if not self.qualification.qualified:
            raise PermissionError(
                "cannot arm live control: " + "; ".join(self.qualification.reasons())
            )
        # The OperatingTwin's own token guard protects against arming the wrong
        # / an unqualified configuration even if the gate above were bypassed.
        self.twin.arm_actuation(token)

    def step(
        self,
        snapshot: SensorSnapshot,
        observed_deposit_thickness_um: Optional[float] = None,
    ) -> LiveControlStep:
        """Run one closed-loop step: safety eval + calibrated bounded command.

        The command's current is the operating twin's bounded/ramped request,
        capped by the calibrated-FE-preserving target.  If either pillar is
        missing, or a trip fires, the twin returns ADVISORY/TRIPPED with 0 A.
        """
        state = self.twin.update(snapshot, now_s=snapshot.timestamp_s)
        qualified = self.qualification.qualified
        ce = (
            self.qualification.calibration.efficiency_at(
                self.qualification.operating_current_density_mA_cm2
            )
            if qualified
            else 0.0
        )
        reasons: List[str] = list(state.trip_reasons) if state.trip_reasons else []

        if not qualified:
            reasons = reasons + list(self.qualification.reasons())
            current_A = 0.0
            mode = TwinMode.ADVISORY if state.mode != TwinMode.TRIPPED else state.mode
        else:
            cmd = self.twin.command(now_s=snapshot.timestamp_s)
            # Bound the commanded current by the calibrated-FE target.
            target = self.qualification.bounded_target_current_A(self.area_cm2)
            current_A = min(cmd.current_A, target)
            mode = cmd.mode
            if current_A <= 0.0 and mode == TwinMode.ACTUATION:
                mode = TwinMode.ADVISORY
                reasons.append("bounded_to_zero_by_calibrated_FE")

        step = LiveControlStep(
            timestamp_s=snapshot.timestamp_s,
            qualification=self.qualification,
            observed_deposit_thickness_um=(
                float(observed_deposit_thickness_um)
                if observed_deposit_thickness_um is not None
                else 0.0
            ),
            calibrated_fe=float(ce),
            command_current_A=float(current_A),
            mode=mode,
            reasons=tuple(reasons),
        )
        self._steps.append(step)
        return step

    @property
    def steps(self) -> List[LiveControlStep]:
        return list(self._steps)

    def live_control_target(self) -> Dict:
        """The L5 live-control target definition (machine-readable)."""
        return {
            "deliverable": "D6",
            "level": "L5",
            "title": "Wire operating_twin closed-loop to a calibrated + observed cell",
            "cell_id": self.qualification.cell_id,
            "cathode_area_cm2": self.area_cm2,
            "fail_closed_gate": "require calibration AND observability before arming",
            "qualification": self.qualification.to_dict(),
            "observed_suite": list(self.qualification.observability.full_tags),
            "evidence_gates": {
                "q1_mass_charge_energy_ledgers": "pending",
                "q3_rde_volumetric_h2": "pending",
                "q4_membrane_anode_viability": "pending",
                "q5_adhesion_release_stress_h2": "pending",
                "d1_immutable_reference_spec": "pending",
                "d2_d3_calibration_and_observability_basis": "pending",
            },
            "note": (
                "Design / wiring contract demonstrated in-silico. Real Q1-Q5 + "
                "D1/D2/D3 evidence must land before hardware actuation is armed."
            ),
        }


def build_reference_cell_qualification(
    kinetics: Optional[DepositionKinetics] = None,
    *,
    cell_id: str = "RC-1",
    run_co_location_contract: bool = True,
    operating_current_density_mA_cm2: float = 300.0,
    min_fe_floor: float = DEFAULT_MIN_FE_FLOOR,
) -> CellQualification:
    """Build a :class:`CellQualification` for the reference cell from an
    (optionally synthetic/real) calibrated kinetics model and the G0 proof.

    Parameters
    ----------
    kinetics : DepositionKinetics, optional
        Fitted kinetics (from real data, or synthetic for the dry run).  If
        None, a screening-default kinetics is used but marked source='none'
        (so the calibration pillar does NOT qualify).
    run_co_location_contract : bool
        If True, run the in-silico G0 co-location contract to obtain the
        observability evidence.  If False, an empty non-qualifying observability
        evidence is returned (source='none') — useful for testing the gate.
    """
    cal = CalibrationEvidence(
        cell_id=cell_id,
        kinetics=(kinetics if kinetics is not None else _screening_default_kinetics()),
        source=("real_data" if kinetics is not None else "none"),
    )
    if run_co_location_contract:
        contract = evaluate_co_location_contract()
        full_tags = tuple(contract.full_tags)
        obs = ObservabilityEvidence(cell_id=cell_id, contract=contract, full_tags=full_tags)
    else:
        obs = ObservabilityEvidence(
            cell_id=cell_id,
            contract=None,  # type: ignore[arg-type]
            full_tags=(),
        )
    return CellQualification(
        cell_id=cell_id,
        calibration=cal,
        observability=obs,
        min_fe_floor=min_fe_floor,
        operating_current_density_mA_cm2=operating_current_density_mA_cm2,
    )


def _screening_default_kinetics() -> DepositionKinetics:
    """A physically plausible screening kinetics (Fe on Fe, mildly acidic bath).

    Used ONLY as a placeholder so the wiring/types hold when no fit is supplied;
    because the calibration evidence source is 'none', it never qualifies.
    """
    return DepositionKinetics(
        pH=2.0,
        temperature_C=60.0,
        fe_i0=1.0e-2,
        her_i0=1.0e-4,
        fe_tafel_V=0.120,
        her_tafel_V=0.140,
        fe_conc_M=1.0,
        boundary_layer_m=5e-5,
    )
