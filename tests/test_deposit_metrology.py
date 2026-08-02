"""L0 dry-run tests for the co-live optical + ultrasonic deposit-metrology pair.

Covers the disjoint error models, the discrepancy/agreement logic, and the
end-to-end operating-twin safe-state outcome (agree => normal; disagree =>
trip).  The co-live pair is a *validator* capability upstream of the twin: it
does not modify the base L1 sensor maps or the EKF, so the baseline twin stays
byte-identical (docs/DEPOSIT_METROLOGY.md sm3/sm5).
"""

import math

import numpy as np

from models.deposit_metrology import (
    DEGRADE_Z,
    FAULT_Z,
    OPTICAL_ERROR,
    ULTRASONIC_ERROR,
    assess_pair,
    optical_reading,
    run_deposit_metrology_demo,
    ultrasonic_reading,
)


def test_combined_uncertainty_uses_disjoint_envelopes():
    """The two channels' total uncertainties are independent and RSS-combined."""
    assert OPTICAL_ERROR.total_unc_um < ULTRASONIC_ERROR.total_unc_um
    # A coupled/blanket noise source would not be disjoint; the pair is only
    # useful if neither channel's envelope dominates the other entirely.
    assert ULTRASONIC_ERROR.total_unc_um < 3 * OPTICAL_ERROR.total_unc_um


def test_agree_when_channels_track_truth():
    """Identical channels (truth + small noise) must agree with quality 'ok'."""
    rng = np.random.default_rng(0)
    individual_pass = True
    for _ in range(200):
        truth = 50.0
        v = assess_pair(optical_reading(truth, 0.0, rng),
                        ultrasonic_reading(truth, 0.0, rng))
        individual_pass = individual_pass and v.agreed
        individual_pass = individual_pass and all(
            q == "ok" for q in v.sensor_quality.values())
    assert individual_pass


def test_optical_fault_escalates_pair_to_faulted():
    """A large +bias on the optical channel makes the pair discordant (faulted).

    With only two co-live channels the pair cannot attribute which one is wrong
    (a biased channel pulls the fused mean toward itself), so both are flagged
    and the operating twin holds.  Attribution needs a 3rd independent reference.
    """
    truth = 100.0
    opt = optical_reading(truth, 2.0, np.random.default_rng(1),
                          fault={"magnitude_um": 60.0, "ramp_hr": 0.5})
    us = ultrasonic_reading(truth, 2.0, np.random.default_rng(1))
    v = assess_pair(opt, us)
    assert v.level == "faulted"
    assert v.suspect == "both"
    assert all(q == "faulted" for q in v.sensor_quality.values())
    assert v.z_score > FAULT_Z


def test_ultrasonic_fault_escalates_pair_to_faulted():
    """A large -bias on the ultrasonic channel also makes the pair faulted."""
    truth = 100.0
    opt = optical_reading(truth, 2.0, np.random.default_rng(2))
    us = ultrasonic_reading(truth, 2.0, np.random.default_rng(2),
                            fault={"magnitude_um": -50.0, "ramp_hr": 0.5})
    v = assess_pair(opt, us)
    assert v.level == "faulted"
    assert v.suspect == "both"
    assert v.z_score > FAULT_Z


def test_discrepancy_escalates_through_degrade_to_fault():
    """A modest bias reads 'degraded' (z between gates); a large one 'faulted'."""
    u_c = math.hypot(OPTICAL_ERROR.total_unc_um, ULTRASONIC_ERROR.total_unc_um)
    half = DEGRADE_Z * u_c * 1.2 / 2  # |d| = 1.2 * DEGRADE_Z * u_c  (just under fault)
    small = assess_pair(100.0 + half, 100.0 - half)
    assert not small.agreed
    assert small.level == "degraded"
    assert DEGRADE_Z < small.z_score <= FAULT_Z
    large = assess_pair(100.0 + 60.0, 100.0)
    assert large.level == "faulted"


def test_dry_run_agree_normal_operation():
    """Healthy pair: no trip, actuation allowed, deposit estimate bounded."""
    report = run_deposit_metrology_demo(seed=42)
    agree = next(r for r in report["scenarios"] if r["scenario"] == "agree")
    assert agree["passed"]
    assert agree["final_verdict"]["level"] == "agree"
    assert agree["operating_twin"]["mode"] == "actuation"
    assert agree["operating_twin"]["command_current_A"] > 0.0


def test_dry_run_disagree_trips_twin():
    """Each single-channel fault must trip the operating twin to zero current."""
    report = run_deposit_metrology_demo(seed=42)
    assert report["all_passed"]
    for name in ("optical_foul", "ultrasound_interface"):
        row = next(r for r in report["scenarios"] if r["scenario"] == name)
        assert row["passed"]
        assert row["expected_trip"]
        assert row["operating_twin"]["mode"] == "tripped"
        assert row["operating_twin"]["command_current_A"] == 0.0
        assert row["shutdown_action"] == "sensor_fault_hold"
