"""
Deterministic observability & sensor-placement analysis driver (L0).

Run with::

    aq-steel-observability          # or: python -m models.run_observability

Prints the rank/conditioning table per operating point, per-state observability
flags, the candidate-sensor marginal-information ranking, and the recommended
minimum sensor set that restores full observability of the 7-state EKF twin.

This is an analysis-only tool: it reads the twin's existing numerical Jacobians
and never touches the EKF dynamics or ``h_obs``, and it does not fetch or
require any real data.
"""

from __future__ import annotations

from .digital_twin import STATE_KEYS, get_default_process_model
from .observability import (
    CANDIDATE_SENSORS,
    RECOMMENDED_MINIMUM_SENSORS,
    MINIMUM_SET_FOR_FULL_OBSERVABILITY,
    characterize_current_suite,
    evaluate_sensor_set_over_grid,
    operating_points_from_model,
    rank_candidate_sensors,
    state_vector_from_operating_point,
)


def _fmt(x: float) -> str:
    if x == float("inf"):
        return "inf"
    return f"{x:.3e}"


def main() -> None:
    model = get_default_process_model()
    ops = operating_points_from_model(model)

    print("=" * 78)
    print("Observability & sensor-placement analysis (L0 screening)")
    print("=" * 78)
    print(f"\nRepresentative operating points ({len(ops)}):")
    for name, j, T, fe2 in ops:
        print(f"  {name:22s} j={j:6.1f} mA/cm2  T={T:5.1f} C  fe2={fe2:.2f} M")

    # 1. Current suite
    print("\n--- Current 5-sensor suite: rank / conditioning per operating point ---")
    results = characterize_current_suite(model)
    print(f"{'operating_point':24s} {'rank':>5s} {'sv_min_nz':>12s} {'cond':>10s}")
    for name, r in results.items():
        print(
            f"{name:24s} {r.rank:5d} {_fmt(r.smallest_nonzero_singular_value):>12s} "
            f"{_fmt(r.condition_number):>10s}"
        )

    print("\n--- Per-state observability (nominal operating point) ---")
    r = results["nominal"]
    print(f"{'state':22s} {'flag':24s} {'score':>6s} {'rel_energy':>11s} {'sigma':>8s}")
    for i, key in enumerate(STATE_KEYS):
        print(
            f"{key:22s} {r.per_state_flag[key]:24s} {r.per_state_score[i]:6.3f} "
            f"{r.per_state_rel_energy[i]:11.4f} {r.per_state_sigma[i]:8.4f}"
        )

    # 2. Candidate sensor ranking (nominal)
    print("\n--- Candidate-sensor marginal-information ranking (nominal) ---")
    x = state_vector_from_operating_point(ops[0], model)
    print(
        f"{'tag':10s} {'target':22s} {'var_before':>11s} {'var_after':>11s} "
        f"{'reduction':>11s} {'rel':>6s} {'resolves':>9s} {'div':>5s}"
    )
    for rank in rank_candidate_sensors(x, model=model):
        print(
            f"{rank.sensor.tag:10s} {rank.target_key:22s} "
            f"{rank.variance_before:11.4f} {rank.variance_after:11.5f} "
            f"{rank.variance_reduction:11.4f} {rank.relative_reduction:6.2f} "
            f"{str(rank.resolves_unobservability):>9s} {str(rank.divergent_before):>5s}"
        )

    # 3. Recommended minimum set
    print("\n--- Recommended minimum sensor set ---")
    for s in RECOMMENDED_MINIMUM_SENSORS:
        print(
            f"  {s.tag:10s} {s.quantity:20s} {s.unit:4s} target=state {s.target_state} "
            f"({STATE_KEYS[s.target_state]}) noise={s.noise_std}"
        )
    print(
        "\n  Strict minimum for full observability (rank 7): "
        + ", ".join(s.tag for s in MINIMUM_SET_FOR_FULL_OBSERVABILITY)
    )

    print("\n--- Recommended set: rank per operating point ---")
    rec = evaluate_sensor_set_over_grid(list(RECOMMENDED_MINIMUM_SENSORS), model)
    all_ok = True
    for name, rr in rec.items():
        flags = set(rr.per_state_flag.values())
        ok = rr.rank == len(STATE_KEYS) and not any(f.startswith("unobservable") for f in flags)
        all_ok = all_ok and ok
        print(
            f"  {name:24s} rank={rr.rank} sv_min={_fmt(rr.smallest_singular_value)} "
            f"cond={_fmt(rr.condition_number)} full_observable={ok}"
        )
    print(f"\nAll 7 states observable at every tested operating point: {all_ok}")

    print("\nCandidates evaluated (not all needed):")
    for s in CANDIDATE_SENSORS:
        print(
            f"  {s.tag:10s} {s.quantity:20s} -> state {s.target_state} "
            f"({STATE_KEYS[s.target_state]}): {s.rationale}"
        )


if __name__ == "__main__":
    main()
