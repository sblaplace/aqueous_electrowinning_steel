"""
Dry-runnable G0 co-location coverage-contract proof driver (L0, in-silico).

Run with::

    aq-steel-observability-g0              # or: python -m models.run_observability_g0

Prints the coverage-contract verdict (full rank 7 + covariance stability at every
operating point) to stdout and writes a short markdown report.  The default report
path is ``outputs/G0_CO_LOCATION_REPORT.md``; override with ``--out PATH`` or pass
``--no-write`` to only print.

Fails (exit code 1) if the contract does NOT hold — this is the turn-key "proof"
job: before buying L1 hardware, run this and require exit 0 + all-PASS verdict.

Reuses ``models/digital_twin.py`` numerical Jacobians and the
``models/observability.py`` Gramian/Riccati machinery via
``models/g0_co_location.py``.  Additive; nothing destructive.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

from .g0_co_location import (
    FULL_TAGS,
    evaluate_co_location_contract,
    render_markdown_report,
    verify_co_location_contract,
)
from .digital_twin import get_default_process_model


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="G0 co-location coverage-contract in-silico proof (L0)."
    )
    parser.add_argument(
        "--out",
        default="outputs/G0_CO_LOCATION_REPORT.md",
        help="Path to write the markdown report (default: outputs/G0_CO_LOCATION_REPORT.md).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print the report to stdout only; do not write a file.",
    )
    args = parser.parse_args(argv)

    model = get_default_process_model()
    result = evaluate_co_location_contract(model)
    report = render_markdown_report(result)

    # Console verdict (compact).
    print("=" * 70)
    print("G0 co-location coverage contract — in-silico proof (L0, NOT gate evidence)")
    print("=" * 70)
    print(f"Full suite tags: {', '.join(FULL_TAGS)}")
    print(f"{'operating point':24s} {'base_rank':>9s} {'full_rank':>9s} "
          f"{'sv_min':>9s} {'cov_stable':>10s}")
    for name, p in result.points.items():
        print(f"{name:24s} {p.base_rank:9d} {p.full_rank:9d} "
              f"{p.full_sv_min:9.3e} {str(p.full_cov_stable):>10s}")

    violations = verify_co_location_contract(result)
    print("-" * 70)
    print(f"FULL RANK ({len(FULL_TAGS) and 7}) AT ALL POINTS: "
          f"{result.all_full_rank}")
    print(f"COVARIANCE STABLE AT ALL POINTS: {result.all_cov_stable}")
    print(f"CONTRACT HOLDS: {result.all_full_rank and result.all_cov_stable and not violations}")

    if not args.no_write:
        out = args.out
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            f.write(report)
        print(f"\nReport written to: {os.path.abspath(out)}")

    return 0 if (result.all_full_rank and result.all_cov_stable and not violations) else 1


if __name__ == "__main__":
    raise SystemExit(main())
