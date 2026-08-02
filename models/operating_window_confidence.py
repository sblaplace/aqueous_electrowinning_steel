"""Level-0 operating-window confidence screen for the reference cell.

This module extends the single-point reference-cell screen in
:mod:`models.theory_confidence` to a temperature × Fe(II) × current-density
surface.  Its results are transparent synthetic predictions, flagged
``unvalidated (L0)``: they are **NOT gate evidence**.  Process gates remain
measurement-only in :mod:`models.process_gates`; no laboratory data are used.

The screen deliberately reuses the acceptance targets and thermal wiring from
``theory_confidence`` rather than creating a second set of route criteria.
"""
from __future__ import annotations

from dataclasses import replace
from statistics import median
from typing import Any, Iterable

from .cell_physics import CellPhysics, OperatingPoint
from .theory_confidence import (
    SCREENING_FLAG,
    ReferenceCell,
    reference_cell,
    solve_reference,
    thermal_balance,
)

_TARGET_NAMES = (
    "fe",
    "v_cell",
    "specific_energy",
    "transport_limit",
    "deposition_rate",
    "thermal",
)


def _point_verdicts(rc: ReferenceCell, op: OperatingPoint) -> dict[str, dict[str, Any]]:
    """Apply the exact non-thermal screening criteria used at the reference."""
    targets = rc.targets
    transport_margin = (
        op.transport_limit_mA_cm2 / op.j_mA_cm2 if op.j_mA_cm2 > 0.0 else 0.0
    )
    return {
        "fe": {"value": op.current_efficiency, "pass": op.current_efficiency >= targets.fe_min},
        "v_cell": {
            "value": op.V_cell,
            "pass": targets.v_cell_min <= op.V_cell <= targets.v_cell_max,
        },
        "specific_energy": {
            "value": op.specific_energy_kWh_t,
            "pass": op.specific_energy_kWh_t <= targets.specific_energy_max_kWh_t,
        },
        "transport_limit": {
            "value": transport_margin,
            "pass": transport_margin >= targets.transport_margin_min,
        },
        "deposition_rate": {
            "value": op.deposition_rate_um_hr,
            "pass": (
                targets.deposit_rate_min_um_hr
                <= op.deposition_rate_um_hr
                <= targets.deposit_rate_max_um_hr
            ),
        },
    }


def _margin_values(rc: ReferenceCell, values: dict[str, float]) -> dict[str, float]:
    """Return dimensionless headroom; positive means all relevant bounds pass."""
    targets = rc.targets
    return {
        "fe": values["fe"] / targets.fe_min - 1.0,
        "v_cell": min(
            values["v_cell"] / targets.v_cell_min - 1.0,
            1.0 - values["v_cell"] / targets.v_cell_max,
        ),
        "specific_energy": 1.0 - values["specific_energy"] / targets.specific_energy_max_kWh_t,
        "transport_limit": values["transport_limit"] / targets.transport_margin_min - 1.0,
        "deposition_rate": min(
            values["deposition_rate"] / targets.deposit_rate_min_um_hr - 1.0,
            1.0 - values["deposition_rate"] / targets.deposit_rate_max_um_hr,
        ),
        "thermal": 1.0 - values["thermal"] / targets.thermal_limit_C,
    }


def _corner(
    t: float,
    fe: float,
    j: float,
    t_grid: tuple[float, ...],
    fe_grid: tuple[float, ...],
    thermal_j: float,
) -> bool:
    """Identify one of four bath extremes at the reference-current slice."""
    return (
        t in (min(t_grid), max(t_grid))
        and fe in (min(fe_grid), max(fe_grid))
        and j == thermal_j
    )


def sweep_window(
    rc: ReferenceCell = reference_cell(), *, t_grid: Iterable[float], fe_grid: Iterable[float], j_grid: Iterable[float]
) -> dict[str, Any]:
    """Sweep a bounded T × Fe(II) × applied-j screening surface.

    The five electrochemical acceptance targets are evaluated at every point.
    At the four extreme temperature × Fe(II) bath corners on the sampled
    current slice closest to the #32 reference current, ``thermal_balance``
    supplies the sixth (cooled-temperature) verdict.  A point is usable only
    when every applicable verdict passes.  Solver failures are retained as
    explicit failed points rather than dropped.
    """
    temperatures = tuple(float(value) for value in t_grid)
    concentrations = tuple(float(value) for value in fe_grid)
    currents = tuple(float(value) for value in j_grid)
    if not temperatures or not concentrations or not currents:
        raise ValueError("t_grid, fe_grid, and j_grid must each contain at least one value")

    reference_j = solve_reference(rc)["current_density_mA_cm2"]
    thermal_j = min(currents, key=lambda current: abs(current - reference_j))
    rows: list[dict[str, Any]] = []
    for temperature in temperatures:
        for fe_concentration in concentrations:
            bath = replace(rc.bath, c_FeSO4_M=fe_concentration)
            conditions = replace(rc.conditions, temperature_C=temperature)
            physics = CellPhysics(bath, rc.geometry, conditions)
            point_rc = replace(rc, bath=bath, conditions=conditions)
            for current_density in currents:
                row: dict[str, Any] = {
                    "T_C": temperature,
                    "fe_M": fe_concentration,
                    "j_mA_cm2": current_density,
                    "corner_thermal_checked": _corner(
                        temperature, fe_concentration, current_density,
                        temperatures, concentrations, thermal_j,
                    ),
                }
                try:
                    op = physics.solve_at_j(current_density)
                    verdicts = _point_verdicts(point_rc, op)
                    row.update({
                        "current_efficiency": op.current_efficiency,
                        "V_cell": op.V_cell,
                        "specific_energy_kWh_t": op.specific_energy_kWh_t,
                        "transport_margin": verdicts["transport_limit"]["value"],
                        "deposition_rate_um_hr": op.deposition_rate_um_hr,
                        "transport_converged": op.transport_converged,
                        "precipitation_active": op.precipitation_active,
                    })
                    if row["corner_thermal_checked"]:
                        thermal = thermal_balance(point_rc, op)
                        verdicts["thermal"] = {
                            "value": thermal["steady_state_T_C"],
                            "pass": thermal["verdict"]["pass"],
                        }
                        row["thermal"] = thermal
                    row["verdicts"] = verdicts
                    row["all_pass"] = bool(all(item["pass"] for item in verdicts.values()))
                except (RuntimeError, ValueError) as exc:
                    row.update({
                        "solver_error": str(exc),
                        "verdicts": {name: {"value": None, "pass": False} for name in _TARGET_NAMES},
                        "all_pass": False,
                    })
                rows.append(row)

    return {
        "flag": SCREENING_FLAG,
        "not_gate_evidence": True,
        "grid": {"T_C": temperatures, "fe_M": concentrations, "j_mA_cm2": currents},
        "results": rows,
        "n_total": len(rows),
        "n_usable": sum(row["all_pass"] for row in rows),
        "thermal_corner_count": sum(row["corner_thermal_checked"] for row in rows),
        "thermal_j_mA_cm2": thermal_j,
        "_reference_cell": rc,
    }


def usable_fraction(result: dict[str, Any]) -> float:
    """Return the fraction of swept points passing all applicable L0 targets."""
    total = result["n_total"]
    return float(result["n_usable"] / total) if total else 0.0


def margins(result: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Summarize dimensionless target headroom over the usable region.

    For two-sided targets, headroom is the smaller distance to either bound.
    Thermal margins use the usable, thermally inspected corner records; the
    thermal target is intentionally evaluated only at those four worst-j bath
    corners, not inferred for the remaining points.
    """
    # Preserve the source cell internally so this summary always uses the
    # exact #32 target bundle supplied to ``sweep_window``.
    rc: ReferenceCell = result["_reference_cell"]
    collected: dict[str, list[float]] = {name: [] for name in _TARGET_NAMES}
    for row in result["results"]:
        if not row["all_pass"]:
            continue
        values = {name: verdict["value"] for name, verdict in row["verdicts"].items()}
        if "thermal" not in values:
            values["thermal"] = None
        for name, value in values.items():
            if value is not None:
                collected[name].append(_margin_values(rc, {**values, "thermal": value}).get(name, 0.0))
    return {
        name: {
            "median": float(median(values)) if values else float("nan"),
            "min": float(min(values)) if values else float("nan"),
        }
        for name, values in collected.items()
    }


def reference_is_interior(result: dict[str, Any], rc: ReferenceCell) -> tuple[bool, dict[str, float]]:
    """Test whether the #32 reference point has strict headroom to every bound."""
    solved = solve_reference(rc)
    physics = CellPhysics(rc.bath, rc.geometry, rc.conditions)
    op = physics.solve_at_j(solved["current_density_mA_cm2"])
    verdicts = _point_verdicts(rc, op)
    thermal = thermal_balance(rc, op)
    values = {name: verdict["value"] for name, verdict in verdicts.items()}
    values["thermal"] = thermal["steady_state_T_C"]
    point_margins = _margin_values(rc, values)
    return all(value > 0.0 for value in point_margins.values()), point_margins


def window_boundary(result: dict[str, Any]) -> dict[str, tuple[str, float] | None]:
    """Locate each target's closest failed grid point and its driving axis.

    Distance is normalized to the supplied sweep range around the #32 reference
    point.  The reported axis is the coordinate with the largest normalized
    departure, making the result an operator-readable first-tripped envelope,
    rather than an arbitrary row-order failure.
    """
    rc: ReferenceCell = result["_reference_cell"]
    reference_j = solve_reference(rc)["current_density_mA_cm2"]
    reference = {"T_C": rc.conditions.temperature_C, "fe_M": rc.bath.c_FeSO4_M, "j_mA_cm2": reference_j}
    grid = result["grid"]

    def distance_and_axis(row: dict[str, Any]) -> tuple[float, str]:
        changes = {
            axis: abs(row[axis] - reference[axis]) / max(max(grid[axis]) - min(grid[axis]), 1.0)
            for axis in reference
        }
        axis = max(changes, key=changes.get)
        return sum(value * value for value in changes.values()), axis

    boundaries: dict[str, tuple[str, float] | None] = {}
    for target in _TARGET_NAMES:
        failures = [
            row for row in result["results"]
            if target in row["verdicts"] and not row["verdicts"][target]["pass"]
        ]
        if not failures:
            boundaries[target] = None
            continue
        failed = min(failures, key=lambda row: distance_and_axis(row)[0])
        axis = distance_and_axis(failed)[1]
        boundaries[target] = (axis, float(failed[axis]))
    return boundaries


def main() -> str:
    """Render a reproducible, human-readable L0 operating-window report."""
    rc = reference_cell()
    reference_j = solve_reference(rc)["current_density_mA_cm2"]
    result = sweep_window(
        rc,
        t_grid=(40.0, 50.0, 60.0),
        fe_grid=(0.75, 1.0, 1.25),
        j_grid=(30.0, reference_j, 140.0, 180.0, 240.0),
    )
    summary = margins(result)
    interior, reference_margins = reference_is_interior(result, rc)
    boundaries = window_boundary(result)
    lines = [
        "=" * 78,
        "OPERATING-WINDOW THEORY-CONFIDENCE SCREEN",
        "Level-0 screening — unvalidated (L0); NOT gate evidence.",
        "Gates are measurement-only (models/process_gates.py). No real lab data.",
        "=" * 78,
        f"Grid: T={result['grid']['T_C']} °C; Fe(II)={result['grid']['fe_M']} M;",
        f"      j={tuple(round(value, 1) for value in result['grid']['j_mA_cm2'])} mA/cm²",
        f"Usable fraction: {result['n_usable']}/{result['n_total']} = {usable_fraction(result):.1%}",
        f"Cooled thermal checks: {result['thermal_corner_count']} extreme bath corners "
        f"at j={result['thermal_j_mA_cm2']:.1f} mA/cm².",
        "",
        "Usable-region margin (median / closest-to-failure):",
    ]
    for target in _TARGET_NAMES:
        lines.append(f"  {target:18s} {summary[target]['median']:+.3f} / {summary[target]['min']:+.3f}")
    lines.append("")
    lines.append(f"Reference point interior to every bound: {'PASS' if interior else 'FAIL'}")
    for target in _TARGET_NAMES:
        lines.append(f"  reference {target:12s} margin {reference_margins[target]:+.3f}")
    lines.append("")
    lines.append("First-tripped boundary (axis, grid value; None means no failure sampled):")
    for target, boundary in boundaries.items():
        lines.append(f"  {target:18s} {boundary}")
    lines.append("")
    lines.append("Closest route approach to each bound is the usable-region minimum margin above.")
    return "\n".join(lines)


if __name__ == "__main__":
    print(main())
