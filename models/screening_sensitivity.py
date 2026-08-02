"""Level-0 screening uncertainty budget for the reference cell.

This module is deliberately a thin, one-at-a-time screen on top of the
reference-cell model in :mod:`models.theory_confidence` and the operating
surface in :mod:`models.operating_window_confidence`.  It is not a calibrated
uncertainty model: every range below is a transparent *screening factor* around
a repository default, not a measured prior.

The output is therefore useful for deciding what to measure first, but it is
not evidence for a process gate.  Gates remain measurement-only in
``models.process_gates`` and there is no real laboratory data in this
repository.
"""

from __future__ import annotations

from dataclasses import replace
from math import isfinite
from typing import Any

from .operating_window_confidence import margins, sweep_window
from .theory_confidence import ReferenceCell, reference_cell, solve_reference

SCREENING_FLAG = "unvalidated (L0)"
NOT_GATE_EVIDENCE = True

# A full #33 surface is intentionally not repeated for every one-at-a-time
# perturbation.  This bounded current slice keeps the budget reproducible and
# CI-tractable while still testing the influence over more than one operating
# point.  The #33 function is used rather than a second window implementation.
WINDOW_J_FACTORS = (0.75, 1.0, 1.25)

_TARGET_NAMES = (
    "fe",
    "v_cell",
    "specific_energy",
    "transport_limit",
    "deposition_rate",
)

# Aliases make the public perturbation helper readable without adding duplicate
# rows to the budget table.  The canonical names are the actual dataclass or
# constructor names used by the reference-cell stack.
_ALIASES = {
    "fe_exchange_current_A_m2": "fe_i0",
    "her_exchange_current_A_m2": "her_i0",
    "fe_tafel_slope_V_decade": "fe_tafel_V",
    "her_tafel_slope_V_decade": "her_tafel_V",
    "boundary_layer_thickness_m": "boundary_layer_m",
    "support_concentration_M": "c_Na2SO4_M",
    "fe_concentration_M": "c_FeSO4_M",
    "membrane_R_ohm_m2": "membrane_area_resistance_ohm_m2",
    "gap_m": "interelectrode_gap_m",
}


def _entry(
    value: float,
    factor_range: tuple[float, float],
    source: str,
    maps_to: str,
    unit: str,
) -> dict[str, Any]:
    """Build one self-describing screening-range record."""
    return {
        "value": float(value),
        "range": factor_range,
        "source": source,
        "maps_to": maps_to,
        "unit": unit,
    }


def define_ranges() -> dict[str, dict[str, Any]]:
    """Return the explicit one-at-a-time screening ranges.

    ``range`` is a pair of multiplicative factors ``(low, high)`` applied to
    ``value`` by :func:`perturb_and_solve`.  The factors are intentionally
    visible instead of being hidden in a sampler.  The provenance strings point
    to the repository parameter/default that supplies each central value and
    explicitly say when the interval is only a screening assumption.
    """
    rc = reference_cell()
    return {
        "fe_i0": _entry(
            rc.conditions.fe_i0,
            (0.10, 10.0),
            "models/cell_physics.py: ProcessConditions.fe_i0 default 1e-2 A/m²; "
            "factor interval is an L0 screening assumption, not a measurement.",
            "conditions.fe_i0 -> CellPhysics._build_transport -> NernstPlanckFilm.fe_i0",
            "A/m²",
        ),
        "her_i0": _entry(
            rc.conditions.her_i0,
            (0.10, 10.0),
            "models/cell_physics.py: ProcessConditions.her_i0 default 1e-6 A/m²; "
            "factor interval is an L0 screening assumption, not a measurement.",
            "conditions.her_i0 -> CellPhysics._build_transport -> NernstPlanckFilm.her_i0",
            "A/m²",
        ),
        "fe_tafel_V": _entry(
            rc.conditions.fe_tafel_V,
            (0.75, 1.25),
            "models/cell_physics.py: ProcessConditions.fe_tafel_V default 0.120 V/decade; "
            "the ±25% interval is an L0 mechanism-screen, not fitted data.",
            "conditions.fe_tafel_V -> CellPhysics._build_transport -> NernstPlanckFilm.fe_tafel_V",
            "V/decade",
        ),
        "her_tafel_V": _entry(
            rc.conditions.her_tafel_V,
            (0.75, 1.25),
            "models/cell_physics.py: ProcessConditions.her_tafel_V default 0.140 V/decade; "
            "the ±25% interval is an L0 mechanism-screen, not fitted data.",
            "conditions.her_tafel_V -> CellPhysics._build_transport -> NernstPlanckFilm.her_tafel_V",
            "V/decade",
        ),
        "boundary_layer_m": _entry(
            rc.conditions.boundary_layer_m,
            (0.40, 2.0),
            "models/cell_physics.py: ProcessConditions.boundary_layer_m default 50 µm; "
            "the 20–100 µm factor screen follows the repo's still/stirred test scale, not a lab fit.",
            "conditions.boundary_layer_m -> CellPhysics._build_transport -> NernstPlanckFilm.boundary_layer_m",
            "m",
        ),
        "c_FeSO4_M": _entry(
            rc.bath.c_FeSO4_M,
            (0.50, 1.50),
            "models/theory_confidence.py: reference_cell bath c_FeSO4_M=1.0 M and "
            "models/operating_window_confidence.py Fe grid; interval is an L0 recipe screen.",
            "bath.c_FeSO4_M -> CellPhysics._build_transport(fe_conc_M)",
            "M",
        ),
        "c_Na2SO4_M": _entry(
            rc.bath.c_Na2SO4_M,
            (0.50, 2.0),
            "models/cell_physics.py: BathRecipe.c_Na2SO4_M and "
            "models/transport.py: NernstPlanckFilm.support_conc_M; interval screens migration.",
            "bath.c_Na2SO4_M -> CellPhysics._build_transport(support_conc_M)",
            "M",
        ),
        "membrane_area_resistance_ohm_m2": _entry(
            rc.geometry.membrane_area_resistance_ohm_m2,
            (0.50, 2.0),
            "models/theory_confidence.py: reference membrane area resistance 3e-4 Ω m²; "
            "models/electrochemistry.py: MembraneModel.R_membrane_ohm_m2 documents the membrane term.",
            "geometry.membrane_area_resistance_ohm_m2 -> CellVoltageModel.membrane",
            "Ω m²",
        ),
        "interelectrode_gap_m": _entry(
            rc.geometry.interelectrode_gap_m,
            (0.50, 2.0),
            "models/theory_confidence.py: reference gap 0.02 m; "
            "models/electrochemistry.py: CellVoltageModel.interelectrode_gap_m controls electrolyte IR.",
            "geometry.interelectrode_gap_m -> CellVoltageModel.IR_electrolyte",
            "m",
        ),
        "contact_resistance_ohm_m2": _entry(
            rc.geometry.contact_resistance_ohm_m2,
            (0.50, 2.0),
            "models/theory_confidence.py: reference contact resistance 5e-4 Ω m²; "
            "models/electrochemistry.py: CellVoltageModel.contact_resistance_ohm_m2.",
            "geometry.contact_resistance_ohm_m2 -> CellVoltageModel.IR_contacts",
            "Ω m²",
        ),
        "anode_bubble_fraction": _entry(
            rc.geometry.anode_bubble_fraction,
            (0.50, 2.0),
            "models/theory_confidence.py: reference anode bubble fraction 0.10; "
            "models/electrochemistry.py: bubble_fraction reduces effective conductivity.",
            "geometry.anode_bubble_fraction -> CellVoltageModel.IR_electrolyte",
            "fraction",
        ),
        "temperature_C": _entry(
            rc.conditions.temperature_C,
            (0.80, 1.20),
            "models/theory_confidence.py: reference temperature 50 °C; "
            "models/operating_window_confidence.py: 40–60 °C screen and "
            "models/electrochemistry.py: conductivity_S_m(T) dependence.",
            "conditions.temperature_C -> speciation, transport and conductivity_S_m(T)",
            "°C",
        ),
    }


def _canonical_param(param: str) -> str:
    """Return a canonical range key, accepting documented aliases."""
    canonical = _ALIASES.get(param, param)
    if canonical not in define_ranges():
        raise KeyError(f"Unknown screening parameter: {param}")
    return canonical


def _current_value(rc: ReferenceCell, param: str) -> float:
    """Read one coefficient from its owning reference-cell dataclass."""
    if param in {
        "fe_i0",
        "her_i0",
        "fe_tafel_V",
        "her_tafel_V",
        "boundary_layer_m",
        "temperature_C",
    }:
        return float(getattr(rc.conditions, param))
    if param in {"c_FeSO4_M", "c_Na2SO4_M"}:
        return float(getattr(rc.bath, param))
    return float(getattr(rc.geometry, param))


def _replace_parameter(rc: ReferenceCell, param: str, value: float) -> ReferenceCell:
    """Rebuild ``rc`` with one dataclass-carried coefficient changed."""
    if param in {
        "fe_i0",
        "her_i0",
        "fe_tafel_V",
        "her_tafel_V",
        "boundary_layer_m",
        "temperature_C",
    }:
        return replace(rc, conditions=replace(rc.conditions, **{param: value}))
    if param in {"c_FeSO4_M", "c_Na2SO4_M"}:
        return replace(rc, bath=replace(rc.bath, **{param: value}))
    return replace(rc, geometry=replace(rc.geometry, **{param: value}))


def _failed_result(
    rc: ReferenceCell,
    param: str,
    factor: float,
    value: float,
    error: Exception,
) -> dict[str, Any]:
    """Represent an infeasible perturbation as an explicit failed screen."""
    return {
        "flag": SCREENING_FLAG,
        "not_gate_evidence": NOT_GATE_EVIDENCE,
        "parameter": param,
        "factor": float(factor),
        "perturbed_value": value,
        "current_efficiency": None,
        "V_cell": None,
        "specific_energy_kWh_t": None,
        "verdicts": {
            name: {"value": None, "pass": False} for name in _TARGET_NAMES
        },
        "all_pass": False,
        "solver_error": str(error),
        "_reference_cell": rc,
    }


def perturb_and_solve(rc: ReferenceCell, param: str, factor: float) -> dict[str, Any]:
    """Multiply one screening coefficient by ``factor`` and solve the stack.

    The reference cell is rebuilt with :func:`dataclasses.replace`; no global
    model defaults are changed.  The returned mapping has the same operating
    point and verdict fields as :func:`models.theory_confidence.solve_reference`
    plus perturbation metadata.  A solver-infeasible extreme is retained as an
    explicit failed result instead of being silently discarded.
    """
    canonical = _canonical_param(param)
    if factor <= 0.0:
        raise ValueError("screening perturbation factor must be positive")
    value = _current_value(rc, canonical) * float(factor)
    if value <= 0.0:
        raise ValueError("perturbed screening coefficient must be positive")
    perturbed_rc = _replace_parameter(rc, canonical, value)
    try:
        result = solve_reference(perturbed_rc)
    except (RuntimeError, ValueError) as error:
        return _failed_result(perturbed_rc, canonical, factor, value, error)

    result = dict(result)
    result.update(
        {
            "not_gate_evidence": NOT_GATE_EVIDENCE,
            "parameter": canonical,
            "factor": float(factor),
            "perturbed_value": value,
            "_reference_cell": perturbed_rc,
        }
    )
    return result


def _value(result: dict[str, Any], key: str) -> float:
    """Read a finite result value, using NaN for an infeasible extreme."""
    raw = result.get(key)
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return float("nan")
    return number if isfinite(number) else float("nan")


def _point_risk(
    central: dict[str, Any], extreme: dict[str, Any], rc: ReferenceCell
) -> float:
    """Score which endpoint is the more decision-threatening one."""
    if not extreme.get("all_pass", False):
        failure_bonus = 10.0
    else:
        failure_bonus = 0.0
    fe_margin = max(central["current_efficiency"] - rc.targets.fe_min, 1e-9)
    v_margin = min(
        central["V_cell"] - rc.targets.v_cell_min,
        rc.targets.v_cell_max - central["V_cell"],
    )
    energy_margin = max(
        rc.targets.specific_energy_max_kWh_t - central["specific_energy_kWh_t"],
        1e-9,
    )
    terms = [
        abs(_value(extreme, "current_efficiency") - central["current_efficiency"])
        / fe_margin,
        abs(_value(extreme, "V_cell") - central["V_cell"]) / max(v_margin, 1e-9),
        abs(_value(extreme, "specific_energy_kWh_t") - central["specific_energy_kWh_t"])
        / energy_margin,
    ]
    finite_terms = [term for term in terms if isfinite(term)]
    return failure_bonus + max(finite_terms, default=0.0)


def _window_margin(rc: ReferenceCell, reference_j: float) -> tuple[float, dict[str, Any]]:
    """Use #33 on a bounded current slice and return its closest margin."""
    if not isfinite(reference_j):
        return 0.0, {}
    low = max(5.0, reference_j * WINDOW_J_FACTORS[0])
    high = reference_j * WINDOW_J_FACTORS[-1]
    window = sweep_window(
        rc,
        t_grid=(float(rc.conditions.temperature_C),),
        fe_grid=(float(rc.bath.c_FeSO4_M),),
        j_grid=(low, float(reference_j), high),
    )
    summary = margins(window)
    finite_margins = [
        values["min"]
        for values in summary.values()
        if isfinite(values["min"])
    ]
    # #33's margins intentionally summarize usable rows only.  Zero here means
    # no usable row remained at this endpoint, which is the most conservative
    # budget result and is distinct from an omitted sweep.
    minimum = min(finite_margins, default=0.0)
    return float(minimum), summary


def _spread(central: float, low: float, high: float) -> float:
    """Return the largest absolute OAT displacement from the centre."""
    candidates = [abs(value - central) for value in (low, high) if isfinite(value)]
    return float(max(candidates, default=float("nan")))


def _priority_score(entry: dict[str, Any]) -> float:
    """Combine output displacement, window risk, and verdict-flip potential."""
    delta_fe = abs(float(entry.get("delta_fe", 0.0)))
    delta_v = abs(float(entry.get("delta_v_cell", 0.0)))
    delta_energy = abs(float(entry.get("delta_specific_energy", 0.0)))
    margin = float(entry.get("min_margin_across_window", 0.0))
    window_risk = 1.0 / (1.0 + max(margin, 0.0))
    flip_bonus = 100.0 if entry.get("flips_pass_at_reference", False) else 0.0
    return (
        flip_bonus
        + delta_fe / 0.80
        + delta_v / 3.50
        + delta_energy / 6000.0
        + window_risk
    )


def sensitivity_profile(rc: ReferenceCell) -> dict[str, dict[str, Any]]:
    """Compute the bounded one-at-a-time FE/V/energy uncertainty profile.

    The FE, voltage and energy deltas are maximum absolute displacements from
    the central reference solve at the low/high factors.  The window margin is
    measured with :func:`sweep_window`/:func:`margins` at the more threatening
    endpoint, over a three-point current slice at the reference bath.  Extra
    low/high fields preserve the direction and make the report auditable while
    the required summary fields remain compact.
    """
    ranges = define_ranges()
    central = solve_reference(rc)
    profile: dict[str, dict[str, Any]] = {}
    for param, definition in ranges.items():
        low_factor, high_factor = definition["range"]
        low = perturb_and_solve(rc, param, low_factor)
        high = perturb_and_solve(rc, param, high_factor)
        low_fe = _value(low, "current_efficiency")
        high_fe = _value(high, "current_efficiency")
        low_v = _value(low, "V_cell")
        high_v = _value(high, "V_cell")
        low_energy = _value(low, "specific_energy_kWh_t")
        high_energy = _value(high, "specific_energy_kWh_t")
        flips = bool(
            not low.get("all_pass", False) or not high.get("all_pass", False)
        )
        endpoint = low if _point_risk(central, low, rc) >= _point_risk(central, high, rc) else high
        endpoint_factor = low_factor if endpoint is low else high_factor
        endpoint_rc = endpoint["_reference_cell"]
        window_min, window_summary = _window_margin(
            endpoint_rc,
            _value(endpoint, "current_density_mA_cm2"),
        )
        entry: dict[str, Any] = {
            "delta_fe": _spread(central["current_efficiency"], low_fe, high_fe),
            "delta_v_cell": _spread(central["V_cell"], low_v, high_v),
            "delta_specific_energy": _spread(
                central["specific_energy_kWh_t"], low_energy, high_energy
            ),
            "flips_pass_at_reference": flips,
            "min_margin_across_window": window_min,
            "low_factor": float(low_factor),
            "high_factor": float(high_factor),
            "low_fe": low_fe,
            "high_fe": high_fe,
            "low_v_cell": low_v,
            "high_v_cell": high_v,
            "low_specific_energy": low_energy,
            "high_specific_energy": high_energy,
            "window_factor": float(endpoint_factor),
            "window_margins": window_summary,
        }
        entry["influence_score"] = _priority_score(entry)
        profile[param] = entry
    return profile


def ranked_calibration_priority(profile: dict[str, dict[str, Any]]) -> list[str]:
    """Return deterministic highest-threat-first calibration priorities.

    A verdict flip is intentionally dominant (a 100-point bonus); remaining
    ties are broken by normalized FE, voltage, energy and window-margin
    influence, then by parameter name.  The final alphabetical tie-break makes
    the result independent of dictionary insertion order.
    """
    scored = []
    for param, entry in profile.items():
        # Recompute from the public summary fields so callers can inspect or
        # update a profile without relying on a cached private score.
        score = _priority_score(entry)
        scored.append((score, param))
    return [param for _score, param in sorted(scored, key=lambda item: (-item[0], item[1]))]


def _pm(value: float, unit: str = "") -> str:
    """Format a plus/minus table value."""
    if not isfinite(value):
        return "n/a"
    return f"±{value:.3g}{unit}"


def main() -> str:
    """Render the reproducible L0 screening-uncertainty report."""
    rc = reference_cell()
    central = solve_reference(rc)
    ranges = define_ranges()
    profile = sensitivity_profile(rc)
    priority = ranked_calibration_priority(profile)

    lines = [
        "=" * 100,
        "REFERENCE-CELL SCREENING-UNCERTAINTY / SENSITIVITY BUDGET",
        "Level-0 synthetic screening — unvalidated (L0); NOT gate evidence.",
        "Gates are measurement-only (models/process_gates.py); no real lab data are used.",
        "=" * 100,
        (
            f"Central reference: j={central['current_density_mA_cm2']:.1f} mA/cm², "
            f"FE={central['current_efficiency']:.4f}, V_cell={central['V_cell']:.3f} V, "
            f"specific energy={central['specific_energy_kWh_t']:.0f} kWh/t; "
            f"all targets={'PASS' if central['all_pass'] else 'FAIL'}."
        ),
        "",
        "OAT influence (factor range; maximum absolute displacement from central):",
        "parameter                                      range          ±FE       ±V_cell       ±energy     flips?  window min",
    ]
    for param in ranges:
        definition = ranges[param]
        entry = profile[param]
        lo, hi = definition["range"]
        lines.append(
            f"{param:44s} {lo:.3g}–{hi:.3g}      "
            f"{_pm(entry['delta_fe']):>8s}  {_pm(entry['delta_v_cell'], ' V'):>12s}  "
            f"{_pm(entry['delta_specific_energy'], ' kWh/t'):>16s}  "
            f"{'YES' if entry['flips_pass_at_reference'] else 'no':>5s}  "
            f"{entry['min_margin_across_window']:+.3f}"
        )

    lines.extend(["", "Ranked calibration priority (measure / calibrate this first):"])
    for index, param in enumerate(priority, start=1):
        entry = profile[param]
        lines.append(
            f"  {index:2d}. {param}  score={entry['influence_score']:.3f}  "
            f"flip={'YES' if entry['flips_pass_at_reference'] else 'no'}"
        )
    if priority:
        dominant = priority[0]
        dominant_entry = profile[dominant]
        if dominant_entry["flips_pass_at_reference"]:
            output = "FE/deposition verdict"
        else:
            output = max(
                (
                    (dominant_entry["delta_fe"], "FE"),
                    (dominant_entry["delta_v_cell"], "V_cell"),
                    (dominant_entry["delta_specific_energy"], "specific energy"),
                ),
                key=lambda item: item[0] if isfinite(item[0]) else float("-inf"),
            )[1]
        lines.extend(
            [
                "",
                f"DOMINANT REMAINING UNKNOWN: {dominant} — closest to flipping the {output};",
                "calibrate this mapped input before spending the first run elsewhere.",
                "This headline is a screening priority, not a validated process conclusion.",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(main())
