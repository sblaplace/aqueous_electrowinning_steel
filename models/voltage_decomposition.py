"""Level-0 decomposition of the RC-1 cell voltage into actionable levers.

This module is a transparent screening calculation.  It reuses the RC-1
configuration and the coupled ``CellPhysics`` solver from
``economics_from_physics``; it does not add measurements or gate evidence.
The energy-gate verdicts reported here are model predictions only.  Gates are
measurement-only and are implemented in ``models/process_gates.py``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .cell_physics import CellPhysics
from .economics_from_physics import ReferenceCell
from .economics_from_physics import reference_cell as _economics_reference_cell
from .electrochemistry import specific_energy_kWh_per_t

SCREENING_FLAG = "unvalidated (L0)"
ENERGY_GATE_KWH_T = 4000.0
REFERENCE_J_MA_CM2 = 300.0

# These are deliberately explicit screening scenarios, not fitted values.
# Each proposed value has a physical basis recorded in ``lever_sensitivity``.
_CONTACT_RESISTANCE_PROPOSED = 1.0e-4
_MEMBRANE_RESISTANCE_PROPOSED = 1.5e-4
_GAP_PROPOSED_M = 1.5e-3
_BUBBLE_FRACTION_PROPOSED = 0.05
_ETA_ANODE_PROPOSED_V = 0.30
_SENSITIVITY_CACHE: dict[tuple[Any, ...], list[dict[str, float | bool | str]]] = {}


def reference_cell() -> ReferenceCell:
    """Return the RC-1 reference configuration used by the economics screen."""
    return _economics_reference_cell()


def _physics(cell: ReferenceCell) -> CellPhysics:
    """Construct the coupled solver without changing the supplied cell."""
    return CellPhysics(cell.bath, cell.geometry, cell.conditions)


def _voltage_model_at(
    cell: ReferenceCell,
    j_mA_cm2: float,
    *,
    eta_anode_override_V: float | None = None,
) -> tuple[CellPhysics, Any, Any]:
    """Solve the cell and rebuild the exact voltage model used by the solve.

    ``CellPhysics.solve_at_j`` intentionally returns a compact operating point,
    while its voltage model is an internal construction.  The public
    decomposition needs the component properties on that model, so this
    helper uses the solver's existing rounded cathode-overpotential hand-off,
    restores the exact value from the returned operating-point voltage, and
    calls ``CellPhysics._build_voltage_model`` as the existing model prescribes.
    It therefore does not reimplement the transport equations or run them a
    second time.
    """
    if j_mA_cm2 <= 0.0:
        raise ValueError("Applied current density must be positive.")

    physics = _physics(cell)
    point = physics.solve_at_j(j_mA_cm2)

    # CellPhysics exposes the cathode overpotential in its existing rounded
    # decomposition.  Use that interface to rebuild the voltage model, then
    # recover the unrounded value from the exact OperatingPoint voltage.  This
    # avoids reimplementing or running the transport solve a second time while
    # preserving exact closure rather than propagating display rounding.
    eta_cathode = float(point.V_decomposition["η_cathode (V)"])
    voltage_model = physics._build_voltage_model(j_mA_cm2, eta_cathode)
    eta_cathode_exact = point.V_cell - (
        voltage_model.E_thermodynamic
        + voltage_model._effective_eta_anode
        + voltage_model._total_ir_drop
    )
    voltage_model.eta_cathode = eta_cathode_exact

    if eta_anode_override_V is not None:
        # The RC-1 model uses CellVoltageModel's fixed-OER fallback rather than
        # an AnodeKinetics object.  Setting that existing model input is the
        # explicit catalyst-improvement scenario; no FE is changed.
        voltage_model.eta_anode = eta_anode_override_V

    # Read the rounded summary as an audit of the model interface.  The
    # returned values below use the model's unrounded properties so closure is
    # exact rather than limited by summary()'s display rounding.
    summary = voltage_model.summary()
    required_summary_keys = {
        "E_thermodynamic (V)",
        "η_cathode (V)",
        "η_anode (V)",
        "iR drop (V)",
        "V_cell (V)",
    }
    if not required_summary_keys.issubset(summary):
        raise RuntimeError("CellVoltageModel.summary() is missing a voltage component.")

    return physics, point, voltage_model


def _breakdown_from_model(point: Any, voltage_model: Any) -> dict[str, float]:
    """Read exact component properties and enforce voltage closure."""
    breakdown: dict[str, float] = {
        "E_thermodynamic": float(voltage_model.E_thermodynamic),
        "eta_cathode": float(voltage_model.eta_cathode),
        "eta_anode": float(voltage_model._effective_eta_anode),
        "IR_electrolyte": float(voltage_model.IR_electrolyte),
        "IR_membrane": float(voltage_model.IR_membrane),
        "IR_contacts": float(voltage_model.IR_contacts),
        "IR_total": float(voltage_model._total_ir_drop),
        "V_cell": float(voltage_model.V_cell),
        "FE": float(point.current_efficiency),
        "specific_energy_kWh_t": float(
            specific_energy_kWh_per_t(voltage_model.V_cell, point.current_efficiency)
        ),
    }

    ohmic_parts = (
        breakdown["IR_electrolyte"]
        + breakdown["IR_membrane"]
        + breakdown["IR_contacts"]
    )
    components = (
        breakdown["E_thermodynamic"]
        + breakdown["eta_cathode"]
        + breakdown["eta_anode"]
        + ohmic_parts
    )
    if abs(ohmic_parts - breakdown["IR_total"]) > 1e-6:
        raise RuntimeError("Ohmic voltage components do not sum to IR_total.")
    if abs(components - breakdown["V_cell"]) > 1e-6:
        raise RuntimeError("Voltage decomposition does not close to V_cell.")

    # Keep the identity explicit at the module boundary.  CellPhysics already
    # reports this same value, but using the shared helper prevents drift.
    if breakdown["specific_energy_kWh_t"] <= 0.0:
        raise RuntimeError("Specific energy must be positive at a positive operating point.")
    return breakdown


def decompose_at(
    cell: ReferenceCell,
    j_mA_cm2: float = REFERENCE_J_MA_CM2,
) -> dict[str, float]:
    """Return the closed RC-1 voltage/energy decomposition at one current."""
    _, point, voltage_model = _voltage_model_at(cell, j_mA_cm2)
    return _breakdown_from_model(point, voltage_model)


def _scenario_cell(cell: ReferenceCell, **geometry_updates: float) -> ReferenceCell:
    """Copy a reference cell with one or more explicit geometry changes."""
    geometry = replace(cell.geometry, **geometry_updates)
    return replace(
        cell,
        geometry=geometry,
        interelectrode_gap_m=geometry.interelectrode_gap_m,
    )


def _gate_pass(energy_kWh_t: float, cell: ReferenceCell) -> bool:
    """Evaluate only the published energy threshold, as a screening verdict."""
    return energy_kWh_t <= cell.targets.specific_energy_max_kWh_t


def _cell_cache_key(cell: ReferenceCell, j_mA_cm2: float) -> tuple[Any, ...]:
    """Make a scalar key so repeated report views do not rerun the solver."""
    bath = cell.bath
    geometry = cell.geometry
    conditions = cell.conditions
    return (
        cell.name,
        cell.cathode_area_cm2,
        cell.interelectrode_gap_m,
        bath.c_FeSO4_M,
        bath.c_Na2SO4_M,
        bath.c_H2SO4_M,
        bath.c_H3BO3_M,
        bath.pH,
        geometry.interelectrode_gap_m,
        geometry.membrane,
        geometry.membrane_area_resistance_ohm_m2,
        geometry.contact_resistance_ohm_m2,
        geometry.anode_bubble_fraction,
        conditions.temperature_C,
        conditions.boundary_layer_m,
        conditions.flow_regime,
        conditions.fe_i0,
        conditions.her_i0,
        conditions.fe_tafel_V,
        conditions.her_tafel_V,
        cell.targets.fe_min,
        cell.targets.specific_energy_max_kWh_t,
        cell.targets.transport_limit_margin_min,
        j_mA_cm2,
    )


def lever_sensitivity(
    cell: ReferenceCell,
    j_mA_cm2: float = REFERENCE_J_MA_CM2,
) -> list[dict[str, float | bool | str]]:
    """Rank explicit single-lever improvements by volts saved.

    FE is held at the baseline ``CellPhysics`` value in every row.  The
    proposed changes are transparent engineering screening scenarios:

    * contact resistance: 5.0e-4 -> 1.0e-4 ohm m², by contact/current-
      collector and busbar optimization;
    * membrane area resistance: 3.0e-4 -> 1.5e-4 ohm m², by a thinner or
      lower-resistance separator;
    * electrode gap: 3.0 -> 1.5 mm, by shortening the electrolyte path;
    * anode bubble fraction: 0.10 -> 0.05, by degassing or improving anolyte
      gas release;
    * anode overpotential: 0.40 -> 0.30 V, by a preferred OER catalyst.

    No rows stack improvements.  A gate result is therefore the honest result
    of that one stated change, not a combined scenario.
    """
    cache_key = _cell_cache_key(cell, j_mA_cm2)
    cached_rows = _SENSITIVITY_CACHE.get(cache_key)
    if cached_rows is not None:
        return [dict(row) for row in cached_rows]

    baseline = decompose_at(cell, j_mA_cm2)
    baseline_fe = float(baseline["FE"])
    baseline_v = float(baseline["V_cell"])

    # The order here is descriptive only; rank_levers performs the numerical
    # sort.  Current values are read from the supplied RC-1 object.
    scenarios: tuple[dict[str, Any], ...] = (
        {
            "lever": "contact resistance",
            "parameter": "contact_resistance_ohm_m2",
            "current_value": float(cell.geometry.contact_resistance_ohm_m2),
            "proposed_value": _CONTACT_RESISTANCE_PROPOSED,
            "basis": "bolt, busbar, and current-collector contact optimization",
            "unit": "ohm m²",
            "cell": _scenario_cell(
                cell, contact_resistance_ohm_m2=_CONTACT_RESISTANCE_PROPOSED
            ),
            "eta_anode_override_V": None,
        },
        {
            "lever": "membrane area resistance",
            "parameter": "membrane_area_resistance_ohm_m2",
            "current_value": float(cell.geometry.membrane_area_resistance_ohm_m2),
            "proposed_value": _MEMBRANE_RESISTANCE_PROPOSED,
            "basis": "thinner or lower-area-resistance separator",
            "unit": "ohm m²",
            "cell": _scenario_cell(
                cell, membrane_area_resistance_ohm_m2=_MEMBRANE_RESISTANCE_PROPOSED
            ),
            "eta_anode_override_V": None,
        },
        {
            "lever": "electrode gap",
            "parameter": "interelectrode_gap_m",
            "current_value": float(cell.geometry.interelectrode_gap_m),
            "proposed_value": _GAP_PROPOSED_M,
            "basis": "halve the electrolyte path, subject to the RC-1 geometry constraint",
            "unit": "m",
            "cell": _scenario_cell(cell, interelectrode_gap_m=_GAP_PROPOSED_M),
            "eta_anode_override_V": None,
        },
        {
            "lever": "anode bubble fraction",
            "parameter": "anode_bubble_fraction",
            "current_value": float(cell.geometry.anode_bubble_fraction),
            "proposed_value": _BUBBLE_FRACTION_PROPOSED,
            "basis": "degas or improve higher-conductivity anolyte gas release",
            "unit": "fraction",
            "cell": _scenario_cell(
                cell, anode_bubble_fraction=_BUBBLE_FRACTION_PROPOSED
            ),
            "eta_anode_override_V": None,
        },
        {
            "lever": "anode overpotential",
            "parameter": "eta_anode",
            "current_value": float(baseline["eta_anode"]),
            "proposed_value": _ETA_ANODE_PROPOSED_V,
            "basis": "preferred OER catalyst with lower fixed-model overpotential",
            "unit": "V",
            "cell": cell,
            "eta_anode_override_V": _ETA_ANODE_PROPOSED_V,
        },
    )

    rows: list[dict[str, float | bool | str]] = []
    for scenario in scenarios:
        _, scenario_point, scenario_model = _voltage_model_at(
            scenario["cell"],
            j_mA_cm2,
            eta_anode_override_V=scenario["eta_anode_override_V"],
        )
        scenario_v = float(scenario_model.V_cell)
        delta_v = baseline_v - scenario_v
        if delta_v <= 0.0:
            raise RuntimeError(
                f"Proposed {scenario['lever']} change did not reduce V_cell: "
                f"{delta_v:.9f} V."
            )
        scenario_fe = float(scenario_point.current_efficiency)
        if abs(scenario_fe - baseline_fe) > 1e-12:
            raise RuntimeError(
                f"FE changed in the {scenario['lever']} voltage-only scenario."
            )
        energy_after = float(specific_energy_kWh_per_t(scenario_v, baseline_fe))
        rows.append(
            {
                "lever": scenario["lever"],
                "parameter": scenario["parameter"],
                "current_value": scenario["current_value"],
                "proposed_value": scenario["proposed_value"],
                "delta_V": delta_v,
                "V_after": scenario_v,
                "FE_after": baseline_fe,
                "energy_before": float(
                    specific_energy_kWh_per_t(baseline_v, baseline_fe)
                ),
                "energy_after": energy_after,
                "gate_pass_after": _gate_pass(energy_after, cell),
                "basis": scenario["basis"],
                "unit": scenario["unit"],
                "flag": SCREENING_FLAG,
            }
        )
    _SENSITIVITY_CACHE[cache_key] = [dict(row) for row in rows]
    return [dict(row) for row in rows]


def rank_levers(
    cell: ReferenceCell,
    j_mA_cm2: float = REFERENCE_J_MA_CM2,
) -> list[str]:
    """Return lever names sorted from largest to smallest volts saved."""
    rows = lever_sensitivity(cell, j_mA_cm2)
    return [row["lever"] for row in sorted(rows, key=lambda row: row["delta_V"], reverse=True)]


def buy_next_measurement(
    cell: ReferenceCell,
    j_mA_cm2: float = REFERENCE_J_MA_CM2,
) -> dict[str, float | int | str]:
    """Recommend one inexpensive measurement for the highest-ranked lever."""
    rows = lever_sensitivity(cell, j_mA_cm2)
    ranked = sorted(rows, key=lambda row: row["delta_V"], reverse=True)
    top = ranked[0]
    lever = str(top["lever"])
    measurement_names = {
        "contact resistance": "measured terminal-to-electrode contact resistance",
        "membrane area resistance": "measured membrane area resistance",
        "electrode gap": "measured electrode-to-membrane gap and electrolyte path",
        "anode bubble fraction": "measured anolyte bubble fraction or conductivity under load",
        "anode overpotential": "measured anode polarization/overpotential at the operating current",
    }
    measurement = measurement_names.get(lever, f"measurement of the {lever} lever")
    reason = (
        f"The {lever} lever ranks first in the unvalidated (L0) decomposition and "
        f"saves {float(top['delta_V']):.3f} V in the stated single-lever scenario; "
        f"{measurement} directly tests the largest predicted voltage contribution."
    )
    return {
        "measurement": measurement,
        "recommendation": f"Buy {measurement} next.",
        "lever": lever,
        "rank": 1,
        "predicted_delta_V": float(top["delta_V"]),
        "reason": reason,
        "flag": SCREENING_FLAG,
        "gate_note": "This recommendation is unvalidated (L0), not gate evidence; gates are measurement-only.",
    }


def _print_row(label: str, value: float | bool | str, *, unit: str = "") -> None:
    """Print one report value with its Level-0 status marker."""
    suffix = f" {unit}" if unit else ""
    print(f"{label}: {value}{suffix} [{SCREENING_FLAG}]")


def main() -> None:
    """Print the decomposition, single-lever table, ranking, and recommendation."""
    cell = reference_cell()
    j_mA_cm2 = REFERENCE_J_MA_CM2
    decomposition = decompose_at(cell, j_mA_cm2)
    rows = lever_sensitivity(cell, j_mA_cm2)
    recommendation = buy_next_measurement(cell, j_mA_cm2)

    print("VOLTAGE DECOMPOSITION — RC-1 reference cell")
    print(f"All predicted values are [{SCREENING_FLAG}].")
    print("NOT gate evidence; gates are measurement-only (models/process_gates.py).")
    _print_row("Current density", f"{j_mA_cm2:.1f}", unit="mA/cm²")
    print("\nVoltage and energy breakdown")
    for key in (
        "E_thermodynamic",
        "eta_cathode",
        "eta_anode",
        "IR_electrolyte",
        "IR_membrane",
        "IR_contacts",
        "IR_total",
        "V_cell",
        "FE",
        "specific_energy_kWh_t",
    ):
        unit = "V" if key not in {"FE", "specific_energy_kWh_t"} else ""
        value = decomposition[key]
        if key == "FE":
            value = f"{float(value):.5f}"
            unit = "fraction"
        elif key == "specific_energy_kWh_t":
            value = f"{float(value):.1f}"
            unit = "kWh/t Fe"
        else:
            value = f"{float(value):.6f}"
        _print_row(key, value, unit=unit)

    print("\nSingle-lever sensitivity (no improvements stacked)")
    headers = "lever | current -> proposed | delta_V | energy_after | gate_pass_after"
    print(headers)
    for row in rows:
        print(
            f"{row['lever']} | {row['current_value']} -> {row['proposed_value']} "
            f"| {float(row['delta_V']):.6f} V "
            f"| {float(row['energy_after']):.1f} kWh/t Fe "
            f"| {row['gate_pass_after']} [{SCREENING_FLAG}]"
        )
        print(f"  basis: {row['basis']}")

    ranked = rank_levers(cell, j_mA_cm2)
    print("\nRanked levers (largest volts saved first)")
    for index, lever in enumerate(ranked, start=1):
        print(f"{index}. {lever} [{SCREENING_FLAG}]")

    print("\nBuy-next-measurement recommendation")
    print(f"{recommendation['recommendation']} [{SCREENING_FLAG}]")
    print(recommendation["reason"])
    print(recommendation["gate_note"])


if __name__ == "__main__":
    main()
