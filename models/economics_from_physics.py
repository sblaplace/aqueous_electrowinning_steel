"""Level-0 economics derived from the coupled cell-physics model.

This module is deliberately a transparent screening calculation, not a plant
forecast or gate evidence.  Its FE and voltage originate in ``CellPhysics``;
the only deliberately assumed FE is the labelled 0.90 contrast case retained
from ``run_technoeconomic.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .cell_physics import BathRecipe, CellGeometry, CellPhysics, ProcessConditions
from .electrochemistry import specific_energy_kWh_per_t
from .technoeconomic import CAPEXModel, ElectrolyzerParams, LevelizedCost, OPEXModel

# PROGRAM_SUMMARY.md decision-grade kill criterion at j >= 300 mA/cm².
FE_TARGET_MIN = 0.70
# PROGRAM_SUMMARY.md decision-grade kill criterion: net DC <= 4,000 kWh/t Fe.
SPECIFIC_ENERGY_TARGET_MAX_KWH_T = 4000.0
# The same criterion requires the migration-enhanced transport limit exceed j.
TRANSPORT_LIMIT_MARGIN_MIN = 1.0
REFERENCE_J_MA_CM2 = 300.0
HARD_CODED_BASELINE_FE = 0.90
FE_UNCERTAINTY_ABS = 0.05  # NEXT_STEPS acceptance/measurement tolerance: +/-5 pp.
VOLTAGE_UNCERTAINTY_V = 0.10  # Transparent L0 voltage sensitivity span.
SCREENING_FLAG = "unvalidated (L0)"


@dataclass(frozen=True)
class ScreeningTargets:
    """Published L0 decision thresholds; none is a measurement acceptance."""

    fe_min: float = FE_TARGET_MIN
    specific_energy_max_kWh_t: float = SPECIFIC_ENERGY_TARGET_MAX_KWH_T
    transport_limit_margin_min: float = TRANSPORT_LIMIT_MARGIN_MIN


@dataclass(frozen=True)
class ReferenceCell:
    """Explicit immutable configuration for the named divided-cell screen."""

    name: str
    cathode_area_cm2: float
    interelectrode_gap_m: float
    bath: BathRecipe
    geometry: CellGeometry
    conditions: ProcessConditions
    targets: ScreeningTargets = ScreeningTargets()


def reference_cell() -> ReferenceCell:
    """Build the explicit RC-1-like divided-cell configuration (L0 only)."""
    return ReferenceCell(
        name="physics-economics-reference-divided-cell-v1",
        cathode_area_cm2=200.0,
        interelectrode_gap_m=0.02,
        bath=BathRecipe(
            c_FeSO4_M=1.0,
            c_Na2SO4_M=0.5,
            c_H2SO4_M=0.01,
            c_H3BO3_M=0.4,
            pH=2.0,
        ),
        geometry=CellGeometry(
            interelectrode_gap_m=0.02,
            membrane=True,
            membrane_area_resistance_ohm_m2=3.0e-4,
            contact_resistance_ohm_m2=5.0e-4,
            anode_bubble_fraction=0.10,
        ),
        conditions=ProcessConditions(
            temperature_C=50.0,
            boundary_layer_m=50e-6,
            flow_regime="moderate",
        ),
    )


def _physics(cell: ReferenceCell) -> CellPhysics:
    return CellPhysics(cell.bath, cell.geometry, cell.conditions)


def _validated_point(cell: ReferenceCell, j_mA_cm2: float):
    """Solve once and reject currents at/beyond the model transport limit."""
    if j_mA_cm2 <= 0:
        raise ValueError("Applied current density must be positive.")
    point = _physics(cell).solve_at_j(j_mA_cm2)
    if point.transport_limit_mA_cm2 <= j_mA_cm2:
        raise ValueError(
            f"Invalid operating point: {j_mA_cm2:.1f} mA/cm² is at/beyond "
            f"the transport limit {point.transport_limit_mA_cm2:.1f} mA/cm²."
        )
    return point


def derived_operating_point(
    reference_cell: ReferenceCell, j_mA_cm2: float = REFERENCE_J_MA_CM2
) -> dict[str, Any]:
    """Return FE, voltage and target verdicts from the coupled physics solve."""
    point = _validated_point(reference_cell, j_mA_cm2)
    targets = reference_cell.targets
    transport_margin = point.transport_limit_mA_cm2 / j_mA_cm2
    energy_from_identity = specific_energy_kWh_per_t(point.V_cell, point.current_efficiency)
    if abs(point.specific_energy_kWh_t - energy_from_identity) > 1e-9:
        raise RuntimeError("Cell physics energy is inconsistent with the electrochemistry identity.")
    verdicts = {
        "FE": {"threshold": f">= {targets.fe_min:.2f}", "pass": point.current_efficiency >= targets.fe_min},
        "specific_energy": {
            "threshold": f"<= {targets.specific_energy_max_kWh_t:.0f} kWh/t Fe",
            "pass": point.specific_energy_kWh_t <= targets.specific_energy_max_kWh_t,
        },
        "transport_limit": {
            "threshold": f"> {j_mA_cm2:.1f} mA/cm²",
            "pass": transport_margin > targets.transport_limit_margin_min,
        },
    }
    return {
        "j_mA_cm2": j_mA_cm2,
        "current_efficiency": point.current_efficiency,
        "V_cell": point.V_cell,
        "specific_energy_kWh_t": point.specific_energy_kWh_t,
        "deposition_rate_um_hr": point.deposition_rate_um_hr,
        "transport_limit_mA_cm2": point.transport_limit_mA_cm2,
        "transport_margin": transport_margin,
        "verdicts": verdicts,
        "all_targets_pass": all(item["pass"] for item in verdicts.values()),
        "flag": SCREENING_FLAG,
    }


def _cost_stack(
    cell: ReferenceCell, j_mA_cm2: float, fe: float, voltage: float, electricity_price_kWh: float
) -> dict[str, float]:
    params = ElectrolyzerParams(
        current_density_mA_cm2=j_mA_cm2,
        current_efficiency=fe,
        cell_voltage=voltage,
        temperature_C=cell.conditions.temperature_C,
        electrode_area_m2=cell.cathode_area_cm2 / 10_000.0,
        n_cells=100,
        electrolyte_type="acidic",
    )
    n_stacks = 10
    capex = CAPEXModel().estimate(params, n_stacks=n_stacks)
    opex = OPEXModel(electricity_price_kWh=electricity_price_kWh).estimate(
        params, capex["Total CAPEX ($)"], n_stacks=n_stacks
    )
    lcofe = LevelizedCost().calculate(
        capex["Total CAPEX ($)"], opex["Total OPEX ($/yr)"], capex["Annual capacity (t/yr)"]
    )
    return {
        "LCOFe_usd_per_t": float(lcofe["LCOFe ($/t Fe)"]),
        "annual_capacity_t_yr": float(capex["Annual capacity (t/yr)"]),
    }


def physics_lcofe(
    reference_cell: ReferenceCell,
    j_mA_cm2: float = REFERENCE_J_MA_CM2,
    *,
    electricity_price_kWh: float = 0.04,
) -> dict[str, Any]:
    """Feed physics-derived FE/V into the existing CAPEX/OPEX/LCOFe stack."""
    derived = derived_operating_point(reference_cell, j_mA_cm2)
    costs = _cost_stack(
        reference_cell, j_mA_cm2, derived["current_efficiency"], derived["V_cell"], electricity_price_kWh
    )
    baseline = _cost_stack(
        reference_cell, j_mA_cm2, HARD_CODED_BASELINE_FE, derived["V_cell"], electricity_price_kWh
    )
    gap = costs["LCOFe_usd_per_t"] - baseline["LCOFe_usd_per_t"]
    return {
        "j_mA_cm2": j_mA_cm2,
        "FE": derived["current_efficiency"],
        "V_cell": derived["V_cell"],
        "specific_energy_kWh_t": derived["specific_energy_kWh_t"],
        **costs,
        "hardcoded_0p90_LCOFe_usd_per_t": baseline["LCOFe_usd_per_t"],
        "LCOFe_gap_usd_per_t": gap,
        "LCOFe_gap_sign": "higher" if gap > 0 else "lower" if gap < 0 else "neutral",
        "flag": SCREENING_FLAG,
    }


def sweep_economics(reference_cell: ReferenceCell) -> list[dict[str, Any]]:
    """Map physics-derived economics from 50 to 500 mA/cm².

    ``CellPhysics.sweep`` supplies the feasible portion.  Requested points it
    cannot solve are retained as explicit invalid rows rather than being
    mistaken for economic predictions.
    """
    requested_j = tuple(float(j) for j in range(50, 501, 50))
    window = _physics(reference_cell).sweep(j_min=50.0, j_max=500.0, n_points=10)
    valid_points = {round(point.j_mA_cm2, 8): point for point in window.points}
    rows = []
    for j in requested_j:
        point = valid_points.get(round(j, 8))
        if point is None or point.transport_limit_mA_cm2 <= point.j_mA_cm2:
            rows.append({"j_mA_cm2": j, "invalid": True, "flag": SCREENING_FLAG})
            continue
        cost = physics_lcofe(reference_cell, j)
        rows.append({
            "j_mA_cm2": j,
            "FE": point.current_efficiency,
            "V_cell": point.V_cell,
            "specific_energy_kWh_t": point.specific_energy_kWh_t,
            "LCOFe_usd_per_t": cost["LCOFe_usd_per_t"],
            "invalid": False,
            "flag": SCREENING_FLAG,
        })
    return rows


def uncertainty_propagation(
    reference_cell: ReferenceCell, j_mA_cm2: float = REFERENCE_J_MA_CM2
) -> dict[str, Any]:
    """Enumerate +/-5 pp FE and +/-0.10 V uncertainty around the L0 prediction."""
    base = physics_lcofe(reference_cell, j_mA_cm2)
    base_fe, base_v = base["FE"], base["V_cell"]
    values = []
    for fe in (max(0.01, base_fe - FE_UNCERTAINTY_ABS), base_fe + FE_UNCERTAINTY_ABS):
        for voltage in (base_v - VOLTAGE_UNCERTAINTY_V, base_v + VOLTAGE_UNCERTAINTY_V):
            values.append(_cost_stack(reference_cell, j_mA_cm2, fe, voltage, 0.04)["LCOFe_usd_per_t"])
    fe_shift = max(
        abs(_cost_stack(reference_cell, j_mA_cm2, max(0.01, base_fe - FE_UNCERTAINTY_ABS), base_v, 0.04)["LCOFe_usd_per_t"] - base["LCOFe_usd_per_t"]),
        abs(_cost_stack(reference_cell, j_mA_cm2, base_fe + FE_UNCERTAINTY_ABS, base_v, 0.04)["LCOFe_usd_per_t"] - base["LCOFe_usd_per_t"]),
    )
    voltage_shift = max(
        abs(_cost_stack(reference_cell, j_mA_cm2, base_fe, base_v + VOLTAGE_UNCERTAINTY_V, 0.04)["LCOFe_usd_per_t"] - base["LCOFe_usd_per_t"]),
        abs(_cost_stack(reference_cell, j_mA_cm2, base_fe, base_v - VOLTAGE_UNCERTAINTY_V, 0.04)["LCOFe_usd_per_t"] - base["LCOFe_usd_per_t"]),
    )
    driver = "FE" if fe_shift >= voltage_shift else "V_cell"
    return {
        "base_LCOFe_usd_per_t": base["LCOFe_usd_per_t"],
        "LCOFe_min_usd_per_t": min(values),
        "LCOFe_max_usd_per_t": max(values),
        "FE_spread_pp": 5.0,
        "V_cell_spread_V": VOLTAGE_UNCERTAINTY_V,
        "largest_driver": driver,
        "measurement_recommendation": f"Measure {driver} first; it moves LCOFe most in this L0 sensitivity.",
        "flag": SCREENING_FLAG,
    }


def main() -> None:
    """Print a compact, explicitly non-gate-evidence screening report."""
    cell = reference_cell()
    point = derived_operating_point(cell)
    cost = physics_lcofe(cell)
    uncertainty = uncertainty_propagation(cell)
    print("PHYSICS-DERIVED ECONOMICS — unvalidated (L0)")
    print("NOT gate evidence; gates are measurement-only (models/process_gates.py).")
    print(f"300 mA/cm²: FE {point['current_efficiency']:.1%}, V_cell {point['V_cell']:.3f} V, "
          f"energy {point['specific_energy_kWh_t']:.0f} kWh/t Fe [unvalidated (L0)]")
    print(f"Targets pass: {point['all_targets_pass']} [unvalidated (L0)]")
    print(f"LCOFe: ${cost['LCOFe_usd_per_t']:.0f}/t; versus hardcoded FE=0.90: "
          f"{cost['LCOFe_gap_sign']} by ${cost['LCOFe_gap_usd_per_t']:+.0f}/t [unvalidated (L0)]")
    print(f"LCOFe range: ${uncertainty['LCOFe_min_usd_per_t']:.0f}–${uncertainty['LCOFe_max_usd_per_t']:.0f}/t; "
          f"buy measurement: {uncertainty['largest_driver']} [unvalidated (L0)]")


if __name__ == "__main__":
    main()
