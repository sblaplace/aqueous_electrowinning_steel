"""Joint operating-point optimizer for aqueous iron electrowinning.

Performs a deterministic scan across current densities (j) and physical lever sets
(electrode gap, membrane resistance, contact resistance) to determine:
1. Does any physically defensible combination passes the energy gate (<= 4,000 kWh/t Fe)?
2. What is the cost-optimal operating point (minimizing production-scale LCOFe)?
3. How measured contact resistance (from the protocol in Deliverable A) impacts the optimum.

Level-0 screening prediction; NOT gate evidence.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .cell_physics import CellPhysics
from .economics_from_physics import (
    ReferenceCell,
    reference_cell as _economics_reference_cell,
    _cost_stack,
    FE_TARGET_MIN,
    SPECIFIC_ENERGY_TARGET_MAX_KWH_T,
    TRANSPORT_LIMIT_MARGIN_MIN,
)
from .contact_resistance_protocol import expected_contact_resistance_range

SCREENING_FLAG = "unvalidated (L0)"

_WINDOW_CACHE: dict[tuple[Any, ...], tuple[dict[str, Any], ...]] = {}


def reference_cell() -> ReferenceCell:
    """Return the RC-1 reference configuration."""
    return _economics_reference_cell()


def _cell_cache_key(cell: ReferenceCell, contact_resistance_ohm_m2: float | None) -> tuple[Any, ...]:
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
        cell.targets.fe_min,
        cell.targets.specific_energy_max_kWh_t,
        cell.targets.transport_limit_margin_min,
        contact_resistance_ohm_m2,
    )


def solve_window(
    cell: ReferenceCell | None = None,
    *,
    contact_resistance_ohm_m2: float | None = None,
) -> list[dict[str, Any]]:
    """Perform a deterministic grid scan over j and physical levers.

    Levers scanned:
      - j_mA_cm2: [150.0, 300.0]
      - interelectrode_gap_m: [1.5e-3, 3.0e-3]
      - membrane_area_resistance_ohm_m2: [3.0e-4]
      - contact_resistance_ohm_m2: supplied override or [1.0e-4, 5.0e-4]
    """
    if cell is None:
        cell = reference_cell()

    cache_key = _cell_cache_key(cell, contact_resistance_ohm_m2)
    cached = _WINDOW_CACHE.get(cache_key)
    if cached is not None:
        return [dict(row) for row in cached]

    j_values = (150.0, 300.0)
    gap_values = (1.5e-3, 3.0e-3)
    mem_values = (3.0e-4,)

    if contact_resistance_ohm_m2 is not None:
        contact_values = (float(contact_resistance_ohm_m2),)
    else:
        contact_values = (1.0e-4, 5.0e-4)

    rows: list[dict[str, Any]] = []

    for j in j_values:
        for gap in gap_values:
            for mem in mem_values:
                for contact in contact_values:
                    geom = replace(
                        cell.geometry,
                        interelectrode_gap_m=gap,
                        membrane_area_resistance_ohm_m2=mem,
                        contact_resistance_ohm_m2=contact,
                    )
                    mod_cell = replace(cell, geometry=geom, interelectrode_gap_m=gap)

                    physics = CellPhysics(mod_cell.bath, mod_cell.geometry, mod_cell.conditions)

                    try:
                        point = physics.solve_at_j(j)
                        transport_limit = point.transport_limit_mA_cm2
                        transport_margin = transport_limit / j
                        valid_transport = bool(transport_margin > TRANSPORT_LIMIT_MARGIN_MIN)

                        if valid_transport:
                            fe = point.current_efficiency
                            v_cell = point.V_cell
                            energy = point.specific_energy_kWh_t
                            cost_res = _cost_stack(
                                mod_cell, j, fe, v_cell, electricity_price_kWh=0.04
                            )
                            lcofe = cost_res["LCOFe_usd_per_t"]
                            valid = True
                        else:
                            fe = point.current_efficiency
                            v_cell = point.V_cell
                            energy = point.specific_energy_kWh_t
                            lcofe = None
                            valid = False
                    except (ValueError, RuntimeError):
                        valid = False
                        fe = 0.0
                        v_cell = 0.0
                        energy = 0.0
                        transport_limit = 0.0
                        transport_margin = 0.0
                        lcofe = None

                    rows.append({
                        "j_mA_cm2": j,
                        "interelectrode_gap_m": gap,
                        "membrane_area_resistance_ohm_m2": mem,
                        "contact_resistance_ohm_m2": contact,
                        "V_cell": v_cell,
                        "FE": fe,
                        "specific_energy_kWh_t": energy,
                        "transport_limit_mA_cm2": transport_limit,
                        "transport_margin": transport_margin,
                        "LCOFe_usd_per_t": lcofe,
                        "valid": bool(valid),
                        "flag": SCREENING_FLAG,
                    })

    _WINDOW_CACHE[cache_key] = tuple(dict(r) for r in rows)
    return [dict(row) for row in rows]


def energy_gate_reachable(
    cell: ReferenceCell | None = None,
    *,
    contact_resistance_ohm_m2: float | None = None,
) -> dict[str, Any]:
    """Check if any physically defensible combination passes the energy gate (<= 4,000 kWh/t)."""
    window = solve_window(cell, contact_resistance_ohm_m2=contact_resistance_ohm_m2)
    valid_rows = [r for r in window if r["valid"] and r["FE"] >= FE_TARGET_MIN]

    if not valid_rows:
        return {
            "reachable": False,
            "min_energy_kWh_t": float("inf"),
            "best_combination": None,
            "verdict": "No valid operational points found in the scanned window meeting FE >= 70%.",
            "flag": SCREENING_FLAG,
        }

    min_energy_row = min(valid_rows, key=lambda r: r["specific_energy_kWh_t"])
    min_energy = min_energy_row["specific_energy_kWh_t"]
    reachable = bool(min_energy <= SPECIFIC_ENERGY_TARGET_MAX_KWH_T)

    if reachable:
        verdict = (
            f"Energy gate IS reachable: minimum specific energy is {min_energy:.1f} kWh/t Fe "
            f"(<= {SPECIFIC_ENERGY_TARGET_MAX_KWH_T:.0f} kWh/t threshold)."
        )
    else:
        verdict = (
            f"Energy gate is NOT reachable: minimum specific energy achieved across the L0 joint space "
            f"is {min_energy:.1f} kWh/t Fe, which exceeds the {SPECIFIC_ENERGY_TARGET_MAX_KWH_T:.0f} kWh/t threshold."
        )

    return {
        "reachable": reachable,
        "min_energy_kWh_t": min_energy,
        "best_combination": min_energy_row,
        "verdict": verdict,
        "flag": SCREENING_FLAG,
    }


def best_operating_point(
    cell: ReferenceCell | None = None,
    *,
    contact_resistance_ohm_m2: float | None = None,
) -> dict[str, Any]:
    """Return the single best feasible operating point (optimizing LCOFe among valid points)."""
    window = solve_window(cell, contact_resistance_ohm_m2=contact_resistance_ohm_m2)
    valid_rows = [
        r for r in window
        if r["valid"] and r["FE"] >= FE_TARGET_MIN and r["LCOFe_usd_per_t"] is not None
    ]

    if not valid_rows:
        valid_rows = [r for r in window if r["valid"]]
        if not valid_rows:
            return dict(window[0]) if window else {}

    best_row = min(valid_rows, key=lambda r: r["LCOFe_usd_per_t"])
    energy_pass = bool(best_row["specific_energy_kWh_t"] <= SPECIFIC_ENERGY_TARGET_MAX_KWH_T)

    result = dict(best_row)
    result["energy_gate_pass"] = energy_pass
    result["verdict"] = (
        f"Cost-optimal operating point at {best_row['j_mA_cm2']:.1f} mA/cm²: "
        f"LCOFe = ${best_row['LCOFe_usd_per_t']:.0f}/t, "
        f"Energy = {best_row['specific_energy_kWh_t']:.1f} kWh/t "
        f"(Energy gate pass <=4000: {energy_pass})."
    )
    return result


def sweep_table(
    cell: ReferenceCell | None = None,
    *,
    contact_resistance_ohm_m2: float | None = None,
) -> list[dict[str, Any]]:
    """Return cost-minimizing and energy-minimizing rows across current densities."""
    window = solve_window(cell, contact_resistance_ohm_m2=contact_resistance_ohm_m2)
    valid_rows = [r for r in window if r["valid"] and r["LCOFe_usd_per_t"] is not None]

    if not valid_rows:
        return []

    j_groups: dict[float, list[dict[str, Any]]] = {}
    for r in valid_rows:
        j_groups.setdefault(r["j_mA_cm2"], []).append(r)

    summary_rows = []
    for j, rows in sorted(j_groups.items()):
        min_lcofe_row = min(rows, key=lambda r: r["LCOFe_usd_per_t"])
        min_energy_row = min(rows, key=lambda r: r["specific_energy_kWh_t"])
        summary_rows.append({
            "j_mA_cm2": j,
            "cost_optimal_LCOFe": min_lcofe_row["LCOFe_usd_per_t"],
            "cost_optimal_gap_m": min_lcofe_row["interelectrode_gap_m"],
            "cost_optimal_contact_ohm_m2": min_lcofe_row["contact_resistance_ohm_m2"],
            "energy_optimal_kWh_t": min_energy_row["specific_energy_kWh_t"],
            "energy_optimal_gap_m": min_energy_row["interelectrode_gap_m"],
            "energy_optimal_contact_ohm_m2": min_energy_row["contact_resistance_ohm_m2"],
            "flag": SCREENING_FLAG,
        })

    return summary_rows


def main() -> None:
    """Print the operating-point optimizer report."""
    print("JOINT OPERATING-POINT OPTIMIZER REPORT — L0 screening")
    print(f"Status: {SCREENING_FLAG}. NOT gate evidence; gates are measurement-only.")
    print("-" * 60)

    cell = reference_cell()
    reach = energy_gate_reachable(cell)
    print(f"Reachability Verdict:\n  {reach['verdict']} [{SCREENING_FLAG}]")
    if reach["best_combination"]:
        bc = reach["best_combination"]
        print(f"  Best energy combination: j={bc['j_mA_cm2']} mA/cm², "
              f"gap={bc['interelectrode_gap_m']*1e3:.1f} mm, "
              f"contact={bc['contact_resistance_ohm_m2']*1e4:.1f}e-4 Ω·m² → "
              f"Energy = {bc['specific_energy_kWh_t']:.1f} kWh/t [{SCREENING_FLAG}]")

    print("\nBest Cost-Optimal Operating Point:")
    best = best_operating_point(cell)
    print(f"  {best['verdict']} [{SCREENING_FLAG}]")

    print("\nSweep Table (Cost vs Energy across Current Densities):")
    table = sweep_table(cell)
    print("  j (mA/cm²) | LCOFe ($/t) | Min Energy (kWh/t) | Cost-Opt Gap/Contact | Energy-Opt Gap/Contact")
    for r in table:
        print(f"  {r['j_mA_cm2']:10.1f} | ${r['cost_optimal_LCOFe']:9.0f} | {r['energy_optimal_kWh_t']:18.1f} | "
              f"{r['cost_optimal_gap_m']*1e3:3.1f}mm / {r['cost_optimal_contact_ohm_m2']*1e4:3.1f}e-4 | "
              f"{r['energy_optimal_gap_m']*1e3:3.1f}mm / {r['energy_optimal_contact_ohm_m2']*1e4:3.1f}e-4 "
              f"[{SCREENING_FLAG}]")

    proto_typical = expected_contact_resistance_range()["typical"]["value"]
    print(f"\nDemonstrating A → B Wiring (using protocol expected typical contact resistance = {proto_typical:.1e} Ω·m²):")
    best_wired = best_operating_point(cell, contact_resistance_ohm_m2=proto_typical)
    print(f"  Wired Best Point: {best_wired['verdict']} [{SCREENING_FLAG}]")


if __name__ == "__main__":
    main()
