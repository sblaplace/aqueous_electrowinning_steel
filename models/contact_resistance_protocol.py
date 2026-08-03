"""Contact-resistance measurement protocol for the RC-1 build.

Level-0 screening protocol and prior expectation (not a fitted model or gate evidence).
Turns the recommendation of #39 into an actionable measurement plan and evaluates
its downstream decision impact via the voltage decomposition and physics model.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .economics_from_physics import ReferenceCell
from .economics_from_physics import reference_cell as _economics_reference_cell
from .voltage_decomposition import decompose_at

SCREENING_FLAG = "unvalidated (L0)"
ENERGY_GATE_KWH_T = 4000.0


def reference_cell() -> ReferenceCell:
    """Return the RC-1 reference configuration."""
    return _economics_reference_cell()


def protocol_overview() -> dict[str, Any]:
    """Structured measurement protocol plan for the RC-1 build."""
    return {
        "title": "RC-1 Terminal-to-Electrode Contact Resistance Measurement Protocol",
        "status": SCREENING_FLAG,
        "method": "4-wire (Kelvin) DC current injection and differential voltage drop measurement",
        "target_interfaces": [
            "Busbar-to-current collector interface",
            "Current collector-to-electrode coupon interface",
            "Aggregate terminal-to-electrode joint (total path)",
        ],
        "expected_signal": (
            "Apply known stable DC currents (1.0 A to 10.0 A) through each joint "
            "while measuring the potential drop (mV to µV) across calibrated probe spans."
        ),
        "recorded_units": (
            "Area-normalized specific resistance in Ω·m² (ohms·m²), "
            "calculated as (V_drop / I_current) * Area_m2, directly compatible "
            "with CellGeometry.contact_resistance_ohm_m2."
        ),
        "note": "Measurement-only plan; does not constitute gate evidence.",
    }


def expected_contact_resistance_range() -> dict[str, Any]:
    """Explicit Level-0 expected prior range for the RC-1 bench build."""
    return {
        "status": "expected, not measured",
        "flag": SCREENING_FLAG,
        "unit": "Ω·m²",
        "min": {
            "value": 1.0e-4,
            "basis": "Optimized mechanical bolting with silver-filled conductive epoxy or gold plating on copper busbars.",
        },
        "typical": {
            "value": 5.0e-4,
            "basis": "RC-1 standard bolted copper busbar and titanium/steel coupon clamp baseline (current model default).",
        },
        "max": {
            "value": 1.0e-3,
            "basis": "Unoptimized mechanical contact with native oxide films or minor clamping non-uniformity.",
        },
    }


def impact_if_measured(cell: ReferenceCell | None = None, j_mA_cm2: float = 300.0) -> dict[str, Any]:
    """Evaluate decision consequence (volts saved and specific energy) if contact resistance is measured at min, typical, and max expected values."""
    if cell is None:
        cell = reference_cell()

    base_decomp = decompose_at(cell, j_mA_cm2)
    base_v = base_decomp["V_cell"]
    base_energy = base_decomp["specific_energy_kWh_t"]

    range_data = expected_contact_resistance_range()
    results = {}

    r_base = cell.geometry.contact_resistance_ohm_m2

    for key in ("min", "typical", "max"):
        r_val = range_data[key]["value"]
        new_geom = replace(cell.geometry, contact_resistance_ohm_m2=r_val)
        new_cell = replace(cell, geometry=new_geom)

        decomp = decompose_at(new_cell, j_mA_cm2)
        v_cell = decomp["V_cell"]
        energy = decomp["specific_energy_kWh_t"]
        delta_v = base_v - v_cell
        pass_gate = energy <= ENERGY_GATE_KWH_T

        results[key] = {
            "contact_resistance_ohm_m2": r_val,
            "V_cell": v_cell,
            "delta_V": delta_v,
            "specific_energy_kWh_t": energy,
            "gate_pass": pass_gate,
            "basis": range_data[key]["basis"],
        }

    return {
        "j_mA_cm2": j_mA_cm2,
        "baseline_contact_resistance_ohm_m2": r_base,
        "baseline_V_cell": base_v,
        "baseline_specific_energy_kWh_t": base_energy,
        "scenarios": results,
        "flag": SCREENING_FLAG,
        "gate_threshold_kWh_t": ENERGY_GATE_KWH_T,
    }


def instrument_requirements() -> list[str]:
    """Conservative, build-realistic instrumentation requirements for the RC-1 build."""
    return [
        "Constant-current DC power supply (0–10 A range, ripple < 0.1%).",
        "High-precision 6.5-digit digital multimeter or nanovoltmeter for microvolt potential drop measurement.",
        "Kelvin (4-wire) probe leads or dedicated spring-loaded sense pins positioned across joint interfaces.",
        "Calibrated torque wrench and mechanical clamping fixture to standardize contact pressure during testing.",
        "Multi-point fixture to test busbar-to-collector and collector-to-coupon interfaces independently and aggregately.",
        "Statistical replication protocol: minimum 5 replicate measurements per joint interface at ambient and operating temperature (50 °C), reporting mean and standard error.",
    ]


def main() -> None:
    """Print the contact resistance protocol report."""
    print("CONTACT-RESISTANCE MEASUREMENT PROTOCOL — RC-1 Build")
    print(f"Status: {SCREENING_FLAG}. NOT gate evidence; gates are measurement-only.")
    print("-" * 60)

    overview = protocol_overview()
    print(f"Overview Method: {overview['method']}")
    print(f"Recorded Units: {overview['recorded_units']}")

    print("\nExpected Prior Range:")
    rng = expected_contact_resistance_range()
    for k in ("min", "typical", "max"):
        item = rng[k]
        print(f"  {k.upper()}: {item['value']:.1e} Ω·m² — {item['basis']}")

    print("\nDecision Impact Analysis (at 300 mA/cm²):")
    impact = impact_if_measured(j_mA_cm2=300.0)
    for k, sc in impact["scenarios"].items():
        print(f"  {k.upper()} ({sc['contact_resistance_ohm_m2']:.1e} Ω·m²): "
              f"V_cell = {sc['V_cell']:.3f} V (ΔV = {sc['delta_V']:+.3f} V), "
              f"Energy = {sc['specific_energy_kWh_t']:.1f} kWh/t, "
              f"Gate pass (<=4000) = {sc['gate_pass']} [{SCREENING_FLAG}]")

    print("\nInstrument Requirements:")
    for req in instrument_requirements():
        print(f"  - {req}")


if __name__ == "__main__":
    main()
