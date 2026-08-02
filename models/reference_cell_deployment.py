"""Generate RC-1 deployment artifacts from the selected design configuration.

The deployment package translates the synthesized reference-cell geometry into
an operator/procurement-ready P&ID, electrical safety boundary, instrument
schedule, and controlled bill of materials.  It intentionally does not issue
hardware commands; the independent shutdown remains a physical design
requirement.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json

from .reference_cell_design import ReferenceCellConfig

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DeploymentPackage:
    """Versioned RC-1 deployment artifacts derived from selected design data."""

    configuration_id: str
    selected_design: Dict[str, Any]
    instruments: List[Dict[str, str]]
    bom: List[Dict[str, str]]
    markdown: str

    def manifest(self) -> Dict[str, Any]:
        return {
            "configuration_id": self.configuration_id,
            "status": "pre-procurement_deployment_package",
            "selected_design": self.selected_design,
            "instruments": self.instruments,
            "bom": self.bom,
            "required_pre_energization_tests": [
                "dimensional and membrane-cassette inspection",
                "water leak and pressure test",
                "flow verification at 0.10, 0.50, and 1.00 L/min per loop",
                "electrical dummy-load and independent-shutdown proof test",
                "instrument calibration and synchronized-timestamp check",
            ],
        }


def _selected(report: Dict[str, Any]) -> Dict[str, Any]:
    selected = report.get("selected_design")
    if not isinstance(selected, dict):
        raise ValueError("design report lacks selected_design")
    candidate = selected.get("candidate", {})
    operating = selected.get("operating", {})
    for key in ("active_area_cm2", "channel_depth_mm", "flow_L_min"):
        if key not in candidate:
            raise ValueError(f"selected design lacks candidate.{key}")
    for key in ("current_A", "current_density_mA_cm2", "cell_voltage_V"):
        if key not in operating:
            raise ValueError(f"selected design lacks operating.{key}")
    return selected


def instrument_schedule(config: ReferenceCellConfig) -> List[Dict[str, str]]:
    """Return the RC-1 measurement and control-point schedule.

    Tags intentionally align with the existing digital/operating twin where
    possible, allowing the deployment package to become the runtime mapping.
    """
    return [
        {"tag": "TT-101", "service": "catholyte reservoir temperature", "range": "0–100 °C", "accuracy": "±0.5 °C", "location": "TK-101 well", "purpose": "thermal balance / high-temperature trip"},
        {"tag": "TT-201", "service": "anolyte reservoir temperature", "range": "0–100 °C", "accuracy": "±0.5 °C", "location": "TK-201 well", "purpose": "thermal balance / high-temperature trip"},
        {"tag": "TT-102", "service": "catholyte cell-outlet temperature", "range": "0–100 °C", "accuracy": "±0.5 °C", "location": "EC-101 cathode outlet", "purpose": "cell heat-rise measurement"},
        {"tag": "TT-202", "service": "anolyte cell-outlet temperature", "range": "0–100 °C", "accuracy": "±0.5 °C", "location": "EC-101 anode outlet", "purpose": "cell heat-rise measurement"},
        {"tag": "FT-201", "service": "catholyte recirculation flow", "range": "0–1.5 L/min", "accuracy": "±5% reading", "location": "P-101 discharge", "purpose": "flow / transport state"},
        {"tag": "FT-202", "service": "anolyte recirculation flow", "range": "0–1.5 L/min", "accuracy": "±5% reading", "location": "P-201 discharge", "purpose": "flow / transport state"},
        {"tag": "PT-101", "service": "catholyte cell differential pressure", "range": "0–10 kPa", "accuracy": "±1% full scale", "location": "EC-101 cathode inlet/outlet", "purpose": "channel fouling / flow verification"},
        {"tag": "PT-201", "service": "anolyte cell differential pressure", "range": "0–10 kPa", "accuracy": "±1% full scale", "location": "EC-101 anode inlet/outlet", "purpose": "channel fouling / flow verification"},
        {"tag": "pHAT-101", "service": "catholyte pH", "range": "pH 0–7", "accuracy": "±0.05 pH", "location": "TK-101 sample loop", "purpose": "bath control / twin observation"},
        {"tag": "AT-202", "service": "anolyte pH", "range": "pH 0–7", "accuracy": "±0.05 pH", "location": "TK-201 sample loop", "purpose": "anolyte drift"},
        {"tag": "FE2P-101", "service": "catholyte Fe(II) sample analysis", "range": "0–2 M", "accuracy": "±0.02 M target", "location": "SP-101", "purpose": "iron balance / EKF observability"},
        {"tag": "CT-201", "service": "rectifier cathodic current", "range": "0–10 A", "accuracy": "±1% reading", "location": "PS-101 output", "purpose": "charge ledger / current-density state"},
        {"tag": "VT-201", "service": "cell voltage", "range": "0–10 V", "accuracy": "±0.01 V", "location": "EC-101 terminals", "purpose": "energy ledger / voltage model"},
        {"tag": "GT-101", "service": "cathode hydrogen monitor", "range": "0–4% H₂ in air minimum", "accuracy": "alarm before LEL", "location": "high point near exhaust", "purpose": "independent gas safety alarm"},
    ]


def controlled_bom(config: ReferenceCellConfig, selected: Dict[str, Any]) -> List[Dict[str, str]]:
    c = selected["candidate"]
    o = selected["operating"]
    return [
        {"id": "EC-101", "item": "Divided recirculating cell body", "qty": "1", "specification": f"Vertical 50 × 20 mm channels; selected {c['channel_depth_mm']:.1f} mm gap; clamped removable membrane cassette; borosilicate/PP/PVDF/PTFE/FEP wetted path only", "acceptance": "dimensional inspection, dry assembly, water leak test"},
        {"id": "MEM-101", "item": "Cation-exchange membrane", "qty": "3 coupons", "specification": "Nafion N117 initial comparator; 50 × 50 mm nominal wetted cassette; retain lot and hydration history", "acceptance": "visual integrity, wetted area recorded, area-resistance baseline"},
        {"id": "CA-101", "item": "Cathode coupon set", "qty": "10+", "specification": f"316L, 1 mm, masked to {c['active_area_cm2']:.1f} cm²; controlled surface-preparation record", "acceptance": "mass/area traceability and mask verification"},
        {"id": "AN-101", "item": "Anode assembly", "qty": "1 + spare", "specification": "Controlled OER-capable experimental anode, exposed area ≥ cathode area; material/lot fixed per campaign", "acceptance": "area and material recorded; post-run inspection plan"},
        {"id": "TK-101/TK-201", "item": "Covered catholyte/anolyte reservoirs", "qty": "2", "specification": f"1.0 L working volume each; compatible sample, drain, vent, and temperature ports; total nominal inventory {config.total_volume_L:.1f} L", "acceptance": "level marks, leak test, compatible labels"},
        {"id": "P-101/P-201", "item": "Recirculation pump loops", "qty": "2", "specification": "Independently adjustable; compatible wetted materials; verified 0.10–1.00 L/min installed-loop range", "acceptance": "three-point installed flow curve and pressure test"},
        {"id": "HT-101/HT-201", "item": "Thermal control", "qty": "2 zones", "specification": f"50–70 °C qualified range; target {config.target_temperature_C:.0f} °C; independent temperature measurement", "acceptance": "heat-up/hold test and over-temperature trip test"},
        {"id": "PS-101", "item": "DC rectifier", "qty": "1", "specification": f"30 V / 10 A CC/CV bench supply; normal RC-1 duty {o['current_A']:.2f} A, qualified ceiling {config.max_current_A:.2f} A", "acceptance": "dummy-load CC verification and output-off behavior"},
        {"id": "K-101/ESD-101", "item": "Independent DC shutdown chain", "qty": "1", "specification": "hardwired latching emergency stop and contactor/enable interruption; separate from software twin", "acceptance": "documented proof test removes rectifier enable"},
        {"id": "DAQ-101", "item": "Synchronized data acquisition", "qty": "1", "specification": "logs current, voltage, temperature, flow, pressure, pH, and event timestamps; immutable raw export retained", "acceptance": "timestamp and channel-calibration check"},
        {"id": "GX-101", "item": "Vent/exhaust and hydrogen monitoring", "qty": "1", "specification": "separate anode/cathode vents to site-approved exhaust; H₂ monitor/alarm; no sealed gas path", "acceptance": "site ventilation review and alarm functional test"},
        {"id": "CT-101", "item": "Secondary containment", "qty": "1", "specification": "compatible tray sized for complete 2 L inventory plus margin; segregated from electrical equipment", "acceptance": "capacity and drain-path inspection"},
    ]


def _p_and_id(selected: Dict[str, Any]) -> str:
    c = selected["candidate"]
    return f"""```mermaid
flowchart LR
  TK101[TK-101\nCatholyte reservoir\n1 L] --> P101[P-101\nCatholyte pump]
  P101 --> FT201[FT-201]
  FT201 --> PT101[PT-101 ΔP]
  PT101 --> EC101[EC-101\nDivided cell\n{c['active_area_cm2']:.0f} cm² cathode\n{c['channel_depth_mm']:.1f} mm channel]
  EC101 --> TT102[TT-102 outlet]
  TT102 --> TK101
  TK201[TK-201\nAnolyte reservoir\n1 L] --> P201[P-201\nAnolyte pump]
  P201 --> FT202[FT-202]
  FT202 --> PT201[PT-201 ΔP]
  PT201 --> EC101
  EC101 --> TT202[TT-202 outlet]
  TT202 --> TK201
  TK101 --- pH101[pHAT-101 / SP-101\npH and Fe(II)]
  TK201 --- pH201[AT-202\npH]
  EC101 --> G101[Cathode vent → GX-101\nH₂ monitor/exhaust]
  EC101 --> G201[Anode vent → GX-101\nseparate exhaust]
  PS101[PS-101\nDC supply] --> K101[K-101 hardwired DC enable]
  K101 --> EC101
  ESD101[ESD-101 / H₂ alarm] -. independent trip .-> K101
  CT201[CT-201] --- PS101
  VT201[VT-201] --- EC101
```"""


def _wiring() -> str:
    return """```text
AC mains → GFCI/RCD → PS-101 (30 V / 10 A CC/CV)
                              │
                              ├─ low-voltage measurement: CT-201 series current meter
                              ├─ low-voltage measurement: VT-201 directly across EC-101 terminals
                              └─ DC enable path: ESD-101 latching stop → K-101 contactor/enable → EC-101

GT-101 H₂ alarm, ESD-101, and the physical emergency-stop circuit are a diverse,
hardwired safety path.  DAQ-101 and the software operating twin may request a
shutdown but have no authority to replace K-101 or ESD-101.

DAQ-101 records: TT-101/102/201/202, FT-201/202, PT-101/201, pHAT-101,
AT-202, CT-201, VT-201, and GT-101 alarm state on one synchronized time base.
```"""


def render_deployment_markdown(config: ReferenceCellConfig, report: Dict[str, Any]) -> DeploymentPackage:
    """Render the controlled RC-1 P&ID, electrical, BOM, and sensor schedule."""
    selected = _selected(report)
    instruments = instrument_schedule(config)
    bom = controlled_bom(config, selected)
    c, o, h = selected["candidate"], selected["operating"], selected["hydraulics"]

    inst_rows = "\n".join(
        f"| `{x['tag']}` | {x['service']} | {x['range']} | {x['accuracy']} | {x['location']} | {x['purpose']} |"
        for x in instruments
    )
    bom_rows = "\n".join(
        f"| `{x['id']}` | {x['item']} | {x['qty']} | {x['specification']} | {x['acceptance']} |"
        for x in bom
    )
    markdown = f"""# RC-1 deployment package

**Configuration:** `{config.configuration_id}`

**Status:** pre-procurement deployment design

**Source configuration:** `processes/reference_cell_rc1.yaml`

**Source calculation:** `outputs/reference_cell_rc1_design_report.json`

## Selected deployable duty

| Parameter | Selected value |
|---|---:|
| Cathode active area | {c['active_area_cm2']:.1f} cm² |
| Channel depth / electrode gap | {c['channel_depth_mm']:.1f} mm |
| Catholyte and anolyte recirculation | {c['flow_L_min']:.2f} L/min per loop |
| Current / current density | {o['current_A']:.2f} A / {o['current_density_mA_cm2']:.0f} mA/cm² |
| Modeled cell voltage | {o['cell_voltage_V']:.2f} V |
| Modeled FE / deposit rate | {o['faradaic_efficiency']:.1%} / {o['deposit_rate_um_hr']:.1f} µm/h |
| Channel Reynolds number / pressure drop | {h['reynolds_number']:.0f} / {h['pressure_drop_Pa']:.1f} Pa |
| Heat generation / all-HER H₂ bound | {selected['utilities_and_gas']['heat_generation_W']:.1f} W / {selected['utilities_and_gas']['h2_design_rate_L_h']:.2f} L/h |

## Process and instrumentation diagram

{_p_and_id(selected)}

### Required plumbing rules

- `EC-101` is vertical, with bottom inlets and top outlets; retain the separate catholyte/anolyte loops.
- `SP-101` and the corresponding anolyte sample point are upstream of makeup additions and are labelled with run and bath-batch IDs.
- Keep cathode and anode gas paths separate through `GX-101`; neither path may be sealed or vent into the enclosure.
- Install pressure sensing across each cell channel, not only across the pump, so deposit/fouling restrictions are observable.
- All wetted materials follow the RC-1 controlled configuration: borosilicate, PP, PVDF, PTFE, FEP, and qualified gasket material only.

## Electrical and independent shutdown boundary

{_wiring()}

## Instrument schedule

| Tag | Service | Range | Required accuracy | Location | Design use |
|---|---|---|---|---|---|
{inst_rows}

## Controlled procurement BOM

| ID | Item | Qty | Controlled specification | Receiving / acceptance check |
|---|---|---:|---|---|
{bom_rows}

## Build, commissioning, and release sequence

1. Receive against the controlled BOM; record manufacturer, part number, lot, material certificate where applicable, and substitution approval.
2. Assemble the dry cell, record as-built active area, channel depth, membrane area, gasket thickness, and every wetted material.
3. Water-test each loop at 0.10, 0.50, and 1.00 L/min.  Record flow, EC-101 differential pressure, leaks, and dye/step-response behavior.
4. Prove the `ESD-101 → K-101` independent disable path, H₂ alarm, GFCI/RCD, and manual restart discipline before electrolyte enters the system.
5. Calibrate pH, temperature, flow, current, voltage, and pressure channels; prove synchronized timestamps and immutable raw-export storage.
6. Commission electrolyte at 1.00 A / 100 mA·cm⁻² before advancing to the selected 3.00 A / 300 mA·cm⁻² duty.

## Runtime mapping

The build uses `models.reference_cell_design.build_reference_cell_digital_twin()` and
`build_reference_cell_operating_twin()` so that the actual 10 cm² geometry, 2 L
inventory, 3 A duty, and sensor tags—not the generic pilot defaults—govern runtime
state estimation and advisory safety logic.  Hardware actuation remains disabled
until the qualification record is complete.
"""
    return DeploymentPackage(config.configuration_id, selected, instruments, bom, markdown)


def write_deployment_package(package: DeploymentPackage, markdown_path: str | Path, manifest_path: str | Path) -> None:
    """Write the human-readable package and machine-readable manifest."""
    markdown_path, manifest_path = Path(markdown_path), Path(manifest_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(package.markdown, encoding="utf-8")
    manifest_path.write_text(json.dumps(package.manifest(), indent=2) + "\n", encoding="utf-8")
