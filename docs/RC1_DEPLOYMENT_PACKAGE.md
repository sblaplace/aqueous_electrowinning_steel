# RC-1 deployment package

**Configuration:** `RC-1`

**Status:** pre-procurement deployment design

**Source configuration:** `processes/reference_cell_rc1.yaml`

**Source calculation:** `outputs/reference_cell_rc1_design_report.json`

## Selected deployable duty

| Parameter | Selected value |
|---|---:|
| Cathode active area | 10.0 cm² |
| Channel depth / electrode gap | 3.0 mm |
| Catholyte and anolyte recirculation | 0.50 L/min per loop |
| Current / current density | 3.00 A / 300 mA/cm² |
| Modeled cell voltage | 5.67 V |
| Modeled FE / deposit rate | 99.2% / 393.6 µm/h |
| Channel Reynolds number / pressure drop | 1643 / 4.7 Pa |
| Heat generation / all-HER H₂ bound | 13.2 W / 1.37 L/h |

## Process and instrumentation diagram

```mermaid
flowchart LR
  TK101[TK-101
Catholyte reservoir
1 L] --> P101[P-101
Catholyte pump]
  P101 --> FT201[FT-201]
  FT201 --> PT101[PT-101 ΔP]
  PT101 --> EC101[EC-101
Divided cell
10 cm² cathode
3.0 mm channel]
  EC101 --> TT102[TT-102 outlet]
  TT102 --> TK101
  TK201[TK-201
Anolyte reservoir
1 L] --> P201[P-201
Anolyte pump]
  P201 --> FT202[FT-202]
  FT202 --> PT201[PT-201 ΔP]
  PT201 --> EC101
  EC101 --> TT202[TT-202 outlet]
  TT202 --> TK201
  TK101 --- pH101[pHAT-101 / SP-101
pH and Fe(II)]
  TK201 --- pH201[AT-202
pH]
  EC101 --> G101[Cathode vent → GX-101
H₂ monitor/exhaust]
  EC101 --> G201[Anode vent → GX-101
separate exhaust]
  PS101[PS-101
DC supply] --> K101[K-101 hardwired DC enable]
  K101 --> EC101
  ESD101[ESD-101 / H₂ alarm] -. independent trip .-> K101
  CT201[CT-201] --- PS101
  VT201[VT-201] --- EC101
```

### Required plumbing rules

- `EC-101` is vertical, with bottom inlets and top outlets; retain the separate catholyte/anolyte loops.
- `SP-101` and the corresponding anolyte sample point are upstream of makeup additions and are labelled with run and bath-batch IDs.
- Keep cathode and anode gas paths separate through `GX-101`; neither path may be sealed or vent into the enclosure.
- Install pressure sensing across each cell channel, not only across the pump, so deposit/fouling restrictions are observable.
- All wetted materials follow the RC-1 controlled configuration: borosilicate, PP, PVDF, PTFE, FEP, and qualified gasket material only.

## Electrical and independent shutdown boundary

```text
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
```

## Instrument schedule

| Tag | Service | Range | Required accuracy | Location | Design use |
|---|---|---|---|---|---|
| `TT-101` | catholyte reservoir temperature | 0–100 °C | ±0.5 °C | TK-101 well | thermal balance / high-temperature trip |
| `TT-201` | anolyte reservoir temperature | 0–100 °C | ±0.5 °C | TK-201 well | thermal balance / high-temperature trip |
| `TT-102` | catholyte cell-outlet temperature | 0–100 °C | ±0.5 °C | EC-101 cathode outlet | cell heat-rise measurement |
| `TT-202` | anolyte cell-outlet temperature | 0–100 °C | ±0.5 °C | EC-101 anode outlet | cell heat-rise measurement |
| `FT-201` | catholyte recirculation flow | 0–1.5 L/min | ±5% reading | P-101 discharge | flow / transport state |
| `FT-202` | anolyte recirculation flow | 0–1.5 L/min | ±5% reading | P-201 discharge | flow / transport state |
| `PT-101` | catholyte cell differential pressure | 0–10 kPa | ±1% full scale | EC-101 cathode inlet/outlet | channel fouling / flow verification |
| `PT-201` | anolyte cell differential pressure | 0–10 kPa | ±1% full scale | EC-101 anode inlet/outlet | channel fouling / flow verification |
| `pHAT-101` | catholyte pH | pH 0–7 | ±0.05 pH | TK-101 sample loop | bath control / twin observation |
| `AT-202` | anolyte pH | pH 0–7 | ±0.05 pH | TK-201 sample loop | anolyte drift |
| `FE2P-101` | catholyte Fe(II) sample analysis | 0–2 M | ±0.02 M target | SP-101 | iron balance / EKF observability |
| `CT-201` | rectifier cathodic current | 0–10 A | ±1% reading | PS-101 output | charge ledger / current-density state |
| `VT-201` | cell voltage | 0–10 V | ±0.01 V | EC-101 terminals | energy ledger / voltage model |
| `GT-101` | cathode hydrogen monitor | 0–4% H₂ in air minimum | alarm before LEL | high point near exhaust | independent gas safety alarm |

## Controlled procurement BOM

| ID | Item | Qty | Controlled specification | Receiving / acceptance check |
|---|---|---:|---|---|
| `EC-101` | Divided recirculating cell body | 1 | Vertical 50 × 20 mm channels; selected 3.0 mm gap; clamped removable membrane cassette; borosilicate/PP/PVDF/PTFE/FEP wetted path only | dimensional inspection, dry assembly, water leak test |
| `MEM-101` | Cation-exchange membrane | 3 coupons | Nafion N117 initial comparator; 50 × 50 mm nominal wetted cassette; retain lot and hydration history | visual integrity, wetted area recorded, area-resistance baseline |
| `CA-101` | Cathode coupon set | 10+ | 316L, 1 mm, masked to 10.0 cm²; controlled surface-preparation record | mass/area traceability and mask verification |
| `AN-101` | Anode assembly | 1 + spare | Controlled OER-capable experimental anode, exposed area ≥ cathode area; material/lot fixed per campaign | area and material recorded; post-run inspection plan |
| `TK-101/TK-201` | Covered catholyte/anolyte reservoirs | 2 | 1.0 L working volume each; compatible sample, drain, vent, and temperature ports; total nominal inventory 2.0 L | level marks, leak test, compatible labels |
| `P-101/P-201` | Recirculation pump loops | 2 | Independently adjustable; compatible wetted materials; verified 0.10–1.00 L/min installed-loop range | three-point installed flow curve and pressure test |
| `HT-101/HT-201` | Thermal control | 2 zones | 50–70 °C qualified range; target 60 °C; independent temperature measurement | heat-up/hold test and over-temperature trip test |
| `PS-101` | DC rectifier | 1 | 30 V / 10 A CC/CV bench supply; normal RC-1 duty 3.00 A, qualified ceiling 3.00 A | dummy-load CC verification and output-off behavior |
| `K-101/ESD-101` | Independent DC shutdown chain | 1 | hardwired latching emergency stop and contactor/enable interruption; separate from software twin | documented proof test removes rectifier enable |
| `DAQ-101` | Synchronized data acquisition | 1 | logs current, voltage, temperature, flow, pressure, pH, and event timestamps; immutable raw export retained | timestamp and channel-calibration check |
| `GX-101` | Vent/exhaust and hydrogen monitoring | 1 | separate anode/cathode vents to site-approved exhaust; H₂ monitor/alarm; no sealed gas path | site ventilation review and alarm functional test |
| `CT-101` | Secondary containment | 1 | compatible tray sized for complete 2 L inventory plus margin; segregated from electrical equipment | capacity and drain-path inspection |

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
