# Reference divided-cell design basis — RC-1

**Status:** pre-procurement engineering basis, version 0.1

**Date:** 2026-08-02

**Decision:** build one conservative, modular, recirculating *reference cell* before any pilot or production geometry.  This document freezes the parts that can be designed without electrochemical calibration and identifies every quantity that must remain adjustable or measured.

RC-1 is a research apparatus.  It is designed to answer the chemistry and cell-performance gates in `PROGRAM_SUMMARY.md`; it is **not** evidence that a production architecture has been selected or that the model's predicted FE, voltage, deposit quality, or durability will be achieved.

Machine-readable nominal values are in [`processes/reference_cell_rc1.yaml`](../processes/reference_cell_rc1.yaml).  A physical build must record all as-built deviations in the run metadata.

The executable design synthesis is `python -m models.run_reference_cell_design`.  It uses the controlled YAML as its input, evaluates the declared area/channel/flow candidate space at the 300 mA/cm² design duty, and writes `outputs/reference_cell_rc1_design_report.json`.  It is the source for selecting a procurement configuration; it is not a replacement for the physical qualification tests in §5.

## 1. Design requirements

| ID | Requirement | RC-1 design response | Verification before energized operation |
|---|---|---|---|
| R1 | Operate a divided sulfate cell from 50–70 °C at up to 300 mA/cm². | 10.0 cm² nominal cathode area; 3.0 A maximum operating current; 30 V / 10 A CC supply. | Measure exposed area and verify CC limit with a dummy load. |
| R2 | Obtain independent charge, iron, and energy ledgers. | Synchronized current/voltage trace, weighable coupon, bath samples, auxiliary-energy logging. | Measurement-system-analysis (MSA) record is accepted. |
| R3 | Allow membrane, electrode, and flow-path changes without rebuilding the test stand. | Clamped membrane cassette, coupon holders, independent catholyte/anolyte loops, removable inserts. | Dry assembly/disassembly and leak test. |
| R4 | Avoid an unobserved transport or thermal limitation. | Known channel geometry, variable recirculation, temperature probes at both inlet/outlet, sample ports. | Water flow map, pressure/leak test, and heat-up test. |
| R5 | Control credible acid, hydrogen, electrical, and hot-liquid hazards. | Secondary containment, external vent path, H₂ monitor, hardwired independent rectifier disable, GFCI/RCD. | Signed run authorization and trip proof test. |
| R6 | Support parameter fitting rather than just a plating demonstration. | Reference-electrode port / voltage taps, sample ports, gas measurement takeoff, fixed geometry/version ID. | Sensor checkout and timestamp synchronization. |

## 2. Frozen nominal configuration

### 2.1 Cell stack

| Item | Nominal RC-1 value | Rationale / procurement constraint |
|---|---:|---|
| Cathode active area | 50 mm × 20 mm = **10.0 cm²** | Reaches the 300 mA/cm² program target at 3.0 A, inside a 10 A bench supply's useful range. |
| Cathode | 316L coupon, 1 mm thick, masked to defined area | Low-cost, removable, weighable commissioning substrate.  It is not a harvest-architecture choice. |
| Anode | Dimensionally stable OER-capable anode or specified experimental anode in a separate anolyte | Do not let anode material silently change between runs.  Anode identity and area are controlled variables. |
| Anode exposed area | ≥10 cm² | Keeps anode area ratio at least 1:1; record the actual value. |
| Separator | Nafion N117, nominal 50 mm × 50 mm wetted coupon | First membrane comparator in `BATH_SPEC.md`; retain a removable cassette so alternatives can be tested. |
| Cathode-side channel | 50 mm long × 20 mm wide × 3 mm nominal depth | Gives a defined flow cross-section and a small electrolyte gap without making a precision-machining claim. |
| Anode-side channel | Same nominal planform and depth | Symmetry simplifies first current/flow interpretation. |
| Electrode-to-membrane gap | **3.0 mm nominal per side**, measured after compression | The value is a geometry input, not an assumed 2 cm beaker spacing.  Record spacer thickness and as-built gap. |
| Wetted construction materials | Borosilicate, PP, PVDF, PTFE, FEP, EPDM/Viton only after compatibility check | Do not use unqualified metals, adhesives, or 3-D-print resin in hot acidic electrolyte. |
| Seals | Replaceable chemically compatible gasket; compression stops | Gasket compression must not change active area or collapse the flow channel. |
| Cell orientation | Vertical channels, bottom inlet / top outlet | Deliberately gives generated gas a buoyant escape path.  `models/gas_holdup.py` now screens that path quantitatively (drift-flux void fraction, Bruggeman conductivity, current redistribution) and predicts <2 % outlet void fraction at 300 mA/cm² in this geometry — but it remains a **screening (L0) bubble model**, not a validated one, until `gas_holdup.measurement_protocol()` is run. |

The membrane cassette must define the exposed membrane area independently of the coupon mask.  Measure and log both areas.  No performance result may be compared across runs if either changes without a new configuration version.

### 2.2 Balance of plant

| Item | Nominal RC-1 value | Required capability |
|---|---:|---|
| Catholyte and anolyte inventory | 1.0 L each (2.0 L total nominal) | Independent reservoirs, level marks, covered headspace, and sample ports. |
| Recirculation | Independent, adjustable loops | Measure actual flow; start with a broad **0.1–1.0 L/min** capability rather than asserting a boundary-layer thickness. |
| Pump wetted materials | PP/PVDF/PTFE/FEP or documented compatible equivalent | Pump must be isolated from the supply and have a measured flow curve in the installed loop. |
| Temperature control | Heating and optional cooling provision | 50–70 °C; independent temperature probes at reservoir and cell outlet. |
| Power | 30 V / 10 A CC/CV supply plus independent DMM and coulomb counter | Normal RC-1 operating ceiling is 3.0 A; the 10 A rating is not permission to use 10 A. |
| Gas handling | Separate cathode/anode headspace vents to safe exhaust; optional measured cathode-gas takeoff | Never use a closed gas path.  A gas measurement train must not add meaningful backpressure. |
| Containment | Secondary tray sized for the complete liquid inventory plus margin | Keep acid, electrical equipment, and drain paths physically separated. |

At 3.0 A, the 10 cm² cathode corresponds to 300 mA/cm².  The RC-1 geometry therefore reaches the program's decision current density without an oversized power supply or a large, poorly characterized cell.

## 3. Pre-build engineering calculations

These calculations are requirements for sizing and safety, **not** predictions of electrochemical success.

### 3.1 Electrical and thermal envelope

Use 3.0 A and an **8.0 V RC-1 screening ceiling** for procurement and fault planning.  The present L0 voltage model returns roughly 6 V at the 300 mA/cm² decision duty with its conservative membrane/contact assumptions, so a 5 V ceiling would preclude the very characterization run RC-1 is built to make.  The 30 V supply is not an operating limit.

- maximum stack electrical power: **24 W**;
- the existing thermal model's 2 L case can be used to size heating/cooling and the thermal-abort test; actual heat generation is measured as `I × V` and must not be inferred from an assumed voltage;
- all clips, leads, fuse/contactor, current meter, and emergency disable path must be rated above the supply fault capability, not merely the nominal 3 A setpoint.

### 3.2 Hydrogen design envelope

If all 3.0 A is diverted to HER, the maximum cathode H₂ rate is approximately **1.25 L/h at 25 °C and 1 atm**.  This is a conservative electrical upper bound, not a predicted HER rate:

\[
\dot n_{H_2}=I/(2F), \qquad \dot V_{H_2}\approx1.25\ \mathrm{L/h}\quad\text{at }I=3\ \mathrm{A}.
\]

The external ventilation/exhaust design must be verified by the site EHS authority.  Do not derive a safe room ventilation rate solely from this calculation.  The H₂ monitor alarm, rectifier-disable trip, and manual emergency stop are mandatory before energized operation.

### 3.3 Flow and residence-time envelope

At 0.1–1.0 L/min, each 1 L loop has a nominal reservoir turnover time of 10–1 min.  In the 20 mm × 3 mm channel, the nominal superficial velocity is approximately 0.028–0.28 m/s.  These values are useful for selecting a pump and avoiding an obviously stagnant channel; they do **not** establish wall shear, bubble coverage, boundary-layer thickness, or a mass-transfer coefficient.

Before electrolyte is used, document a water-flow test at a minimum of three setpoints with measured flow, pressure drop, leak observation, and a dye or conductivity step-response observation.  If the flow distribution is materially asymmetric, correct the hardware before electrochemical experiments.

## 4. CFD and model work: allowed claims and boundaries

A pre-build CFD study is useful once the CAD geometry is frozen.  Its required outputs are:

1. pressure drop versus flow rate;
2. velocity distribution, recirculation zones, and residence-time distribution;
3. temperature field for bounded heat loads;
4. gas-free current-path/electrolyte resistance sensitivity to gap and conductivity; and
5. a mesh/conservation and sensitivity record.

It must report a **family of cases**, not a single predicted operating point:

- low/nominal/high electrolyte conductivity;
- low/nominal/high heat load;
- clean-channel and conservative gas-void blockage cases;
- dimensional tolerance cases for channel depth and gasket compression;
- inlet flow cases spanning the installed pump range.

The model may claim geometry-dependent flow, pressure-drop, and thermal *bounds*.  Until calibrated, it may not claim a particular FE, bubble coverage, local pH, deposition rate, deposit morphology, membrane lifetime, or production-cell performance.  Those require the measurements listed in §6.

A reduced-order hydraulic/thermal model is sufficient for RC-1 if it passes conservation checks and shows no problematic dead zone.  Escalate to two-phase CFD only if the transparent/observable reference cell shows gas hold-up or maldistribution that the reduced model cannot represent.

That reduced-order two-phase model now exists: `models/gas_holdup.py` (`aq-steel-gas-holdup`).  It predicts, for the RC-1 geometry at the 300 mA/cm² kill-criterion point and 85 % FE, an outlet void fraction near 1.2 %, an ohmic penalty under 1 %, and axial current spread near 1 % — i.e. **hold-up should not be observable at bench scale**, which is itself the falsifiable prediction.  Its `measurement_protocol()` (~$450, 3 days, level-swell + segmented cathode + backlit bubble video + AC resistance) defines the escalation trigger concretely: recalibrate on a systematic offset with the right trend, and go to CFD only on observed maldistribution, dead zones, slugging or channelling that a 1-D axial model cannot represent.

## 5. Measurement and safety design package required before purchase

No purchase order should be released until these artifacts are approved:

- [ ] RC-1 general arrangement drawing, membrane-cassette drawing, and flow/electrical schematic;
- [ ] as-built controlled BOM with material compatibility and replacement-part IDs;
- [ ] vendor datasheets for membrane, pumps, tubing, gaskets, cell body, electrodes, and power equipment;
- [ ] a measurement map: sensor ID, range, accuracy, calibration standard, sample rate, timestamp source, and location;
- [ ] MSA protocol for mass, current, voltage, temperature, flow, pH, and gas collection;
- [ ] vent/exhaust design and H₂ detector location approved for the actual site;
- [ ] independent hardwired shutdown schematic and proof-test procedure, consistent with `INDEPENDENT_SHUTDOWN.md`;
- [ ] chemical compatibility review and waste handling route;
- [ ] dry leak/pressure test, water flow-distribution test, and electrical dummy-load test procedure;
- [ ] operating envelope, abort thresholds, restart criteria, and named run authority.

A cheap H-cell may be used only for qualitative membrane or chemistry scouting.  It is not RC-1 unless it meets the controlled-area, flow, sensing, containment, and traceability requirements above.

## 6. Calibration measurements RC-1 is built to obtain

| Unknown / model term | Measurement replacing the assumption | Initial acceptance evidence |
|---|---|---|
| Fe/HER kinetics | RDE/LSV plus divided-cell polarization on actual bath and cathode surface | Replicates, reference scale recorded, raw exports mapped through `kinetics_fit_pipeline.py`. |
| FE and deposit rate | Dry mass, Fe composition, integrated charge, thickness map | Replicated measured FE; do not call mass-only FE verified iron FE. |
| Voltage decomposition | Full-cell voltage plus reference-electrode/voltage-tap measurements | Current/voltage trace synchronized with temperature and flow. |
| Membrane resistance/crossover | Area resistance, catholyte/anolyte Fe speciation and volume samples | Crossover reported as measured flux/rate with uncertainty. |
| Hydrodynamics | Installed flow, pressure drop, dye/step response, observed gas behavior | As-built geometry and flow record attached to every run. |
| Thermal model | Inlet/outlet/reservoir temperature and auxiliary energy | Heat balance residual reported; no unlogged heater power. |
| Gas/H absorption | Cathode gas measurement and deposit H analysis where available | Charge ledger remains explicitly partial until products are measured. |

The canonical raw-data and ledger requirements remain those in `DATA_CONTRACT.md`.  RC-1 does not relax them.

## 7. Operating envelope and staged progression

The apparatus is commissioned progressively; do not start at the program kill criterion.

| Stage | Purpose | Current-density ceiling | Advance only when |
|---|---|---:|---|
| 0 — dry/water | Leak, flow, sensor, shutdown, and timestamp test | No electrolysis | All preflight checks in §5 pass. |
| 1 — electrolyte commissioning | Verify stable electrical/thermal operation and sampling | 100 mA/cm² (1.0 A) | No leak, no unplanned trip, complete trace, and post-run balance samples obtained. |
| 2 — reference-condition replication | Establish FE/V/deposit repeatability | 100–200 mA/cm² | Three independently prepared/recorded runs meet predeclared MSA and balance checks. |
| 3 — decision-current matrix | Test chemistry gate at useful current | up to 300 mA/cm² (3.0 A) | Safety envelope remains demonstrated; actual cell voltage stays within the qualified supply/instrument range. |
| 4 — disturbance/durability | Flow, feed, membrane, and time sensitivity | Same qualified envelope | Calibration residuals and component inspection justify the exposure. |

The program criterion remains: at **j ≥300 mA/cm²**, replicated divided-cell runs must sustain **FE ≥70%** and **net DC specific energy ≤4,000 kWh/t Fe** after appropriate concentration, temperature, and flow optimization.  It is a decision criterion, not a design assumption.

## 8. Open decisions deliberately retained

| Decision | Why it is not frozen now | Decision evidence |
|---|---|---|
| Production architecture | Powder/flakes vs foil and continuous harvesting remain separate branches. | Reference-cell chemistry, peel coupon, and architecture gates. |
| Final membrane | N117 is the first comparator, not proven optimum. | Resistance, crossover, fouling, and durability measurements. |
| Final anode | Anode gas, ferric chemistry, and degradation are chemistry-specific. | Anode polarization, gas analysis, inspection, and impurity data. |
| Final flow rate/channel geometry | Boundary layer and bubble behavior are not known from geometry alone. | RC-1 flow/voltage/FE data calibrated against bounded model cases. |
| Representative-feed treatment | Copperas/SPL impurity spectrum must be assayed rather than invented. | Certificate of analysis + independent assay + impurity challenge results. |

## 9. Design-review decision

**Recommended approval:** approve RC-1 only as a modular reference apparatus after §5 is complete.  Do not approve a production drum, rotating cylinder, or pilot design from uncalibrated CFD or screening-model output.

This gives the program a real, buildable and safe design before purchasing while preserving the ability to learn from the measurements that determine whether the chemistry is viable.
