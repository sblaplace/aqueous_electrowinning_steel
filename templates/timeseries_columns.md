# Timeseries CSV format

Every plating experiment produces a timeseries CSV. This file records **raw
instrument readings** — not derived quantities.  Post-processing (charge
integration, current density, FE) happens in `models/plating_data.py`.

## Columns

| Column               | Unit    | Required | Description                                                        |
|----------------------|---------|:--------:|--------------------------------------------------------------------|
| `timestamp_s`        | s       | ✓        | Seconds since experiment start (monotonically increasing)          |
| `current_setpoint_A` | A       |          | Commanded current from the power supply                            |
| `current_actual_A`   | A       | ✓        | Measured current (cathodic negative by default)                    |
| `voltage_V`          | V       | ✓        | Cell voltage across working/counter electrodes                     |
| `temperature_C`      | °C      |          | Bath temperature from thermocouple or inline probe                 |
| `pH`                 | —       |          | Bath pH from inline probe or spot measurement                      |
| `fe2_concentration_M`| mol/L   |          | Fe²⁺ concentration from titration or inline sensor                 |
| `notes`              | —       |          | Event markers: "added H₃BO₃", "power outage", "sample taken", etc |

## Conventions

- **First row**: `timestamp_s = 0` marks the moment current is applied.
- **Sign**: Cathodic current is negative.  If your instrument reports
  cathodic current as positive, negate it before recording or note the
  convention in the manifest.
- **Sampling**: Record as fast as the instrument allows.  1 Hz is a good
  default for galvanostatic runs; higher for pulse-reverse.
- **No interpolation**: If a sensor drops out, leave the cell blank or
  record `NaN`.  Do **not** forward-fill or interpolate in the raw file.
- **Event notes**: Use the `notes` column to mark physical events
  (stirring change, sample extraction, pH adjustment).  This makes
  video correlation possible.

## Example

```
timestamp_s,current_setpoint_A,current_actual_A,voltage_V,temperature_C,pH,notes
0,-2.00,-1.98,2.41,25.3,3.2,
1,-2.00,-2.01,2.42,25.3,3.2,
2,-2.00,-1.99,2.40,25.4,3.2,
300,-2.00,-2.02,2.45,25.6,3.1,added 5 mL H3BO3 solution
301,-2.00,-2.01,2.44,25.6,3.1,
...
```
