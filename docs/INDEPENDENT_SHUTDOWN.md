# Independent-shutdown contract

The operating twin is a supervisory software component, not the safety
instrumented function.  It may detect an unsafe condition and emit a
`ShutdownRequest`; it **never owns shutdown authority**.  A diverse,
independently powered/hardwired channel consumes that request (and its own
physical interlocks) and executes the safe state: remove rectifier enable,
stop hazardous feeds, isolate energy, and/or elevate the crate as specified by
the site design.

## Interface

`models.operating_twin.ShutdownRequest` is the software boundary:

- `request_id`: monotonic/run-scoped correlation identity;
- `action`: requested safe-state action, such as `storm_mode_hold_high_wind`,
  `flood_hold_elevate_and_shutdown`, or `sensor_fault_hold`;
- `reasons`: independently auditable detection signals;
- `source_run_id`: traceability to the synchronized snapshot.

The request has serialization only.  There is no `execute()`, contactor,
valve, rectifier, GPIO, or network actuator in `OperatingTwin`.  A consumer
must authenticate and validate it, then use the separate physical channel.
Loss of the request link must itself be fail-safe in that channel; the twin
request is not a substitute for a hardwired trip.

The replay invariant is intentionally strict: after any trip the twin is
`TRIPPED`, emits a request, and `command()` returns zero current with no
actuation command.  The replay tests do **not** claim hardware independence;
they prove only the software discipline and serialization boundary.

## Failure-mode table (L5 acceptance item)

These are screening limits for the scripted harness, not qualified hardware
limits.  They must be replaced by the signed hardware qualification record.

| Detection signal | Abort limit | Recovery action |
|---|---:|---|
| Wind-gust sensor / crate load | >40 m/s gust | request storm hold; independent channel disables rectifier; verify ballast/tie-down and inspect before reset |
| Flood-depth sensor | >0.10 m | request flood hold; independent channel isolates power and elevates/isolates crate |
| Rain / ingress | >100 mm/h or ingress switch | request ingress hold; independent channel isolates electrical equipment and drain/seal |
| Temperature | outside configured 0–80 °C replay envelope | independent channel removes energy; investigate cooling/heating fault |
| Voltage | >5 V or invalid power quality | remove rectifier enable; qualify supply and sensor before reset |
| Current / current density | >10 A / >200 mA cm⁻² | remove rectifier enable; inspect rectifier and electrodes |
| pH / Fe²⁺ | pH outside 0.5–5.0 / Fe²⁺ outside 0.2–2.0 M | hold; sample and correct bath, then operator-authorized reset |
| Sensor quality / bias / spike | quality not `ok/good/valid`, or hard limit crossed | hold; prove sensor plausibility with diverse measurement |
| Stale or stuck timestamp | >5 s old | hold; restore independent time/sensor path |
| Freeze detection | freeze flag with protection required | request freeze hold; maintain heat/trace protection and inspect |
| Loss of twin request link | channel-specific watchdog timeout | hardwired channel trips without relying on the twin |

All model, crate, and site claims remain **L0** until a real fault/test
campaign, load test, and independent shutdown proof test are completed.
