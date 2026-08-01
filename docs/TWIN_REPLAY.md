# Composed-twin replay campaign

Run the deterministic matrix with:

```bash
python -m models.run_twin_replay
# or, after installing the project entry points:
aq-steel-twin-replay
```

The implementation is `models/twin_replay.py`. It composes the cell physics
surrogate and `DigitalTwin` with `Crate` and `OperatingTwin`; each row checks
crate mounting guidance, a latched operating-twin trip, a zero-current
post-trip command, and a `ShutdownRequest`. Outputs are
`experiments/data/twin_replay_report.json` and
`docs/figures/twin_replay_scenario_matrix.png`.

The matrix includes storm, flood, heavy rain/ingress, sensor bias, sensor
stuck/stale, sensor spike, power loss, freeze, and storm+ingress. The campaign
is synthetic fault injection and therefore does not raise any layer above L0.
See [INDEPENDENT_SHUTDOWN.md](INDEPENDENT_SHUTDOWN.md) for the interface and
failure-mode table.
