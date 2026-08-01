# Cell ↔ crate environmental-disturbance coupling (L0)

**Tier:** L0 (screening). **Data:** none. **Builds on:** PR #29 coupled
bath/recirculation dynamics. **Status:** implemented — closes the whole-system
loop (cell ↔ crate ↔ site) called for in `docs/SYSTEM_TWIN.md`.

## What this adds

The crate layer (`models/crate.py`) turns weather/site into a *stability
verdict*; the operating twin (`models/operating_twin.py`) turns the same
observations into *trip/shutdown decisions*. Neither previously perturbed the
cell process. This module is the missing adapter: it maps measured/modeled
environmental + crate observations into **process disturbance inputs** that the
cell thermal / Fe2+ / pH balances consume.

Key guarantee: **coupling is OFF by default.** `disturbance_from_environment`
with no observations returns a disabled, all-zero `DisturbanceInputs`, so the
EKF is byte-identical to the uncoupled twin. Coupling changes nothing unless
real env/crate data is supplied and enabled.

## Mapping

`models/env_coupling.py::disturbance_from_environment(env_state, crate_state)`
is pure and deterministic. With any of wind / rain / flood / ingress present it
returns `enabled=True`:

| Input | Source | Output | Units | Default |
|------:|--------|--------|-------|---------|
| `wind_gust_m_s` | env or crate `WindLoad.gust_m_s` | convective coefficient `h_conv` | W/m²K | 0 |
| `T_ambient_C` | env or crate wind | ambient temperature `T_ambient_C` | °C | 25 |
| `rain_intensity_mm_hr` | env | rain cooling `rain_cooling` | W/m² | 0 |
| `flood_depth_m` / `ingress_detected` | env | ingress dilution `dilution` | 1/hr | 0 |

### Wind-driven convection
```
h_conv = H_CONV_BASE + H_CONV_WIND_K · gust^H_CONV_WIND_EXP     [W/m²K]
       = 5.0            + 3.0              · gust^0.7
```
Natural-convection floor 5 W/m²K; a 40 m/s gust → ≈ 45 W/m²K.

### Rain cooling
```
rain_cooling = RAIN_COOLING_W_M2_PER_MMHR · rain_intensity     [W/m²]
             = 0.5 · rain_intensity
```
100 mm/hr → 50 W/m² of convective-equivalent cooling.

### Ingress dilution
```
dilution = INGRESS_DILUTION_PER_M_FLOOD · flood_depth + (INGRESS_DILUTION_BASE if ingress)
         = 0.10 · flood_depth + 0.05·(ingress ? 1 : 0)          [1/hr]
```

## Application in the cell dynamics

`models/bath_dynamics.py::step()` reads `design_point["_env_dist"]` (a
`DisturbanceInputs`). When enabled it:

- **Overrides the ambient temperature** in the thermal balance (`T_ambient`).
- **Adds convective + rain cooling** to the catholyte/anolyte energy balances
  over the exposed `heat_exchange_area_m2` (default 10 m²):
  ```
  Q_conv = h_conv · A_heat · (T - T_ambient)     [W]
  Q_rain = rain_cooling · A_heat                  [W]
  ```
- **Dilutes bulk Fe2+** toward Fe2+-free water and **drags pH toward neutral**
  (`pH = 7`) at the ingress rate:
  ```
  dfe2/dt -= dilution · fe2
  dpH/dt  += dilution · (7 - pH)
  ```

When `_env_dist` is absent or `enabled=False`, every term above is zero — the
`step()` code path is unchanged from PR #29.

## Wiring

`models/digital_twin.py::DigitalTwin` gains two optional constructor args,
`env_state` and `crate_state`, and a `set_environment(env_state, crate_state)`
method that (re)computes the disturbance into `design_point["_env_dist"]`.
Call it whenever the environment / site snapshot updates.

```python
twin = DigitalTwin()
twin.set_environment({"wind_gust_m_s": 40.0, "T_ambient_C": 0.0}, {})
# ... twin.update(sensor_readings) — now coupled; storm wind cools the cell.
```

## Fail-safe contract

The adapter does **not** touch the operating twin's storm-mode
`ShutdownRequest` / `ControlCommand` (current → 0) boundary; that stays in
`models/operating_twin.py`. The 9-scenario replay (`tests/test_twin_replay.py`)
and `tests/test_operating_twin.py` / `tests/test_crate.py` are untouched and
green. Coupling only perturbs the cell dynamics; it never grants shutdown
authority (see `docs/INDEPENDENT_SHUTDOWN.md`).

## Tests

`tests/test_env_coupling.py` locks in the guarantees: coupling-off
byte-identical, higher wind → stronger cooling → colder cell, rain cooling
magnitude, ingress dilution of Fe2+ and pH neutralization, and
`DigitalTwin.set_environment` integration. Run:

```bash
python -m pytest tests/test_env_coupling.py tests/test_bath_dynamics.py \
    tests/test_digital_twin.py tests/test_twin_replay.py \
    tests/test_operating_twin.py tests/test_crate.py -q
```
