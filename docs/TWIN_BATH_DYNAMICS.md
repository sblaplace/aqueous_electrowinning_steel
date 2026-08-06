# Twin Bath Dynamics — Coupled Mass/Energy Balance Model

## Overview

This document describes the coupled bath/recirculation dynamics implemented in `models/bath_dynamics.py` for the digital twin's Extended Kalman Filter (EKF). The dynamics replace the previous ad-hoc mean-reversion model with physically-consistent conservation laws for mass and energy, plus recirculation exchange with a finite reservoir.

**Model tier:** L0 scaffold. No real data. Conservation-law dynamics ready for L1 data-fitting.

## Governing Equations

### 1. Fe²⁺ Mass Balance (State Index 2)

**Equation:**
```
d(fe2)/dt = -consumption + recirculation + makeup
```

**Terms:**
- **Consumption** (Faraday deposition):
  ```
  consumption [M/hr] = (j_A_m2 × FE / (z × F)) × area / V_cath × 3600
  ```
  where:
  - `j_A_m2` = current density (A/m²)
  - `FE` = Faradaic efficiency (from physics model)
  - `z` = 2 (electrons per Fe²⁺)
  - `F` = 96485.3321 C/mol (Faraday constant)
  - `area` = electrode area (m²)
  - `V_cath` = catholyte volume (L)

- **Recirculation exchange:**
  ```
  recirculation [M/hr] = (flow / V_cath) × (fe2_res - fe2)
  ```
  where `fe2_res` is the reservoir Fe²⁺ concentration.

- **Makeup source:**
  ```
  makeup [M/hr] = fe2_makeup_rate_M_hr
  ```
  Control input for FeSO₄ addition or sacrificial Fe dissolution.

**Reservoir balance:**
```
d(fe2_res)/dt = (flow / V_res) × (fe2 - fe2_res) + makeup × (V_cath / V_res)
```

**Redox transfer (Fe³⁺ shuttle extension, off by default):** with
`fe3_shuttle_enabled` on, the Fe²⁺ balance additionally receives
`shuttle_return − r_prod` (Section 7): autoxidation drains it, the cathodic
Fe³⁺ shuttle returns it, so dissolved iron inventory leaves only through the
Fe(OH)₃ sludge ledger.

### 2. pH / Buffer Dynamics (State Index 3)

**Equation:**
```
d(pH)/dt = -net_proton_rate / β + recirculation
```

**Terms:**
- **HER hydroxide production:**
  ```
  OH_production [M/hr] = (j_A_m2 × (1 - FE) / F) × area / V_cath × 3600
  ```
  HER: 2H₂O + 2e⁻ → H₂ + 2OH⁻ (1 mol OH⁻ per mol e⁻)

- **Acid dose:**
  ```
  acid_dose [M/hr] = acid_dose_rate_M_hr
  ```
  Control input for acid addition (positive = adds H⁺).

- **Net proton rate:**
  ```
  net_proton [M/hr] = acid_dose - OH_production
  ```

- **Buffer capacity:**
  ```
  β [mol/(L·pH)] = buffer_capacity_beta
  ```
  Typical for borate buffer: 0.01–0.1 mol/(L·pH).

- **Recirculation:**
  ```
  recirculation [pH/hr] = (flow / V_cath) × (pH_res - pH)
  ```

**Bounds:** pH clamped to [0, 14].

### 3. Thermal Balance (State Indices 0, 1)

**Catholyte energy balance:**
```
dT_cath/dt = (Q_cath - Q_cool - Q_membrane - Q_amb_cath + Q_recirc_cath) / (m_cath × Cp)
```

**Terms:**
- **Joule heating:**
  ```
  Q_joule [W] = V_cell × I
  Q_cath [W] = Q_joule × joule_heat_fraction_catholyte
  Q_anol [W] = Q_joule × joule_heat_fraction_anolyte
  ```

- **Cooling:**
  ```
  Q_cool [W] = cooling_power_W
  ```
  If not set, auto-balances: `Q_cool = Q_cath` (steady-state default).

- **Membrane crossover:**
  ```
  Q_membrane [W] = UA_membrane × (T_cath - T_anol)
  ```

- **Ambient losses:**
  ```
  Q_amb_cath [W] = UA_ambient × 1.0 × (T_cath - T_amb)
  Q_amb_anol [W] = UA_ambient × 1.0 × (T_anol - T_amb)
  ```

- **Recirculation heat exchange:**
  ```
  flow_thermal [W/K] = flow × ρ × Cp / 3600
  Q_recirc_cath [W] = flow_thermal × (T_res - T_cath)
  Q_recirc_anol [W] = flow_thermal × (T_res - T_anol)
  ```

**Thermal mass:**
```
m_cath [J/K] = V_cath × ρ × Cp
m_anol [J/K] = V_anol × ρ × Cp
m_res [J/K] = V_res × ρ × Cp
```

**Anolyte balance:**
```
dT_anol/dt = (Q_anol + Q_membrane - Q_amb_anol + Q_recirc_anol) / (m_anol × Cp)
```

**Reservoir balance:**
```
dT_res/dt = (Q_recirc_from_cath + Q_recirc_from_anol - Q_amb_res) / (m_res × Cp)
```

### 4. Cell Voltage (State Index 6)

**Equation:**
```
dV/dt = (V_predicted - V) / τ_V
```

**Time constant:**
```
τ_V [hr] = max(τ_electrical + τ_mass_transfer, V_relax_min_hr)
```

where:
- `τ_electrical [s] = R_ohm × C_dl × area`
- `R_ohm [Ω] = gap / (σ × area)`
- `τ_mass_transfer [s] = 1.0 / max(fe2, 0.01)` (heuristic)

**Default:** `V_relax_min_hr = 10.0` hr (slow tracking for stability).

### 5. Current Density (State Index 4)

**Equation:**
```
j(t+dt) = j(t) + α × (j_setpoint - j(t))
```

where `α = 1 - exp(-dt / τ_j)` and `τ_j = tau_j_hr` (default 0.5 hr).

### 6. Deposit Thickness (State Index 5)

**Equation:**
```
d(deposit)/dt = deposit_rate_um_hr × (j − i_shuttle·10)/j_applied
```

from physics model prediction, scaled by the galvanostatic shuttle slip
(identity factor 1.0 when the Fe³⁺ extension is off). Clamped to ≥ 0.

### 7. Fe³⁺ Redox Shuttle (Auxiliary CSTR Extension; off by default)

Enable with `fe3_shuttle_enabled: true` (see `apply_fe3_scenario`).
Static steady-state counterpart: `models/fe3_shuttle.py`; shipped note and
unit errata: `docs/SIM_BATH_REDOX.md`.

**States** (auxiliary, NOT in the 7-state EKF vector):
`fe3_catholyte_M`, `fe3_reservoir_M`, `fe3_sludge_cumulative_mol`.

**Catholyte balance:**
```
fe3*  = (r_prod + (flow/V_cath)·fe3_res) / (k_shuttle + flow/V_cath)
fe3⁻  = fe3* + (fe3 − fe3*)·exp(−(k_shuttle + flow/V_cath)·dt)   (exact exponential)
fe3⁺  = fe3⁻ − max(0, fe3⁻ − cap(pH))                            (instant Fe(OH)₃ cap)
```
with
- `r_prod [M/hr]` = homogeneous autoxidation (`bath_startup.fe2_oxidation_rate`,
  at current T/pH/Fe²⁺, O₂ pinned at `fe3_o2_fraction_of_sat` of Weiss air
  saturation) + `4·fe3_crossover_o2_flux·A/V_L` (anolyte crossover fault);
- `k_shuttle [1/hr] = (D_Fe3/δ)·(A/V_cath)·3600` — mass-transfer-limited
  cathodic Fe³⁺ → Fe²⁺ reduction;
- `cap(pH) = Ksp(Fe(OH)₃)/[OH⁻]³` (`fe3_shuttle.fe3_solubility_cap_M`);
- precipitated excess joins `fe3_sludge_cumulative_mol` (×V_cath).

**Back-couplings:**
- **Fe²⁺ balance**: `−r_prod + ∫k_shuttle·fe3 dt/dt` (exact step-integral of
  the shuttle return) — net inventory loss only via sludge;
- **galvanostatic split**: the Fe/HER pair shares `j − i_sh` with
  `i_sh [A/m²] = F·k_m·[Fe³⁺]`; scales Fe consumption, HER/OH⁻ production and
  the deposit rate (`fe3_shuttle.ce_penalty_at_j` semantics);
- **pH balance**: `−r_prod + 3·precip_rate` added to `net_proton` (−1 H⁺ per
  Fe²⁺ oxidised, +3 H⁺ per Fe(OH)₃ precipitated; net +2 H⁺ per sludge mol).

**Reservoir Fe³⁺:** passive mixing plus the same hydrolysis cap at
`pH_reservoir`; its precipitation joins the same sludge ledger. Reservoir
autoxidation is NOT modelled (documented limitation — scenario O₂ pinning
describes the catholyte).

**Steady state:** held at fixed (T, pH, Fe²⁺) with precipitation inactive,
the dynamic states relax to the static closed form
`[Fe³⁺]_ss = r_prod/(k_m·A/V)` — the recirculation terms cancel identically
at mutual steady state (fe3_res = fe3). Cross-validated in
`tests/test_bath_fe3_cstr.py`.

## Surrogate validity guard (physics model)

Every on-grid quantity above (`FE`, `v_cell`, `deposit_rate`, `surface_pH`)
comes from the fast interpolated surrogate in `models/twin_physics.py`. Its
interpolators extrapolate linearly outside the calibrated grid
(`bounds_error=False, fill_value=None`), which can yield physically impossible
values — negative deposit rate and absurd cell voltage — during transients
(startup, faults, or env-coupling storm cases) that leave the grid.

`CellProcessModel.predict()` therefore:
- detects out-of-grid queries (`CellProcessModel.in_bounds` / `grid_bounds`
  report the validity envelope; `ProcessPrediction.extrapolated` flags the
  result);
- clamps physical outputs to the calibrated range (`deposit_rate ≥ 0`, `v_cell`
  and `deposit_rate` within the grid extremes) so the surrogate never feeds
  nonsense into the energy/mass integrators;
- leaves in-grid interpolation byte-identical (the guard is a no-op inside the
  envelope).

Out-of-grid queries are a signal, not an error: the flag lets callers (and the
future uncertainty layer) downgrade confidence in the twin's prediction rather
than trust chemistry the surrogate was never calibrated for.

## Design-Point Parameters

All parameters have explicit defaults in `BATH_DYNAMICS_DEFAULTS` (see `models/bath_dynamics.py`).

### Recirculation Loop
| Parameter | Unit | Default | Description |
|-----------|------|---------|-------------|
| `recirculation_flow_L_hr` | L/hr | 6000 | Total recirculation flow rate |
| `reservoir_volume_L` | L | 50000 | External reservoir/balance tank volume |
| `catholyte_volume_L` | L | 800 | Catholyte compartment volume |
| `anolyte_volume_L` | L | 2000 | Anolyte compartment volume |

### Fe²⁺ Makeup
| Parameter | Unit | Default | Description |
|-----------|------|---------|-------------|
| `fe2_makeup_rate_M_hr` | M/hr | 0.0 | FeSO₄ makeup rate to catholyte |
| `fe2_reservoir_M` | M | 1.0 | Initial reservoir Fe²⁺ concentration |

### pH / Buffer Control
| Parameter | Unit | Default | Description |
|-----------|------|---------|-------------|
| `buffer_capacity_beta` | mol/(L·pH) | 0.05 | Bath buffer capacity |
| `acid_dose_rate_M_hr` | M/hr | 0.0 | Acid dose rate (positive = add acid) |
| `pH_reservoir` | pH | 3.5 | Reservoir pH |

### Thermal Control
| Parameter | Unit | Default | Description |
|-----------|------|---------|-------------|
| `cooling_power_W` | W | *auto* | Cooling power (auto-balances Joule heating if not set) |
| `joule_heat_fraction_catholyte` | - | 0.6 | Fraction of Joule heat into catholyte |
| `joule_heat_fraction_anolyte` | - | 0.3 | Fraction of Joule heat into anolyte |
| `UA_membrane_W_K` | W/K | 50 | Membrane heat transfer coefficient × area |
| `UA_ambient_W_K` | W/K | 5 | Ambient heat loss coefficient |
| `T_ambient_C` | °C | 25 | Ambient temperature |
| `T_reservoir_C` | °C | 55 | Initial reservoir temperature |

### Electrical Relaxation
| Parameter | Unit | Default | Description |
|-----------|------|---------|-------------|
| `electrolyte_conductivity_S_m` | S/m | 10 | Electrolyte conductivity |
| `electrode_gap_m` | m | 0.02 | Inter-electrode gap |
| `C_dl_F_m2` | F/m² | 0.02 | Double-layer capacitance |
| `V_relax_min_hr` | hr | 10.0 | Minimum voltage relaxation time |

### Current Density Tracking
| Parameter | Unit | Default | Description |
|-----------|------|---------|-------------|
| `tau_j_hr` | hr | 0.5 | Current density setpoint tracking time constant |

### Fe³⁺ Redox Shuttle (CSTR extension)
| Parameter | Unit | Default | Description |
|-----------|------|---------|-------------|
| `fe3_shuttle_enabled` | bool | false | Master switch (off → byte-identical dynamics) |
| `fe3_o2_fraction_of_sat` | - | 0.005 | Dissolved O₂ as fraction of air saturation (sealed cell) |
| `fe3_crossover_o2_flux_mol_m2_s` | mol/m²/s | 0.0 | Anolyte O₂ crossover fault flux |
| `fe3_d_m2_s` | m²/s | 5.5e-10 | Fe³⁺ diffusivity (screening family) |
| `fe3_boundary_layer_m` | m | 50e-6 | Cathode diffusion-layer thickness for Fe³⁺ |
| `fe3_k_ox_ref` | M⁻¹ s⁻¹ | 1.0e-4 | Autoxidation k_ref (bath_startup screening value) |
| `fe3_Ea_ox_J_mol` | J/mol | 50000 | Autoxidation apparent activation energy |
| `fe3_catholyte_M`, `fe3_reservoir_M`, `fe3_sludge_cumulative_mol` | M, M, mol | 0.0 | Optional initial values for the auxiliary Fe³⁺ states |

## Auxiliary State (BathAux)

The reservoir state (T_reservoir, fe2_reservoir, pH_reservoir) is tracked as auxiliary state in `design_point["_bath_aux"]`. It is **not** part of the 7-state EKF vector but is integrated alongside the EKF state by the same dynamics.

As of 2026-08 the auxiliary state optionally additionally carries the Fe³⁺ redox-shuttle states (`fe3_catholyte_M`, `fe3_reservoir_M`, `fe3_sludge_cumulative_mol`; zero and untouched while `fe3_shuttle_enabled` is off).

## Conservation Checks

The dynamics satisfy the following conservation laws:

1. **Fe²⁺ mass balance:** Total Fe²⁺ in (catholyte + reservoir) changes only due to Faraday consumption and makeup addition. With the Fe³⁺ extension on, total **iron** (dissolved Fe²⁺/Fe³⁺ in both compartments + cumulative Fe(OH)₃ sludge + Faraday-plated Fe) is the closing ledger — pinned in `tests/test_bath_fe3_cstr.py`.
2. **Energy balance:** Total thermal energy changes only due to Joule heating, cooling, and ambient losses.
3. **Proton balance:** Total protons change only due to acid/base dose and HER hydroxide production (plus autoxidation/precipitation terms with the Fe³⁺ extension on).

See `tests/test_bath_dynamics.py` for automated conservation checks.

## L1 Data-Fitting Targets

The following terms are L0 scaffolds to be replaced by data-fitted models at L1:

- **Reservoir makeup rate:** Currently a control input; at L1, fit to actual FeSO₄ dissolution kinetics or feed rate.
- **Buffer capacity β:** Currently constant; at L1, fit to actual bath speciation and complexation.
- **Heat transfer coefficients (UA_membrane, UA_ambient):** Currently estimated; at L1, fit to thermal transient data.
- **Voltage relaxation time τ_V:** Currently heuristic; at L1, fit to impedance spectroscopy or step-response data.
- **Joule heat fractions:** Currently fixed (0.6 catholyte, 0.3 anolyte); at L1, fit to calorimetry data.

## Usage

```python
from models.digital_twin import DigitalTwin

# Create twin with default bath dynamics
twin = DigitalTwin(seed=42)

# Or customize design_point
design_point = {
    "temperature_C": 60.0,
    "pH": 3.5,
    "j_avg_mA_cm2": 150.0,
    "electrode_area_m2": 1.0,
    "fe2_M": 1.0,
    # Bath dynamics parameters
    "recirculation_flow_L_hr": 6000.0,
    "fe2_makeup_rate_M_hr": 0.05,  # balance consumption
    "acid_dose_rate_M_hr": 0.01,   # balance HER
}
twin = DigitalTwin(design_point=design_point, seed=42)

# Run EKF updates
for readings in sensor_stream:
    state = twin.update(readings, dt_hr=0.1)
```

## Implementation Notes

- The `step()` function in `bath_dynamics.py` is pure (no side effects) and unit-testable standalone.
- The Jacobian `_F_jacobian` snapshots the auxiliary state before each perturbation to ensure consistent numerical differentiation.
- Auto-balance cooling (`Q_cool = Q_cath` if not set) provides stable thermal dynamics across operating points.
- Large reservoir volume (50000 L default) acts as a quasi-infinite source/sink for stability.

## References

- `models/bath_dynamics.py` — Implementation
- `models/digital_twin.py` — EKF integration
- `tests/test_bath_dynamics.py` — Conservation checks
- `docs/NEXT_STEPS.md` §3.5 — Dynamic plant loop motivation
