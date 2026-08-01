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
d(deposit)/dt = deposit_rate_um_hr
```

from physics model prediction. Clamped to ≥ 0.

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

## Auxiliary State (BathAux)

The reservoir state (T_reservoir, fe2_reservoir, pH_reservoir) is tracked as auxiliary state in `design_point["_bath_aux"]`. It is **not** part of the 7-state EKF vector but is integrated alongside the EKF state by the same dynamics.

## Conservation Checks

The dynamics satisfy the following conservation laws:

1. **Fe²⁺ mass balance:** Total Fe²⁺ in (catholyte + reservoir) changes only due to Faraday consumption and makeup addition.
2. **Energy balance:** Total thermal energy changes only due to Joule heating, cooling, and ambient losses.
3. **Proton balance:** Total protons change only due to acid/base dose and HER hydroxide production.

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
